r"""Pick at one coil, drive, place at another -- the warehouse version.

Same two ways to run it as ``run_in_isaacsim.py``.

A) From the Isaac Sim GUI
-------------------------
       C:\isaacsim\isaac-sim.bat --/rtx/ecoMode/enabled=True

   Window > Script Editor, then:

       p = r"C:\Users\user\Documents\인턴소프트웨어\isaacsim_wpt\run_pick_and_place.py"
       exec(compile(open(p, encoding="utf-8").read(), p, "exec"))

B) From the terminal
--------------------
       C:\isaacsim\python.bat run_pick_and_place.py
       C:\isaacsim\python.bat run_pick_and_place.py --headless --start=1 --target=4

What happens
------------
1. The robot is hand-placed on the source coil with a 10 mm placement error and re-aligns on
   it using the AprilTag registration.
2. The cute_arm (3-DOF, 12 + 12 cm links, from gyeong-yeon2718/cute_arm and its parent
   elevenMiles/Robotic_Arm_Seven) picks the payload off the pick shelf.
3. It folds the arm in and drives to the destination coil, aligning there.
4. It places the payload on the marked drop pad.

Why this is the same project rather than a second one: the arm has three joints and no
sensor pointed at the shelf, so it goes exactly where its joint angles say from wherever the
robot is parked.  The placement error therefore *is* the docking error, and the run reports
both so you can see it.  The wireless-charging alignment is what makes a blind arm
repeatable.

Honest scope: the arm is kinematic.  Servo torque, backlash, the arm's inertia reacting on
the chassis and friction grasping are not simulated -- see ``isaac/arm_build.py`` for why, and
for what is.
"""

from __future__ import annotations

import builtins
import os
import sys

# --------------------------------------------------------------------------
# Settings you are likely to change
# --------------------------------------------------------------------------

START_COIL = 1          # pick here
TARGET_COIL = 4         # place here
SEED = 20260811
HEADLESS = False
HOLD_SECONDS = 25.0
LOG_PATH: str | None = None

FALLBACK_PROJECT_DIR = r"C:\Users\user\Documents\인턴소프트웨어\isaacsim_wpt"
TOP_PLATE_STL_NAMES = ["top_plate.stl"]
TOWER_STL_NAMES = ["tier3_battery_box.stl"]
TOP_PLATE_STL_FALLBACKS = [r"C:\Users\user\Documents\카카오톡 받은 파일\상판 최종 버전.stl"]
TOWER_STL_FALLBACKS = [r"C:\Users\user\Downloads\터틀봇초안 3층_홈.stl"]

# --------------------------------------------------------------------------
# Mode detection -- must happen before any isaacsim / omni / pxr import
# --------------------------------------------------------------------------


def _inside_kit() -> bool:
    """True when this file is running inside an already-booted Kit application.

    ``builtins.ISAAC_LAUNCHED_FROM_TERMINAL`` is *absent* at this point in both modes, so it
    cannot be used here; importing ``omni.kit.app`` succeeds only once Kit's plugin system
    has booted, which is a direct test for "is there an app".
    """
    try:
        import omni.kit.app

        return omni.kit.app.get_app() is not None
    except Exception:                    # noqa: BLE001
        return False


_TERMINAL_FLAG = getattr(builtins, "ISAAC_LAUNCHED_FROM_TERMINAL", None)
IN_KIT = _inside_kit()
IN_GUI = IN_KIT and _TERMINAL_FLAG is not False
NEED_SIMULATION_APP = not IN_KIT

PROJECT_DIR = FALLBACK_PROJECT_DIR
try:
    _candidate = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _candidate = ""
if _candidate and os.path.isdir(os.path.join(_candidate, "wpt_dock")):
    PROJECT_DIR = _candidate
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


def _first_existing(names: list[str], fallbacks: list[str]) -> str | None:
    for name in names:
        p = os.path.join(PROJECT_DIR, "assets", name)
        if os.path.exists(p):
            return p
    for p in fallbacks:
        if p and os.path.exists(p):
            return p
    return None


def _take_arg(flag: str, default):
    prefix = f"--{flag}"
    for i, token in enumerate(list(sys.argv)):
        if token == prefix:
            sys.argv.pop(i)
            return True
        if token.startswith(prefix + "="):
            sys.argv.pop(i)
            return token.split("=", 1)[1]
    return default


simulation_app = None

if not IN_GUI:
    _headless = _take_arg("headless", False)
    HEADLESS = bool(_headless) if _headless is not False else HEADLESS
    START_COIL = int(_take_arg("start", START_COIL))
    TARGET_COIL = int(_take_arg("target", TARGET_COIL))
    SEED = int(_take_arg("seed", SEED))
    _log = _take_arg("log", None)
    LOG_PATH = _log if isinstance(_log, str) else LOG_PATH
    _hold = _take_arg("hold", None)
    HOLD_SECONDS = float(_hold) if isinstance(_hold, str) else HOLD_SECONDS

    if NEED_SIMULATION_APP:
        from isaacsim import SimulationApp  # noqa: E402

        simulation_app = SimulationApp(
            {"headless": bool(HEADLESS), "width": 1600, "height": 900}
        )

# --------------------------------------------------------------------------
# Everything below may import omni / pxr / isaacsim
# --------------------------------------------------------------------------

from wpt_dock.arm import ArmSpec                                     # noqa: E402
from wpt_dock.config import DEFAULTS                                 # noqa: E402
from wpt_dock.isaac.pickplace_runner import PickPlaceRunner          # noqa: E402
from wpt_dock.isaac.runner import RunConfig                          # noqa: E402


def _make_runner():
    return PickPlaceRunner(
        DEFAULTS,
        RunConfig(
            start_coil=START_COIL,
            target_coil=TARGET_COIL,
            seed=SEED,
            top_plate_stl=_first_existing(TOP_PLATE_STL_NAMES, TOP_PLATE_STL_FALLBACKS),
            tower_stl=_first_existing(TOWER_STL_NAMES, TOWER_STL_FALLBACKS),
            log_path=LOG_PATH,
            verbose=True,
        ),
        ArmSpec(),
    )


def _build_scene(world, runner):
    from isaacsim.core.utils.stage import get_current_stage
    from isaacsim.robot.wheeled_robots.robots import WheeledRobot

    stage = get_current_stage()
    handles = runner.build(stage)
    robot = WheeledRobot(
        prim_path=handles.prim_path,
        name="turtlebot_wpt",
        wheel_dof_names=list(handles.wheel_joints),
        create_robot=False,
    )
    world.scene.add(robot)
    runner.attach_robot(robot)
    return handles


def _aim_camera(target_coil: int) -> None:
    """Frame the whole workcell, not just the board -- the shelves are the point here."""
    try:
        from isaacsim.core.utils.viewports import set_camera_view

        pos = DEFAULTS.board.coil_positions[target_coil]
        set_camera_view(
            eye=[pos[0] - 0.75, pos[1] - 0.85, 0.70],
            target=[0.22, 0.13, 0.08],
            camera_prim_path="/OmniverseKit_Persp",
        )
    except Exception as exc:                           # noqa: BLE001
        print(f"  [view] could not aim the viewport ({exc})", flush=True)


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------


async def _gui_main():
    import omni.kit.app
    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import create_new_stage_async

    runner = _make_runner()
    await create_new_stage_async()
    world = World(physics_dt=DEFAULTS.sim.physics_dt, rendering_dt=DEFAULTS.sim.render_dt,
                  stage_units_in_meters=1.0)
    await world.initialize_simulation_context_async()
    _build_scene(world, runner)
    await world.reset_async()
    runner.after_reset()
    _aim_camera(TARGET_COIL)
    await omni.kit.app.get_app().next_update_async()

    state = {"done": False}

    def on_physics(step_size: float) -> None:
        try:
            runner.on_step(step_size)
            if runner.finished and not state["done"]:
                state["done"] = True
                print(runner.summary(), flush=True)
                runner.write_log()
        except Exception:                              # noqa: BLE001
            import traceback

            state["done"] = True
            traceback.print_exc()

    world.add_physics_callback("wpt_pickplace_step", on_physics)

    def rearm(_event) -> None:
        if not world.physics_callback_exists("wpt_pickplace_step"):
            world.add_physics_callback("wpt_pickplace_step", on_physics)

    world.add_timeline_callback("wpt_pickplace_rearm", rearm)
    await world.play_async()
    print("  [gui] running. The runner is on builtins.WPT_PICKPLACE_RUNNER.", flush=True)
    builtins.WPT_PICKPLACE_RUNNER = runner


# --------------------------------------------------------------------------
# Standalone
# --------------------------------------------------------------------------


def _app_running() -> bool:
    if simulation_app is not None:
        return bool(simulation_app.is_running())
    import omni.kit.app

    return bool(omni.kit.app.get_app().is_running())


def _standalone_main():
    from isaacsim.core.api import World

    runner = _make_runner()
    world = World(physics_dt=DEFAULTS.sim.physics_dt, rendering_dt=DEFAULTS.sim.render_dt,
                  stage_units_in_meters=1.0)
    _build_scene(world, runner)
    world.reset()
    runner.after_reset()
    _aim_camera(TARGET_COIL)
    world.add_physics_callback("wpt_pickplace_step", runner.on_step)
    world.play()

    hold = 0.0
    reported = False
    while _app_running():
        world.step(render=not HEADLESS)
        if not world.is_playing():
            continue
        if runner.finished:
            if not reported:
                reported = True
                print(runner.summary(), flush=True)
                runner.write_log()
            hold += DEFAULTS.sim.render_dt
            if HEADLESS or hold >= HOLD_SECONDS:
                break

    if not reported:
        print(runner.summary(), flush=True)
        runner.write_log()
    # fast_shutdown skips Python's atexit, so buffered output is lost without this.
    sys.stdout.flush()
    sys.stderr.flush()
    if simulation_app is not None:
        simulation_app.close()


if IN_GUI:
    try:
        from omni.kit.async_engine import run_coroutine

        run_coroutine(_gui_main())
    except ImportError:
        import asyncio

        asyncio.ensure_future(_gui_main())
else:
    _standalone_main()
