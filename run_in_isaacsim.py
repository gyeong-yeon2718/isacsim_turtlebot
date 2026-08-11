r"""ONE file, two ways to run it.

A) From the Isaac Sim GUI  (this is the flow you asked for)
----------------------------------------------------------
1. Launch the GUI.  Only ``isaacsim.exp.full.kit`` -- what ``isaac-sim.bat`` starts --
   enables the Script Editor, so use the normal launcher.  On an 8 GB card the eco
   flag is worth adding:

       C:\isaacsim\isaac-sim.bat --/rtx/ecoMode/enabled=True

2. Window > Script Editor, then paste these two lines and press Ctrl+Enter:

       p = r"C:\Users\user\Documents\인턴소프트웨어\isaacsim_wpt\run_in_isaacsim.py"
       exec(compile(open(p, encoding="utf-8").read(), p, "exec"))

   Passing the path to ``compile`` sets ``__file__``, so the script finds its own
   package directory without needing the constant below to be right.

3. That is all.  The scene builds, PLAY starts by itself, and the console prints a
   line per state change.  Watch the target coil's ring fill and the coil light up.

   Nothing blocks the UI: the per-frame work is registered as a **physics callback**
   and the setup runs on Kit's own async loop.  A ``while True: world.step()`` in the
   Script Editor freezes Kit solid, because the Run button executes on the UI
   thread -- which is why this file never does it.

B) Headless or windowed from the terminal
-----------------------------------------
       C:\isaacsim\python.bat run_in_isaacsim.py
       C:\isaacsim\python.bat run_in_isaacsim.py --headless          # fast, no window
       C:\isaacsim\python.bat run_in_isaacsim.py --start=2 --target=3
       C:\isaacsim\python.bat run_in_isaacsim.py --seed=7 --log=run.csv

Before running either, screen the controller without a GPU -- it takes seconds and
tells you far more than one Isaac run can:

       python validate.py

What it does
------------
Drives the TurtleBot3 from one wireless-charging coil to another on the 0.80 x
0.60 m plywood stage, using ``/odom``-style wheel odometry to get close and AprilTag
registration for the final alignment, then verifies the alignment and lights the
coil.  Set ``START_COIL`` / ``TARGET_COIL`` below (or pass ``--start`` / ``--target``).

Coil numbering, following the upstream convention: coil 1 is the origin, +X points
to coil 2, +Y points to coil 3, coil 4 is the diagonal opposite.
"""

from __future__ import annotations

import builtins
import os
import sys

# --------------------------------------------------------------------------
# Settings you are likely to change
# --------------------------------------------------------------------------

START_COIL = 1
TARGET_COIL = 4
SEED = 20260811
HEADLESS = False
# How long to keep the window open after the mission ends, then close.  The window
# closing is not a crash: standalone mode owns the app and shuts it down on purpose.
# Pass --hold=600 to linger, or run from the GUI Script Editor, which never closes.
HOLD_SECONDS = 25.0
LOG_PATH: str | None = None

# Only used if the script cannot work out its own location (a bare ``exec`` in the
# Script Editor with no ``__file__``).  Prefer the two-line launcher above, which
# sets ``__file__`` for you and makes this irrelevant.
FALLBACK_PROJECT_DIR = r"C:\Users\user\Documents\인턴소프트웨어\isaacsim_wpt"

# The user's printed parts.  Project-local copies first so the scene keeps working
# if the originals are moved or cleaned out of Downloads.
TOP_PLATE_STL_NAMES = ["top_plate.stl"]
TOWER_STL_NAMES = ["tier3_battery_box.stl"]
TOP_PLATE_STL_FALLBACKS = [
    r"C:\Users\user\Documents\카카오톡 받은 파일\상판 최종 버전.stl",
]
TOWER_STL_FALLBACKS = [
    r"C:\Users\user\Downloads\터틀봇초안 3층_홈.stl",
]

# --------------------------------------------------------------------------
# Mode detection -- must happen before any isaacsim / omni / pxr import
# --------------------------------------------------------------------------


def _inside_kit() -> bool:
    """True when this file is running inside an already-booted Kit application.

    This, not ``builtins.ISAAC_LAUNCHED_FROM_TERMINAL``, is the reliable test.  That
    flag is set to False by ``SimulationApp`` and to True by ``isaacsim.core.api`` --
    but *neither* has run yet at this point in the file, so in a fresh
    ``python.bat`` process the flag is simply **absent**, exactly as it is in the
    GUI.  Defaulting absent to "GUI" sends a standalone run down the async path,
    ``SimulationApp`` never gets constructed, and the first ``pxr`` import dies with
    ``No module named 'pxr'``.  (Which is precisely what happened the first time.)

    ``omni.kit.app`` is an extension module: it only reaches ``sys.path`` once Kit's
    plugin system has booted, so importing it *is* the test for "is there an app".
    """
    try:
        import omni.kit.app

        return omni.kit.app.get_app() is not None
    except Exception:                    # noqa: BLE001 - any failure means "no Kit"
        return False


# Explicitly False means a SimulationApp already exists in this process.
_TERMINAL_FLAG = getattr(builtins, "ISAAC_LAUNCHED_FROM_TERMINAL", None)
IN_KIT = _inside_kit()
IN_GUI = IN_KIT and _TERMINAL_FLAG is not False
NEED_SIMULATION_APP = not IN_KIT


def _resolve_project_dir() -> str:
    try:
        candidate = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        candidate = ""
    # When exec()'d without compile(), the ambient __file__ can belong to something
    # else entirely, so only trust it if the package is actually sitting next to it.
    if candidate and os.path.isdir(os.path.join(candidate, "wpt_dock")):
        return candidate
    return FALLBACK_PROJECT_DIR


PROJECT_DIR = _resolve_project_dir()
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
    """Read and *remove* ``--flag=value`` from argv.

    Removal matters: leftover unknown arguments are forwarded into Kit's own
    command-line parsing, where an unrecognised flag can change app behaviour or
    abort the launch outright.
    """
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
        from isaacsim import SimulationApp  # noqa: E402 -- must precede every isaacsim import

        simulation_app = SimulationApp(
            {
                "headless": bool(HEADLESS),   # NOTE: SimulationApp defaults to headless=True
                "width": 1600,
                "height": 900,
            }
        )

# --------------------------------------------------------------------------
# Everything below may import omni / pxr / isaacsim
# --------------------------------------------------------------------------

from wpt_dock.config import DEFAULTS                              # noqa: E402
from wpt_dock.isaac.runner import RunConfig, SimulationRunner     # noqa: E402


def _make_runner():
    return SimulationRunner(
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
    )


def _build_scene(world, runner):
    """Author the stage and register the articulation.  Shared by both entry paths."""
    from isaacsim.core.utils.stage import get_current_stage
    from isaacsim.robot.wheeled_robots.robots import WheeledRobot

    stage = get_current_stage()
    handles = runner.build(stage)

    # create_robot=False: the prim already exists -- either the referenced official
    # TurtleBot3 asset or the primitive fallback.  This wrapper is only here for the
    # articulation view: joint index lookup, apply_wheel_actions, get_world_pose,
    # get_wheel_velocities.
    #
    # The joint names come from ``handles``, i.e. from walking the loaded asset, not
    # from a constant.  A wrong name here does not raise; it yields a robot that never
    # moves.
    robot = WheeledRobot(
        prim_path=handles.prim_path,
        name="turtlebot_wpt",
        wheel_dof_names=list(handles.wheel_joints),
        create_robot=False,
    )
    world.scene.add(robot)
    runner.attach_robot(robot)
    return robot


# --------------------------------------------------------------------------
# A) GUI: async setup + physics callback, never blocking the UI thread
# --------------------------------------------------------------------------


async def _gui_main():
    import omni.kit.app
    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import create_new_stage_async

    runner = _make_runner()

    await create_new_stage_async()
    world = World(
        physics_dt=DEFAULTS.sim.physics_dt,
        rendering_dt=DEFAULTS.sim.render_dt,
        stage_units_in_meters=1.0,
    )
    # In the Extensions workflow SimulationContext.__init__ skips stage setup, so
    # this await is not optional: without it the physics context stays None and
    # reset() raises.
    await world.initialize_simulation_context_async()

    _build_scene(world, runner)

    await world.reset_async()
    runner.after_reset()
    await omni.kit.app.get_app().next_update_async()   # hand a frame back to the UI

    state = {"done": False}

    def on_physics(step_size: float) -> None:
        # An exception escaping a physics callback is swallowed and the callback
        # silently abandoned, so failures are caught and printed rather than
        # leaving a scene that looks fine and does nothing.
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

    world.add_physics_callback("wpt_dock_step", on_physics)

    def rearm(_event) -> None:
        # STOP wipes every physics callback (SimulationContext clears
        # _physics_callback_functions on TimelineEventType.STOP), so STOP then PLAY
        # would otherwise leave a dead scene that still looks correct.
        if not world.physics_callback_exists("wpt_dock_step"):
            world.add_physics_callback("wpt_dock_step", on_physics)

    world.add_timeline_callback("wpt_dock_rearm", rearm)

    await world.play_async()
    print(
        "  [gui] running. STOP/PLAY re-arms the physics callback; re-run this script "
        "for a fresh scene. The runner is on builtins.WPT_DOCK_RUNNER.",
        flush=True,
    )
    builtins.WPT_DOCK_RUNNER = runner


# --------------------------------------------------------------------------
# B) Standalone
# --------------------------------------------------------------------------


def _app_running() -> bool:
    """Loop condition that works whether or not *we* created the app.

    Re-running this file inside an already-booted standalone session must not
    construct a second ``SimulationApp``, so ``simulation_app`` can legitimately be
    ``None`` here.
    """
    if simulation_app is not None:
        return bool(simulation_app.is_running())
    import omni.kit.app

    return bool(omni.kit.app.get_app().is_running())


def _aim_camera_at_board(target_coil: int) -> None:
    """Point the viewport at the stage from a three-quarter view.

    The board is 0.80 x 0.60 m and the default perspective camera sits metres away, so
    without this the scene is a speck.  Framing it here means a capture -- and the GUI's
    first frame -- shows the thing the run is about.
    """
    try:
        from isaacsim.core.utils.viewports import set_camera_view

        pos = DEFAULTS.board.coil_positions[target_coil]
        set_camera_view(
            eye=[pos[0] - 0.55, pos[1] - 0.62, 0.52],
            target=[pos[0], pos[1], 0.04],
            camera_prim_path="/OmniverseKit_Persp",
        )
    except Exception as exc:                           # noqa: BLE001
        print(f"  [view] could not aim the viewport ({exc})", flush=True)


def _standalone_main():
    from isaacsim.core.api import World

    runner = _make_runner()
    world = World(
        physics_dt=DEFAULTS.sim.physics_dt,
        rendering_dt=DEFAULTS.sim.render_dt,
        stage_units_in_meters=1.0,
    )
    _build_scene(world, runner)
    world.reset()
    runner.after_reset()
    _aim_camera_at_board(TARGET_COIL)

    # The same callback the GUI uses, so the two paths cannot drift apart.
    world.add_physics_callback("wpt_dock_step", runner.on_step)
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
    # Isaac Sim defaults to fast_shutdown=True, which exits hard enough that Python's
    # buffered stdout is never flushed -- the run summary silently vanished from the
    # first redirected log even though the CSV had been written.
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
