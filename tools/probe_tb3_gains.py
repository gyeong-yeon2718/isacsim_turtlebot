r"""Find the API that actually changes the articulation's velocity-drive gains.

    C:\isaacsim\python.bat tools\probe_tb3_gains.py

Evidence this exists to act on, from ``tools/probe_tb3_drive.py``:

    cmd rad/s   actual rad/s
        1.000        0.1551
        2.120        1.1130
        4.000        2.9499
        6.600        5.5900

A **constant** shortfall of about 1.0 rad/s at every command -- the signature of a finite
drive damping working against a constant load torque, where ``deficit = T_load / damping``.
Below roughly 1 rad/s (33 mm/s of body speed) the shortfall exceeds the command and the
wheel cannot turn forwards at all, which is exactly the stall the mission hit at 12-14 mm/s.

Crucially, authoring ``drive:angular:physics:damping`` and ``maxForce`` in USD changed
nothing: the 1.5 N.m and 1e6 N.m sweeps were bit-identical.  ``WheeledRobot.post_reset()``
calls ``switch_control_mode("velocity")``, which writes the gains into the articulation
itself, so a USD-level edit made before that is simply overwritten.

So this probe reports the gains the articulation is really using, sets them through the
Isaac API, and re-measures.  No guessing about which call works -- the table says.
"""

from __future__ import annotations

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402
from isaacsim.robot.wheeled_robots.robots import WheeledRobot  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402

ROOT = "/World/tb3"
WHEEL_RADIUS = 0.033
RATES = [0.15, 0.30, 0.45, 0.60, 1.00, 2.12, 4.00, 6.60]

world = World(physics_dt=1.0 / 120.0, rendering_dt=1.0 / 60.0, stage_units_in_meters=1.0)
stage = get_current_stage()

floor = UsdGeom.Cube.Define(stage, Sdf.Path("/World/floor"))
floor.CreateSizeAttr(1.0)
xf = UsdGeom.Xformable(floor)
xf.ClearXformOpOrder()
xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.05))
xf.AddScaleOp().Set(Gf.Vec3f(6.0, 6.0, 0.1))
UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
mat = UsdShade.Material.Define(stage, Sdf.Path("/World/Looks/Floor"))
UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
mapi = UsdPhysics.MaterialAPI(mat.GetPrim())
mapi.CreateStaticFrictionAttr(0.9)
mapi.CreateDynamicFrictionAttr(0.8)
mapi.CreateRestitutionAttr(0.0)
UsdShade.MaterialBindingAPI.Apply(floor.GetPrim()).Bind(
    mat, bindingStrength=UsdShade.Tokens.weakerThanDescendants, materialPurpose="physics"
)

root = get_assets_root_path()
head, _, _ = root.rpartition("/")
url = f"{head}/4.2/Isaac/Robots/Turtlebot/turtlebot3_burger.usd"
add_reference_to_stage(usd_path=url, prim_path=ROOT)
stage.GetPrimAtPath(Sdf.Path(ROOT)).Load(Usd.LoadWithDescendants)

names = [
    p.GetName()
    for p in Usd.PrimRange(stage.GetPrimAtPath(Sdf.Path(ROOT)))
    if p.IsA(UsdPhysics.RevoluteJoint) and "wheel" in p.GetName().lower()
]
robot = WheeledRobot(prim_path=ROOT, name="tb3", wheel_dof_names=names, create_robot=False)
world.scene.add(robot)
world.reset()
world.play()

print(f"\nwheel joints: {names}", flush=True)
controller = robot.get_articulation_controller()
print(f"controller: {type(controller).__name__}", flush=True)
print("  gain-related members:",
      [m for m in dir(controller) if "gain" in m.lower()], flush=True)
view = getattr(robot, "_articulation_view", None)
print(f"view: {type(view).__name__ if view is not None else None}", flush=True)
if view is not None:
    print("  gain-related members:", [m for m in dir(view) if "gain" in m.lower()], flush=True)
    try:
        kps, kds = view.get_gains()
        print(f"  gains AS SIMULATED: kps={np.asarray(kps).ravel()}  kds={np.asarray(kds).ravel()}",
              flush=True)
    except Exception as exc:
        print(f"  get_gains failed: {exc}", flush=True)


def sweep(label: str) -> None:
    print(f"\n=== {label} ===", flush=True)
    print("  cmd rad/s   actual rad/s   body m/s   expected m/s   ratio", flush=True)
    for rate in RATES:
        for _ in range(240):
            robot.apply_wheel_actions(
                ArticulationAction(joint_velocities=np.array([rate, rate], dtype=float))
            )
            world.step(render=False)
        wv = robot.get_wheel_velocities()
        lin = robot.get_linear_velocity()
        _, quat = robot.get_world_pose()
        w, x, y, z = (float(v) for v in quat)
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        body = float(lin[0]) * np.cos(yaw) + float(lin[1]) * np.sin(yaw)
        expected = rate * WHEEL_RADIUS
        print(f"  {rate:9.3f}   {float(wv[0]):12.4f}   {body:8.4f}   {expected:12.4f}   "
              f"{body / expected if expected > 1e-9 else 0.0:5.2f}", flush=True)
    for _ in range(120):
        robot.apply_wheel_actions(ArticulationAction(joint_velocities=np.array([0.0, 0.0])))
        world.step(render=False)


sweep("as loaded")

for kd in (1.0e3, 1.0e5):
    applied = False
    try:
        n = len(names)
        view.set_gains(kps=np.zeros((1, n)), kds=np.full((1, n), kd))
        applied = True
        print(f"\nset_gains on the view: kds={kd:g}", flush=True)
    except Exception as exc:
        print(f"\nview.set_gains({kd:g}) failed: {exc}", flush=True)
    if not applied:
        try:
            controller.set_gains(kps=np.zeros(len(names)), kds=np.full(len(names), kd))
            applied = True
            print(f"\nset_gains on the controller: kds={kd:g}", flush=True)
        except Exception as exc:
            print(f"controller.set_gains({kd:g}) failed: {exc}", flush=True)
    if applied:
        try:
            kps, kds = view.get_gains()
            print(f"  gains now: kps={np.asarray(kps).ravel()}  kds={np.asarray(kds).ravel()}",
                  flush=True)
        except Exception:
            pass
        sweep(f"kds = {kd:g}")

import sys  # noqa: E402

sys.stdout.flush()
simulation_app.close()
