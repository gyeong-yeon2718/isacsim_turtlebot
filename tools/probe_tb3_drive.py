r"""Measure the TurtleBot3 asset's drivetrain in isolation.

    C:\isaacsim\python.bat tools\probe_tb3_drive.py

Why: with the published asset the robot drives at 70 mm/s but stops dead at the 12-14 mm/s
the controller tapers to over the last centimetre of a leg, then sits there until the
mission times out.  Mass is right (1.0017 kg, matching the Burger datasheet) and joint
friction is already zero, so the remaining candidates are the drive gains and the wheel
contact.  Guessing between them has already cost two runs, so this measures instead:
command a fixed wheel rate, wait, and report what the wheel and the chassis actually did.

A drive problem shows up as *actual wheel rate* falling short of the command.
A contact problem shows up as the wheel turning at the commanded rate while the chassis
does not move.  The two are distinguishable in one table.
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
WHEEL_SEP = 0.160
RATES = [0.15, 0.30, 0.45, 0.60, 1.00, 2.12, 4.00, 6.60]   # rad/s at the wheel


def build(max_force: float, damping: float):
    world = World(physics_dt=1.0 / 120.0, rendering_dt=1.0 / 60.0, stage_units_in_meters=1.0)
    stage = get_current_stage()

    # A plain static floor with explicit friction, so the ground side of the contact pair
    # is a known quantity rather than the scene default.
    floor = UsdGeom.Cube.Define(stage, Sdf.Path("/World/floor"))
    floor.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(floor)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.05))
    xf.AddScaleOp().Set(Gf.Vec3f(6.0, 6.0, 0.1))
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    mat = UsdShade.Material.Define(stage, Sdf.Path("/World/Looks/Floor"))
    UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api = UsdPhysics.MaterialAPI(mat.GetPrim())
    api.CreateStaticFrictionAttr(0.9)
    api.CreateDynamicFrictionAttr(0.8)
    api.CreateRestitutionAttr(0.0)
    UsdShade.MaterialBindingAPI.Apply(floor.GetPrim()).Bind(
        mat, bindingStrength=UsdShade.Tokens.weakerThanDescendants, materialPurpose="physics"
    )

    root = get_assets_root_path()
    head, _, _ = root.rpartition("/")
    url = f"{head}/4.2/Isaac/Robots/Turtlebot/turtlebot3_burger.usd"
    print(f"asset: {url}", flush=True)
    add_reference_to_stage(usd_path=url, prim_path=ROOT)
    prim = stage.GetPrimAtPath(Sdf.Path(ROOT))
    prim.Load(Usd.LoadWithDescendants)

    joints = []
    for p in Usd.PrimRange(prim):
        if p.IsA(UsdPhysics.RevoluteJoint) and "wheel" in p.GetName().lower():
            joints.append(p)
    for p in joints:
        drive = UsdPhysics.DriveAPI.Get(p, "angular") or UsdPhysics.DriveAPI.Apply(p, "angular")
        drive.CreateTypeAttr().Set("force")
        drive.CreateStiffnessAttr(0.0)
        drive.CreateDampingAttr(float(damping))
        drive.CreateTargetVelocityAttr(0.0)
        drive.CreateMaxForceAttr(float(max_force))
    names = [p.GetName() for p in joints]

    robot = WheeledRobot(prim_path=ROOT, name="tb3", wheel_dof_names=names, create_robot=False)
    world.scene.add(robot)
    world.reset()
    world.play()
    return world, robot, names


def sweep(world, robot, label: str):
    print(f"\n=== {label} ===", flush=True)
    print("  cmd rad/s   actual rad/s   body m/s   expected m/s   ratio", flush=True)
    for rate in RATES:
        for _ in range(240):        # 2 s at 120 Hz
            robot.apply_wheel_actions(
                ArticulationAction(joint_velocities=np.array([rate, rate], dtype=float))
            )
            world.step(render=False)
        wv = robot.get_wheel_velocities()
        lin = robot.get_linear_velocity()
        pos, quat = robot.get_world_pose()
        w, x, y, z = (float(v) for v in quat)
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        body = float(lin[0]) * np.cos(yaw) + float(lin[1]) * np.sin(yaw)
        expected = rate * WHEEL_RADIUS
        ratio = body / expected if expected > 1e-9 else 0.0
        print(f"  {rate:9.3f}   {float(wv[0]):12.4f}   {body:8.4f}   {expected:12.4f}   {ratio:5.2f}",
              flush=True)
    # Park it before the next sweep so the next case starts from rest.
    for _ in range(120):
        robot.apply_wheel_actions(ArticulationAction(joint_velocities=np.array([0.0, 0.0])))
        world.step(render=False)


world, robot, names = build(max_force=1.5, damping=1.0e4)
print(f"wheel joints: {names}, dofs: {list(robot.dof_names)}", flush=True)
sweep(world, robot, "maxForce 1.5 N.m, damping 1e4")

# Second case: lift the torque ceiling entirely.  If the low rates come alive, the cap was
# the limit; if they do not, the resistance is in the contact, not the actuator.
stage = get_current_stage()
for p in Usd.PrimRange(stage.GetPrimAtPath(Sdf.Path(ROOT))):
    if p.IsA(UsdPhysics.RevoluteJoint) and "wheel" in p.GetName().lower():
        UsdPhysics.DriveAPI.Get(p, "angular").CreateMaxForceAttr(1.0e6)
world.reset()
world.play()
sweep(world, robot, "maxForce 1e6 N.m, damping 1e4")

import sys  # noqa: E402

sys.stdout.flush()
simulation_app.close()
