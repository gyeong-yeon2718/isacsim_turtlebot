"""Can two kinematic pads, driven every step, hold a dynamic box by friction?

Minimal scene: no robot, no articulation, just the mechanism in question.  If this works the
crash is about where the pads live, not about the idea; if it crashes here it is the idea.
"""
import sys

from isaacsim import SimulationApp

app = SimulationApp({"headless": True}, experience="")

OUT = r"C:\Users\user\.claude\turtlebot_isacsim\pads.txt"
lines = []


def say(s):
    lines.append(s)
    sys.__stderr__.write(s + "\n")
    open(OUT, "w", encoding="utf-8").write("\n".join(lines))


try:
    import numpy as np
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    from isaacsim.core.api import World

    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 120.0)
    world.scene.add_default_ground_plane()
    stage = world.stage

    def box(path, size, pos, *, dynamic=None, mass=0.04):
        x = UsdGeom.Xform.Define(stage, Sdf.Path(path))
        xf = UsdGeom.Xformable(x)
        t = xf.AddTranslateOp()
        t.Set(Gf.Vec3d(*pos))
        o = xf.AddOrientOp()
        o.Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        if dynamic is not None:
            rb = UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
            rb.CreateKinematicEnabledAttr(not dynamic)
            UsdPhysics.MassAPI.Apply(x.GetPrim()).CreateMassAttr(mass)
        c = UsdGeom.Cube.Define(stage, Sdf.Path(path + "/geo"))
        c.CreateSizeAttr(1.0)
        cx = UsdGeom.Xformable(c)
        # A UsdGeom.Cube with size 1 spans -0.5..+0.5, so the scale IS the edge length.
        # Halving it, as the first version did, built a 12.5 mm box the 24.4 mm pads never
        # touched -- and the "grasp failed" that produced was a geometry error, not physics.
        cx.AddScaleOp().Set(Gf.Vec3f(size[0], size[1], size[2]))
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
        return t, o

    W = 0.025                      # the payload, 25 mm
    PAD = 0.004
    # Payload sitting on the ground, dynamic.
    box("/World/payload", (W, W, W), (0.0, 0.0, W / 2 + 0.001), dynamic=True)
    # Two kinematic pads either side of it, starting apart.
    open_gap = 0.030
    lt, lo_ = box("/World/pad_left", (0.014, PAD, 0.016),
                  (0.0, +(open_gap + PAD) / 2, W / 2 + 0.001), dynamic=False, mass=0.004)
    rt, ro = box("/World/pad_right", (0.014, PAD, 0.016),
                 (0.0, -(open_gap + PAD) / 2, W / 2 + 0.001), dynamic=False, mass=0.004)

    # Default PhysX friction (about 0.5 static) on purpose: if the grasp only holds with an
    # invented 0.95, that is a tuned result rather than a physical one, and the question here is
    # whether the mechanism works at all.
    say("scene built")

    world.reset()
    say("reset OK")

    from isaacsim.core.prims import RigidPrim

    view = RigidPrim(prim_paths_expr="/World/payload", name="pv", reset_xform_properties=False)
    view.initialize()

    SQUEEZE = 0.0006
    shut = (W - SQUEEZE + PAD) / 2
    for step in range(600):
        # Close over the first 120 steps, then lift over the next 240.
        f = min(1.0, step / 120.0)
        half = (open_gap + PAD) / 2 + f * (shut - (open_gap + PAD) / 2)
        lift = 0.0 if step < 150 else min(0.12, (step - 150) * 0.0008)
        z = W / 2 + 0.001 + lift
        lt.Set(Gf.Vec3d(0.0, +half, z))
        rt.Set(Gf.Vec3d(0.0, -half, z))
        world.step(render=False)
        if step in (0, 100, 140, 160, 200, 300, 400, 599):
            p, _q = view.get_world_poses()
            say(f"  step {step:3d} gap {2*half-PAD:.4f} lift {lift:.4f}  "
                f"payload z {float(p[0][2]):+.4f}  y {float(p[0][1]):+.4f}")
    p, _q = view.get_world_poses()
    held = float(p[0][2]) > 0.05
    say(f"VERDICT: payload final z = {float(p[0][2]):.4f} m -> "
        f"{'HELD by friction' if held else 'DROPPED'}")
except Exception as exc:                              # noqa: BLE001
    import traceback
    say("FAILED: " + traceback.format_exc())

app.close()
