"""Where does jaw_stl actually end up?  Measure it in the built scene, do not reason about it."""
import sys

from isaacsim import SimulationApp

app = SimulationApp({"headless": True}, experience="")

OUT = r"C:\Users\user\.claude\turtlebot_isacsim\jaw.txt"
lines = []
try:
    sys.path.insert(0, r"C:\Users\user\.claude\turtlebot_isacsim")
    from pxr import Sdf, Usd, UsdGeom

    from isaacsim.core.api import World

    from wpt_dock.arm import ArmSpec
    from wpt_dock.config import DEFAULTS
    from wpt_dock.isaac.arm_build import build_arm
    from wpt_dock.isaac.robot_build import build_robot

    world = World(stage_units_in_meters=1.0)
    stage = world.stage
    handles = build_robot(stage, DEFAULTS, position=(0.0, 0.0), yaw=0.0,
                          top_plate_stl=None, tower_stl=None)
    spec = ArmSpec()
    notes = []
    rig = build_arm(stage, handles.chassis_path, spec,
                    plate_top_local_z=handles.plate_top_z - DEFAULTS.robot.wheel_radius,
                    stl_dir=r"C:\Users\user\.claude\turtlebot_isacsim\assets\arm",
                    notes=notes)

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"])
    xf = UsdGeom.XformCache(Usd.TimeCode.Default())

    def report(label, path):
        prim = stage.GetPrimAtPath(Sdf.Path(path))
        if not prim or not prim.IsValid():
            lines.append(f"{label:26s} MISSING  {path}")
            return None
        t = xf.GetLocalToWorldTransform(prim).ExtractTranslation()
        try:
            r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            mn, mx = r.GetMin(), r.GetMax()
            lines.append(f"{label:26s} origin ({t[0]:+.4f},{t[1]:+.4f},{t[2]:+.4f})  "
                         f"bbox x{mn[0]:+.4f}..{mx[0]:+.4f} y{mn[1]:+.4f}..{mx[1]:+.4f} "
                         f"z{mn[2]:+.4f}..{mx[2]:+.4f}")
        except Exception:
            lines.append(f"{label:26s} origin ({t[0]:+.4f},{t[1]:+.4f},{t[2]:+.4f})  (no bbox)")
        return t

    grip = rig.tcp_path.rsplit("/", 1)[0]
    lines.append("--- notes from build_arm ---")
    lines.extend("  " + n for n in notes)
    lines.append("--- world positions ---")
    report("chassis", handles.chassis_path)
    report("arm root", rig.root)
    report("shoulder", rig.root + "/yaw/shoulder")
    report("elbow", rig.root + "/yaw/shoulder/elbow")
    report("fore_link_stl", rig.root + "/yaw/shoulder/elbow/fore_link_stl")
    report("gripper frame", grip)
    report("tcp", rig.tcp_path)
    report("jaw_left frame", grip + "/jaw_left")
    jaw = report("jaw_left/jaw_stl", grip + "/jaw_left/jaw_stl")
    report("jaw_fixed frame", grip + "/jaw_fixed")

    # The number that matters: how far is the jaw mesh from the frame it is parented to?
    f = xf.GetLocalToWorldTransform(stage.GetPrimAtPath(Sdf.Path(grip + "/jaw_left"))).ExtractTranslation()
    if jaw is not None:
        d = [float(jaw[i] - f[i]) for i in range(3)]
        lines.append(f"\njaw_stl origin minus jaw_left frame = ({d[0]:+.4f}, {d[1]:+.4f}, {d[2]:+.4f}) m")
    # And where its geometry actually sits relative to that frame.
    p = stage.GetPrimAtPath(Sdf.Path(grip + "/jaw_left/jaw_stl"))
    if p and p.IsValid():
        r = cache.ComputeWorldBound(p).ComputeAlignedRange()
        mn, mx = r.GetMin(), r.GetMax()
        lines.append(f"jaw_stl bbox relative to jaw_left frame: "
                     f"x{mn[0]-f[0]:+.4f}..{mx[0]-f[0]:+.4f} "
                     f"y{mn[1]-f[1]:+.4f}..{mx[1]-f[1]:+.4f} "
                     f"z{mn[2]-f[2]:+.4f}..{mx[2]-f[2]:+.4f} m")
except Exception as exc:                              # noqa: BLE001
    import traceback
    lines.append("FAILED: " + traceback.format_exc())

open(OUT, "w", encoding="utf-8").write("\n".join(lines))
sys.__stderr__.write(f"wrote {len(lines)} lines\n")
app.close()
