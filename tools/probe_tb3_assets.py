r"""Which published TurtleBot3 asset actually loads with its meshes intact?

    C:\isaacsim\python.bat tools\probe_tb3_assets.py

Background: the 5.1 asset at ``Isaac/Robots/Turtlebot/Turtlebot3/turtlebot3_burger.usd``
resolves, and its 11 MB payload composes, but the stage then reports

    Unresolved reference prim path
      @.../configuration/turtlebot3_burger_physics.usd@</visuals/base_footprint>

The *file* is there (6501 bytes); the *prim path* inside it is not.  The result is a
robot with links and joints and no visible geometry -- which is exactly what an
appearance complaint looks like from the outside.

Rather than guess which of the published variants is intact, this loads each one and
counts what actually arrived: boundable prims, world bounds, and the revolute joints the
controller needs.  One app boot, three answers, and the numbers go in the log.
"""

from __future__ import annotations

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from pxr import Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402

from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402

CANDIDATES = {
    "5.1 nested": "/Isaac/Robots/Turtlebot/Turtlebot3/turtlebot3_burger.usd",
    "4.5 flat": "/Isaac/Robots/Turtlebot/turtlebot3_burger.usd",
    "4.2 flat": "/Isaac/Robots/Turtlebot/turtlebot3_burger.usd",
}
ROOTS = {"5.1 nested": "5.1", "4.5 flat": "4.5", "4.2 flat": "4.2"}

BASE = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac"

print(f"configured asset root: {get_assets_root_path(skip_check=True)}", flush=True)

import omni.usd  # noqa: E402

stage = omni.usd.get_context().get_stage()

for i, (label, rel) in enumerate(CANDIDATES.items()):
    url = f"{BASE}/{ROOTS[label]}{rel}"
    path = f"/probe{i}"
    print(f"\n=== {label} ===\n  {url}", flush=True)
    try:
        add_reference_to_stage(usd_path=url, prim_path=path)
        prim = stage.GetPrimAtPath(Sdf.Path(path))
        if not prim or not prim.IsValid():
            print("  FAIL: no prim", flush=True)
            continue
        prim.Load(Usd.LoadWithDescendants)

        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
        )
        boundable = []
        joints = []
        bodies = []
        for p in Usd.PrimRange(prim):
            if p.IsA(UsdPhysics.RevoluteJoint):
                joints.append(p.GetName())
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                bodies.append(p.GetName())
            if UsdGeom.Boundable(p):
                rng = cache.ComputeWorldBound(p).ComputeAlignedRange()
                if not rng.IsEmpty():
                    boundable.append((str(p.GetPath())[len(path):], p.GetTypeName(),
                                      float(rng.GetMin()[2]), float(rng.GetMax()[2])))
        whole = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        print(f"  boundable geometry prims: {len(boundable)}", flush=True)
        print(f"  rigid bodies: {bodies}", flush=True)
        print(f"  revolute joints: {joints}", flush=True)
        if not whole.IsEmpty():
            print(f"  subtree bounds z: {float(whole.GetMin()[2]) * 1000:+.1f} .. "
                  f"{float(whole.GetMax()[2]) * 1000:+.1f} mm", flush=True)
        for rel_path, kind, z0, z1 in sorted(boundable, key=lambda r: -r[3])[:14]:
            print(f"    {z1 * 1000:8.1f} top  {z0 * 1000:8.1f} bot  {kind:12} {rel_path}",
                  flush=True)
    except Exception as exc:                     # noqa: BLE001
        print(f"  FAIL: {exc}", flush=True)

import sys  # noqa: E402

sys.stdout.flush()
simulation_app.close()
