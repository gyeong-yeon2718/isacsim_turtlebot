"""Build the robot: verified physics, ROBOTIS geometry, the user's printed parts on top.

The split, and why
------------------
The lecture material's route is to use the official ``turtlebot3_burger.usd`` (p.6), and
that is where the geometry comes from.  Its **drivetrain**, however, does not work at the
speeds this task needs, and that was measured rather than assumed
(``tools/probe_tb3_drive.py``, ``tools/probe_tb3_gains.py``):

* a *constant* wheel-rate shortfall of about 1.0 rad/s at every commanded rate, so below
  roughly 1 rad/s -- 33 mm/s of body speed -- the wheels do not turn forwards at all.  The
  mission taper reaches 12-14 mm/s over the last centimetre of a leg, so the robot stopped
  9 mm short and sat there until the mission timed out;
* the articulation's simulated gains are already ``kds = 5.7e8``; forcing them to 1e3 or
  1e5 through ``Articulation.set_gains`` moves the numbers by less than run-to-run noise;
* USD-level ``damping`` / ``maxForce`` edits have no effect at all, because
  ``WheeledRobot.post_reset()`` writes the gains into the articulation afterwards -- the
  1.5 N.m and 1e6 N.m sweeps came back bit-identical.

So the resistance is somewhere in the published asset's own physics configuration.  Rather
than keep digging, the parts are used for what each is good for: the **procedural
articulation below** provides the physics, which is measured good (3-6 % slip, 300/300
docking success), and the asset's **meshes are grafted onto it** for the looks, including
onto the driven wheel bodies so they spin.

Visual mesh over simplified collision is a standard split.  Stating it plainly: the robot
on screen is real ROBOTIS geometry; the robot in the solver is a two-wheel differential
drive built to the Burger's published dimensions.  Also note the asset was still worth
loading for a second reason -- only the **burger** variant exists, which settles the variant
question the upstream repos never answered, and its chassis mesh tops out at 151.5 mm,
which is where the printed plate goes.

Physics choices that each cost a debug cycle
--------------------------------------------
* **Casters must not share the ground plane with the wheels.**  With both caster spheres
  tangent to z = 0, exactly like the wheels, the solver put the load on the casters --
  friction 0.04 by design -- and the driven wheels spun in the air: 98 % slip, 43 mm of
  travel in 49 s.  Nothing about the controller was wrong.  So the rear caster touches,
  the front sits 2 mm high as a pitch stop, and the mass centre is 10 mm behind the axle,
  matching the real robot's rear battery box.  That puts 87 % of the weight on the driven
  wheels.
* **Torque ceiling** of 1.5 N.m, near the XL430-W250 stall figure.  Without it the wheels
  are infinitely strong and every timing result flatters itself.
* **Separate friction for wheels and casters.**  A caster with tyre friction is a brake
  that fights every turn; a tyre with caster friction cannot pull.  Both look like
  controller faults from the outside.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from ..config import CameraSpec, Settings, default_cameras
from .stl_mesh import add_stl_mesh, try_read
from .usd_helpers import (
    add_box,
    add_cylinder,
    add_physics_material,
    add_sphere,
    bind_physics_material,
    set_display_colour,
)

LEFT_JOINT = "wheel_left_joint"
RIGHT_JOINT = "wheel_right_joint"

_DARK = (0.13, 0.13, 0.14)
_TYRE = (0.06, 0.06, 0.07)
_PCB = (0.10, 0.35, 0.16)
_LIDAR_KEYWORDS = ("scan", "lidar", "lds", "laser")


@dataclass
class RobotHandles:
    prim_path: str
    wheel_joints: tuple[str, str]
    chassis_path: str
    camera_prims: dict[str, str]
    plate_lens_z: float          # m, measured lens height above the running surface
    visuals_from_asset: bool
    notes: list[str]


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------


def _geometry_tops(stage, root: str, exclude: tuple[str, ...]) -> list[tuple[str, float, float]]:
    """Per-geometry world z extents, excluding named subtrees, tallest first.

    A whole-subtree bounding box is the wrong measurement for "where does the deck sit":
    on the Burger it answers 172.5 mm -- the top of the lidar mast -- which put the printed
    plate 37 mm too high, floating above a part that is not on the real robot any more.
    Hiding the lidar does not help either, because bounds come from geometry extents and do
    not care what is visible.  So the exclusion is explicit and the result is logged, which
    turns a choice between two plausible numbers into a one-look check.
    """
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    )
    out: list[tuple[str, float, float]] = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(Sdf.Path(root))):
        path = str(prim.GetPath())
        if any(k in path.lower() for k in exclude):
            continue
        if not UsdGeom.Boundable(prim):
            continue
        rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if rng.IsEmpty():
            continue
        out.append((path, float(rng.GetMin()[2]), float(rng.GetMax()[2])))
    out.sort(key=lambda r: -r[2])
    return out


def _make_invisible(stage, path: str) -> None:
    """Hide a prim without disabling it.

    Used on the procedural collision shapes once the real meshes are grafted on.
    Visibility is a render-time property, so the colliders keep working -- which is the
    whole point: keep the physics that was measured good, show the geometry that looks
    right.
    """
    prim = stage.GetPrimAtPath(Sdf.Path(path))
    if prim and prim.IsValid():
        imageable = UsdGeom.Imageable(prim)
        if imageable:
            imageable.MakeInvisible()


# ---------------------------------------------------------------------------
# Grafting the official meshes
# ---------------------------------------------------------------------------


def graft_asset_meshes(
    stage,
    url: str,
    targets: dict[str, tuple[str, Gf.Matrix4d]],
    skip: tuple[str, ...] = _LIDAR_KEYWORDS,
) -> list[str]:
    """Copy the official TurtleBot3 meshes onto our own articulation.

    ``targets`` maps a path keyword to ``(destination prim, transform of that destination
    relative to the robot root)``.  Keywords are tried in insertion order, so the wheels
    must come before the chassis -- ``/base_link/visuals`` and ``/wheel_left_link/visuals``
    both contain "base" once the full path is lowercased.

    Mesh *data* is copied rather than referenced, so no rigid bodies, joints or
    articulation roots come along with it.  The source subtree is then deactivated, which
    prunes it from composition entirely -- stronger and cheaper than stripping APIs one at
    a time, and it guarantees the asset's own articulation never reaches the solver.
    """
    from isaacsim.core.utils.stage import add_reference_to_stage

    scratch = "/World/_tb3_mesh_source"
    grafted: list[str] = []

    add_reference_to_stage(usd_path=url, prim_path=scratch)
    src_root = stage.GetPrimAtPath(Sdf.Path(scratch))
    if not src_root or not src_root.IsValid():
        raise RuntimeError("could not reference the mesh source asset")
    # The wrapper is a small stub whose real content is a payload, and payloads are not
    # composed until asked for.  Without this the subtree looks empty.
    src_root.Load(Usd.LoadWithDescendants)

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    root_inv = cache.GetLocalToWorldTransform(src_root).GetInverse()

    for prim in Usd.PrimRange(src_root):
        mesh = UsdGeom.Mesh(prim)
        if not mesh:
            continue
        low = str(prim.GetPath()).lower()
        if any(k in low for k in skip):
            continue
        match = next(((k, *targets[k]) for k in targets if k in low), None)
        if match is None:
            continue
        keyword, dest_path, dest_xform = match

        points = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not points or not counts or not indices:
            continue

        # Row-vector convention: mesh-local -> destination-local is
        # (mesh -> source root) * inverse(destination -> source root).
        local = cache.GetLocalToWorldTransform(prim) * root_inv * dest_xform.GetInverse()
        copy = UsdGeom.Mesh.Define(stage, Sdf.Path(f"{dest_path}/tb3_visual_{keyword}"))
        copy.CreatePointsAttr(points)
        copy.CreateFaceVertexCountsAttr(counts)
        copy.CreateFaceVertexIndicesAttr(indices)
        normals = mesh.GetNormalsAttr().Get()
        if normals:
            copy.CreateNormalsAttr(normals)
            copy.SetNormalsInterpolation(mesh.GetNormalsInterpolation())
        copy.CreateSubdivisionSchemeAttr(mesh.GetSubdivisionSchemeAttr().Get() or "none")
        extent = mesh.GetExtentAttr().Get()
        if extent:
            copy.CreateExtentAttr(extent)
        xf = UsdGeom.Xformable(copy)
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(local)
        set_display_colour(copy.GetPrim(), (0.17, 0.17, 0.19))
        grafted.append(f"{keyword}({len(points)} pts)")

    src_root.SetActive(False)
    return grafted


def asset_url(settings: Settings) -> str:
    """Full URL of the intact TurtleBot3 package.

    ``tools/probe_tb3_assets.py`` loads all three published variants and counts what
    arrives: the 5.1 and 4.5 packages compose with links and joints but **zero mesh
    prims**, because their base layer references
    ``configuration/turtlebot3_burger_physics.usd@</visuals/...>`` and that prim path is
    not in the file.  4.2 is self-contained and intact.  The version segment of whatever
    asset root is configured is substituted, so a user pointing at a mirror still gets the
    right file.
    """
    from isaacsim.storage.native import get_assets_root_path

    root = get_assets_root_path()
    override = settings.robot.asset_version_override
    if override:
        head, _, tail = root.rpartition("/")
        if head and tail:
            root = f"{head}/{override}"
    return root + settings.robot.asset_relative_path


# ---------------------------------------------------------------------------
# Custom superstructure
# ---------------------------------------------------------------------------


def _add_camera_marker(stage, parent: str, cam: CameraSpec) -> str:
    """A camera board seated in its plate hole, lens looking down.

    The transform is ``translate -> rotateZ(yaw) -> rotateY(-pitch)``, matching
    ``apriltag.CameraModel`` exactly, so in that frame the optical axis is local +X -- which
    is why the lens cylinder is authored along X.  A marker pointing somewhere the model
    does not makes the picture disagree with the geometry the estimator uses, and that is a
    miserable thing to diagnose from a screenshot.

    The board sits behind the lens along the optical axis, i.e. up inside the 9 mm pocket,
    which is how the real ones are mounted: ribbon connectors facing up, lens facing down
    through the through-hole.
    """
    path = f"{parent}/cameras/{cam.name}"
    xf = UsdGeom.Xform.Define(stage, Sdf.Path(path))
    x = UsdGeom.Xformable(xf)
    x.ClearXformOpOrder()
    x.AddTranslateOp().Set(Gf.Vec3d(*cam.position))
    x.AddRotateZOp().Set(math.degrees(cam.yaw))
    x.AddRotateYOp().Set(math.degrees(-cam.pitch))

    add_box(stage, f"{path}/board", (0.004, 0.019, 0.019), (-0.006, 0.0, 0.0), _PCB)
    add_box(stage, f"{path}/ribbon", (0.002, 0.014, 0.004), (-0.011, 0.0, 0.0), (0.85, 0.84, 0.80))
    add_cylinder(stage, f"{path}/lens", 0.0042, 0.007, (0.0005, 0.0, 0.0), (0.04, 0.04, 0.07),
                 axis="X")
    return path


def _attach_custom_parts(
    stage,
    settings: Settings,
    chassis: str,
    *,
    deck_top_z: float,
    chassis_origin_z: float,
    top_plate_stl: str | None,
    tower_stl: str | None,
    notes: list[str],
) -> tuple[dict[str, str], float]:
    """Mount the printed plate, battery box, CC/CV module, Rx coil and cameras.

    ``deck_top_z`` is the *measured* world height of the robot's top deck.  The printed
    plate **replaces** the stock third tier rather than stacking on it, so it sits flush --
    the 1 mm is only there to keep two coplanar surfaces from z-fighting.  On the Burger the
    three tiers are a single visual mesh and cannot be removed individually, so flush
    mounting is what makes it read as a replacement instead of a fourth tier.
    """
    r = settings.robot

    def local(z: float) -> float:
        return z - chassis_origin_z

    # Tier 1 is *derived*, not guessed.  The rack's own geometry fixes the spacing: its
    # underside rests on tier 1 and the underside of its top slab touches the top of
    # tier 3, so those two features are 133.2 mm apart and tier 1 follows from the measured
    # deck.  See RobotSpec for the measurements this comes from.
    rack_span = r.rack_top_underside_local - r.rack_bottom_local
    tier1_top_z = deck_top_z - rack_span
    rack_origin_z = tier1_top_z - r.rack_bottom_local          # world z of the rack's local 0
    rack_rear_x = -0.5 * r.base_footprint[1] - r.rear_extension
    notes.append(
        f"rack span {rack_span * 1000:.1f} mm fixes tier 1 at {tier1_top_z * 1000:.1f} mm "
        f"from the measured tier-3 top of {deck_top_z * 1000:.1f} mm"
    )

    # The extra half plate on the back of tier 1, which the rack stands on.
    add_box(
        stage, f"{chassis}/tier1_rear_extension",
        (r.rear_extension, r.base_footprint[0], r.rear_extension_thickness),
        (rack_rear_x + 0.5 * r.rear_extension, 0.0,
         local(tier1_top_z - 0.5 * r.rear_extension_thickness)),
        (0.16, 0.16, 0.17),
    )

    # The rack.  Placed by its own coordinates -- no recentring -- because its local frame is
    # what carries the two contact constraints.  Rear face flush with the extension's rear
    # edge, laterally centred, opening forwards.
    rack = try_read(tower_stl) if tower_stl else None
    if rack is not None:
        notes.append(f"rear rack STL: {rack.describe(0.001)}")
        add_stl_mesh(
            stage, f"{chassis}/rear_rack", rack, scale=0.001,
            translate=(rack_rear_x, -r.rack_y_centre_local, local(rack_origin_z)),
            colour=(0.09, 0.09, 0.10), recenter_xy=False, zero_bottom=False,
        )
        rack_top_z = rack_origin_z + r.rack_top_upper_local
    else:
        notes.append("rear rack STL unavailable; using a box of the measured outline")
        add_box(stage, f"{chassis}/rear_rack",
                (r.rack_size[0], r.rack_size[1], r.rack_size[2]),
                (rack_rear_x + 0.5 * r.rack_size[0], 0.0,
                 local(tier1_top_z + 0.5 * r.rack_size[2])), (0.09, 0.09, 0.10))
        rack_top_z = tier1_top_z + r.rack_size[2]

    # The printed plate sits on the rack's top slab, which is the topmost structure.
    plate_bottom_z = rack_top_z
    plate_thickness = r.plate_size[2]
    lens_z = plate_bottom_z + 0.002        # lens 2 mm into the 9 mm through-hole

    plate = try_read(top_plate_stl) if top_plate_stl else None
    if plate is not None:
        notes.append(f"top plate STL: {plate.describe(0.001)}")
        add_stl_mesh(
            stage, f"{chassis}/custom_top_plate", plate,
            scale=0.001, translate=(0.0, 0.0, local(plate_bottom_z)),
            colour=(0.11, 0.11, 0.12), recenter_xy=True, zero_bottom=True,
        )
    else:
        notes.append("top plate STL unavailable; using a box of the measured outline")
        add_box(stage, f"{chassis}/custom_top_plate",
                (r.plate_size[0], r.plate_size[1], plate_thickness),
                (0.0, 0.0, local(plate_bottom_z + 0.5 * plate_thickness)), (0.11, 0.11, 0.12))

    # The 18650 3S2P pack on the rack's middle shelf.  The CC/CV module that used to sit on
    # risers above the plate is deliberately not drawn: it is electrical hardware with no
    # geometric role in this task, and on the real robot it is not part of the structure the
    # alignment depends on.
    add_box(stage, f"{chassis}/battery_pack", (0.068, 0.075, 0.042),
            (rack_rear_x + 0.040, 0.0, local(rack_origin_z + 0.085 + 0.021)), (0.20, 0.20, 0.24))

    rx = add_cylinder(
        stage, f"{chassis}/rx_coil", r.rx_coil_radius, 0.005,
        (r.rx_coil_offset[0], r.rx_coil_offset[1], local(0.012)), (0.85, 0.55, 0.25),
    )
    set_display_colour(rx.GetPrim(), (0.85, 0.55, 0.25))

    camera_prims: dict[str, str] = {}
    for cam in default_cameras(lens_z=lens_z):
        seated = CameraSpec(
            cam.name, cam.width, cam.height, cam.hfov,
            (cam.position[0], cam.position[1], local(cam.position[2])),
            cam.yaw, cam.pitch, cam.fps,
        )
        camera_prims[cam.name] = _add_camera_marker(stage, chassis, seated)

    notes.append(
        f"deck top measured at z = {deck_top_z * 1000:.1f} mm; printed plate underside "
        f"{plate_bottom_z * 1000:.1f} mm; camera lenses {lens_z * 1000:.1f} mm"
    )
    return camera_prims, lens_z


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------


def _rigid_body(stage, path: str, mass: float, com: tuple[float, float, float] | None = None):
    xform = UsdGeom.Xform.Define(stage, Sdf.Path(path))
    prim = xform.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(float(mass))
    if com is not None:
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*com))
    return xform


def _revolute_wheel_joint(
    stage, path: str, body0: str, body1: str,
    anchor_in_body0: tuple[float, float, float], *, damping: float, max_force: float,
) -> UsdPhysics.RevoluteJoint:
    """A free-spinning revolute joint about body Y, with a velocity drive.

    No limits are authored, which in USD means unlimited rotation -- correct for a wheel,
    and easy to get wrong by leaving the default 0/0 limits, which locks it solid.

    Unit trap *avoided* rather than solved: a USD angular drive's ``targetVelocity`` is in
    **degrees per second** while ``ArticulationAction(joint_velocities=...)`` is in
    **radians per second**.  The authored target is 0 and all run-time commanding goes
    through the Isaac API, so the two never meet.  Mixing them scales commands by 57.3.
    """
    joint = UsdPhysics.RevoluteJoint.Define(stage, Sdf.Path(path))
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    joint.CreateAxisAttr("Y")
    joint.CreateLocalPos0Attr(Gf.Vec3f(*anchor_in_body0))
    joint.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr().Set("force")
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(float(damping))
    drive.CreateTargetVelocityAttr(0.0)
    drive.CreateMaxForceAttr(float(max_force))
    return joint


def build_robot(
    stage,
    settings: Settings,
    *,
    prim_path: str = "/World/turtlebot",
    position: tuple[float, float] = (0.0, 0.0),
    yaw: float = 0.0,
    top_plate_stl: str | None = None,
    tower_stl: str | None = None,
    graft_official_meshes: bool = True,
) -> RobotHandles:
    r = settings.robot
    notes: list[str] = []

    root = UsdGeom.Xform.Define(stage, Sdf.Path(prim_path))
    xf = UsdGeom.Xformable(root)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(position[0], position[1], 0.0))
    xf.AddRotateZOp().Set(math.degrees(yaw))
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())

    wheel_mat = add_physics_material(stage, "/World/Looks/PhysicsWheel", 1.10, 0.95)
    caster_mat = add_physics_material(stage, "/World/Looks/PhysicsCaster", 0.06, 0.04)

    base_path = f"{prim_path}/base_link"
    base = _rigid_body(stage, base_path, r.chassis_mass, com=(-0.010, 0.0, 0.010))
    UsdGeom.Xformable(base).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, r.wheel_radius))

    body_len, body_wid, body_h = r.base_footprint[1], 0.130, 0.056
    add_box(stage, f"{base_path}/hull", (body_len, body_wid, body_h), (0.0, 0.0, 0.012),
            _DARK, collision=True)

    caster_r = 0.008
    ground_z = -r.wheel_radius + caster_r
    for name, cx, lift in (
        ("caster_rear", -0.5 * body_len + 0.014, 0.0),
        ("caster_front", 0.5 * body_len - 0.018, 0.002),
    ):
        sph = add_sphere(stage, f"{base_path}/{name}", caster_r, (cx, 0.0, ground_z + lift),
                         (0.35, 0.35, 0.38), collision=True)
        bind_physics_material(sph.GetPrim(), caster_mat)

    half_sep = 0.5 * r.wheel_separation
    for side, sign, joint_name in (("left", +1.0, LEFT_JOINT), ("right", -1.0, RIGHT_JOINT)):
        wheel_path = f"{prim_path}/wheel_{side}"
        wheel = _rigid_body(stage, wheel_path, r.wheel_mass)
        UsdGeom.Xformable(wheel).AddTranslateOp().Set(
            Gf.Vec3d(0.0, sign * half_sep, r.wheel_radius)
        )
        geo = add_cylinder(stage, f"{wheel_path}/tyre", r.wheel_radius, 0.018, (0.0, 0.0, 0.0),
                           _TYRE, axis="Y", collision=True)
        bind_physics_material(geo.GetPrim(), wheel_mat)
        add_cylinder(stage, f"{wheel_path}/hub", r.wheel_radius * 0.45, 0.020, (0.0, 0.0, 0.0),
                     (0.55, 0.56, 0.58), axis="Y")
        _revolute_wheel_joint(
            stage, f"{prim_path}/{joint_name}", base_path, wheel_path,
            (0.0, sign * half_sep, 0.0), damping=1.0e3, max_force=1.5,
        )

    # --- looks -----------------------------------------------------------
    grafted = False
    if graft_official_meshes:
        try:
            url = asset_url(settings)
            # Wheels before the chassis: every full path contains "base" once lowercased.
            targets = {
                "wheel_left": (f"{prim_path}/wheel_left",
                               Gf.Matrix4d().SetTranslate(Gf.Vec3d(0.0, half_sep, r.wheel_radius))),
                "wheel_right": (f"{prim_path}/wheel_right",
                                Gf.Matrix4d().SetTranslate(Gf.Vec3d(0.0, -half_sep, r.wheel_radius))),
                "base": (base_path,
                         Gf.Matrix4d().SetTranslate(Gf.Vec3d(0.0, 0.0, r.wheel_radius))),
            }
            report = graft_asset_meshes(stage, url, targets)
            if not report:
                raise RuntimeError("asset contained no usable meshes")
            notes.append(f"grafted official ROBOTIS meshes from {url}")
            notes.append(f"  meshes: {report}  (lidar skipped -- removed on the real robot)")
            for hide in (f"{base_path}/hull", f"{base_path}/caster_rear",
                         f"{base_path}/caster_front", f"{prim_path}/wheel_left/tyre",
                         f"{prim_path}/wheel_left/hub", f"{prim_path}/wheel_right/tyre",
                         f"{prim_path}/wheel_right/hub"):
                _make_invisible(stage, hide)
            notes.append("  procedural collision shapes hidden (still colliding, just not drawn)")
            grafted = True
        except Exception as exc:                       # noqa: BLE001
            notes.append(f"could not graft the official meshes ({exc}); using primitives")

    if not grafted:
        for i, z in enumerate((0.052, 0.086, 0.120)):
            add_box(stage, f"{base_path}/tier{i}",
                    (r.base_footprint[1], r.base_footprint[0], 0.004),
                    (0.0, 0.0, z - r.wheel_radius), (0.16, 0.16, 0.17))

    tops = _geometry_tops(stage, prim_path, _LIDAR_KEYWORDS)
    deck_top_z = tops[0][2] if tops else (r.wheel_radius + 0.012 + 0.5 * body_h)
    notes.append("tallest geometry (top z in mm):")
    for path, _z0, z1 in tops[:4]:
        notes.append(f"    {z1 * 1000:7.1f}  {path}")

    camera_prims, lens_z = _attach_custom_parts(
        stage, settings, base_path, deck_top_z=deck_top_z, chassis_origin_z=r.wheel_radius,
        top_plate_stl=top_plate_stl, tower_stl=tower_stl, notes=notes,
    )

    return RobotHandles(
        prim_path=prim_path, wheel_joints=(LEFT_JOINT, RIGHT_JOINT), chassis_path=base_path,
        camera_prims=camera_prims, plate_lens_z=lens_z, visuals_from_asset=grafted, notes=notes,
    )
