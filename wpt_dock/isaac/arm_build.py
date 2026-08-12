"""The cute_arm as a kinematic USD chain, plus kinematic carrying of the payload.

Why kinematic and not an articulation
-------------------------------------
The arm is built as a nested ``Xform`` chain whose joint angles are written every frame,
not as three more PhysX joints on the robot's articulation.  That is a deliberate scope
line, and it is worth being explicit about what it does and does not buy.

*Not* simulated: servo torque and stall, gearbox backlash, the arm's inertia reacting on
the chassis, and friction-based grasping.  Four SG90s holding a payload at 24 cm is
genuinely marginal on real hardware and would deserve a dynamic model if the question were
"can it lift this".

*Still* simulated, and it is the question that matters here: where the gripper ends up in
the world.  Placement error is dominated by the **base pose**, because a 3-DOF arm has no
sensor pointed at the shelf -- it goes where its own joint angles say, from wherever the
robot happens to be parked.  So the accuracy of the whole pick-and-place is the accuracy of
the WPT alignment, and that part is fully simulated, noise and all.  Adding servo dynamics
would change the timing and not the answer.

The payload is a dynamic rigid body that goes kinematic only while it is carried, and is handed
back to the solver at release so it settles under gravity.  What is *not* modelled is the grasp
itself: the jaws close onto the box rather than through it, but they do not squeeze it, and
nothing but this code is holding it.  ``GripperAttachment`` says exactly what that costs and
records the dead ends -- including one claim it stated as fact and had to retract.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from ..arm import ArmPose, ArmSpec
from .stl_mesh import add_stl_mesh, try_read
from .usd_helpers import (
    add_box,
    add_cylinder,
    add_physics_material,
    bind_physics_material,
    set_display_colour,
    set_transform,
)


def _hide(stage, path: str) -> None:
    """Make a prim invisible without removing it.

    Visibility is a render-time property, so a hidden stand-in keeps whatever role it had -- the
    same trick ``robot_build`` uses to keep measured collision boxes working behind the grafted
    ROBOTIS meshes.
    """
    prim = stage.GetPrimAtPath(Sdf.Path(path))
    if prim and prim.IsValid():
        imageable = UsdGeom.Imageable(prim)
        if imageable:
            imageable.MakeInvisible()

_METAL = (0.72, 0.73, 0.76)
_SERVO = (0.10, 0.10, 0.12)
_LINK = (0.88, 0.88, 0.90)
_JAW = (0.20, 0.22, 0.26)
_HORN = (0.93, 0.93, 0.95)

# SG90-class micro servo, from the datasheet outline.  These are what the printed brackets are
# built around, so the bracket dimensions below are derived from them rather than chosen.
SERVO_BODY = (0.0228, 0.0122, 0.0226)   # m, body without the mounting flanges
SERVO_FLANGE = (0.0322, 0.0122, 0.0025)  # m, the lugs the bracket bolts through
SERVO_HORN_R = 0.0058                    # m, output disc radius
BRACKET_T = 0.0025                       # m, printed wall thickness


#: Logical arm part -> the upstream STL stems that provide it, most specific first.
#:
#: **The upstream names are inverted relative to this module's, and that is a trap.**
#: ``elevenMiles/Robotic_Arm_Seven`` names its links by where they sit in the chain read from the
#: gripper down, so ``lower_arm.stl`` is the *shoulder-to-elbow* segment -- what this module calls
#: ``upper_link`` -- and ``upper_arm.stl`` is the *elbow-to-gripper* segment, the forearm.  Wiring
#: them the obvious way round produces an arm with both links present, both the right length, and
#: the wrong shapes on the wrong joints: nothing throws and no test catches it.
#:
#: Worse, ``gripper_upper_arm.stl`` also contains the substring ``upper_arm``.  Matching by
#: substring alone would hand the wrist bracket to the forearm.  So the primary match is on the
#: exact stem, and the wrist is resolved before the forearm regardless.
ARM_STL_STEMS: dict[str, tuple[str, ...]] = {
    "base": ("base_slim",),                    # the pedestal housing the base servo
    "turret": ("base_rotation",),              # turntable on the base servo's horn
    "gripper_bracket": ("gripper_upper_arm",),  # resolved BEFORE fore_link, see above
    "upper_link": ("lower_arm",),              # shoulder -> elbow, 120 mm  (NOT "upper_arm")
    "fore_link": ("upper_arm",),               # elbow -> gripper, 120 mm  (NOT "lower_arm")
    "jaw": ("gripper",),                       # both jaws, probably in one mesh
}

#: Never loaded even if present.  The user's build omits the large base and the pen holder; the
#: pen holder ships as two variants, so three files are excluded to honour "those two parts".
#: The upstream README's own print table lists exactly the six used files plus ``base_large``,
#: which corroborates that the pen holder is an alternative end effector nobody printed.
ARM_STL_EXCLUDE: tuple[str, ...] = ("base_large", "pen_holder")


def discover_arm_stls(directory: str) -> tuple[dict[str, str], list[str]]:
    """Map logical arm parts to STL files in ``directory``.  Returns ``(found, notes)``.

    Returns only what it finds.  Every part has a procedural fallback, so a partial set is
    useful -- the same degradation the printed top plate and the rear rack already have, for the
    same reason: the mission depends on the numbers in ``config``, never on the geometry.

    Nothing is downloaded here, by design.  Put the files in ``assets/arm/`` and they are picked
    up; ``assets/*.stl`` is gitignored so the hardware design is not published by pushing.
    """
    import glob
    import os

    found: dict[str, str] = {}
    notes: list[str] = []
    # Deduplicated by normalised case: on Windows the two globs match the same files twice, and
    # the duplicates show up in the build log as every excluded part being listed twice.
    seen: dict[str, str] = {}
    for pattern in ("*.stl", "*.STL"):
        for p in glob.glob(os.path.join(directory, pattern)):
            seen.setdefault(os.path.normcase(os.path.abspath(p)), p)
    paths = sorted(seen.values())
    if not paths:
        return found, [f"no arm STLs in {directory}; using the procedural parts"]

    def stem(p: str) -> str:
        base = os.path.basename(p).rsplit(".", 1)[0]
        return base.lower().replace("-", "_").replace(" ", "_")

    skipped = [p for p in paths if any(bad in stem(p) for bad in ARM_STL_EXCLUDE)]
    usable = [p for p in paths if p not in skipped]
    for part, stems in ARM_STL_STEMS.items():
        for want in stems:
            hit = next((p for p in usable if stem(p) == want and p not in found.values()), None)
            if hit is not None:
                found[part] = hit
                break
    notes.append(f"arm STLs: matched {len(found)}/{len(ARM_STL_STEMS)} parts from {directory}")
    for part in ARM_STL_STEMS:
        if part not in found:
            notes.append(f"  {part}: no STL, using the procedural part")
    if skipped:
        notes.append("  excluded by request: "
                     + ", ".join(os.path.basename(p) for p in sorted(skipped)))
    return found, notes


def _servo(stage, path: str, centre: tuple[float, float, float], *, horn_axis: str = "Y",
           horn_offset: float = 0.0) -> None:
    """A micro servo: body, mounting flanges, and a round output horn.

    The horn is the part that makes an arm stop looking like a stack of cubes -- it is the one
    visibly circular feature at every joint, and every joint has one.
    """
    add_box(stage, f"{path}/body", SERVO_BODY, centre, _SERVO)
    add_box(stage, f"{path}/flange", SERVO_FLANGE,
            (centre[0], centre[1], centre[2] + 0.5 * SERVO_BODY[2] - 0.004), _SERVO)
    if horn_axis == "Y":
        hc = (centre[0], centre[1] + horn_offset, centre[2] + 0.5 * SERVO_BODY[2] + 0.002)
    else:
        hc = (centre[0], centre[1], centre[2] + 0.5 * SERVO_BODY[2] + 0.002 + horn_offset)
    add_cylinder(stage, f"{path}/horn", SERVO_HORN_R, 0.0035, hc, _HORN, axis=horn_axis)


def _u_bracket(stage, path: str, *, span: float, length: float, height: float,
               centre: tuple[float, float, float], colour=_LINK, back: bool = True) -> None:
    """A printed U bracket: two parallel cheeks and, optionally, a back plate joining them.

    This is the shape the whole arm is made of.  ``span`` is the inside distance between the
    cheeks -- the servo sits in that gap -- so the outside width is ``span + 2 * BRACKET_T`` and
    the bracket wraps its servo instead of pretending to be a solid block.
    """
    cx, cy, cz = centre
    half = 0.5 * span + 0.5 * BRACKET_T
    for sy, tag in ((+1, "cheek_p"), (-1, "cheek_n")):
        add_box(stage, f"{path}/{tag}", (length, BRACKET_T, height),
                (cx, cy + sy * half, cz), colour)
    if back:
        add_box(stage, f"{path}/back", (BRACKET_T, span + 2 * BRACKET_T, height),
                (cx - 0.5 * length + 0.5 * BRACKET_T, cy, cz), colour)


def _jaw(stage, path: str, spec: ArmSpec, sign: float = 1.0) -> None:
    """One scissor finger: the vertical pivot screw, the lever, and the gripping face.

    Authored in the finger's own frame, whose origin **is** the pivot screw, so the lever runs out
    along +x and the face stands at the far end.  ``sign`` puts the face on the inner side.

    The pad is the one part whose dimensions are load bearing rather than cosmetic --
    ``jaw_pad_thickness`` is what makes the usable gap ``span - 4 mm``, and the grasp angle is
    computed from it.  So it is taken from the spec, and changing it changes both the picture and
    the kinematics together.
    """
    r = spec.jaw_tip_reach
    # The vertical pivot -- the servo horn on the moving jaw, a screw on the fixed one.
    add_cylinder(stage, f"{path}/hub", 0.0048, 0.005, (0.0, 0.0, 0.0), _METAL, axis="Z")
    # A flat arm running out to the tip, thin in z, as the printed part is.
    add_box(stage, f"{path}/arm", (r, 0.018, 0.004), (0.5 * r, 0.0, 0.0), _JAW)
    # The hooked tip.  Both jaws' tips sit on the centreline so that at zero swing they meet --
    # that is the structure the user described and photographed, and it is what puts the tool
    # centre point at the tip rather than near the wrist.
    add_box(stage, f"{path}/tip", (0.012, 0.010, 0.016), (r, 0.0, sign * 0.008), _JAW)
    # No collider on the face, and the reason is structural rather than a scope choice.  It hangs
    # off the arm, which hangs off the chassis -- an articulation link, i.e. a rigid body.  A
    # collider on that link whose *local* transform is rewritten every frame (the jaw separation
    # is) makes Kit re-enter its own tasking mutex: "Recursion not allowed",
    # carb/tasking/Mutex.cpp:103, a modal assertion that stops the process dead on the first
    # physics step.  The physical pads in ``build_pad_bodies`` exist precisely because collision
    # has to live outside the articulation, not inside it.


def _link_plate(stage, path: str, *, length: float, height: float, thickness: float,
                centre: tuple[float, float, float], colour=_LINK) -> None:
    """An arm link: a plate along +X with **rounded ends**, as a printed link is made.

    Axes matter here and getting them wrong is easy: the link runs along its frame's +X, its
    plate plane is X-Z, and ``thickness`` is the printed wall in **Y**.  The end bosses turn
    about the joint axis, which for both bending joints is Y -- so they are cylinders about Y
    with the plate's own height as their diameter.  Putting them about Z instead produces discs
    lying flat across the arm, which is a different machine.

    The rounding is not decoration: a link pivots on a boss at each end, so the material follows
    the bolt circle.  A bare box reads as machined stock, which this arm is not.
    """
    cx, cy, cz = centre
    r = 0.5 * height
    add_box(stage, f"{path}/web", (max(1e-4, length - height), thickness, height),
            (cx, cy, cz), colour)
    for sx, tag in ((-1, "boss_near"), (+1, "boss_far")):
        add_cylinder(stage, f"{path}/{tag}", r, thickness,
                     (cx + sx * 0.5 * (length - height), cy, cz), colour, axis="Y")


@dataclass
class ArmRig:
    """Handles for the transform ops that get written every frame."""

    spec: ArmSpec
    root: str
    base_rot: UsdGeom.XformOp
    shoulder_rot: UsdGeom.XformOp
    elbow_rot: UsdGeom.XformOp
    jaw_left: UsdGeom.XformOp
    jaw_right: UsdGeom.XformOp
    tcp_path: str
    grip_path: str = ""
    #: World-space pad bodies, ``(translate_op, orient_op, sign)`` per side.  These exist outside
    #: the robot's prim tree on purpose -- see ``build_pad_bodies``.
    pads: tuple = ()
    pose: ArmPose = field(default=ArmPose(0.0, 0.0, 0.0))
    gripper: float = 0.0

    def set_pose(self, pose: ArmPose, gripper: float) -> None:
        """Write the joint angles.

        Signs: USD ``rotateY`` by a positive angle takes +X towards -Z, while this project's
        convention has ``shoulder`` positive *up*.  Hence the negation -- and hence writing it
        down, because a sign error here produces an arm that mirrors its whole workspace
        below the deck and still looks like a plausible arm.
        """
        self.pose = pose
        self.gripper = gripper
        self.base_rot.Set(math.degrees(pose.base))
        self.shoulder_rot.Set(-math.degrees(pose.shoulder))
        self.elbow_rot.Set(-math.degrees(pose.elbow))
        # One moving jaw, one fixed jaw.  Only ``jaw_left`` -- the part on the servo horn -- turns;
        # the fixed jaw belongs to the bracket and has no op to write.  Zero is shut, tips
        # touching, which is why the sign is positive to open.
        self.jaw_left.Set(math.degrees(self.spec.jaw_rotation(gripper)))

    def tcp_world(self, stage) -> Gf.Matrix4d:
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        return cache.GetLocalToWorldTransform(stage.GetPrimAtPath(Sdf.Path(self.tcp_path)))

    def exclude_pads_from(self, stage, prim_path: str) -> None:
        """Stop the pads colliding with the robot they are part of.

        Not optional.  The pads are kinematic, which in PhysX means infinitely massive, and they
        are *inside* the robot's own volume by construction -- so every pad-versus-robot contact
        is spurious and each one kicks the articulation.  Left unfiltered they threw the robot to
        (-0.095, 0.109) at -156 degrees within six seconds; the emergency-stop guard caught it,
        which is the only reason it read as a mission failure rather than a mystery.
        """
        from pxr import UsdPhysics as _UP

        for translate, _orient, _sign, _x in self.pads:
            prim = translate.GetAttr().GetPrim()
            api = _UP.FilteredPairsAPI.Apply(prim)
            api.CreateFilteredPairsRel().AddTarget(Sdf.Path(prim_path))

    def drive_pads(self, stage) -> None:
        """Put the two physical pads where the gripper's jaws are, in world space.

        Called every control step, after ``set_pose``.  The pads cannot live under the gripper
        prim -- see ``build_pad_bodies`` -- so their world poses are recomputed from the gripper
        frame each step instead of being composed by USD.
        """
        if not self.pads:
            return
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        m = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(Sdf.Path(self.grip_path)))
        rot = m.ExtractRotationQuat()
        quat = Gf.Quatf(float(rot.GetReal()), *[float(v) for v in rot.GetImaginary()])
        half = 0.5 * self.spec.gripper_span(self.gripper)
        for translate, orient, sign, local_x in self.pads:
            # The pad's centre in the gripper frame, then through the gripper's world transform.
            local = Gf.Vec3d(local_x, sign * half, 0.0)
            translate.Set(m.Transform(local))
            orient.Set(quat)


#: Parts whose STL replaces the procedural geometry, and where each one goes.
#:
#: ``frame`` is which joint frame it is parented to, so a loaded mesh rotates with its joint.
#: The transforms are *starting points*, not measurements: an STL's origin is wherever the CAD
#: happened to put it, and none of these files publish a datum.  ``build_arm`` prints each mesh's
#: measured extent into the build notes -- the same thing ``robot_build`` does for the printed top
#: plate -- so the offsets can be corrected in one pass once the files are actually present.
#:
#: ``jaw`` is absent on purpose.  ``gripper.stl`` is a single 742-triangle mesh that almost
#: certainly holds *both* fingers, and the fingers have to move independently: the jaw separation
#: is driven every frame and the grasp angle is a measured quantity, not decoration.  One mesh
#: cannot be two moving parts, so the procedural pads stay and the file is skipped with a note.
#: Splitting it into left and right in a mesh editor is what would change that.
#: Measured extents of the six used files, in mm, so the placements below are derived rather than
#: guessed.  Read straight off the meshes:
#:
#:   base_slim              150.00 x 150.00 x  22.00     min (-75.00, -75.00, -20.00)
#:   base_rotation           65.00 x  38.00 x  46.20     min (-35.00, -18.00, +13.80)
#:   lower_arm              155.17 x  29.61 x   4.00     min (+32.39, +84.20, +101.41)
#:   upper_arm              130.08 x  11.22 x  25.11     min (+89.92, +108.88, +165.52)
#:   gripper_upper_arm      149.00 x  19.01 x  23.00     min (+83.00, +103.59, +165.50)
#:   gripper                 92.16 x  29.00 x  20.44     min (+139.68, +97.56, +179.50)
#:
#: Two things follow that are not obvious.  First, the four distal parts have large positive
#: origins in one shared frame -- they were exported **in place from an assembly**, so their own
#: coordinates are meaningless in isolation and every one has to be recentred onto its joint.
#: Second, ``lower_arm`` is 4 mm thick **in Z** while every other link is thick in Y: it is
#: exported in its *print* pose, lying flat on the bed, which is exactly what the upstream print
#: table implies by giving it no supports.  It gets rotated upright; the others do not.
#:
#: The links are recentred in all three axes and then pushed out to their segment midpoint.  That
#: is exact to the extent the end bosses are symmetric, and they are: 155.17 mm of part for a
#: 120 mm pivot spacing is 17.6 mm of boss at each end, and 130.08 mm is 5 mm at each end.
ARM_STL_PLACEMENT: dict[str, dict] = {
    "base": dict(frame="root", translate=(0.0, 0.0, 0.0),
                 recenter_xy=True, recenter_z=False, zero_bottom=True, rot_x_deg=0.0),
    "turret": dict(frame="yaw", translate=(0.0, 0.0, BRACKET_T + SERVO_BODY[2]),
                   recenter_xy=True, recenter_z=False, zero_bottom=True, rot_x_deg=0.0),
    "upper_link": dict(frame="shoulder", translate=("half_upper", 0.0, 0.0),
                       recenter_xy=True, recenter_z=True, zero_bottom=False, rot_x_deg=90.0),
    "fore_link": dict(frame="elbow", translate=("half_fore", 0.0, 0.0),
                      recenter_xy=True, recenter_z=True, zero_bottom=False, rot_x_deg=0.0),
    "gripper_bracket": dict(frame="gripper", translate=(0.0, 0.0, 0.0),
                            recenter_xy=True, recenter_z=True, zero_bottom=False, rot_x_deg=0.0),
}


def build_pad_bodies(stage, spec: ArmSpec, root: str = "/World/gripper_pads") -> tuple:
    """The two jaw pads as **kinematic rigid bodies**, outside the robot's prim tree.

    This is what makes the grasp a contact grasp rather than a teleport, and the placement is
    forced by PhysX rather than chosen: a rigid body may not be nested inside another rigid body,
    and the arm hangs off the chassis, which is an articulation link.  So pads authored under the
    gripper would be rigid bodies inside a rigid body -- undefined at best.  They live at the top
    level and their world poses are driven from the gripper frame every control step
    (``ArmRig.drive_pads``).

    Kinematic, not static: a static collider that is moved by writing its transform does not
    impart motion to what it touches -- PhysX does not compute a velocity for it, so it tunnels or
    simply fails to push.  A kinematic body is the correct object for something whose motion is
    commanded and whose contacts must still be real, and it is also what a position-controlled
    servo is: infinitely stiff against the load, which is why the box cannot squeeze the jaws open.

    The pads are the only part of the arm with collision.  The links do not have it, and that is
    still deliberate -- what the grasp needs is the two faces that touch the object.

    **This does not currently work, and it is off by default.**  With the pads present PhysX dies
    on the first physics step: no traceback, no summary, the process simply ends after
    "Simulation App Startup Complete" -- the same signature as the fixed-joint attempt recorded in
    ``GripperAttachment``.  Two things were ruled out first and both were real bugs worth keeping
    fixed: the pads spawned at the world origin, which is coil 1, so they materialised inside the
    robot and threw it to (-0.095, 0.109) at -156 degrees (caught by the emergency-stop guard);
    and pad-versus-robot contact is spurious by construction because the pads *are* the robot, so
    it is filtered.  Neither cured the crash.  What is left untried is a top-level kinematic body
    whose pose is written every step while the payload it touches is dynamic -- the payload itself
    does exactly that and survives, so the difference is worth finding.  Kept behind
    ``RunConfig.physical_grasp`` so the next attempt starts from working code.
    """
    from pxr import UsdPhysics as _UP

    UsdGeom.Xform.Define(stage, Sdf.Path(root))
    material = add_physics_material(stage, "/World/Looks/PhysicsJaw", 0.95, 0.85)
    pads = []
    pad_x = 0.021          # matches the procedural pad's centre in the gripper frame
    for tag, sign in (("left", +1.0), ("right", -1.0)):
        body_path = f"{root}/{tag}"
        body = UsdGeom.Xform.Define(stage, Sdf.Path(body_path))
        rb = _UP.RigidBodyAPI.Apply(body.GetPrim())
        rb.CreateKinematicEnabledAttr(True)
        _UP.MassAPI.Apply(body.GetPrim()).CreateMassAttr(0.004)
        xf = UsdGeom.Xformable(body)
        translate = xf.AddTranslateOp()
        orient = xf.AddOrientOp()
        orient.Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        box = add_box(stage, f"{body_path}/pad", (0.014, spec.jaw_pad_thickness, 0.016),
                      (0.0, 0.0, 0.0), _JAW, collision=True)
        bind_physics_material(box.GetPrim(), material)
        pads.append((translate, orient, sign, pad_x))
    return tuple(pads)


def _mount_jaw_pair(stage, path: str, grip_path: str, spec: ArmSpec,
                    notes: list[str] | None) -> None:
    """Mount ``gripper.stl`` twice, mirrored, one on each jaw frame.

    An earlier revision refused to mount this file at all, on the stated grounds that one
    742-triangle mesh "almost certainly holds both fingers" and fingers have to move
    independently.  That was an inference and it was wrong: a connected-component count over the
    welded vertices returns **one** component, and the part is not mirror-symmetric in y (1689
    vertices below its own y centre against 537 above).  So it is a *single* finger, the real
    gripper uses two copies facing each other, and it can be used.  Measure the file.

    Placement is derived from the mesh, not chosen.  Slicing along x shows a 2 mm-thick blade for
    the first 46 mm with a tall boss at x = 0, then a flat plate fanning out to 29 mm wide.  The
    boss is the pivot, so the part is shifted to put that boss on the jaw frame's origin.

    What this cannot do is recover the part's *assembled* orientation.  The six files are laid out
    on a print bed, not in an assembly -- proven by their thin axes disagreeing: this jaw and the
    shoulder link are thin in z while the forearm is thin in y.  So the blade is authored along
    +x, which is a default and not a measurement.  ``notes`` therefore reports where the pad face
    lands against the modelled tool centre point, because if the real blade is angled in the
    assembly then 92 mm of jaw along +x is too long -- it would add 92 mm to a documented 240 mm
    reach -- and that number is the thing to look at.
    """
    data = try_read(path)
    if data is None:
        return
    tris = data.triangles
    lo = tris.reshape(-1, 3).min(axis=0)
    hi = tris.reshape(-1, 3).max(axis=0)
    # The pivot boss: the slice within 10 mm of the blade end.
    near = tris.reshape(-1, 3)
    near = near[near[:, 0] <= lo[0] + 10.0]
    piv = (lo[0], 0.5 * (near[:, 1].min() + near[:, 1].max()),
           0.5 * (near[:, 2].min() + near[:, 2].max()))
    s = 0.001
    for tag, mirror in (("jaw_left", False), ("jaw_right", True)):
        # Mirroring negates y, so the pivot's y offset has to be negated with it.
        py = -piv[1] if mirror else piv[1]
        add_stl_mesh(
            stage, f"{grip_path}/{tag}/jaw_stl", data, scale=s,
            translate=(-piv[0] * s, -py * s, -piv[2] * s),
            colour=_JAW, mirror_y=mirror,
        )
    if notes is not None:
        reach = (hi[0] - lo[0]) * s
        notes.append(
            f"  jaw: {os.path.basename(path)} -- one connected component, not two, so it is a "
            f"single finger mounted twice (mirrored).  {data.describe(s)}"
        )
        notes.append(
            f"  !! the jaw is {reach * 1000:.0f} mm from pivot to tip, but the model's tool centre "
            f"point is {spec.l_tool * 1000:.0f} mm out.  Authored along +x as a default -- the "
            f"files are a print-bed layout, not an assembly, so the blade's real angle is not "
            f"recoverable from them.  If it looks wrong, the blade is angled on the real gripper."
        )


def build_arm(
    stage,
    chassis: str,
    spec: ArmSpec,
    *,
    plate_top_local_z: float,
    stl_dir: str | None = None,
    notes: list[str] | None = None,
    spec_plate_size: tuple[float, float] | None = None,
    physical_grasp: bool = False,
) -> ArmRig:
    """Author the arm on the custom top plate.

    ``plate_top_local_z`` is the plate's upper surface in the chassis frame, so the arm sits
    on the plate the cameras are mounted in rather than at a guessed height -- the same
    measured chain everything else on this robot hangs off.
    """
    # Two frames, not one, and the split is the whole point.
    #
    # ``root`` is **static**: it carries the mount offset and holds the parts that are bolted to
    # the robot's plate -- the base and the base servo's body.  ``yaw`` carries the base rotation
    # and holds everything the servo actually turns.  An earlier version put the rotation on
    # ``root``, so the base plate and the servo body swung round with the arm; the user saw it
    # immediately.  A servo cannot rotate its own stator, and a base plate bolted to the deck
    # cannot rotate at all.
    root = f"{chassis}/arm"
    base = UsdGeom.Xform.Define(stage, Sdf.Path(root))
    xb = UsdGeom.Xformable(base)
    xb.ClearXformOpOrder()
    xb.AddTranslateOp().Set(Gf.Vec3d(spec.mount_offset[0], spec.mount_offset[1], plate_top_local_z))

    # --- static: the base and the servo bolted into it -------------------------------------
    # The base plate is the part the user excluded from the build ("base large"), so what is
    # modelled is the *small* base: a disc footprint just wide enough for the bolt circle.
    add_cylinder(stage, f"{root}/base_plate", 0.026, BRACKET_T,
                 (0.0, 0.0, 0.5 * BRACKET_T), _LINK, axis="Z")
    _servo(stage, f"{root}/base_servo", (0.0, 0.0, BRACKET_T + 0.5 * SERVO_BODY[2]),
           horn_axis="Z")

    # --- rotating: everything from the turntable up ----------------------------------------
    yaw_path = f"{root}/yaw"
    yw = UsdGeom.Xform.Define(stage, Sdf.Path(yaw_path))
    xy = UsdGeom.Xformable(yw)
    xy.ClearXformOpOrder()
    base_rot = xy.AddRotateZOp()
    base_rot.Set(0.0)

    # Turntable: the disc the whole arm rotates on, riding on the base servo's horn.
    add_cylinder(stage, f"{yaw_path}/turret", 0.021, 0.005,
                 (0.0, 0.0, BRACKET_T + SERVO_BODY[2] + 0.006), _METAL, axis="Z")

    shoulder_path = f"{yaw_path}/shoulder"
    sh = UsdGeom.Xform.Define(stage, Sdf.Path(shoulder_path))
    xs = UsdGeom.Xformable(sh)
    xs.ClearXformOpOrder()
    xs.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, spec.base_height))
    shoulder_rot = xs.AddRotateYOp()
    shoulder_rot.Set(0.0)

    # --- shoulder: a U bracket standing on the turntable, wrapping the shoulder servo ------
    # The bracket is authored in the *rotating* frame, so it turns with the joint exactly as the
    # printed part does.  Its cheeks straddle the servo, which is why the gap is the servo's
    # own width rather than a chosen number.
    _u_bracket(stage, f"{shoulder_path}/bracket", span=SERVO_BODY[1], length=0.030,
               height=0.030, centre=(0.004, 0.0, -0.006))
    _servo(stage, f"{shoulder_path}/servo", (0.0, 0.0, -0.008),
           horn_axis="Y", horn_offset=0.5 * SERVO_BODY[1] + 0.004)

    # Upper arm: a plate along +X with rounded ends, bolted to the shoulder horn.
    _link_plate(stage, f"{shoulder_path}/upper_link", length=spec.l_upper, height=0.022,
                thickness=0.005, centre=(0.5 * spec.l_upper, 0.0, 0.0))
    # The second cheek, on the far side: a printed arm carries the load in two shear planes,
    # which is also why the elbow servo ends up sandwiched rather than cantilevered.
    _link_plate(stage, f"{shoulder_path}/upper_link_far", length=spec.l_upper, height=0.022,
                thickness=0.005, centre=(0.5 * spec.l_upper, -0.0165, 0.0))

    elbow_path = f"{shoulder_path}/elbow"
    el = UsdGeom.Xform.Define(stage, Sdf.Path(elbow_path))
    xe = UsdGeom.Xformable(el)
    xe.ClearXformOpOrder()
    xe.AddTranslateOp().Set(Gf.Vec3d(spec.l_upper, 0.0, 0.0))
    elbow_rot = xe.AddRotateYOp()
    elbow_rot.Set(0.0)

    # --- elbow: servo sandwiched between the upper arm's cheeks, driving the forearm -------
    _servo(stage, f"{elbow_path}/servo", (0.006, -0.008, 0.0),
           horn_axis="Y", horn_offset=0.5 * SERVO_BODY[1] + 0.003)
    _link_plate(stage, f"{elbow_path}/fore_link", length=spec.l_fore, height=0.018,
                thickness=0.004, centre=(0.5 * spec.l_fore, 0.008, 0.0))
    _link_plate(stage, f"{elbow_path}/fore_link_far", length=spec.l_fore, height=0.018,
                thickness=0.004, centre=(0.5 * spec.l_fore, -0.014, 0.0))

    grip_path = f"{elbow_path}/gripper"
    gp = UsdGeom.Xform.Define(stage, Sdf.Path(grip_path))
    xg = UsdGeom.Xformable(gp)
    xg.ClearXformOpOrder()
    xg.AddTranslateOp().Set(Gf.Vec3d(spec.l_fore, 0.0, 0.0))
    # --- gripper: servo in a bracket, driving two pivoting jaws -----------------------------
    _u_bracket(stage, f"{grip_path}/bracket", span=SERVO_BODY[1], length=0.026, height=0.026,
               centre=(0.002, 0.0, 0.0))
    _servo(stage, f"{grip_path}/servo", (-0.002, 0.0, 0.0),
           horn_axis="Y", horn_offset=0.5 * SERVO_BODY[1] + 0.003)

    # Jaw rest separations come from the spec, not from literals.  They used to be authored at
    # +-0.012 and then immediately overwritten by ``set_pose``, so the numbers in the file said
    # one thing and the scene showed another; anyone reading them would have inferred a 20 mm
    # gap that the mechanism never uses.
    # Each finger's frame sits **at its pivot screw** and carries a rotation, not a slide.  The
    # pivot is where the force is applied and where the part is bolted, so putting the frame
    # anywhere else -- as the sliding version did -- puts the whole mechanism's reference point in
    # the wrong place, which is what the user saw.
    # The moving jaw: on the servo horn, turning about that horn's vertical axis.
    mv = UsdGeom.Xform.Define(stage, Sdf.Path(f"{grip_path}/jaw_left"))
    xm = UsdGeom.Xformable(mv)
    xm.ClearXformOpOrder()
    xm.AddTranslateOp().Set(Gf.Vec3d(spec.jaw_pivot_x, 0.0, 0.006))
    jaw_left = xm.AddRotateZOp()
    jaw_left.Set(0.0)
    _jaw(stage, f"{grip_path}/jaw_left", spec, +1.0)

    # The fixed jaw: part of the bracket, reaching forward to meet the moving one.  It gets a
    # frame so the two are authored the same way, but nothing ever writes to it -- ``jaw_right``
    # is kept only so ``ArmRig``'s shape does not change for callers.
    fx = UsdGeom.Xform.Define(stage, Sdf.Path(f"{grip_path}/jaw_fixed"))
    xfx = UsdGeom.Xformable(fx)
    xfx.ClearXformOpOrder()
    xfx.AddTranslateOp().Set(Gf.Vec3d(spec.jaw_pivot_x, 0.0, -0.004))
    jaw_right = xfx.AddRotateZOp()
    jaw_right.Set(0.0)
    _jaw(stage, f"{grip_path}/jaw_fixed", spec, -1.0)

    # The tool centre point: where a grasped object's centre sits.  Between the jaws, out
    # along the forearm.  Everything about carrying and placing is expressed relative to
    # this one prim so there is a single definition of "where the gripper is".
    #
    # Driven from ``spec.l_tool`` rather than a literal, because this offset also has to be in
    # the kinematics -- it was a literal here and absent there, and the resulting 16 mm was
    # enough to bury every placed box in the shelf.  One number, one place.
    tcp_path = f"{grip_path}/tcp"
    tcp = UsdGeom.Xform.Define(stage, Sdf.Path(tcp_path))
    set_transform(UsdGeom.Xformable(tcp), (spec.l_tool, 0.0, 0.0), 0.0, None)

    # --- real printed parts, if the user has them ------------------------------------------
    # Each loaded mesh *replaces* its procedural stand-in: the stand-in is made invisible rather
    # than deleted, exactly as the robot's collision boxes are, so nothing that referenced it
    # breaks and the shapes never double up.
    if stl_dir:
        frames = {"root": root, "yaw": yaw_path, "shoulder": shoulder_path,
                  "elbow": elbow_path, "gripper": grip_path,
                  "jaw_left": f"{grip_path}/jaw_left", "jaw_right": f"{grip_path}/jaw_right"}
        procedural = {
            "base": (f"{root}/base_plate",),
            "turret": (f"{yaw_path}/turret",),
            "jaw": (f"{grip_path}/jaw_left/finger", f"{grip_path}/jaw_left/pad",
                    f"{grip_path}/jaw_left/pivot", f"{grip_path}/jaw_right/finger",
                    f"{grip_path}/jaw_right/pad", f"{grip_path}/jaw_right/pivot"),
            "upper_link": (f"{shoulder_path}/upper_link", f"{shoulder_path}/upper_link_far"),
            "fore_link": (f"{elbow_path}/fore_link", f"{elbow_path}/fore_link_far"),
            "gripper_bracket": (f"{grip_path}/bracket",),
        }
        found, stl_notes = discover_arm_stls(stl_dir)
        if notes is not None:
            notes.extend(stl_notes)
        for part, path in found.items():
            if part == "jaw":
                _mount_jaw_pair(stage, path, grip_path, spec, notes)
                for stand_in in procedural.get("jaw", ()):
                    _hide(stage, stand_in)
                continue
            place = ARM_STL_PLACEMENT.get(part)
            if place is None:
                if notes is not None:
                    notes.append(f"  {part}: STL present but not mounted -- see ARM_STL_PLACEMENT")
                continue
            data = try_read(path)
            if data is None:
                continue
            # Symbolic offsets, resolved from the spec so the mesh lands on its segment midpoint
            # rather than on a number typed in here.
            tx = place["translate"][0]
            if tx == "half_upper":
                tx = 0.5 * spec.l_upper
            elif tx == "half_fore":
                tx = 0.5 * spec.l_fore
            mesh_path = f"{frames[place['frame']]}/{part}_stl"
            add_stl_mesh(
                stage, mesh_path, data, scale=0.001,
                translate=(tx, place["translate"][1], place["translate"][2]),
                colour=_LINK, recenter_xy=place["recenter_xy"],
                recenter_z=place["recenter_z"], zero_bottom=place["zero_bottom"],
                rot_x_deg=place["rot_x_deg"],
            )
            for stand_in in procedural.get(part, ()):
                _hide(stage, stand_in)
            if notes is not None:
                notes.append(f"  {part}: {os.path.basename(path)} -- {data.describe(0.001)}")
                if part == "base":
                    lo, hi = data.bounds
                    ext = (hi - lo) * 0.001
                    plate = spec_plate_size
                    if plate and (ext[0] > plate[0] or ext[1] > plate[1]):
                        notes.append(
                            f"  !! base STL is {ext[0] * 1000:.0f} x {ext[1] * 1000:.0f} mm but the "
                            f"printed plate is {plate[0] * 1000:.0f} x {plate[1] * 1000:.0f} mm -- "
                            f"it overhangs.  Either the arm is not bolted to this plate on the real "
                            f"robot, or base_slim carries mounting wings that are trimmed in the "
                            f"build.  Reported rather than hidden or silently scaled."
                        )

    rig = ArmRig(
        spec=spec, root=root, base_rot=base_rot, shoulder_rot=shoulder_rot,
        elbow_rot=elbow_rot, jaw_left=jaw_left, jaw_right=jaw_right, tcp_path=tcp_path,
        grip_path=grip_path,
        pads=build_pad_bodies(stage, spec) if physical_grasp else (),
    )
    from ..arm import HOME

    rig.set_pose(HOME, spec.gripper_open)
    return rig


# ---------------------------------------------------------------------------
# Payload carrying
# ---------------------------------------------------------------------------


def _level_orientation(tcp_world: Gf.Matrix4d) -> Gf.Quatf:
    """A level (no roll, no pitch) orientation whose +Y follows the tool's jaw axis.

    USD's ``Gf.Matrix4d`` is row-major, so row *i* is the world image of the local basis
    vector *e_i*; row 1 is therefore the jaw axis.  Projecting it into the ground plane and
    rebuilding a pure-Z rotation gives the orientation a gripped cube actually holds.

    Degenerate case: if the jaw axis were vertical the projection would vanish, and the yaw is
    then genuinely undefined.  This arm cannot reach that pose -- the jaw axis is the rotation
    axis of both bending joints, so it is always horizontal -- but a zero-length ``atan2`` is
    silent rather than loud, so it is handled instead of assumed away.
    """
    jx, jy = float(tcp_world[1][0]), float(tcp_world[1][1])
    if math.hypot(jx, jy) < 1e-9:
        return Gf.Quatf(1.0, 0.0, 0.0, 0.0)
    # R_z(theta) maps +Y to (-sin theta, cos theta), so match that against the jaw axis.
    theta = math.atan2(-jx, jy)
    return Gf.Quatf(math.cos(0.5 * theta), 0.0, 0.0, math.sin(0.5 * theta))


class GripperAttachment:
    """Drives the payload's pose from the tool centre point while it is held.

    What is physical here and what is not, stated plainly, because two earlier versions of
    this docstring claimed more than the code delivered:

    **Physical.**  The payload has mass and a collider, and so do the conveyor surface, the drop
    table, and the robot's custom top plate.  That last one is what turned the arm trajectory
    into a real constraint: lifting the payload straight up before slewing it over the deck is
    necessary rather than cosmetic, and there is a unit test that measures the clearance.  The
    placement error is the arm's genuine achieved position, driven by the WPT alignment error at
    the base, which is fully simulated with noise.

    The payload is a **dynamic** rigid body that rests on the conveyor under gravity, becomes
    kinematic for the seconds it is carried, and is handed back to the solver at release so it
    falls the last fraction of a millimetre and settles.  So the drop is a real settle, and the
    reported placement error is a settled position rather than a number this code wrote down.

    **Not physical.**  The jaws do not squeeze -- the arm links carry no colliders, so the grasp
    cannot slip and the payload cannot be knocked loose.  They do close *onto* the box rather
    than through it (see ``ArmSpec.grip_angle_for``), but nothing is holding it except this
    class.  A friction grasp would need jaw colliders and material tuning, which is a study in
    its own right and not the question this project is asking.

    Two dead ends are recorded because each looked right going in, and one because it was
    recorded here as fact and was not:

    1. A ``UsdPhysics.FixedJoint`` to a kinematic gripper anchor, enabled at grasp.  It
       **crashed PhysX** -- a native fault in ``_physx.pyd`` about three seconds in.  Toggling
       ``physics:jointEnabled`` on a live joint is not the attribute-only write it appears to
       be; PhysX rebuilds the joint and does not survive it here.
    2. Zeroing ``physics:velocity`` on the USD prim before handing the body back.  That
       attribute is an initial condition, not live state, so it changes nothing.  The physics
       API is the right handle -- see ``release``.
    3. **A retracted claim.**  An earlier version of this docstring stated that PhysX does not
       pick up ``kinematicEnabled`` mid-simulation, and used that to justify a payload that was
       kinematic for the entire run -- deterministic placement with no settle.  It is false.
       The flip works on the frame it is written, confirmed against this install by watching
       ``get_physics_stats()`` move ``numKinematicBodies`` 1 -> 0 with no step in between.  The
       claim was inferred from a run in which the payload was ejected to z = -23.5 m, and the
       real cause of that was the collision documented in
       ``exclude_from_collision_with`` -- the same one that launched the robot off the board.
       One failure, two symptoms, and attributing it to the nearest suspicious API cost a
       working feature.  Measure the thing.
    """

    def __init__(self, stage, payload_path: str, *, physical_grasp: bool = True) -> None:
        self.stage = stage
        self.payload_path = payload_path
        # With a physical grasp the payload is never pose-driven: it stays dynamic and the two
        # kinematic pads hold it by friction.  ``False`` restores the kinematic carry, kept as a
        # fallback because a friction grasp can genuinely drop the box and that is a real result
        # worth being able to compare against rather than a bug to hide.
        self.physical_grasp = physical_grasp
        self.held = False
        prim = stage.GetPrimAtPath(Sdf.Path(payload_path))
        if not prim or not prim.IsValid():
            raise RuntimeError(f"payload prim missing: {payload_path}")
        self._prim = prim
        self._rb = UsdPhysics.RigidBodyAPI(prim)
        self._kinematic = self._rb.CreateKinematicEnabledAttr(False)
        self._view = None
        self._np = None
        # The payload prim is a bare wrapper Xform whose child carries the geometry, so it
        # arrives with no transform ops and these two are the only ones it will ever have.
        # Deliberately *not* clearing an existing op order: doing that on a prim built by
        # ``add_box`` drops its scale op, and a 35 mm box silently becomes a 1 m cube.
        xf = UsdGeom.Xformable(prim)
        self._translate = xf.AddTranslateOp()
        self._orient = xf.AddOrientOp()
        self._orient.Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    def exclude_from_collision_with(self, prim_path: str) -> None:
        """Stop the payload and the robot from colliding, authored before the sim starts.

        This is not a convenience -- without it the run is destroyed.  The payload is
        kinematic, which in PhysX means *infinitely massive*: when its collider swept into the
        newly-added top-plate collider during the carry move, it did not get blocked, it
        **punted the whole articulation off the board**.  The robot ended up at
        (0.85, 1.23, -0.23) m -- through the floor -- and froze there, while the mission logic
        went on reporting a clean dock because the fault was downstream of everything it
        measures.  Diagnosing that took two runs of blaming the arm; ``_frame_report`` in
        ``pickplace_runner`` exists so the next person spends one.

        Filtering *this one pair* is the correct scope.  The plate collider still makes the
        robot solid against the shelves and crates, and the payload still collides with the
        world; what is removed is a contact that can only ever be wrong, because a body the
        gripper is holding must not be able to push the gripper.  What keeps the payload from
        passing *through* the deck is the trajectory -- lift in place, then slew -- and that
        has a unit test measuring the clearance.
        """
        api = UsdPhysics.FilteredPairsAPI.Apply(self._prim)
        api.CreateFilteredPairsRel().AddTarget(Sdf.Path(prim_path))

    def place_at(self, position: tuple[float, float, float]) -> None:
        self._translate.Set(Gf.Vec3d(*position))

    def world_position(self) -> Gf.Vec3d:
        """Where the payload actually is, asked of **physics** and not of USD.

        This distinction is the whole difference between measuring a result and reporting an
        intention.  Once the body is dynamic the solver owns its pose and USD holds whatever was
        last authored -- which after a release is the pose the jaws let go at.  Reading that back
        would report a perfect placement no matter what the box then did, including falling
        through the floor.  The USD read is kept only as the pre-simulation fallback.
        """
        try:
            pos, _ = self._rigid_view().get_world_poses()
            return Gf.Vec3d(float(pos[0][0]), float(pos[0][1]), float(pos[0][2]))
        except Exception:                         # noqa: BLE001 - before play, or no view
            cache = UsdGeom.XformCache(Usd.TimeCode.Default())
            return cache.GetLocalToWorldTransform(self._prim).ExtractTranslation()

    def rest_report(self) -> str:
        """Speed and tilt at the moment of asking -- the evidence that it settled.

        A frozen kinematic body and a settled dynamic one look identical in a position readout.
        They do not look identical here: a settled box reads a speed of a few tenths of a
        millimetre per second and a pitch of zero.
        """
        try:
            view = self._rigid_view()
            vel = view.get_velocities()[0]
            lin = float(self._np.linalg.norm(vel[:3]))
            ang = float(self._np.linalg.norm(vel[3:]))
            _, quat = view.get_world_poses()
            w, x, y, z = (float(v) for v in quat[0])
            # Tilt of the body's own +Z away from world +Z.
            up_z = 1.0 - 2.0 * (x * x + y * y)
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, up_z))))
            return (f"at rest: speed {lin * 1000:.2f} mm/s, spin {math.degrees(ang):.2f} deg/s, "
                    f"tilt {tilt:.2f} deg off level")
        except Exception as exc:                  # noqa: BLE001
            return f"rest state unavailable ({exc})"

    def _rigid_view(self):
        """The physics-API handle, built lazily because it needs a live simulation view.

        ``reset_xform_properties=False`` matters: the default rewrites the prim's transform ops,
        and this class owns them.  Letting it rewrite them risks the failure recorded above --
        a dropped scale op turning a 25 mm box into a 1 m cube.
        """
        if self._view is None:
            import numpy as np
            from isaacsim.core.prims import RigidPrim

            self._np = np
            self._view = RigidPrim(prim_paths_expr=self.payload_path, name="payload_view",
                                   reset_xform_properties=False)
            self._view.initialize()
        return self._view

    def grasp(self) -> None:
        """Take the payload out of the solver's hands and drive it from the tool centre point.

        PhysX picks this attribute up on the frame it is written -- measured, not assumed:
        ``get_physics_stats()`` moves ``numKinematicBodies`` 0 -> 1 immediately, with no step in
        between.
        """
        self.held = True
        if not self.physical_grasp:
            self._kinematic.Set(True)

    def release(self) -> None:
        """Hand the payload back to the solver so it falls the last fraction and settles.

        Order is not interchangeable and was measured: flip the attribute first, *then* zero the
        velocity.  ``set_velocities`` on a body that is still kinematic returns success and
        changes nothing -- PhysX derives a kinematic actor's reported velocity from its pose
        targets, so it is not settable.

        The zeroing is insurance rather than the fix: the kinematic-to-dynamic transition
        discards velocity by itself (a probe released a 60 m/s kinematic actor and it fell
        straight down).  It is kept because it costs nothing and because ``get_velocities()``
        reading exactly zero afterwards is a cheap assertion that the hand-off happened.

        Deliberately *not* done here, all measured as unnecessary or harmful:
        ``flush_changes()``, ``enable_rigid_body_physics()``, ``wake_up()``, and a fixed joint.
        Cycling ``rigidBodyEnabled`` or removing and re-applying ``RigidBodyAPI`` to force the
        change permanently breaks the tensor view ("Failed to get rigid body velocities from
        backend") and is not needed.
        """
        self.held = False
        if self.physical_grasp:
            # Nothing to hand back -- it was dynamic the whole time.  Opening the jaws is the
            # release, and gravity does the rest.
            return
        self._kinematic.Set(False)
        try:
            view = self._rigid_view()
            view.set_velocities(self._np.zeros((1, 6), dtype=self._np.float32))
        except Exception as exc:                  # noqa: BLE001 - report and carry on
            # The transition discards velocity without this, so a failure here costs realism in
            # an edge case, not the run.  Silence would be worse than a line of output.
            print(f"  [arm] note: could not zero the payload velocity ({exc})", flush=True)

    def follow(self, rig: ArmRig) -> None:
        """Drive the payload to the tool centre point -- only in the non-physical fallback.

        With ``physical_grasp`` set, this does nothing at all: the payload is a dynamic body and
        the two kinematic pads carry it by contact and friction, which is the whole point.
        Writing its pose on top of that would be exactly the "보여주기 식" the user objected to,
        and worse than before, because it would silently override a real result.

        The payload takes the tool's **position and yaw only** -- it stays level.  That is not
        a simplification, it is what the mechanism does.  The shoulder and elbow rotate about
        the body +Y, and the jaws separate along that same +Y, so the jaw axis stays horizontal
        at every arm pose.  Two flat pads closing on a cube's opposite *vertical* faces cannot
        impose a pitch on it: the box was sitting level on the shelf, and it is still level in
        the jaws.

        Copying the tool's full rotation instead -- which is what this did -- rolled the cube
        by the forearm's elevation.  At the place pose that is about 81 degrees, so the box was
        carried and then left standing on a corner, and being kinematic it stayed there.  The
        user's report was that the object is placed tilted and stays tilted.

        The yaw is taken from the jaw axis rather than from ``rig.pose.base``, so it stays
        correct without this function having to know how the arm is mounted or which way the
        chassis is pointing.
        """
        if self.physical_grasp or not self.held:
            return
        mat = rig.tcp_world(self.stage)
        self._translate.Set(mat.ExtractTranslation())
        self._orient.Set(_level_orientation(mat))
