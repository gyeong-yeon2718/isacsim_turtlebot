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
from .usd_helpers import add_box, add_cylinder, set_display_colour, set_transform


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


def _jaw(stage, path: str, spec: ArmSpec) -> None:
    """One finger: a pivot boss, an L-shaped finger, and the flat pad that touches the object.

    The pad is the one part whose dimensions are load bearing rather than cosmetic --
    ``jaw_pad_thickness`` is what makes the usable gap ``span - 4 mm``, and the grasp angle is
    computed from it.  So it is taken from the spec, and changing it changes both the picture and
    the kinematics together.
    """
    t = spec.jaw_pad_thickness
    # Pivot boss at the jaw's own origin, turning about Z with the finger.
    add_cylinder(stage, f"{path}/pivot", 0.004, 0.010, (0.0, 0.0, 0.0), _JAW, axis="Z")
    # The finger reaching forward, then the pad facing inward across the gap.
    add_box(stage, f"{path}/finger", (0.020, t, 0.014), (0.011, 0.0, 0.0), _JAW)
    add_box(stage, f"{path}/pad", (0.014, t, 0.016), (0.021, 0.0, 0.0), _JAW)


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
        half = 0.5 * self.spec.gripper_span(gripper)
        self.jaw_left.Set(Gf.Vec3d(0.0, half, 0.0))
        self.jaw_right.Set(Gf.Vec3d(0.0, -half, 0.0))

    def tcp_world(self, stage) -> Gf.Matrix4d:
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        return cache.GetLocalToWorldTransform(stage.GetPrimAtPath(Sdf.Path(self.tcp_path)))


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
ARM_STL_PLACEMENT: dict[str, dict] = {
    "base": dict(frame="root", translate=(0.0, 0.0, 0.0), recenter_xy=True, zero_bottom=True),
    "turret": dict(frame="root", translate=(0.0, 0.0, BRACKET_T + SERVO_BODY[2]),
                   recenter_xy=True, zero_bottom=True),
    "upper_link": dict(frame="shoulder", translate=(0.0, 0.0, 0.0),
                       recenter_xy=False, zero_bottom=False),
    "fore_link": dict(frame="elbow", translate=(0.0, 0.0, 0.0),
                      recenter_xy=False, zero_bottom=False),
    "gripper_bracket": dict(frame="gripper", translate=(0.0, 0.0, 0.0),
                            recenter_xy=False, zero_bottom=False),
}


def build_arm(
    stage,
    chassis: str,
    spec: ArmSpec,
    *,
    plate_top_local_z: float,
    stl_dir: str | None = None,
    notes: list[str] | None = None,
) -> ArmRig:
    """Author the arm on the custom top plate.

    ``plate_top_local_z`` is the plate's upper surface in the chassis frame, so the arm sits
    on the plate the cameras are mounted in rather than at a guessed height -- the same
    measured chain everything else on this robot hangs off.
    """
    root = f"{chassis}/arm"
    base = UsdGeom.Xform.Define(stage, Sdf.Path(root))
    xb = UsdGeom.Xformable(base)
    xb.ClearXformOpOrder()
    xb.AddTranslateOp().Set(Gf.Vec3d(spec.mount_offset[0], spec.mount_offset[1], plate_top_local_z))
    base_rot = xb.AddRotateZOp()
    base_rot.Set(0.0)

    # --- base: a mounting plate, the base servo standing in it, and the turntable ---------
    # The base plate is the part the user excluded from the build ("base large"), so what is
    # modelled is the *small* base: a disc footprint just wide enough for the bolt circle.
    add_cylinder(stage, f"{root}/base_plate", 0.026, BRACKET_T,
                 (0.0, 0.0, 0.5 * BRACKET_T), _LINK, axis="Z")
    _servo(stage, f"{root}/base_servo", (0.0, 0.0, BRACKET_T + 0.5 * SERVO_BODY[2]),
           horn_axis="Z")
    # Turntable: the disc the whole arm rotates on, riding on the base servo's horn.
    add_cylinder(stage, f"{root}/turret", 0.021, 0.005,
                 (0.0, 0.0, BRACKET_T + SERVO_BODY[2] + 0.006), _METAL, axis="Z")

    shoulder_path = f"{root}/shoulder"
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
    rest_half = 0.5 * spec.gripper_span(spec.gripper_open)
    jaw_l = UsdGeom.Xform.Define(stage, Sdf.Path(f"{grip_path}/jaw_left"))
    jl = UsdGeom.Xformable(jaw_l)
    jl.ClearXformOpOrder()
    jaw_left = jl.AddTranslateOp()
    jaw_left.Set(Gf.Vec3d(0.0, rest_half, 0.0))
    _jaw(stage, f"{grip_path}/jaw_left", spec)

    jaw_r = UsdGeom.Xform.Define(stage, Sdf.Path(f"{grip_path}/jaw_right"))
    jr = UsdGeom.Xformable(jaw_r)
    jr.ClearXformOpOrder()
    jaw_right = jr.AddTranslateOp()
    jaw_right.Set(Gf.Vec3d(0.0, -rest_half, 0.0))
    _jaw(stage, f"{grip_path}/jaw_right", spec)

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
        frames = {"root": root, "shoulder": shoulder_path, "elbow": elbow_path,
                  "gripper": grip_path}
        procedural = {
            "base": (f"{root}/base_plate",),
            "turret": (f"{root}/turret",),
            "upper_link": (f"{shoulder_path}/upper_link", f"{shoulder_path}/upper_link_far"),
            "fore_link": (f"{elbow_path}/fore_link", f"{elbow_path}/fore_link_far"),
            "gripper_bracket": (f"{grip_path}/bracket",),
        }
        found, stl_notes = discover_arm_stls(stl_dir)
        if notes is not None:
            notes.extend(stl_notes)
        for part, path in found.items():
            place = ARM_STL_PLACEMENT.get(part)
            if place is None:
                if notes is not None:
                    notes.append(f"  {part}: STL present but not mounted -- see ARM_STL_PLACEMENT")
                continue
            data = try_read(path)
            if data is None:
                continue
            mesh_path = f"{frames[place['frame']]}/{part}_stl"
            add_stl_mesh(
                stage, mesh_path, data, scale=0.001, translate=place["translate"],
                colour=_LINK, recenter_xy=place["recenter_xy"],
                zero_bottom=place["zero_bottom"],
            )
            for stand_in in procedural.get(part, ()):
                _hide(stage, stand_in)
            if notes is not None:
                notes.append(f"  {part}: {os.path.basename(path)} -- {data.describe(0.001)}")

    rig = ArmRig(
        spec=spec, root=root, base_rot=base_rot, shoulder_rot=shoulder_rot,
        elbow_rot=elbow_rot, jaw_left=jaw_left, jaw_right=jaw_right, tcp_path=tcp_path,
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

    def __init__(self, stage, payload_path: str) -> None:
        self.stage = stage
        self.payload_path = payload_path
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
        self._kinematic.Set(False)
        try:
            view = self._rigid_view()
            view.set_velocities(self._np.zeros((1, 6), dtype=self._np.float32))
        except Exception as exc:                  # noqa: BLE001 - report and carry on
            # The transition discards velocity without this, so a failure here costs realism in
            # an edge case, not the run.  Silence would be worse than a line of output.
            print(f"  [arm] note: could not zero the payload velocity ({exc})", flush=True)

    def follow(self, rig: ArmRig) -> None:
        """Drive the payload to the tool centre point.  Only while held.

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
        if not self.held:
            return
        mat = rig.tcp_world(self.stage)
        self._translate.Set(mat.ExtractTranslation())
        self._orient.Set(_level_orientation(mat))
