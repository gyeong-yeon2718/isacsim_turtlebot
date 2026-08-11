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

The payload is carried the same way: it is a kinematic rigid body whose transform is driven
from the tool centre point.  It still has mass and a collider and so do the shelves and the
robot's deck, but the grasp itself is not a friction grasp and the release is not a gravity
settle.  ``GripperAttachment`` says exactly what that costs and lists the three dynamic
approaches that were tried and failed, one of which crashed PhysX.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from ..arm import ArmPose, ArmSpec
from .usd_helpers import add_box, add_cylinder, set_display_colour, set_transform

_METAL = (0.72, 0.73, 0.76)
_SERVO = (0.10, 0.10, 0.12)
_LINK = (0.88, 0.88, 0.90)
_JAW = (0.20, 0.22, 0.26)


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


def build_arm(
    stage,
    chassis: str,
    spec: ArmSpec,
    *,
    plate_top_local_z: float,
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

    # Base servo block and the rotating turret it drives.
    add_box(stage, f"{root}/servo_base", (0.023, 0.013, 0.026), (0.0, 0.0, 0.013), _SERVO)
    add_cylinder(stage, f"{root}/turret", 0.019, 0.008, (0.0, 0.0, 0.030), _METAL)

    shoulder_path = f"{root}/shoulder"
    sh = UsdGeom.Xform.Define(stage, Sdf.Path(shoulder_path))
    xs = UsdGeom.Xformable(sh)
    xs.ClearXformOpOrder()
    xs.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, spec.base_height))
    shoulder_rot = xs.AddRotateYOp()
    shoulder_rot.Set(0.0)
    add_box(stage, f"{shoulder_path}/servo", (0.023, 0.013, 0.026), (0.0, 0.0, -0.008), _SERVO)

    # Upper arm: a slim beam along +X of the shoulder frame.
    add_box(stage, f"{shoulder_path}/upper_link", (spec.l_upper, 0.014, 0.022),
            (0.5 * spec.l_upper, 0.0, 0.0), _LINK)

    elbow_path = f"{shoulder_path}/elbow"
    el = UsdGeom.Xform.Define(stage, Sdf.Path(elbow_path))
    xe = UsdGeom.Xformable(el)
    xe.ClearXformOpOrder()
    xe.AddTranslateOp().Set(Gf.Vec3d(spec.l_upper, 0.0, 0.0))
    elbow_rot = xe.AddRotateYOp()
    elbow_rot.Set(0.0)
    add_box(stage, f"{elbow_path}/servo", (0.023, 0.013, 0.026), (0.0, 0.0, 0.0), _SERVO)
    add_box(stage, f"{elbow_path}/fore_link", (spec.l_fore, 0.012, 0.018),
            (0.5 * spec.l_fore, 0.0, 0.0), _LINK)

    grip_path = f"{elbow_path}/gripper"
    gp = UsdGeom.Xform.Define(stage, Sdf.Path(grip_path))
    xg = UsdGeom.Xformable(gp)
    xg.ClearXformOpOrder()
    xg.AddTranslateOp().Set(Gf.Vec3d(spec.l_fore, 0.0, 0.0))
    add_box(stage, f"{grip_path}/servo", (0.023, 0.013, 0.026), (-0.004, 0.0, 0.0), _SERVO)

    jaw_l = UsdGeom.Xform.Define(stage, Sdf.Path(f"{grip_path}/jaw_left"))
    jl = UsdGeom.Xformable(jaw_l)
    jl.ClearXformOpOrder()
    jaw_left = jl.AddTranslateOp()
    jaw_left.Set(Gf.Vec3d(0.0, 0.012, 0.0))
    add_box(stage, f"{grip_path}/jaw_left/pad", (0.026, 0.004, 0.016), (0.013, 0.0, 0.0), _JAW)

    jaw_r = UsdGeom.Xform.Define(stage, Sdf.Path(f"{grip_path}/jaw_right"))
    jr = UsdGeom.Xformable(jaw_r)
    jr.ClearXformOpOrder()
    jaw_right = jr.AddTranslateOp()
    jaw_right.Set(Gf.Vec3d(0.0, -0.012, 0.0))
    add_box(stage, f"{grip_path}/jaw_right/pad", (0.026, 0.004, 0.016), (0.013, 0.0, 0.0), _JAW)

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


class GripperAttachment:
    """Drives the payload's pose from the tool centre point while it is held.

    What is physical here and what is not, stated plainly, because two earlier versions of
    this docstring claimed more than the code delivered:

    **Physical.**  The payload has mass and a collider, and so do the shelves, the station
    tops, and -- as of this revision -- the robot's custom top plate.  That last one is what
    turned the arm trajectory into a real constraint: lifting the payload straight up before
    slewing it over the deck is now necessary rather than cosmetic, and there is a unit test
    that measures the clearance.  The placement error is the arm's genuine achieved position,
    driven by the WPT alignment error at the base, which is fully simulated with noise.

    **Not physical.**  The jaws do not squeeze -- the arm links carry no colliders, so the
    grasp cannot slip and the payload cannot be knocked loose.  And the payload is kinematic
    for the whole run, so the drop is placement, not a settle.

    Three approaches to a genuinely dynamic grasp were tried and are recorded here rather than
    quietly dropped, because each one looked right going in:

    1. A ``UsdPhysics.FixedJoint`` to a kinematic gripper anchor, enabled at grasp.  It
       **crashed PhysX** -- a native fault in ``_physx.pyd`` about three seconds in.  Toggling
       ``physics:jointEnabled`` on a live joint is not the attribute-only write it appears to
       be; PhysX rebuilds the joint and does not survive it here.
    2. Dynamic body, switched to kinematic at grasp and back at release.  PhysX does not pick
       up ``kinematicEnabled`` mid-simulation, so the body stayed dynamic the whole time and
       the solver and this class fought over its pose.  It looked fine only because nothing
       else touched the payload; adding the shelf collider exposed it -- a zero-gap initial
       overlap flicked the box off the shelf at t = 0 and it finished a metre from the pad.
    3. Zeroing ``physics:velocity`` before handing the body back.  That attribute is an
       initial condition, not live state, so it changed nothing and the payload was ejected to
       z = -23.5 m.

    Hence the current design: kinematic from creation, pose always authored.  A friction grasp
    would need jaw colliders and material tuning, which is a study in its own right and not the
    question this project is asking.
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
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        return cache.GetLocalToWorldTransform(self._prim).ExtractTranslation()

    def grasp(self) -> None:
        """Start driving the payload from the tool centre point.

        No physics state changes here.  The payload is authored kinematic at build time (see
        ``warehouse.build_warehouse``) precisely so that grasping and releasing are pure
        bookkeeping -- switching ``kinematicEnabled`` at runtime does not reach PhysX, and the
        earlier version that tried it produced a payload that PhysX was still simulating while
        this class wrote transforms on top of it.
        """
        self.held = True

    def release(self) -> None:
        """Stop driving the payload.  It stays on the pad, kinematic, where the jaws left it.

        A kinematic body holds its authored pose, so this is deterministic placement rather
        than a gravity settle: nothing tips, rolls, or falls the last fraction of a
        millimetre.  The reported placement error is the arm's achieved position, which is the
        quantity being measured, but it is not a settled-under-gravity result and should not be
        read as one.
        """
        self.held = False

    def follow(self, rig: ArmRig) -> None:
        """Drive the payload to the tool centre point.  Only while held."""
        if not self.held:
            return
        mat = rig.tcp_world(self.stage)
        self._translate.Set(mat.ExtractTranslation())
        rot = mat.ExtractRotationQuat()
        self._orient.Set(Gf.Quatf(float(rot.GetReal()), *[float(v) for v in rot.GetImaginary()]))
