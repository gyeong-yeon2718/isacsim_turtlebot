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

The payload is carried the same way: its rigid body is switched to kinematic and its
transform is driven from the tool centre point, then handed back to the solver on release
so it settles onto the target under gravity.  A friction grasp between two printed jaws
and a cube is a notoriously fiddly thing to tune and would only add a failure mode that is
not being studied.
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
    tcp_path = f"{grip_path}/tcp"
    tcp = UsdGeom.Xform.Define(stage, Sdf.Path(tcp_path))
    set_transform(UsdGeom.Xformable(tcp), (0.016, 0.0, 0.0), 0.0, None)

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


class KinematicCarry:
    """Attaches a rigid body to the tool centre point without simulating a grasp.

    On grasp the payload is switched to kinematic and its transform is driven from the TCP;
    on release it is switched back and left to settle under gravity.  Switching rather than
    creating and destroying a joint at run time is the predictable option: adding joints to
    a live scene is exactly the sort of thing that works until it does not.
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

    def place_at(self, position: tuple[float, float, float]) -> None:
        self._translate.Set(Gf.Vec3d(*position))

    def world_position(self) -> Gf.Vec3d:
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        return cache.GetLocalToWorldTransform(self._prim).ExtractTranslation()

    def grasp(self) -> None:
        self.held = True
        self._rb.CreateKinematicEnabledAttr(True)

    def release(self) -> None:
        self.held = False
        self._rb.CreateKinematicEnabledAttr(False)
        vel = self._rb.GetVelocityAttr()
        if vel:
            vel.Set(Gf.Vec3f(0.0, 0.0, 0.0))
        ang = self._rb.GetAngularVelocityAttr()
        if ang:
            ang.Set(Gf.Vec3f(0.0, 0.0, 0.0))

    def follow(self, rig: ArmRig) -> None:
        """Drive the payload to the TCP.  Called every frame while held."""
        if not self.held:
            return
        mat = rig.tcp_world(self.stage)
        self._translate.Set(mat.ExtractTranslation())
        rot = mat.ExtractRotationQuat()
        self._orient.Set(Gf.Quatf(float(rot.GetReal()), *[float(v) for v in rot.GetImaginary()]))
