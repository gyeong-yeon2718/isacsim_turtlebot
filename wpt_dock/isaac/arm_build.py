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
