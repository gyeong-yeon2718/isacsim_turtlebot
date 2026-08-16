"""The pick-and-place runner: the alignment runner plus an arm and a warehouse.

Subclassing rather than forking is the point.  Everything about odometry, tag registration,
the estimator, wheel commanding, telemetry and the coil glow is inherited unchanged, so the
alignment behaviour on this branch is *the same code* that was validated on ``main``.  Only
three things are overridden: what gets built, which mission runs, and what happens to the
arm after each control step.
"""

from __future__ import annotations

import math

from ..arm import ArmSpec, workspace_report
from ..config import Settings
from ..pickplace import DONE, FAILED, PICK, PLACE, PickPlaceMission
from .arm_build import ArmRig, GripperAttachment, build_arm
from .runner import RunConfig, SimulationRunner
from .warehouse import PAYLOAD_SIZE, WarehouseScene, build_warehouse


class PickPlaceRunner(SimulationRunner):
    def __init__(self, settings: Settings, run: RunConfig, arm: ArmSpec | None = None) -> None:
        super().__init__(settings, run)
        self.arm_spec = arm or ArmSpec()
        self.rig: ArmRig | None = None
        self.warehouse: WarehouseScene | None = None
        self.carry: GripperAttachment | None = None
        self.stage = None
        self._arm_notes: list[str] = []
        self._grasped_at: float | None = None
        self._released_at: float | None = None

    # -- construction --------------------------------------------------------

    def build(self, stage):
        handles = super().build(stage)
        self.stage = stage
        self.warehouse = build_warehouse(
            stage, self.s, self.arm_spec,
            source_coil=self.run.start_coil, target_coil=self.run.target_coil,
            plate_top_z=handles.plate_top_z,
        )
        # The rig is parented to the chassis, so the arm rides with the robot; the plate's
        # measured top surface is where it bolts on.
        self.rig = build_arm(
            stage, handles.chassis_path, self.arm_spec,
            plate_top_local_z=handles.plate_top_z - self.s.robot.wheel_radius,
            stl_dir=self.run.arm_stl_dir,
            notes=self._arm_notes,
            spec_plate_size=(self.s.robot.plate_size[0], self.s.robot.plate_size[1]),
            physical_grasp=self.run.physical_grasp,
        )
        # The carry object owns the payload's transform ops, so it has to exist before the
        # payload is positioned.
        # The pads are part of the robot, so they must not collide with it -- and they have to be
        # put where the gripper is before the first physics step, or they spawn at the world origin
        # (which is coil 1, i.e. inside the robot) and launch it.
        self.rig.exclude_pads_from(stage, handles.prim_path)
        self.rig.drive_pads(stage)

        self.carry = GripperAttachment(stage, self.warehouse.payload_path,
                                       physical_grasp=self.run.physical_grasp)
        # Payload-versus-robot collision is filtered, and this is a **known compromise** rather
        # than a settled design.  Both states have been run and both are wrong in different ways:
        #
        #   filtered   the box passes through the printed top plate, which the user saw and which
        #              is plainly not physics
        #   unfiltered the box jams against the robot somewhere and tips the whole TurtleBot over,
        #              and the run time goes from 33 s to 73 s fighting the contact
        #
        # Filtered is the one that leaves a usable simulation, so it is what ships until the jam
        # is located.  What the jam is *not*: the carry trajectory.  Measured with the current
        # geometry, the payload clears the plate by 130-162 mm along the whole lift-slew-carry
        # path, so widening that clearance would fix nothing.  The contact is at the pick or the
        # place moment, or against a robot part that is not the plate -- the rear rack and the
        # battery box are the candidates, and locating it wants the placement probe pointed at the
        # payload through the grasp rather than another guess.
        self.carry.exclude_from_collision_with(handles.prim_path)
        # Spawned 1 mm clear of the rollers rather than exactly in contact.  The payload is a
        # dynamic body now, so it drops that millimetre in about 14 ms and is at rest long
        # before the arm arrives at t = 4.4 s -- and starting a body in exact surface contact
        # asks the solver to resolve a zero-gap overlap on frame one, which is a question worth
        # not asking.
        gx, gy, gz = self.warehouse.pick.grasp_point
        self.carry.place_at((gx, gy, gz + 0.001))
        self._notes = list(handles.notes) + list(self.warehouse.notes) + list(self._arm_notes)
        self._notes.extend(workspace_report(self.arm_spec).splitlines())
        return handles

    def after_reset(self) -> None:
        super().after_reset()
        # Replace the alignment-only mission with the supervisor.  PickPlaceStatus
        # duck-types MissionStatus, so nothing in the inherited step loop changes.
        assert self.warehouse is not None and self.handles is not None
        self.mission = PickPlaceMission(
            self.s, self.arm_spec, self.run.start_coil, self.run.target_coil,
            grasp_world=self.warehouse.pick.grasp_point,
            drop_world=self.warehouse.place.grasp_point,
            plate_top_z=self.handles.plate_top_z,
            payload_width=PAYLOAD_SIZE,
        )
        if self.run.verbose:
            print(f"  [arm] payload starts on the {self.warehouse.pick.name} at "
                  f"{self.warehouse.pick.grasp_point}, drops on the "
                  f"{self.warehouse.place.name} at {self.warehouse.place.grasp_point}",
                  flush=True)

    # -- diagnostics ---------------------------------------------------------

    def _frame_report(self, label: str) -> None:
        """Print the chassis and TCP world poses.

        Kept in the code rather than thrown away: the payload ending up a metre from the pad
        looked like a grasp bug for two runs, and the thing that actually settled it was
        comparing the chassis' *USD* transform against the pose the controller believed it had.
        If those two disagree, no amount of arm-side fixing will help.
        """
        from pxr import Sdf, Usd, UsdGeom

        assert self.rig is not None and self.handles is not None and self.stage is not None
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        chassis = self.stage.GetPrimAtPath(Sdf.Path(self.handles.chassis_path))
        cw = cache.GetLocalToWorldTransform(chassis).ExtractTranslation()
        tw = self.rig.tcp_world(self.stage).ExtractTranslation()
        truth = self.true_pose()
        print(f"  [frame:{label}] chassis USD ({cw[0]:+.4f}, {cw[1]:+.4f}, {cw[2]:+.4f}) "
              f"TCP USD ({tw[0]:+.4f}, {tw[1]:+.4f}, {tw[2]:+.4f}) "
              f"chassis physics ({truth[0]:+.4f}, {truth[1]:+.4f})", flush=True)

    # -- per physics step ----------------------------------------------------

    def _each_physics_step(self, dt: float, believed) -> None:
        """Advance the servos and everything that rides on them, at the physics rate.

        This used to happen in ``_after_control``, at ``control_hz`` = 30 Hz, while physics steps
        several times faster -- so the arm and the payload it carried advanced in 33 ms jumps and
        the box visibly juddered.  Servos are kinematics and belong here; the decisions about what
        the arm has *achieved* stay at the control rate in ``_after_control``.
        """
        if self.rig is None or self.carry is None or not isinstance(self.mission, PickPlaceMission):
            return
        state = self.mission.advance_arm(believed, dt)
        self.rig.set_pose(state.pose, state.gripper)
        # The physical pads live outside the robot's prim tree, so USD does not compose their
        # world poses -- they are driven here, right after the joints move.
        self.rig.drive_pads(self.stage)
        self.carry.follow(self.rig)
        self._grasp_events(state)

    # -- per control step ----------------------------------------------------

    def _grasp_events(self, state) -> None:
        if self.rig is None or self.carry is None or not isinstance(self.mission, PickPlaceMission):
            return

        # Grasp and release are driven by the *gripper's own angle*, not by a phase counter.
        # The jaws closing is the physical event; keying off anything else means the payload
        # can attach a step early or late, which looks like the arm passing through it.
        spec = self.arm_spec
        eps = math.radians(1.0)
        phase = self.mission.phase
        # "Closed" means closed *onto the payload*, not to the mechanical stop -- the jaws stop
        # at the box's width.  Keying off gripper_closed here would never fire once the
        # sequence started commanding the contact angle instead.
        closed = state.gripper <= self.mission.grip_angle + eps
        opened = state.gripper >= spec.gripper_open - eps

        if phase == PICK and closed and not self.carry.held:
            self.carry.grasp()
            self._grasped_at = self.t
            if self.run.verbose:
                print(f"  [arm] t={self.t:6.2f}s grasped the payload", flush=True)
                self._frame_report("grasp")
        elif phase == PLACE and opened and self.carry.held:
            self.carry.release()
            self._released_at = self.t
            if self.run.verbose:
                print(f"  [arm] t={self.t:6.2f}s released the payload", flush=True)
                self._frame_report("release")

    # -- reporting -----------------------------------------------------------

    def summary(self) -> str:
        base = super().summary()
        if not isinstance(self.mission, PickPlaceMission) or self.warehouse is None:
            return base
        lines = [base, self.mission.summary()]
        if self.carry is not None:
            landed = self.carry.world_position()
            target = self.warehouse.place.grasp_point
            dx = float(landed[0]) - target[0]
            dy = float(landed[1]) - target[1]
            dz = float(landed[2]) - target[2]
            lines.append(
                f"  PAYLOAD landed at ({float(landed[0]):.4f}, {float(landed[1]):.4f}, "
                f"{float(landed[2]):.4f}) m"
            )
            lines.append(
                f"  placement error vs the drop pad: {dx * 1000:+.2f}, {dy * 1000:+.2f} mm "
                f"lateral, {dz * 1000:+.2f} mm vertical, "
                f"radial {math.hypot(dx, dy) * 1000:.2f} mm"
            )
            lines.append("  PAYLOAD " + self.carry.rest_report())
            lines.append(
                "  the arm has no sensor on the shelf, so this error is the docking error "
                "plus the arm's own geometry -- which is the whole reason the alignment matters"
            )
        if self._grasped_at is not None:
            lines.append(f"  grasped at t={self._grasped_at:.2f} s"
                         + (f", released at t={self._released_at:.2f} s"
                            if self._released_at is not None else ", never released"))
        return "\n".join(lines)
