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
from .arm_build import ArmRig, KinematicCarry, build_arm
from .runner import RunConfig, SimulationRunner
from .warehouse import WarehouseScene, build_warehouse


class PickPlaceRunner(SimulationRunner):
    def __init__(self, settings: Settings, run: RunConfig, arm: ArmSpec | None = None) -> None:
        super().__init__(settings, run)
        self.arm_spec = arm or ArmSpec()
        self.rig: ArmRig | None = None
        self.warehouse: WarehouseScene | None = None
        self.carry: KinematicCarry | None = None
        self._grasped_at: float | None = None
        self._released_at: float | None = None

    # -- construction --------------------------------------------------------

    def build(self, stage):
        handles = super().build(stage)
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
        )
        # The carry object owns the payload's transform ops, so it has to exist before the
        # payload is positioned.
        self.carry = KinematicCarry(stage, self.warehouse.payload_path)
        self.carry.place_at(self.warehouse.pick.grasp_point)
        self._notes = list(handles.notes) + list(self.warehouse.notes)
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
        )
        if self.run.verbose:
            print(f"  [arm] payload starts on the {self.warehouse.pick.name} at "
                  f"{self.warehouse.pick.grasp_point}, drops on the "
                  f"{self.warehouse.place.name} at {self.warehouse.place.grasp_point}",
                  flush=True)

    # -- per control step ----------------------------------------------------

    def _after_control(self, dt: float) -> None:
        if self.rig is None or self.carry is None or not isinstance(self.mission, PickPlaceMission):
            return
        state = self.mission.arm_state
        self.rig.set_pose(state.pose, state.gripper)

        # Grasp and release are driven by the *gripper's own angle*, not by a phase counter.
        # The jaws closing is the physical event; keying off anything else means the payload
        # can attach a step early or late, which looks like the arm passing through it.
        spec = self.arm_spec
        eps = math.radians(1.0)
        phase = self.mission.phase
        closed = state.gripper <= spec.gripper_closed + eps
        opened = state.gripper >= spec.gripper_open - eps

        if phase == PICK and closed and not self.carry.held:
            self.carry.grasp()
            self._grasped_at = self.t
            if self.run.verbose:
                print(f"  [arm] t={self.t:6.2f}s grasped the payload", flush=True)
        elif phase == PLACE and opened and self.carry.held:
            self.carry.release()
            self._released_at = self.t
            if self.run.verbose:
                print(f"  [arm] t={self.t:6.2f}s released the payload", flush=True)

        self.carry.follow(self.rig)

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
            lines.append(
                "  the arm has no sensor on the shelf, so this error is the docking error "
                "plus the arm's own geometry -- which is the whole reason the alignment matters"
            )
        if self._grasped_at is not None:
            lines.append(f"  grasped at t={self._grasped_at:.2f} s"
                         + (f", released at t={self._released_at:.2f} s"
                            if self._released_at is not None else ", never released"))
        return "\n".join(lines)
