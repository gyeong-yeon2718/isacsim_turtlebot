"""Pick at one coil, place at another: a supervisor over the alignment machine.

    ALIGN_SOURCE -> PICK -> TRANSIT (drives and aligns) -> PLACE -> DONE

The point of building it this way -- as a supervisor that *runs the existing*
``MissionController`` twice rather than a new state machine -- is that the alignment
behaviour stays the validated one.  Nothing about docking is re-implemented here; this
file only decides when to hand control to the arm.

Why the wireless-charging alignment is the load-bearing part of a pick-and-place task
------------------------------------------------------------------------------------
The cute_arm has three joints and no sensor pointed at the shelf.  It goes exactly where
its own joint angles say, from wherever the robot happens to be parked -- so the placement
error *is* the base pose error, one for one, with nothing in between to notice or correct
it (``tests/test_core.py`` asserts that the transform neither amplifies nor hides it).

That makes the two halves of this project one system rather than two demos: the AprilTag
alignment that exists to seat a receiver coil over a transmitter coil is the same mechanism
that makes a blind arm repeatable at a shelf.  A 1 cm docking error is a 1 cm miss on the
drop pad, and the run reports both so the connection is visible rather than asserted.

Phase notes
-----------
``ALIGN_SOURCE`` re-aligns on the coil the robot was hand-placed on, using the
``start == target`` path of the mission controller.  It matters that this is a real
approach and not just a measurement: hand placement is modelled with a 10 mm sigma, which
is outside the 15 mm box often enough to matter, and the controller handles it by backing
off along the approach ray and driving in.

``TRANSIT`` is the ordinary coil-to-coil mission, which ends with the destination
alignment already done -- so there is no separate ALIGN_TARGET phase.  The arm stays folded
in ``CARRY`` throughout, which keeps the payload inside the support footprint so it cannot
swing out past the plywood edge, and keeps it out of the downward cameras' view.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .arm import (
    CARRY,
    HOME,
    ArmPose,
    ArmSequencer,
    ArmSpec,
    ArmState,
    pick_sequence,
    place_sequence,
)
from .config import Settings
from .coupling import LinkState
from .fsm import MissionController
from .geometry import Pose, compose, invert
from .routes import plan_route

ALIGN_SOURCE = "ALIGN_SOURCE"
PICK = "PICK"
TRANSIT = "TRANSIT"
PLACE = "PLACE"
DONE = "DONE"
FAILED = "FAILED"


@dataclass
class PickPlaceStatus:
    """Duck-types ``MissionStatus`` so the Isaac runner needs no special cases."""

    state: str
    v: float
    w: float
    leg_index: int = 0
    n_legs: int = 0
    target_coil: int = 0
    message: str = ""
    lateral: float = 0.0
    heading_error: float = 0.0
    errors: tuple[float, float, float] | None = None
    link: LinkState | None = None
    retries: int = 0
    required_distance: float = 0.0
    remaining: float = 0.0
    energised: bool = False
    finished: bool = False
    success: bool = False
    extras: dict = field(default_factory=dict)

    # pick-and-place specific
    phase: str = ALIGN_SOURCE
    arm_pose: ArmPose = HOME
    gripper: float = 0.0
    holding: bool = False
    arm_message: str = ""


class PickPlaceMission:
    def __init__(
        self,
        settings: Settings,
        arm: ArmSpec,
        source_coil: int,
        target_coil: int,
        *,
        grasp_world: tuple[float, float, float],
        drop_world: tuple[float, float, float],
        plate_top_z: float,
        payload_width: float | None = None,
    ) -> None:
        self.s = settings
        self.arm_spec = arm
        self.source_coil = int(source_coil)
        self.target_coil = int(target_coil)
        self.grasp_world = grasp_world
        self.drop_world = drop_world
        self.plate_top_z = plate_top_z
        # The width the jaws must close onto.  ``None`` means "close to the mechanical stop",
        # which is only right when there is nothing in the jaws.
        self.payload_width = payload_width
        # The angle the sequence actually commands, squeeze included, so the runner's "closed"
        # test matches what the jaws are told to do rather than where the object surface is.
        self.grip_angle = (
            arm.gripper_closed if payload_width is None
            else arm.grip_angle_for(max(arm.gripper_clear(arm.gripper_closed),
                                        payload_width - arm.grip_squeeze))
        )

        route = plan_route(settings.board, self.source_coil, self.target_coil)
        self.source_heading = route.legs[0].heading if route.legs else 0.0
        self.route_note = route.describe()

        self.phase = ALIGN_SOURCE
        self.mission = MissionController(
            settings, self.source_coil, self.source_coil, initial_heading=self.source_heading
        )
        self.sequencer = ArmSequencer(arm)
        self.sequencer.state.pose = HOME
        self.sequencer.state.gripper = arm.gripper_open
        self.message = f"aligning on coil {self.source_coil} before picking"
        self.elapsed = 0.0
        self.finished = False
        self.success = False
        self.align_errors: dict[str, tuple[float, float, float]] = {}
        self._offset = settings.robot.rx_coil_offset

    # -- interface the runner expects ---------------------------------------

    def coil_errors(self, pose: Pose) -> tuple[float, float, float]:
        return self.mission.coil_errors(pose)

    @property
    def current_coil(self) -> int:
        """The coil the *active* alignment is working on, which is not the final target.

        During ``ALIGN_SOURCE`` and ``PICK`` this is the source coil.  Reporting the final
        target here instead is what made the destination coil glow -- and go charging-green --
        at t = 0, while the robot was actually sitting on the source.
        """
        return self.mission.current_coil

    @property
    def arm_state(self) -> ArmState:
        return self.sequencer.state

    def _robot_pose(self, control_pose: Pose) -> Pose:
        """Undo the receiver-coil offset to recover the chassis pose the arm mounts on."""
        ox, oy = self._offset
        return compose(control_pose, invert((ox, oy, 0.0)))

    # -- stepping -----------------------------------------------------------

    def step(self, control_pose: Pose, dt: float) -> PickPlaceStatus:
        self.elapsed += dt
        if self.phase in (DONE, FAILED):
            return self._status(0.0, 0.0, finished=True, success=self.phase == DONE)

        if self.phase in (ALIGN_SOURCE, TRANSIT):
            return self._step_drive(control_pose, dt)
        return self._step_arm(control_pose, dt)

    def _step_drive(self, control_pose: Pose, dt: float) -> PickPlaceStatus:
        status = self.mission.step(control_pose, dt)
        if not status.finished:
            self.message = f"{self.phase.lower()}: {status.message}"
            return self._status(status.v, status.w, inner=status)

        if not status.success:
            self.phase = FAILED
            self.message = f"{self.phase} while {status.state}: {status.message}"
            return self._status(0.0, 0.0, finished=True, success=False, inner=status)

        # Record the alignment the arm is about to rely on.
        errors = self.mission.coil_errors(control_pose)
        if self.phase == ALIGN_SOURCE:
            self.align_errors["source"] = errors
            self.phase = PICK
            self.sequencer.start(
                pick_sequence(self.grasp_world, self.arm_spec, grip_width=self.payload_width)
            )
            self.message = (
                f"aligned on coil {self.source_coil} "
                f"(believed x {errors[0] * 1000:+.1f} mm, y {errors[1] * 1000:+.1f} mm, "
                f"yaw {math.degrees(errors[2]):+.2f} deg); picking"
            )
        else:
            self.align_errors["target"] = errors
            self.phase = PLACE
            self.sequencer.start(place_sequence(self.drop_world, self.arm_spec))
            self.message = (
                f"aligned on coil {self.target_coil} "
                f"(believed x {errors[0] * 1000:+.1f} mm, y {errors[1] * 1000:+.1f} mm, "
                f"yaw {math.degrees(errors[2]):+.2f} deg); placing"
            )
        return self._status(0.0, 0.0, inner=status)

    def advance_arm(self, control_pose: Pose, dt: float) -> ArmState:
        """Move the servos on, at whatever rate the caller runs.  Separate from ``step`` on purpose.

        The joints used to advance inside ``step``, which runs at ``SimSpec.control_hz`` -- 30 Hz.
        Physics steps several times faster, so the arm and everything it carried moved in 33 ms
        jumps and the payload visibly juddered.  Advancing the servos is *kinematics* and belongs
        at the physics rate; deciding that a waypoint is done and the phase should change is
        *supervision* and belongs at the control rate.  Splitting them is the fix, and it is also
        the more honest arrangement -- a real servo does not wait for the controller's next tick.

        Refreshes the IK context from the pose the alignment *achieved* rather than the one it was
        asked for.  The robot is stationary here, so it is cheap, and it is the difference between
        the arm reaching for where it is and where it thinks it should be.
        """
        if self.phase not in (PICK, PLACE):
            return self.sequencer.state
        self.sequencer.set_ik_context(self._robot_pose(control_pose), self.plate_top_z)
        return self.sequencer.step(dt)

    def _step_arm(self, control_pose: Pose, dt: float) -> PickPlaceStatus:
        # The joints have already been advanced this frame by ``advance_arm``; this only decides
        # what the result means.  Stepping the sequencer again here would double its rate.
        state = self.sequencer.state

        if not self.sequencer.finished:
            self.message = f"{self.phase.lower()}: {state.message}"
            return self._status(0.0, 0.0)

        if not self.sequencer.succeeded:
            self.phase = FAILED
            self.message = f"arm failed during {self.phase}: {state.message}"
            return self._status(0.0, 0.0, finished=True, success=False)

        if self.phase == PICK:
            self.phase = TRANSIT
            self.mission = MissionController(
                self.s, self.source_coil, self.target_coil, initial_heading=self.source_heading
            )
            self.message = f"picked; {self.route_note}"
            return self._status(0.0, 0.0)

        self.phase = DONE
        self.finished = True
        self.success = True
        self.message = "placed on the drop pad"
        return self._status(0.0, 0.0, finished=True, success=True)

    # -- reporting ----------------------------------------------------------

    def _status(
        self, v: float, w: float, *, finished: bool = False, success: bool = False, inner=None
    ) -> PickPlaceStatus:
        state = self.sequencer.state
        holding = self.phase in (TRANSIT, PLACE) and not (
            self.phase == PLACE and self.sequencer.state.step >= 4
        )
        st = PickPlaceStatus(
            state=f"{self.phase}/{inner.state}" if inner is not None else self.phase,
            v=v, w=w, target_coil=self.target_coil, message=self.message,
            phase=self.phase, arm_pose=state.pose, gripper=state.gripper,
            holding=holding, arm_message=state.message,
            finished=finished, success=success,
        )
        if inner is not None:
            st.leg_index = inner.leg_index
            st.n_legs = inner.n_legs
            st.lateral = inner.lateral
            st.heading_error = inner.heading_error
            st.errors = inner.errors
            st.link = inner.link
            st.retries = inner.retries
            st.remaining = inner.remaining
            st.energised = inner.energised
        return st

    def summary(self) -> str:
        lines = [f"pick-and-place: coil {self.source_coil} -> coil {self.target_coil}",
                 f"  {self.route_note}"]
        for key in ("source", "target"):
            if key in self.align_errors:
                ex, ey, eyaw = self.align_errors[key]
                lines.append(
                    f"  {key} alignment as believed: x {ex * 1000:+.2f} mm, y {ey * 1000:+.2f} mm, "
                    f"yaw {math.degrees(eyaw):+.2f} deg"
                )
        lines.append(f"  phase {self.phase} after {self.elapsed:.1f} s")
        return "\n".join(lines)
