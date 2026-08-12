"""The per-frame bridge between PhysX and the control stack.

This is the only place where Isaac Sim and the algorithm meet, and it is written so
that the *same* object serves both entry paths: the standalone loop calls
:meth:`SimulationRunner.on_step` from its own ``while``, and the GUI registers it as
a physics callback.  Nothing in the control stack knows which one it is.

Two things about the sensing here are better than in the headless twin, and worth
being explicit about because they are the reason to run in Isaac Sim at all:

* **Wheel slip is real, not modelled.**  Odometry is derived from
  ``robot.get_wheel_velocities()``, exactly as real encoders would: the wheel turned
  this much, therefore I think I moved this far.  When PhysX's friction solve lets a
  wheel slip, the encoder still reports the rotation and the estimate genuinely
  diverges from truth.  ``OdometryModel`` then adds only the calibration and
  electrical error on top.  In the kinematic twin the slip term was a noise model;
  here it comes out of the contact solver.
* **The chassis has dynamics.**  It can rock on its wheels while settling, which is
  precisely why the state machine has a SETTLE state and a dwell requirement instead
  of trusting the first frame that lands inside tolerance.

Ground truth is read from the simulator, but only for three things: synthesising
what the cameras would see, driving the coil's brightness, and logging.  It never
reaches the controller.  A run whose numbers look good is therefore a run the
estimator earned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..apriltag import (
    TagDetectorSim,
    build_tags,
    nominal_cameras,
    perturbed_cameras,
    visible_tag_report,
)
from ..config import Settings, with_camera_height
from ..coupling import LinkMonitor, LinkState
from ..estimator import PoseEstimator
from ..fsm import CHARGING, MissionController, MissionStatus
from ..geometry import Pose, compose, wrap_angle, yaw_from_quat_wxyz
from ..routes import plan_route
from ..sensors import OdometryModel
from .board_build import BoardScene, build_board
from .robot_build import LEFT_JOINT, RIGHT_JOINT, build_robot
from .visuals import CoilGlow


@dataclass
class RunConfig:
    start_coil: int = 1
    target_coil: int = 4
    seed: int = 20260811
    placement_pos_sigma: float = 0.010          # m, hand placement on the start coil
    placement_yaw_sigma: float = math.radians(3.0)
    top_plate_stl: str | None = None
    tower_stl: str | None = None
    # Directory of the robot arm's printed parts.  Unused by the alignment-only run; the
    # pick-and-place runner passes it to ``build_arm``, which falls back to procedural geometry
    # for any part it cannot find.
    arm_stl_dir: str | None = None
    # Contact-and-friction grasp instead of a kinematic carry.  Off because it does not work yet:
    # with the pad bodies present PhysX dies on the first physics step.  See
    # ``arm_build.build_pad_bodies`` for what was ruled out and what is left to try.
    physical_grasp: bool = False
    log_path: str | None = None
    verbose: bool = True


@dataclass
class TelemetryRow:
    t: float
    state: str
    v: float
    w: float
    true_x: float
    true_y: float
    true_yaw: float
    est_x: float
    est_y: float
    est_yaw: float
    err_lon: float
    err_lat: float
    err_yaw: float
    efficiency: float
    relative: float


class SimulationRunner:
    """Owns the scene, the estimator, the mission, and the visuals."""

    def __init__(self, settings: Settings, run: RunConfig) -> None:
        self.s = settings
        self.run = run
        self.rng = np.random.default_rng(run.seed)

        self.board_scene: BoardScene | None = None
        self.robot = None
        self.glow: CoilGlow | None = None
        self.estimator: PoseEstimator | None = None
        self.mission: MissionController | None = None
        self.truth_link = LinkMonitor(settings.dock, settings.wpt)

        self.tags = build_tags(settings.board)
        self.odometry = OdometryModel(settings.odometry, self.rng)
        # The camera models are built in build(), not here: their mount height comes
        # from the *measured* bounding box of the TurtleBot3 asset, so it is not known
        # until the robot exists.  Deriving it rather than guessing is what keeps the
        # extrinsics the estimator believes identical to the plate you can see.
        self.cams_nominal: dict = {}
        self.cams_true: dict = {}
        self.detector: TagDetectorSim | None = None
        self.handles = None

        route = plan_route(settings.board, run.start_coil, run.target_coil)
        self.route = route
        self.start_heading = route.legs[0].heading if route.legs else 0.0
        self.nominal_start: Pose = (*settings.board.coil_positions[run.start_coil], self.start_heading)
        self.true_start: Pose = (
            self.nominal_start[0] + float(self.rng.normal(0.0, run.placement_pos_sigma)),
            self.nominal_start[1] + float(self.rng.normal(0.0, run.placement_pos_sigma)),
            wrap_angle(self.nominal_start[2] + float(self.rng.normal(0.0, run.placement_yaw_sigma))),
        )

        self.t = 0.0
        self._control_accum = 0.0
        self._camera_accum = 0.0
        self._last_state = ""
        self._last_print = -1.0
        self.status: MissionStatus | None = None
        self.link_state: LinkState | None = None
        self.finished = False
        self.rows: list[TelemetryRow] = []
        self._dof_left = 0
        self._dof_right = 1
        self._notes: list[str] = []

    # -- construction ------------------------------------------------------

    def build(self, stage):
        """Author the whole scene.  Must run before ``world.reset()``."""
        self.board_scene = build_board(stage, self.s)
        handles = build_robot(
            stage,
            self.s,
            position=(self.true_start[0], self.true_start[1]),
            yaw=self.true_start[2],
            top_plate_stl=self.run.top_plate_stl,
            tower_stl=self.run.tower_stl,
        )
        self.handles = handles
        self._notes = handles.notes

        # Adopt the measured camera height, then build the models.  Both the nominal
        # set (what the robot believes) and the perturbed set (reality) derive from the
        # same specs, so the only difference between them is calibration error.
        self.s = with_camera_height(self.s, handles.plate_lens_z)
        self.cams_nominal = nominal_cameras(self.s.cameras)
        self.cams_true = perturbed_cameras(self.s.cameras, self.s.detection, self.rng)
        self.detector = TagDetectorSim(self.cams_true, self.tags, self.s.detection, self.rng)

        self.glow = CoilGlow(self.board_scene, wpt=self.s.wpt, board=self.s.board)
        self.glow.all_off()
        self.glow.set_target(self.run.target_coil)
        return handles

    @property
    def wheel_joint_names(self) -> tuple[str, str]:
        """The joint names actually present in the loaded robot.

        Discovered by walking the stage rather than hard-coded, because a wrong joint
        name does not raise -- it produces a robot that never moves.
        """
        if self.handles is not None:
            return self.handles.wheel_joints
        return (LEFT_JOINT, RIGHT_JOINT)

    def attach_robot(self, robot) -> None:
        """Hand over the ``WheeledRobot`` wrapper, created by the caller."""
        self.robot = robot

    def after_reset(self) -> None:
        """Resolve joint indices and start the estimator.  Call after ``world.reset()``."""
        if self.robot is not None:
            left, right = self.wheel_joint_names
            try:
                self._dof_left = self.robot.get_dof_index(left)
                self._dof_right = self.robot.get_dof_index(right)
            except Exception as exc:                     # noqa: BLE001
                print(f"[runner] could not resolve wheel dof indices ({exc}); using 0/1")
            try:
                print(f"  [robot] dofs {list(self.robot.dof_names)}; "
                      f"driving {left} (idx {self._dof_left}) and {right} (idx {self._dof_right})",
                      flush=True)
            except Exception:                            # noqa: BLE001
                pass
        self.estimator = PoseEstimator(self.s, self.tags, self.cams_nominal, self.nominal_start)
        self.mission = MissionController(
            self.s, self.run.start_coil, self.run.target_coil, initial_heading=self.start_heading
        )
        self.truth_link = LinkMonitor(self.s.dock, self.s.wpt)
        self.t = 0.0
        self._control_accum = 0.0
        self._camera_accum = 0.0
        self.finished = False
        if self.run.verbose:
            self.banner()

    def banner(self) -> None:
        s = self.s
        print("=" * 78)
        print(f"  WPT coil alignment  |  coil {self.run.start_coil} -> coil {self.run.target_coil}")
        print(f"  {self.route.describe()}")
        if self.route.note:
            print(f"  {self.route.note}")
        print(f"  stage {s.board.stage_size[0]:.2f} x {s.board.stage_size[1]:.2f} m, "
              f"coil spacing {s.board.coil_spacing[0]:.3f} x {s.board.coil_spacing[1]:.3f} m")
        print(f"  tolerance {s.dock.pos_tol * 1000:.0f} mm per axis, "
              f"{math.degrees(s.dock.yaw_tol):.1f} deg, dwell {s.dock.hold_time:.1f} s, "
              f"lock needs eta >= {s.wpt.lock_efficiency * 100:.0f} %")
        placed = (
            (self.true_start[0] - self.nominal_start[0]) * 1000,
            (self.true_start[1] - self.nominal_start[1]) * 1000,
            math.degrees(wrap_angle(self.true_start[2] - self.nominal_start[2])),
        )
        print(f"  hand-placement error: {placed[0]:+.1f}, {placed[1]:+.1f} mm, {placed[2]:+.2f} deg "
              f"(the robot is told the nominal pose, not this one)")
        print(f"  {self.odometry.describe()}")
        for note in self._notes:
            print(f"  [robot] {note}")
        print("=" * 78)

    # -- per frame ---------------------------------------------------------

    def true_pose(self) -> Pose:
        pos, quat = self.robot.get_world_pose()
        return (float(pos[0]), float(pos[1]), yaw_from_quat_wxyz(quat))

    def encoder_twist(self) -> tuple[float, float]:
        """Body twist as the wheel encoders would report it.

        Deliberately *not* ``get_linear_velocity()``: that is the chassis's true
        motion, which a robot cannot measure.  Wheel rotation is what an encoder
        sees, so slip appears as an honest discrepancy instead of being erased.
        """
        w = self.robot.get_wheel_velocities()
        return self.s.robot.wheels_to_body(float(w[0]), float(w[1]))

    def _each_physics_step(self, dt: float, believed: Pose) -> None:
        """Extension point, called every **physics** step.

        Separate from ``_after_control`` because the two rates mean different things.  Anything
        that is kinematics -- servos moving, a carried object following the gripper -- belongs here,
        or it advances in control-period jumps and judders.  Anything that is a decision belongs in
        ``_after_control``, at the rate the controller actually runs.
        """

    def _after_control(self, dt: float) -> None:
        """Extension point, called once per control step after the wheels are commanded.

        No-op here.  The pick-and-place runner uses it to push the arm's joint angles into
        the rig and to carry the payload, which has to happen at the control rate and after
        the mission has decided what the arm should be doing.
        """

    def wheel_slip(self, commanded_v: float) -> float:
        """Fraction of commanded forward speed that the chassis is *not* achieving.

        Worth surfacing rather than inferring: the first Isaac run of this scene sat
        at 98 % slip because the casters were carrying the load, and the symptom on
        screen -- odometry marching forward while the robot stayed put -- looks exactly
        like a broken controller.  One number tells the two apart immediately.
        """
        if abs(commanded_v) < 1e-4:
            return 0.0
        lin = self.robot.get_linear_velocity()
        _, _, yaw = self.true_pose()
        actual = float(lin[0]) * math.cos(yaw) + float(lin[1]) * math.sin(yaw)
        return max(0.0, min(1.0, 1.0 - actual / commanded_v))

    def command(self, v: float, w: float) -> None:
        from isaacsim.core.utils.types import ArticulationAction

        left, right = self.s.robot.body_to_wheels(v, w)
        limit = self.s.robot.max_wheel_rate
        left = max(-limit, min(limit, left))
        right = max(-limit, min(limit, right))
        self.robot.apply_wheel_actions(
            ArticulationAction(joint_velocities=np.array([left, right], dtype=float))
        )

    def on_step(self, step_size: float) -> None:
        """Physics-step callback.  Safe to call at any physics rate."""
        if self.robot is None or self.mission is None or self.estimator is None:
            return
        if self.finished:
            self.command(0.0, 0.0)
            return

        self.t += step_size
        truth = self.true_pose()
        offset = self.s.robot.rx_coil_offset
        true_coil_pose = compose(truth, (offset[0], offset[1], 0.0))

        # Camera cadence: 10 Hz, matching the hardware.  Running the fix at the
        # physics rate would be a free accuracy gain the real robot does not get.
        self._camera_accum += step_size
        camera_period = 1.0 / self.s.sim.camera_hz
        if self._camera_accum >= camera_period:
            self._camera_accum = 0.0
            self.estimator.update(self.detector.detect(truth))

        # Kinematics run every physics step, before the control block, so a subclass driving an
        # arm sees the freshest estimate and moves smoothly rather than in control-period jumps.
        # The estimate is only refreshed at the control rate, which is correct: the *sensor* is
        # slow, the *servos* are not.
        believed_now = compose(self.estimator.pose, (offset[0], offset[1], 0.0))
        self._each_physics_step(step_size, believed_now)

        # Control cadence.
        self._control_accum += step_size
        control_period = 1.0 / self.s.sim.control_hz
        if self._control_accum >= control_period:
            dt = self._control_accum
            self._control_accum = 0.0

            v_enc, w_enc = self.encoder_twist()
            self.estimator.predict(self.odometry.measure(v_enc, w_enc), dt)

            believed = compose(self.estimator.pose, (offset[0], offset[1], 0.0))
            self.status = self.mission.step(believed, dt)
            self.command(self.status.v, self.status.w)
            self._after_control(dt)

            if self.status.finished:
                self.finished = True

        # Visual feedback always runs from ground truth: the coil lights because the
        # coils really are aligned, not because the robot thinks so.
        errors = self.mission.coil_errors(true_coil_pose)
        self.link_state = self.truth_link.update(step_size, *errors)
        if self.glow is not None and self.status is not None:
            # Every coil, from the receiver's true position -- so the coil the robot starts on
            # glows before it moves, and one it crosses on a diagonal route lights up as it
            # passes.  Lighting only the target made the other three look broken.
            self.glow.update_all(
                (true_coil_pose[0], true_coil_pose[1]),
                self.run.target_coil,
                self.link_state,
                # link_state was measured against *this* coil, so this is the only coil the
                # lock decision may be applied to.  During a pick-and-place it is the source
                # coil for the first third of the run.
                current=self.mission.current_coil,
                energised=self.status.energised or CHARGING in str(self.status.state)
                or "VERIFY" in str(self.status.state) or "SETTLE" in str(self.status.state),
                believed_locked=CHARGING in str(self.status.state),
                time=self.t,
            )

        if self.status is not None:
            self._record(truth, errors)
            if self.run.verbose:
                self._report()

    # -- reporting ---------------------------------------------------------

    def _record(self, truth: Pose, errors: tuple[float, float, float]) -> None:
        if self.link_state is None or self.status is None or self.estimator is None:
            return
        est = self.estimator.pose
        self.rows.append(
            TelemetryRow(
                t=self.t, state=self.status.state, v=self.status.v, w=self.status.w,
                true_x=truth[0], true_y=truth[1], true_yaw=truth[2],
                est_x=est[0], est_y=est[1], est_yaw=est[2],
                err_lon=errors[0], err_lat=errors[1], err_yaw=errors[2],
                efficiency=self.link_state.efficiency, relative=self.link_state.relative,
            )
        )

    def _report(self) -> None:
        if self.status is None or self.link_state is None or self.estimator is None:
            return
        state_changed = self.status.state != self._last_state
        tick = self.t - self._last_print >= 1.0
        if not (state_changed or tick or self.status.finished):
            return
        self._last_state = self.status.state
        self._last_print = self.t
        lon, lat, yaw = self.status.errors or (0.0, 0.0, 0.0)
        est_err = self.estimator.error_against(self.true_pose())
        print(
            f"  t={self.t:6.2f}s {self.status.state:9s}"
            f" | believed x={lon * 1000:+7.2f} y={lat * 1000:+7.2f} mm"
            f" yaw={math.degrees(yaw):+6.2f} deg"
            f" | TRUE radial={self.link_state.radial_offset * 1000:6.2f} mm"
            f" eta={self.link_state.efficiency * 100:5.1f}%"
            f" | est err {est_err[0] * 1000:+6.2f},{est_err[1] * 1000:+6.2f} mm"
            f" {math.degrees(est_err[2]):+5.2f} deg"
            f" slip={self.wheel_slip(self.status.v) * 100:3.0f}%"
            f" | {self.status.message[:48]}"
        )

    def summary(self) -> str:
        if self.status is None or self.link_state is None or self.estimator is None:
            return "no steps taken"
        lines = ["-" * 78]
        verdict = "CHARGING" if self.status.success else f"FAILED ({self.status.state})"
        lines.append(f"  result: {verdict}   after {self.t:.2f} s, {self.status.retries} retry/retries")
        lines.append(f"  {self.status.message}")
        row = self.rows[-1] if self.rows else None
        if row is not None:
            lines.append(
                f"  GROUND TRUTH at the coil: longitudinal {row.err_lon * 1000:+.2f} mm, "
                f"lateral {row.err_lat * 1000:+.2f} mm, yaw {math.degrees(row.err_yaw):+.2f} deg"
            )
            lines.append(
                f"  radial offset {self.link_state.radial_offset * 1000:.2f} mm  ->  "
                f"k = {self.link_state.coupling_k:.3f}, eta = {self.link_state.efficiency * 100:.2f} % "
                f"({self.link_state.relative * 100:.1f} % of the aligned maximum)"
            )
        est_err = self.estimator.error_against(self.true_pose())
        lines.append(
            f"  estimator error at the end: {est_err[0] * 1000:+.2f}, {est_err[1] * 1000:+.2f} mm, "
            f"{math.degrees(est_err[2]):+.3f} deg   ({self.estimator.summary()})"
        )
        lines.extend(self.plausibility_check())
        lines.append("-" * 78)
        return "\n".join(lines)

    # -- simulation-only sanity gate -----------------------------------------

    def plausibility_check(self) -> list[str]:
        """Refuse to let a physically impossible run report success.

        This exists because one did.  A collision launched the robot off the board and through
        the floor at t = 9 s, it froze there, and the mission still printed ``CHARGING`` --
        because with no tag in any camera the estimator had nothing to correct against, so it
        coasted on wheel odometry that kept turning, and odometry said it had arrived.  Every
        number the mission logic can see was self-consistent.  The only thing that could have
        caught it was ground truth, and nothing was looking at ground truth.

        So this does, and it is deliberately a *ground-truth* check rather than a cleverer
        estimator: it is asking "is this simulation still simulating the thing I asked about",
        which is a question only the simulator can answer.  The real-hardware analogue is not
        this function -- it is a watchdog on "claiming alignment while no tag has been seen for
        N seconds", which is the transferable half of the same idea and is noted in the README
        as unimplemented.
        """
        out: list[str] = []
        x, y, _ = self.true_pose()
        board = self.s.board
        ox, oy = board.stage_origin
        w, h = board.stage_size
        # A generous box: the robot legitimately drives into the margin outside the coils, and
        # legitimately retreats.  Half a metre past the plywood edge is not legitimate.
        pad = 0.50
        z = float(self.robot.get_world_pose()[0][2])
        faults: list[str] = []
        if not (ox - pad <= x <= ox + w + pad and oy - pad <= y <= oy + h + pad):
            faults.append(f"the robot is off the board at ({x:+.3f}, {y:+.3f}) m")
        if not (-0.05 <= z <= 0.25):
            faults.append(f"the robot's base is at z = {z:+.3f} m, which is not on the floor")
        if faults:
            out.append("  !! IMPLAUSIBLE RUN -- the result above is not trustworthy:")
            out.extend(f"     - {f}" for f in faults)
            out.append("     A collision has moved the robot somewhere it cannot drive to. "
                       "Fix the scene, not the controller.")
        return out
        return "\n".join(lines)

    def write_log(self) -> None:
        if not self.run.log_path or not self.rows:
            return
        import csv

        with open(self.run.log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "t", "state", "v", "w", "true_x", "true_y", "true_yaw",
                    "est_x", "est_y", "est_yaw", "err_lon", "err_lat", "err_yaw",
                    "efficiency", "relative",
                ]
            )
            for r in self.rows:
                writer.writerow(
                    [
                        f"{r.t:.4f}", r.state, f"{r.v:.5f}", f"{r.w:.5f}",
                        f"{r.true_x:.6f}", f"{r.true_y:.6f}", f"{r.true_yaw:.6f}",
                        f"{r.est_x:.6f}", f"{r.est_y:.6f}", f"{r.est_yaw:.6f}",
                        f"{r.err_lon:.6f}", f"{r.err_lat:.6f}", f"{r.err_yaw:.6f}",
                        f"{r.efficiency:.5f}", f"{r.relative:.5f}",
                    ]
                )
        print(f"  telemetry written to {self.run.log_path} ({len(self.rows)} rows)", flush=True)

    def tag_report(self) -> str:
        return visible_tag_report(self.detector.detect(self.true_pose()))


