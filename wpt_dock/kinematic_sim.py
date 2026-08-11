"""A fast headless twin of the whole loop, with no Isaac Sim and no GPU.

Why this exists
---------------
The request was for an approach that works, after two that did not.  "It looks
right in the code" is not evidence, and Isaac Sim is a slow and expensive place to
discover that a gain is wrong -- a single run costs minutes of start-up and one run
tells you nothing about repeatability.  So the identical control stack --
``OdometryModel`` -> ``PoseEstimator`` -> ``MissionController`` -> ``path_follow`` --
is driven here against a unicycle integrated in closed form, which runs a full
mission in milliseconds and can therefore be repeated hundreds of times over
randomised placements and sensor noise.

Only the vehicle is replaced.  Every line of estimation and control that runs in
Isaac Sim runs here, so a success rate measured here is a real statement about the
algorithm.  What it deliberately does *not* cover is contact physics: wheel slip is
a noise model here rather than a friction solve, the chassis cannot rock on its
suspension, and nothing can tip.  Those are exactly the questions Isaac Sim is
for, which is why this is a screening tool and not a substitute.

Actuation realism that is easy to omit and matters at millimetre scale:

* **First-order motor lag.**  A commanded wheel speed is not the actual one.  With
  a 50 ms time constant and a 30 Hz controller, the last few millimetres of an
  approach are governed by the lag, not by the gain.
* **A stiction deadband.**  Below a few percent of full speed the wheels do not
  turn at all.  This is the physical reason ``SpeedProfile.min_move`` exists, and
  a simulation without it will happily report arbitrarily fine positioning that
  hardware cannot reproduce.
* **Per-wheel multiplicative slip**, which is what makes odometry drift rather
  than merely be noisy.

Hand placement is modelled too: the upstream procedure is to physically put the
robot on a known coil facing +X, so the estimator starts believing the *nominal*
coil pose while the truth carries the placement error.  That initial disagreement
is a genuine part of the problem, not a detail.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .apriltag import TagDetectorSim, build_tags, nominal_cameras, perturbed_cameras
from .config import Settings
from .coupling import LinkMonitor
from .estimator import PoseEstimator
from .fsm import MissionController
from .geometry import Pose, compose, wrap_angle
from .routes import plan_route
from .sensors import OdometryModel


@dataclass
class Actuator:
    """Wheel-level actuation: lag, stiction deadband, multiplicative slip."""

    settings: Settings
    rng: np.random.Generator
    tau: float = 0.050              # s, motor time constant
    # TurtleBot3 drives Dynamixel XL430 servos in closed-loop velocity mode, so the
    # floor is the servo's own velocity quantisation (about 0.024 rad/s) rather than
    # the much larger stiction of an open-loop brushed motor.  1 % of full scale is
    # roughly that, and it is what makes 12 mm/s a usable command and 4 mm/s not.
    deadband: float = 0.010         # -, fraction of max wheel rate below which nothing moves
    slip_sigma: float = 0.010       # -, per-wheel per-step multiplicative slip
    left: float = 0.0
    right: float = 0.0

    def apply(self, v_cmd: float, w_cmd: float, dt: float) -> tuple[float, float]:
        """Command in, *actual* body twist out."""
        robot = self.settings.robot
        l_target, r_target = robot.body_to_wheels(v_cmd, w_cmd)
        limit = robot.max_wheel_rate
        l_target = max(-limit, min(limit, l_target))
        r_target = max(-limit, min(limit, r_target))

        alpha = 1.0 - math.exp(-dt / self.tau) if self.tau > 0.0 else 1.0
        self.left += alpha * (l_target - self.left)
        self.right += alpha * (r_target - self.right)

        dead = self.deadband * limit
        left = 0.0 if abs(self.left) < dead else self.left
        right = 0.0 if abs(self.right) < dead else self.right

        left *= 1.0 + float(self.rng.normal(0.0, self.slip_sigma))
        right *= 1.0 + float(self.rng.normal(0.0, self.slip_sigma))
        return robot.wheels_to_body(left, right)


def integrate(pose: Pose, v: float, w: float, dt: float) -> Pose:
    """Exact constant-curvature arc.

    Not the Euler step ``x += v*cos(theta)*dt``: that step's error is always
    towards the inside of the turn, so it does not average out, and over a 90 deg
    corner at 30 Hz it accumulates into a visible heading bias.  Since the point of
    this harness is to measure millimetres, the integrator must not be the thing
    contributing them.
    """
    x, y, th = pose
    if abs(w) < 1e-9:
        return (x + v * dt * math.cos(th), y + v * dt * math.sin(th), wrap_angle(th))
    r = v / w
    th2 = th + w * dt
    return (
        x + r * (math.sin(th2) - math.sin(th)),
        y - r * (math.cos(th2) - math.cos(th)),
        wrap_angle(th2),
    )


@dataclass
class EpisodeResult:
    success: bool
    state: str
    message: str
    duration: float
    true_errors: tuple[float, float, float]      # (ex, ey, eyaw) at the target coil, ground truth
    radial_offset: float
    efficiency: float
    relative_efficiency: float
    retries: int
    fixes_accepted: int
    fixes_rejected: int
    estimate_error: tuple[float, float, float]   # believed minus true, at the end
    frames: int
    trace: list = field(default_factory=list)

    def one_line(self) -> str:
        ex, ey, eyaw = self.true_errors
        tag = "CHARGING" if self.success else self.state
        return (
            f"{tag:9s} t={self.duration:6.2f}s  x={ex * 1000:+7.2f}mm  y={ey * 1000:+7.2f}mm  "
            f"yaw={math.degrees(eyaw):+6.2f}deg  r={self.radial_offset * 1000:5.2f}mm  "
            f"eta={self.efficiency * 100:5.1f}%  retries={self.retries}  fixes={self.fixes_accepted}"
        )


def run_episode(
    settings: Settings,
    start_coil: int,
    target_coil: int,
    seed: int,
    *,
    placement_pos_sigma: float = 0.010,
    placement_yaw_sigma: float = math.radians(3.0),
    collect_trace: bool = False,
) -> EpisodeResult:
    rng = np.random.default_rng(seed)
    board = settings.board
    tags = build_tags(board)
    cams_nominal = nominal_cameras(settings.cameras)
    cams_true = perturbed_cameras(settings.cameras, settings.detection, rng)
    detector = TagDetectorSim(cams_true, tags, settings.detection, rng)
    odom = OdometryModel(settings.odometry, rng)

    route = plan_route(board, start_coil, target_coil)
    start_heading = route.legs[0].heading if route.legs else 0.0

    nominal_start: Pose = (*board.coil_positions[start_coil], start_heading)
    true_pose: Pose = (
        nominal_start[0] + float(rng.normal(0.0, placement_pos_sigma)),
        nominal_start[1] + float(rng.normal(0.0, placement_pos_sigma)),
        wrap_angle(nominal_start[2] + float(rng.normal(0.0, placement_yaw_sigma))),
    )

    # The robot is told where it was placed, not where it actually is.
    estimator = PoseEstimator(settings, tags, cams_nominal, nominal_start)
    mission = MissionController(settings, start_coil, target_coil, initial_heading=start_heading)
    truth_link = LinkMonitor(settings.dock, settings.wpt)

    dt = 1.0 / settings.sim.control_hz
    camera_every = max(1, int(round(settings.sim.control_hz / settings.sim.camera_hz)))
    offset = settings.robot.rx_coil_offset
    max_steps = int(settings.sim.mission_timeout * settings.sim.control_hz) + 10

    actuator = Actuator(settings, rng)
    trace: list = []
    status = None
    t = 0.0

    for step in range(max_steps):
        if step % camera_every == 0:
            estimator.update(detector.detect(true_pose))

        believed_coil_pose = compose(estimator.pose, (offset[0], offset[1], 0.0))
        status = mission.step(believed_coil_pose, dt)

        true_coil_pose = compose(true_pose, (offset[0], offset[1], 0.0))
        tex, tey, teyaw = mission.coil_errors(true_coil_pose)
        truth_state = truth_link.evaluate(tex, tey, teyaw)

        if collect_trace:
            trace.append(
                {
                    "t": t,
                    "state": status.state,
                    "true": true_pose,
                    "est": estimator.pose,
                    "v": status.v,
                    "w": status.w,
                    "true_err": (tex, tey, teyaw),
                    "eta": truth_state.efficiency,
                }
            )

        if status.finished:
            break

        v_act, w_act = actuator.apply(status.v, status.w, dt)
        true_pose = integrate(true_pose, v_act, w_act, dt)
        estimator.predict(odom.measure(v_act, w_act), dt)
        t += dt

    true_coil_pose = compose(true_pose, (offset[0], offset[1], 0.0))
    tex, tey, teyaw = mission.coil_errors(true_coil_pose)
    final = truth_link.evaluate(tex, tey, teyaw)
    est_err = estimator.error_against(true_pose)

    return EpisodeResult(
        success=bool(status and status.success),
        state=status.state if status else "NO_STEP",
        message=status.message if status else "",
        duration=t,
        true_errors=(tex, tey, teyaw),
        radial_offset=final.radial_offset,
        efficiency=final.efficiency,
        relative_efficiency=final.relative,
        retries=status.retries if status else 0,
        fixes_accepted=estimator.fixes_accepted,
        fixes_rejected=estimator.fixes_rejected,
        estimate_error=est_err,
        frames=step + 1,
        trace=trace,
    )


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------


@dataclass
class Campaign:
    results: list[EpisodeResult] = field(default_factory=list)

    def add(self, r: EpisodeResult) -> None:
        self.results.append(r)

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def successes(self) -> list[EpisodeResult]:
        return [r for r in self.results if r.success]

    def summary(self) -> str:
        if not self.results:
            return "no episodes"
        ok = self.successes
        rate = 100.0 * len(ok) / self.n
        lines = [f"episodes {self.n}, charging {len(ok)} ({rate:.1f} %)"]

        if ok:
            def stats(vals: list[float], scale: float, unit: str) -> str:
                a = np.abs(np.asarray(vals)) * scale
                return (
                    f"mean {a.mean():6.2f} {unit}  p95 {np.percentile(a, 95):6.2f} {unit}  "
                    f"max {a.max():6.2f} {unit}"
                )

            lines.append("  ground-truth error at the coil, over charging runs:")
            lines.append(f"    |longitudinal| {stats([r.true_errors[0] for r in ok], 1000, 'mm')}")
            lines.append(f"    |lateral|      {stats([r.true_errors[1] for r in ok], 1000, 'mm')}")
            lines.append(f"    |yaw|          {stats([r.true_errors[2] for r in ok], 180.0 / math.pi, 'deg')}")
            lines.append(f"    radial         {stats([r.radial_offset for r in ok], 1000, 'mm')}")
            etas = np.asarray([r.efficiency for r in ok]) * 100.0
            lines.append(f"    eta            min {etas.min():5.1f} %  mean {etas.mean():5.1f} %")
            durations = np.asarray([r.duration for r in ok])
            lines.append(f"    duration       mean {durations.mean():5.1f} s  max {durations.max():5.1f} s")
            retries = np.asarray([r.retries for r in ok])
            lines.append(f"    retries        mean {retries.mean():4.2f}  max {int(retries.max())}")
            lines.append("  estimator error at the end (believed minus true):")
            lines.append(f"    |dx|,|dy|      {stats([r.estimate_error[0] for r in ok], 1000, 'mm')}")
            lines.append(f"    |dyaw|         {stats([r.estimate_error[2] for r in ok], 180.0 / math.pi, 'deg')}")

        failures: dict[str, int] = {}
        for r in self.results:
            if not r.success:
                failures[f"{r.state}: {r.message[:90]}"] = failures.get(f"{r.state}: {r.message[:90]}", 0) + 1
        if failures:
            lines.append("  failures:")
            for msg, count in sorted(failures.items(), key=lambda kv: -kv[1]):
                lines.append(f"    x{count}  {msg}")
        return "\n".join(lines)


def run_campaign(
    settings: Settings,
    pairs: list[tuple[int, int]],
    repeats: int,
    *,
    base_seed: int = 20260811,
) -> dict[str, Campaign]:
    """One campaign per coil pair, plus an ``all`` aggregate."""
    out: dict[str, Campaign] = {"all": Campaign()}
    for start, target in pairs:
        key = f"{start}->{target}"
        camp = Campaign()
        for i in range(repeats):
            seed = base_seed + 1000 * (10 * start + target) + i
            r = run_episode(settings, start, target, seed)
            camp.add(r)
            out["all"].add(r)
        out[key] = camp
    return out


