"""Registration ("정합") and state estimation.

Two distinct meanings of 정합 are both implemented here, because the task needs
both and conflating them is how docking demos end up cheating:

1. **Geometric registration** -- ``fit_rigid_2d`` recovers the rigid transform
   between the known marker layout on the dock and the marker positions the
   robot actually observes.  Closed form, no iteration, and it reports its own
   uncertainty.
2. **Physical alignment** -- the controller then drives that transform to
   identity.  See ``dock.py``.

Nothing in this package feeds ground-truth pose to the controller.  The robot
navigates on wheel odometry fused with these registrations, so the accuracy the
demo achieves is an accuracy it earned.  Reading the simulator's exact transform
into the control loop would make any tolerance reachable and would prove nothing.

Why closed form instead of ICP: correspondences here are *known*.  Each marker is
a distinguishable target with an ID, exactly as an AprilTag or a coded
retroreflector would be on real hardware.  ICP exists to discover unknown
correspondences, and running it when correspondences are known throws away
information, adds local minima, and turns an exact one-shot least-squares
solution into an iteration that can fail to converge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .geometry import (
    Pose,
    compose,
    compose_jacobian_right,
    invert,
    invert_jacobian,
    rot2,
    wrap_angle,
)


@dataclass
class RegistrationResult:
    """Outcome of one marker fit.

    ``pose`` is the transform that maps *model* coordinates into the *observation*
    frame.  With model = pad-frame marker layout and observations expressed in the
    robot frame, ``pose`` is the pad's pose as seen by the robot.
    """

    ok: bool
    pose: Pose
    covariance: np.ndarray          # 3x3 on (x, y, yaw)
    residual_rms: float             # m
    n_points: int
    reason: str = ""


_J2 = np.array([[0.0, -1.0], [1.0, 0.0]])


def fit_rigid_2d(
    model: np.ndarray,
    observed: np.ndarray,
    sigma: float = 0.002,
    *,
    weights: np.ndarray | None = None,
    max_residual: float = 0.030,
) -> RegistrationResult:
    """Least-squares planar rigid transform, in closed form.

    ``weights`` are per-point ``1 / sigma_i^2``.  They matter here rather than
    being a nicety: a tag corner back-projected from the front camera at 0.25 m
    and 60 deg obliquity is several times noisier than one seen from a bottom
    camera looking straight down at 0.12 m, and letting those vote equally throws
    away most of the precision the close view offers.  When ``weights`` is given
    the covariance comes from the information matrix directly and ``sigma`` is
    ignored.

    Minimises ``sum |q_i - (R(th) p_i + t)|^2``.  Subtracting the two centroids
    removes ``t`` from the rotation sub-problem, and the remaining scalar problem
    has the exact solution

        th = atan2( sum (p_i' x q_i'), sum (p_i' . q_i') )

    (cross and dot of the centred vectors).  ``t`` then follows directly.  This is
    the 2D specialisation of the Kabsch/Umeyama result, with the reflection case
    excluded by construction since a single ``atan2`` cannot produce one.

    The covariance is the Gauss-Newton one, ``sigma^2 * (J^T J)^-1``, built from
    the analytic Jacobian rather than assumed diagonal.  It has real
    cross-covariance between yaw and translation whenever the marker centroid is
    not at the model origin, and pretending otherwise would make the filter
    overconfident precisely along the direction that matters.

    The reported ``sigma`` is the larger of the a-priori sensor sigma and the
    unbiased residual estimate.  Trusting the residual alone is unwise with four
    points (5 degrees of freedom), and trusting the a-priori value alone hides
    genuine model error such as an occluded or mis-associated marker.
    """
    p = np.asarray(model, dtype=float).reshape(-1, 2)
    q = np.asarray(observed, dtype=float).reshape(-1, 2)
    n = len(p)
    if len(q) != n:
        return RegistrationResult(False, (0.0, 0.0, 0.0), np.eye(3), math.inf, n, "size mismatch")
    if n < 2:
        return RegistrationResult(False, (0.0, 0.0, 0.0), np.eye(3), math.inf, n, "need >= 2 points")

    if weights is None:
        w = np.ones(n)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if len(w) != n or not np.all(np.isfinite(w)) or float(w.sum()) <= 0.0:
            return RegistrationResult(False, (0.0, 0.0, 0.0), np.eye(3), math.inf, n, "bad weights")

    w_sum = float(w.sum())
    p_bar = (w[:, None] * p).sum(axis=0) / w_sum
    q_bar = (w[:, None] * q).sum(axis=0) / w_sum
    pc = p - p_bar
    qc = q - q_bar

    if float((w * (pc * pc).sum(axis=1)).sum()) < 1e-12:
        return RegistrationResult(False, (0.0, 0.0, 0.0), np.eye(3), math.inf, n, "degenerate model")

    cross = float((w * (pc[:, 0] * qc[:, 1] - pc[:, 1] * qc[:, 0])).sum())
    dot = float((w * (pc * qc).sum(axis=1)).sum())
    theta = math.atan2(cross, dot)

    r = rot2(theta)
    t = q_bar - r @ p_bar

    residual = q - (p @ r.T + t)
    per_point = (residual * residual).sum(axis=1)
    ss = float(per_point.sum())
    dof = max(1, 2 * n - 3)
    rms_unbiased = math.sqrt(ss / dof)

    if rms_unbiased > max_residual:
        return RegistrationResult(
            False, (float(t[0]), float(t[1]), theta), np.eye(3) * 1e3,
            rms_unbiased, n, f"residual {rms_unbiased:.4f} m exceeds gate",
        )

    a = (r @ (_J2 @ p.T)).T            # d(residual_i)/d(theta) up to sign, shape (n, 2)
    jtj = np.zeros((3, 3))
    jtj[0, 0] = jtj[1, 1] = w_sum
    jtj[0, 2] = jtj[2, 0] = float((w * a[:, 0]).sum())
    jtj[1, 2] = jtj[2, 1] = float((w * a[:, 1]).sum())
    jtj[2, 2] = float((w * (a * a).sum(axis=1)).sum())

    try:
        if weights is None:
            sigma_eff = max(float(sigma), rms_unbiased)
            cov = (sigma_eff * sigma_eff) * np.linalg.inv(jtj)
        else:
            cov = np.linalg.inv(jtj)
            # Variance inflation when the fit is worse than the weights claim.
            # Without this, a systematic error -- a mis-measured mount angle, say --
            # produces a tight covariance around a wrong pose, and the filter then
            # trusts it over perfectly good odometry.  Inflating never makes the
            # estimate worse; omitting it can make the whole system confidently
            # wrong, which is the failure mode that is hardest to notice.
            chi2 = float((w * per_point).sum())
            cov = cov * max(1.0, chi2 / dof)
    except np.linalg.LinAlgError:
        return RegistrationResult(False, (float(t[0]), float(t[1]), theta), np.eye(3) * 1e3,
                                  rms_unbiased, n, "singular information matrix")

    # Symmetrise: the inverse of a symmetric matrix is symmetric in exact
    # arithmetic, and a filter fed a slightly asymmetric covariance can pick up a
    # bias that is very hard to find later.
    cov = 0.5 * (cov + cov.T)

    return RegistrationResult(True, (float(t[0]), float(t[1]), wrap_angle(theta)), cov, rms_unbiased, n)


def robot_pose_from_registration(
    fit: RegistrationResult,
    model_frame_in_world: Pose = (0.0, 0.0, 0.0),
) -> tuple[Pose, np.ndarray]:
    """Turn "where the model frame is, as seen by me" into "where I am, in the world".

    ``fit.pose`` maps model coordinates into the robot frame, so it *is* the model
    frame's pose as observed from the robot: ``T_rob_model``.  Two frame changes
    get us to a robot pose, each with its covariance carried through the analytic
    Jacobian rather than assumed to pass through unchanged:

        T_model_rob = inverse(T_rob_model)              -- Jacobian: invert_jacobian
        T_world_rob = T_world_model * T_model_rob       -- Jacobian: a rotation

    For the tag work the model frame is the board frame and the board frame is the
    world frame, so the default identity applies and the second step is a no-op.
    The argument exists for the case where the tag layout is surveyed into a
    larger frame -- and note that if that survey were itself uncertain, its
    covariance would have to be added here rather than ignored.
    """
    t_rob_model = fit.pose
    t_model_rob = invert(t_rob_model)
    j_inv = invert_jacobian(t_rob_model)
    cov_model_rob = j_inv @ fit.covariance @ j_inv.T

    t_world_rob = compose(model_frame_in_world, t_model_rob)
    j_comp = compose_jacobian_right(model_frame_in_world)
    cov_world_rob = j_comp @ cov_model_rob @ j_comp.T
    return t_world_rob, 0.5 * (cov_world_rob + cov_world_rob.T)


# ---------------------------------------------------------------------------
# Extended Kalman filter on SE(2)
# ---------------------------------------------------------------------------


class Ekf2D:
    """(x, y, yaw) filter: unicycle odometry prediction, absolute pose updates.

    The prediction step integrates the *exact* constant-curvature arc rather than
    the Euler step ``x += v*cos(theta)*dt``.  At 60 Hz and 1.2 rad/s the Euler
    error per step is small but it is systematically *inward* on every turn, so it
    accumulates into a heading-dependent bias -- the classic "odometry says I
    turned less than I did" artefact.  The arc form has no such bias.

    Process noise is proportional to the commanded motion (slip scales with how
    far you drive), plus a small floor.  Without the floor the covariance decays
    towards zero while parked and the filter then refuses to believe a perfectly
    good registration.
    """

    def __init__(
        self,
        pose: Pose,
        position_sigma: float = 0.05,
        yaw_sigma: float = 0.05,
        *,
        floor_position_sigma: float = 0.0,
        floor_yaw_sigma: float = 0.0,
    ) -> None:
        self.x = np.array([pose[0], pose[1], wrap_angle(pose[2])], dtype=float)
        self.p = np.diag([position_sigma ** 2, position_sigma ** 2, yaw_sigma ** 2])
        self.floor = np.array(
            [floor_position_sigma ** 2, floor_position_sigma ** 2, floor_yaw_sigma ** 2], dtype=float
        )
        self.updates = 0
        self.consecutive_rejections = 0

    def enforce_floor(self) -> None:
        """Stop the covariance from shrinking below what calibration allows.

        Absolute measurements here share a fixed calibration error, so they are not
        independent samples and averaging cannot beat that error.  A filter that
        assumes independence drives ``P`` towards zero, and an overconfident filter
        does something worse than report a wrong uncertainty: its own gate starts
        rejecting the correct measurements that disagree with it, and it never comes
        back.  The floor is the statement "no number of frames tells me my pose
        better than my mounts are known".

        Applied by scaling row and column ``i`` by ``sqrt(floor_i / P_ii)``.  That is
        ``D P D`` with ``D`` diagonal and positive, so symmetry and positive
        definiteness both survive -- unlike overwriting the diagonal entry, which can
        leave the matrix indefinite and make later updates produce nonsense.
        """
        for i in range(3):
            f = float(self.floor[i])
            if f <= 0.0:
                continue
            v = float(self.p[i, i])
            if v < f:
                scale = math.sqrt(f / max(v, 1e-18))
                self.p[i, :] *= scale
                self.p[:, i] *= scale
                self.p[i, i] = f

    @property
    def pose(self) -> Pose:
        return (float(self.x[0]), float(self.x[1]), float(self.x[2]))

    @property
    def position_sigma(self) -> float:
        return math.sqrt(max(0.0, float(self.p[0, 0] + self.p[1, 1])) / 2.0)

    @property
    def yaw_sigma(self) -> float:
        return math.sqrt(max(0.0, float(self.p[2, 2])))

    def predict(
        self,
        v: float,
        w: float,
        dt: float,
        *,
        speed_noise: float,
        yaw_noise: float,
        yaw_from_speed: float,
        floor_pos: float,
        floor_yaw: float,
    ) -> None:
        if dt <= 0.0:
            return
        th = float(self.x[2])
        small = abs(w) < 1e-4

        if small:
            # Second-order expansion, continuous with the exact arc as w -> 0.
            dx = v * dt * math.cos(th) - 0.5 * v * w * dt * dt * math.sin(th)
            dy = v * dt * math.sin(th) + 0.5 * v * w * dt * dt * math.cos(th)
            f = np.array([
                [1.0, 0.0, -v * dt * math.sin(th) - 0.5 * v * w * dt * dt * math.cos(th)],
                [0.0, 1.0, v * dt * math.cos(th) - 0.5 * v * w * dt * dt * math.sin(th)],
                [0.0, 0.0, 1.0],
            ])
            g = np.array([
                [dt * math.cos(th) - 0.5 * w * dt * dt * math.sin(th), -0.5 * v * dt * dt * math.sin(th)],
                [dt * math.sin(th) + 0.5 * w * dt * dt * math.cos(th), 0.5 * v * dt * dt * math.cos(th)],
                [0.0, dt],
            ])
        else:
            r = v / w
            th2 = th + w * dt
            s0, c0 = math.sin(th), math.cos(th)
            s1, c1 = math.sin(th2), math.cos(th2)
            dx = r * (s1 - s0)
            dy = -r * (c1 - c0)
            f = np.array([
                [1.0, 0.0, r * (c1 - c0)],
                [0.0, 1.0, r * (s1 - s0)],
                [0.0, 0.0, 1.0],
            ])
            g = np.array([
                [(s1 - s0) / w, -v * (s1 - s0) / (w * w) + r * dt * c1],
                [-(c1 - c0) / w, v * (c1 - c0) / (w * w) + r * dt * s1],
                [0.0, dt],
            ])

        self.x = np.array([self.x[0] + dx, self.x[1] + dy, wrap_angle(th + w * dt)])

        sv = speed_noise * abs(v)
        sw = yaw_noise * abs(w) + yaw_from_speed * abs(v)
        q_u = np.diag([sv * sv, sw * sw])
        q_floor = np.diag([floor_pos ** 2, floor_pos ** 2, floor_yaw ** 2])
        self.p = f @ self.p @ f.T + g @ q_u @ g.T + q_floor
        self.p = 0.5 * (self.p + self.p.T)

    def update_pose(self, z: Pose, cov: np.ndarray, *, max_rejections: int = 5) -> bool:
        """Absolute (x, y, yaw) measurement.  Returns False if it was rejected."""
        r = np.asarray(cov, dtype=float).reshape(3, 3)
        y = np.array([z[0] - self.x[0], z[1] - self.x[1], wrap_angle(z[2] - self.x[2])])
        s = self.p + r
        try:
            s_inv = np.linalg.inv(s)
        except np.linalg.LinAlgError:
            return False

        # Chi-square gate on 3 dof.  16.3 is the 0.999 quantile: loose enough not to
        # fight a genuinely surprising but correct fix after odometry drift, tight
        # enough to drop a mis-associated tag set, which would otherwise teleport the
        # estimate straight off the board.
        mahal = float(y @ s_inv @ y)
        if not math.isfinite(mahal) or mahal > 16.3:
            self.consecutive_rejections += 1
            # A gate that keeps firing is evidence against the *filter*, not against
            # the measurements: consistent rejections mean the state is wrong and too
            # confident to admit it.  Inflating lets the next fix back in.  Without
            # this escape a single early bad update is permanent, which is precisely
            # the divergence the floor above is there to prevent -- belt and braces,
            # because the cost of being wrong here is a silent 30 mm error.
            if self.consecutive_rejections >= max_rejections:
                self.p *= 4.0
                self.consecutive_rejections = 0
            return False

        self.consecutive_rejections = 0
        k = self.p @ s_inv
        self.x = self.x + k @ y
        self.x[2] = wrap_angle(float(self.x[2]))
        i_kh = np.eye(3) - k
        self.p = i_kh @ self.p @ i_kh.T + k @ r @ k.T   # Joseph form: stays positive definite
        self.p = 0.5 * (self.p + self.p.T)
        self.enforce_floor()
        self.updates += 1
        return True
