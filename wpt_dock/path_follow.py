"""The steering law.

This is the part the request asked me to design rather than borrow, so the
reasoning is written out.

**What is wrong with pure pursuit for this task.**  Pure pursuit picks a point
one look-ahead ahead on the path and drives the unique circular arc to it.  Two
consequences make it the wrong tool for wireless-power docking:

* It has no terminal-orientation authority at all.  The arc is chosen to hit a
  *position*; the heading the robot happens to have on arrival is whatever the
  geometry produced.  Docking needs position *and* heading.
* As the look-ahead point approaches the end of the path, the commanded
  curvature ``2*sin(alpha)/L`` degenerates -- the denominator shrinks while the
  numerator does not -- so the last few centimetres are exactly where the law is
  least trustworthy.  Shrinking the look-ahead to fix accuracy makes it
  oscillate; growing it to fix oscillation makes it cut corners.  There is no
  setting that is both accurate and stable at the end of a path.

**What this uses instead.**  A Frenet-frame law with a Lyapunov certificate.
With cross-track error ``e`` (positive left), heading error ``psi = theta -
theta_path``, and path curvature ``k``, the exact error dynamics of a unicycle
are

    s_dot = v*cos(psi) / (1 - k*e)
    e_dot = v*sin(psi)
    psi_dot = w - k*s_dot

Command

    w = k*s_dot - k_psi*psi - k_cross*v*sinc(psi)*e            (sinc(x)=sin x / x)

and take ``V = 0.5*k_cross*e^2 + 0.5*psi^2``.  Then

    V_dot = k_cross*e*v*sin(psi) + psi*(w - k*s_dot)
          = k_cross*e*v*sin(psi) - k_psi*psi^2 - k_cross*v*e*sin(psi)
          = -k_psi*psi^2  <= 0

The cross-track terms cancel *identically* -- no small-angle assumption, no
linearisation, no bound on ``e``.  That is the whole reason for the ``sinc``
factor, and it is why this law does not have pure pursuit's "accurate or stable,
pick one" problem.

**Gains that mean something.**  Rewriting the linearised dynamics in arc length
``x`` instead of time (``d/dx = (1/v) d/dt``) gives

    e'' + 2*zeta*lam*e' + lam^2*e = 0      with   k_cross = lam^2,  k_psi = 2*zeta*lam*v

which contains no ``v``.  The error therefore decays per *metre travelled*, not
per second, so one pair of gains works from cruise speed down to docking crawl.
``lam`` is "inverse e-folding distance of the cross-track error" in 1/m and
``zeta`` is the damping ratio.  Both are quantities you can reason about with a
ruler, which is what makes the feasibility test at the bottom of this file
possible at all.

**Saturation preserves curvature.**  When a limit binds, ``v`` and ``w`` are
scaled by the *same* factor.  Clipping them independently changes ``w/v``, i.e.
it changes the arc the robot drives, so the robot silently stops tracking the
geometry it was commanded -- a slow, hard-to-see failure that looks like a
tuning problem and is not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .config import FollowGains, RobotSpec, SpeedProfile
from .geometry import clamp, sinc_unnormalised, wrap_angle
from .routes import Projection


@dataclass
class SteerResult:
    v: float                     # m/s, forward speed command (negative = reverse)
    w: float                     # rad/s, yaw-rate command
    lateral: float               # m, signed cross-track error at this instant
    heading_error: float         # rad, wrapped
    progress: float              # m, arc length along the reference
    remaining: float             # m, arc length left
    index: int                   # reference sample index, feed back as the next hint
    mode: str                    # "follow" | "spin" | "arrived"
    saturation: float = 1.0      # <1 means a limit bound and the command was scaled
    curvature_cmd: float = 0.0   # 1/m, w/v of the issued command
    extras: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Actuator limits
# ---------------------------------------------------------------------------


def saturate(v: float, w: float, robot: RobotSpec, max_yaw: float) -> tuple[float, float, float]:
    """Scale ``(v, w)`` down until it is achievable, keeping ``w/v`` fixed.

    Returns ``(v, w, scale)``.  Two limits are enforced: the body yaw-rate cap
    (a control-authority reserve, so the docking stage always has some turning
    left over) and the per-wheel angular rate.
    """
    scale = 1.0
    if max_yaw > 0.0 and abs(w) > max_yaw:
        s = max_yaw / abs(w)
        v, w, scale = v * s, w * s, scale * s

    left, right = robot.body_to_wheels(v, w)
    peak = max(abs(left), abs(right))
    limit = robot.max_wheel_rate
    if peak > limit and peak > 0.0:
        s = limit / peak
        v, w, scale = v * s, w * s, scale * s
    return v, w, scale


# ---------------------------------------------------------------------------
# Speed shaping
# ---------------------------------------------------------------------------


def shape_speed(
    base: float,
    curvature: float,
    heading_error: float,
    remaining: float,
    speeds: SpeedProfile,
    approach_gain: float = 1.4,
) -> float:
    """Multiplicatively reduce the base speed for curvature, misalignment, and arrival.

    Every factor is <= 1, so the result never exceeds ``base``.  The arrival term
    is a floor-limited linear ramp rather than the usual ``sqrt(2*a*d)``: a square
    root has infinite slope at ``d = 0``, which is precisely where we want the
    gentlest behaviour.
    """
    v = abs(base)

    if abs(curvature) > 1e-6 and speeds.lateral_accel_budget > 0.0:
        v = min(v, math.sqrt(speeds.lateral_accel_budget / abs(curvature)))

    psi = abs(heading_error)
    if psi > speeds.heading_taper:
        span = max(1e-6, speeds.heading_gate - speeds.heading_taper)
        v *= max(0.0, (speeds.heading_gate - psi) / span)

    if math.isfinite(remaining):
        v = min(v, max(speeds.creep, approach_gain * max(0.0, remaining)))

    return v


def spin_command(current_yaw: float, target_yaw: float, speeds: SpeedProfile) -> tuple[float, float]:
    """In-place rotation towards ``target_yaw``.

    Two-speed on purpose: a single gain either takes forever from 180 deg away or
    overshoots at 2 deg away, because the wheels have static friction that a
    proportional command below ``min_move`` cannot break.
    """
    err = wrap_angle(target_yaw - current_yaw)
    rate = speeds.spin_rate if abs(err) > speeds.spin_fine_band else speeds.spin_rate_fine
    return 0.0, math.copysign(rate, err)


# ---------------------------------------------------------------------------
# The follower
# ---------------------------------------------------------------------------


def follow(
    reference,
    pose: tuple[float, float, float],
    *,
    gains: FollowGains,
    speeds: SpeedProfile,
    robot: RobotSpec,
    hint: int = 0,
    reverse: bool = False,
    speed_cap: float | None = None,
    stop_at: float | None = None,
    arrive_tol: float = 0.01,
) -> SteerResult:
    """One control step against a reference curve.

    ``reverse=True`` drives the robot backwards along ``reference``.  It is
    implemented by treating the vehicle as a *virtual* unicycle whose heading is
    rotated by pi and whose speed is positive, then negating the resulting ``v``.
    The law is then applied unchanged, which is legitimate because a unicycle
    driving backwards along a curve is exactly a unicycle driving forwards along
    it with the body flipped -- so the same Lyapunov certificate carries over
    rather than being re-derived by hand and hoped for.  The caller must pass the
    reference already oriented in the direction of travel (see
    ``RayReference.reversed``).
    """
    x, y, theta = pose
    if reverse:
        theta = wrap_angle(theta + math.pi)

    proj = reference.project(x, y, hint)
    psi = wrap_angle(theta - proj.heading)

    remaining = proj.remaining if stop_at is None else (stop_at - proj.s)
    if remaining <= arrive_tol:
        return SteerResult(
            v=0.0, w=0.0, lateral=proj.lateral, heading_error=psi,
            progress=proj.s, remaining=remaining, index=proj.index, mode="arrived",
        )

    # Large heading error: a differential drive should spin, not swing out on an
    # arc.  Swinging out is what produces the "robot loops around the goal"
    # behaviour that plagues arc-based trackers started off-heading.
    if abs(psi) > speeds.heading_gate:
        target = proj.heading
        v_cmd, w_cmd = spin_command(theta, target, speeds)
        w_cmd = clamp(w_cmd, -gains.max_yaw_rate, gains.max_yaw_rate)
        v_out = -0.0 if reverse else 0.0
        return SteerResult(
            v=v_out, w=w_cmd, lateral=proj.lateral, heading_error=psi,
            progress=proj.s, remaining=remaining, index=proj.index, mode="spin",
        )

    base = speeds.cruise if speed_cap is None else speed_cap
    v = shape_speed(base, proj.curvature, psi, remaining, speeds)
    v = max(v, speeds.min_move) if v > 0.0 else 0.0

    # Frenet feed-forward.  The (1 - k*e) factor is exact, not a refinement:
    # without it the feed-forward is wrong by the fraction of the turn radius the
    # robot is offset by, which on a tight corner is tens of percent.  Clamped
    # because at e = 1/k the robot sits on the centre of curvature and s_dot is
    # genuinely undefined.
    denom = clamp(1.0 - proj.curvature * proj.lateral, 0.2, 5.0)
    s_dot = v * math.cos(psi) / denom

    k_psi = gains.k_heading(v)
    w = proj.curvature * s_dot - k_psi * psi - gains.k_cross * v * sinc_unnormalised(psi) * proj.lateral

    v, w, scale = saturate(v, w, robot, gains.max_yaw_rate)
    curvature_cmd = (w / v) if abs(v) > 1e-9 else 0.0

    if reverse:
        v = -v

    return SteerResult(
        v=v, w=w, lateral=proj.lateral, heading_error=psi,
        progress=proj.s, remaining=remaining, index=proj.index, mode="follow",
        saturation=scale, curvature_cmd=curvature_cmd,
        extras={"s_dot": s_dot, "k_psi": k_psi},
    )


# ---------------------------------------------------------------------------
# Feasibility: can this alignment finish in the distance that is left?
# ---------------------------------------------------------------------------


def convergence_distance(
    lateral: float,
    heading_error: float,
    gains: FollowGains,
    pos_tol: float,
    yaw_tol: float,
    *,
    horizon: float = 4.0,
    step: float = 0.004,
) -> float:
    """Arc length after which the follower is *guaranteed* to be inside tolerance.

    This is the piece that turns docking from hopeful into decidable.  Because
    the gain design made the error dynamics purely spatial

        e'' + 2*zeta*lam*e' + lam^2*e = 0,   e' = psi

    the trajectory of ``(e, psi)`` from the current state is fully determined by
    distance travelled -- speed does not enter.  So we can integrate it forward
    and ask: after what distance is the error inside tolerance *and staying
    there*?  Comparing that number with the distance actually remaining to the
    pad tells the state machine, before it commits, whether this approach can
    possibly succeed.  If it cannot, backing off and re-approaching is the only
    correct move, and the machine takes it instead of grinding into the pad
    misaligned.

    Implementation detail: the answer is the *last* distance at which either
    error is still out of tolerance, plus one step -- not the first distance at
    which both happen to be inside.  With any damping the trajectory can dip
    through tolerance and come back out, and a "first crossing" answer would be
    optimistic exactly when it matters.

    Returns ``inf`` if it never settles inside ``horizon``.
    """
    lam = gains.decay_rate
    zeta = gains.damping
    e = float(lateral)
    psi = float(heading_error)

    n = max(1, int(horizon / step))
    last_bad = -1

    def accel(e_: float, psi_: float) -> float:
        return -2.0 * zeta * lam * psi_ - lam * lam * e_

    for i in range(n + 1):
        if abs(e) > pos_tol or abs(psi) > yaw_tol:
            last_bad = i
        # RK4 on (e' = psi, psi' = -2*zeta*lam*psi - lam^2*e)
        k1e, k1p = psi, accel(e, psi)
        k2e, k2p = psi + 0.5 * step * k1p, accel(e + 0.5 * step * k1e, psi + 0.5 * step * k1p)
        k3e, k3p = psi + 0.5 * step * k2p, accel(e + 0.5 * step * k2e, psi + 0.5 * step * k2p)
        k4e, k4p = psi + step * k3p, accel(e + step * k3e, psi + step * k3p)
        e += step * (k1e + 2.0 * k2e + 2.0 * k3e + k4e) / 6.0
        psi += step * (k1p + 2.0 * k2p + 2.0 * k3p + k4p) / 6.0

    if last_bad < 0:
        return 0.0
    if last_bad >= n:
        return math.inf
    return (last_bad + 1) * step
