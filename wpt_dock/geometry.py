"""Planar rigid-body maths: SE(2) poses, angle bookkeeping, quaternion bridge.

Kept dependency-light on purpose -- only ``numpy``, which Isaac Sim's bundled
Python always has.  Everything in here is pure and unit-testable without Isaac.

Conventions, stated once so the rest of the package can stop repeating them:

* A pose is ``(x, y, yaw)``: position in metres, yaw in radians, measured
  counter-clockwise from +X, right-handed about +Z.
* ``T_a_b`` names the transform that takes coordinates *expressed in b* and
  returns them *expressed in a*.  Composition reads left to right:
  ``T_a_c = compose(T_a_b, T_b_c)``.
* Quaternions from USD / Isaac Sim are ``(w, x, y, z)``.  ``scipy`` and many
  ROS-flavoured APIs use ``(x, y, z, w)``.  The two helpers below are named for
  their layout so the ordering can never be ambiguous at a call site.
"""

from __future__ import annotations

import math

import numpy as np

TAU = 2.0 * math.pi

Pose = tuple[float, float, float]


# ---------------------------------------------------------------------------
# Angles
# ---------------------------------------------------------------------------


def wrap_angle(a: float) -> float:
    """Fold an angle into (-pi, pi]."""
    a = math.remainder(a, TAU)
    # math.remainder returns a value in [-pi, pi]; normalise the -pi edge to +pi
    # so that wrap(pi) and wrap(-pi) agree and comparisons stay stable.
    if a <= -math.pi:
        a += TAU
    return a


def wrap_angle_array(a: np.ndarray) -> np.ndarray:
    return (np.asarray(a, dtype=float) + math.pi) % TAU - math.pi


def angle_diff(a: float, b: float) -> float:
    """Smallest signed rotation taking ``b`` to ``a``."""
    return wrap_angle(a - b)


def sinc_unnormalised(x: float) -> float:
    """``sin(x) / x`` with the removable singularity at 0 handled exactly.

    The path-following law needs this factor; evaluating ``sin(x)/x`` naively
    is a division by zero precisely at the equilibrium the controller is trying
    to reach, which is the one place the code must not blow up.  For small ``x``
    the Taylor series is both faster and more accurate than the quotient.
    """
    ax = abs(x)
    if ax < 1e-4:
        x2 = x * x
        return 1.0 - x2 / 6.0 + x2 * x2 / 120.0
    return math.sin(x) / x


# ---------------------------------------------------------------------------
# Rotations and SE(2)
# ---------------------------------------------------------------------------


def rot2(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=float)


def compose(t_a_b: Pose, t_b_c: Pose) -> Pose:
    """``T_a_c = T_a_b * T_b_c``."""
    xa, ya, ta = t_a_b
    xb, yb, tb = t_b_c
    c, s = math.cos(ta), math.sin(ta)
    return (xa + c * xb - s * yb, ya + s * xb + c * yb, wrap_angle(ta + tb))


def invert(t_a_b: Pose) -> Pose:
    """``T_b_a`` given ``T_a_b``."""
    x, y, t = t_a_b
    c, s = math.cos(t), math.sin(t)
    return (-(c * x + s * y), -(-s * x + c * y), wrap_angle(-t))


def transform_point(t_a_b: Pose, p_b: np.ndarray) -> np.ndarray:
    """Map one point, or an ``(N, 2)`` array of points, from frame b into frame a."""
    x, y, t = t_a_b
    p = np.asarray(p_b, dtype=float)
    r = rot2(t)
    return p @ r.T + np.array([x, y])


def relative_pose(t_w_ref: Pose, t_w_target: Pose) -> Pose:
    """Pose of ``target`` expressed in the frame of ``ref``.

    This is the workhorse of the docking controller: with ``ref`` = the pad and
    ``target`` = the robot, the three components are exactly the longitudinal,
    lateral and heading errors the task is specified in.
    """
    return compose(invert(t_w_ref), t_w_target)


# ---------------------------------------------------------------------------
# Jacobians (needed to carry covariance through the frame changes)
# ---------------------------------------------------------------------------


def invert_jacobian(t_a_b: Pose) -> np.ndarray:
    """d(invert(T)) / dT for T in SE(2), as a 3x3 matrix on ``(x, y, yaw)``.

    Derivation: with ``T = (t, th)``, ``T^-1 = (-R(th)^T t, -th)``.
    ``d/dth R(th)^T = -R(th)^T J`` with ``J = [[0,-1],[1,0]]``, so the
    translation block picks up ``+R(-th) J t``.
    """
    x, y, th = t_a_b
    rt = rot2(-th)
    j = np.array([[0.0, -1.0], [1.0, 0.0]])
    out = np.zeros((3, 3))
    out[:2, :2] = -rt
    out[:2, 2] = (rt @ j @ np.array([x, y]))
    out[2, 2] = -1.0
    return out


def compose_jacobian_right(t_a_b: Pose) -> np.ndarray:
    """d(compose(T_a_b, T_b_c)) / d(T_b_c), holding ``T_a_b`` fixed.

    Only the rotation of the left operand matters, so this is a block-diagonal
    rotation.  Used to push a covariance expressed in frame ``b`` into frame
    ``a`` when ``T_a_b`` itself is known exactly (our pad pose is map data, so
    it is exact by construction).
    """
    _, _, th = t_a_b
    out = np.eye(3)
    out[:2, :2] = rot2(th)
    return out


# ---------------------------------------------------------------------------
# Quaternion bridge (USD / Isaac Sim use w-first)
# ---------------------------------------------------------------------------


def yaw_from_quat_wxyz(q) -> float:
    """Extract yaw about +Z from a (w, x, y, z) quaternion.

    Uses the full atan2 form rather than a small-angle shortcut so it stays
    correct for any orientation, and tolerates the un-normalised quaternions
    that occasionally come back from physics queries.
    """
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return 0.0
    w, x, y, z = w / n, x / n, y / n, z / n
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def quat_wxyz_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    half = 0.5 * yaw
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def roll_pitch_from_quat_wxyz(q) -> tuple[float, float]:
    """Roll and pitch, used only to notice that the robot has tipped over."""
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return 0.0, 0.0
    w, x, y, z = w / n, x / n, y / n, z / n
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sinp)
    return roll, pitch


# ---------------------------------------------------------------------------
# Small helpers used in several modules
# ---------------------------------------------------------------------------


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def clamp_abs(v: float, limit: float) -> float:
    limit = abs(limit)
    return clamp(v, -limit, limit)


def hypot2(dx: float, dy: float) -> float:
    return math.sqrt(dx * dx + dy * dy)
