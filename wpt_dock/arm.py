"""Kinematics and sequencing for the ``cute_arm`` 3-DOF arm.

Hardware, from the two upstream repositories (gyeong-yeon2718/cute_arm and its parent
elevenMiles/Robotic_Arm_Seven):

* base yaw, shoulder, elbow -- **three** joints -- plus a gripper servo; 4x SG90.
* shoulder-to-elbow 12.0 cm, elbow-to-gripper 12.0 cm (both repos agree).
* usable reach 3 cm to 23 cm measured from the shoulder pivot, against a 24 cm
  geometric maximum.
* documented rest pose ``(12, 0, 12)`` cm with the origin at the shoulder pivot.
* gripper open 45 deg, closed 0 deg; servo speed 30-90 deg/s.

Two consequences of *three* joints that shape everything below
--------------------------------------------------------------
**The arm cannot choose its approach orientation.**  Base yaw plus two in-plane joints
gives exactly three degrees of freedom, which is enough to place the gripper at a
commanded *position* and nothing left over to aim it.  Whatever wrist angle the elbow-up
solution happens to produce is the angle you get.  So the pick and place targets below
are specified as positions with a vertical approach, and the gripper's tilt is reported
rather than commanded.  Any code that tried to command a grasp orientation would be
writing a cheque the mechanism cannot cash.

**The inverse kinematics are closed form, and are derived here rather than ported.**
Equal 12 cm links make the planar sub-problem a textbook two-link solution, so there is
no reason to reach for an iterative solver.  Deliberately *not* copied from the upstream
firmware: cute_arm's own ``docs/findings.md`` is a list of firmware bugs and
documentation discrepancies, so porting that implementation would mean inheriting them.
The derivation here is checked against the one pose the upstream documents -- the
``(12, 0, 12)`` cm rest pose -- in ``tests/test_core.py``.

Joint angle convention, stated once:

* ``base``     rotation about the vertical; 0 points along the robot's +X.
* ``shoulder`` upper-arm elevation from horizontal, positive up.
* ``elbow``    forearm angle **relative to the upper arm**; 0 is straight out,
  negative folds the forearm down.  This is the servo's own variable, which is why the
  solver returns it rather than the forearm's absolute angle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .geometry import clamp, wrap_angle


@dataclass(frozen=True)
class ArmSpec:
    """Geometry and limits.  MEASURED unless marked otherwise."""

    l_upper: float = 0.120           # m, shoulder -> elbow, MEASURED (both repos)
    l_fore: float = 0.120            # m, elbow -> gripper, MEASURED (both repos)
    reach_min: float = 0.030         # m, MEASURED (documented usable minimum)
    reach_max: float = 0.230         # m, MEASURED (documented usable maximum)

    # ESTIMATED: neither repository documents the height from the mounting surface to the
    # shoulder pivot.  0.055 m is a typical SG90 base-rotation bracket plus shoulder
    # mount.  It shifts the whole workspace vertically, so measure it if a target ends up
    # marginally out of reach.
    base_height: float = 0.055       # m, plate top -> shoulder pivot
    mount_offset: tuple[float, float] = (-0.015, 0.0)  # m, on the custom top plate

    # Servo travel.  SG90s are nominally 0-180 deg; these are the sub-ranges the
    # mechanism can actually use without the links colliding.  ESTIMATED.
    base_range: tuple[float, float] = (math.radians(-90.0), math.radians(90.0))
    shoulder_range: tuple[float, float] = (math.radians(-10.0), math.radians(150.0))
    elbow_range: tuple[float, float] = (math.radians(-150.0), math.radians(10.0))

    gripper_open: float = math.radians(45.0)    # MEASURED
    gripper_closed: float = 0.0                 # MEASURED

    # How close a joint has to get before a move counts as arrived.  0.5 deg is about the
    # best an SG90 resolves, so this is a hardware limit rather than a tuning knob -- and it
    # is a *visible* one: at the ~0.18 m working radius used here, half a degree is 1.6 mm
    # of gripper position, and two joints contribute.  It is the largest single term in the
    # measured placement error, ahead of the docking error itself.  Tightening it would
    # simulate a better servo than the robot has.
    joint_tolerance: float = math.radians(0.5)
    # 30-90 deg/s is the documented range; the middle of it is the honest default, and it
    # is what makes the arm take a believable few seconds rather than snapping.
    servo_rate: float = math.radians(60.0)      # rad/s
    gripper_rate: float = math.radians(120.0)   # rad/s

    @property
    def max_reach(self) -> float:
        return self.l_upper + self.l_fore

    def gripper_span(self, opening: float) -> float:
        """Jaw separation for a gripper servo angle, for drawing the jaws.

        Linear in the servo angle, which is what a simple two-finger linkage gives to
        first order.  ESTIMATED 34 mm at full open.
        """
        frac = 0.0 if self.gripper_open <= 0 else clamp(opening / self.gripper_open, 0.0, 1.0)
        return 0.006 + 0.028 * frac


@dataclass(frozen=True)
class ArmPose:
    """A joint-space pose.  ``elbow`` is relative to the upper arm."""

    base: float
    shoulder: float
    elbow: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.base, self.shoulder, self.elbow)


HOME = ArmPose(0.0, math.radians(90.0), math.radians(-90.0))
"""The documented rest pose: upper arm vertical, forearm horizontal, tip at (12, 0, 12) cm."""

CARRY = ArmPose(0.0, math.radians(115.0), math.radians(-125.0))
"""Folded in over the deck.  Used while driving: it keeps the payload inside the robot's
support footprint so a carried object cannot swing out past the plywood edge, and it keeps
the arm clear of the downward cameras' field of view."""


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------


def forward_kinematics(spec: ArmSpec, pose: ArmPose) -> tuple[float, float, float]:
    """Gripper tip position relative to the **shoulder pivot**."""
    q_fore = pose.shoulder + pose.elbow          # forearm's absolute elevation
    r = spec.l_upper * math.cos(pose.shoulder) + spec.l_fore * math.cos(q_fore)
    z = spec.l_upper * math.sin(pose.shoulder) + spec.l_fore * math.sin(q_fore)
    return (r * math.cos(pose.base), r * math.sin(pose.base), z)


def elbow_position(spec: ArmSpec, pose: ArmPose) -> tuple[float, float, float]:
    r = spec.l_upper * math.cos(pose.shoulder)
    z = spec.l_upper * math.sin(pose.shoulder)
    return (r * math.cos(pose.base), r * math.sin(pose.base), z)


@dataclass
class IkResult:
    ok: bool
    pose: ArmPose | None = None
    reason: str = ""
    reach: float = 0.0
    gripper_tilt: float = 0.0     # rad, forearm elevation -- reported, not commanded


def solve_ik(spec: ArmSpec, target: tuple[float, float, float], *, elbow_up: bool = True) -> IkResult:
    """Closed-form inverse kinematics.  ``target`` is relative to the shoulder pivot.

    Base yaw comes straight from ``atan2``.  What remains is a two-link planar reach in the
    vertical plane containing the target, which has the standard cosine-rule solution:

        d      = |target|
        alpha  = angle at the shoulder between the target line and the upper arm
        beta   = interior angle at the elbow
        shoulder = atan2(z, r) + alpha        (elbow-up branch; minus for elbow-down)
        elbow    = -(pi - beta)               (relative, negative folds down)

    Reachability is checked against the *documented usable* 3-23 cm envelope rather than
    the 24 cm geometric maximum: at full extension the links are colinear, the Jacobian is
    singular, and the servos stall trying to hold it.  Refusing those targets here is why
    the caller never has to handle a pose that looks valid and cannot be held.
    """
    x, y, z = target
    r = math.hypot(x, y)
    d = math.hypot(r, z)

    if d > spec.reach_max:
        return IkResult(False, reason=f"target {d * 100:.1f} cm beyond the {spec.reach_max * 100:.0f} cm reach", reach=d)
    if d < spec.reach_min:
        return IkResult(False, reason=f"target {d * 100:.1f} cm inside the {spec.reach_min * 100:.0f} cm dead zone", reach=d)

    l1, l2 = spec.l_upper, spec.l_fore
    cos_alpha = clamp((d * d + l1 * l1 - l2 * l2) / (2.0 * l1 * d), -1.0, 1.0)
    cos_beta = clamp((l1 * l1 + l2 * l2 - d * d) / (2.0 * l1 * l2), -1.0, 1.0)
    alpha = math.acos(cos_alpha)
    beta = math.acos(cos_beta)

    base = math.atan2(y, x)
    sign = 1.0 if elbow_up else -1.0
    shoulder = math.atan2(z, r) + sign * alpha
    elbow = -sign * (math.pi - beta)

    pose = ArmPose(wrap_angle(base), wrap_angle(shoulder), wrap_angle(elbow))
    for name, value, (lo, hi) in (
        ("base", pose.base, spec.base_range),
        ("shoulder", pose.shoulder, spec.shoulder_range),
        ("elbow", pose.elbow, spec.elbow_range),
    ):
        if not (lo - 1e-9 <= value <= hi + 1e-9):
            return IkResult(
                False,
                reason=f"{name} would need {math.degrees(value):+.1f} deg, limit "
                       f"[{math.degrees(lo):+.0f}, {math.degrees(hi):+.0f}]",
                reach=d,
            )
    return IkResult(True, pose=pose, reach=d, gripper_tilt=pose.shoulder + pose.elbow)


def base_frame_target(
    spec: ArmSpec,
    robot_pose: tuple[float, float, float],
    plate_top_z: float,
    world_target: tuple[float, float, float],
) -> tuple[float, float, float]:
    """World point -> shoulder-pivot frame, given the robot's planar pose.

    This is where the whole project joins up: the arm's accuracy at a fixed shelf is the
    accuracy of ``robot_pose``, and ``robot_pose`` is what the WPT alignment pins down.  A
    1 cm base error is a 1 cm placement error -- the arm cannot see the shelf and has no way
    to notice.
    """
    x, y, yaw = robot_pose
    mx, my = spec.mount_offset
    c, s = math.cos(yaw), math.sin(yaw)
    pivot = (x + c * mx - s * my, y + s * mx + c * my, plate_top_z + spec.base_height)
    dx, dy, dz = (world_target[0] - pivot[0], world_target[1] - pivot[1], world_target[2] - pivot[2])
    return (c * dx + s * dy, -s * dx + c * dy, dz)


# ---------------------------------------------------------------------------
# Sequencing
# ---------------------------------------------------------------------------

IDLE = "IDLE"
MOVING = "MOVING"
GRIPPING = "GRIPPING"
DONE = "DONE"
FAILED = "FAILED"


@dataclass
class Waypoint:
    """One step of a pick or place sequence."""

    name: str
    world_target: tuple[float, float, float] | None = None   # solve IK to reach this
    joint_target: ArmPose | None = None                      # or go straight to these angles
    gripper: float | None = None                             # or drive the gripper to this
    settle: float = 0.15                                     # s to hold once arrived


@dataclass
class ArmState:
    pose: ArmPose = HOME
    gripper: float = math.radians(45.0)
    holding: bool = False
    step: int = 0
    phase: str = IDLE
    message: str = ""


class ArmSequencer:
    """Rate-limited joint-space playback of a waypoint list.

    Joints move at the servo rate rather than jumping, because the timing matters to the
    demo and because an instantaneous joint change would fling a kinematically carried
    payload.  Arrival is per joint, so a move finishes when the slowest one does.
    """

    def __init__(self, spec: ArmSpec, tolerance: float | None = None) -> None:
        self.spec = spec
        self.tolerance = spec.joint_tolerance if tolerance is None else tolerance
        self.state = ArmState()
        self.waypoints: list[Waypoint] = []
        self._hold = 0.0
        self._target = HOME
        self._gripper_target = spec.gripper_open
        self._ik_context: tuple[tuple[float, float, float], float] | None = None

    # -- control ---------------------------------------------------------

    def start(self, waypoints: list[Waypoint]) -> None:
        self.waypoints = list(waypoints)
        self.state.step = 0
        self.state.phase = MOVING if self.waypoints else DONE
        self.state.message = "starting" if self.waypoints else "nothing to do"
        self._hold = 0.0
        self._begin_step()

    @property
    def finished(self) -> bool:
        return self.state.phase in (DONE, FAILED)

    @property
    def succeeded(self) -> bool:
        return self.state.phase == DONE

    def _begin_step(self) -> None:
        if self.state.step >= len(self.waypoints):
            self.state.phase = DONE
            self.state.message = "sequence complete"
            return
        wp = self.waypoints[self.state.step]
        self._hold = 0.0
        if wp.gripper is not None:
            self._gripper_target = wp.gripper
            self.state.phase = GRIPPING
            self.state.message = f"{wp.name}: gripper -> {math.degrees(wp.gripper):.0f} deg"
            return
        if wp.joint_target is not None:
            self._target = wp.joint_target
        elif wp.world_target is not None and self._ik_context is not None:
            robot_pose, plate_z = self._ik_context
            local = base_frame_target(self.spec, robot_pose, plate_z, wp.world_target)
            result = solve_ik(self.spec, local)
            if not result.ok or result.pose is None:
                self.state.phase = FAILED
                self.state.message = f"{wp.name}: unreachable -- {result.reason}"
                return
            self._target = result.pose
        else:
            self.state.phase = FAILED
            self.state.message = f"{wp.name}: no target and no IK context"
            return
        self.state.phase = MOVING
        self.state.message = wp.name

    def set_ik_context(self, robot_pose: tuple[float, float, float], plate_top_z: float) -> None:
        """Where the robot is, for solving world-frame waypoints.

        Refreshed every step while stationary so the arm uses the pose the alignment
        actually achieved, not the pose it was commanded to.
        """
        self._ik_context = (robot_pose, plate_top_z)

    # -- integration -----------------------------------------------------

    def step(self, dt: float) -> ArmState:
        if self.finished:
            return self.state
        wp = self.waypoints[self.state.step]

        if self.state.phase == GRIPPING:
            arrived = self._advance_gripper(dt)
        else:
            arrived = self._advance_joints(dt)

        if arrived:
            self._hold += dt
            if self._hold >= wp.settle:
                self.state.step += 1
                self._begin_step()
        return self.state

    def _advance_joints(self, dt: float) -> bool:
        limit = self.spec.servo_rate * dt
        cur = self.state.pose.as_tuple()
        tgt = self._target.as_tuple()
        out = []
        arrived = True
        for c, t in zip(cur, tgt):
            err = t - c
            if abs(err) > self.tolerance:
                arrived = False
            out.append(c + clamp(err, -limit, limit))
        self.state.pose = ArmPose(*out)
        return arrived

    def _advance_gripper(self, dt: float) -> bool:
        limit = self.spec.gripper_rate * dt
        err = self._gripper_target - self.state.gripper
        self.state.gripper += clamp(err, -limit, limit)
        return abs(err) <= self.tolerance


# ---------------------------------------------------------------------------
# Sequences
# ---------------------------------------------------------------------------


def pick_sequence(grasp_point: tuple[float, float, float], spec: ArmSpec,
                  approach: float = 0.060) -> list[Waypoint]:
    """Vertical approach, close, lift, fold for transit.

    The approach is straight down onto the object.  With three joints the gripper's tilt is
    whatever the elbow-up solution gives, so "vertical approach" here means the *path* is
    vertical, not that the jaws are held vertical -- see the module docstring.
    """
    above = (grasp_point[0], grasp_point[1], grasp_point[2] + approach)
    return [
        Waypoint("open the gripper", gripper=spec.gripper_open),
        Waypoint("reach above the object", world_target=above, settle=0.2),
        Waypoint("descend onto the object", world_target=grasp_point, settle=0.3),
        Waypoint("close the gripper", gripper=spec.gripper_closed, settle=0.3),
        Waypoint("lift", world_target=above, settle=0.2),
        Waypoint("fold for transit", joint_target=CARRY, settle=0.2),
    ]


def place_sequence(drop_point: tuple[float, float, float], spec: ArmSpec,
                   approach: float = 0.060) -> list[Waypoint]:
    """Unfold, descend onto the target, release, retract."""
    above = (drop_point[0], drop_point[1], drop_point[2] + approach)
    return [
        Waypoint("unfold", joint_target=HOME, settle=0.2),
        Waypoint("reach above the target", world_target=above, settle=0.2),
        Waypoint("descend to the target", world_target=drop_point, settle=0.3),
        Waypoint("release", gripper=spec.gripper_open, settle=0.3),
        Waypoint("retract", world_target=above, settle=0.2),
        Waypoint("home", joint_target=HOME, settle=0.1),
    ]


def workspace_report(spec: ArmSpec) -> str:
    lines = [
        f"cute_arm: links {spec.l_upper * 100:.1f} + {spec.l_fore * 100:.1f} cm, "
        f"usable reach {spec.reach_min * 100:.0f}-{spec.reach_max * 100:.0f} cm "
        f"(geometric max {spec.max_reach * 100:.0f} cm)",
        f"  shoulder pivot {spec.base_height * 1000:.0f} mm above the plate, mounted at "
        f"({spec.mount_offset[0] * 1000:+.0f}, {spec.mount_offset[1] * 1000:+.0f}) mm",
        "  3 joints: position-only IK, gripper tilt is a consequence not a command",
    ]
    tip = forward_kinematics(spec, HOME)
    lines.append(f"  HOME tip at ({tip[0] * 100:.1f}, {tip[1] * 100:.1f}, {tip[2] * 100:.1f}) cm "
                 f"relative to the pivot -- upstream documents (12.0, 0.0, 12.0)")
    return "\n".join(lines)
