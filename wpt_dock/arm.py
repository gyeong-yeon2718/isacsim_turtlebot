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

    # Gripper mount -> the point a held object's centre sits at, along the forearm axis.
    # DESIGN, and it has to exist: without it the kinematics solve for the *wrist*, while the
    # thing that has to land on the pad is the payload centre 16 mm further out.  The two
    # differ by exactly this much along the forearm, and with the forearm pointing nearly
    # straight down at a shelf that is almost all vertical -- which is how it was found.  Every
    # place and pick came out 15-16 mm low, burying the box in the shelf, and the bias was
    # constant because it is geometry and not noise.  It matches the ``tcp`` prim in
    # ``isaac/arm_build.py``; the two must move together.
    # Gripper mount -> the point a held object's centre sits at, along the forearm axis.
    #
    # **Derived from the jaw geometry, not a constant.**  It used to be 16 mm, which put the tool
    # point just past the wrist -- nowhere near where this gripper actually holds anything.  The
    # tips meet at ``jaw_pivot_x + jaw_tip_reach`` from the gripper mount, so that is where an
    # object is gripped and that is the point the kinematics must aim.  Getting this wrong is not
    # cosmetic: the IK solves for this point, so a tool offset that does not match the hardware
    # moves every grasp and every placement by the difference.
    @property
    def l_tool(self) -> float:
        """Gripper mount -> tool centre point: the tip contact, where the jaws meet."""
        return self.jaw_pivot_x + self.jaw_tip_reach

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
    def distal(self) -> float:
        """Elbow -> tool centre point.  The effective second link for FK and IK.

        The forearm and the tool offset are rigidly in line, so the kinematics cannot tell
        them apart and there is no reason to carry them separately through the solve.  They
        are separate *inputs* because one is measured and one is a design choice.
        """
        return self.l_fore + self.l_tool

    @property
    def max_reach(self) -> float:
        return self.l_upper + self.distal

    # Jaw pad *centre* separation at the servo's travel limits, and the pad thickness.
    #
    # ESTIMATED, and the estimate is this project's own: both upstream repositories document
    # the gripper purely as an angle-commanded peripheral (``GO`` / ``GC`` / ``G(30)``) and
    # state no jaw travel, no opening width in millimetres, and no maximum graspable object
    # size anywhere in their READMEs, findings, or firmware notes.  The only remaining physical
    # source of truth upstream is ``gripper.stl``.  So these are labelled guesses, and when the
    # payload did not fit, the payload is what moved -- widening the jaws would have meant
    # inventing a second unsourced number on top of the first.
    span_closed: float = 0.006       # m, pad centre separation at gripper_closed
    span_open: float = 0.034         # m, pad centre separation at gripper_open
    jaw_pad_thickness: float = 0.004  # m, matches the pad boxes in isaac/arm_build.py
    # How far past the object's surface the jaws are commanded, so contact carries real force.
    # DESIGN: printed PLA jaws on an SG90 flex about this much before the servo stalls.
    # Corroborated by cute_arm's own README, which warns that ``gripper_close = 0`` puts the servo
    # against a mechanical stopper and asks whether it overheats -- so closing onto the object
    # rather than to the stop is what the hardware wants, not just what looks better.
    grip_squeeze: float = 0.0006      # m

    # --- the gripper is a scissor pair, not parallel jaws ----------------------------------
    #
    # Each finger pivots on a **vertical 2 mm screw**.  That is measured, not assumed:
    # ``gripper.stl`` contains exactly one circular bore, 1.90 mm across, and its axis lies along
    # the file's Y with the bore passing through a 2 mm wall.  1.90 mm is a 2 mm screw from the
    # BOM, not an SG90's 4.8 mm spline, so it is a pivot and not a drive shaft -- which is what the
    # user meant by assembling these with screws.  Requiring that bore to end up vertical is also
    # what finally fixes the part's assembled orientation: rotating the print pose 90 degrees about
    # X takes the bore from Y to Z, and the same rotation turns the thin blade into a horizontal
    # lever and the far plate into a 29 mm-tall vertical gripping face.  A scissor gripper.
    #
    # MEASURED from the mesh, in the rotated (assembled) frame:
    #   bore centre        8.08 mm along the lever from the part's near end
    #   gripping face      46..92 mm along the lever, so its middle is about 61 mm from the bore
    # ONE moving jaw against ONE fixed jaw, and the tips meet when it shuts.  From the user's
    # photographs of the built arm: ``gripper_upper_arm.stl`` is bolted to the forearm and does not
    # move -- it reaches forward and its hooked tip *is* the fixed jaw.  ``gripper.stl``, of which
    # there is exactly one, is screwed to the gripper servo's horn and swings about that horn's
    # vertical axis.  One servo, one moving part, like a pair of pliers.
    #
    # This replaces a symmetric two-finger scissor model that was wrong in structure, not just in
    # numbers: it used two copies of the jaw and moved both.
    # Both MEASURED, and the measurement corrected an earlier guess twice over.
    #
    # ``gripper.stl`` has TWO bores along its Y, at the same height: a 1.80 mm screw hole at
    # x = 8.08 and an **8.20 mm** hole at x = 22.04.  The user described exactly this -- two small
    # circles and a slightly larger one in the middle -- and the larger one is the SG90 horn's
    # spline boss, so *that* is the pivot.  An earlier revision used the 1.80 mm hole and put the
    # pivot 14 mm too far back.
    #
    # ``gripper_upper_arm.stl`` is the forearm *and* the gripper's fixed half: 149 mm long, with
    # the servo pocket cut between x = 104 and 134, so the horn axis lands near 120 mm from the
    # elbow.  That is the documented ``LENGTH_ELBOW_GRIPPER = 12 cm``, which is a genuine
    # cross-check rather than a coincidence -- the documented forearm length is measured to the
    # gripper *servo*, not to the end of the part.  So the gripper frame already sits on the horn
    # and the jaw pivots at that frame's origin.
    jaw_pivot_x: float = 0.0          # m, gripper frame -> horn axis: they coincide
    jaw_tip_reach: float = 0.0701     # m, horn axis -> tip.  92.16 - 22.04 from the mesh

    def jaw_rotation(self, opening: float) -> float:
        """Angle the **single** moving jaw is swung open, for a servo angle.  Zero is shut.

        ``gripper_span`` stays the authority on the opening -- it is what ``grip_angle_for``
        inverts and what the tests pin -- and this converts that gap into the swing that produces
        it.  Because the tips meet when closed, the gap *is* the tip's lateral travel, so
        ``gap = jaw_tip_reach * sin(theta)`` and nothing else enters.

        The consequence worth noticing: an 84 mm jaw opening 30 mm needs 21 degrees of swing, so
        the servo's half-degree resolution is worth about 0.7 mm at the tip.  That is a real
        limitation of the mechanism rather than of the model.
        """
        s = self.gripper_span(opening) / self.jaw_tip_reach
        return math.asin(clamp(s, -1.0, 1.0))

    def gripper_span(self, opening: float) -> float:
        """Jaw pad **centre** separation for a gripper servo angle.

        Linear in the servo angle, which is what a simple two-finger linkage gives to
        first order.  This is the quantity the USD build writes to the jaw transforms.
        """
        frac = 0.0 if self.gripper_open <= 0 else clamp(opening / self.gripper_open, 0.0, 1.0)
        return self.span_closed + (self.span_open - self.span_closed) * frac

    def gripper_clear(self, opening: float) -> float:
        """The usable gap between the pads' inner faces.

        The pads have thickness, and it is the *faces* that touch an object, not the centres
        the transforms are placed at.  Ignoring the 4 mm leaves the pads overlapping the object
        by 2 mm each -- which reads as jaws sunk into the box, the very thing being fixed.
        """
        return self.gripper_span(opening) - self.jaw_pad_thickness

    def grip_angle_for(self, width: float) -> float:
        """The servo angle at which the pad faces come to rest on an object ``width`` wide.

        A gripper does not close to its mechanical limit around an object; it closes *until it
        touches*, and the object's width sets where that is.  Commanding ``gripper_closed``
        instead is why the jaws visibly passed through the payload: the servo went to 6 mm of
        separation while the box was 35 mm across, so 14.5 mm of box stuck out through each
        jaw.  The user's report was "the object is too big for the gripper", and it was both
        that and this.

        Raises rather than clamps if the object cannot be gripped at all.  Clamping would give
        a pose that looks like a grasp and is not one, which is the failure mode this whole
        module is trying to avoid.
        """
        lo = self.gripper_clear(self.gripper_closed)
        hi = self.gripper_clear(self.gripper_open)
        if not (lo <= width <= hi):
            raise ValueError(
                f"a {width * 1000:.1f} mm object is outside the jaws' usable "
                f"{lo * 1000:.0f}-{hi * 1000:.0f} mm gap"
            )
        span = width + self.jaw_pad_thickness
        frac = (span - self.span_closed) / (self.span_open - self.span_closed)
        return self.gripper_open * frac


# The payload the gripper carries.  It lives here, beside the jaw travel that determines it,
# rather than in the USD scene module: it is a *specification* constrained by the hardware, and
# putting it behind a ``pxr`` import meant the tests could not reference it and hardcoded 17.5 mm
# instead -- which then silently disagreed with the scene.
PAYLOAD_SIZE = 0.025             # m, cube edge.  Must lie strictly inside the jaws' travel.
PAYLOAD_MASS = 0.040             # kg, a light printed/cardboard box an SG90 could lift
PAYLOAD_HALF = 0.5 * PAYLOAD_SIZE


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
    """**Tool centre point** relative to the shoulder pivot.

    Not the wrist: this is where a held object's centre goes, which is the point every
    waypoint in this module is expressed in and the point the USD ``tcp`` prim marks.
    """
    q_fore = pose.shoulder + pose.elbow          # forearm's absolute elevation
    r = spec.l_upper * math.cos(pose.shoulder) + spec.distal * math.cos(q_fore)
    z = spec.l_upper * math.sin(pose.shoulder) + spec.distal * math.sin(q_fore)
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

    l1, l2 = spec.l_upper, spec.distal
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
    # Keep the *current* base yaw and apply only the shoulder/elbow of ``joint_target``.
    # This is what lets the arm lift straight up in its own vertical plane before slewing --
    # the standard way to leave a shelf without dragging the payload across whatever is beside
    # it.  Rotating and lifting at the same time is what put the payload through the deck.
    hold_base: bool = False


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
            jt = wp.joint_target
            self._target = (
                ArmPose(self.state.pose.base, jt.shoulder, jt.elbow) if wp.hold_base else jt
            )
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


def payload_height(spec: ArmSpec, pose: ArmPose, plate_top_z: float) -> float:
    """World z of the carried payload's centre for a joint pose.

    Exists because the sequences below have to be checked against the robot's own deck, not
    just against reachability.  ``plate_top_z`` is the printed plate's upper surface; the
    shoulder pivot is ``base_height`` above it.
    """
    return plate_top_z + spec.base_height + forward_kinematics(spec, pose)[2]


def path_clearance(
    spec: ArmSpec, a: ArmPose, b: ArmPose, plate_top_z: float, payload_half: float,
    plate_half: tuple[float, float] = (0.0462, 0.1150), samples: int = 60,
) -> float:
    """Smallest gap between the payload's underside and the plate along a joint-space move.

    Only samples where the payload is actually **over** the plate in XY are counted.  The
    first version of this compared height everywhere, which flagged the perfectly legal pose
    of reaching down to a shelf beside the robot -- the plate is not above the shelf.  A
    clearance metric that fires on safe poses is worse than none, because it gets ignored.

    Joint-space interpolation does not move the tip in a straight line, so two endpoints that
    both clear the deck can still be joined by an arc that does not.  Returns ``inf`` if the
    move never passes over the plate, and a negative number if the payload goes through it.
    """
    worst = math.inf
    for i in range(samples + 1):
        t = i / samples
        pose = ArmPose(
            a.base + t * (b.base - a.base),
            a.shoulder + t * (b.shoulder - a.shoulder),
            a.elbow + t * (b.elbow - a.elbow),
        )
        tip = forward_kinematics(spec, pose)
        x = spec.mount_offset[0] + tip[0]
        y = spec.mount_offset[1] + tip[1]
        if abs(x) > plate_half[0] + payload_half or abs(y) > plate_half[1] + payload_half:
            continue                      # payload is beside the deck, not above it
        z = plate_top_z + spec.base_height + tip[2]
        worst = min(worst, z - payload_half - plate_top_z)
    return worst


def pick_sequence(grasp_point: tuple[float, float, float], spec: ArmSpec,
                  approach: float = 0.060, grip_width: float | None = None) -> list[Waypoint]:
    """Vertical approach, close, lift, raise clear of the deck, then fold for transit.

    The approach is straight down onto the object.  With three joints the gripper's tilt is
    whatever the elbow-up solution gives, so "vertical approach" means the *path* is
    vertical, not that the jaws are held vertical -- see the module docstring.

    The ``raise clear`` step is not decoration.  Interpolating straight from the lift pose to
    ``CARRY`` in joint space swings the payload's underside down to about z = 0.160 m while
    the printed plate's top is at 0.165 m -- so it passed through the deck, which is exactly
    what it looked like.  Going by way of ``HOME`` keeps the whole arc above the plate, and
    ``path_clearance`` is the function that checks it rather than trusting the eye.
    """
    above = (grasp_point[0], grasp_point[1], grasp_point[2] + approach)
    # Close onto the object, not to the mechanical stop.  ``grip_angle_for`` refuses an object
    # the jaws cannot span, so an unsuitable payload fails here rather than being mimed.
    #
    # ``squeeze`` matters once the grasp is a contact grasp: commanding exactly the box's width
    # leaves the pads just touching, contact force near zero, and friction with nothing to work
    # against -- the box slides out.  A position-controlled servo does not stop at the surface, it
    # drives *into* it and stalls, and the printed jaw flexes.  0.6 mm of overlap is what turns
    # "the pads are adjacent to the box" into "the pads are holding the box".
    close_to = (spec.gripper_closed if grip_width is None
                else spec.grip_angle_for(max(spec.gripper_clear(spec.gripper_closed),
                                             grip_width - spec.grip_squeeze)))
    return [
        Waypoint("open the gripper", gripper=spec.gripper_open),
        Waypoint("reach above the object", world_target=above, settle=0.2),
        Waypoint("descend onto the object", world_target=grasp_point, settle=0.35),
        Waypoint("close onto the object", gripper=close_to, settle=0.4),
        Waypoint("lift", world_target=above, settle=0.2),
        Waypoint("raise in place", joint_target=HOME, hold_base=True, settle=0.15),
        Waypoint("slew to centre", joint_target=HOME, settle=0.15),
        Waypoint("fold for transit", joint_target=CARRY, settle=0.2),
    ]


def place_sequence(drop_point: tuple[float, float, float], spec: ArmSpec,
                   approach: float = 0.060) -> list[Waypoint]:
    """Unfold clear of the deck, descend onto the target, release, retract."""
    above = (drop_point[0], drop_point[1], drop_point[2] + approach)
    return [
        Waypoint("unfold", joint_target=HOME, settle=0.2),
        # Slew over the deck at HOME elevation, *then* descend.  Same reason as the pick:
        # combining the yaw and the descent sweeps the payload low across the plate.
        Waypoint("slew towards the target", world_target=(above[0], above[1], above[2] + 0.070),
                 settle=0.15),
        Waypoint("reach above the target", world_target=above, settle=0.2),
        Waypoint("descend to the target", world_target=drop_point, settle=0.35),
        Waypoint("release", gripper=spec.gripper_open, settle=0.4),
        Waypoint("retract", world_target=above, settle=0.2),
        Waypoint("home", joint_target=HOME, settle=0.1),
    ]


def workspace_report(spec: ArmSpec) -> str:
    lines = [
        f"cute_arm: links {spec.l_upper * 100:.1f} + {spec.l_fore * 100:.1f} cm "
        f"(+{spec.l_tool * 1000:.0f} mm to the tool centre point), "
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
