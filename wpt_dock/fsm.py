"""The mission state machine: coil N -> coil M, then align and charge.

State graph
-----------

    TRANSIT --(leg done)--> CORNER --> TRANSIT ... --> APPROACH
                                                        ^   |
                                                        |   v
                                                    RETREAT  SETTLE --> VERIFY --> CHARGING
                                                        ^      ^  |
                                                        |      |  v
                                                (no room)------+  TRIM

Four decisions in here are the substance of the design.

**1. The approach ray, not a waypoint.**  The final leg's reference is anchored at
the target coil and points along the direction of travel, so its tangent at the
goal *is* the required docked heading.  A follower that drives cross-track and
heading error to zero therefore solves position and orientation at once.  The
upstream design instead ran pure pursuit to a point and then tried to fix
orientation in a separate fine-align stage -- with the robot already parked on the
coil, out of room, and only able to correct by spinning, which moves the coil off
centre again.  That circular dependency is designed out here rather than tuned
around.

**2. The projection *is* the error.**  Because the ray sits on the coil, its arc
length ``s`` is the longitudinal error, its ``lateral`` is the lateral error, and
the heading error is the yaw error.  There is no second coordinate transform
between "what the tracker minimises" and "what the tolerance is written in", so
the two cannot disagree and no sign can flip.

**3. Feasibility before commitment.**  A differential drive cannot remove lateral
error without driving forward.  So the machine asks ``convergence_distance``:
given the error I have and the gains I am using, how much distance does
convergence need?  If that exceeds what is left, no gain change will rescue the
attempt and the correct action is to back off and re-approach.  On an 0.80 x
0.60 m board that room is genuinely scarce, so the retreat distance is *computed*
against the board edge rather than assumed -- promising a 0.30 m retreat on a
stage with 0.17 m of margin would just drive the robot onto the floor.

**4. Arriving is not docking.**  The machine stops, waits for the chassis to
settle, and then requires the tolerances to hold continuously for the full dwell
(1.0 s, the upstream 10-frames-at-10-Hz rule) before it closes the relay.  If the
residual is yaw-only and small it spins in place -- free for a differential drive --
instead of discarding a good approach.

Why leg-1 accuracy matters more than it looks
---------------------------------------------
On a diagonal route the two legs are perpendicular, so a longitudinal error at the
end of leg 1 becomes a *lateral* error at the start of leg 2.  Lateral error is the
expensive kind: it costs approach distance to remove.  That is why non-final legs
are held to a 2 mm arrival tolerance rather than the "close enough" a waypoint
would normally get.

Estimated versus true pose
--------------------------
Everything here runs on the estimate from ``estimator.py``.  Ground truth is used
only by the runner, for logging and for the coil's brightness -- so an estimator
failure shows up as the robot confidently declaring a lock while the coil stays
dim, which is visible, instead of being hidden by a demo that reads the answer key.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .config import Settings
from .coupling import LinkMonitor, LinkState
from .geometry import Pose, wrap_angle
from .path_follow import SteerResult, convergence_distance, follow, saturate, spin_command
from .routes import (
    Leg,
    RayReference,
    Route,
    footprint_is_on_board,
    plan_route,
    retreat_room,
    route_fits_on_board,
)

TRANSIT = "TRANSIT"
CORNER = "CORNER"
APPROACH = "APPROACH"
SETTLE = "SETTLE"
VERIFY = "VERIFY"
TRIM = "TRIM"
RETREAT = "RETREAT"
CHARGING = "CHARGING"
FAILED = "FAILED"

_LEG_ARRIVE_TOL = 0.002       # m, tight because leg-1 error becomes leg-2 lateral error
_APPROACH_ARRIVE_TOL = 0.0015  # m, about one control step at the 12 mm/s floor
_CORNER_TOL = math.radians(1.5)


@dataclass
class MissionStatus:
    state: str
    v: float
    w: float
    leg_index: int
    n_legs: int
    target_coil: int
    message: str = ""
    lateral: float = 0.0
    heading_error: float = 0.0
    errors: tuple[float, float, float] | None = None    # believed (ex, ey, eyaw) at the target coil
    link: LinkState | None = None
    retries: int = 0
    required_distance: float = 0.0
    remaining: float = 0.0
    energised: bool = False
    finished: bool = False
    success: bool = False
    extras: dict = field(default_factory=dict)


class MissionController:
    def __init__(
        self,
        settings: Settings,
        start_coil: int,
        target_coil: int,
        *,
        initial_heading: float = 0.0,
    ) -> None:
        self.s = settings
        self.board = settings.board
        self.start_coil = int(start_coil)
        self.target_coil = int(target_coil)
        # Two different footprints, for two different questions.  Planning asks "is
        # this route safe at any heading the robot will pass through", so it uses the
        # swept radius.  The running check asks "is the robot on the board right now",
        # so it uses the actual rectangle at the actual heading.
        self.radius = settings.robot.swept_radius
        self.half_extents = settings.robot.footprint_half_extents

        self.route: Route = plan_route(self.board, self.start_coil, self.target_coil)
        self.legs: tuple[Leg, ...] = self.route.legs
        self.leg_index = 0
        self.state = TRANSIT
        self.message = self.route.describe()
        self.retries = 0
        self.trims = 0
        self.timer = 0.0
        self.elapsed = 0.0
        self.last_required = 0.0
        self._feasibility_clock = 0.0
        self.link = LinkMonitor(settings.dock, settings.wpt)
        self.final_errors: tuple[float, float, float] | None = None

        if self.legs:
            final = self.legs[-1]
            self.final_ray = RayReference(final.target, final.heading)
            self.final_leg: Leg | None = final
        else:
            # "Re-align on the coil I am already parked on."  There is no leg to
            # take a heading from, so the caller's current heading defines the
            # approach axis.
            pos = self.board.coil_positions[self.target_coil]
            self.final_ray = RayReference(pos, initial_heading)
            self.final_leg = None
            self.state = SETTLE
            self.message = "already on the target coil; verifying alignment"

        ok, why = route_fits_on_board(self.board, self.radius, self.route)
        if not ok:
            self.state = FAILED
            self.message = f"route does not fit on the plywood: {why}"

        self.retreat_target = -self.s.dock.approach_distance

    # -- helpers -------------------------------------------------------------

    @property
    def n_legs(self) -> int:
        return len(self.legs)

    def _current_leg(self) -> Leg:
        return self.legs[min(self.leg_index, self.n_legs - 1)]

    def coil_errors(self, pose: Pose) -> tuple[float, float, float]:
        """(longitudinal, lateral, yaw) error at the target coil, from the approach ray."""
        proj = self.final_ray.project(pose[0], pose[1])
        return proj.s, proj.lateral, wrap_angle(pose[2] - proj.heading)

    def _approach_speed(self, remaining: float) -> float:
        sp = self.s.speeds
        if remaining > self.s.dock.approach_radius:
            return sp.cruise
        if remaining > 0.030:
            return sp.approach
        return sp.dock

    def _needed_room(self, lateral: float, psi: float) -> float:
        d = convergence_distance(
            lateral, psi, self.s.docking, self.s.dock.pos_tol, self.s.dock.yaw_tol
        )
        return math.inf if not math.isfinite(d) else d + self.s.dock.feasibility_margin

    def _status(self, state: str, v: float, w: float, **kw) -> MissionStatus:
        return MissionStatus(
            state=state, v=v, w=w, leg_index=self.leg_index, n_legs=self.n_legs,
            target_coil=self.target_coil, retries=self.retries, **kw
        )

    def _finish(self, success: bool, message: str) -> MissionStatus:
        self.state = CHARGING if success else FAILED
        self.message = message
        return self._status(
            self.state, 0.0, 0.0, message=message, finished=True, success=success,
            energised=success, errors=self.final_errors,
        )

    # -- main step -----------------------------------------------------------

    def step(self, pose: Pose, dt: float) -> MissionStatus:
        """``pose`` is the *receiver coil* pose (see ``RobotSpec.rx_coil_offset``)."""
        self.elapsed += dt

        if self.state in (CHARGING, FAILED):
            ex, ey, eyaw = self.coil_errors(pose)
            link = self.link.update(dt, ex, ey, eyaw) if self.state == CHARGING else None
            return self._status(
                self.state, 0.0, 0.0, message=self.message, finished=True,
                success=self.state == CHARGING, energised=self.state == CHARGING,
                errors=(ex, ey, eyaw), link=link,
            )

        if self.elapsed > self.s.sim.mission_timeout:
            return self._finish(False, f"timed out after {self.elapsed:.1f} s in state {self.state}")

        if not footprint_is_on_board(self.board, self.half_extents, pose[0], pose[1], pose[2]):
            return self._finish(
                False,
                f"emergency stop: at ({pose[0]:.3f}, {pose[1]:.3f}, "
                f"{math.degrees(pose[2]):.1f} deg) the footprint overhangs the plywood",
            )

        handler = {
            TRANSIT: self._step_transit,
            CORNER: self._step_corner,
            APPROACH: self._step_approach,
            SETTLE: self._step_settle,
            VERIFY: self._step_verify,
            TRIM: self._step_trim,
            RETREAT: self._step_retreat,
        }[self.state]
        return handler(pose, dt)

    # -- states --------------------------------------------------------------

    def _step_transit(self, pose: Pose, dt: float) -> MissionStatus:
        leg = self._current_leg()
        if leg.is_final:
            self.state = APPROACH
            self._feasibility_clock = 0.0
            self.message = f"final approach to coil {self.target_coil}"
            return self._status(APPROACH, 0.0, 0.0, message=self.message)

        res = follow(
            leg.reference(), pose,
            gains=self.s.cruise, speeds=self.s.speeds, robot=self.s.robot,
            speed_cap=self.s.speeds.cruise, stop_at=0.0, arrive_tol=_LEG_ARRIVE_TOL,
        )
        remaining = -res.progress

        if res.mode == "arrived":
            self.leg_index += 1
            self.state = CORNER
            self.message = f"leg {leg.name} done, turning for the next leg"
            return self._status(CORNER, 0.0, 0.0, message=self.message)

        return self._status(
            TRANSIT, res.v, res.w,
            message=f"leg {leg.name} ({res.mode}), {remaining * 100:.1f} cm to go",
            lateral=res.lateral, heading_error=res.heading_error, remaining=remaining,
        )

    def _step_corner(self, pose: Pose, dt: float) -> MissionStatus:
        """In-place rotation onto the next leg's heading.

        A dedicated state, not a side effect of the follower's spin gate, because
        the tolerance wanted here (1.5 deg) is far tighter than that gate: entering
        the final leg crooked spends approach distance the feasibility budget then
        has to pay back.
        """
        if self.leg_index >= self.n_legs:
            self.state = FAILED
            return self._finish(False, "internal: corner state past the last leg")

        target = self._current_leg().heading
        err = wrap_angle(target - pose[2])
        if abs(err) <= _CORNER_TOL:
            self.state = TRANSIT
            self.message = f"aligned to leg {self._current_leg().name}"
            return self._status(TRANSIT, 0.0, 0.0, message=self.message)

        v, w = spin_command(pose[2], target, self.s.speeds)
        v, w, _ = saturate(v, w, self.s.robot, self.s.cruise.max_yaw_rate)
        return self._status(
            CORNER, v, w,
            message=f"turning {math.degrees(err):+.1f} deg", heading_error=err,
        )

    def _step_approach(self, pose: Pose, dt: float) -> MissionStatus:
        remaining_hint = -self.final_ray.project(pose[0], pose[1]).s
        res: SteerResult = follow(
            self.final_ray, pose,
            gains=self.s.docking, speeds=self.s.speeds, robot=self.s.robot,
            speed_cap=self._approach_speed(remaining_hint),
            stop_at=0.0, arrive_tol=_APPROACH_ARRIVE_TOL,
        )
        remaining = -res.progress
        ex, ey, eyaw = self.coil_errors(pose)
        link = self.link.evaluate(ex, ey, eyaw)

        self._feasibility_clock += dt
        if remaining > 0.040 and self._feasibility_clock >= 0.20:
            self._feasibility_clock = 0.0
            need = self._needed_room(res.lateral, res.heading_error)
            self.last_required = 99.0 if not math.isfinite(need) else need
            # Dead band on the decision, not just on the estimate.  The shortfall is
            # computed from a live pose estimate that jitters by a millimetre or two,
            # so an exact comparison converts that jitter into a manoeuvre -- which is
            # what produced a 16 s back-and-forth for a 1.2 cm shortfall.  Riding a
            # marginal case out costs nothing, because the follower keeps reducing the
            # error all the way in and SETTLE/VERIFY is the real judge.
            if need > remaining + self.s.dock.retreat_shortfall:
                return self._begin_retreat(
                    need,
                    f"needs {need * 100:.1f} cm to converge, only {remaining * 100:.1f} cm left",
                )

        if res.mode == "arrived":
            self.state = SETTLE
            self.timer = 0.0
            self.message = "on the coil, letting the chassis settle"
            return self._status(SETTLE, 0.0, 0.0, message=self.message, errors=(ex, ey, eyaw), link=link)

        return self._status(
            APPROACH, res.v, res.w,
            message=(
                f"approach: {remaining * 100:.1f} cm, lateral {res.lateral * 1000:+.1f} mm, "
                f"eta {link.efficiency * 100:.1f}%"
            ),
            lateral=res.lateral, heading_error=res.heading_error, errors=(ex, ey, eyaw),
            link=link, required_distance=self.last_required, remaining=remaining,
            extras={"saturation": res.saturation, "speed_cap": self._approach_speed(remaining)},
        )

    def _begin_retreat(self, need: float, why: str) -> MissionStatus:
        d = self.s.dock
        if self.retries >= d.max_retries:
            return self._finish(False, f"alignment infeasible ({why}); retry budget exhausted")
        self.retries += 1
        wanted = min(d.approach_distance_max, max(d.approach_distance, need) + d.retry_margin * self.retries)
        room = wanted
        if self.final_leg is not None:
            room = retreat_room(self.board, self.radius, self.final_leg, wanted)
        if room < d.pos_tol * 2.0:
            return self._finish(
                False,
                f"alignment infeasible ({why}) and the board edge leaves only "
                f"{room * 100:.1f} cm to back off into",
            )
        self.retreat_target = -room
        self.state = RETREAT
        self.message = f"retry {self.retries}/{d.max_retries}: {why} -> backing off {room * 100:.1f} cm"
        if room < wanted - 1e-6:
            self.message += f" (wanted {wanted * 100:.1f} cm, board allows {room * 100:.1f} cm)"
        return self._status(RETREAT, 0.0, 0.0, message=self.message, required_distance=need)

    def _step_settle(self, pose: Pose, dt: float) -> MissionStatus:
        self.timer += dt
        ex, ey, eyaw = self.coil_errors(pose)
        self.final_errors = (ex, ey, eyaw)
        link = self.link.evaluate(ex, ey, eyaw)

        if self.timer < self.s.dock.settle_time:
            return self._status(SETTLE, 0.0, 0.0, message="settling",
                                errors=(ex, ey, eyaw), link=link)

        d = self.s.dock
        pos_ok = abs(ex) <= d.pos_tol and abs(ey) <= d.pos_tol
        yaw_ok = abs(eyaw) <= d.yaw_tol

        if pos_ok and yaw_ok:
            self.state = VERIFY
            self.timer = 0.0
            self.link.reset()
            self.message = "inside tolerance, running the dwell check"
        elif pos_ok and abs(eyaw) <= d.trim_yaw_limit and self.trims < 3:
            self.trims += 1
            self.state = TRIM
            self.message = f"yaw trim {self.trims}/3: {math.degrees(eyaw):+.2f} deg"
        else:
            return self._begin_retreat(
                self._needed_room(ey, eyaw),
                f"out of tolerance (x {ex * 1000:+.1f} mm, y {ey * 1000:+.1f} mm, "
                f"yaw {math.degrees(eyaw):+.2f} deg)",
            )

        return self._status(self.state, 0.0, 0.0, message=self.message,
                            errors=(ex, ey, eyaw), link=link)

    def _step_verify(self, pose: Pose, dt: float) -> MissionStatus:
        self.timer += dt
        ex, ey, eyaw = self.coil_errors(pose)
        self.final_errors = (ex, ey, eyaw)
        link = self.link.update(dt, ex, ey, eyaw)

        if link.locked:
            return self._finish(
                True,
                f"charging on coil {self.target_coil}: x {ex * 1000:+.1f} mm, y {ey * 1000:+.1f} mm, "
                f"yaw {math.degrees(eyaw):+.2f} deg, radial {link.radial_offset * 1000:.1f} mm, "
                f"eta {link.efficiency * 100:.1f}% (held {link.held:.2f} s)",
            )

        if not link.in_tolerance:
            self.state = SETTLE
            self.timer = 0.0
            self.message = f"drifted out on {link.worst_axis} during the dwell, re-settling"
            return self._status(SETTLE, 0.0, 0.0, message=self.message,
                                errors=(ex, ey, eyaw), link=link)

        if self.timer > self.s.dock.hold_time + 2.0:
            self.state = SETTLE
            self.timer = 0.0
            self.message = "dwell did not complete, re-settling"

        return self._status(
            VERIFY, 0.0, 0.0,
            message=f"dwell {link.held:.2f}/{self.s.dock.hold_time:.2f} s, eta {link.efficiency * 100:.1f}%",
            errors=(ex, ey, eyaw), link=link,
        )

    def _step_trim(self, pose: Pose, dt: float) -> MissionStatus:
        ex, ey, eyaw = self.coil_errors(pose)
        link = self.link.evaluate(ex, ey, eyaw)
        if abs(eyaw) <= 0.4 * self.s.dock.yaw_tol:
            self.state = SETTLE
            self.timer = 0.0
            self.message = "trim done, re-settling"
            return self._status(SETTLE, 0.0, 0.0, message=self.message,
                                errors=(ex, ey, eyaw), link=link)

        # Pure rotation is the only motion a differential drive can make that, to
        # first order, leaves its own centre where it is -- which is why this is
        # safe to do while parked on the coil, and why the result is re-verified
        # rather than assumed exact.
        _, w = spin_command(eyaw, 0.0, self.s.speeds)
        _, w, _ = saturate(0.0, -w, self.s.robot, self.s.docking.max_yaw_rate)
        return self._status(
            TRIM, 0.0, w, message=f"yaw trim {math.degrees(eyaw):+.2f} deg",
            errors=(ex, ey, eyaw), heading_error=eyaw, link=link,
        )

    def _step_retreat(self, pose: Pose, dt: float) -> MissionStatus:
        proj = self.final_ray.project(pose[0], pose[1])
        if proj.s <= self.retreat_target + 0.005:
            self.state = CORNER if self.final_leg is not None else SETTLE
            if self.final_leg is not None:
                self.leg_index = self.n_legs - 1
            self.message = f"backed off to {-proj.s * 100:.1f} cm, re-entering"
            return self._status(self.state, 0.0, 0.0, message=self.message)

        # Same follower, run in reverse on the flipped ray, so the retreat also
        # *reduces* lateral error instead of merely undoing progress.  The next
        # approach therefore starts from a better state, which is what keeps the
        # retry count at one or two instead of exhausting the budget.
        res = follow(
            self.final_ray.reversed(), pose,
            gains=self.s.docking, speeds=self.s.speeds, robot=self.s.robot,
            reverse=True, speed_cap=self.s.speeds.approach,
            stop_at=-self.retreat_target, arrive_tol=0.005,
        )
        ex, ey, eyaw = self.coil_errors(pose)
        return self._status(
            RETREAT, res.v, res.w,
            message=f"retreating to {-self.retreat_target * 100:.1f} cm (at {-proj.s * 100:.1f} cm)",
            lateral=proj.lateral, heading_error=res.heading_error, errors=(ex, ey, eyaw),
        )

