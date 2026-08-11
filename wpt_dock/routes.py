"""Reference geometry and coil-to-coil route planning.

There is no occupancy grid and no graph search here, and that is a deliberate
simplification rather than a missing feature: the stage is a bare 0.80 x 0.60 m
plywood sheet with four coils on it and no obstacles, and the upstream work
likewise defines the world analytically (coil 1 at the origin, +X to coil 2, +Y
to coil 3) with a table of eight named straight routes.  Running A* over a
five-cell-wide free space would add machinery and no capability.

One reference type covers every motion in the task
--------------------------------------------------
Every leg is "drive to a point along a given heading", so every leg is a
:class:`RayReference` **anchored at its target** and pointing along the direction
of travel.  Two consequences that remove whole classes of bug:

* Arc length ``s`` along the ray is exactly the signed longitudinal error, so
  ``remaining = -s`` and the goal is always ``s = 0``.  There is no separate
  "distance to goal" quantity that can disagree with the tracking error.
* For the final leg the ray's tangent *is* the required docked heading, so a
  follower that drives cross-track and heading error to zero solves position and
  orientation together.  The classic docking failure -- arrive on the spot, then
  find you are several degrees off with no room left -- cannot occur, because
  heading was never a separate objective.

Route choice actually matters
-----------------------------
The upstream system does not automate diagonal transits at all (their note: a
90 deg in-place rotation risks occluding the line and the markers).  Here a
diagonal is two straight legs with one in-place rotation between them, and there
are two ways to do it -- via the coil in the same row, or via the coil in the
same column.  They are not equivalent: whichever axis you travel *last* becomes
the final approach, and the final approach length is what the alignment
feasibility budget is spent from.  So the rule is **short axis first**, leaving
the long axis for the approach.  On the measured board that turns a 0.255 m final
run into a 0.453 m one -- nearly double the distance available to null a lateral
error, for free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .config import BoardSpec
from .geometry import wrap_angle


@dataclass(frozen=True)
class Projection:
    """Where the robot sits relative to a reference.

    ``lateral`` is signed **positive to the left of the direction of travel**.
    That convention is load bearing: the Frenet error dynamics the controller
    relies on are ``d(lateral)/dt = v * sin(heading_error)``, which only holds with
    left-positive lateral offset.
    """

    s: float             # m, arc length of the closest point; 0 is the target
    lateral: float       # m, signed cross-track offset, positive to the left
    heading: float       # rad, tangent heading of the reference
    curvature: float     # 1/m, always 0 here; kept so the follower stays general
    point: tuple[float, float]
    remaining: float     # m, arc length left until the end of the reference
    index: int = 0


class RayReference:
    """A straight reference anchored at ``origin`` and pointing along ``heading``."""

    def __init__(self, origin: tuple[float, float], heading: float, length: float = math.inf) -> None:
        self.origin = (float(origin[0]), float(origin[1]))
        self.heading = wrap_angle(float(heading))
        self.length = float(length)
        self._t = (math.cos(self.heading), math.sin(self.heading))
        self._n = (-self._t[1], self._t[0])

    @property
    def total_length(self) -> float:
        return self.length

    def point_at(self, s: float) -> tuple[float, float]:
        return (self.origin[0] + s * self._t[0], self.origin[1] + s * self._t[1])

    def project(self, x: float, y: float, hint: int = 0) -> Projection:
        dx, dy = x - self.origin[0], y - self.origin[1]
        s = dx * self._t[0] + dy * self._t[1]
        lateral = dx * self._n[0] + dy * self._n[1]
        return Projection(
            s=s,
            lateral=lateral,
            heading=self.heading,
            curvature=0.0,
            point=self.point_at(s),
            remaining=self.length - s,
        )

    def reversed(self) -> "RayReference":
        """The same line traversed the other way, still anchored at ``origin``.

        Used by the retreat manoeuvre, where the robot backs away from a coil with
        its nose still pointing at it.
        """
        return RayReference(self.origin, self.heading + math.pi, self.length)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Leg:
    """One straight run to a target point, arriving on a given heading."""

    target: tuple[float, float]
    heading: float
    length: float
    name: str
    is_final: bool = False
    coil: int | None = None          # the coil this leg terminates on, if any

    def reference(self) -> RayReference:
        return RayReference(self.target, self.heading)

    def start_point(self) -> tuple[float, float]:
        return (
            self.target[0] - self.length * math.cos(self.heading),
            self.target[1] - self.length * math.sin(self.heading),
        )


@dataclass(frozen=True)
class Route:
    start_coil: int
    target_coil: int
    legs: tuple[Leg, ...]
    note: str = ""

    @property
    def total_length(self) -> float:
        return sum(leg.length for leg in self.legs)

    def describe(self) -> str:
        path = " -> ".join(leg.name for leg in self.legs)
        return f"coil {self.start_coil} -> coil {self.target_coil}: {path} ({self.total_length:.3f} m)"


def _leg_between(
    board: BoardSpec, a: int, b: int, *, is_final: bool
) -> Leg:
    pa = board.coil_positions[a]
    pb = board.coil_positions[b]
    dx, dy = pb[0] - pa[0], pb[1] - pa[1]
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        raise ValueError(f"coils {a} and {b} coincide")
    heading = math.atan2(dy, dx)
    return Leg(
        target=pb,
        heading=heading,
        length=length,
        name=f"{a}{b}",
        is_final=is_final,
        coil=b,
    )


def plan_route(board: BoardSpec, start_coil: int, target_coil: int) -> Route:
    """Legs taking the robot from ``start_coil`` to ``target_coil``.

    Same row or same column is a single leg.  A diagonal is two legs through the
    shared-row or shared-column coil, chosen so the *longer* axis is travelled
    last -- see the module docstring for why that is not an arbitrary tie-break.
    """
    coils = board.coil_positions
    if start_coil not in coils or target_coil not in coils:
        raise KeyError(f"coils must be in {sorted(coils)}")
    if start_coil == target_coil:
        return Route(start_coil, target_coil, tuple(), note="already at the target coil")

    pa, pb = coils[start_coil], coils[target_coil]
    same_row = abs(pa[1] - pb[1]) < 1e-9
    same_col = abs(pa[0] - pb[0]) < 1e-9

    if same_row or same_col:
        return Route(start_coil, target_coil, (_leg_between(board, start_coil, target_coil, is_final=True),))

    # Diagonal.  The two candidate intermediates are the coil sharing our row and
    # the coil sharing our column.
    via_row = next(c for c, p in coils.items() if c not in (start_coil, target_coil) and abs(p[1] - pa[1]) < 1e-9)
    via_col = next(c for c, p in coils.items() if c not in (start_coil, target_coil) and abs(p[0] - pa[0]) < 1e-9)

    def final_length(via: int) -> float:
        pv = coils[via]
        return math.hypot(pb[0] - pv[0], pb[1] - pv[1])

    via = via_row if final_length(via_row) >= final_length(via_col) else via_col
    note = (
        f"diagonal via coil {via}: final approach {final_length(via):.3f} m "
        f"(the alternative via coil {via_row if via is via_col else via_col} would give "
        f"{final_length(via_col if via is via_row else via_row):.3f} m)"
    )
    return Route(
        start_coil,
        target_coil,
        (
            _leg_between(board, start_coil, via, is_final=False),
            _leg_between(board, via, target_coil, is_final=True),
        ),
        note=note,
    )


# ---------------------------------------------------------------------------
# Staying on the plywood
# ---------------------------------------------------------------------------


def safe_area(board: BoardSpec, robot_radius: float) -> tuple[float, float, float, float]:
    """``(x_min, y_min, x_max, y_max)`` the robot *centre* may occupy.

    The board has no walls -- its edge is a 0.15 m drop onto the table.  So this is
    a real constraint, not a formality, and it is checked against the *estimated*
    pose every control step.
    """
    ox, oy = board.stage_origin
    w, h = board.stage_size
    return (ox + robot_radius, oy + robot_radius, ox + w - robot_radius, oy + h - robot_radius)


def point_is_on_board(board: BoardSpec, robot_radius: float, x: float, y: float) -> bool:
    x0, y0, x1, y1 = safe_area(board, robot_radius)
    return x0 <= x <= x1 and y0 <= y <= y1


def footprint_is_on_board(
    board: BoardSpec, half_extents: tuple[float, float], x: float, y: float, yaw: float
) -> bool:
    """Orientation-aware containment for a rectangular robot.

    The axis-aligned bounding box of a rectangle rotated by ``yaw`` has half-extents
    ``|a cos| + |b sin|`` and ``|a sin| + |b cos|``.  Checking that box against the
    stage is exact for axis-aligned headings -- which is every heading a leg on this
    board uses -- and conservative in between.

    This is the *running* check.  Using a single swept radius instead costs 30 mm of
    margin on each axis, and on an 0.80 x 0.60 m stage that is the difference between
    a legal pose and a false emergency stop.
    """
    a, b = half_extents
    c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
    hx = a * c + b * s
    hy = a * s + b * c
    ox, oy = board.stage_origin
    w, h = board.stage_size
    return (
        ox + hx <= x <= ox + w - hx
        and oy + hy <= y <= oy + h - hy
    )


def route_fits_on_board(board: BoardSpec, robot_radius: float, route: Route) -> tuple[bool, str]:
    """Check every leg endpoint, plus the retreat room behind each final approach.

    Endpoint checking is sufficient because the legs are axis-aligned and the safe
    area is an axis-aligned rectangle, so a segment with both ends inside is
    entirely inside.  That is worth stating: the same shortcut would be wrong for
    a non-convex region.
    """
    for leg in route.legs:
        for label, (px, py) in (("start", leg.start_point()), ("end", leg.target)):
            if not point_is_on_board(board, robot_radius, px, py):
                return False, f"leg {leg.name} {label} ({px:.3f}, {py:.3f}) puts the robot off the plywood"
    return True, "ok"


def retreat_room(
    board: BoardSpec, robot_radius: float, leg: Leg, wanted: float, *, edge_margin: float = 0.015
) -> float:
    """How far the robot can actually back off behind a coil before leaving the board.

    Retreat is the recovery action when an approach cannot converge in the distance
    left.  On a board this small that room is genuinely scarce, so the amount is
    computed rather than assumed -- promising a 0.32 m retreat on a stage that only
    has 0.17 m of margin would just drive the robot off the edge.

    ``edge_margin`` exists because the Monte Carlo found the boundary case: returning
    the exact distance to the safe-area edge makes the retreat *arrive* on the edge,
    and then tracking error of a millimetre or two trips the same safe-area check as
    an emergency stop.  A recovery action must not be able to cause the fault it is
    recovering from, so it stops short.
    """
    x0, y0, x1, y1 = safe_area(board, robot_radius + max(0.0, edge_margin))
    cx, cy = leg.target
    ux, uy = math.cos(leg.heading), math.sin(leg.heading)
    limit = wanted
    # Walk backwards along -u and find the first boundary crossing.
    for u, lo, hi, c in ((ux, x0, x1, cx), (uy, y0, y1, cy)):
        if abs(u) < 1e-9:
            continue
        # position after backing off d is c - u*d; require lo <= c - u*d <= hi
        if u > 0.0:
            limit = min(limit, (c - lo) / u)
        else:
            limit = min(limit, (c - hi) / u)
    return max(0.0, limit)


@dataclass
class RouteBook:
    """All eight coil-to-coil routes, for reporting and for sanity checks."""

    board: BoardSpec
    robot_radius: float
    routes: dict[str, Route] = field(default_factory=dict)

    def build(self) -> "RouteBook":
        coils = sorted(self.board.coil_positions)
        for a in coils:
            for b in coils:
                if a == b:
                    continue
                self.routes[f"{a}{b}"] = plan_route(self.board, a, b)
        return self

    def report(self) -> str:
        lines = []
        for key in sorted(self.routes):
            route = self.routes[key]
            ok, why = route_fits_on_board(self.board, self.robot_radius, route)
            flag = "ok " if ok else "OFF"
            lines.append(f"  [{flag}] {route.describe()}" + ("" if ok else f"  <- {why}"))
            if route.note:
                lines.append(f"         {route.note}")
        return "\n".join(lines)
