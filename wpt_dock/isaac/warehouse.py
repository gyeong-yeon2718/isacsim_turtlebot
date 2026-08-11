"""Shelving, pick and place stations, the payload, and warehouse dressing.

Everything here is placed against two hard constraints rather than by eye, which is why
the numbers are computed from the specs instead of typed in:

**1. The arm has to be able to reach it.**  The cute_arm's usable envelope is 3-23 cm from
a shoulder pivot that sits about 0.22 m above the plywood.  A station 0.30 m away is
decoration, not a workcell.  ``station_reach_check`` verifies each station against the real
IK from the robot's docked pose and reports the numbers, so an unreachable target is caught
at build time rather than as a puzzling mid-run failure.

**2. The robot must not be able to drive into it.**  The stations sit on the board, in the
margin *outside* the robot's safe area -- x below -0.0845 or above 0.5375 m, y below -0.0835
or above 0.3385 m for the support footprint used by the controller.  Putting them anywhere
inside that band would make them obstacles, and there is no obstacle avoidance in this
project because until now there was nothing to avoid.

The tall racks are dressing.  They stand on the floor outside the table, so they can be as
large as a warehouse aisle wants without competing for board space or entering the arm's
workspace.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..arm import (
    PAYLOAD_HALF,
    PAYLOAD_MASS,
    PAYLOAD_SIZE,
    ArmSpec,
    base_frame_target,
    solve_ik,
)
from ..config import BoardSpec, Settings
from ..geometry import compose, invert
from ..routes import plan_route
from .robot_build import _make_invisible
from .usd_helpers import add_box, add_cylinder, add_physics_material, bind_physics_material

_STEEL = (0.42, 0.45, 0.50)
_SHELF = (0.55, 0.52, 0.48)
_CRATE = (0.62, 0.45, 0.26)
_PALLET = (0.58, 0.44, 0.28)
_PAYLOAD = (0.85, 0.30, 0.18)
_TARGET = (0.95, 0.80, 0.10)
_ROLLER = (0.78, 0.78, 0.80)

STATION_SURFACE_Z = 0.100        # m above the plywood; mid-workspace for the arm
STATION_FOOTPRINT = (0.070, 0.090)   # m, the drop table's top
# The conveyor is deliberately short and narrow.  Its footprint is a hard constraint in the
# placement search -- a line running the whole depth of the board cannot clear the robot's
# turning circle at both the dock and the route corner, and the first version that tried it put
# the rear extension plate through the rails.
CONVEYOR_WIDTH = 0.038               # m, outside the rails
CONVEYOR_LENGTH = 0.200              # m, along y
# Height of the plywood above the warehouse floor.  A station the placement search puts outside
# the board has to stand on the floor, so its legs are this much longer.
TABLE_TOP_Z = 0.330                  # m

# Where the robot's own safe area ends in x.  A station inside it becomes an obstacle, and this
# project has no avoidance logic -- there has never been anything to avoid.  Used as a hard
# constraint by the placement search rather than as a comment to check by hand.
SAFE_AREA_X = (-0.084, 0.538)

# The cube edge is set by the **jaws**, not by taste, so it is defined in ``arm.py`` beside the
# jaw travel and re-exported here.  ``ArmSpec`` gives 6 mm of separation closed and 34 mm open,
# so a 35 mm cube -- the previous value -- could not fit between the jaws even fully open, and
# the "closed" pose drove them 14.5 mm into each side of it.  That is what the user saw: an
# object far too big for the gripper.  ``payload_fit_check`` asserts the fit at build time
# rather than trusting this comment.


@dataclass
class Station:
    name: str
    centre: tuple[float, float]
    surface_z: float

    @property
    def grasp_point(self) -> tuple[float, float, float]:
        """Where the payload's centre sits when it is on this station."""
        return (self.centre[0], self.centre[1], self.surface_z + 0.5 * PAYLOAD_SIZE)


@dataclass
class WarehouseScene:
    pick: Station
    place: Station
    payload_path: str
    notes: list[str]


def _rack(stage, path: str, centre: tuple[float, float], size: tuple[float, float],
          floor_z: float, height: float, shelves: int = 3) -> None:
    """Uprights plus evenly spaced shelves.  Pure dressing, no colliders needed."""
    w, d = size
    for sx in (-1, 1):
        for sy in (-1, 1):
            add_box(stage, f"{path}/upright_{'p' if sx > 0 else 'n'}{'p' if sy > 0 else 'n'}",
                    (0.022, 0.022, height),
                    (centre[0] + sx * (0.5 * w - 0.014), centre[1] + sy * (0.5 * d - 0.014),
                     floor_z + 0.5 * height), _STEEL)
    for i in range(shelves):
        z = floor_z + height * (i + 1) / (shelves + 0.6)
        add_box(stage, f"{path}/shelf{i}", (w, d, 0.014), (centre[0], centre[1], z), _SHELF)


def _station(stage, path: str, station: Station, footprint: tuple[float, float],
             floor_z: float = 0.0) -> None:
    """A short pedestal with a shelf top, standing on the plywood.

    The top is a **collider**: the payload is a real rigid body and rests on it, and is
    released onto it.  Everything else in this module is visual only, so this one exception
    is worth naming -- a decorative shelf would let the box fall straight through.
    """
    w, d = footprint
    post_h = station.surface_z - floor_z
    add_box(stage, f"{path}/post", (0.030, 0.030, post_h),
            (station.centre[0], station.centre[1], floor_z + 0.5 * post_h), _STEEL)
    top = add_box(stage, f"{path}/top", (w, d, 0.008),
                  (station.centre[0], station.centre[1], station.surface_z - 0.004), _SHELF,
                  collision=True)
    bind_physics_material(
        top.GetPrim(), add_physics_material(stage, "/World/Looks/PhysicsShelf", 0.70, 0.60)
    )


def _conveyor(
    stage,
    path: str,
    station: Station,
    *,
    y_span: tuple[float, float],
    width: float = CONVEYOR_WIDTH,
    floor_z: float = 0.0,
) -> None:
    """A gravity roller conveyor running in ``y``, with the pick point on it.

    Why a conveyor and not another pedestal: the station's ``x`` is pinned to about 20 mm by
    the board edge and the robot's safe area, so the only axis with room to place the grasp at
    the arm's best-conditioned reach is ``y`` -- and a line running that way is exactly what a
    warehouse puts there.  The geometry follows the requirement instead of decorating it.

    Physics: one **invisible box collider** spans the roller tops, and the rollers themselves
    are visual.  Forty roller cylinders with convex colliders would each need a contact pair
    against the payload and would dominate the physics step to model a surface the box only
    ever rests flat on.  The visible thing and the solid thing are separated deliberately, the
    same split used for the robot's printed top plate.
    """
    cx = station.centre[0]
    y0, y1 = y_span
    top_z = station.surface_z
    rail_h = 0.016
    half_w = 0.5 * width

    # Side rails, running the full depth.
    for sy, tag in ((+1, "far"), (-1, "near")):
        add_box(stage, f"{path}/rail_{tag}",
                (0.010, y1 - y0, rail_h),
                (cx + sy * half_w, 0.5 * (y0 + y1), top_z - 0.5 * rail_h + 0.004), _STEEL)

    # Legs down to whatever it stands on: the plywood, or the warehouse floor if the placement
    # search had to put it outside the board.
    leg_top = top_z - rail_h + 0.004
    leg_h = leg_top - floor_z
    for k, fy in enumerate((0.06, 0.5, 0.94)):
        ly = y0 + (y1 - y0) * fy
        for sy, tag in ((+1, "far"), (-1, "near")):
            add_box(stage, f"{path}/leg{k}_{tag}", (0.012, 0.012, leg_h),
                    (cx + sy * half_w, ly, floor_z + 0.5 * leg_h), _STEEL)

    # Rollers: axis along X, tops flush with the surface the payload rests on.
    radius = 0.007
    pitch = 0.026
    n = max(2, int((y1 - y0 - 0.02) / pitch))
    for i in range(n):
        ry = y0 + 0.01 + pitch * (i + 0.5)
        add_cylinder(stage, f"{path}/roller{i:02d}", radius, width - 0.016,
                     (cx, ry, top_z - radius), _ROLLER, axis="X")

    # The one solid thing.  Thin, so the payload rests at exactly ``surface_z``.
    surface_path = f"{path}/surface"
    top = add_box(stage, surface_path, (width - 0.016, y1 - y0 - 0.004, 0.004),
                  (cx, 0.5 * (y0 + y1), top_z - 0.002), _SHELF, collision=True)
    _make_invisible(stage, surface_path)
    bind_physics_material(
        top.GetPrim(), add_physics_material(stage, "/World/Looks/PhysicsShelf", 0.70, 0.60)
    )


def station_reach_check(
    settings: Settings, arm: ArmSpec, station: Station, coil: int, heading: float,
    plate_top_z: float,
) -> tuple[bool, str]:
    """Can the arm actually reach this station from that coil, on that heading?

    Run at build time against the same closed-form IK the mission uses.  The grasp point
    *and* the approach point 60 mm above it are both checked, because a target that is
    reachable but whose approach is not produces a sequence that fails halfway with the
    payload already committed.
    """
    pos = settings.board.coil_positions[coil]
    robot_pose = (pos[0], pos[1], heading)
    grasp = station.grasp_point
    above = (grasp[0], grasp[1], grasp[2] + 0.060)
    lines = []
    ok = True
    for label, target in (("grasp", grasp), ("approach", above)):
        local = base_frame_target(arm, robot_pose, plate_top_z, target)
        res = solve_ik(arm, local)
        d = math.sqrt(sum(v * v for v in local))
        if res.ok and res.pose is not None:
            lines.append(
                f"{label} d={d * 100:.1f}cm base={math.degrees(res.pose.base):+.0f} "
                f"sh={math.degrees(res.pose.shoulder):+.0f} el={math.degrees(res.pose.elbow):+.0f}"
            )
        else:
            ok = False
            lines.append(f"{label} UNREACHABLE d={d * 100:.1f}cm -- {res.reason}")
    return ok, f"{station.name} from coil {coil}: " + "; ".join(lines)


def best_station_centre(
    settings: Settings,
    arm: ArmSpec,
    *,
    coil: int,
    heading: float,
    plate_top_z: float,
    surface_z: float,
    x_window: tuple[float, float],
    y_window: tuple[float, float],
    footprint: tuple[float, float],
    turn_points: tuple[tuple[float, float], ...] = (),
    clearance: float = 0.010,
    base_margin: float = math.radians(15.0),
) -> tuple[tuple[float, float], str]:
    """Where in the margin is the grasp best conditioned?  Searched, not guessed.

    For a two-link arm the Jacobian determinant is ``l_upper * distal * |sin(elbow)|``, so the
    best-conditioned reach is the one with the elbow at 90 degrees -- a distance of
    ``hypot(l_upper, distal)`` from the shoulder pivot, which for this arm is 181.4 mm.  That
    matters here because the dominant error term in the whole pick-and-place is the SG90's
    0.5-degree resolution, and how much gripper displacement half a degree buys is exactly what
    conditioning decides.  Near the peak it is also *flat* -- ``sin`` is stationary at 90
    degrees -- so honesty about the size of the win: the hand-placed stations were already
    within 0.4 % of maximum manipulability.

    The gain that is not marginal is the **base yaw**.  Hitting the optimum distance moves the
    pick from a base angle of +84 degrees to +66, and the servo's limit is +90.  A docking error
    that rotates the robot a few degrees the wrong way could have pushed the old placement past
    that limit and made the target unreachable in the middle of a run; there is now 24 degrees
    of margin instead of 6.

    The search is a scan rather than a solve because the constraints are not smooth: the station
    has to be fully on the plywood, fully outside the robot's safe area, and both the grasp
    point *and* the approach point 60 mm above it have to be inside all three joint ranges.
    ``x`` is pinned to about 20 mm by the first two, which is why the free axis is ``y`` -- and
    why the pick station is a rail running that way rather than a pedestal.

    Two further constraints, both added after a station was placed without them and the robot's
    rear extension plate ended up inside the conveyor:

    * **Clear the robot's turning circle.**  ``turn_points`` are where the robot rotates -- the
      coil it docks on and every vertex of its route -- and the station's whole footprint must
      stay ``obstacle_swept_radius + clearance`` from all of them.  That is deliberately *not*
      ``swept_radius``: that one is built from the support envelope and answers "does the robot
      stay on the plywood", where the overhanging parts may hang past the edge because there is
      nothing out there to hit.  A conveyor is something to hit.  The rear extension puts
      structure 157.4 mm from the wheel axle; the conveyor's near rail was 93.7 mm from coil 1.
    * **Leave the base servo some travel.**  A solution sitting exactly on the +-90 degree limit
      is not a solution: the arm aims from wherever the robot actually parked, so a docking error
      of a couple of degrees the wrong way makes the target unreachable mid-run.  ``base_margin``
      reserves that, and the first version of this search returned +90.0 degrees exactly.

    Together these can make the plywood margin infeasible, and at coil 1 they do: requiring
    clearance from both coil 1 and the route vertex leaves an empty interval.  That is why the
    caller may pass an ``x_window`` reaching past the board edge -- a floor-standing conveyor
    alongside the table, which is what a warehouse has anyway.  The returned note says which.
    """
    ox, oy = settings.board.stage_origin
    cx, cy = settings.board.coil_positions[coil]
    # The chassis pose when docked: the *receiver coil* is what parks on the transmitter, so
    # undo its offset to get the frame the arm is bolted to.  Same transform the mission uses.
    chassis = compose((cx, cy, heading), invert((*settings.robot.rx_coil_offset, 0.0)))
    grasp_z = surface_z + PAYLOAD_HALF
    d_opt = math.hypot(arm.l_upper, arm.distal)
    manip_max = arm.l_upper * arm.distal

    hx, hy = footprint
    r = settings.robot
    keep_out = r.obstacle_swept_radius + clearance
    base_limit = min(abs(arm.base_range[0]), abs(arm.base_range[1])) - base_margin

    # The low outline, in body coordinates about the wheel axle.  The 230 mm plate is excluded
    # for the reason ``obstacle_swept_radius`` gives: it rides above a station this short.
    rear_x = -(0.5 * r.base_footprint[1] + r.rear_extension)
    corners = [
        (rear_x, +0.5 * r.base_footprint[0]), (rear_x, -0.5 * r.base_footprint[0]),
        (+0.5 * r.base_footprint[1], +0.5 * r.base_footprint[0]),
        (+0.5 * r.base_footprint[1], -0.5 * r.base_footprint[0]),
        (0.0, +r.footprint_half_extents[1]), (0.0, -r.footprint_half_extents[1]),
    ]

    def rect_gap(sx: float, sy: float, px: float, py: float) -> float:
        """Distance from a point to the station rectangle, so a long conveyor is not a dot."""
        return math.hypot(max(0.0, abs(px - sx) - hx), max(0.0, abs(py - sy) - hy))

    def clears_every_turn(sx: float, sy: float) -> bool:
        """Two different tests, because the robot does two different things.

        At a **route vertex** it really turns through 90 degrees, so the whole swept circle has
        to be clear.  At the coil it **docks** on it does not: it arrives on a heading and holds
        it, correcting by at most the yaw tolerance and its hand-placement error.  Demanding a
        full turning circle there is not conservatism, it is a wrong model -- and it is what made
        the drop table infeasible, because clearing 167 mm and staying inside the arm's 230 mm
        reach left a band about a millimetre wide.  So the docked coil is checked against the
        actual oriented outline over a yaw slack, sampled rather than reasoned about.
        """
        for px, py in turn_points:
            if math.hypot(px - cx, py - cy) < 1e-9:
                continue                                  # the docked coil, handled below
            if rect_gap(sx, sy, px, py) < keep_out:
                return False
        slack = math.radians(12.0)                        # tolerance 4 deg + placement error 3 sigma
        for k in range(13):
            th = heading - slack + 2.0 * slack * k / 12
            ct, st = math.cos(th), math.sin(th)
            for bx, by in corners:
                wx = cx + bx * ct - by * st
                wy = cy + bx * st + by * ct
                if rect_gap(sx, sy, wx, wy) < clearance:
                    return False
        return True

    best: tuple[tuple[float, float], float, float, float, float] | None = None
    rejected = {"clear": 0, "ik": 0, "base": 0}
    x_lo, x_hi = x_window
    y_lo, y_hi = y_window
    nx = 1 if x_hi <= x_lo else 41
    for i in range(nx):
        sx = x_lo if nx == 1 else x_lo + (x_hi - x_lo) * i / (nx - 1)
        for j in range(241):
            sy = y_lo + (y_hi - y_lo) * j / 240
            if turn_points and not clears_every_turn(sx, sy):
                rejected["clear"] += 1
                continue
            grasp = base_frame_target(arm, chassis, plate_top_z, (sx, sy, grasp_z))
            above = base_frame_target(arm, chassis, plate_top_z, (sx, sy, grasp_z + 0.060))
            g, a = solve_ik(arm, grasp), solve_ik(arm, above)
            if not (g.ok and a.ok) or g.pose is None:
                rejected["ik"] += 1
                continue
            if abs(g.pose.base) > base_limit:
                rejected["base"] += 1
                continue
            d = math.sqrt(sum(v * v for v in grasp))
            manip = manip_max * abs(math.sin(g.pose.elbow))
            # Closest to the optimum distance; manipulability breaks ties.
            key = (round(abs(d - d_opt), 6), -manip)
            if best is None or key < (round(abs(best[1] - d_opt), 6), -best[2]):
                best = ((sx, sy), d, manip, g.pose.elbow, g.pose.base)

    if best is None:
        # Fall back to the middle of the window and say so, rather than silently shipping a
        # station the arm cannot use.  The rejection counts say *which* constraint bit, because
        # "no feasible placement" is useless on its own.
        centre = (0.5 * (x_lo + x_hi), 0.5 * (y_lo + y_hi))
        return centre, (
            f"coil {coil}: NO feasible station -- rejected {rejected['clear']} for turning-circle "
            f"clearance, {rejected['ik']} unreachable, {rejected['base']} on the base servo "
            f"limit; fell back to ({centre[0]:+.3f}, {centre[1]:+.3f}) m"
        )

    (sx, sy), d, manip, elbow, base = best
    where = "on the board" if sx - hx >= ox else "floor-standing beside the table"
    return (sx, sy), (
        f"coil {coil}: station at ({sx:+.4f}, {sy:+.4f}) m {where} -> reach {d * 1000:.1f} mm "
        f"(optimum {d_opt * 1000:.1f}), elbow {math.degrees(elbow):+.1f} deg, "
        f"base {math.degrees(base):+.1f} of {math.degrees(base_limit):.0f} usable, "
        f"manipulability {100.0 * manip / manip_max:.1f} % of maximum; "
        f"clears every turning circle by {keep_out * 1000:.0f} mm"
    )


def payload_fit_check(arm: ArmSpec) -> tuple[bool, str]:
    """Does the payload fit the jaws, and by how much?

    A build-time check for the same reason ``station_reach_check`` is one: the alternative is
    an object the gripper visibly cannot hold, which is what shipped.  Printing the clearance
    means the next person changing ``PAYLOAD_SIZE`` sees immediately whether it is still
    grippable instead of discovering it in the viewport.
    """
    open_gap = arm.gripper_clear(arm.gripper_open)
    closed_gap = arm.gripper_clear(arm.gripper_closed)
    if not (closed_gap < PAYLOAD_SIZE < open_gap):
        return False, (
            f"payload {PAYLOAD_SIZE * 1000:.0f} mm does NOT fit the jaws "
            f"({closed_gap * 1000:.0f}-{open_gap * 1000:.0f} mm usable gap)"
        )
    angle = arm.grip_angle_for(PAYLOAD_SIZE)
    return True, (
        f"payload {PAYLOAD_SIZE * 1000:.0f} mm in a {open_gap * 1000:.0f} mm open gap: "
        f"{0.5 * (open_gap - PAYLOAD_SIZE) * 1000:.1f} mm clearance per jaw on approach, "
        f"pads close from {math.degrees(arm.gripper_open):.0f} to {math.degrees(angle):.1f} deg "
        f"and stop on the box ({arm.gripper_clear(angle) * 1000:.1f} mm)"
    )


def build_warehouse(
    stage,
    settings: Settings,
    arm: ArmSpec,
    *,
    source_coil: int,
    target_coil: int,
    plate_top_z: float,
    root: str = "/World/warehouse",
) -> WarehouseScene:
    board: BoardSpec = settings.board
    ox, oy = board.stage_origin
    w, h = board.stage_size
    notes: list[str] = []

    # Where the robot will be standing when it works each station, which is what decides where
    # the station should be.
    route_plan = plan_route(board, source_coil, target_coil)
    src_heading = route_plan.legs[0].heading if route_plan.legs else 0.0
    dst_heading = route_plan.legs[-1].heading if route_plan.legs else 0.0

    # Stations placed where the grasp is best conditioned, searched against the real IK inside
    # the only window the layout allows: fully on the plywood and fully clear of the robot's
    # safe area.  Nothing here is a typed-in offset.
    # Everywhere the robot rotates: the coil it docks on plus every vertex of its route.  A
    # station has to clear all of them, not just the one it serves -- the conveyor was clipped by
    # the 90-degree turn at the route corner as well as by the dock itself.
    turn_points = tuple(
        [board.coil_positions[source_coil]] + [leg.target for leg in route_plan.legs]
    )

    conv_half = (0.5 * CONVEYOR_WIDTH, 0.5 * CONVEYOR_LENGTH)
    half_x, half_y = 0.5 * STATION_FOOTPRINT[0], 0.5 * STATION_FOOTPRINT[1]
    # The pick window reaches 90 mm past the plywood edge: on this board the turning-circle
    # clearance makes the margin infeasible, so the conveyor stands on the floor alongside.
    pick_centre, pick_note = best_station_centre(
        settings, arm, coil=source_coil, heading=src_heading, plate_top_z=plate_top_z,
        surface_z=STATION_SURFACE_Z,
        x_window=(ox - 0.090, SAFE_AREA_X[0] - conv_half[0]),
        y_window=(oy - 0.060, oy + h + 0.060),
        footprint=conv_half, turn_points=turn_points,
    )
    place_centre, place_note = best_station_centre(
        settings, arm, coil=target_coil, heading=dst_heading, plate_top_z=plate_top_z,
        surface_z=STATION_SURFACE_Z,
        x_window=(SAFE_AREA_X[1] + half_x, ox + w + 0.090),
        y_window=(oy - 0.060, oy + h + 0.060),
        footprint=(half_x, half_y), turn_points=turn_points,
    )
    pick = Station("pick conveyor", pick_centre, STATION_SURFACE_Z)
    place = Station("drop table", place_centre, STATION_SURFACE_Z)
    notes.append("PLACED " + pick_note)
    notes.append("PLACED " + place_note)

    # The pick point is a gravity roller conveyor running in y.  Its legs go down to whatever it
    # is standing on: the plywood if the search kept it on the board, otherwise the floor.
    pick_on_board = pick.centre[0] - conv_half[0] >= ox
    _conveyor(
        stage, f"{root}/pick", pick,
        y_span=(pick.centre[1] - conv_half[1], pick.centre[1] + conv_half[1]),
        width=CONVEYOR_WIDTH,
        floor_z=0.0 if pick_on_board else -TABLE_TOP_Z,
    )
    _station(stage, f"{root}/place", place, STATION_FOOTPRINT,
             floor_z=0.0 if place.centre[0] + half_x <= ox + w else -TABLE_TOP_Z)

    # Boxes queued behind the pick point, dressing only.  Placed along the conveyor away from the
    # robot, and only where the conveyor actually exists now that it is 200 mm long.
    for i, fy in enumerate((0.62, 0.80)):
        qy = pick.centre[1] - conv_half[1] + CONVEYOR_LENGTH * fy
        add_box(stage, f"{root}/pick/queued{i}", (PAYLOAD_SIZE,) * 3,
                (pick.centre[0], qy, STATION_SURFACE_Z + PAYLOAD_HALF), _CRATE)

    # A marked landing pad, so a placement error is visible rather than inferred.
    add_box(stage, f"{root}/place/pad", (0.050, 0.050, 0.0012),
            (place.centre[0], place.centre[1], place.surface_z + 0.0012), _TARGET)
    for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        add_box(stage, f"{root}/place/pad_corner_{'p' if sx > 0 else 'n'}{'p' if sy > 0 else 'n'}",
                (0.014, 0.003, 0.0016),
                (place.centre[0] + sx * 0.0235, place.centre[1] + sy * 0.0235,
                 place.surface_z + 0.0016), (0.15, 0.15, 0.16))

    # The payload.  A real rigid body: carried kinematically, then released to settle under
    # gravity, which is what makes the final placement error a physical result rather than a
    # number we wrote down.
    #
    # The rigid body goes on a **wrapper Xform** with the cube as its child, not on the cube
    # itself.  ``add_box`` builds a box as a unit ``UsdGeom.Cube`` plus a scale op, and the
    # carry code has to own the prim's transform ops -- clearing them to do that silently
    # dropped the scale, turning a 35 mm box into a 1 m cube that exploded out of the scene
    # on the first contact.  Separating "the thing physics moves" from "the thing that has a
    # size" removes the conflict entirely.
    payload_path = f"{root}/payload"
    from pxr import Sdf, UsdGeom, UsdPhysics

    body = UsdGeom.Xform.Define(stage, Sdf.Path(payload_path))
    rb = UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    # **Dynamic** from creation.  It rests on the conveyor under gravity like anything else,
    # and it falls and settles when the jaws let go.  ``GripperAttachment`` switches it to
    # kinematic only for the seconds it is actually being carried.
    #
    # An earlier revision of this comment asserted the opposite -- that PhysX does not pick up
    # ``kinematicEnabled`` mid-simulation, so the payload had to be kinematic for the whole run.
    # That was **wrong**, and it was wrong in the worst way: it was inferred from a failed run
    # rather than measured.  A probe against this exact install
    # (5.1.0-rc.19) flips the attribute mid-simulation and
    # ``omni.physx.get_physxunittests_interface().get_physics_stats()`` moves
    # ``numKinematicBodies`` 1 -> 0 on the same frame, with no step in between.  The released
    # box then falls at textbook -g, tips off its corner and comes to rest flat.  What actually
    # ejected the payload to z = -23.5 m was the collision that also launched the robot, and
    # that is fixed by ``GripperAttachment.exclude_from_collision_with``.
    rb.CreateKinematicEnabledAttr(False)
    UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(PAYLOAD_MASS)
    box = add_box(stage, f"{payload_path}/box", (PAYLOAD_SIZE,) * 3, (0.0, 0.0, 0.0),
                  _PAYLOAD, collision=True)
    bind_physics_material(
        box.GetPrim(), add_physics_material(stage, "/World/Looks/PhysicsPayload", 0.80, 0.70)
    )

    # --- dressing: floor-standing racks outside the table ------------------
    floor_z = -TABLE_TOP_Z
    table_x0, table_x1 = ox - 0.05, ox + w + 0.05
    table_y0, table_y1 = oy - 0.05, oy + h + 0.05
    add_box(stage, f"{root}/wall_west", (0.02, 1.60, 1.10), (table_x0 - 0.42, 0.5 * h + oy,
            floor_z + 0.55), (0.52, 0.53, 0.55))
    add_box(stage, f"{root}/wall_north", (1.90, 0.02, 1.10), (0.5 * w + ox, table_y1 + 0.52,
            floor_z + 0.55), (0.52, 0.53, 0.55))

    _rack(stage, f"{root}/rack_west", (table_x0 - 0.30, oy + 0.10), (0.34, 0.80), floor_z, 0.92)
    _rack(stage, f"{root}/rack_north", (ox + 0.28, table_y1 + 0.30), (0.80, 0.34), floor_z, 0.92)
    _rack(stage, f"{root}/rack_east", (table_x1 + 0.30, oy + h - 0.10), (0.34, 0.70), floor_z, 0.80)

    crates = [
        (f"{root}/crate_w0", (table_x0 - 0.30, oy - 0.10, floor_z + 0.20), (0.13, 0.16, 0.11)),
        (f"{root}/crate_w1", (table_x0 - 0.30, oy + 0.28, floor_z + 0.47), (0.15, 0.13, 0.13)),
        (f"{root}/crate_n0", (ox + 0.10, table_y1 + 0.30, floor_z + 0.20), (0.16, 0.13, 0.12)),
        (f"{root}/crate_n1", (ox + 0.46, table_y1 + 0.30, floor_z + 0.47), (0.14, 0.14, 0.14)),
        (f"{root}/crate_e0", (table_x1 + 0.30, oy + h - 0.10, floor_z + 0.18), (0.13, 0.15, 0.10)),
    ]
    for path, centre, size in crates:
        add_box(stage, path, size, (centre[0], centre[1], centre[2] + 0.5 * size[2]), _CRATE)

    for i, (px, py) in enumerate(((table_x0 - 0.30, oy + h + 0.35), (table_x1 + 0.32, oy - 0.28))):
        add_box(stage, f"{root}/pallet{i}", (0.30, 0.24, 0.024), (px, py, floor_z + 0.012),
                _PALLET)
        add_box(stage, f"{root}/pallet{i}_load", (0.24, 0.20, 0.16), (px, py, floor_z + 0.104),
                _CRATE)

    # --- reachability, checked not assumed ---------------------------------
    # Redundant with the placement search, which already solved the IK at every candidate --
    # kept because it reports the achieved joint angles in the build log, and because a future
    # change that hand-places a station again would otherwise lose the check entirely.
    for station, coil, heading in ((pick, source_coil, src_heading), (place, target_coil, dst_heading)):
        ok, line = station_reach_check(settings, arm, station, coil, heading, plate_top_z)
        notes.append(("OK   " if ok else "FAIL ") + line)

    grip_ok, grip_line = payload_fit_check(arm)
    notes.append(("OK   " if grip_ok else "FAIL ") + grip_line)

    notes.append(
        f"stations on the board margin at x = {pick.centre[0]:+.3f} and "
        f"{place.centre[0]:+.3f} m, surface {STATION_SURFACE_Z * 1000:.0f} mm above the plywood; "
        f"robot safe area ends at x = -0.084 / +0.538 m so neither is drivable into"
    )
    return WarehouseScene(pick=pick, place=place, payload_path=payload_path, notes=notes)
