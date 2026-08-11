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

from ..arm import ArmSpec, base_frame_target, solve_ik
from ..config import BoardSpec, Settings
from .usd_helpers import add_box, add_cylinder, add_physics_material, bind_physics_material

_STEEL = (0.42, 0.45, 0.50)
_SHELF = (0.55, 0.52, 0.48)
_CRATE = (0.62, 0.45, 0.26)
_PALLET = (0.58, 0.44, 0.28)
_PAYLOAD = (0.85, 0.30, 0.18)
_TARGET = (0.95, 0.80, 0.10)

STATION_SURFACE_Z = 0.100        # m above the plywood; mid-workspace for the arm
PAYLOAD_SIZE = 0.035             # m, cube edge
PAYLOAD_MASS = 0.040             # kg, a light printed/cardboard box an SG90 could lift


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


def _station(stage, path: str, station: Station, footprint: tuple[float, float]) -> None:
    """A short pedestal with a shelf top, standing on the plywood.

    The top is a **collider**: the payload is a real rigid body and rests on it, and is
    released onto it.  Everything else in this module is visual only, so this one exception
    is worth naming -- a decorative shelf would let the box fall straight through.
    """
    w, d = footprint
    add_box(stage, f"{path}/post", (0.030, 0.030, station.surface_z),
            (station.centre[0], station.centre[1], 0.5 * station.surface_z), _STEEL)
    top = add_box(stage, f"{path}/top", (w, d, 0.008),
                  (station.centre[0], station.centre[1], station.surface_z - 0.004), _SHELF,
                  collision=True)
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

    # Stations in the board margin outside the robot's safe area.  Offsets are derived from
    # the board edge so they move with the board rather than being magic numbers.
    pick = Station("pick shelf", (ox + 0.038, board.coil_positions[source_coil][1]),
                   STATION_SURFACE_Z)
    place = Station("drop shelf", (ox + w - 0.038, board.coil_positions[target_coil][1]),
                    STATION_SURFACE_Z)

    _station(stage, f"{root}/pick", pick, (0.070, 0.090))
    _station(stage, f"{root}/place", place, (0.070, 0.090))

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
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(PAYLOAD_MASS)
    box = add_box(stage, f"{payload_path}/box", (PAYLOAD_SIZE,) * 3, (0.0, 0.0, 0.0),
                  _PAYLOAD, collision=True)
    bind_physics_material(
        box.GetPrim(), add_physics_material(stage, "/World/Looks/PhysicsPayload", 0.80, 0.70)
    )

    # --- dressing: floor-standing racks outside the table ------------------
    floor_z = -0.330
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
    from ..routes import plan_route

    source_heading = plan_route(board, source_coil, target_coil)
    src_heading = source_heading.legs[0].heading if source_heading.legs else 0.0
    dst_heading = source_heading.legs[-1].heading if source_heading.legs else 0.0
    for station, coil, heading in ((pick, source_coil, src_heading), (place, target_coil, dst_heading)):
        ok, line = station_reach_check(settings, arm, station, coil, heading, plate_top_z)
        notes.append(("OK   " if ok else "FAIL ") + line)

    notes.append(
        f"stations on the board margin at x = {pick.centre[0]:+.3f} and "
        f"{place.centre[0]:+.3f} m, surface {STATION_SURFACE_Z * 1000:.0f} mm above the plywood; "
        f"robot safe area ends at x = -0.084 / +0.538 m so neither is drivable into"
    )
    return WarehouseScene(pick=pick, place=place, payload_path=payload_path, notes=notes)
