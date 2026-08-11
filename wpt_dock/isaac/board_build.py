"""Build the plywood stage: coils, tape, AprilTags, lights, and the glow rig.

Everything is generated from :class:`wpt_dock.config.BoardSpec`, the same object
the planner and the tag detector read.  Nothing about the geometry is duplicated
here, so the picture on screen and the numbers the algorithm reasons about cannot
drift apart -- which is the usual way a docking demo ends up "working" against a
board that is not the board.

About the tag faces
-------------------
The 10 x 10 pattern drawn on each tag is a **visual stand-in**: a white quiet
zone, a black border, and a 6 x 6 payload generated deterministically from the tag
id.  It is *not* a valid tag36h11 codeword.  It cannot be, without shipping the
codebook, and nothing here depends on it: the detector in this package is
geometric (see ``apriltag.py``), so the bits are decoration that makes the board
recognisable in a screenshot.  If you later render real camera images and run a
real decoder, replace these faces with actual tag36h11 images -- and expect the
minimum-pixel and obliqueness gates to start mattering much more than they do now.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..config import BoardSpec, Settings
from .usd_helpers import (
    GlowMaterial,
    add_box,
    add_cylinder,
    add_distant_light,
    add_dome_light,
    add_physics_material,
    add_quad_grid_mesh,
    add_sphere_light,
    bind_physics_material,
    set_display_colour,
)

GAUGE_SEGMENTS = 16

_WOOD = (0.78, 0.62, 0.40)
_WOOD_DARK = (0.55, 0.42, 0.26)
_TAPE = (0.06, 0.06, 0.07)
_COPPER = (0.72, 0.45, 0.20)
_FERRITE = (0.20, 0.19, 0.18)


@dataclass
class CoilVisual:
    number: int
    position: tuple[float, float]
    face_prim: object
    core_prim: object
    material: GlowMaterial | None
    lamp_intensity: object
    lamp_colour: object
    gauge_prims: list = field(default_factory=list)
    beacon_prim: object = None
    beacon_material: GlowMaterial | None = None


@dataclass
class BoardScene:
    root: str
    coils: dict[int, CoilVisual]
    tag_prims: dict[int, object]

    def coil(self, number: int) -> CoilVisual:
        return self.coils[number]


def _tag_cells(tag_id: int) -> list[list[tuple[float, float, float]]]:
    """10 x 10 cell colours: quiet zone, black border, deterministic 6 x 6 payload."""
    white = (0.94, 0.94, 0.92)
    black = (0.04, 0.04, 0.05)
    # Small LCG so each id gets a distinct, reproducible pattern with no codebook.
    state = (tag_id * 1103515245 + 12345) & 0x7FFFFFFF
    bits: list[int] = []
    for _ in range(36):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        bits.append((state >> 16) & 1)

    cells: list[list[tuple[float, float, float]]] = []
    for r in range(10):
        row: list[tuple[float, float, float]] = []
        for c in range(10):
            if r == 0 or r == 9 or c == 0 or c == 9:
                row.append(white)
            elif r == 1 or r == 8 or c == 1 or c == 8:
                row.append(black)
            else:
                row.append(black if bits[(r - 2) * 6 + (c - 2)] else white)
        cells.append(row)
    return cells


def _add_tape_strip(stage, path: str, a, b, width: float, z: float) -> None:
    ax, ay = a
    bx, by = b
    length = math.hypot(bx - ax, by - ay)
    if length < 1e-6:
        return
    yaw = math.atan2(by - ay, bx - ax)
    add_box(
        stage, path, (length, width, 0.0006),
        (0.5 * (ax + bx), 0.5 * (ay + by), z),
        _TAPE, yaw=yaw,
    )


def _add_tape_ring(stage, root: str, centre, radius: float, width: float, z: float, segments: int = 28) -> None:
    """A circular tape ring approximated by short chords.

    The ring exists because the upstream config has a rule about it: the tape
    circling each coil must *not* be treated as a line to follow, which is why
    their ``coil_line_ignore_radius_m`` (0.11 m) is set larger than the ring
    itself.  Drawing it keeps that constraint visible instead of implicit.
    """
    step = 2.0 * math.pi / segments
    chord = 2.0 * radius * math.sin(0.5 * step) * 1.06
    for i in range(segments):
        a = i * step
        cx = centre[0] + radius * math.cos(a)
        cy = centre[1] + radius * math.sin(a)
        add_box(stage, f"{root}/seg{i:02d}", (chord, width, 0.0006), (cx, cy, z), _TAPE,
                yaw=a + 0.5 * math.pi)


def build_board(stage, settings: Settings, *, root: str = "/World/board") -> BoardScene:
    board: BoardSpec = settings.board
    ox, oy = board.stage_origin
    w, h = board.stage_size
    cx, cy = ox + 0.5 * w, oy + 0.5 * h
    t = board.plywood_thickness

    # Floor well below the table, purely for visual context.  No default ground
    # plane is used: it would land at z = 0, which is the *top* of the plywood, and
    # the robot would end up driving on an invisible surface at the same height.
    add_box(stage, f"{root}/floor", (4.0, 4.0, 0.02), (cx, cy, -0.34), (0.30, 0.31, 0.33),
            collision=True)
    add_box(stage, f"{root}/table", (w + 0.10, h + 0.10, 0.30), (cx, cy, -t - 0.15),
            (0.24, 0.25, 0.28), collision=True)

    # The plywood.  Top surface at z = 0 by construction: every tag, coil and tape
    # strip is authored just above it, and the robot's wheels roll on it.
    #
    # The friction is bound explicitly rather than left to the scene default.  One
    # side of the tyre/floor contact pair being "whatever the default is" makes a
    # traction problem much harder to diagnose, and this run already lost one debug
    # cycle to exactly that class of question.
    plywood = add_box(stage, f"{root}/plywood", (w, h, t), (cx, cy, -0.5 * t), _WOOD,
                      collision=True)
    bind_physics_material(
        plywood.GetPrim(), add_physics_material(stage, "/World/Looks/PhysicsPlywood", 0.90, 0.80)
    )
    add_box(stage, f"{root}/plywood_edge", (w + 0.004, h + 0.004, 0.004),
            (cx, cy, -t - 0.002), _WOOD_DARK)

    if board.guard_rail:
        # A sim-only lip. The real board has no rail; the safe-area check in the
        # controller is what keeps the robot on the plywood, and this only exists so
        # that a failure during development is recoverable instead of ending with the
        # robot on the floor.  It sits 0.115 m outside the closest legal pose, so it
        # is never touched during a correct run.
        rh, rt = 0.022, 0.006
        add_box(stage, f"{root}/rail_south", (w, rt, rh), (cx, oy + 0.5 * rt, 0.5 * rh),
                (0.70, 0.72, 0.75), collision=True)
        add_box(stage, f"{root}/rail_north", (w, rt, rh), (cx, oy + h - 0.5 * rt, 0.5 * rh),
                (0.70, 0.72, 0.75), collision=True)
        add_box(stage, f"{root}/rail_west", (rt, h, rh), (ox + 0.5 * rt, cy, 0.5 * rh),
                (0.70, 0.72, 0.75), collision=True)
        add_box(stage, f"{root}/rail_east", (rt, h, rh), (ox + w - 0.5 * rt, cy, 0.5 * rh),
                (0.70, 0.72, 0.75), collision=True)

    # --- tape grid -------------------------------------------------------
    coils = board.coil_positions
    sx, sy = board.coil_spacing
    tape_z = 0.0004
    for i, y in enumerate(sorted({p[1] for p in coils.values()})):
        _add_tape_strip(stage, f"{root}/tape/row{i}", (ox + 0.02, y), (ox + w - 0.02, y),
                        board.tape_width, tape_z)
    for i, x in enumerate(sorted({p[0] for p in coils.values()})):
        _add_tape_strip(stage, f"{root}/tape/col{i}", (x, oy + 0.02), (x, oy + h - 0.02),
                        board.tape_width, tape_z)
    for n, pos in coils.items():
        _add_tape_ring(stage, f"{root}/tape/ring{n}", pos, board.tape_ring_radius,
                       board.tape_width, tape_z)

    # --- coils -----------------------------------------------------------
    coil_visuals: dict[int, CoilVisual] = {}
    for n, (px, py) in coils.items():
        base = f"{root}/coil{n}"
        face = add_cylinder(stage, f"{base}/face", board.coil_radius, 0.004,
                            (px, py, 0.002), _COPPER)
        core = add_cylinder(stage, f"{base}/core", board.coil_radius * 0.45, 0.005,
                            (px, py, 0.0025), _FERRITE)

        material: GlowMaterial | None = None
        try:
            material = GlowMaterial(stage, f"/World/Looks/CoilGlow{n}", _COPPER)
            material.bind(face.GetPrim())
        except Exception as exc:                  # noqa: BLE001
            # MDL creation is the one step here that depends on an optional
            # extension.  If it is unavailable the demo must still be readable, so
            # the visuals also drive displayColor unconditionally -- see visuals.py.
            print(f"[board] emissive material unavailable for coil {n} ({exc}); "
                  f"falling back to display colour and the lamp only")

        _, lamp_i, lamp_c = add_sphere_light(
            stage, f"{base}/lamp", (px, py, 0.045), radius=0.030,
            colour=(1.0, 0.4, 0.1), intensity=0.0,
        )
        # A vertical beacon above the coil.  A flat pad on a flat board is a hard thing
        # to notice from a three-quarter view, however bright it is -- the lit area is
        # tiny and mostly facing away from the camera.  A column has silhouette from
        # every angle, so it reads as "this coil is live" without hunting for it.
        beacon = add_cylinder(stage, f"{base}/beacon", 0.010, 0.170, (px, py, 0.086),
                              (0.10, 0.10, 0.12))
        beacon_mat: GlowMaterial | None = None
        try:
            beacon_mat = GlowMaterial(stage, f"/World/Looks/CoilBeacon{n}", (0.06, 0.06, 0.07))
            beacon_mat.bind(beacon.GetPrim())
        except Exception:                             # noqa: BLE001
            beacon_mat = None

        gauge: list = []
        gauge_r = board.tape_ring_radius + 0.017
        for i in range(GAUGE_SEGMENTS):
            a = 0.5 * math.pi - i * (2.0 * math.pi / GAUGE_SEGMENTS)
            gx = px + gauge_r * math.cos(a)
            gy = py + gauge_r * math.sin(a)
            seg = add_box(stage, f"{base}/gauge/seg{i:02d}", (0.009, 0.005, 0.0012),
                          (gx, gy, 0.0008), (0.22, 0.22, 0.24), yaw=a + 0.5 * math.pi)
            gauge.append(seg.GetPrim())

        coil_visuals[n] = CoilVisual(
            number=n, position=(px, py), face_prim=face.GetPrim(), core_prim=core.GetPrim(),
            material=material, lamp_intensity=lamp_i, lamp_colour=lamp_c, gauge_prims=gauge,
            beacon_prim=beacon.GetPrim(), beacon_material=beacon_mat,
        )

    # --- AprilTags -------------------------------------------------------
    tag_prims: dict[int, object] = {}
    for tid, (tx, ty, tyaw) in board.tag_positions().items():
        cells = _tag_cells(tid)
        mesh = add_quad_grid_mesh(
            stage, f"{root}/tags/tag{tid}", cells,
            cell_size=board.tag_size / 10.0, center=(tx, ty, 0.0008), yaw=tyaw,
        )
        tag_prims[tid] = mesh.GetPrim()

    # --- lighting --------------------------------------------------------
    # Deliberately dim.  Emission only reads as *glow* if the scene around it is
    # darker than the emitter -- at the previous 650/1500 the plywood was already near
    # white, so a lit coil just looked like a slightly different shade of pad.  Enough
    # ambient to see the board and the tags, and no more.
    add_dome_light(stage, "/World/lighting/dome", intensity=180.0)
    add_distant_light(stage, "/World/lighting/sun", intensity=450.0)

    return BoardScene(root=root, coils=coil_visuals, tag_prims=tag_prims)
