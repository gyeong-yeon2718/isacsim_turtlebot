"""Load the user's 3D-printed parts from STL into USD meshes.

Why parse the file here instead of using an importer extension: ``omni.kit
.asset_converter`` and the CAD importer are asynchronous, are not guaranteed to be
enabled in every ``.kit`` configuration, and would put a conversion step between
the user's file and what appears on screen.  Binary STL is a fifty-byte-per-
triangle format; reading it directly is about forty lines, works in any Kit
configuration, and lets the units and orientation be set explicitly -- which
matters, because STL carries no unit information at all and these files are in
millimetres.

Deliberate choice: the imported meshes are **visual only**.  Collision uses the
simple boxes in ``robot_build.py``.  A convex decomposition of a honeycombed
printed plate would produce hundreds of hulls, cost far more per physics step than
the whole rest of this scene, and buy nothing -- the plate never touches anything.
Mesh colliders are the classic way to make a small simulation inexplicably slow.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np
from pxr import Gf, Sdf, UsdGeom

from .usd_helpers import set_transform, set_display_colour


@dataclass
class StlData:
    triangles: np.ndarray        # (n, 3, 3) in the file's own units
    source: str

    @property
    def count(self) -> int:
        return len(self.triangles)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        flat = self.triangles.reshape(-1, 3)
        return flat.min(axis=0), flat.max(axis=0)

    def describe(self, scale: float = 1.0) -> str:
        lo, hi = self.bounds
        ext = (hi - lo) * scale
        return (
            f"{self.count} triangles, extent "
            f"{ext[0]:.4f} x {ext[1]:.4f} x {ext[2]:.4f} (scaled), "
            f"min ({lo[0] * scale:+.4f}, {lo[1] * scale:+.4f}, {lo[2] * scale:+.4f})"
        )


def read_stl(path: str) -> StlData:
    """Read binary or ASCII STL.

    Format detection is by *size*, not by the leading "solid" keyword: plenty of
    binary exporters write "solid" into their 80-byte header, so the keyword alone
    misidentifies them.  A binary file is exactly ``84 + 50 * n`` bytes, and that
    test is unambiguous.
    """
    with open(path, "rb") as f:
        blob = f.read()

    if len(blob) >= 84:
        count = struct.unpack_from("<I", blob, 80)[0]
        if len(blob) == 84 + 50 * count and count > 0:
            # Each 50-byte record is: 3 floats normal, 9 floats vertices, 2 bytes attr.
            rec = np.frombuffer(blob, dtype=np.uint8, count=50 * count, offset=84).reshape(count, 50)
            verts = rec[:, 12:48].copy().view("<f4").reshape(count, 3, 3)
            return StlData(np.asarray(verts, dtype=float), path)

    text = blob.decode("utf-8", errors="replace")
    tris: list[list[list[float]]] = []
    current: list[list[float]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0] == "vertex":
            current.append([float(parts[1]), float(parts[2]), float(parts[3])])
            if len(current) == 3:
                tris.append(current)
                current = []
    if not tris:
        raise ValueError(f"{path}: not a recognisable binary or ASCII STL")
    return StlData(np.asarray(tris, dtype=float), path)


def add_stl_mesh(
    stage,
    prim_path: str,
    stl: StlData,
    *,
    scale: float = 0.001,
    translate: tuple[float, float, float] = (0.0, 0.0, 0.0),
    yaw: float = 0.0,
    colour: tuple[float, float, float] = (0.14, 0.14, 0.15),
    recenter_xy: bool = False,
    zero_bottom: bool = False,
) -> UsdGeom.Mesh:
    """Author the triangles as a ``UsdGeom.Mesh``.

    Vertices are **not** welded.  STL is a per-face format with no shared normals,
    so authoring one vertex per corner and no normals attribute lets the renderer
    derive flat facets -- which is what a printed part actually looks like.  Welding
    would smooth-shade across the sharp edges of the honeycomb and make the part
    look like melted plastic.

    ``recenter_xy`` / ``zero_bottom`` exist because an STL's origin is wherever the
    CAD happened to put it: these two files have origins that are neither centred
    nor on the base.  Normalising here means the mount offsets in ``config`` are
    measured from something meaningful rather than from an arbitrary CAD datum.
    """
    tris = stl.triangles * float(scale)
    lo = tris.reshape(-1, 3).min(axis=0)
    hi = tris.reshape(-1, 3).max(axis=0)
    shift = np.zeros(3)
    if recenter_xy:
        shift[0] = -0.5 * (lo[0] + hi[0])
        shift[1] = -0.5 * (lo[1] + hi[1])
    if zero_bottom:
        shift[2] = -lo[2]
    tris = tris + shift

    flat = tris.reshape(-1, 3)
    points = [Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in flat]
    n = len(tris)
    counts = [3] * n
    indices = list(range(3 * n))

    mesh = UsdGeom.Mesh.Define(stage, Sdf.Path(prim_path))
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    lo2 = flat.min(axis=0)
    hi2 = flat.max(axis=0)
    mesh.CreateExtentAttr([Gf.Vec3f(*lo2.astype(float)), Gf.Vec3f(*hi2.astype(float))])
    set_transform(UsdGeom.Xformable(mesh), translate, yaw, None)
    set_display_colour(mesh.GetPrim(), colour)
    return mesh


def try_read(path: str) -> StlData | None:
    """Read an STL, or return ``None`` with a printed reason.

    A missing custom part must not stop the mission from running: the algorithm
    does not depend on the printed geometry, only on the numbers in ``config``.  So
    a bad path degrades the picture and nothing else.
    """
    try:
        return read_stl(path)
    except FileNotFoundError:
        print(f"[stl] not found, skipping: {path}")
    except Exception as exc:                      # noqa: BLE001 - report and carry on
        print(f"[stl] could not read {path}: {exc}")
    return None

