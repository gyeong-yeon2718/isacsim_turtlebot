"""Thin wrappers over raw ``pxr`` for the primitives this scene needs.

Raw USD on purpose.  The higher-level ``isaacsim.core.api.objects`` wrappers are
convenient but they bring assumptions about scene registration and reset
behaviour that are not wanted for decoration, and every one is an extra API whose
signature can change between releases.  ``UsdGeom`` / ``UsdPhysics`` /
``UsdShade`` are stable USD schemas, so authoring against them directly is the
lower-risk choice for everything except the articulation itself, which does need
Isaac's wrapper to be controllable.

Everything here must be imported *after* ``SimulationApp`` has been constructed
in standalone mode -- ``pxr`` is not on ``sys.path`` before Kit boots.
"""

from __future__ import annotations

import math

from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade


def set_transform(
    prim: UsdGeom.Xformable,
    translate: tuple[float, float, float] | None = None,
    yaw: float = 0.0,
    scale: tuple[float, float, float] | None = None,
) -> None:
    """Author translate / rotateZ / scale, in that order.

    Ops are added explicitly rather than via ``AddTransformOp`` so the order is
    visible and stable: USD applies ``xformOpOrder`` as written, and a scale
    authored before a rotate produces a sheared prim, which is a confusing thing
    to debug from a screenshot.
    """
    prim.ClearXformOpOrder()
    if translate is not None:
        prim.AddTranslateOp().Set(Gf.Vec3d(*translate))
    if abs(yaw) > 1e-12:
        prim.AddRotateZOp().Set(math.degrees(yaw))
    if scale is not None:
        prim.AddScaleOp().Set(Gf.Vec3f(*scale))


def set_display_colour(prim, colour: tuple[float, float, float]) -> None:
    """Set a single constant display colour.

    The interpolation is declared ``constant`` explicitly.  Without it Fabric logs
    ``attribute primvars:displayColor:indices not found`` on every prim whose colour
    is written at run time -- a one-element array with no declared interpolation is
    ambiguous, and the per-frame glow updates write exactly that.
    """
    gprim = UsdGeom.Gprim(prim)
    attr = gprim.GetDisplayColorAttr()
    if not attr:
        attr = gprim.CreateDisplayColorAttr()
        UsdGeom.Primvar(attr).SetInterpolation(UsdGeom.Tokens.constant)
    attr.Set([Gf.Vec3f(*colour)])


def add_box(
    stage,
    path: str,
    size: tuple[float, float, float],
    center: tuple[float, float, float],
    colour: tuple[float, float, float] = (0.6, 0.6, 0.6),
    *,
    yaw: float = 0.0,
    collision: bool = False,
) -> UsdGeom.Cube:
    """A box built from a unit ``UsdGeom.Cube`` plus a scale op.

    ``UsdGeom.Cube`` is uniform by definition, so a non-cubic box has to come from
    a scale.  ``size`` is set to 1.0 (extent -0.5..0.5) so the scale factors read
    directly as metres, which makes the call sites checkable by eye.
    """
    cube = UsdGeom.Cube.Define(stage, Sdf.Path(path))
    cube.CreateSizeAttr(1.0)
    cube.CreateExtentAttr([Gf.Vec3f(-0.5, -0.5, -0.5), Gf.Vec3f(0.5, 0.5, 0.5)])
    set_transform(UsdGeom.Xformable(cube), center, yaw, size)
    set_display_colour(cube.GetPrim(), colour)
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


def add_cylinder(
    stage,
    path: str,
    radius: float,
    height: float,
    center: tuple[float, float, float],
    colour: tuple[float, float, float] = (0.6, 0.6, 0.6),
    *,
    axis: str = "Z",
    yaw: float = 0.0,
    collision: bool = False,
) -> UsdGeom.Cylinder:
    cyl = UsdGeom.Cylinder.Define(stage, Sdf.Path(path))
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    cyl.CreateAxisAttr(axis)
    half = 0.5 * height
    if axis == "Z":
        extent = [Gf.Vec3f(-radius, -radius, -half), Gf.Vec3f(radius, radius, half)]
    elif axis == "Y":
        extent = [Gf.Vec3f(-radius, -half, -radius), Gf.Vec3f(radius, half, radius)]
    else:
        extent = [Gf.Vec3f(-half, -radius, -radius), Gf.Vec3f(half, radius, radius)]
    cyl.CreateExtentAttr(extent)
    set_transform(UsdGeom.Xformable(cyl), center, yaw, None)
    set_display_colour(cyl.GetPrim(), colour)
    if collision:
        UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
    return cyl


def add_sphere(
    stage,
    path: str,
    radius: float,
    center: tuple[float, float, float],
    colour: tuple[float, float, float] = (0.6, 0.6, 0.6),
    *,
    collision: bool = False,
) -> UsdGeom.Sphere:
    sph = UsdGeom.Sphere.Define(stage, Sdf.Path(path))
    sph.CreateRadiusAttr(float(radius))
    sph.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -radius), Gf.Vec3f(radius, radius, radius)])
    set_transform(UsdGeom.Xformable(sph), center, 0.0, None)
    set_display_colour(sph.GetPrim(), colour)
    if collision:
        UsdPhysics.CollisionAPI.Apply(sph.GetPrim())
    return sph


def add_physics_material(
    stage, path: str, static: float, dynamic: float, restitution: float = 0.0
) -> UsdShade.Material:
    """A friction/restitution material.

    Worth being explicit about even for the floor: an unbound surface falls back to
    the scene default, and "why does the robot barely move" is a much harder
    question to answer when one side of the contact pair is whatever the default
    happens to be this release.
    """
    mat = UsdShade.Material.Define(stage, Sdf.Path(path))
    UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api = UsdPhysics.MaterialAPI(mat.GetPrim())
    api.CreateStaticFrictionAttr(float(static))
    api.CreateDynamicFrictionAttr(float(dynamic))
    api.CreateRestitutionAttr(float(restitution))
    return mat


def bind_physics_material(prim, material: UsdShade.Material) -> None:
    if prim.HasAPI(UsdShade.MaterialBindingAPI):
        binding = UsdShade.MaterialBindingAPI(prim)
    else:
        binding = UsdShade.MaterialBindingAPI.Apply(prim)
    binding.Bind(
        material,
        bindingStrength=UsdShade.Tokens.weakerThanDescendants,
        materialPurpose="physics",
    )


def add_quad_grid_mesh(
    stage,
    path: str,
    cells: list[list[tuple[float, float, float]]],
    cell_size: float,
    center: tuple[float, float, float],
    yaw: float,
) -> UsdGeom.Mesh:
    """One mesh holding an ``n x m`` grid of coloured quads on the XY plane.

    Used for the tag faces.  A grid of separate cubes would be 100 prims per tag
    and 1600 for the board; a single mesh with a uniform-interpolation
    ``displayColor`` is one prim, needs no texture files, and gives crisp cell
    edges because there is no texture filtering involved at all.
    """
    rows = len(cells)
    cols = len(cells[0]) if rows else 0
    points: list[Gf.Vec3f] = []
    counts: list[int] = []
    indices: list[int] = []
    colours: list[Gf.Vec3f] = []

    half_w = 0.5 * cols * cell_size
    half_h = 0.5 * rows * cell_size
    n = 0
    for r in range(rows):
        for c in range(cols):
            x0 = -half_w + c * cell_size
            y0 = half_h - (r + 1) * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            points.extend(
                [
                    Gf.Vec3f(x0, y0, 0.0),
                    Gf.Vec3f(x1, y0, 0.0),
                    Gf.Vec3f(x1, y1, 0.0),
                    Gf.Vec3f(x0, y1, 0.0),
                ]
            )
            indices.extend([n, n + 1, n + 2, n + 3])
            counts.append(4)
            colours.append(Gf.Vec3f(*cells[r][c]))
            n += 4

    mesh = UsdGeom.Mesh.Define(stage, Sdf.Path(path))
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform).Set(colours)
    set_transform(UsdGeom.Xformable(mesh), center, yaw, None)
    return mesh


# ---------------------------------------------------------------------------
# Emissive material
# ---------------------------------------------------------------------------


class GlowMaterial:
    """An OmniPBR material whose emission can be driven every frame.

    Every shader input that will be written later is created **now**, before the
    first physics step.  That is not tidiness: creating a shader input mid-run
    changes the material's network and forces a shader recompile, which stalls the
    render thread; ``Set`` on an input that already exists is a cheap value write.
    """

    def __init__(self, stage, path: str, base_colour: tuple[float, float, float]) -> None:
        import omni.kit.commands
        import omni.usd

        self.path = path
        omni.kit.commands.execute(
            "CreateMdlMaterialPrim",
            mtl_url="OmniPBR.mdl",
            mtl_name="OmniPBR",
            mtl_path=path,
            select_new_prim=False,
        )
        prim = stage.GetPrimAtPath(path)
        self.material = UsdShade.Material(prim)
        self.shader = UsdShade.Shader(omni.usd.get_shader_from_material(prim, get_prim=True))

        self._diffuse = self.shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f)
        self._enable = self.shader.CreateInput("enable_emission", Sdf.ValueTypeNames.Bool)
        self._colour = self.shader.CreateInput("emissive_color", Sdf.ValueTypeNames.Color3f)
        self._intensity = self.shader.CreateInput("emissive_intensity", Sdf.ValueTypeNames.Float)

        self._diffuse.Set(Gf.Vec3f(*base_colour))
        self._enable.Set(False)
        self._colour.Set(Gf.Vec3f(*base_colour))
        self._intensity.Set(0.0)

    def bind(self, prim) -> None:
        if prim.HasAPI(UsdShade.MaterialBindingAPI):
            binding = UsdShade.MaterialBindingAPI(prim)
        else:
            binding = UsdShade.MaterialBindingAPI.Apply(prim)
        binding.Bind(self.material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)

    def set_emission(self, colour: tuple[float, float, float], intensity: float) -> None:
        on = intensity > 1e-6
        self._enable.Set(bool(on))
        self._colour.Set(Gf.Vec3f(*colour))
        self._intensity.Set(float(max(0.0, intensity)))

    def set_diffuse(self, colour: tuple[float, float, float]) -> None:
        self._diffuse.Set(Gf.Vec3f(*colour))


def add_sphere_light(
    stage,
    path: str,
    center: tuple[float, float, float],
    radius: float = 0.02,
    colour: tuple[float, float, float] = (1.0, 1.0, 1.0),
    intensity: float = 0.0,
):
    """A point-ish light, returned with its intensity and colour attributes.

    The attribute handles are returned along with the light because re-resolving
    ``inputs:intensity`` by name on every frame is pure overhead, and the
    per-frame path here runs inside a physics callback.
    """
    lamp = UsdLux.SphereLight.Define(stage, Sdf.Path(path))
    lamp.CreateRadiusAttr(float(radius))
    lamp.CreateIntensityAttr(float(intensity))
    lamp.CreateColorAttr(Gf.Vec3f(*colour))
    set_transform(UsdGeom.Xformable(lamp), center, 0.0, None)
    return lamp, lamp.GetIntensityAttr(), lamp.GetColorAttr()


def add_dome_light(stage, path: str, intensity: float = 900.0):
    dome = UsdLux.DomeLight.Define(stage, Sdf.Path(path))
    dome.CreateIntensityAttr(float(intensity))
    dome.CreateColorAttr(Gf.Vec3f(0.92, 0.94, 1.0))
    return dome


def add_distant_light(stage, path: str, intensity: float = 1800.0, angle_deg: float = 35.0):
    light = UsdLux.DistantLight.Define(stage, Sdf.Path(path))
    light.CreateIntensityAttr(float(intensity))
    light.CreateAngleAttr(0.8)
    xf = UsdGeom.Xformable(light)
    xf.ClearXformOpOrder()
    xf.AddRotateXYZOp().Set(Gf.Vec3f(-float(angle_deg), 0.0, 25.0))
    return light

