"""Every tunable number in one place, with its unit and its provenance.

Provenance tags used in the comments, because on this task the difference matters
a great deal:

* ``MEASURED``   -- a number that exists in the upstream repos as a recorded
                    measurement of the physical rig.
* ``REPO``       -- a number configured in the upstream repos (a setting they
                    chose, not necessarily a measurement).
* ``POSTER``     -- taken from the project poster's stated model.
* ``SPEC``       -- from the ROBOTIS TurtleBot3 datasheet.
* ``ESTIMATED``  -- derived here from the STL bounding boxes and the photographs.
                    Not measured. Change it if you can measure it.
* ``DESIGN``     -- a requirement or gain we are choosing, with the reasoning
                    written next to it.

Units are SI everywhere: metres, seconds, radians, kilograms.  Isaac Sim stages
default to ``metersPerUnit = 1.0``, so these go into USD unchanged.

Known conflict, deliberately surfaced rather than silently resolved
------------------------------------------------------------------
Coil spacing appears three times in the upstream work with three different
values: ``0.453 / 0.255`` (repo A ``shelf_layout.py``, carrying a dated
measurement annotation, and repeated in ``coil_transit.py``), ``0.470 / 0.270``
(repo B ``purepursuit_pid.yaml``) and "approximately 0.45 / 0.30" (repo A
README).  The default below is the annotated-measurement pair, because a
measurement beats a config value and beats prose.  Both alternatives are
provided as named constructors so a re-measurement is a one-line change.  A wrong
spacing does not crash anything -- it silently biases every dead-reckoned
approach, which is the worst class of error to leave implicit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

# ---------------------------------------------------------------------------
# Robot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RobotSpec:
    """TurtleBot3 Burger geometry, published limits, and the custom superstructure.

    Which variant?  Neither upstream repo states it.  Inferred as **Burger** from
    two pieces of evidence: the custom top plate STL is 230 mm across, and in the
    photographs it visibly overhangs the chassis on both sides -- which is true of
    a 138 mm-wide Burger deck and false of a 281 mm-wide Waffle deck.  If the rig
    is actually a Waffle, ``wheel_separation`` nearly doubles and every
    dead-reckoned heading is wrong, so this is worth confirming before trusting a
    result.
    """

    # SPEC: ROBOTIS TurtleBot3 Burger.
    wheel_radius: float = 0.033          # m
    wheel_separation: float = 0.160      # m, between wheel contact points
    base_footprint: tuple[float, float] = (0.138, 0.178)   # m, (width, length)
    max_body_speed: float = 0.22         # m/s
    max_body_yaw_rate: float = 2.84      # rad/s

    # ESTIMATED from the STL bounding boxes and the photographs.
    plate_size: tuple[float, float, float] = (0.0924, 0.2300, 0.0090)  # m, custom top plate (x, y, z)
    plate_height: float = 0.1200         # m, top-plate underside above the running surface
    # --- the printed rear rack ("터틀봇초안 3층_홈") -------------------------------
    # MEASURED from the STL, not chosen.  It is a three-shelf rack with a 5 mm back wall at
    # local x = 0..5 that opens towards +x, and its local frame is:
    #
    #     bottom slab       z = -5.00 .. 0.00     (rests on the tier-1 plate)
    #     middle shelf      z = 80.20 .. 85.00    (the battery pack sits here)
    #     top slab          z = 128.20 .. 133.00
    #     footprint         x = 0 .. 75, y = -40 .. +80 mm
    #
    # The user's two constraints -- "the bottom sits on tier 1" and "the underside of the
    # top part touches right on top of tier 3" -- together *fix* the tier spacing:
    #
    #     tier3_top - tier1_top = 128.20 - (-5.00) = 133.20 mm
    #
    # so tier 1 is derived from the measured deck rather than guessed.  With the asset's
    # tier-3 top at 151.5 mm that puts tier 1 at 18.3 mm, which is a believable height for
    # the Burger's bottom plate (its chassis mesh starts at 10.0 mm).
    rack_bottom_local: float = -0.0050        # m, local z of the rack's underside
    rack_top_underside_local: float = 0.1282  # m, local z of the top slab's underside
    rack_top_upper_local: float = 0.1330      # m, local z of the top slab's upper face
    rack_size: tuple[float, float, float] = (0.0750, 0.1200, 0.1380)   # m, fallback box
    rack_y_centre_local: float = 0.0200       # m, local y midpoint, for lateral centring

    # How far the tier-1 plate is extended rearwards by the extra half plate.
    #
    # NOT read from the DWG: ``TB3_Waffle_Plate-IPL-01.dwg`` is a binary AutoCAD 2007
    # (AC1021) file and nothing dimensional survives a text scan of it -- only the Autodesk
    # signature and two font names.  52.5 mm is half of ROBOTIS's 105 mm waffle plate, and
    # it is also the value the rest of the geometry wants: the rack is 75 mm deep, so with a
    # 52.5 mm extension its top slab still overlaps the rear of tier 3 by 22.5 mm, which is
    # what "touching on top of tier 3" requires.  A 105 mm half-plate (52.5 mm) works; half
    # of the 178 mm chassis length (89 mm) would push the rack clear of tier 3 entirely and
    # the contact could not happen.  Measure it and correct this one number if it is wrong.
    rear_extension: float = 0.0525            # m
    rear_extension_thickness: float = 0.0040  # m
    chassis_mass: float = 1.30           # kg, Burger plus battery pack, CC/CV board and printed parts
    wheel_mass: float = 0.05             # kg

    # DESIGN: the receiver coil is under the chassis.  Its centre is the point the
    # whole task is about -- tolerances are on *this* point, not on the robot
    # origin -- so it gets its own offset instead of being assumed to be zero.
    rx_coil_offset: tuple[float, float] = (0.0, 0.0)   # m, in the body frame
    rx_coil_radius: float = 0.026        # m, ESTIMATED from the photographs

    # The official ROBOTIS model.  This is the route the lecture material takes
    # ("Moving the Turtlebot in Isaac Sim -- Turtlebot3_burger.usd", p.6), and the
    # asset is reachable on the 5.1 CDN: HTTP 200, a 4.5 kB wrapper whose payload is
    # an 11.1 MB base USD.  Note the folder is ``Turtlebot/Turtlebot3/`` -- *not*
    # ``Robotis/`` -- which is why nothing in the local install can name this path and
    # why the first version of this project built the robot from primitives instead.
    #
    # Only the burger variant exists in the 5.1 tree; there is no waffle.  That
    # settles the variant question the upstream repos never answered.
    # Which published variant, and why this one.  ``tools/probe_tb3_assets.py`` loads all
    # three and counts what arrives:
    #
    #   5.1  /Isaac/Robots/Turtlebot/Turtlebot3/turtlebot3_burger.usd   0 mesh prims
    #   4.5  /Isaac/Robots/Turtlebot/turtlebot3_burger.usd              0 mesh prims
    #   4.2  /Isaac/Robots/Turtlebot/turtlebot3_burger.usd              4 mesh prims  <- this one
    #
    # The 5.1 and 4.5 packages are broken in the same way: their base layer references
    # ``configuration/turtlebot3_burger_physics.usd@</visuals/...>`` and that prim path is
    # not in the file, so the robot composes with links and joints but **no visible
    # geometry**.  The 4.2 package is self-contained and intact.  Same robot, same
    # ``wheel_left_joint`` / ``wheel_right_joint``, 191 mm tall as the Burger datasheet says.
    #
    # ``asset_version_override`` substitutes the version segment of whatever asset root is
    # configured, so this keeps working if the root is repointed.
    asset_relative_path: str = "/Isaac/Robots/Turtlebot/turtlebot3_burger.usd"
    asset_version_override: str | None = "4.2"
    # Substrings identifying the LDS-01 lidar the user physically removed.  Matching
    # prims are made invisible rather than deleted, so the articulation topology --
    # and therefore the joint indices -- is untouched.
    hide_prim_keywords: tuple[str, ...] = ("lidar", "lds", "scan", "laser")
    # The printed plate *replaces* the stock third tier rather than sitting on top of it,
    # so hide the stock one where the asset exposes it separately.  Kept narrow on
    # purpose: a bare "plate" would also match the first and second tiers, and on many
    # TurtleBot3 builds all three are a single visual mesh anyway.
    replaced_plate_keywords: tuple[str, ...] = ("tier3", "tier_3", "top_plate", "plate_top")

    @property
    def footprint_half_extents(self) -> tuple[float, float]:
        """``(along body X, along body Y)`` half-extents of what must stay on the plywood.

        This is the **support** envelope -- the wheels and casters -- not the vehicle's
        overall outline, and the distinction is load bearing.  The custom plate is 230 mm
        wide and sits 156 mm above the running surface; the rear extension plate hangs off
        the back at 18 mm.  Both can overhang the edge of the board freely, because there
        is nothing at those heights to hit.  What cannot leave the board is whatever
        touches it.

        Getting this wrong is not academic: with the 230 mm plate treated as the constraint,
        the extended robot no longer fits the stage at all -- coil 1 and coil 4 both land
        outside the computed safe area by about 9 mm, and every route is rejected.  The
        robot is fine; the model of it was wrong.

        Wheels sit at ``y = +-0.080`` and are 18 mm wide, so 0.089 covers them; the casters
        span ``x = -0.075 .. +0.071``, so 0.089 covers those too.
        """
        return (0.089, 0.089)

    @property
    def overall_half_extents(self) -> tuple[float, float]:
        """Full outline including the overhanging plate and the rear extension.

        Reporting only -- never used for containment, for the reason above.
        """
        a = 0.5 * (self.base_footprint[1] + self.rear_extension)
        b = 0.5 * max(self.base_footprint[0], self.plate_size[1])
        return (a, b)

    @property
    def swept_radius(self) -> float:
        """Radius the support envelope sweeps during an in-place rotation.

        Used for *planning* -- a turn must be safe at every heading it passes through.  It
        is deliberately not used for the running check: on this board the coil column sits
        only a few centimetres from the safe-area edge, so applying a swept radius while
        driving a straight leg reports the robot off the plywood when it is comfortably on
        it.  The Monte Carlo found exactly that false alarm.  Plan with the swept radius,
        drive with the oriented footprint.
        """
        a, b = self.footprint_half_extents
        return math.hypot(a, b)

    @property
    def obstacle_swept_radius(self) -> float:
        """Radius swept by everything low enough to hit a **structure**, not the board edge.

        A different question from ``swept_radius``, and confusing the two put a roller conveyor
        through the robot's rear extension plate.  ``swept_radius`` asks "does the robot stay on
        the plywood", and for that the support envelope is right, because the overhanging parts
        can hang over the edge freely -- there is nothing at those heights to hit.  A conveyor
        standing on the board *is* something at those heights to hit, so containment logic is
        the wrong tool and this is the right one.

        Measured about the rotation centre, which for a differential drive is the wheel axle
        midpoint -- the chassis origin here.  The outline is deliberately **not** treated as
        centred on it: the rear extension puts structure 141.5 mm behind the axle while the
        front reaches only 89 mm, so a half-extent of the bounding box would understate the
        rear corner by 26 mm.  Per corner, at this robot's numbers:

            rear extension  (-0.1415, +-0.069)  ->  157.4 mm   <- binding
            rear rack       (-0.1415, +-0.060)  ->  153.7 mm
            front           (+0.0890, +-0.069)  ->  112.6 mm
            wheels          ( 0.0000, +-0.089)  ->   89.0 mm

        The 230 mm-wide printed plate is excluded on purpose: it sits about 156 mm up, its
        corner is only 123.9 mm out anyway, and it genuinely does pass over a low obstacle.  An
        obstacle tall enough to reach the plate would need its own check -- the plate's height is
        *measured* from the grafted meshes at build time, so it is not a number this spec can
        carry, and ``build_warehouse`` is where such a check would belong.
        """
        rear_x = 0.5 * self.base_footprint[1] + self.rear_extension
        candidates = [
            (rear_x, 0.5 * self.base_footprint[0]),
            (rear_x, 0.5 * self.rack_size[1]),
            (0.5 * self.base_footprint[1], 0.5 * self.base_footprint[0]),
            (0.0, self.footprint_half_extents[1]),
        ]
        return max(math.hypot(x, y) for x, y in candidates)

    @property
    def max_wheel_rate(self) -> float:
        """Wheel rate that makes *both* published limits reachable.

        Straight-line needs ``v_max / r``; spin-in-place needs
        ``w_max * L / (2 r)``.  Taking the smaller would silently make one of the
        two datasheet numbers unattainable.
        """
        return max(
            self.max_body_speed / self.wheel_radius,
            self.max_body_yaw_rate * self.wheel_separation / (2.0 * self.wheel_radius),
        )

    def body_to_wheels(self, v: float, w: float) -> tuple[float, float]:
        half = 0.5 * self.wheel_separation
        return (v - w * half) / self.wheel_radius, (v + w * half) / self.wheel_radius

    def wheels_to_body(self, left: float, right: float) -> tuple[float, float]:
        v = 0.5 * self.wheel_radius * (right + left)
        w = self.wheel_radius * (right - left) / self.wheel_separation
        return v, w


# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraSpec:
    """One camera: intrinsics, and its pose on the robot.

    The extrinsics here are ESTIMATED.  The upstream configs give only a device
    index and a resolution -- mount height, pitch and offset appear nowhere.  That
    omission is very likely part of why the hardware alignment underperformed: a
    tag-based pose estimate is only as good as the transform from camera to robot,
    and an unrecorded extrinsic is an unbounded error.  They are written out
    explicitly here so they can be measured and corrected.

    Frame convention: the camera looks down its own +Z.  ``yaw`` then ``pitch``
    are applied in the robot frame; ``pitch = -90 deg`` points straight down.
    """

    name: str
    width: int
    height: int
    hfov: float                                  # rad, horizontal field of view
    position: tuple[float, float, float]         # m, in the body frame
    yaw: float                                   # rad, about +Z, 0 = forward
    pitch: float                                 # rad, negative = tilted downwards
    fps: float = 10.0

    @property
    def fx(self) -> float:
        return 0.5 * self.width / math.tan(0.5 * self.hfov)

    @property
    def fy(self) -> float:
        # Square pixels: the vertical focal length equals the horizontal one, and
        # the vertical field of view follows from the aspect ratio rather than
        # being a second free parameter.
        return self.fx

    @property
    def cx(self) -> float:
        return 0.5 * self.width

    @property
    def cy(self) -> float:
        return 0.5 * self.height

    @property
    def vfov(self) -> float:
        return 2.0 * math.atan(0.5 * self.height / self.fy)


# Camera mount points, MEASURED from the user's top-plate STL rather than guessed.
#
# The plate was rasterised at 0.5 mm and its interior voids ranked by area (see
# ``tools/plate_holes.py``).  Exactly three of them are camera-board sized
# through-holes, and they are the three the boards sit in on the real robot:
#
#     front  centre (+25.11,    0.00) mm   bbox 21.0 x 25.0 mm
#     left   centre (-12.64, +101.50) mm   bbox 25.5 x 19.0 mm
#     right  centre (-12.64, -101.50) mm   bbox 25.5 x 19.0 mm
#
# The remaining large voids are a 49.5 x 13 mm slot pair at y = +-79.5 and a
# 14 x 48 mm midline slot -- cable routing, not camera pockets.
#
# Coordinates are in the plate frame *after* centring its bounding box in XY, which
# is how ``robot_build`` mounts it, so the +2.2025 mm centring shift is already
# folded in.
PLATE_HOLE_FRONT = (0.02731, 0.00000)
PLATE_HOLE_LEFT = (-0.01044, 0.10150)
PLATE_HOLE_RIGHT = (-0.01044, -0.10150)


def default_cameras(lens_z: float = 0.1545) -> tuple[CameraSpec, ...]:
    """The three USB cameras, seated in the plate's own camera holes.

    All three look **straight down**.  That is what the hardware geometry forces,
    not a simplification: the pockets are through-holes in a 9 mm flat plate, the
    boards drop into them, and the photographs show the ribbon connectors facing up
    -- so the lenses face down through the holes.  An earlier version had the front
    camera pitched 52 deg forward, which was an invention; the plate cannot mount it
    that way.

    The default ``lens_z`` is **measured**, not chosen: the intact TurtleBot3 asset's
    chassis mesh tops out at 151.5 mm once the lidar is hidden, the printed plate sits
    1 mm above that, and the lens is 2 mm into the 9 mm pocket -- 154.5 mm.  The Isaac
    build re-measures and overrides it via ``with_camera_height``, so the extrinsics the
    estimator believes and the plate you can see are always the same numbers.  A camera
    extrinsic that disagrees with the model is an unbounded error, and it is exactly what
    the upstream configs never recorded.

    Ground coverage at 0.1545 m with a 62 deg horizontal field of view (footprint about
    0.186 x 0.139 m):

    * left/right, at ``y = +-0.1015``, cover ``|y|`` from about 0.009 to 0.194 m, so the
      west/east coil tags at 0.0975 m are in view *at the docked pose* -- which is what
      makes the fine alignment observable when it matters, not only during the run-in;
    * front, at ``x = +0.0273``, covers ``x`` from -0.066 to +0.120, so it picks up the
      north/south tag at the docked pose too.  At the 0.115 m height guessed before the
      plate was measured, it did not -- the extra height is what buys the third tag.

    Resolution follows repo B (640x480) rather than repo A (320x240): corner-noise
    pose uncertainty scales with pixel size and there is no reason to inherit the
    tighter constraint.  ``fps = 10`` is kept, because that is the real sensing
    cadence and pretending otherwise would flatter the estimator.
    """
    hfov = math.radians(62.0)     # ESTIMATED, typical low-cost USB / Pi camera
    down = math.radians(-90.0)
    return (
        CameraSpec("front", 640, 480, hfov, (*PLATE_HOLE_FRONT, lens_z), 0.0, down),
        CameraSpec("left_bottom", 640, 480, hfov, (*PLATE_HOLE_LEFT, lens_z), 0.0, down),
        CameraSpec("right_bottom", 640, 480, hfov, (*PLATE_HOLE_RIGHT, lens_z), 0.0, down),
    )


@dataclass(frozen=True)
class DetectionSpec:
    """Noise and gating for the AprilTag detector.

    ``corner_sigma`` is the one number that sets the accuracy floor of the whole
    system, so it is worth being explicit: a well-lit tag36h11 corner is usually
    localised to a few tenths of a pixel, and 0.35 px is a fair figure for a
    640x480 USB camera that has actually been calibrated.  ``intrinsic_bias`` is
    the part that does *not* average away -- lens calibration error is fixed for
    the run -- and it is why more frames cannot buy unlimited precision.
    """

    corner_sigma: float = 0.35        # px, per corner per axis, per frame
    intrinsic_bias: float = 0.30      # px, fixed per run, does not average out
    min_tag_pixels: float = 14.0      # px, smallest apparent tag edge the detector can decode
    max_view_angle: float = math.radians(72.0)   # rad, obliqueness limit off the tag normal
    dropout: float = 0.03             # -, probability a visible tag is missed in a frame
    min_tags_for_fix: int = 1         # one tag gives 4 points, which already determines a 2D pose
    margin_px: float = 6.0            # px, keep detections off the image border

    # The floor that no amount of averaging can beat, and the fix for a real bug
    # the Monte Carlo exposed.
    #
    # Random corner noise is tiny once back-projected: 0.35 px at 0.115 m through a
    # 532 px focal length is about 0.08 mm on the floor.  Averaged over 12 corners
    # and 130 frames, a filter that treats those frames as independent drives its
    # own covariance to about 0.1 mm -- and then, being that confident, its
    # chi-square gate starts *rejecting* perfectly good fixes that disagree with it,
    # and it never recovers.  That is filter divergence through overconfidence, and
    # it was producing 30 mm errors that the robot sincerely believed were zero.
    #
    # The physical reality is that successive frames are *not* independent: mount
    # pose and intrinsics are calibration constants, so their error is common to
    # every frame from that camera.  Correctly, it enters at the pose level, not the
    # point level, and it is what these two numbers represent.
    #
    # Derivation from the calibration spec in ``apriltag.perturbed_cameras``, at the
    # measured 0.1545 m lens height:
    #   a 0.3 deg mount-pitch error displaces a back-projected point by
    #   0.1545 * tan(0.3 deg) ~ 0.81 mm; a 1 mm mount-position error contributes 1 mm
    #   directly; combined per camera ~1.3 mm, and a fit blending two or three cameras
    #   lands near 1.8 mm.  The two bottom cameras sit 0.203 m apart, so the
    #   *differential* part, ~1.8 mm, appears as yaw: 1.8 mm / 0.203 m ~ 9 mrad ~ 0.5 deg.
    # Both scale with lens height, which is why they moved when the plate was measured
    # rather than guessed.
    systematic_pos_sigma: float = 0.0018           # m
    systematic_yaw_sigma: float = math.radians(0.55)  # rad


# ---------------------------------------------------------------------------
# Board / task geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoardSpec:
    """The plywood stage and its four transmitter coils.

    Frame: coil 1 is the origin, +X points to coil 2, +Y points to coil 3 -- the
    convention stated in repo B's config.  Global compass names for the tags:
    north = +Y, south = -Y, west = -X, east = +X.
    """

    stage_size: tuple[float, float] = (0.80, 0.60)   # m, MEASURED (repo A shelf_layout.py)
    coil_spacing: tuple[float, float] = (0.453, 0.255)  # m, MEASURED (repo A, dated annotation)
    plywood_thickness: float = 0.018     # m, ESTIMATED from the photographs
    coil_center_to_tag: float = 0.0975   # m, REPO (repo B purepursuit_pid.yaml)
    tag_size: float = 0.030              # m, ESTIMATED -- upstream says only "measure your printed tag"
    coil_radius: float = 0.030           # m, ESTIMATED transmitter coil radius
    tape_ring_radius: float = 0.055      # m, ESTIMATED insulation-tape ring around each coil
    line_ignore_radius: float = 0.11     # m, REPO -- larger than the tape ring on purpose
    tape_width: float = 0.019            # m, ESTIMATED standard electrical tape
    guard_rail: bool = True              # DESIGN: sim-only lip so a runaway does not fall off the table

    @property
    def coil_positions(self) -> dict[int, tuple[float, float]]:
        """Shelf 1..4 at (row, col) = {1:(0,0), 2:(0,1), 3:(1,0), 4:(1,1)}.

        Row/col indexing is repo A's ``SHELF_ROW`` / ``SHELF_COL``; row advances
        along +Y and column along +X, which is what makes coil 1 the origin and
        coil 4 the diagonal opposite.
        """
        sx, sy = self.coil_spacing
        return {1: (0.0, 0.0), 2: (sx, 0.0), 3: (0.0, sy), 4: (sx, sy)}

    @property
    def stage_origin(self) -> tuple[float, float]:
        """Bottom-left corner of the plywood in coil-1 coordinates.

        The coil array is centred on the stage: with a 0.453 m span on an 0.80 m
        board the margin is 0.1735 m a side, and 0.1725 m on the 0.60 m axis.
        Those two coming out nearly equal is a decent consistency check on the
        measured spacing.
        """
        sx, sy = self.coil_spacing
        w, h = self.stage_size
        return (-(w - sx) * 0.5, -(h - sy) * 0.5)

    def tag_positions(self) -> dict[int, tuple[float, float, float]]:
        """Tag id -> (x, y, yaw) on the board plane.

        ``id = shelf * 10 + position`` with north=1, south=2, west=3, east=4
        (repo A/B ``tag_layout.py``).  ``yaw`` is the tag's own +X axis in board
        coordinates; each tag is rotated to face outward from its coil, which is
        the orientation a printed sheet naturally gets when laid around a circle.
        """
        d = self.coil_center_to_tag
        out: dict[int, tuple[float, float, float]] = {}
        for shelf, (cx, cy) in self.coil_positions.items():
            out[shelf * 10 + 1] = (cx, cy + d, math.pi / 2)     # north
            out[shelf * 10 + 2] = (cx, cy - d, -math.pi / 2)    # south
            out[shelf * 10 + 3] = (cx - d, cy, math.pi)         # west
            out[shelf * 10 + 4] = (cx + d, cy, 0.0)             # east
        return out


def board_repo_b() -> BoardSpec:
    """The alternative spacing from repo B's yaml, for comparison runs."""
    return BoardSpec(coil_spacing=(0.470, 0.270))


# ---------------------------------------------------------------------------
# Wireless power transfer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WptLinkModel:
    """The project's own link-efficiency model, implemented as stated.

    POSTER:  ``U = k * sqrt(Q_tx * Q_rx)``  and  ``eta = U^2 / (1 + sqrt(1 + U^2))^2``
    with ``Q_tx = Q_rx = 20``, ``k_aligned = 0.50 -> eta = 81.9 %`` and
    ``k_misaligned = 0.12 -> eta = 44.4 %``.  Both anchor points reproduce to three
    significant figures from this formula, which is why it is used verbatim
    instead of a surrogate.

    The one thing the poster does *not* give is ``k`` as a function of offset -- it
    shows a curve, not an equation.  So exactly one free parameter is introduced
    here, ``misalign_offset``: the offset at which ``k`` has fallen to
    ``k_misaligned``.  ``k(d) = k_aligned * exp(-(d / d0)^2)`` with ``d0`` solved
    to hit that anchor.  The Gaussian form is a choice; the two endpoint values
    are not.  Treat the shape between the anchors as illustrative.

    Note on yaw: these are circular coaxial coils, so coupling is rotationally
    symmetric and ``eta`` depends on radial offset only.  The yaw tolerance in
    ``DockSpec`` is therefore *not* an electromagnetic requirement -- it exists so
    the robot ends a leg pointing correctly for the next one, and because a
    crooked park costs alignment margin on the following approach.  Conflating the
    two would be an easy and wrong simplification.
    """

    q_tx: float = 20.0
    q_rx: float = 20.0
    k_aligned: float = 0.50           # POSTER
    k_misaligned: float = 0.12        # POSTER
    misalign_offset: float = 0.020    # m, the single fitted parameter -- UNVERIFIED
    stable_offset: float = 0.010      # m, POSTER: "valid reception within +-1 cm"
    # eta required to declare a lock.  0.55 corresponds to about 17.6 mm radial, which
    # pairs with the relaxed 15 mm-per-axis box in DockSpec: the box is the coarse
    # gate, this is the one with physical meaning.  The poster's own +-1 cm edge sits
    # at eta = 0.75, which is what ``strict()`` uses.
    lock_efficiency: float = 0.55     # -

    @property
    def _d0(self) -> float:
        ratio = max(1e-6, min(0.999999, self.k_misaligned / self.k_aligned))
        return self.misalign_offset / math.sqrt(-math.log(ratio))

    def coupling_k(self, offset: float) -> float:
        d = abs(offset)
        return self.k_aligned * math.exp(-((d / self._d0) ** 2))

    def efficiency(self, offset: float) -> float:
        u = self.coupling_k(offset) * math.sqrt(self.q_tx * self.q_rx)
        return (u * u) / (1.0 + math.sqrt(1.0 + u * u)) ** 2

    @property
    def efficiency_aligned(self) -> float:
        return self.efficiency(0.0)

    def relative_efficiency(self, offset: float) -> float:
        """``eta / eta_aligned`` -- the quantity the poster's curve actually plots."""
        peak = self.efficiency_aligned
        return 0.0 if peak <= 0.0 else self.efficiency(offset) / peak


# ---------------------------------------------------------------------------
# Path following
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FollowGains:
    """Gains for the Lyapunov path-following law, in *spatial* units.

    The law linearises along a straight reference into

        e'' + 2*zeta*lam*e' + lam^2*e = 0      (derivative with respect to arc length)

    so with ``k_cross = lam^2`` and ``k_heading = 2*zeta*lam*v`` the cross-track
    error decays as ``exp(-lam * distance travelled)`` and the speed cancels out
    entirely.  ``decay_rate`` is therefore the inverse e-folding *distance* in
    1/m, and ``damping`` is the damping ratio.  Both are things you can check with
    a ruler, which is what makes the feasibility test in ``path_follow.py``
    possible.

    Why the docking value is so large, and how it was chosen: the *shortest* final
    approach on this board is the 0.255 m axis, and hand placement can leave a
    25 mm lateral error at the start of it.  The Monte Carlo showed ``lam = 12/m``
    needing 0.286 m for that case -- more than the leg has -- so the feasibility
    test correctly refused, retreated, ran out of board, and failed.  ``lam = 18/m``
    brings the same case to about 0.17 m and it succeeds outright with no retreat.

    The gain is therefore set by the geometry, not by taste, and the check that it
    is *achievable* is separate: at 18/m the yaw-rate demand for a 25 mm error at
    docking speed is ``lam^2 * v * e = 324 * 0.015 * 0.025 ~ 0.12 rad/s``, inside
    the cap below.  When the cap does bind -- which it does briefly at cruise speed
    with a large error -- ``path_follow.saturate`` scales ``v`` and ``w`` together, so
    the robot slows down and still drives the commanded arc rather than drifting
    off it.
    """

    decay_rate: float = 6.0      # 1/m
    damping: float = 1.0         # -, 1.0 = critically damped, no overshoot
    max_yaw_rate: float = 1.2    # rad/s, REPO (repo B max_angular_radps)

    @property
    def k_cross(self) -> float:
        return self.decay_rate * self.decay_rate

    def k_heading(self, speed: float) -> float:
        return 2.0 * self.damping * self.decay_rate * abs(speed)


@dataclass(frozen=True)
class SpeedProfile:
    """Speed shaping.  All limits come from the upstream configs where they exist."""

    cruise: float = 0.070                # m/s, REPO (repo B max_linear_mps)
    approach: float = 0.030              # m/s, REPO-ish (repo A shelf_entry_linear 0.025)
    dock: float = 0.015                  # m/s, REPO (repo A coil_max_linear)

    # DESIGN, and the first thing the Monte Carlo caught.  There is a floor below
    # which a wheel does not turn at all -- servo velocity quantisation plus
    # stiction -- so a controller that tapers its command towards zero as it
    # approaches the goal stops *short* of the goal and stays there.  The upstream
    # config's 0.004 m/s minimum is below that floor.  The fix is not a smaller
    # taper but a hard floor above the stiction limit: drive at 12 mm/s until the
    # goal is within one step, then stop dead.  At 30 Hz one step is 0.4 mm, and
    # with a 50 ms motor lag the coast afterwards is about 0.6 mm -- together an
    # order of magnitude inside the 10 mm requirement, so nothing is lost by
    # refusing to creep.
    creep: float = 0.012                 # m/s
    min_move: float = 0.012              # m/s, must stay above the actuation floor

    lateral_accel_budget: float = 0.20   # m/s^2, DESIGN, caps v on curvature
    heading_gate: float = math.radians(35.0)   # rad, above this stop and rotate in place
    heading_taper: float = math.radians(12.0)  # rad, start reducing v here
    spin_rate: float = 0.80              # rad/s, DESIGN, in-place turns between legs
    spin_rate_fine: float = 0.15         # rad/s, DESIGN, same stiction floor applies to spinning
    spin_fine_band: float = math.radians(4.0)


# ---------------------------------------------------------------------------
# Docking
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DockSpec:
    """Alignment requirement and the approach/retry geometry.

    The tolerances are stated in **millimetres and degrees**, which is the thing the
    upstream ``docs/calibration.md`` says it cannot do -- its thresholds are
    ``4 px / 4 px / 2.0 deg``, a quantity that depends on camera height and lens and
    is therefore not comparable between rigs.

    Preset: **single-shot**.  The defaults here trade tightness for landing in one
    pass instead of backing off and re-approaching.  What changed and why:

    * ``yaw_tol`` 2 deg -> 4 deg.  This is the one that mattered.  Every retreat
      observed in Isaac Sim was triggered by the *yaw* term of the feasibility test,
      not by position -- a 3 deg residual needs distance to unwind, and near the coil
      there is none left.  Position was never the problem: measured runs land at
      1.5-3.4 mm radial, a factor of three inside even the strict box.
    * ``pos_tol`` 10 mm -> 15 mm, with the efficiency gate doing the real work.
    * ``max_retries`` 4 -> 1, and ``retreat_shortfall`` added so a *marginal*
      shortfall is ridden out rather than turned into a 16 s round trip.

    Honest consequence: the poster's "valid reception within +-1 cm" corresponds to
    eta = 75 %, and this preset locks down to eta = 55 % (about 17.6 mm radial).  So a
    lock here means "aligned well enough to charge with reduced efficiency", not
    "inside the stated sweet spot".  ``strict()`` restores the tighter definition.
    """

    approach_distance: float = 0.200      # m, straight run before the coil, DESIGN
    approach_distance_max: float = 0.320  # m, cap on a retry request
    pos_tol: float = 0.015               # m, per axis
    yaw_tol: float = math.radians(4.0)   # rad
    hold_time: float = 1.0               # s, REPO (10 stable frames at 10 Hz)
    settle_time: float = 0.35            # s, wait after stopping before trusting the pose
    max_retries: int = 1
    retry_margin: float = 0.05           # m, extra room asked for on each retry
    feasibility_margin: float = 0.02     # m, safety added to the computed required distance
    # Only give up on an approach when it is short by *this much*, not merely short.
    # The feasibility estimate is computed from a live pose estimate that jitters by a
    # millimetre or two, so a threshold with no dead band turns estimator noise into
    # a manoeuvre.  Riding out a marginal shortfall costs nothing: the follower keeps
    # reducing the error the whole way in, and SETTLE/VERIFY is the real judge.
    retreat_shortfall: float = 0.030     # m
    trim_yaw_limit: float = math.radians(12.0)   # rad, above this a spin is not worth it
    approach_radius: float = 0.085       # m, REPO (coil_approach_radius_m)
    fine_align_radius: float = 0.005     # m, REPO (fine_align_radius_m)


# ---------------------------------------------------------------------------
# Odometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OdometrySpec:
    """Wheel-odometry error model.

    The upstream system had **no odometry at all** -- its only ROS interface is a
    ``/cmd_vel`` publisher, so position came from commanded velocity times time.
    Using ``/odom`` is the single largest change here, so it deserves an honest
    error model rather than a flattering one: the dominant terms are proportional
    to distance travelled, and the wheel-radius mismatch part is *fixed for the
    run*, so no amount of filtering averages it away.
    """

    speed_noise: float = 0.04         # -, 1-sigma fractional error on measured v
    yaw_noise: float = 0.05           # -, 1-sigma fractional error on measured w
    yaw_from_speed: float = 0.03      # rad/s per m/s, from unequal wheel radii
    floor_pos: float = 8.0e-5         # m per step, keeps the covariance from collapsing
    floor_yaw: float = 2.5e-4         # rad per step


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimSpec:
    physics_dt: float = 1.0 / 120.0   # s, 120 Hz keeps wheel contact stable at millimetre scale
    render_dt: float = 1.0 / 60.0     # s
    control_hz: float = 30.0          # Hz, DESIGN (upstream ran 10 Hz; sim can afford more)
    camera_hz: float = 10.0           # Hz, REPO -- the real sensing cadence, kept honest
    mission_timeout: float = 180.0    # s
    seed: int = 20260811


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    robot: RobotSpec = field(default_factory=RobotSpec)
    cameras: tuple[CameraSpec, ...] = field(default_factory=default_cameras)
    detection: DetectionSpec = field(default_factory=DetectionSpec)
    board: BoardSpec = field(default_factory=BoardSpec)
    wpt: WptLinkModel = field(default_factory=WptLinkModel)
    cruise: FollowGains = field(default_factory=lambda: FollowGains(decay_rate=6.0, max_yaw_rate=1.2))
    docking: FollowGains = field(default_factory=lambda: FollowGains(decay_rate=18.0, max_yaw_rate=0.35))
    speeds: SpeedProfile = field(default_factory=SpeedProfile)
    dock: DockSpec = field(default_factory=DockSpec)
    odometry: OdometrySpec = field(default_factory=OdometrySpec)
    sim: SimSpec = field(default_factory=SimSpec)

    def camera(self, name: str) -> CameraSpec:
        for c in self.cameras:
            if c.name == name:
                return c
        raise KeyError(name)


def with_camera_height(settings: Settings, lens_z: float) -> Settings:
    """Rebuild the camera specs at a measured lens height.

    Called once, after the TurtleBot3 asset's bounding box tells us where the custom
    plate actually sits.  The point is that the extrinsics the estimator believes and
    the plate on screen come from the *same* number: a camera height that disagrees
    with the model is an unbounded pose error and is invisible in the picture.
    """
    return replace(settings, cameras=default_cameras(lens_z=float(lens_z)))


def strict() -> Settings:
    """The tighter definition of alignment: the poster's +-1 cm sweet spot.

    ``pos_tol`` 10 mm, ``yaw_tol`` 2 deg, ``eta >= 75 %``, and the full retry budget.
    Measured 100 % over 300 randomised runs, but it does occasionally back off and
    re-approach -- which is the behaviour the single-shot default trades away.
    """
    base = Settings()
    return replace(
        base,
        dock=DockSpec(
            pos_tol=0.010,
            yaw_tol=math.radians(2.0),
            max_retries=4,
            retreat_shortfall=0.0,
            feasibility_margin=0.03,
        ),
        wpt=replace(base.wpt, lock_efficiency=0.75),
    )


DEFAULTS = Settings()
