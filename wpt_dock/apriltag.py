"""Pinhole cameras, AprilTag geometry, and the seam where a real detector plugs in.

The porting seam
----------------
:class:`TagDetection` is exactly what ``pupil_apriltags`` (or ``cv2.aruco``) hands
you on real hardware: a tag id, which camera saw it, and four corner positions in
pixels.  Everything *downstream* of that struct in this package -- back-projection
to the board plane, rigid registration, covariance, the filter, the controller --
is the code that would run unchanged on the robot.  Only :class:`TagDetectorSim`,
which manufactures the struct, is simulation-only.

That is the whole reason this module does not rasterise images and run a real
decoder.  Doing so would consume most of an 8 GB GPU for three camera streams and
would add a decoder's failure modes to a study that is not about decoders, while
changing *none* of the code that has to transfer to hardware.  What does matter --
projective geometry, field-of-view and obliqueness limits, pixel noise, and
calibration error that cannot be averaged away -- is modelled explicitly below.

Why back-project to the plane instead of solving a full PnP
-----------------------------------------------------------
Every tag lies flat on the plywood.  With the camera pose on the robot known, a
pixel therefore determines a unique point on that plane -- a ray/plane
intersection, no iteration and no pose ambiguity.  Planar PnP from four coplanar
points has a well-known two-fold ambiguity that has to be resolved by
reprojection error, and it estimates six numbers when the task only has three.
Back-projecting the corners turns the problem into 2D point-set registration,
which has a closed-form optimum (see ``registration.fit_rigid_2d``) and a
covariance you can write down.

Camera frame convention
-----------------------
OpenCV: the camera looks along its own **+Z**, image **+X** is right, image **+Y**
is down.  The mount is described by a yaw about the body +Z followed by a pitch,
with ``pitch = -90 deg`` pointing straight at the floor.

Not modelled, and stated so it is not mistaken for modelled: lens distortion.
The upstream project had no calibration file at all, so a radial-distortion term
here would be a fiction on top of a fiction.  On real hardware, undistort before
constructing :class:`TagDetection`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import BoardSpec, CameraSpec, DetectionSpec
from .geometry import Pose, rot2

# Camera axes expressed in the body frame for a camera pointing straight forward:
# camera +X -> body -Y (right), camera +Y -> body -Z (down), camera +Z -> body +X.
_R_BODY_CAM0 = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
)


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------


@dataclass
class CameraModel:
    """A camera bolted to the robot, with the intrinsics the *estimator* believes.

    ``bias_*`` fields hold the difference between belief and reality.  They are
    zero for the nominal model the estimator uses and non-zero for the twin the
    simulator projects through, which is precisely how calibration error behaves:
    fixed for the run, invisible to the filter, and not reducible by averaging
    frames.
    """

    spec: CameraSpec
    bias_cx: float = 0.0
    bias_cy: float = 0.0
    bias_focal: float = 1.0        # multiplicative
    bias_pitch: float = 0.0        # rad
    bias_yaw: float = 0.0          # rad
    bias_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)   # m

    def __post_init__(self) -> None:
        yaw = self.spec.yaw + self.bias_yaw
        pitch = self.spec.pitch + self.bias_pitch
        self._r_body_cam = _rot_z(yaw) @ _rot_y(-pitch) @ _R_BODY_CAM0
        self._t_body_cam = np.array(self.spec.position, dtype=float) + np.array(self.bias_pos, dtype=float)
        self._fx = self.spec.fx * self.bias_focal
        self._fy = self.spec.fy * self.bias_focal
        self._cx = self.spec.cx + self.bias_cx
        self._cy = self.spec.cy + self.bias_cy

    # -- geometry ------------------------------------------------------------

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def position_body(self) -> np.ndarray:
        return self._t_body_cam

    @property
    def optical_axis_body(self) -> np.ndarray:
        return self._r_body_cam @ np.array([0.0, 0.0, 1.0])

    # -- forward projection --------------------------------------------------

    def project(self, p_body: np.ndarray) -> tuple[float, float, float] | None:
        """Body-frame 3D point -> (u, v, depth).  ``None`` if behind the camera."""
        rel = np.asarray(p_body, dtype=float) - self._t_body_cam
        p_cam = self._r_body_cam.T @ rel
        z = float(p_cam[2])
        if z <= 1e-6:
            return None
        return (
            self._fx * float(p_cam[0]) / z + self._cx,
            self._fy * float(p_cam[1]) / z + self._cy,
            z,
        )

    def in_image(self, u: float, v: float, margin: float = 0.0) -> bool:
        return (
            margin <= u <= self.spec.width - 1.0 - margin
            and margin <= v <= self.spec.height - 1.0 - margin
        )

    # -- inverse projection onto the running surface -------------------------

    def backproject_to_ground(self, u: float, v: float, plane_z: float = 0.0) -> np.ndarray | None:
        """Pixel -> the point on ``z = plane_z`` (body frame) that projects to it.

        Returns ``None`` when the ray does not descend towards the plane, which is
        the correct answer rather than an error: a pixel above the horizon simply
        has no ground point.
        """
        d_cam = np.array([(u - self._cx) / self._fx, (v - self._cy) / self._fy, 1.0])
        d_body = self._r_body_cam @ d_cam
        if d_body[2] >= -1e-9:
            return None
        t = (plane_z - self._t_body_cam[2]) / d_body[2]
        if t <= 0.0:
            return None
        p = self._t_body_cam + t * d_body
        return p[:2]

    def ground_point_sigma(self, u: float, v: float, sigma_px: float, plane_z: float = 0.0) -> float:
        """Isotropic 1-sigma of a back-projected corner, in metres.

        Obtained from the numeric Jacobian of the back-projection rather than the
        usual ``sigma_px * range / f`` rule of thumb, because that rule ignores
        obliqueness -- and obliqueness is exactly what makes the front camera's
        far corners much worse than the bottom cameras' near ones.  Getting this
        wrong would not bias the fit, but it would mis-weight it, and the whole
        point of weighting is that a 5 mm corner and a 0.5 mm corner should not
        vote equally.

        Returned as the isotropic equivalent of the 2x2 corner covariance:
        ``sigma^2 = 0.5 * trace(J J^T) * sigma_px^2``.
        """
        h = 0.5
        p0 = self.backproject_to_ground(u, v, plane_z)
        if p0 is None:
            return math.inf
        pu1 = self.backproject_to_ground(u + h, v, plane_z)
        pu0 = self.backproject_to_ground(u - h, v, plane_z)
        pv1 = self.backproject_to_ground(u, v + h, plane_z)
        pv0 = self.backproject_to_ground(u, v - h, plane_z)
        if pu1 is None or pu0 is None or pv1 is None or pv0 is None:
            return math.inf
        j_u = (pu1 - pu0) / (2.0 * h)
        j_v = (pv1 - pv0) / (2.0 * h)
        var = 0.5 * (float(j_u @ j_u) + float(j_v @ j_v)) * sigma_px * sigma_px
        return math.sqrt(max(var, 1e-18))


def nominal_cameras(specs: tuple[CameraSpec, ...]) -> dict[str, CameraModel]:
    """The models the estimator uses: it believes the mount is exactly as designed."""
    return {s.name: CameraModel(s) for s in specs}


def perturbed_cameras(
    specs: tuple[CameraSpec, ...],
    detection: DetectionSpec,
    rng: np.random.Generator,
    *,
    mount_angle_sigma: float = math.radians(0.30),
    mount_pos_sigma: float = 0.0010,
    focal_sigma: float = 0.004,
) -> dict[str, CameraModel]:
    """The models reality uses.  Draw once per run, never redrawn.

    These three defaults are the accuracy floor of the whole system, so they are
    worth reading as a claim: a mount angle known to 0.3 deg, a lens position
    known to 1 mm, and a focal length known to 0.4 %.  For a camera 0.115 m above
    the surface, the 0.3 deg term alone puts about 0.6 mm of unremovable bias into
    every back-projected point.  That is why the 10 mm requirement is achievable
    and a 1 mm one -- which the upstream calibration notes correctly say is not
    reachable by camera alone -- is not.
    """
    out: dict[str, CameraModel] = {}
    for s in specs:
        out[s.name] = CameraModel(
            s,
            bias_cx=float(rng.normal(0.0, detection.intrinsic_bias)),
            bias_cy=float(rng.normal(0.0, detection.intrinsic_bias)),
            bias_focal=1.0 + float(rng.normal(0.0, focal_sigma)),
            bias_pitch=float(rng.normal(0.0, mount_angle_sigma)),
            bias_yaw=float(rng.normal(0.0, mount_angle_sigma)),
            bias_pos=tuple(float(x) for x in rng.normal(0.0, mount_pos_sigma, size=3)),
        )
    return out


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TagModel:
    """One printed tag36h11 lying flat on the board.

    ``corners`` are ordered counter-clockwise starting from the tag frame's
    ``(-h, -h)`` corner.  A real detector returns its corners in a fixed order
    relative to the decoded tag frame, so correspondence between model and
    observation is by index -- which is why no ICP-style association search is
    needed anywhere in this package.
    """

    tag_id: int
    center: tuple[float, float]
    yaw: float
    size: float

    @property
    def shelf(self) -> int:
        return self.tag_id // 10

    @property
    def position_name(self) -> str:
        return {1: "north", 2: "south", 3: "west", 4: "east"}.get(self.tag_id % 10, "?")

    def corners_board(self) -> np.ndarray:
        h = 0.5 * self.size
        local = np.array([[-h, -h], [h, -h], [h, h], [-h, h]])
        return local @ rot2(self.yaw).T + np.array(self.center)


def build_tags(board: BoardSpec) -> dict[int, TagModel]:
    """All sixteen coil tags: ``id = shelf * 10 + {north 1, south 2, west 3, east 4}``."""
    return {
        tid: TagModel(tid, (x, y), yaw, board.tag_size)
        for tid, (x, y, yaw) in board.tag_positions().items()
    }


@dataclass
class TagDetection:
    """**The porting seam.**  Identical in shape to a real AprilTag detector's output."""

    tag_id: int
    camera: str
    corners_px: np.ndarray          # (4, 2), same order as TagModel.corners_board()
    decision_margin: float = 1.0    # detector confidence; only used for reporting here


class TagDetectorSim:
    """Manufactures :class:`TagDetection` from the true robot pose.

    Gating, in the order a real detector fails:

    1. every corner must project in front of the camera and land inside the image
       (with a margin, because a tag touching the border does not decode);
    2. the apparent edge length must exceed ``min_tag_pixels``, since a tag too
       small in the image cannot carry its payload bits;
    3. the view must not be more oblique than ``max_view_angle`` off the surface
       normal;
    4. a random ``dropout`` fraction is lost anyway, which is what motion blur and
       specular glare on tape do in practice.
    """

    def __init__(
        self,
        cameras_true: dict[str, CameraModel],
        tags: dict[int, TagModel],
        detection: DetectionSpec,
        rng: np.random.Generator,
    ) -> None:
        self.cameras = cameras_true
        self.tags = tags
        self.detection = detection
        self.rng = rng

    def detect(self, true_pose: Pose) -> list[TagDetection]:
        out: list[TagDetection] = []
        d = self.detection
        r_wb = rot2(true_pose[2])
        origin = np.array([true_pose[0], true_pose[1]])

        for tag in self.tags.values():
            corners_board = tag.corners_board()
            # Board -> body: rotate by -yaw about the robot origin.
            corners_body2 = (corners_board - origin) @ r_wb
            corners_body = np.column_stack([corners_body2, np.zeros(len(corners_body2))])

            for cam in self.cameras.values():
                px = np.zeros((4, 2))
                ok = True
                for i in range(4):
                    proj = cam.project(corners_body[i])
                    if proj is None:
                        ok = False
                        break
                    u, v, _ = proj
                    if not cam.in_image(u, v, d.margin_px):
                        ok = False
                        break
                    px[i] = (u, v)
                if not ok:
                    continue

                edges = [float(np.linalg.norm(px[(i + 1) % 4] - px[i])) for i in range(4)]
                if min(edges) < d.min_tag_pixels:
                    continue

                # Obliqueness: angle between the surface normal and the direction
                # from the tag centre to the lens, both in the body frame.
                cam_pos = cam.position_body
                centre_body = corners_body.mean(axis=0)
                to_cam = cam_pos - centre_body
                norm = float(np.linalg.norm(to_cam))
                if norm < 1e-9:
                    continue
                incidence = math.acos(max(-1.0, min(1.0, float(to_cam[2]) / norm)))
                if incidence > d.max_view_angle:
                    continue

                if self.rng.random() < d.dropout:
                    continue

                noisy = px + self.rng.normal(0.0, d.corner_sigma, size=(4, 2))
                out.append(TagDetection(tag.tag_id, cam.name, noisy, decision_margin=min(edges)))
        return out


# ---------------------------------------------------------------------------
# Detections -> weighted planar correspondences
# ---------------------------------------------------------------------------


@dataclass
class GroundCorrespondences:
    """Model points (board frame) paired with observations (robot frame)."""

    model: np.ndarray            # (n, 2)
    observed: np.ndarray         # (n, 2)
    sigmas: np.ndarray           # (n,), 1-sigma in metres per point
    tag_ids: tuple[int, ...]
    cameras: tuple[str, ...]

    @property
    def n(self) -> int:
        return len(self.model)

    @property
    def weights(self) -> np.ndarray:
        return 1.0 / np.maximum(self.sigmas, 1e-6) ** 2


def to_ground_correspondences(
    detections: list[TagDetection],
    cameras_nominal: dict[str, CameraModel],
    tags: dict[int, TagModel],
    detection: DetectionSpec,
) -> GroundCorrespondences:
    """Turn detector output into the point pairs the registration consumes.

    Uses the **nominal** camera models, i.e. what the robot believes about its own
    mounts.  That is the point: any mismatch with reality shows up as registration
    residual and pose bias, exactly as it would on hardware, instead of being
    quietly cancelled by using the same numbers on both sides.
    """
    model: list[np.ndarray] = []
    observed: list[np.ndarray] = []
    sigmas: list[float] = []
    ids: list[int] = []
    cams: list[str] = []

    for det in detections:
        tag = tags.get(det.tag_id)
        cam = cameras_nominal.get(det.camera)
        if tag is None or cam is None:
            continue
        corners_board = tag.corners_board()
        for i in range(4):
            u, v = float(det.corners_px[i, 0]), float(det.corners_px[i, 1])
            ground = cam.backproject_to_ground(u, v)
            if ground is None:
                continue
            sigma = cam.ground_point_sigma(u, v, detection.corner_sigma)
            if not math.isfinite(sigma):
                continue
            model.append(corners_board[i])
            observed.append(ground)
            sigmas.append(sigma)
            ids.append(det.tag_id)
            cams.append(det.camera)

    if not model:
        empty = np.zeros((0, 2))
        return GroundCorrespondences(empty, empty, np.zeros(0), tuple(), tuple())
    return GroundCorrespondences(
        np.asarray(model), np.asarray(observed), np.asarray(sigmas), tuple(ids), tuple(cams)
    )


def visible_tag_report(detections: list[TagDetection]) -> str:
    by_cam: dict[str, list[int]] = {}
    for d in detections:
        by_cam.setdefault(d.camera, []).append(d.tag_id)
    if not by_cam:
        return "no tags"
    return ", ".join(f"{cam}:{sorted(ids)}" for cam, ids in sorted(by_cam.items()))

