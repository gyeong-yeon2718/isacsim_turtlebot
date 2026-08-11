"""Fusing ``/odom`` with AprilTag registrations into one pose estimate.

This is the "``/odom`` to get near the coil, tags for the final alignment" split,
made concrete.  It is not a mode switch: odometry always propagates and tags
always correct when they are visible, so there is no discontinuity at the
handover and no moment where the robot is flying blind because a threshold has
not tripped yet.

The pipeline per camera frame:

    corner pixels  ->  back-project onto the board plane      (apriltag.py)
                   ->  weighted rigid 2D fit, with covariance (registration.py)
                   ->  robot pose in the board frame          (registration.py)
                   ->  chi-square-gated EKF update            (registration.py)

Every tag with a decoded id contributes, not just the target coil's four.  That is
a free improvement over the upstream design, which only ever looked at the tags
belonging to the coil it was working on: on a diagonal transit the robot passes
directly over an intermediate coil, and its four tags are the best fix available
anywhere on the board at that moment.  Ignoring them would be throwing away the
one measurement that arrives exactly when dead reckoning has drifted furthest.

The controller never sees ground truth.  Only this estimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .apriltag import (
    CameraModel,
    GroundCorrespondences,
    TagDetection,
    TagModel,
    to_ground_correspondences,
)
from .config import Settings
from .geometry import Pose, wrap_angle
from .registration import Ekf2D, RegistrationResult, fit_rigid_2d, robot_pose_from_registration
from .sensors import OdometryReading


@dataclass
class FixReport:
    """What happened on one attempted tag update."""

    accepted: bool
    reason: str
    n_tags: int = 0
    n_points: int = 0
    residual_rms: float = math.inf
    pose: Pose | None = None
    sigma_pos: float = math.inf
    sigma_yaw: float = math.inf
    tag_ids: tuple[int, ...] = field(default_factory=tuple)


class PoseEstimator:
    def __init__(
        self,
        settings: Settings,
        tags: dict[int, TagModel],
        cameras_nominal: dict[str, CameraModel],
        initial_pose: Pose,
        *,
        initial_position_sigma: float = 0.030,
        initial_yaw_sigma: float = math.radians(5.0),
    ) -> None:
        self.s = settings
        self.tags = tags
        self.cameras = cameras_nominal
        self.ekf = Ekf2D(
            initial_pose,
            initial_position_sigma,
            initial_yaw_sigma,
            floor_position_sigma=settings.detection.systematic_pos_sigma,
            floor_yaw_sigma=settings.detection.systematic_yaw_sigma,
        )
        # Pose-level covariance of the calibration error, added to every fix.  It
        # belongs here rather than in the per-corner weights because a mount or
        # intrinsic error is common to all corners a given camera contributes: it
        # therefore produces no residual for the fit to notice, and no amount of
        # per-point weighting can represent it.
        self._systematic = np.diag(
            [
                settings.detection.systematic_pos_sigma ** 2,
                settings.detection.systematic_pos_sigma ** 2,
                settings.detection.systematic_yaw_sigma ** 2,
            ]
        )
        self.fixes_accepted = 0
        self.fixes_rejected = 0
        self.last_fix: FixReport | None = None
        self.frames_since_fix = 0

    # -- accessors -----------------------------------------------------------

    @property
    def pose(self) -> Pose:
        return self.ekf.pose

    @property
    def position_sigma(self) -> float:
        return self.ekf.position_sigma

    @property
    def yaw_sigma(self) -> float:
        return self.ekf.yaw_sigma

    # -- steps ---------------------------------------------------------------

    def predict(self, odom: OdometryReading, dt: float) -> None:
        o = self.s.odometry
        self.ekf.predict(
            odom.v,
            odom.w,
            dt,
            speed_noise=o.speed_noise,
            yaw_noise=o.yaw_noise,
            yaw_from_speed=o.yaw_from_speed,
            floor_pos=o.floor_pos,
            floor_yaw=o.floor_yaw,
        )
        self.frames_since_fix += 1

    def update(self, detections: list[TagDetection]) -> FixReport:
        d = self.s.detection
        if len(detections) < d.min_tags_for_fix:
            report = FixReport(False, "not enough tags", n_tags=len(detections))
            self.last_fix = report
            return report

        corr: GroundCorrespondences = to_ground_correspondences(
            detections, self.cameras, self.tags, d
        )
        if corr.n < 2:
            report = FixReport(False, "not enough usable corners", n_tags=len(detections), n_points=corr.n)
            self.last_fix = report
            return report

        fit: RegistrationResult = fit_rigid_2d(corr.model, corr.observed, weights=corr.weights)
        if not fit.ok:
            self.fixes_rejected += 1
            report = FixReport(
                False, f"fit rejected: {fit.reason}", n_tags=len(detections),
                n_points=corr.n, residual_rms=fit.residual_rms,
                tag_ids=tuple(sorted(set(corr.tag_ids))),
            )
            self.last_fix = report
            return report

        pose, cov = robot_pose_from_registration(fit)
        cov = cov + self._systematic
        sigma_pos = math.sqrt(max(0.0, 0.5 * (cov[0, 0] + cov[1, 1])))
        sigma_yaw = math.sqrt(max(0.0, cov[2, 2]))

        accepted = self.ekf.update_pose(pose, cov)
        if accepted:
            self.fixes_accepted += 1
            self.frames_since_fix = 0
        else:
            self.fixes_rejected += 1

        report = FixReport(
            accepted=accepted,
            reason="ok" if accepted else "chi-square gate rejected the fix",
            n_tags=len(detections),
            n_points=corr.n,
            residual_rms=fit.residual_rms,
            pose=pose,
            sigma_pos=sigma_pos,
            sigma_yaw=sigma_yaw,
            tag_ids=tuple(sorted(set(corr.tag_ids))),
        )
        self.last_fix = report
        return report

    # -- reporting -----------------------------------------------------------

    def summary(self) -> str:
        return (
            f"fixes accepted {self.fixes_accepted}, rejected {self.fixes_rejected}; "
            f"pose sigma {self.position_sigma * 1000:.2f} mm / "
            f"{math.degrees(self.yaw_sigma):.3f} deg"
        )

    def error_against(self, truth: Pose) -> tuple[float, float, float]:
        """Estimation error, for logging only.  Never fed back into the estimate."""
        p = self.pose
        return (p[0] - truth[0], p[1] - truth[1], wrap_angle(p[2] - truth[2]))
