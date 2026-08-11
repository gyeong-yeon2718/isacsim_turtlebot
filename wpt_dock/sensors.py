"""Wheel odometry.

The upstream system had **no odometry at all**: its entire ROS surface is one
``/cmd_vel`` publisher with zero subscriptions, so position came from commanded
velocity times elapsed time.  That is open-loop dead reckoning on a *command*,
which does not even know whether the wheels turned -- every slip, every stall
against the tape ridge, every millisecond of controller latency is a permanent
position error with nothing to correct it.  Reading ``/odom`` is the single
largest change in this port, and the reason the estimator has anything to
propagate between camera frames.

Modelling it honestly matters, so the error structure is copied from what
actually goes wrong on a small differential drive:

* **Proportional, not additive.**  Slip and wheel-radius error scale with
  distance travelled.  A fixed per-step variance would understate drift on a long
  leg and overstate it while parked.
* **Partly fixed for the run.**  Unequal wheel radii and a mis-measured track
  width are properties of the robot, not noise.  They are drawn once per episode.
  A filter cannot average them away, and pretending it can is how a simulation
  ends up reporting accuracy that hardware never reproduces.
* **Yaw contaminated by forward motion.**  A wheel-radius mismatch turns straight
  driving into a slow arc.  This is the dominant heading error on a real
  TurtleBot, and it is the term a naive model omits entirely.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import OdometrySpec


@dataclass
class OdometryReading:
    v: float          # m/s, measured body speed
    w: float          # rad/s, measured body yaw rate


class OdometryModel:
    """Turns true body motion into what the wheel encoders would report."""

    def __init__(self, spec: OdometrySpec, rng: np.random.Generator) -> None:
        self.spec = spec
        self.rng = rng
        # Fixed for the run: unequal wheel radii, track-width error, and the
        # straight-line-becomes-an-arc coupling they produce together.
        self.scale_bias = 1.0 + float(rng.normal(0.0, 0.5 * spec.speed_noise))
        self.yaw_scale_bias = 1.0 + float(rng.normal(0.0, 0.5 * spec.yaw_noise))
        self.yaw_coupling = float(rng.normal(0.0, spec.yaw_from_speed))

    def measure(self, v_true: float, w_true: float) -> OdometryReading:
        s = self.spec
        v = v_true * self.scale_bias
        w = w_true * self.yaw_scale_bias + self.yaw_coupling * v_true
        v += float(self.rng.normal(0.0, 0.5 * s.speed_noise * abs(v_true) + 5e-5))
        w += float(self.rng.normal(0.0, 0.5 * s.yaw_noise * abs(w_true) + 2e-4))
        return OdometryReading(v, w)

    def describe(self) -> str:
        return (
            f"odometry biases for this run: speed x{self.scale_bias:.4f}, "
            f"yaw x{self.yaw_scale_bias:.4f}, yaw-from-speed {self.yaw_coupling:+.4f} rad/s per m/s"
        )
