"""Link efficiency and the lock decision that drives the light.

Unlike a hand-waved "alignment score", the number the coil's brightness follows is
the project's **own** stated model, implemented verbatim (see
``config.WptLinkModel``):

    U   = k * sqrt(Q_tx * Q_rx)          with Q_tx = Q_rx = 20
    eta = U^2 / (1 + sqrt(1 + U^2))^2

Both anchor points on the poster reproduce from it to three significant figures
(``k = 0.50 -> 81.9 %``, ``k = 0.12 -> 44.4 %``), which is good evidence the
formula was transcribed correctly rather than merely plausibly.

Two honest caveats, stated here so they are not lost:

1. ``k`` as a function of offset is **not** given by the source -- only a curve is.
   The Gaussian interpolation between the two anchors carries one fitted
   parameter (``misalign_offset``).  Endpoint values: sourced.  Shape between
   them: illustrative.
2. These are circular coaxial coils, so coupling is rotationally symmetric and
   ``eta`` is a function of **radial offset only**.  The 2 deg yaw tolerance is
   therefore not an electromagnetic requirement.  It is a pose requirement: the
   robot has to leave a coil pointing correctly for the next leg, and a crooked
   park spends alignment margin on the following approach.  Folding yaw into an
   efficiency term would look tidier and would be wrong.

The lock decision is deliberately stricter than "the score looks high":

* each error component must be inside its own tolerance -- a scalar score can be
  high with one axis badly out, and a scalar gate would let that through;
* the efficiency must clear its threshold, which couples the axes;
* and the whole condition must hold continuously for ``hold_time``.  Without the
  dwell, a robot still rolling reports a lock during the single frame it sweeps
  through the tolerance box.  The upstream config requires 10 consecutive frames
  for the same reason; that is 1.0 s at their 10 Hz, and it is kept.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import DockSpec, WptLinkModel


@dataclass
class LinkState:
    """Everything worth showing about the current alignment."""

    radial_offset: float     # m, Rx coil centre to Tx coil centre
    efficiency: float        # -, absolute eta from the poster's model
    relative: float          # -, eta / eta_aligned, the quantity the poster plots
    coupling_k: float        # -, the k that produced it
    in_tolerance: bool
    locked: bool
    held: float              # s, how long the full condition has held
    worst_axis: str
    margin: float            # normalised worst-axis slack; negative = outside


class LinkMonitor:
    """Debounced charging verdict for one target coil.

    ``margin`` is the *normalised* worst-axis slack, so one number says how close
    to failing the lock is without the reader having to mentally compare
    millimetres against degrees.
    """

    def __init__(self, dock: DockSpec, wpt: WptLinkModel) -> None:
        self.dock = dock
        self.wpt = wpt
        self.held = 0.0
        self.ever_locked = False
        self.best_offset = math.inf

    def reset(self) -> None:
        self.held = 0.0

    def evaluate(self, ex: float, ey: float, eyaw: float) -> LinkState:
        """Stateless scoring -- no dwell bookkeeping, safe to call for display."""
        d = math.hypot(ex, ey)
        eta = self.wpt.efficiency(d)
        slack = {
            "longitudinal": (self.dock.pos_tol - abs(ex)) / self.dock.pos_tol,
            "lateral": (self.dock.pos_tol - abs(ey)) / self.dock.pos_tol,
            "yaw": (self.dock.yaw_tol - abs(eyaw)) / self.dock.yaw_tol,
        }
        worst = min(slack, key=slack.get)
        margin = slack[worst]
        in_tol = margin >= 0.0
        return LinkState(
            radial_offset=d,
            efficiency=eta,
            relative=self.wpt.relative_efficiency(d),
            coupling_k=self.wpt.coupling_k(d),
            in_tolerance=in_tol,
            locked=False,
            held=self.held,
            worst_axis=worst,
            margin=margin,
        )

    def update(self, dt: float, ex: float, ey: float, eyaw: float) -> LinkState:
        state = self.evaluate(ex, ey, eyaw)
        self.best_offset = min(self.best_offset, state.radial_offset)

        condition = state.in_tolerance and state.efficiency >= self.wpt.lock_efficiency
        if condition:
            self.held += max(0.0, dt)
        else:
            self.held = 0.0

        locked = self.held >= self.dock.hold_time
        if locked:
            self.ever_locked = True

        return LinkState(
            radial_offset=state.radial_offset,
            efficiency=state.efficiency,
            relative=state.relative,
            coupling_k=state.coupling_k,
            in_tolerance=state.in_tolerance,
            locked=locked,
            held=self.held,
            worst_axis=state.worst_axis,
            margin=state.margin,
        )


# ---------------------------------------------------------------------------
# Visual mapping
# ---------------------------------------------------------------------------


def glow_colour(relative: float, locked: bool) -> tuple[float, float, float]:
    """Red -> amber -> green ramp on relative efficiency, with a step change at lock.

    Two channels of information on one object: hue reports how good the alignment
    is *right now*, and the jump to saturated green reports that the decision has
    been **made** -- tolerances met and held for the full dwell.  A continuous ramp
    alone cannot show the second thing, and the dwell is the part a viewer most
    needs to see, because it is the difference between passing through tolerance
    and actually parking inside it.
    """
    e = max(0.0, min(1.0, relative))
    if locked:
        return (0.10, 1.00, 0.35)
    if e < 0.5:
        t = e / 0.5
        return (1.00, 0.10 + 0.55 * t, 0.05)
    t = (e - 0.5) / 0.5
    return (1.00 - 0.75 * t, 0.65 + 0.20 * t, 0.05 + 0.10 * t)


def glow_intensity(relative: float, locked: bool, *, standby: float, peak: float) -> float:
    """Light intensity for the target coil.

    Scaled by relative efficiency rather than absolute, because absolute ``eta``
    never falls below about 0.44 in the modelled range -- a light driven by it
    would be more than half lit even when badly misaligned, which reads as "nearly
    working" when it is not.  Relative efficiency spans the full 0..1 and is also
    the quantity the source curve plots.
    """
    e = max(0.0, min(1.0, relative))
    base = standby + (peak - standby) * (e ** 2)
    return peak * 1.35 if locked else base
