"""Turn alignment quality into something a person can see across the room.

Three channels, each carrying different information, because one channel cannot
carry all of it:

* **The gauge ring** around the target coil fills with *relative* link efficiency.
  Continuous, so you can watch the approach converge.
* **The coil face and its lamp** light up with the same quantity, and step to a
  saturated green the moment the lock decision is taken -- tolerances met *and* held
  for the full dwell.  A ramp alone cannot show a decision; that step is the part
  worth watching for, because it is the difference between sweeping through the
  tolerance box and parking inside it.
* **The core dot** marks which coil is the target, before anything lights up.

Why relative efficiency and not absolute: the poster's model never drops below
about 44 % in the modelled range, so a light driven by absolute ``eta`` would sit
more than half lit while badly misaligned -- reading as "nearly working" when it is
not.  Relative efficiency spans the full 0..1, and it is also the quantity the
poster's own curve plots.

One deliberately alarming case: if the robot declares a lock but the *true*
efficiency is below threshold, the coil turns magenta instead of green.  That means
the estimator is wrong and the robot does not know it -- the single most dangerous
failure this system can have, and the only honest thing to do with it is make it
impossible to miss rather than let a green light hide it.
"""

from __future__ import annotations

import math

from ..coupling import LinkState, glow_colour, glow_intensity
from .board_build import GAUGE_SEGMENTS, BoardScene
from .usd_helpers import set_display_colour

_DARK_SEG = (0.20, 0.20, 0.22)
_COIL_IDLE = (0.72, 0.45, 0.20)
_FERRITE = (0.20, 0.19, 0.18)
_WARN = (1.0, 0.15, 0.85)

STANDBY_INTENSITY = 900.0
PEAK_INTENSITY = 320000.0
EMISSIVE_BASE = 4000.0
EMISSIVE_PEAK = 160000.0
BEACON_PEAK = 90000.0


class CoilGlow:
    """Drives the board's lights from the *true* link state.

    **Every** coil is lit from its own coupling, not just the mission's target.  That is
    both what a bench of energised pads actually does and what makes the run readable: the
    coil the robot starts on glows before it moves, a coil it drives across on a diagonal
    route lights up as it passes and fades as it leaves, and the destination comes up as it
    arrives.  Lighting only the target made the other three look broken.

    Note for fidelity: on the real rig a relay closes for one selected transmitter at a
    time, so a hardware-faithful scene would light exactly one.  This is the "all pads live"
    configuration, which is the more informative of the two -- and the target is still
    distinguished, by its core marker and by the step change at lock.
    """

    def __init__(self, scene: BoardScene, wpt=None, board=None) -> None:
        self.scene = scene
        self.wpt = wpt
        self.board = board
        self.target: int | None = None
        self._last_target: int | None = None

    def set_target(self, coil: int) -> None:
        if coil == self._last_target:
            return
        self.target = coil
        self._last_target = coil
        for n, visual in self.scene.coils.items():
            set_display_colour(visual.core_prim, (0.15, 0.45, 0.95) if n == coil else _FERRITE)

    def update_all(
        self,
        rx_xy: tuple[float, float],
        target: int,
        target_link: LinkState,
        *,
        energised: bool,
        believed_locked: bool,
        time: float = 0.0,
    ) -> None:
        """Light every coil from the receiver's distance to it.

        ``rx_xy`` is the receiver coil's true position on the board, so each transmitter is
        driven by the coupling it would really deliver.  Non-target coils get the same colour
        ramp but never the lock step change -- that decision belongs to the mission.
        """
        self.set_target(target)
        self.update(target, target_link, energised=energised,
                    believed_locked=believed_locked, time=time)
        if self.wpt is None or self.board is None:
            return
        for n, (cx, cy) in self.board.coil_positions.items():
            if n == target:
                continue
            offset = math.hypot(rx_xy[0] - cx, rx_xy[1] - cy)
            rel = self.wpt.relative_efficiency(offset)
            if rel <= 0.02:
                self._darken(n)
                continue
            self._paint(self.scene.coils[n], glow_colour(rel, False), rel,
                        glow_intensity(rel, False, standby=STANDBY_INTENSITY,
                                      peak=PEAK_INTENSITY))

    def _darken(self, coil: int) -> None:
        visual = self.scene.coils[coil]
        visual.lamp_intensity.Set(0.0)
        set_display_colour(visual.face_prim, _COIL_IDLE)
        if visual.material is not None:
            visual.material.set_emission(_COIL_IDLE, 0.0)
        if visual.beacon_material is not None:
            visual.beacon_material.set_emission((0.06, 0.06, 0.07), 0.0)
        if visual.beacon_prim is not None:
            set_display_colour(visual.beacon_prim, (0.09, 0.09, 0.11))
        for seg in visual.gauge_prims:
            set_display_colour(seg, _DARK_SEG)

    def update(
        self,
        target: int,
        link: LinkState,
        *,
        energised: bool,
        believed_locked: bool,
        time: float = 0.0,
    ) -> None:
        self.set_target(target)
        visual = self.scene.coils[target]

        truly_locked = energised and link.in_tolerance
        disagreement = believed_locked and not link.in_tolerance

        if disagreement:
            colour = _WARN
            # Fast blink: a steady magenta could be mistaken for a design choice.
            intensity = PEAK_INTENSITY * (0.35 + 0.65 * (0.5 + 0.5 * math.sin(time * 22.0)))
        else:
            colour = glow_colour(link.relative, truly_locked)
            intensity = glow_intensity(
                link.relative, truly_locked, standby=STANDBY_INTENSITY, peak=PEAK_INTENSITY
            )
            if truly_locked:
                # A slow throb once charging, so a locked coil reads as *active*
                # rather than as a static prop.
                intensity *= 0.80 + 0.20 * (0.5 + 0.5 * math.sin(time * 6.0))

        self._paint(visual, colour, min(1.0, max(0.0, link.relative)), intensity)

    def _paint(self, visual, colour, rel: float, intensity: float) -> None:
        visual.lamp_intensity.Set(float(intensity))
        visual.lamp_colour.Set(tuple(float(c) for c in colour))
        set_display_colour(visual.face_prim, colour)
        if visual.material is not None:
            # Squared, so the last few millimetres of alignment produce most of the visible
            # change.  A linear ramp spends its brightness on the part of the approach where
            # nothing is decided yet.
            visual.material.set_emission(colour, EMISSIVE_BASE + EMISSIVE_PEAK * rel * rel)

        if visual.beacon_material is not None:
            visual.beacon_material.set_emission(colour, BEACON_PEAK * rel * rel)
        if visual.beacon_prim is not None:
            set_display_colour(visual.beacon_prim, colour if rel > 0.05 else (0.09, 0.09, 0.11))

        filled = int(round(rel * GAUGE_SEGMENTS))
        for i, seg in enumerate(visual.gauge_prims):
            set_display_colour(seg, colour if i < filled else _DARK_SEG)

    def all_off(self) -> None:
        for n in self.scene.coils:
            self._darken(n)
