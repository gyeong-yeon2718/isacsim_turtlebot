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
    """Drives the board's lights from the *true* link state."""

    def __init__(self, scene: BoardScene) -> None:
        self.scene = scene
        self.target: int | None = None
        self._last_target: int | None = None

    def set_target(self, coil: int) -> None:
        if coil == self._last_target:
            return
        self.target = coil
        self._last_target = coil
        for n, visual in self.scene.coils.items():
            set_display_colour(visual.core_prim, (0.15, 0.45, 0.95) if n == coil else _FERRITE)
            if n != coil:
                self._darken(n)

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

        rel = min(1.0, max(0.0, link.relative))
        visual.lamp_intensity.Set(float(intensity))
        visual.lamp_colour.Set(tuple(float(c) for c in colour))
        set_display_colour(visual.face_prim, colour)
        if visual.material is not None:
            # Squared, so the last few millimetres of alignment produce most of the
            # visible change.  A linear ramp spends its brightness on the part of the
            # approach where nothing is decided yet.
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
