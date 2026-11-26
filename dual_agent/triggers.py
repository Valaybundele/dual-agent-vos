"""Streaming eff-IoU variance trigger.

Fires when the rolling standard deviation of SAM 3's eff-IoU over the previous
`w` frames exceeds `thr` for `persist` consecutive frames, with at least
`min_spacing` frames between fires. NaN eff-IoU frames are skipped (not added
to the window).
"""

import math
from typing import List


class EffIouVarTrigger:
    def __init__(self, w: int = 4, thr: float = 0.15, persist: int = 2, min_spacing: int = 8):
        self.w = max(2, w)
        self.thr = thr
        self.persist = max(1, persist)
        self.min_spacing = max(1, min_spacing)
        self.last_fire_t: int = -10**9
        self._t = -1
        self._buf: List[float] = []
        self._consec = 0

    def step(self, eff_iou: float) -> bool:
        """Advance one frame; return True if the trigger fires on this frame."""
        self._t += 1
        raw = self._update(eff_iou)
        if raw and (self._t - self.last_fire_t) >= self.min_spacing:
            self.last_fire_t = self._t
            return True
        return False

    def _update(self, eff) -> bool:
        if eff is None or (isinstance(eff, float) and math.isnan(eff)):
            return self._consec >= self.persist
        # Std over the window BEFORE the current frame is appended.
        if len(self._buf) >= self.w:
            m = sum(self._buf) / len(self._buf)
            v = sum((x - m) * (x - m) for x in self._buf) / len(self._buf)
            sd = v ** 0.5
            if sd > self.thr:
                self._consec += 1
            else:
                self._consec = 0
            fired = self._consec >= self.persist
        else:
            fired = False
            self._consec = 0
        self._buf.append(float(eff))
        if len(self._buf) > self.w:
            self._buf.pop(0)
        return fired
