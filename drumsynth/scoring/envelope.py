"""Broadband RMS envelope.

The cheapest descriptor in the chain and the one that catches the most obvious
failures: a generated hit whose envelope peaks at sample zero when the reference
peaks at 9.8 ms is wrong in a way no amount of modal accuracy repairs.
"""

from __future__ import annotations

import numpy as np

from ..core.constants import Audio, Decibels
from ..core.dsp import EnvelopeFollower
from .descriptors import EnvelopeCurve


class EnvelopeAnalyzer:
    """Framed RMS in dB, plus the three numbers that summarize the attack."""

    def __init__(self, sr: int = Audio.DEFAULT_SR, hop_seconds: float = 0.005) -> None:
        self.sr = int(sr)
        self.hop_seconds = float(hop_seconds)
        self._follower = EnvelopeFollower(self.sr, hop_seconds, hop_seconds * 2.0)

    def analyze(self, x: np.ndarray) -> EnvelopeCurve:
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return EnvelopeCurve(np.zeros(0), np.zeros(0), 0.0, 0.0, 0.0)

        times, rms = self._follower.rms(x)
        db = Decibels.from_amplitude(rms)
        peak_index = int(np.argmax(rms))
        peak_value = float(rms[peak_index])

        return EnvelopeCurve(
            times=times,
            db=db,
            peak_time=float(times[peak_index]),
            rise_10_90=self._rise_time(times, rms, peak_index, peak_value),
            crest_factor_db=self._crest_factor(x),
        )

    @staticmethod
    def _rise_time(
        times: np.ndarray, rms: np.ndarray, peak_index: int, peak_value: float
    ) -> float:
        """10% to 90% of peak amplitude, measured on the rising edge only.

        Zero is a meaningful answer, not a failure: it means the signal was
        already above 90% of its peak in the first frame, which is exactly what
        an impulse-excited modal bank produces.
        """
        if peak_value <= 0.0 or peak_index == 0:
            return 0.0
        rising = rms[: peak_index + 1]
        low = np.flatnonzero(rising >= 0.1 * peak_value)
        high = np.flatnonzero(rising >= 0.9 * peak_value)
        if low.size == 0 or high.size == 0:
            return 0.0
        return float(max(0.0, times[high[0]] - times[low[0]]))

    @staticmethod
    def _crest_factor(x: np.ndarray) -> float:
        """Peak-to-RMS over the whole signal, in dB."""
        rms = float(np.sqrt(np.mean(x**2)))
        peak = float(np.max(np.abs(x)))
        if rms <= 0.0 or peak <= 0.0:
            return 0.0
        return float(Decibels.from_amplitude(peak / rms))

    def energy_fractions(self, x: np.ndarray, *windows: float) -> tuple[float, ...]:
        """Fraction of total energy inside each leading window, in seconds."""
        x = np.asarray(x, dtype=np.float64)
        cumulative = np.cumsum(x**2)
        total = float(cumulative[-1]) if cumulative.size else 0.0
        if total <= 0.0:
            return tuple(0.0 for _ in windows)
        out = []
        for seconds in windows:
            index = min(len(cumulative) - 1, int(round(seconds * self.sr)))
            out.append(float(cumulative[index] / total))
        return tuple(out)
