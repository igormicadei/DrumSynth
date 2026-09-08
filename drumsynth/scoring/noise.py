"""Statistics of the stochastic component.

Noise is never compared sample-wise. Two realizations of the same process have
roughly zero correlation, so a sample-wise comparison of two correct noise
bursts reports total failure. Only aggregates are matchable: how much energy
sits in each band, and how fast each band falls.

When mode estimates are supplied the modal part is resynthesized and subtracted
first, so the statistics describe the residual rather than the whole signal.
That is not a null test — it is subtracting a model from the signal it was
estimated on, which is what a residual is.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..core.constants import Audio, Decibels
from ..core.dsp import EnvelopeFollower, FilterDesign, LinearRegression
from .descriptors import ModeEstimate, NoiseStats


class NoiseAnalyzer:
    """Band energies, band decay rates and attack statistics of the residual."""

    DEFAULT_BANDS: tuple[tuple[float, float], ...] = (
        (200, 800),
        (800, 2000),
        (2000, 6000),
        (6000, 15000),
    )

    #: Window for the spectral-flux measurement of the attack.
    FLUX_WINDOW: int = 512
    FLUX_SECONDS: float = 0.05

    #: Modes quieter than this, relative to the loudest, are not worth
    #: subtracting — their contribution to the residual is below the noise.
    SUBTRACT_FLOOR_DB: float = -50.0

    def __init__(
        self,
        sr: int = Audio.DEFAULT_SR,
        bands: Sequence[tuple[float, float]] | None = None,
    ) -> None:
        self.sr = int(sr)
        self.bands = tuple(bands) if bands is not None else NoiseAnalyzer.DEFAULT_BANDS
        self._follower = EnvelopeFollower(self.sr, 0.005, 0.015)
        self._filters = {
            band: FilterDesign.bandpass(band[0], band[1], self.sr)
            for band in self.bands
            if band[0] < 0.5 * self.sr
        }

    def analyze(
        self, x: np.ndarray, modes: list[ModeEstimate] | None = None
    ) -> NoiseStats:
        """Statistics of `x`, with any supplied modes removed first."""
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return NoiseStats()

        residual = self.residual(x, modes) if modes else x
        total_db = Decibels.from_amplitude(np.sqrt(np.mean(residual**2)))

        energies: dict[tuple[float, float], float] = {}
        slopes: dict[tuple[float, float], float] = {}
        for band, sos in self._filters.items():
            filtered = FilterDesign.apply(sos, residual)
            energies[band] = float(
                Decibels.from_amplitude(np.sqrt(np.mean(filtered**2)))
            )
            slopes[band] = self._band_slope(filtered)

        return NoiseStats(
            band_energy_db=energies,
            band_slope_db_s=slopes,
            energy_fraction_30ms=self._energy_fraction(residual, 0.030),
            energy_fraction_150ms=self._energy_fraction(residual, 0.150),
            spectral_flux_attack=self.spectral_flux(residual),
            modal_residual_db=float(total_db),
        )

    # -- modal subtraction ----------------------------------------------------

    def residual(self, x: np.ndarray, modes: list[ModeEstimate]) -> np.ndarray:
        """`x` with its estimated modes removed.

        Phase matters here and is why `ModeEstimate` carries one: without it the
        resynthesized mode is as likely to add energy as to remove it.
        """
        x = np.asarray(x, dtype=np.float64)
        if not modes:
            return x.copy()

        loudest = max(mode.amplitude for mode in modes)
        threshold = loudest * Decibels.to_amplitude(NoiseAnalyzer.SUBTRACT_FLOOR_DB)

        out = x.copy()
        for mode in modes:
            if mode.amplitude >= threshold:
                out -= mode.render(len(x), self.sr)
        return out

    # -- statistics -----------------------------------------------------------

    def _band_slope(self, filtered: np.ndarray) -> float:
        """dB/s over the band's decay. NaN when the band never rises."""
        times, db = self._follower.db(filtered)
        if db.size < 5:
            return float("nan")
        peak = int(np.argmax(db))
        cutoff = db[peak] - 40.0
        tail = db[peak:]
        below = np.flatnonzero(tail <= cutoff)
        stop = peak + (int(below[0]) if below.size else len(tail))
        if stop - peak < 5:
            return float("nan")
        fit = LinearRegression.fit(times[peak:stop], db[peak:stop])
        return float(fit.slope) if fit.is_valid else float("nan")

    def _energy_fraction(self, x: np.ndarray, seconds: float) -> float:
        cumulative = np.cumsum(x**2)
        total = float(cumulative[-1]) if cumulative.size else 0.0
        if total <= 0.0:
            return 0.0
        index = min(len(cumulative) - 1, int(round(seconds * self.sr)))
        return float(cumulative[index] / total)

    def spectral_flux(self, x: np.ndarray) -> float:
        """Summed positive spectral change over the attack.

        A stick has a bright, fast-changing onset; a mallet does not. This is
        the cheapest number in the chain that responds to `contact_time`, which
        makes it the one to watch once velocity fitting starts.
        """
        window = NoiseAnalyzer.FLUX_WINDOW
        hop = window // 4
        span = min(len(x), int(NoiseAnalyzer.FLUX_SECONDS * self.sr))
        if span < 2 * window:
            return 0.0

        taper = np.hanning(window)
        frames = [
            np.abs(np.fft.rfft(x[start : start + window] * taper))
            for start in range(0, span - window, hop)
        ]
        if len(frames) < 2:
            return 0.0

        flux = 0.0
        for previous, current in zip(frames, frames[1:]):
            flux += float(np.sum(np.maximum(current - previous, 0.0)))
        return flux / (len(frames) - 1) / window
