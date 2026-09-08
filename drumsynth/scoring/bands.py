"""Per-band decay, fitted as a straight line in dB.

The robust fallback for everywhere modal extraction is unreliable, and the
first thing worth building: `SignalPrep` + `BandDecayAnalyzer` + `GlideAnalyzer`
already give a usable comparison, and none of the three is hard.

A low r_squared here is informative, not a failure. It means the band holds
several modes with different t60, which is exactly what the reference showed —
and it is the reason there is no `decay_shape` parameter anywhere in the synth.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..core.constants import Audio, Decay, Decibels
from ..core.dsp import EnvelopeFollower, FilterDesign, LinearRegression
from .descriptors import BandDecay


class BandDecayAnalyzer:
    """Bandpass, envelope, fit a line between the peak and the noise floor."""

    DEFAULT_BANDS: tuple[tuple[float, float], ...] = (
        (40, 130),
        (130, 230),
        (230, 400),
        (400, 900),
        (900, 2000),
        (2000, 6000),
        (6000, 15000),
    )

    #: Distance above the noise floor at which fitting stops. Six decibels of
    #: clearance keeps the fit out of the transition into the floor, where the
    #: slope flattens and every t60 comes out long.
    FLOOR_MARGIN_DB: float = 6.0

    #: Fitting stops here even when the floor is lower — the last 60 dB of a
    #: decay is all the -60 dB definition needs.
    MAX_RANGE_DB: float = 60.0

    #: Fewer frames than this and the fit is not reported.
    MIN_POINTS: int = 5

    #: Quantile of a band's own frame levels taken as that band's noise floor.
    #:
    #: The broadband floor is not the floor in every band. A reference whose
    #: 900-2000 Hz content sits 50 dB below its fundamental has almost nothing
    #: but noise up there, and a fit anchored to the BROADBAND floor never
    #: clips — so the "decay" measured in that band is the noise, reported with
    #: a confident t60 several times too long. Each band gets its own floor.
    BAND_FLOOR_QUANTILE: float = 0.10

    def __init__(
        self,
        sr: int = Audio.DEFAULT_SR,
        bands: Sequence[tuple[float, float]] | None = None,
        hop_seconds: float = 0.01,
    ) -> None:
        self.sr = int(sr)
        self.bands = tuple(bands) if bands is not None else BandDecayAnalyzer.DEFAULT_BANDS
        self.hop_seconds = float(hop_seconds)
        self._follower = EnvelopeFollower(self.sr, hop_seconds, hop_seconds * 3.0)
        self._filters = {
            band: FilterDesign.bandpass(band[0], band[1], self.sr)
            for band in self.bands
            if band[0] < 0.5 * self.sr
        }

    def analyze(self, x: np.ndarray, noise_floor_db: float = -np.inf) -> list[BandDecay]:
        """One BandDecay per band. `noise_floor_db` is relative to signal peak."""
        x = np.asarray(x, dtype=np.float64)
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        absolute_floor = (
            Decibels.from_amplitude(peak) + noise_floor_db
            if np.isfinite(noise_floor_db) and peak > 0
            else -np.inf
        )

        results = []
        for band in self.bands:
            if band not in self._filters:
                continue
            filtered = FilterDesign.apply(self._filters[band], x)
            times, db = self._follower.db(filtered)
            floor = max(absolute_floor, self.band_floor_db(db))
            slope, level, valid, r_squared = self.fit_slope(db, times, floor)
            results.append(
                BandDecay(
                    f_low=float(band[0]),
                    f_high=float(band[1]),
                    level_db=level,
                    slope_db_s=slope,
                    t60=Decay.slope_db_s_to_t60(slope),
                    valid_range=valid,
                    r_squared=r_squared,
                )
            )
        return results

    def band_floor_db(self, db_curve: np.ndarray) -> float:
        """This band's own noise floor, as an absolute dB level.

        A low quantile of the band's frame levels. In a band that decayed into
        the noise, most frames ARE the noise and the quantile lands on it; in a
        band still ringing at the end of the file it lands on the quietest part
        of a real decay, which is conservative in the right direction.
        """
        finite = db_curve[np.isfinite(db_curve)]
        if finite.size < 4:
            return -np.inf
        return float(np.quantile(finite, BandDecayAnalyzer.BAND_FLOOR_QUANTILE))

    def fit_slope(
        self, db_curve: np.ndarray, times: np.ndarray, floor_db: float
    ) -> tuple[float, float, tuple[float, float], float]:
        """Fit only the region between the band's peak and the noise floor.

        Returns (slope_db_s, level_db_at_0, valid_range, r_squared).
        """
        db_curve = np.asarray(db_curve, dtype=np.float64)
        times = np.asarray(times, dtype=np.float64)
        if db_curve.size == 0:
            return np.nan, -np.inf, (0.0, 0.0), 0.0

        peak_index = int(np.argmax(db_curve))
        peak_db = float(db_curve[peak_index])
        cutoff = max(
            floor_db + BandDecayAnalyzer.FLOOR_MARGIN_DB,
            peak_db - BandDecayAnalyzer.MAX_RANGE_DB,
        )

        # Stop at the first frame that drops below the cutoff rather than
        # keeping every frame above it: a band that dips and recovers would
        # otherwise contribute the recovery to the decay fit.
        tail = db_curve[peak_index:]
        below = np.flatnonzero(tail <= cutoff)
        stop = peak_index + (int(below[0]) if below.size else len(tail))

        indices = np.arange(peak_index, stop)
        if indices.size < BandDecayAnalyzer.MIN_POINTS:
            return np.nan, peak_db, (float(times[peak_index]), float(times[peak_index])), 0.0

        fit = LinearRegression.fit(times[indices], db_curve[indices])
        valid = (float(times[indices[0]]), float(times[indices[-1]]))
        if not fit.is_valid:
            return np.nan, peak_db, valid, 0.0
        return float(fit.slope), float(fit.intercept), valid, float(fit.r_squared)
