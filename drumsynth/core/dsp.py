"""Shared DSP primitives.

These are used by both the synthesizer (noise bands) and the analysis chain
(band decays, envelopes, glide). Keeping one implementation of each means the
noise band a `NoiseVoice` renders and the band a `BandDecayAnalyzer` measures
are filtered by the same design, so a measured band slope is comparable to the
`t60` that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as _signal

from .constants import Audio, Decibels


class FilterDesign:
    """Bandpass / lowpass design in second-order-section form.

    SOS rather than transfer-function coefficients: a 4th-order 40-130 Hz
    bandpass at 44.1 kHz has poles close enough to the unit circle that the
    `ba` form loses accuracy outright.
    """

    DEFAULT_ORDER: int = 4

    @staticmethod
    def bandpass(
        f_low: float, f_high: float, sr: int, order: int = DEFAULT_ORDER
    ) -> np.ndarray:
        """Butterworth bandpass, edges clamped inside (0, nyquist)."""
        nyquist = 0.5 * sr
        low = max(float(f_low), 1e-3)
        high = min(float(f_high), nyquist * 0.999)
        if low >= high:
            raise ValueError(f"empty band after clamping: {f_low}-{f_high} Hz at sr={sr}")
        if high >= nyquist * 0.99:
            return _signal.butter(order, low / nyquist, btype="highpass", output="sos")
        return _signal.butter(
            order, [low / nyquist, high / nyquist], btype="bandpass", output="sos"
        )

    @staticmethod
    def lowpass(f_cut: float, sr: int, order: int = DEFAULT_ORDER) -> np.ndarray:
        nyquist = 0.5 * sr
        return _signal.butter(order, min(f_cut, nyquist * 0.999) / nyquist,
                              btype="lowpass", output="sos")

    @staticmethod
    def apply(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Causal filtering. Never filtfilt — zero-phase filtering smears a
        transient backwards in time and moves the measured onset."""
        return _signal.sosfilt(sos, np.asarray(x, dtype=np.float64))

    @staticmethod
    def response_at(sos: np.ndarray, freqs: np.ndarray, sr: int) -> np.ndarray:
        """|H(f)| of an SOS chain at arbitrary frequencies.

        Used to undo a bandpass's own gain when reporting the amplitude of a
        mode measured through it.
        """
        f = np.atleast_1d(np.asarray(freqs, dtype=float))
        z = np.exp(2j * np.pi * f / sr)
        h = np.ones_like(z, dtype=complex)
        for section in np.atleast_2d(sos):
            b0, b1, b2, _, a1, a2 = section
            num = b0 + b1 / z + b2 / z**2
            den = 1.0 + a1 / z + a2 / z**2
            h *= num / den
        return np.abs(h)

    @staticmethod
    def center_and_q(f_low: float, f_high: float) -> tuple[float, float]:
        """Geometric center frequency and Q of a band."""
        center = float(np.sqrt(max(f_low, 1e-9) * max(f_high, 1e-9)))
        bandwidth = float(f_high - f_low)
        return center, (center / bandwidth if bandwidth > 0 else np.inf)


class EnvelopeFollower:
    """Framed RMS envelope in dB.

    Frame-based rather than a one-pole follower: the analysis side needs a
    known, identical time grid on both signals, and a one-pole's smoothing
    interacts with the decay rate it is trying to measure.
    """

    def __init__(
        self,
        sr: int = Audio.DEFAULT_SR,
        hop_seconds: float = 0.005,
        window_seconds: float | None = None,
    ) -> None:
        self.sr = int(sr)
        self.hop = max(1, int(round(hop_seconds * sr)))
        window = window_seconds if window_seconds is not None else hop_seconds * 4.0
        self.window = max(self.hop, int(round(window * sr)))

    def rms(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (times, linear RMS) on this follower's frame grid."""
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return np.zeros(0), np.zeros(0)

        n_frames = max(1, 1 + (len(x) - 1) // self.hop)
        padded = np.concatenate([x, np.zeros(self.window, dtype=np.float64)])
        # Cumulative sum of squares turns the whole framing into two lookups.
        power = np.concatenate([[0.0], np.cumsum(padded**2)])
        starts = np.arange(n_frames) * self.hop
        ends = starts + self.window
        frame_energy = power[ends] - power[starts]
        rms = np.sqrt(frame_energy / self.window)
        times = (starts + 0.5 * self.window) / self.sr
        return times, rms

    def db(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (times, dB envelope). dB is absolute, not peak-relative."""
        times, rms = self.rms(x)
        return times, Decibels.from_amplitude(rms)


@dataclass
class LineFit:
    """Result of a least-squares straight-line fit, with its own quality measure."""

    slope: float
    intercept: float
    r_squared: float
    n_points: int

    def at(self, x):
        return self.slope * np.asarray(x, dtype=float) + self.intercept

    @property
    def is_valid(self) -> bool:
        return self.n_points >= 3 and np.isfinite(self.slope)


class LinearRegression:
    """Least squares on a straight line. Separate class because every decay
    measurement in the project is a line fit in dB, and they should all report
    r_squared the same way."""

    @staticmethod
    def fit(x: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> LineFit:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        if weights is not None:
            weights = np.asarray(weights, dtype=np.float64)[finite]

        if x.size < 2:
            return LineFit(slope=np.nan, intercept=np.nan, r_squared=0.0, n_points=int(x.size))

        w = np.ones_like(x) if weights is None else np.maximum(weights, 0.0)
        w_sum = w.sum()
        if w_sum <= 0:
            return LineFit(slope=np.nan, intercept=np.nan, r_squared=0.0, n_points=int(x.size))

        mean_x = float(np.sum(w * x) / w_sum)
        mean_y = float(np.sum(w * y) / w_sum)
        var_x = float(np.sum(w * (x - mean_x) ** 2))
        if var_x <= 0:
            return LineFit(slope=np.nan, intercept=float(mean_y), r_squared=0.0, n_points=int(x.size))

        slope = float(np.sum(w * (x - mean_x) * (y - mean_y)) / var_x)
        intercept = mean_y - slope * mean_x
        residual = y - (slope * x + intercept)
        ss_res = float(np.sum(w * residual**2))
        ss_tot = float(np.sum(w * (y - mean_y) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return LineFit(slope, intercept, max(0.0, r_squared), int(x.size))


class SpectralPeak:
    """Sub-bin peak location.

    A 0.25 s window at 44.1 kHz has 4 Hz bins — 72 cents at 92.5 Hz, far coarser
    than the +/-10 cent tolerance the glide is scored against. Parabolic
    interpolation on the log magnitude recovers roughly a tenth of a bin.
    """

    @staticmethod
    def refine(magnitudes: np.ndarray, index: int) -> tuple[float, float]:
        """Return (fractional bin index, interpolated magnitude)."""
        mags = np.asarray(magnitudes, dtype=np.float64)
        if index <= 0 or index >= len(mags) - 1:
            return float(index), float(mags[index])

        left, center, right = (
            np.log(max(mags[index - 1], Audio.EPS)),
            np.log(max(mags[index], Audio.EPS)),
            np.log(max(mags[index + 1], Audio.EPS)),
        )
        denominator = left - 2.0 * center + right
        if abs(denominator) < 1e-30:
            return float(index), float(mags[index])
        delta = 0.5 * (left - right) / denominator
        delta = float(np.clip(delta, -0.5, 0.5))
        peak_log = center - 0.25 * (left - right) * delta
        return float(index) + delta, float(np.exp(peak_log))

    @staticmethod
    def find_all(magnitudes: np.ndarray, relative_threshold_db: float = -20.0) -> list[int]:
        """Indices of local maxima above `relative_threshold_db` from the max."""
        mags = np.asarray(magnitudes, dtype=np.float64)
        if mags.size < 3:
            return []
        floor = np.max(mags) * (10.0 ** (relative_threshold_db / 20.0))
        interior = np.arange(1, len(mags) - 1)
        is_peak = (mags[1:-1] > mags[:-2]) & (mags[1:-1] >= mags[2:]) & (mags[1:-1] >= floor)
        return interior[is_peak].tolist()
