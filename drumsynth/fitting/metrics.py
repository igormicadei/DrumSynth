"""How a reconstruction is measured against the sound it came from.

The objective is relative waveform mean-squared error:

    relative_mse = Σ(y - ŷ)² / Σ y²

Waveform, not spectrum: a spectral distance can be small while the phase is
wrong enough to change what the hit sounds like, and this model stores phase
explicitly, so there is no reason to look away from it. The measure is
relative so a target like 1e-5 means the same thing for a quiet sample and a
loud one, and it maps straight to a signal-to-noise ratio: 1e-5 is 50 dB.

Correlation, RMS ratio and peak ratio come along because they fail in
different ways. A reconstruction can track the waveform almost perfectly and
still sit 3 dB low; that shows up in the RMS ratio and nowhere else.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


def rms(signal: np.ndarray) -> float:
    x = np.asarray(signal, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


def peak(signal: np.ndarray) -> float:
    x = np.asarray(signal, dtype=np.float64)
    return float(np.max(np.abs(x))) if x.size else 0.0


def relative_mse(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Error energy over reference energy. Zero for a silent reference."""
    y = np.asarray(reference, dtype=np.float64)
    y_hat = np.asarray(estimate, dtype=np.float64)
    energy = float(np.sum(y * y))
    if energy <= 0.0:
        return 0.0
    return float(np.sum((y - y_hat) ** 2) / energy)


def correlation(reference: np.ndarray, estimate: np.ndarray) -> float:
    y = np.asarray(reference, dtype=np.float64)
    y_hat = np.asarray(estimate, dtype=np.float64)
    if y.size < 2 or np.std(y) <= 0.0 or np.std(y_hat) <= 0.0:
        return 0.0
    return float(np.corrcoef(y, y_hat)[0, 1])


def snr_db(relative_error: float) -> float:
    """The same number as `relative_mse`, in decibels of signal to noise."""
    if relative_error <= 0.0:
        return math.inf
    return float(-10.0 * math.log10(relative_error))


@dataclass(frozen=True)
class Quality:
    """Everything measured about one reconstruction."""

    relative_mse: float
    snr_db: float
    correlation: float
    reference_rms: float
    estimate_rms: float
    reference_peak: float
    estimate_peak: float

    @classmethod
    def measure(cls, reference: np.ndarray, estimate: np.ndarray) -> "Quality":
        error = relative_mse(reference, estimate)
        return cls(
            relative_mse=error,
            snr_db=snr_db(error),
            correlation=correlation(reference, estimate),
            reference_rms=rms(reference),
            estimate_rms=rms(estimate),
            reference_peak=peak(reference),
            estimate_peak=peak(estimate),
        )

    @property
    def rms_ratio(self) -> float:
        return self.estimate_rms / self.reference_rms if self.reference_rms else 0.0

    @property
    def peak_ratio(self) -> float:
        return self.estimate_peak / self.reference_peak if self.reference_peak else 0.0

    def to_dict(self) -> dict:
        data = asdict(self)
        data["rms_ratio"] = self.rms_ratio
        data["peak_ratio"] = self.peak_ratio
        return data
