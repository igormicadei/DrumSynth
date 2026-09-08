"""Multi-resolution log-magnitude STFT loss.

A single scalar for automated fitting. Reported alongside the ScoreCard but
deliberately NOT folded into `total`: a scalar tells you "worse", which is
almost useless when you are tuning 109 numbers by hand.

Two decisions carry this loss:

  * Log magnitude. Linear-magnitude L2 is dominated by the loudest bins — in a
    tom, the fundamental in the first 200 ms, some 40 dB above everything else
    — so it reports "very similar" while the entire transient and the whole
    3-second tail are wrong. Log puts a -60 dB error and a -6 dB error on
    comparable footing, roughly matching perception. This matters more than
    everything else here combined.
  * Multiple resolutions. The 88.0/92.8 Hz pair needs ~500 ms to resolve; the
    4.4 ms attack needs ~5 ms. No single window does both.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..core.constants import Audio


class MultiResolutionSTFTLoss:
    """Mean absolute log-magnitude difference, summed over window sizes."""

    def __init__(
        self,
        sr: int = Audio.DEFAULT_SR,
        fft_sizes: Sequence[int] = (256, 1024, 4096, 16384),
        log_eps: float = 1e-7,
        floor_db: float = -100.0,
    ) -> None:
        self.sr = int(sr)
        self.fft_sizes = tuple(int(size) for size in fft_sizes)
        self.log_eps = float(log_eps)
        self.floor_db = float(floor_db)
        self._windows = {size: np.hanning(size) for size in self.fft_sizes}

    def __call__(self, reference: np.ndarray, generated: np.ndarray) -> float:
        per_resolution = self.per_resolution(reference, generated)
        values = [value for value in per_resolution.values() if np.isfinite(value)]
        return float(np.mean(values)) if values else float("nan")

    def per_resolution(
        self, reference: np.ndarray, generated: np.ndarray
    ) -> dict[int, float]:
        """Loss broken out by window size.

        A large error only at short windows points at the excitation; only at
        long windows points at the modal bank. That split is the one genuinely
        diagnostic thing a scalar loss can tell you.
        """
        reference, generated = self._align(reference, generated)
        return {
            size: self._loss_at(reference, generated, size) for size in self.fft_sizes
        }

    # -- internals ------------------------------------------------------------

    @staticmethod
    def _align(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Zero-pad the shorter signal. Truncating instead would silently drop
        the part of the tail that is most likely to be wrong."""
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        length = max(len(a), len(b))
        return (
            np.pad(a, (0, length - len(a))),
            np.pad(b, (0, length - len(b))),
        )

    def _spectrogram_db(self, x: np.ndarray, size: int) -> np.ndarray:
        hop = size // 4
        if len(x) < size:
            x = np.pad(x, (0, size - len(x)))
        window = self._windows[size]
        starts = range(0, len(x) - size + 1, hop)
        frames = np.stack([x[start : start + size] * window for start in starts])
        magnitude = np.abs(np.fft.rfft(frames, axis=1))
        db = 20.0 * np.log10(magnitude + self.log_eps)
        return np.maximum(db, np.max(db) + self.floor_db if db.size else self.floor_db)

    def _loss_at(self, reference: np.ndarray, generated: np.ndarray, size: int) -> float:
        if len(reference) < size:
            return float("nan")
        reference_db = self._spectrogram_db(reference, size)
        generated_db = self._spectrogram_db(generated, size)
        rows = min(len(reference_db), len(generated_db))
        if rows == 0:
            return float("nan")
        return float(
            np.mean(np.abs(reference_db[:rows] - generated_db[:rows]))
        )
