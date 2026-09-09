"""Short-time Fourier analysis and the overlap-add synthesis that inverts it.

One window (periodic Hann), one framing convention, one inverse. Everything
downstream — bin selection, envelope coding, phase coding — is expressed in
terms of the complex spectrogram this module produces, and the reconstruction
is always measured through :func:`synthesize`, never assumed.

The inverse is a *weighted* overlap-add. Analysis multiplies each frame by the
window and synthesis multiplies it again, so the accumulated signal is the
input times the summed window power; dividing that out is exact wherever the
denominator is non-zero, for any hop. Keeping an unmodified spectrogram round
tripping to machine precision is what makes a reconstruction error meaningful:
whatever error a model shows is the model's, not the transform's.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

#: Below this the summed window power is treated as no coverage at all.
_POWER_FLOOR = 1e-12


@dataclass(frozen=True)
class StftSpec:
    """The framing of an analysis: transform size and hop, in samples.

    `n_fft` must be a multiple of `hop`. That is not a mathematical
    requirement of the transform — it makes the overlap-add a handful of
    vectorized adds instead of a Python loop over frames, and the search
    evaluates tens of thousands of reconstructions.
    """

    n_fft: int
    hop: int

    def __post_init__(self) -> None:
        if self.n_fft <= 0 or self.n_fft % 2:
            raise ValueError(f"n_fft must be positive and even, got {self.n_fft}")
        if self.hop <= 0:
            raise ValueError(f"hop must be positive, got {self.hop}")
        if self.n_fft % self.hop:
            raise ValueError(
                f"n_fft must be a multiple of hop, got n_fft={self.n_fft} hop={self.hop}"
            )

    @property
    def n_bins(self) -> int:
        """Number of real-FFT bins, DC and Nyquist included."""
        return self.n_fft // 2 + 1

    @property
    def pad(self) -> int:
        """Samples of zero padding placed before the signal, so frame 0 is centred on sample 0."""
        return self.n_fft // 2

    @property
    def overlap(self) -> int:
        """How many frames cover any given sample."""
        return self.n_fft // self.hop

    def n_frames(self, n_samples: int) -> int:
        return 1 + int(n_samples) // self.hop

    def bin_frequencies(self, sr: int) -> np.ndarray:
        return np.fft.rfftfreq(self.n_fft, 1.0 / float(sr))

    @property
    def window(self) -> np.ndarray:
        return _window(self.n_fft)


@lru_cache(maxsize=8)
def _window(n_fft: int) -> np.ndarray:
    """Periodic Hann. Periodic, not symmetric: the symmetric one does not sum flat."""
    n = np.arange(n_fft, dtype=np.float64)
    w = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / n_fft)
    w.setflags(write=False)
    return w


def analyze(signal: np.ndarray, spec: StftSpec) -> np.ndarray:
    """Complex spectrogram of `signal`, shaped (n_bins, n_frames)."""
    x = np.asarray(signal, dtype=np.float64).ravel()
    n_frames = spec.n_frames(x.size)

    padded = np.zeros((n_frames - 1) * spec.hop + spec.n_fft, dtype=np.float64)
    padded[spec.pad : spec.pad + x.size] = x

    frames = np.lib.stride_tricks.sliding_window_view(padded, spec.n_fft)[:: spec.hop]
    return np.fft.rfft(frames[:n_frames] * spec.window, axis=1).T


def synthesize(spectrogram: np.ndarray, spec: StftSpec, n_samples: int) -> np.ndarray:
    """Weighted overlap-add back to `n_samples` samples of audio."""
    n_frames = spectrogram.shape[1]
    frames = np.fft.irfft(spectrogram.T, n=spec.n_fft, axis=1) * spec.window

    signal = _overlap_add(frames, spec.hop)
    power = _window_power(spec, n_frames)

    out = np.divide(signal, power, out=np.zeros_like(signal), where=power > _POWER_FLOOR)
    return out[spec.pad : spec.pad + n_samples]


def _overlap_add(frames: np.ndarray, hop: int) -> np.ndarray:
    """Sum `frames` (n_frames, n_fft) back into one signal at `hop` spacing.

    With n_fft a multiple of hop, every frame splits into `overlap` blocks of
    exactly `hop` samples and the overlap-add is that many shifted block adds.
    """
    n_frames, n_fft = frames.shape
    overlap = n_fft // hop

    blocks = frames.reshape(n_frames, overlap, hop)
    out = np.zeros((n_frames + overlap - 1, hop), dtype=np.float64)
    for shift in range(overlap):
        out[shift : shift + n_frames] += blocks[:, shift, :]
    return out.reshape(-1)


@lru_cache(maxsize=16)
def _window_power(spec: StftSpec, n_frames: int) -> np.ndarray:
    """Summed window² at every output sample — the divisor that makes OLA exact."""
    squared = np.broadcast_to(spec.window**2, (n_frames, spec.n_fft))
    return _overlap_add(squared, spec.hop)
