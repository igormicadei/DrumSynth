"""Turning audio into a :class:`SpectralModel` for a given candidate.

This is the canonical encode path: analyze, pick bins, take the component
block, code it. The search in :mod:`drumsynth.fitting` calls the same codec
methods with the analysis and the factorizations cached across candidates, so
what it measures is exactly what this function produces.
"""

from __future__ import annotations

import numpy as np

from .bins import select_bins
from .components import to_components
from .model import Candidate, SpectralModel
from .stft import analyze


def encode(signal: np.ndarray, sample_rate: int, candidate: Candidate) -> SpectralModel:
    """Encode `signal` under `candidate`."""
    x = np.asarray(signal, dtype=np.float64).ravel()
    spec = candidate.spec

    spectrogram = analyze(x, spec)
    bins = select_bins(spectrogram, candidate.n_components)
    block = to_components(spectrogram, bins, spec)

    codec = candidate.codec_class
    arrays = codec.encode(
        codec.prepare(block), candidate.codec_param, candidate.phase_stride
    )

    return SpectralModel(
        candidate=candidate,
        sample_rate=int(sample_rate),
        n_samples=int(x.size),
        n_frames=int(spectrogram.shape[1]),
        bins=bins,
        arrays=arrays,
    )
