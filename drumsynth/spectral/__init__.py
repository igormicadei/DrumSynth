"""The representation: STFT in, a compact model out, audio back again.

    drumsynth.spectral.stft         framing, analysis, weighted overlap-add
    drumsynth.spectral.bins         which bins a model keeps
    drumsynth.spectral.components   the component block and the codecs for it
    drumsynth.spectral.phase        storing phase at or below the frame rate
    drumsynth.spectral.model        Candidate and SpectralModel
    drumsynth.spectral.encode       audio + Candidate -> SpectralModel

Nothing here chooses a candidate. Choosing is a measurement problem and lives
in :mod:`drumsynth.fitting`.
"""

from .bins import bin_energy, energy_order, select_bins
from .components import (
    CODECS,
    ComponentCodec,
    LowRankComponents,
    RawComponents,
    SharedComponents,
    codec_for,
    drift,
    from_components,
    to_components,
)
from .encode import encode
from .model import FORMAT, Candidate, SpectralModel
from .stft import StftSpec, analyze, synthesize, synthesize_frames

__all__ = [
    "CODECS",
    "FORMAT",
    "Candidate",
    "ComponentCodec",
    "LowRankComponents",
    "RawComponents",
    "SharedComponents",
    "SpectralModel",
    "StftSpec",
    "analyze",
    "bin_energy",
    "codec_for",
    "drift",
    "encode",
    "energy_order",
    "from_components",
    "select_bins",
    "synthesize",
    "synthesize_frames",
    "to_components",
]
