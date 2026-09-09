"""The component block, and the three ways a model stores it.

A model keeps `k` FFT bins. Bin `b` is a sinusoid at that bin's own centre
frequency, so across frames its phase advances by a known constant,

    2π b hop / n_fft   radians per frame,

whatever the sound does. Divide that rotation out and what is left is a
*component block* Z: `k` rows of complex amplitude that move slowly — the
envelope in the modulus, the detuning and any phase wander in the argument.

    Z = D[bins] · exp(-i · drift)

Every codec here stores Z and hands it back. They differ in what they assume
about it:

    raw       modulus and argument of every entry, as measured
    shared    a rank-r factorization of the modulus, argument stored separately
    lowrank   a rank-r factorization of Z itself, complex — argument included

Which one wins is a property of the sound, not of the mathematics, and the
gap is large. On a tom, `lowrank` reaches the same error as `raw` in a quarter
of the numbers, because its partials really are a handful of decaying complex
exponentials. On a cymbal it collapses — noise has no low-rank structure to
find — and `shared` with interpolated phase takes the frontier instead. That
is the whole reason this project searches rather than decides.

Two rules, both measured rather than assumed (see docs/FINDINGS.md):

* the factorizations work on the *linear* modulus, not on a log envelope. The
  objective is squared error, and a truncated SVD is exactly the best rank-r
  approximation under squared error. Compressing log-amplitude instead
  measured about 100x worse at equal size.
* a truncated `shared` block may go slightly negative in a decayed tail. It is
  left alone: clipping it to zero would only move the reconstruction away from
  the least-squares optimum, and a negative amplitude is just a phase flip
  60 dB down.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from . import phase as phase_codec
from .stft import StftSpec


def drift(bins: np.ndarray, spec: StftSpec, n_frames: int) -> np.ndarray:
    """Phase each kept bin accrues per frame from its own centre frequency."""
    frames = np.arange(n_frames, dtype=np.float64)
    per_frame = 2.0 * np.pi * np.asarray(bins, dtype=np.float64) * spec.hop / spec.n_fft
    return per_frame[:, None] * frames[None, :]


def to_components(spectrogram: np.ndarray, bins: np.ndarray, spec: StftSpec) -> np.ndarray:
    """The component block of the kept bins: bin rotation divided out."""
    rows = spectrogram[bins]
    return rows * np.exp(-1j * drift(bins, spec, rows.shape[1]))


def from_components(block: np.ndarray, bins: np.ndarray, spec: StftSpec) -> np.ndarray:
    """Put the bin rotation back on, giving spectrogram rows again."""
    return block * np.exp(1j * drift(bins, spec, block.shape[1]))


class ComponentCodec:
    """How one component block is stored.

    `prepare` does the analysis that does not depend on the setting — a
    factorization, a polar split — so a search over ranks and strides pays for
    it once per block instead of once per candidate. `encode` then applies one
    setting, quantizing to float32 exactly as the file will, and `decode`
    reverses it with nothing but the stored arrays.
    """

    name: ClassVar[str] = ""
    param_name: ClassVar[str] = "param"
    #: Whether `phase_stride` means anything for this codec.
    uses_phase: ClassVar[bool] = True

    @staticmethod
    def prepare(block: np.ndarray):
        raise NotImplementedError

    @staticmethod
    def encode(prepared, param: int, phase_stride: int) -> dict[str, np.ndarray]:
        raise NotImplementedError

    @staticmethod
    def decode(arrays: dict[str, np.ndarray], n_frames: int) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def n_scalars(n_components: int, n_frames: int, param: int, phase_stride: int) -> int:
        raise NotImplementedError

    @staticmethod
    def clamp(param: int, n_components: int, n_frames: int) -> int:
        """The largest setting that still means something for a block this size."""
        return param


class RawComponents(ComponentCodec):
    """Modulus and argument of every entry. The baseline everything must beat."""

    name = "raw"
    param_name = "unused"
    uses_phase = True

    @staticmethod
    def prepare(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.abs(block), phase_codec.unwrap(np.angle(block))

    @staticmethod
    def encode(prepared, param: int, phase_stride: int) -> dict[str, np.ndarray]:
        magnitude, phase = prepared
        indices, values = phase_codec.sample(phase, phase_stride)
        return {
            "magnitude": magnitude.astype(np.float32),
            "phase_indices": indices,
            "phase_values": values,
        }

    @staticmethod
    def decode(arrays: dict[str, np.ndarray], n_frames: int) -> np.ndarray:
        magnitude = np.asarray(arrays["magnitude"], dtype=np.float64)
        phase = phase_codec.decode(arrays["phase_indices"], arrays["phase_values"], n_frames)
        return magnitude * np.exp(1j * phase)

    @staticmethod
    def n_scalars(n_components: int, n_frames: int, param: int, phase_stride: int) -> int:
        return n_components * n_frames + phase_codec.n_scalars(
            n_components, n_frames, phase_stride
        )

    @staticmethod
    def clamp(param: int, n_components: int, n_frames: int) -> int:
        return 0


class SharedComponents(ComponentCodec):
    """Rank-r factorization of the modulus; the argument stored on its own.

        |Z| ≈ weights @ basis

    The basis is temporal and shared by every component: one set of curves that
    all the partials of a hit are scaled copies of. Partials of a drum do tend
    to share an attack and a decay shape, which is why a rank of 8 or 16 often
    reproduces a whole block.
    """

    name = "shared"
    param_name = "rank"
    uses_phase = True

    @staticmethod
    def prepare(block: np.ndarray):
        u, s, vt = np.linalg.svd(np.abs(block), full_matrices=False)
        return u, s, vt, phase_codec.unwrap(np.angle(block))

    @staticmethod
    def encode(prepared, param: int, phase_stride: int) -> dict[str, np.ndarray]:
        u, s, vt, phase = prepared
        rank = SharedComponents.clamp(param, u.shape[0], vt.shape[1])
        indices, values = phase_codec.sample(phase, phase_stride)
        return {
            "weights": u[:, :rank].astype(np.float32),
            "basis": (s[:rank, None] * vt[:rank]).astype(np.float32),
            "phase_indices": indices,
            "phase_values": values,
        }

    @staticmethod
    def decode(arrays: dict[str, np.ndarray], n_frames: int) -> np.ndarray:
        magnitude = np.asarray(arrays["weights"], dtype=np.float64) @ np.asarray(
            arrays["basis"], dtype=np.float64
        )
        phase = phase_codec.decode(arrays["phase_indices"], arrays["phase_values"], n_frames)
        return magnitude * np.exp(1j * phase)

    @staticmethod
    def n_scalars(n_components: int, n_frames: int, param: int, phase_stride: int) -> int:
        rank = SharedComponents.clamp(param, n_components, n_frames)
        return rank * (n_components + n_frames) + phase_codec.n_scalars(
            n_components, n_frames, phase_stride
        )

    @staticmethod
    def clamp(param: int, n_components: int, n_frames: int) -> int:
        return max(1, min(int(param), n_components, n_frames))


class LowRankComponents(ComponentCodec):
    """Rank-r factorization of the complex block — amplitude and phase together.

        Z ≈ weights @ basis,  both complex

    One rank-1 term is a fixed spectral pattern multiplied by one complex curve
    in time: a group of partials that decay together and share a detuning. A
    struck tonal drum is a few of those, which is why this codec can be four
    times smaller than storing the phase frame by frame at the same error. It
    has nothing to offer a sound whose phase is noise, and on a cymbal it loses
    to every other option, which is exactly what the search is for.

    There is no phase stride here: the phase is inside the factorization.
    """

    name = "lowrank"
    param_name = "rank"
    uses_phase = False

    @staticmethod
    def prepare(block: np.ndarray):
        return np.linalg.svd(block, full_matrices=False)

    @staticmethod
    def encode(prepared, param: int, phase_stride: int = 1) -> dict[str, np.ndarray]:
        u, s, vt = prepared
        rank = LowRankComponents.clamp(param, u.shape[0], vt.shape[1])
        return {
            "weights": u[:, :rank].astype(np.complex64),
            "basis": (s[:rank, None] * vt[:rank]).astype(np.complex64),
        }

    @staticmethod
    def decode(arrays: dict[str, np.ndarray], n_frames: int) -> np.ndarray:
        return np.asarray(arrays["weights"], dtype=np.complex128) @ np.asarray(
            arrays["basis"], dtype=np.complex128
        )

    @staticmethod
    def n_scalars(
        n_components: int, n_frames: int, param: int, phase_stride: int = 1
    ) -> int:
        rank = LowRankComponents.clamp(param, n_components, n_frames)
        return 2 * rank * (n_components + n_frames)  # complex: two numbers each

    @staticmethod
    def clamp(param: int, n_components: int, n_frames: int) -> int:
        return max(1, min(int(param), n_components, n_frames))


CODECS: dict[str, type[ComponentCodec]] = {
    codec.name: codec for codec in (RawComponents, SharedComponents, LowRankComponents)
}


def codec_for(name: str) -> type[ComponentCodec]:
    try:
        return CODECS[name]
    except KeyError:
        raise ValueError(
            f"unknown component codec {name!r}; known: {sorted(CODECS)}"
        ) from None
