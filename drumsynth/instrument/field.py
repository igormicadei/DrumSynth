"""How a drum's magnitude changes with how hard it is hit.

Stack the magnitude block of every velocity layer and you have a tensor
`M[v, bin, frame]`: the same drum, the same partials, struck harder. It is
enormously redundant — a harder hit is mostly the same shape louder, brighter
and ringing a little longer — and that redundancy is what makes a velocity
model smaller than the recordings, and what makes interpolation mean anything.

Every codec here ends up in the same form,

    M[v] ≈ Σ_r weights[v, r] · patterns[r]

so a velocity that was never recorded is `weights` interpolated between its
neighbours and combined with the same patterns. The codecs differ only in how
`patterns` is stored:

    full        one pattern per layer, stored whole — the baseline, and the
                one that cannot extrapolate structure it never saw
    velocity    rank-r factorization across the velocity axis: a handful of
                spectro-temporal shapes, each with its own velocity curve
    separable   the same, with each shape further factorized into spectral
                and temporal parts

Interpolation is linear in the weights, which is linear in magnitude. That
follows the same reasoning as the per-hit codecs: the error being measured is
squared error on the linear magnitude, and the factorization that minimizes it
is the one taken on those numbers rather than on their logarithm
(docs/FINDINGS.md §1, and §8 for the velocity axis specifically).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np


class FieldCodec:
    """How the patterns of a magnitude field are stored."""

    name: ClassVar[str] = ""
    #: What its integer parameters mean, in order.
    param_names: ClassVar[tuple[str, ...]] = ()

    @staticmethod
    def prepare(tensor: np.ndarray):
        raise NotImplementedError

    @staticmethod
    def encode(prepared, rank: int, pattern_rank: int) -> dict[str, np.ndarray]:
        raise NotImplementedError

    @staticmethod
    def decode(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """(weights (V, r), patterns (r, bins, frames))."""
        raise NotImplementedError

    @staticmethod
    def n_scalars(shape: tuple[int, int, int], rank: int, pattern_rank: int) -> int:
        raise NotImplementedError

    @staticmethod
    def clamp(shape: tuple[int, int, int], rank: int, pattern_rank: int) -> tuple[int, int]:
        return rank, pattern_rank


class FullField(FieldCodec):
    """Every layer stored whole. Nothing shared, nothing to get wrong."""

    name = "full"
    param_names = ()

    @staticmethod
    def prepare(tensor: np.ndarray) -> np.ndarray:
        return tensor

    @staticmethod
    def encode(prepared, rank: int = 0, pattern_rank: int = 0) -> dict[str, np.ndarray]:
        return {"patterns": np.asarray(prepared, dtype=np.float32)}

    @staticmethod
    def decode(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        patterns = np.asarray(arrays["patterns"], dtype=np.float64)
        return np.eye(patterns.shape[0], dtype=np.float64), patterns

    @staticmethod
    def n_scalars(shape: tuple[int, int, int], rank: int = 0, pattern_rank: int = 0) -> int:
        layers, bins, frames = shape
        return layers * bins * frames

    @staticmethod
    def clamp(shape, rank: int = 0, pattern_rank: int = 0) -> tuple[int, int]:
        return 0, 0


class VelocityField(FieldCodec):
    """Rank-r factorization across velocity: a few shapes, each with a curve."""

    name = "velocity"
    param_names = ("rank",)

    @staticmethod
    def prepare(tensor: np.ndarray):
        layers = tensor.shape[0]
        u, s, vt = np.linalg.svd(tensor.reshape(layers, -1), full_matrices=False)
        return u, s, vt, tensor.shape

    @staticmethod
    def encode(prepared, rank: int, pattern_rank: int = 0) -> dict[str, np.ndarray]:
        u, s, vt, shape = prepared
        r = min(rank, s.size)
        return {
            "weights": (u[:, :r] * s[:r]).astype(np.float32),
            "patterns": vt[:r].reshape(r, shape[1], shape[2]).astype(np.float32),
        }

    @staticmethod
    def decode(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(arrays["weights"], dtype=np.float64),
            np.asarray(arrays["patterns"], dtype=np.float64),
        )

    @staticmethod
    def n_scalars(shape: tuple[int, int, int], rank: int, pattern_rank: int = 0) -> int:
        layers, bins, frames = shape
        r = min(rank, layers)
        return r * (layers + bins * frames)

    @staticmethod
    def clamp(shape, rank: int, pattern_rank: int = 0) -> tuple[int, int]:
        return max(1, min(rank, shape[0])), 0


class SeparableField(FieldCodec):
    """Rank-r across velocity, then rank-p within each shape.

    A pattern is a spectro-temporal block, and those are low rank for the same
    reason a single hit's magnitudes are: the partials of one drum share their
    envelopes. Factorizing them too turns `r · bins · frames` into
    `r · p · (bins + frames)`, which is where a velocity model stops being the
    size of the recordings it came from.
    """

    name = "separable"
    param_names = ("rank", "pattern_rank")

    @staticmethod
    def prepare(tensor: np.ndarray):
        # The last slot memoizes each pattern's own factorization: a pattern is
        # the same array whatever rank slices it, so a search over ranks
        # factorizes it once rather than once per candidate.
        return (*VelocityField.prepare(tensor), {})

    @staticmethod
    def encode(prepared, rank: int, pattern_rank: int) -> dict[str, np.ndarray]:
        u, s, vt, shape, cache = prepared
        velocity_arrays = VelocityField.encode((u, s, vt, shape), rank)
        r = velocity_arrays["patterns"].shape[0]
        bins, frames = shape[1], shape[2]
        p = max(1, min(pattern_rank, bins, frames))

        weights = np.empty((r, bins, p), dtype=np.float32)
        basis = np.empty((r, p, frames), dtype=np.float32)
        for index in range(r):
            if index not in cache:
                cache[index] = np.linalg.svd(
                    vt[index].reshape(bins, frames), full_matrices=False
                )
            pattern_u, pattern_s, pattern_vt = cache[index]
            weights[index] = pattern_u[:, :p].astype(np.float32)
            basis[index] = (pattern_s[:p, None] * pattern_vt[:p]).astype(np.float32)

        return {
            "weights": velocity_arrays["weights"],
            "pattern_weights": weights,
            "pattern_basis": basis,
        }

    @staticmethod
    def decode(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        weights = np.asarray(arrays["pattern_weights"], dtype=np.float64)
        basis = np.asarray(arrays["pattern_basis"], dtype=np.float64)
        return np.asarray(arrays["weights"], dtype=np.float64), weights @ basis

    @staticmethod
    def n_scalars(shape: tuple[int, int, int], rank: int, pattern_rank: int) -> int:
        layers, bins, frames = shape
        r = min(rank, layers)
        p = max(1, min(pattern_rank, bins, frames))
        return r * layers + r * p * (bins + frames)

    @staticmethod
    def clamp(shape, rank: int, pattern_rank: int) -> tuple[int, int]:
        layers, bins, frames = shape
        return max(1, min(rank, layers)), max(1, min(pattern_rank, bins, frames))


CODECS: dict[str, type[FieldCodec]] = {
    codec.name: codec for codec in (FullField, VelocityField, SeparableField)
}


def codec_for(name: str) -> type[FieldCodec]:
    try:
        return CODECS[name]
    except KeyError:
        raise ValueError(f"unknown field codec {name!r}; known: {sorted(CODECS)}") from None


@dataclass
class MagnitudeField:
    """The fitted field: what magnitude this drum has at any velocity."""

    velocities: np.ndarray
    weights: np.ndarray
    patterns: np.ndarray

    @classmethod
    def from_arrays(
        cls, velocities: np.ndarray, codec: str, arrays: dict[str, np.ndarray]
    ) -> "MagnitudeField":
        weights, patterns = codec_for(codec).decode(arrays)
        return cls(np.asarray(velocities, dtype=np.float64), weights, patterns)

    @property
    def n_layers(self) -> int:
        return int(self.velocities.size)

    @property
    def shape(self) -> tuple[int, int]:
        """(bins, frames) of one magnitude block."""
        return self.patterns.shape[1], self.patterns.shape[2]

    @property
    def range(self) -> tuple[float, float]:
        return float(self.velocities[0]), float(self.velocities[-1])

    def weights_at(self, velocity: float) -> np.ndarray:
        """Interpolate the per-layer weights, clamped outside the recorded range."""
        return np.array(
            [
                np.interp(float(velocity), self.velocities, self.weights[:, column])
                for column in range(self.weights.shape[1])
            ]
        )

    def at(self, velocity: float) -> np.ndarray:
        """The magnitude block this drum has at `velocity`."""
        return np.einsum("r,rkt->kt", self.weights_at(velocity), self.patterns)

    def layers(self) -> np.ndarray:
        """The magnitude block of every recorded layer, as the model holds it."""
        return np.einsum("vr,rkt->vkt", self.weights, self.patterns)
