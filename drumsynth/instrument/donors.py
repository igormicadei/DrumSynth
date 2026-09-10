"""Where a rendered hit gets its phase.

The magnitude field says what the drum sounds like at a velocity. It says
nothing about phase, and phase cannot be averaged or interpolated across
recordings: two strikes of the same drum are two different sets of initial
phases, and crossfading them cancels rather than blends. So phase is not
interpolated — it is *borrowed*, whole, from the recording nearest in velocity.

A donor is one layer's phase field, stored on its own. Three ways to store it:

    exact     the unwrapped phase of every bin and frame, as measured
    lowrank   a rank-r complex factorization of the layer's component block,
              of which only the argument is ever used
    phase     a rank-r factorization of the unwrapped phase itself

`lowrank` throwing away half of what it stores and still winning is not a
contradiction: phase alone has no structure a factorization can hold, while
the complex block does, and the modulus is what carries that structure
(docs/FINDINGS.md §9).
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from ..spectral import phase as phase_codec


class DonorCodec:
    """How one layer's phase field is stored.

    Split into `prepare` and `encode` for the same reason the per-hit codecs
    are: a search over ranks would otherwise refactorize every layer of the
    drum for every candidate, and there are a lot of both.
    """

    name: ClassVar[str] = ""

    @staticmethod
    def prepare(block: np.ndarray):
        """The analysis that does not depend on the rank."""
        raise NotImplementedError

    @staticmethod
    def encode(prepared, rank: int) -> dict[str, np.ndarray]:
        raise NotImplementedError

    @staticmethod
    def decode(arrays: dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def n_scalars(bins: int, frames: int, rank: int) -> int:
        raise NotImplementedError

    @staticmethod
    def clamp(bins: int, frames: int, rank: int) -> int:
        return max(1, min(rank, bins, frames))


class ExactDonor(DonorCodec):
    """Every phase value. The baseline the factorizations have to beat."""

    name = "exact"

    @staticmethod
    def prepare(block: np.ndarray) -> np.ndarray:
        return phase_codec.unwrap(np.angle(block))

    @staticmethod
    def encode(prepared, rank: int = 0) -> dict[str, np.ndarray]:
        return {"phase": np.asarray(prepared, dtype=np.float32)}

    @staticmethod
    def decode(arrays: dict[str, np.ndarray]) -> np.ndarray:
        return np.asarray(arrays["phase"], dtype=np.float64)

    @staticmethod
    def n_scalars(bins: int, frames: int, rank: int = 0) -> int:
        return bins * frames

    @staticmethod
    def clamp(bins: int, frames: int, rank: int = 0) -> int:
        return 0


class LowRankDonor(DonorCodec):
    """The argument of a rank-r complex factorization of the layer's block."""

    name = "lowrank"

    @staticmethod
    def prepare(block: np.ndarray):
        return np.linalg.svd(block, full_matrices=False)

    @staticmethod
    def encode(prepared, rank: int) -> dict[str, np.ndarray]:
        u, s, vt = prepared
        r = LowRankDonor.clamp(u.shape[0], vt.shape[1], rank)
        return {
            "weights": u[:, :r].astype(np.complex64),
            "basis": (s[:r, None] * vt[:r]).astype(np.complex64),
        }

    @staticmethod
    def decode(arrays: dict[str, np.ndarray]) -> np.ndarray:
        weights = np.asarray(arrays["weights"], dtype=np.complex128)
        basis = np.asarray(arrays["basis"], dtype=np.complex128)
        return np.angle(weights @ basis)

    @staticmethod
    def n_scalars(bins: int, frames: int, rank: int) -> int:
        r = LowRankDonor.clamp(bins, frames, rank)
        return 2 * r * (bins + frames)


class PhaseDonor(DonorCodec):
    """A rank-r factorization of the unwrapped phase itself."""

    name = "phase"

    @staticmethod
    def prepare(block: np.ndarray):
        return np.linalg.svd(phase_codec.unwrap(np.angle(block)), full_matrices=False)

    @staticmethod
    def encode(prepared, rank: int) -> dict[str, np.ndarray]:
        u, s, vt = prepared
        r = PhaseDonor.clamp(u.shape[0], vt.shape[1], rank)
        return {
            "weights": u[:, :r].astype(np.float32),
            "basis": (s[:r, None] * vt[:r]).astype(np.float32),
        }

    @staticmethod
    def decode(arrays: dict[str, np.ndarray]) -> np.ndarray:
        return np.asarray(arrays["weights"], dtype=np.float64) @ np.asarray(
            arrays["basis"], dtype=np.float64
        )

    @staticmethod
    def n_scalars(bins: int, frames: int, rank: int) -> int:
        return PhaseDonor.clamp(bins, frames, rank) * (bins + frames)


CODECS: dict[str, type[DonorCodec]] = {
    codec.name: codec for codec in (ExactDonor, LowRankDonor, PhaseDonor)
}


def codec_for(name: str) -> type[DonorCodec]:
    try:
        return CODECS[name]
    except KeyError:
        raise ValueError(f"unknown donor codec {name!r}; known: {sorted(CODECS)}") from None


def choose(velocities: np.ndarray, count: int) -> np.ndarray:
    """Indices of the layers that will act as donors, spread over the range.

    `count` of 0 or more than there are layers means every layer donates, which
    is what a fit picks whenever it can afford it: the further a rendered
    velocity is from the strike its phase came from, the more the model is
    playing back one hit with another hit's spectrum.
    """
    n = int(velocities.size)
    if count <= 0 or count >= n:
        return np.arange(n, dtype=np.int32)
    if count == 1:
        return np.array([n // 2], dtype=np.int32)

    targets = np.linspace(velocities[0], velocities[-1], count)
    return np.unique(np.abs(velocities[None, :] - targets[:, None]).argmin(axis=1)).astype(
        np.int32
    )


def nearest(donor_velocities: np.ndarray, velocity: float) -> int:
    """Which donor a rendered velocity borrows its phase from."""
    return int(np.argmin(np.abs(np.asarray(donor_velocities, dtype=np.float64) - velocity)))
