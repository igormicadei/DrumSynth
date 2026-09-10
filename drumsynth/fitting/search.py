"""The search: build every candidate representation, measure it, keep the best.

The premise of this project is that no formula tells you which representation
a given hit wants. A cymbal with a long shimmering tail and a tight kick are
not compressed well by the same transform size, the same number of bins or the
same codec, and the differences are large — often an order of magnitude in
model size at equal error. So the fit is empirical: every
candidate is encoded, quantized, rendered back to audio and compared against
the input waveform. A representation is never accepted because it looks
compact on paper.

What makes that affordable is that candidates share their expensive parts.
Candidates are grouped by analysis framing and bin count, so one STFT serves
thousands of them, one sort picks the bins, and each codec's factorization is
computed once per block and then sliced to every rank. What remains per
candidate is coding the block at one setting, one inverse STFT and one
comparison — the part that genuinely differs.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Iterator, Sequence

import numpy as np

from ..spectral.bins import energy_order
from ..spectral.components import codec_for, from_components, to_components
from ..spectral.encode import encode
from ..spectral.model import Candidate, SpectralModel
from ..spectral.stft import StftSpec, analyze, synthesize
from .metrics import Quality
from .pareto import pareto_frontier, preferred

#: Default quality target: relative waveform MSE, i.e. 50 dB signal to noise.
DEFAULT_TARGET_MSE = 1e-5


@dataclass(frozen=True)
class SearchSpace:
    """The grid of representations to try.

    `overlaps` is n_fft/hop rather than a hop ratio, because the overlap-add
    needs an integer overlap and naming it that way makes the invalid
    combinations unrepresentable instead of silently skipped.

    The default grid is the one worth running on a hit you care about; `quick`
    is a tenth of it for a first look, and `full` widens every axis for the
    case where the default's answer sits at the edge of a range.
    """

    n_ffts: tuple[int, ...] = (1024, 2048, 4096)
    overlaps: tuple[int, ...] = (2, 4)
    components: tuple[int, ...] = (32, 48, 64, 96, 128, 192, 256, 384, 512)
    ranks: tuple[int, ...] = (1, 2, 4, 8, 16, 24, 32, 48, 64)
    phase_strides: tuple[int, ...] = (1, 2, 4, 8)

    @classmethod
    def quick(cls) -> "SearchSpace":
        """A first look: same shape, a tenth of the points."""
        return cls(
            n_ffts=(1024, 2048),
            overlaps=(2,),
            components=(32, 64, 128, 256),
            ranks=(2, 8, 16, 32),
            phase_strides=(1, 2),
        )

    @classmethod
    def full(cls) -> "SearchSpace":
        """Every axis widened, for when the answer sits against a limit."""
        return cls(
            n_ffts=(512, 1024, 1536, 2048, 3072, 4096, 8192),
            overlaps=(2, 4, 8),
            components=(16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024),
            ranks=(1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128),
            phase_strides=(1, 2, 3, 4, 6, 8, 16),
        )

    def specs(self) -> list[StftSpec]:
        return [
            StftSpec(n_fft, n_fft // overlap)
            for n_fft in self.n_ffts
            for overlap in self.overlaps
            if n_fft % overlap == 0
        ]

    def candidates(
        self, n_samples: int, scalar_budget: int | None = None
    ) -> list[Candidate]:
        """Every candidate worth evaluating for a signal of `n_samples` samples.

        A candidate holding more numbers than the waveform it replaces is
        dropped: whatever else it is, it is not a compression of anything.
        """
        budget = n_samples if scalar_budget is None else scalar_budget
        seen: dict[Candidate, None] = {}

        for spec in self.specs():
            n_frames = spec.n_frames(n_samples)
            for k in self.components:
                if k > spec.n_bins:
                    continue
                for candidate in self._for_bins(spec, k, n_frames):
                    if candidate.n_scalars(n_frames) <= budget:
                        seen.setdefault(candidate, None)

        return list(seen)

    def _for_bins(self, spec: StftSpec, k: int, n_frames: int) -> Iterator[Candidate]:
        strides = [s for s in self.phase_strides if s == 1 or s < n_frames]
        settings: list[tuple[str, int, int]] = [
            ("raw", 0, stride) for stride in strides
        ]
        for rank in self.ranks:
            if rank > min(k, n_frames):
                continue
            clamped = codec_for("shared").clamp(rank, k, n_frames)
            settings += [("shared", clamped, stride) for stride in strides]
            settings.append(("lowrank", clamped, 1))

        for codec, param, stride in settings:
            yield Candidate(
                n_fft=spec.n_fft,
                hop=spec.hop,
                n_components=k,
                codec=codec,
                codec_param=param,
                phase_stride=stride,
            )


@dataclass(frozen=True)
class Evaluation:
    """One candidate, encoded, quantized, rendered and measured end to end."""

    candidate: Candidate
    quality: Quality
    n_scalars: int
    n_frames: int

    @property
    def relative_mse(self) -> float:
        return self.quality.relative_mse

    def to_dict(self) -> dict:
        return {
            "candidate": self.candidate.to_dict(),
            "quality": self.quality.to_dict(),
            "n_scalars": self.n_scalars,
            "n_frames": self.n_frames,
        }


@dataclass(frozen=True)
class Progress:
    """Handed to the `progress` callback as each group of candidates lands."""

    done: int
    total: int
    elapsed: float
    best: Evaluation


@dataclass
class FitResult:
    """What the search found, and everything it looked at on the way."""

    model: SpectralModel
    quality: Quality
    target_mse: float
    evaluations: list[Evaluation]
    elapsed: float
    space: SearchSpace = field(default_factory=SearchSpace)
    sample_rate: int = 0

    @property
    def candidate(self) -> Candidate:
        return self.model.candidate

    @property
    def target_reached(self) -> bool:
        return self.quality.relative_mse <= self.target_mse

    @property
    def frontier(self) -> list[Evaluation]:
        return pareto_frontier(self.evaluations)

    def summary(self) -> str:
        model = self.model
        reached = "reached" if self.target_reached else "NOT REACHED"
        return "\n".join(
            [
                f"representation   {self.candidate.label()}",
                f"frames           {model.n_frames}",
                f"model            {model.n_scalars} numbers, "
                f"{model.n_bytes() / 1024:.1f} kB, "
                f"{1.0 / model.compression_ratio():.1f}x smaller than 16-bit PCM",
                f"relative MSE     {self.quality.relative_mse:.3e} "
                f"({self.quality.snr_db:.1f} dB SNR) — "
                f"target {self.target_mse:.1e} {reached}",
                f"correlation      {self.quality.correlation:.9f}",
                f"level            rms {self.quality.rms_ratio:.6f}x, "
                f"peak {self.quality.peak_ratio:.6f}x",
                f"searched         {len(self.evaluations)} candidates "
                f"in {self.elapsed:.1f} s",
            ]
        )


def fit(
    signal: np.ndarray,
    sample_rate: int,
    *,
    target_mse: float = DEFAULT_TARGET_MSE,
    space: SearchSpace | None = None,
    progress: Callable[[Progress], None] | None = None,
    max_candidates: int | None = None,
    jobs: int = 1,
) -> FitResult:
    """Search `space` for the smallest model of `signal` that stays within `target_mse`.

    `jobs` splits the search across processes; each process takes whole groups
    of candidates that share an analysis, so nothing is analyzed twice. Pass 0
    for one process per core.
    """
    x = np.asarray(signal, dtype=np.float64).ravel()
    if x.size == 0:
        raise ValueError("cannot fit an empty signal")

    space = space or SearchSpace()
    candidates = space.candidates(x.size)
    if max_candidates is not None:
        candidates = candidates[:max_candidates]
    if not candidates:
        raise ValueError(
            "no candidate fits this signal — it may be shorter than one analysis "
            "frame, or the search space may hold nothing smaller than the waveform"
        )

    started = time.perf_counter()
    evaluations: list[Evaluation] = []
    best: Evaluation | None = None

    for group in _run_groups(x, _group(candidates), jobs):
        evaluations.extend(group)
        for evaluation in group:
            best = evaluation if best is None else preferred(best, evaluation, target_mse)

        if progress is not None:
            assert best is not None
            progress(
                Progress(
                    done=len(evaluations),
                    total=len(candidates),
                    elapsed=time.perf_counter() - started,
                    best=best,
                )
            )

    assert best is not None
    model = encode(x, sample_rate, best.candidate)

    return FitResult(
        model=model,
        quality=Quality.measure(x, model.render()),
        target_mse=target_mse,
        evaluations=evaluations,
        elapsed=time.perf_counter() - started,
        space=space,
        sample_rate=int(sample_rate),
    )


# ============================================================================
# Grouping and evaluation
# ============================================================================


def _group(candidates: Sequence[Candidate]) -> list[tuple[StftSpec, int, list[Candidate]]]:
    """Regroup candidates by what they can share: one analysis, one bin set."""
    groups: dict[tuple[StftSpec, int], list[Candidate]] = {}
    for candidate in candidates:
        groups.setdefault((candidate.spec, candidate.n_components), []).append(candidate)
    return [(spec, k, group) for (spec, k), group in groups.items()]


def _run_groups(x, groups, jobs: int) -> Iterator[list[Evaluation]]:
    if jobs == 1:
        for spec, k, group in groups:
            yield evaluate_group(x, spec, k, group)
        return

    workers = jobs if jobs > 0 else (os.cpu_count() or 1)
    # Longest groups first, so the tail of the search is not one slow task.
    ordered = sorted(groups, key=lambda g: -len(g[2]))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(evaluate_group, x, spec, k, group) for spec, k, group in ordered]
        for future in as_completed(futures):
            yield future.result()


def evaluate_group(
    x: np.ndarray, spec: StftSpec, n_components: int, candidates: Sequence[Candidate]
) -> list[Evaluation]:
    """Measure every candidate that shares this analysis and this bin set.

    The analysis, the bin selection and each codec's factorization are done
    once here and then reused; what is left per candidate is coding the block
    at one setting, the inverse transform, and the comparison.
    """
    spectrogram = analyze(x, spec)
    n_frames = spectrogram.shape[1]

    bins = np.sort(energy_order(spectrogram)[:n_components]).astype(np.int32)
    block = to_components(spectrogram, bins, spec)
    prepared = {name: codec_for(name).prepare(block) for name in {c.codec for c in candidates}}

    model_spectrogram = np.zeros((spec.n_bins, n_frames), dtype=np.complex128)
    evaluations = []
    for candidate in candidates:
        codec = candidate.codec_class
        arrays = codec.encode(
            prepared[candidate.codec], candidate.codec_param, candidate.phase_stride
        )
        model_spectrogram[bins] = from_components(
            codec.decode(arrays, n_frames), bins, spec
        )

        evaluations.append(
            Evaluation(
                candidate=candidate,
                quality=Quality.measure(x, synthesize(model_spectrogram, spec, x.size)),
                n_scalars=candidate.n_scalars(n_frames),
                n_frames=n_frames,
            )
        )
    return evaluations
