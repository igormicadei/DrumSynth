"""Fitting one drum across every velocity it was recorded at.

The search is the same shape as the per-hit one: enumerate representations,
build each, render it and compare against the recordings. What differs is that
a velocity model has to answer three separate questions, and only one of them
can be asked with a waveform comparison.

    reconstruction  render the model at a recorded velocity and compare it,
                    sample for sample, against that recording. This is the
                    target the search optimizes, and it is meaningful because
                    the model's phase at that velocity came from that take.

    generalization  the same drum, same velocity, a *different* strike. Its
                    phase is unrelated, so a waveform comparison would measure
                    nothing; the model is judged on magnitude alone.

    interpolation   velocities that were never recorded. Fit the field again
                    with every other layer removed, and see how well it
                    predicts the ones it was not shown — again on magnitude,
                    for the same reason.

A fit reports all three. A model that reconstructs perfectly and generalizes
badly has memorized four takes; the numbers say so rather than leaving it to
be discovered by ear.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field as dataclass_field
from typing import Callable, Iterator, Sequence

import numpy as np

from ..fitting.metrics import relative_mse
from ..fitting.pareto import pareto_frontier, preferred
from ..spectral.stft import StftSpec
from . import field as field_codecs
from .layers import VelocityLayers
from .model import InstrumentAnalysis, InstrumentCandidate, InstrumentModel

#: Default quality target for the reconstruction of a recorded velocity.
DEFAULT_TARGET_MSE = 1e-4

#: How many velocities the search renders per candidate. The chosen model is
#: then measured on every one of them.
PROBE_LAYERS = 6


def magnitude_error(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Relative squared error between two magnitude fields.

    The measure for everything phase cannot be compared on: a different strike,
    or a velocity that was never played.
    """
    reference = np.abs(np.asarray(reference))
    estimate = np.abs(np.asarray(estimate))
    energy = float(np.sum(reference**2))
    if energy <= 0.0:
        return 0.0
    return float(np.sum((reference - estimate) ** 2) / energy)


@dataclass(frozen=True)
class InstrumentSearchSpace:
    """The grid of velocity models to try."""

    n_ffts: tuple[int, ...] = (1024, 2048, 4096)
    overlaps: tuple[int, ...] = (2,)
    components: tuple[int, ...] = (64, 128, 256)
    field_ranks: tuple[int, ...] = (2, 4, 6, 8, 12)
    pattern_ranks: tuple[int, ...] = (8, 16, 32)
    donor_ranks: tuple[int, ...] = (8, 16, 32)

    @classmethod
    def quick(cls) -> "InstrumentSearchSpace":
        return cls(
            n_ffts=(1024, 2048),
            components=(64, 128),
            field_ranks=(4, 8),
            pattern_ranks=(16,),
            donor_ranks=(16,),
        )

    @classmethod
    def full(cls) -> "InstrumentSearchSpace":
        return cls(
            n_ffts=(512, 1024, 2048, 4096, 8192),
            overlaps=(2, 4),
            components=(32, 64, 128, 192, 256, 384, 512),
            field_ranks=(1, 2, 3, 4, 6, 8, 12, 16, 24),
            pattern_ranks=(4, 8, 16, 24, 32, 48),
            donor_ranks=(4, 8, 16, 24, 32, 48),
        )

    def specs(self) -> list[StftSpec]:
        return [
            StftSpec(n_fft, n_fft // overlap)
            for n_fft in self.n_ffts
            for overlap in self.overlaps
            if n_fft % overlap == 0
        ]

    def candidates(
        self, layers: VelocityLayers, n_donors: int = 0
    ) -> list[InstrumentCandidate]:
        """Every candidate worth evaluating for this instrument.

        `n_donors` is not searched over: it is the one choice the fit cannot
        make honestly for you. Every layer donating is what reproduces every
        recorded velocity, and fewer donors is a decision to accept that
        velocities far from a donor play back one strike's phase under another
        strike's spectrum — cheaper, and audible.
        """
        seen: dict[InstrumentCandidate, None] = {}
        for spec in self.specs():
            frames = spec.n_frames(layers.n_samples)
            for k in self.components:
                if k > spec.n_bins:
                    continue
                for candidate in self._for_bins(
                    spec, k, layers.n_layers, frames, n_donors
                ):
                    seen.setdefault(candidate, None)
        return list(seen)

    def _for_bins(
        self, spec: StftSpec, k: int, n_layers: int, n_frames: int, n_donors: int
    ) -> Iterator[InstrumentCandidate]:
        shape = (n_layers, k, n_frames)
        fields: list[tuple[str, int, int]] = [("full", 0, 0)]
        for rank in self.field_ranks:
            if rank > n_layers:
                continue
            clamped, _ = field_codecs.codec_for("velocity").clamp(shape, rank, 0)
            fields.append(("velocity", clamped, 0))
            for pattern_rank in self.pattern_ranks:
                fields.append(
                    ("separable", *field_codecs.codec_for("separable").clamp(shape, rank, pattern_rank))
                )

        donors: list[tuple[str, int]] = [("exact", 0)]
        for rank in self.donor_ranks:
            if rank > min(k, n_frames):
                continue
            donors.append(("lowrank", rank))
            donors.append(("phase", rank))

        for field_codec, field_rank, pattern_rank in dict.fromkeys(fields):
            for donor_codec, donor_rank in dict.fromkeys(donors):
                yield InstrumentCandidate(
                    n_fft=spec.n_fft,
                    hop=spec.hop,
                    n_components=k,
                    field_codec=field_codec,
                    field_rank=field_rank,
                    pattern_rank=pattern_rank,
                    donor_codec=donor_codec,
                    donor_rank=donor_rank,
                    n_donors=n_donors,
                )


@dataclass(frozen=True)
class InstrumentEvaluation:
    """One candidate, built and rendered at a spread of velocities.

    `relative_mse` is the mean over those velocities, each weighted equally —
    not pooled by energy, which would let the quietest layers be anything at
    all and still look fine behind the loudest.
    """

    candidate: InstrumentCandidate
    relative_mse: float
    worst_mse: float
    n_scalars: int
    n_frames: int

    def to_dict(self) -> dict:
        return {
            "candidate": self.candidate.to_dict(),
            "relative_mse": self.relative_mse,
            "worst_mse": self.worst_mse,
            "n_scalars": self.n_scalars,
            "n_frames": self.n_frames,
        }


@dataclass(frozen=True)
class LayerReport:
    """How the model did at one recorded velocity."""

    velocity: float
    reconstruction_mse: float
    generalization_mse: float | None
    is_donor: bool

    def to_dict(self) -> dict:
        return {
            "velocity": self.velocity,
            "reconstruction_mse": self.reconstruction_mse,
            "generalization_mse": self.generalization_mse,
            "is_donor": self.is_donor,
        }


@dataclass(frozen=True)
class Progress:
    done: int
    total: int
    elapsed: float
    best: InstrumentEvaluation


@dataclass
class InstrumentFitResult:
    """The chosen velocity model, and everything measured about it."""

    model: InstrumentModel
    layers: list[LayerReport]
    interpolation_mse: float | None
    target_mse: float
    evaluations: list[InstrumentEvaluation]
    elapsed: float
    n_recordings: int
    averaged: bool = False
    space: InstrumentSearchSpace = dataclass_field(default_factory=InstrumentSearchSpace)

    @property
    def candidate(self) -> InstrumentCandidate:
        return self.model.candidate

    @property
    def reconstruction_mse(self) -> float:
        return float(np.mean([layer.reconstruction_mse for layer in self.layers]))

    @property
    def worst_layer(self) -> LayerReport:
        return max(self.layers, key=lambda layer: layer.reconstruction_mse)

    @property
    def generalization_mse(self) -> float | None:
        measured = [
            layer.generalization_mse
            for layer in self.layers
            if layer.generalization_mse is not None
        ]
        return float(np.mean(measured)) if measured else None

    @property
    def encoded_recordings(self) -> int:
        """How many recordings the model actually holds.

        One per velocity normally; every take when their magnitudes were
        averaged. Comparing the model against recordings that never reached it
        would flatter it.
        """
        return self.n_recordings if self.averaged else len(self.layers)

    @property
    def target_reached(self) -> bool:
        return self.reconstruction_mse <= self.target_mse

    @property
    def frontier(self) -> list[InstrumentEvaluation]:
        return pareto_frontier(self.evaluations)

    def summary(self) -> str:
        model = self.model
        low, high = model.velocity_range
        lines = [
            f"instrument       {model.name}",
            f"velocities       {model.velocities.size} layers, {low:g} to {high:g}, "
            f"from {self.n_recordings} recordings",
            f"representation   {self.candidate.label()}",
            f"model            {model.n_scalars} numbers, {model.n_bytes() / 1024:.1f} kB, "
            f"{1.0 / model.compression_ratio(self.encoded_recordings):.1f}x smaller than "
            f"the {self.encoded_recordings} recordings it holds, as 16-bit PCM",
            f"anchored on      {'the mean of every take' if self.averaged else f'take {0}'}",
            f"reconstruction   {self.reconstruction_mse:.3e} mean, "
            f"{self.worst_layer.reconstruction_mse:.3e} at velocity "
            f"{self.worst_layer.velocity:g} — target {self.target_mse:.1e} "
            f"{'reached' if self.target_reached else 'NOT REACHED'}",
        ]
        if self.generalization_mse is not None:
            lines.append(
                f"generalization   {self.generalization_mse:.3e} against strikes the "
                f"model never saw"
            )
        if self.interpolation_mse is not None:
            lines.append(
                f"interpolation    {self.interpolation_mse:.3e} at held-out velocities"
            )
        lines.append(
            f"searched         {len(self.evaluations)} candidates in {self.elapsed:.1f} s"
        )
        return "\n".join(lines)


def fit_instrument(
    layers: VelocityLayers,
    *,
    target_mse: float = DEFAULT_TARGET_MSE,
    space: InstrumentSearchSpace | None = None,
    take: int = 0,
    average: bool = False,
    n_donors: int = 0,
    progress: Callable[[Progress], None] | None = None,
    max_candidates: int | None = None,
) -> InstrumentFitResult:
    """Search for the smallest velocity model that reconstructs every layer.

    `take` picks which round robin each layer is built from. The others are not
    thrown away — they are what `generalization` is measured against. With
    `average`, the magnitudes come from every take instead of that one, which
    predicts an unseen strike better and reproduces no particular strike
    exactly; the reconstruction numbers then carry that gap and cannot reach
    zero.

    `n_donors` is fixed for the whole search rather than searched over (0, the
    default, means every layer donates its phase). Candidates are rendered at a
    spread of velocities rather than all of them, and with sparse donors those
    numbers are optimistic — but every candidate shares the same donor layout,
    so the comparison between them holds, and the model that wins is then
    measured at every recorded velocity.
    """
    space = space or InstrumentSearchSpace()
    candidates = space.candidates(layers, n_donors)
    if max_candidates is not None:
        candidates = candidates[:max_candidates]
    if not candidates:
        raise ValueError("the search space is empty for this instrument")

    started = time.perf_counter()
    evaluations: list[InstrumentEvaluation] = []
    best: InstrumentEvaluation | None = None
    probes = _probe_layers(layers.n_layers, PROBE_LAYERS)

    for spec, k, group_candidates in _grouped(candidates):
        analysis = _analyse(layers, spec, k, take, average)
        references = [layers.audio(index, take) for index in probes]

        for candidate in group_candidates:
            model = analysis.build(candidate)
            errors = [
                relative_mse(reference, model.render(layers.velocities[index]))
                for index, reference in zip(probes, references)
            ]
            evaluations.append(
                InstrumentEvaluation(
                    candidate=candidate,
                    relative_mse=float(np.mean(errors)),
                    worst_mse=float(np.max(errors)),
                    n_scalars=model.n_scalars,
                    n_frames=model.n_frames,
                )
            )
            best = (
                evaluations[-1]
                if best is None
                else preferred(best, evaluations[-1], target_mse)
            )

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
    model = _build(layers, best.candidate, take, average)

    return InstrumentFitResult(
        model=model,
        layers=measure_layers(model, layers, take),
        interpolation_mse=interpolation_error(layers, best.candidate, take, average),
        averaged=average,
        target_mse=target_mse,
        evaluations=evaluations,
        elapsed=time.perf_counter() - started,
        n_recordings=layers.n_recordings,
        space=space,
    )


def _probe_layers(n_layers: int, count: int) -> list[int]:
    """A spread of velocities to measure candidates on.

    Deliberately offset from the even spacing `donors.choose` uses. Landing the
    probes on the donors would measure a sparse donor layout at its best
    velocities and call the result the average.
    """
    if n_layers <= count:
        return list(range(n_layers))
    steps = (np.arange(count) + 0.5) / count
    return sorted({int(round(step * (n_layers - 1))) for step in steps})


def _grouped(
    candidates: Sequence[InstrumentCandidate],
) -> list[tuple[StftSpec, int, list[InstrumentCandidate]]]:
    """Regroup by what an analysis can be shared across."""
    groups: dict[tuple[StftSpec, int], list[InstrumentCandidate]] = {}
    for candidate in candidates:
        groups.setdefault((candidate.spec, candidate.n_components), []).append(candidate)
    return [(spec, k, group) for (spec, k), group in groups.items()]


def _analyse(
    layers: VelocityLayers,
    spec: StftSpec,
    n_components: int,
    take: int,
    average: bool = False,
) -> InstrumentAnalysis:
    bins = layers.select_bins(spec, n_components)
    magnitudes, blocks = layers.analyse(spec, bins, take=take, average=average)
    return InstrumentAnalysis(
        name=layers.name,
        sample_rate=layers.sample_rate,
        n_samples=layers.n_samples,
        velocities=layers.velocities,
        bins=bins,
        magnitudes=magnitudes,
        blocks=blocks,
    )


def _build(
    layers: VelocityLayers,
    candidate: InstrumentCandidate,
    take: int,
    average: bool = False,
) -> InstrumentModel:
    return _analyse(
        layers, candidate.spec, candidate.n_components, take, average
    ).build(candidate)


def measure_layers(
    model: InstrumentModel, layers: VelocityLayers, take: int = 0
) -> list[LayerReport]:
    """Reconstruction at every recorded velocity, and generalization to other takes."""
    spec = model.candidate.spec
    reports = []

    for index, velocity in enumerate(layers.velocities):
        rendered = model.render(velocity)
        modelled = np.abs(model.field.at(velocity))

        others = [
            other for other in range(layers.takes(index)) if other != take % layers.takes(index)
        ]
        generalization = (
            float(
                np.mean(
                    [
                        magnitude_error(
                            layers.block_of(index, other, spec, model.bins), modelled
                        )
                        for other in others
                    ]
                )
            )
            if others
            else None
        )

        reports.append(
            LayerReport(
                velocity=float(velocity),
                reconstruction_mse=relative_mse(layers.audio(index, take), rendered),
                generalization_mse=generalization,
                is_donor=bool(
                    np.any(np.isclose(model.donor_velocities, velocity))
                ),
            )
        )
    return reports


def interpolation_error(
    layers: VelocityLayers,
    candidate: InstrumentCandidate,
    take: int = 0,
    average: bool = False,
) -> float | None:
    """Refit the field without every other layer, and test it on those layers.

    Only the magnitude field is rebuilt: the question is whether a velocity
    between two recorded ones can be predicted, and phase is borrowed rather
    than predicted at any velocity. Layers outside the kept range are left out
    of the score — the field clamps there rather than interpolating, and
    counting a clamp as a failed interpolation would answer a question nobody
    asked.
    """
    if layers.n_layers < 5:
        return None

    bins = layers.select_bins(candidate.spec, candidate.n_components)
    magnitudes, _ = layers.analyse(candidate.spec, bins, take=take, average=average)

    kept = np.arange(0, layers.n_layers, 2)
    held_out = [
        index
        for index in np.setdiff1d(np.arange(layers.n_layers), kept)
        if kept[0] < index < kept[-1]
    ]
    if not held_out:
        return None

    codec = field_codecs.codec_for(candidate.field_codec)
    arrays = codec.encode(
        codec.prepare(magnitudes[kept]), candidate.field_rank, candidate.pattern_rank
    )
    field = field_codecs.MagnitudeField.from_arrays(
        layers.velocities[kept], candidate.field_codec, arrays
    )

    return float(
        np.mean(
            [
                magnitude_error(magnitudes[index], field.at(layers.velocities[index]))
                for index in held_out
            ]
        )
    )


__all__ = [
    "DEFAULT_TARGET_MSE",
    "InstrumentEvaluation",
    "InstrumentFitResult",
    "InstrumentSearchSpace",
    "LayerReport",
    "Progress",
    "fit_instrument",
    "interpolation_error",
    "magnitude_error",
    "measure_layers",
]
