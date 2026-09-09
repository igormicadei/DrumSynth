"""Turning a sample set into something the fitter can consume.

A drum's library is 100+ WAVs across 25 velocity bands with four round-robins
each. The fitter does not want all of them: it wants one representative hit per
velocity layer, chosen for fit quality, and it wants them prepared identically.

Round-robins at the same velocity are alternate takes of the same layer, so
using all four multiplies the work without adding information about how the
drum responds to velocity. One per layer, picked for SNR and tail length, is
the honest choice — and §8.2's "never fit to a single hit" is about velocity
coverage, not about take count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core.constants import Audio
from ..samples.sample import Sample
from ..samples.sample_set import SampleSet
from ..scoring.prep import SignalPrep


@dataclass
class Layer:
    """One velocity layer, prepared and ready to fit."""

    velocity: float
    velocity_normalized: float
    audio: np.ndarray = field(repr=False)
    source: str = ""
    usable_seconds: float = 0.0
    snr_db: float = 0.0

    @property
    def duration(self) -> float:
        return len(self.audio)


@dataclass
class FitTarget:
    """Everything one drum's fit needs."""

    drum: str
    sr: int
    layers: list[Layer]
    reference_index: int = 0
    warnings: list[str] = field(default_factory=list)

    #: Seconds of each hit the fit looks at. Long enough for the slowest decay
    #: the reference showed (t60 ~2.3 s), short enough that a fit is not
    #: dominated by silence.
    DEFAULT_SECONDS: float = 2.5

    #: Where in the velocity range the frequency measurement is taken.
    FREQUENCY_POSITION: float = 0.25

    @property
    def reference(self) -> Layer:
        return self.layers[self.reference_index]

    def frequency_layer(self) -> Layer:
        """The layer f_static is measured from.

        `f_static` is the frequency AT REST. A loud hit is the drum at its
        tightest — the tension feedback holds every mode up to two semitones
        sharp through the first second — so a soft layer is where the rest
        frequency is actually visible. Not the very softest: that one has the
        worst SNR and recovers the fewest partials.
        """
        if len(self.layers) == 1:
            return self.layers[0]
        index = int(round(FitTarget.FREQUENCY_POSITION * (len(self.layers) - 1)))
        return self.layers[max(0, min(index, len(self.layers) - 1))]

    def velocities(self) -> np.ndarray:
        return np.array([layer.velocity_normalized for layer in self.layers])

    def summary(self) -> str:
        low, high = self.layers[0].velocity, self.layers[-1].velocity
        return (
            f"{self.drum}: {len(self.layers)} velocity layers, v{low:.0f}-v{high:.0f}, "
            f"reference v{self.reference.velocity:.0f} "
            f"({self.reference.usable_seconds:.2f}s usable)"
        )


class TargetBuilder:
    """Builds a `FitTarget` from a `SampleSet`, or from audio directly."""

    #: More layers than this costs time without telling you more about the
    #: shape of the velocity curve, which is only two numbers per quantity.
    MAX_LAYERS: int = 10

    #: Below this, stage 3's experiment cannot run and stage 4's two-parameter
    #: curves are fitted through fewer points than they have parameters.
    MIN_LAYERS: int = 3

    def __init__(
        self,
        sr: int = Audio.DEFAULT_SR,
        seconds: float = FitTarget.DEFAULT_SECONDS,
        max_layers: int = MAX_LAYERS,
    ) -> None:
        self.sr = int(sr)
        self.seconds = float(seconds)
        self.max_layers = int(max_layers)
        self.prep = SignalPrep(self.sr)

    # -- from a real library --------------------------------------------------

    def from_sample_set(self, sample_set: SampleSet,
                        progress=None) -> FitTarget:
        report = progress or (lambda _: None)
        report({"step": f"loading {len(sample_set)} samples"})
        named = len(sample_set)
        sample_set.load_all(skip_missing=True)

        usable = [s for s in sample_set if s.quality is None or s.quality.is_usable]
        warnings = []
        if sample_set.missing:
            examples = ", ".join(
                Path(path).name for path in sample_set.missing[:3])
            warnings.append(
                f"{len(sample_set.missing)} of {named} samples named by the "
                f"manifest are not on disk and were left out ({examples}"
                + (", ..." if len(sample_set.missing) > 3 else "")
                + "). The manifests are committed and the WAVs are not, so a "
                "partial checkout looks exactly like this"
            )
        if len(usable) < len(sample_set):
            warnings.append(
                f"{len(sample_set) - len(usable)} of {len(sample_set)} samples are "
                "unusable (clipped, or more than one hit) and were left out"
            )
        if not usable:
            raise ValueError(f"drum {sample_set.drum!r} has no usable samples")

        report({"step": "calibrating velocity"})
        calibration = sample_set.calibrate()
        if not calibration.is_monotone():
            warnings.append(
                "velocity calibration is NOT monotone: a harder hit measured "
                "quieter. Investigate the session before trusting stage 3 — "
                + "; ".join(
                    f"v{low:.0f}->v{high:.0f} drops {abs(drop):.1f} dB"
                    for low, high, drop in calibration.inversions()
                )
            )

        chosen = self._pick_layers(usable)
        report({"step": f"prepared {len(chosen)} velocity layers"})
        if len(chosen) < TargetBuilder.MIN_LAYERS:
            warnings.append(
                f"only {len(chosen)} velocity layer"
                + ("s" if len(chosen) != 1 else "")
                + " survived, and §8.2 says never fit to a single hit: a "
                "parameter set tuned to one recording is tuned to that "
                "recording. Stage 3 cannot run its experiment and stage 4 has "
                "nothing to fit a curve through, so treat the result as a "
                "starting point to edit by hand, not as a fitted drum"
            )

        layers = [self._to_layer(sample) for sample in chosen]
        reference = self._reference_index(layers)
        return FitTarget(
            drum=sample_set.drum, sr=self.sr, layers=layers,
            reference_index=reference, warnings=warnings,
        )

    def _pick_layers(self, samples: list[Sample]) -> list[Sample]:
        """One sample per velocity band, spread evenly across the range.

        Picked for fit quality inside each band — best SNR, longest usable
        tail — for the same reason `SampleSet.reference_sample` does: the
        loudest take is the most nonlinear and the most likely to be clipped.
        """
        by_band: dict[tuple[float, float], list[Sample]] = {}
        for sample in samples:
            low = sample.velocity_low if sample.velocity_low is not None else sample.velocity
            high = sample.velocity_high if sample.velocity_high is not None else sample.velocity
            by_band.setdefault((float(low), float(high)), []).append(sample)

        def quality(sample: Sample) -> float:
            if sample.quality is None:
                return 0.0
            tail = min(sample.quality.usable_duration / 2.3, 1.0)
            return 0.6 * min(sample.quality.snr_db / 60.0, 1.0) + 0.4 * tail

        best = [max(group, key=quality) for _, group in sorted(by_band.items())]
        if len(best) <= self.max_layers:
            return best
        picks = np.linspace(0, len(best) - 1, self.max_layers).round().astype(int)
        return [best[index] for index in sorted(set(picks.tolist()))]

    def _to_layer(self, sample: Sample) -> Layer:
        audio = self.prep.normalize(sample.audio_window(self.seconds), "none")
        wanted = int(self.seconds * self.sr)
        if len(audio) < wanted:
            audio = np.pad(audio, (0, wanted - len(audio)))
        return Layer(
            velocity=float(sample.velocity),
            velocity_normalized=float(
                sample.velocity_normalized
                if sample.velocity_normalized is not None
                else sample.velocity / 127.0
            ),
            audio=audio[:wanted],
            source=sample.path.name,
            usable_seconds=(
                sample.quality.usable_duration if sample.quality else self.seconds
            ),
            snr_db=sample.quality.snr_db if sample.quality else 0.0,
        )

    @staticmethod
    def _reference_index(layers: list[Layer]) -> int:
        """Which layer stage 1 measures the drum from.

        Fit quality, not loudness, and biased toward the upper middle — the
        loudest hit has the most nonlinear behaviour and is the most likely to
        be clipped, and the quietest has the worst SNR.
        """
        if not layers:
            return 0
        scores = []
        for index, layer in enumerate(layers):
            position = index / max(len(layers) - 1, 1)
            centrality = 1.0 - abs(position - 0.65) / 0.65
            tail = min(layer.usable_seconds / 2.3, 1.0)
            snr = min(layer.snr_db / 60.0, 1.0)
            scores.append(0.4 * tail + 0.35 * snr + 0.25 * max(centrality, 0.0))
        return int(np.argmax(scores))

    # -- from audio, for testing and for one-off references -------------------

    def from_audio(self, drum: str, hits: list[tuple[float, np.ndarray]]) -> FitTarget:
        """`hits` is (velocity 0-127, audio). Velocities are normalized linearly,
        because without measured energy there is nothing better to use."""
        wanted = int(self.seconds * self.sr)
        velocities = np.array([velocity for velocity, _ in hits], dtype=float)
        span = velocities.max() - velocities.min()
        layers = []
        for velocity, audio in hits:
            audio = np.asarray(audio, dtype=np.float64)
            if len(audio) < wanted:
                audio = np.pad(audio, (0, wanted - len(audio)))
            layers.append(
                Layer(
                    velocity=float(velocity),
                    velocity_normalized=float(
                        (velocity - velocities.min()) / span if span > 0 else 1.0
                    ),
                    audio=audio[:wanted],
                    source="in memory",
                    usable_seconds=self.seconds,
                    snr_db=60.0,
                )
            )
        layers.sort(key=lambda item: item.velocity)
        return FitTarget(
            drum=drum, sr=self.sr, layers=layers,
            reference_index=TargetBuilder._reference_index(layers),
        )


class DrumCatalogue:
    """Which drums are available to fit.

    Only membrane drums. Cymbals break the modal bank at the physics level
    (§9) — a struck cymbal cascades energy from low modes into high ones, which
    a linear bank cannot do at any setting — so they are not offered here even
    though the library stores them.
    """

    @staticmethod
    def available(metadata_dir: str | Path = "data/metadata") -> list[dict]:
        directory = Path(metadata_dir)
        if not directory.is_dir():
            return []

        import json

        entries = []
        for path in sorted(directory.glob("drums-*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            rows = data.get("samples", [])
            if not rows:
                continue
            velocities = [row["velocity"] for row in rows]
            entries.append({
                "drum": data.get("drum", path.stem),
                "manifest": str(path),
                "samples": len(rows),
                "velocity_low": min(velocities),
                "velocity_high": max(velocities),
                "bands": len({(row["velocity_low"], row["velocity_high"]) for row in rows}),
            })
        return entries

    @staticmethod
    def load(manifest: str | Path, sr: int = Audio.DEFAULT_SR) -> SampleSet:
        sample_set = SampleSet.from_manifest(manifest)
        sample_set.sr = sr
        return sample_set
