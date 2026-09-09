"""Every hit of ONE drum. The unit the fitter consumes.

    SampleSet (one drum)
      |-- shared:   DrumParams        <- f_static, t60, tension   (fit once, frozen)
      \\-- per-hit:  excitation params <- gain[], noise level[], contact_time

That split is the whole reason this class exists. Deliberately single-drum: two
drums in one set would tempt the fitter into one shared `DrumParams` across
both, which is physically wrong — `f_static` and `t60` are properties of a
specific physical drum, and a drum does not retune itself between a soft and a
hard stroke.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from ..core.constants import Audio
from .calibration import VelocityCalibration
from .sample import Sample


@dataclass
class SampleSet:
    """All the recorded hits of one drum."""

    drum: str
    samples: list[Sample] = field(default_factory=list)
    sr: int = Audio.DEFAULT_SR
    calibration: VelocityCalibration | None = None

    #: Paths named by the manifest whose audio was not on disk, after a
    #: `load_all(skip_missing=True)`. Empty otherwise.
    missing: list[str] = field(default_factory=list)

    #: Longest decay the set is expected to support. From the reference floor
    #: tom: its slowest mode measured t60 ~= 2.3 s, and a set whose longest
    #: usable window is shorter than that cannot fit it.
    EXPECTED_SLOWEST_T60: float = 2.3

    #: Full MIDI controller range. A set that stops short of either end has no
    #: evidence there, and the fitted velocity curve extrapolates rather than
    #: interpolates — which is the harder question and the one more likely to
    #: be wrong. Reported when the shortfall exceeds CONTROLLER_MARGIN.
    CONTROLLER_RANGE: tuple[float, float] = (1.0, 127.0)
    CONTROLLER_MARGIN: float = 8.0

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        drum: str,
        pattern: str | None = None,
        sr: int = Audio.DEFAULT_SR,
        glob: str = "*.wav",
    ) -> "SampleSet":
        """Every WAV in a directory whose filename parses, as one drum's set."""
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"not a directory: {directory}")

        sample_set = cls(drum=drum, sr=sr)
        for path in sorted(directory.glob(glob)):
            try:
                sample_set.add(Sample.from_filename(path, pattern, drum=drum))
            except ValueError:
                continue  # not a sample filename; leave it alone
        return sample_set

    @classmethod
    def from_manifest(cls, path: str | Path) -> "SampleSet":
        """Load from a JSON or CSV manifest.

        Preferred over filename parsing once metadata gets richer than drum and
        velocity: a regex is a bad place to record that take 3 happened after
        the drum was retuned.
        """
        path = Path(path)
        if path.suffix.lower() == ".csv":
            return cls._from_csv(path)
        return cls._from_json(path)

    @classmethod
    def _from_json(cls, path: Path) -> "SampleSet":
        data = json.loads(path.read_text(encoding="utf-8"))
        samples = []
        for entry in data.get("samples", []):
            entry = dict(entry)
            sample_path = Path(entry["path"])
            if not sample_path.is_absolute():
                entry["path"] = str(path.parent / sample_path)
            samples.append(Sample.from_dict(entry))
        sample_set = cls(
            drum=str(data.get("drum") or (samples[0].drum if samples else "")),
            sr=int(data.get("sr", Audio.DEFAULT_SR)),
        )
        for sample in samples:
            sample_set.add(sample)
        return sample_set

    @classmethod
    def _from_csv(cls, path: Path) -> "SampleSet":
        rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
        if not rows:
            raise ValueError(f"manifest {path} has no rows")
        root = path.parent
        sample_set = cls(drum=str(rows[0].get("drum", "")))
        for row in rows:
            entry = {key: value for key, value in row.items() if value not in ("", None)}
            entry["path"] = str(root / entry["path"]) if not Path(
                entry["path"]
            ).is_absolute() else entry["path"]
            sample_set.add(Sample.from_dict(entry))
        return sample_set

    def add(self, sample: Sample) -> None:
        """Raises if sample.drum does not match this set."""
        if sample.drum != self.drum:
            raise ValueError(
                f"sample is drum {sample.drum!r} but this set is {self.drum!r}. "
                "One SampleSet per drum, always — f_static and t60 belong to a "
                "specific physical drum."
            )
        self.samples.append(sample)

    def save_manifest(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "drum": self.drum,
                    "sr": self.sr,
                    "samples": [sample.to_dict() for sample in self.samples],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    # -- preparation ----------------------------------------------------------

    def load_all(self, unload_audio: bool = False,
                 skip_missing: bool = False) -> "SampleSet":
        """Load and quality-check every sample.

        `skip_missing` exists because a manifest and the audio it names are
        separate things. The manifests are committed; 3.5 GB of WAV is not, and
        a partial checkout, an interrupted copy or a single file that failed to
        transfer are all ordinary. Raising on the first one turns "three of 104
        samples are missing" into a stack trace and no fit at all, which is a
        worse answer than fitting the 101 and saying so.
        """
        missing: list[str] = []
        for sample in list(self.samples):
            try:
                sample.load(self.sr)
            except FileNotFoundError:
                if not skip_missing:
                    raise
                missing.append(str(sample.path))
                continue
            if unload_audio:
                # Quality and measured energy survive; the audio does not.
                sample.unload()
        if missing:
            self.samples = [
                sample for sample in self.samples
                if str(sample.path) not in set(missing)
            ]
            self.missing = missing
        return self

    def calibrate(self, mode: str = "energy") -> VelocityCalibration:
        """Fit and apply velocity calibration across the set."""
        if not self.samples:
            raise ValueError(f"drum {self.drum!r} has no samples to calibrate")
        self.load_all()
        self.calibration = VelocityCalibration(mode).fit(self.samples)
        self.calibration.apply(self.samples)
        return self.calibration

    def validate(self, n_bins: int = 8) -> list[str]:
        """Every problem worth knowing before fitting.

        Run this on the real library BEFORE any fitting. Each item is something
        that produces a plausible-looking fit and a wrong drum.
        """
        issues: list[str] = []
        if not self.samples:
            return [f"drum {self.drum!r} has no samples"]

        self.load_all()

        unusable = [s for s in self.samples if s.quality and not s.quality.is_usable]
        for sample in unusable:
            issues.append(f"{sample.path.name}: unusable — {'; '.join(sample.quality.warnings()[:1])}")

        truncated = [s for s in self.samples if s.quality and s.quality.tail_truncated]
        if truncated:
            issues.append(
                f"{len(truncated)} of {len(self.samples)} samples have truncated tails; "
                "their t60 will fit short and nothing about the fit will look wrong"
            )

        longest = self.longest_usable_duration()
        if longest < SampleSet.EXPECTED_SLOWEST_T60:
            issues.append(
                f"longest usable window is {longest:.2f}s, shorter than the "
                f"{SampleSet.EXPECTED_SLOWEST_T60:.1f}s slowest decay expected — "
                "this set cannot fit the fundamental's t60 at all"
            )

        if self.calibration is None:
            self.calibrate()
        if not self.calibration.is_monotone():
            issues.append(
                "velocity calibration is NOT monotone: "
                + "; ".join(
                    f"v{low:.0f}->v{high:.0f} drops {abs(drop):.1f} dB"
                    for low, high, drop in self.calibration.inversions()
                )
                + " — investigate the session, do not fit around it"
            )

        low, high = self.velocity_range()
        covered_low, covered_high = self.controller_coverage()
        controller_low, controller_high = SampleSet.CONTROLLER_RANGE
        if covered_low - controller_low > SampleSet.CONTROLLER_MARGIN:
            issues.append(
                f"nothing recorded below v{covered_low:.0f}; the velocity curve "
                "extrapolates at the quiet end"
            )
        if controller_high - covered_high > SampleSet.CONTROLLER_MARGIN:
            issues.append(
                f"nothing recorded above v{covered_high:.0f} of "
                f"{controller_high:.0f}; the velocity curve extrapolates at the loud "
                "end — which is also where the drum is most nonlinear and where a "
                "hard hit's glide and damping stop behaving like the soft ones"
            )

        coverage = self.velocity_coverage(n_bins)
        empty = np.flatnonzero(coverage == 0)
        if empty.size:
            width = (high - low) / n_bins if n_bins else 0.0
            gaps = ", ".join(
                f"v{low + index * width:.0f}-{low + (index + 1) * width:.0f}"
                for index in empty
            )
            issues.append(
                f"velocity coverage gaps at {gaps}; the fitted velocity curve is "
                "least trustworthy there"
            )

        rates = {sample.sr for sample in self.samples}
        if len(rates) > 1:
            issues.append(f"inconsistent sample rates in one set: {sorted(rates)}")

        spread = self.fundamental_spread_cents()
        if spread is not None and spread > 50.0:
            issues.append(
                f"fundamental varies by {spread:.0f} cents across the set — the drum "
                "was probably retuned mid-session, and one DrumParams cannot "
                "describe both halves"
            )
        return issues

    # -- selection ------------------------------------------------------------

    def by_velocity(self, ascending: bool = True) -> list[Sample]:
        return sorted(self.samples, key=lambda s: s.velocity, reverse=not ascending)

    def nearest_velocity(self, velocity: float) -> Sample:
        if not self.samples:
            raise ValueError(f"drum {self.drum!r} has no samples")
        return min(self.samples, key=lambda s: abs(s.velocity - velocity))

    def usable(self) -> list[Sample]:
        return [s for s in self.samples if s.quality is None or s.quality.is_usable]

    def reference_sample(self) -> Sample:
        """The sample to fit f_static and t60 from, then freeze.

        Picked for fit quality, not loudness: best SNR and longest usable tail,
        which is normally a mid-to-upper velocity. The loudest hit is a bad
        choice — it has the most nonlinear behaviour and is the most likely to
        be clipped.
        """
        candidates = self.usable()
        if not candidates:
            raise ValueError(f"drum {self.drum!r} has no usable samples")
        self.load_all()

        def score(sample: Sample) -> float:
            quality = sample.quality
            if quality is None:
                return 0.0
            tail = min(quality.usable_duration / SampleSet.EXPECTED_SLOWEST_T60, 1.0)
            snr = min(quality.snr_db / 60.0, 1.0)
            # Penalize the extremes of the velocity range from both ends.
            low, high = self.velocity_range()
            span = high - low
            position = (sample.velocity - low) / span if span > 0 else 0.5
            centrality = 1.0 - abs(position - 0.65) / 0.65
            truncation = 0.5 if quality.tail_truncated else 1.0
            return (0.4 * tail + 0.4 * snr + 0.2 * max(centrality, 0.0)) * truncation

        return max(candidates, key=score)

    def filter(
        self, articulation: str | None = None, usable_only: bool = True
    ) -> "SampleSet":
        """A new set with the same drum and a subset of the samples."""
        selected = self.samples
        if usable_only:
            selected = [s for s in selected if s.quality is None or s.quality.is_usable]
        if articulation is not None:
            selected = [s for s in selected if s.articulation == articulation]
        return SampleSet(
            drum=self.drum, samples=list(selected), sr=self.sr, calibration=self.calibration
        )

    # -- statistics -----------------------------------------------------------

    def velocity_range(self) -> tuple[float, float]:
        if not self.samples:
            return (0.0, 0.0)
        velocities = [sample.velocity for sample in self.samples]
        return (float(min(velocities)), float(max(velocities)))

    def controller_coverage(self) -> tuple[float, float]:
        """The controller range the set actually spans.

        Uses each sample's mapped velocity BAND where one exists, not its label.
        A range-mapped layer labelled v120 may cover up to v122, and the
        question here is which controller values have evidence behind them —
        not where the band midpoints happen to fall.
        """
        if not self.samples:
            return (0.0, 0.0)
        lows = [
            sample.velocity_low if sample.velocity_low is not None else sample.velocity
            for sample in self.samples
        ]
        highs = [
            sample.velocity_high if sample.velocity_high is not None else sample.velocity
            for sample in self.samples
        ]
        return (float(min(lows)), float(max(highs)))

    def velocity_coverage(self, n_bins: int = 8) -> np.ndarray:
        """Sample count per velocity bin.

        Sparse regions are where the fitted velocity curve will be least
        trustworthy, and they are invisible once the curve is drawn.
        """
        low, high = self.velocity_range()
        if high <= low:
            return np.array([len(self.samples)] + [0] * (n_bins - 1))
        edges = np.linspace(low, high, n_bins + 1)
        counts, _ = np.histogram([s.velocity for s in self.samples], bins=edges)
        return counts

    def longest_usable_duration(self) -> float:
        durations = [
            sample.quality.usable_duration
            for sample in self.samples
            if sample.quality is not None
        ]
        return float(max(durations)) if durations else 0.0

    def fundamental_spread_cents(self) -> float | None:
        """Cents between the highest and lowest measured fundamental in the set.

        A large spread means the drum was retuned partway through, which breaks
        the one-DrumParams-per-drum assumption the whole module rests on.
        """
        from ..scoring.glide import GlideAnalyzer

        loaded = [s for s in self.samples if s.is_loaded and s.audio.size]
        if len(loaded) < 2:
            return None

        analyzer = GlideAnalyzer(self.sr)
        asymptotes = []
        for sample in loaded:
            track = analyzer.analyze(sample.audio_window())
            if np.isfinite(track.f_asymptote) and track.f_asymptote > 0:
                asymptotes.append(track.f_asymptote)
        if len(asymptotes) < 2:
            return None
        return float(1200.0 * np.log2(max(asymptotes) / min(asymptotes)))

    def summary(self) -> str:
        low, high = self.velocity_range()
        lines = [
            f"SampleSet {self.drum!r}: {len(self.samples)} samples, "
            f"v{low:.0f}-v{high:.0f} (covers v{self.controller_coverage()[0]:.0f}-"
            f"v{self.controller_coverage()[1]:.0f}), sr={self.sr}",
            f"  coverage: {self.velocity_coverage().tolist()}",
            f"  longest usable window: {self.longest_usable_duration():.2f}s",
        ]
        if self.calibration is not None and self.calibration.is_fitted:
            lines.append(f"  calibration: {self.calibration}")
        return "\n".join(lines)

    # -- splits ---------------------------------------------------------------

    def split(
        self, holdout: Sequence[float] | float = 0.2, stratify_by_velocity: bool = True
    ) -> tuple["SampleSet", "SampleSet"]:
        """Train / holdout split.

        Holds out INTERIOR velocities, not the extremes. The test that matters
        is whether the fitted velocity curve interpolates to strengths that were
        never recorded; holding out the endpoints tests extrapolation instead,
        which is a different and much easier-to-fail question.

        Pass an explicit list of velocity labels to hold out specific takes.
        """
        ordered = self.by_velocity()
        if not ordered:
            return self.filter(usable_only=False), SampleSet(self.drum, [], self.sr)

        if not isinstance(holdout, (int, float)):
            wanted = {float(value) for value in holdout}
            held = [s for s in ordered if s.velocity in wanted]
            kept = [s for s in ordered if s.velocity not in wanted]
        elif stratify_by_velocity:
            count = max(1, int(round(len(ordered) * float(holdout))))
            interior = list(range(1, len(ordered) - 1))
            if not interior:
                interior = list(range(len(ordered)))
            # Spread the held-out indices evenly through the interior.
            picks = {
                interior[int(round(position))]
                for position in np.linspace(0, len(interior) - 1, min(count, len(interior)))
            }
            held = [sample for index, sample in enumerate(ordered) if index in picks]
            kept = [sample for index, sample in enumerate(ordered) if index not in picks]
        else:
            count = max(1, int(round(len(ordered) * float(holdout))))
            held, kept = ordered[-count:], ordered[:-count]

        return (
            SampleSet(self.drum, kept, self.sr, self.calibration),
            SampleSet(self.drum, held, self.sr, self.calibration),
        )

    def leave_one_out(self) -> Iterator[tuple["SampleSet", Sample]]:
        for index, sample in enumerate(self.samples):
            rest = self.samples[:index] + self.samples[index + 1 :]
            yield SampleSet(self.drum, rest, self.sr, self.calibration), sample

    # -- dunder ---------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)

    def __getitem__(self, index: int) -> Sample:
        return self.samples[index]
