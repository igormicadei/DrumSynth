"""One recorded hit.

A sample carries a velocity label, but velocity does NOT select a different
drum — it selects a different excitation of the same drum. That distinction is
the reason this module exists and the reason `Sample` holds no `DrumParams`.

Audio and descriptors are lazy and cached: analysis is expensive and a fitter
touches each sample many times.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np

from ..core.audio_io import AudioIO
from ..core.constants import Audio, Decibels
from ..scoring.prep import SignalPrep
from .quality import QualityChecker, SampleQuality


@dataclass
class Sample:
    """One recorded hit, plus everything measured from it on load."""

    path: Path
    drum: str  # "floor_tom_16", "snare", "kick"
    velocity: float  # NOMINAL label as recorded (MIDI 0-127, or whatever you used)

    # -- optional metadata; seams for later, all ignored by the current fitter --
    articulation: str = "center"  # center | edge | rimshot | cross_stick | ...
    strike_position: float | None = None  # 0.0 center -> 1.0 rim, if known
    take: int = 0  # multiple takes at the same velocity
    tuning: str = ""  # session tuning label, if the drum was retuned
    notes: str = ""

    # -- populated on load ----------------------------------------------------
    sr: int = Audio.DEFAULT_SR
    quality: SampleQuality | None = None
    measured_energy_db: float | None = None  # actual energy; calibrates the label
    velocity_normalized: float | None = None  # 0-1 after calibration; FIT AGAINST THIS

    # -- caches ---------------------------------------------------------------
    _audio: np.ndarray | None = field(default=None, repr=False, compare=False)
    _descriptors: object | None = field(default=None, repr=False, compare=False)

    #: Default filename grammar: drum_v<velocity>[_take<n>].wav
    FILENAME_PATTERN: ClassVar[str] = (
        r"^(?P<drum>.+?)[_-]v(?P<velocity>\d+(?:\.\d+)?)"
        r"(?:[_-]take(?P<take>\d+))?$"
    )

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    # -- loading --------------------------------------------------------------

    def load(self, sr: int | None = None, force: bool = False) -> np.ndarray:
        """Read WAV, sum to mono, resample, trim to onset, measure quality.

        Trimming to onset here rather than in the fitter guarantees every sample
        shares t=0, which every decay fit and envelope comparison depends on.

        Does NOT normalize amplitude. Level is the signal being fitted — a
        normalized sample set makes every velocity look identical, which is the
        one thing the excitation fit must be able to see.
        """
        if self._audio is not None and not force and (sr is None or sr == self.sr):
            return self._audio

        target_sr = int(sr or self.sr)
        signal, actual_sr = AudioIO.read(self.path, sr=target_sr)
        self.sr = actual_sr

        prep = SignalPrep(self.sr)
        trimmed = prep.trim_to_onset(signal)

        # Quality is measured on the untrimmed file: pre-onset silence and the
        # position of a second onset are both properties of the file, not of the
        # trimmed hit.
        self.quality = QualityChecker(self.sr).check(signal)
        self._audio = trimmed
        self._descriptors = None
        self.measured_energy_db = Decibels.from_power(self.energy())
        return self._audio

    @property
    def audio(self) -> np.ndarray:
        """Cached, onset-trimmed, un-normalized."""
        return self._audio if self._audio is not None else self.load()

    @property
    def is_loaded(self) -> bool:
        return self._audio is not None

    def audio_window(self, seconds: float | None = None) -> np.ndarray:
        """Audio truncated to `seconds`, or to quality.usable_duration if None.

        The default is the important one. Fitting a decay slope past the noise
        floor measures the recording, not the drum.
        """
        audio = self.audio
        if seconds is None:
            seconds = self.quality.usable_duration if self.quality else 0.0
        if not seconds or seconds <= 0:
            return audio
        return audio[: min(len(audio), int(round(seconds * self.sr)))]

    def unload(self) -> None:
        """Drop cached audio and descriptors. For large libraries."""
        self._audio = None
        self._descriptors = None

    # -- analysis -------------------------------------------------------------

    def descriptors(self, analyzer=None):
        """Cached `SoundDescriptors`, computed over the usable window only."""
        if self._descriptors is not None:
            return self._descriptors

        from ..scoring.analyzer import Analyzer

        analyzer = analyzer or Analyzer(self.sr)
        self._descriptors = analyzer.analyze(
            analyzer.prep.normalize(self.audio_window(), "none")
        )
        return self._descriptors

    def energy(self) -> float:
        """Total signal energy over the usable window.

        The physical stand-in for the velocity label, and computed over the
        usable window so a long, quiet, mostly-noise tail does not add energy
        that the drum never produced.
        """
        window = self.audio_window()
        return float(np.sum(window**2)) if window.size else 0.0

    def peak(self) -> float:
        audio = self.audio
        return float(np.max(np.abs(audio))) if audio.size else 0.0

    def peak_db(self) -> float:
        return float(Decibels.from_amplitude(self.peak()))

    def peak_time(self) -> float:
        """Seconds from onset to envelope maximum. ~10 ms for a membrane drum."""
        from ..scoring.envelope import EnvelopeAnalyzer

        audio = self.audio
        if audio.size == 0:
            return 0.0
        return float(EnvelopeAnalyzer(self.sr, 0.001).analyze(audio).peak_time)

    def duration(self) -> float:
        return len(self.audio) / self.sr if self.is_loaded else 0.0

    # -- io -------------------------------------------------------------------

    @classmethod
    def from_filename(
        cls, path: str | Path, pattern: str | None = None, drum: str | None = None
    ) -> "Sample":
        """Parse drum / velocity / take out of a filename.

        e.g. "floor_tom_16_v100_take2.wav" -> drum, velocity=100, take=2
        Supply `pattern` as a named-group regex to override the default.

        Filename parsing is a convenience for a first pass. Once metadata gets
        richer than drum + velocity, use a manifest — a regex is a bad place to
        keep the fact that take 3 was recorded after the drum was retuned.
        """
        path = Path(path)
        match = re.match(pattern or cls.FILENAME_PATTERN, path.stem)
        if match is None:
            raise ValueError(
                f"cannot parse {path.name!r} with pattern "
                f"{pattern or cls.FILENAME_PATTERN!r}; expected something like "
                "'floor_tom_16_v100_take2.wav'"
            )

        groups = match.groupdict()
        return cls(
            path=path,
            drum=drum or groups.get("drum") or path.stem,
            velocity=float(groups["velocity"]),
            take=int(groups.get("take") or 0),
            articulation=groups.get("articulation") or "center",
        )

    def to_dict(self) -> dict:
        """Metadata only — never the audio."""
        data = {
            key: value
            for key, value in asdict(self).items()
            if not key.startswith("_")
        }
        data["path"] = str(self.path)
        data["quality"] = asdict(self.quality) if self.quality else None
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Sample":
        quality = data.get("quality")
        sample = cls(
            path=Path(data["path"]),
            drum=str(data["drum"]),
            velocity=float(data["velocity"]),
            articulation=str(data.get("articulation", "center")),
            strike_position=data.get("strike_position"),
            take=int(data.get("take", 0)),
            tuning=str(data.get("tuning", "")),
            notes=str(data.get("notes", "")),
            sr=int(data.get("sr", Audio.DEFAULT_SR)),
            measured_energy_db=data.get("measured_energy_db"),
            velocity_normalized=data.get("velocity_normalized"),
        )
        if quality:
            sample.quality = SampleQuality(**{
                key: value for key, value in quality.items()
                if key in SampleQuality.__dataclass_fields__
            })
        return sample

    def __str__(self) -> str:
        velocity = (
            f"v{self.velocity:.0f}"
            if self.velocity_normalized is None
            else f"v{self.velocity:.0f} ({self.velocity_normalized:.3f} norm)"
        )
        return f"{self.drum} {velocity} take{self.take}  {self.path.name}"
