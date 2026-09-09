"""Every fit, kept.

A fit takes minutes and produces a drum you cannot judge in one listen. The
useful comparison is between runs — this drum with 20 modes against the same
drum with 34, the fit before the tension stage was corrected against the one
after — and that comparison is impossible if each run overwrites the last.

So a run is a directory, named by drum and timestamp, holding everything needed
to reopen it later without refitting:

    out/runs/toms-stereo-tom3/2026-09-09T14-22-05/
        run.json          settings, timings, warnings, the whole summary
        params.json       the DrumParams, loadable by the live synth
        score.json        the ScoreCard, per component and per layer
        audio/            one generated WAV per velocity layer
        reference/        the sample each was fitted against

The audio is written because it is the part that cannot be regenerated cheaply:
re-rendering needs the parameters (fine) but the reference needs the 3.5 GB
library, which may not be where it was. A run directory is self-contained and
survives the samples moving.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from ..core.audio_io import AudioIO
from ..core.constants import Audio
from ..synth.params import DrumParams


@dataclass
class RunRecord:
    """One stored fit, read back from disk."""

    directory: Path
    drum: str
    started: str                     # ISO 8601, the directory's own name
    summary: dict = field(default_factory=dict)
    score: dict = field(default_factory=dict)

    #: Filled by `params()`, which reads lazily — a run list should not parse
    #: thirty parameter files to draw a table.
    _params: DrumParams | None = None

    @property
    def label(self) -> str:
        return f"{self.drum} · {self.started.replace('T', ' ')}"

    @property
    def total(self) -> float:
        return float(self.score.get("total", float("nan")))

    @property
    def stft_loss(self) -> float:
        return float(self.score.get("stft_loss", float("nan")))

    @property
    def elapsed(self) -> float:
        return float(self.summary.get("elapsed", 0.0))

    @property
    def modes(self) -> int:
        return int(self.summary.get("modes", 0))

    @property
    def device(self) -> str:
        return str(self.summary.get("device", "")) or "cpu"

    def params(self) -> DrumParams:
        if self._params is None:
            self._params = DrumParams.load(self.directory / "params.json")
        return self._params

    def layers(self) -> list[dict]:
        """(velocity, generated path, reference path) per stored layer."""
        return list(self.summary.get("stored_layers", []))

    def audio(self, velocity: float, sr: int = Audio.DEFAULT_SR
              ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """(generated, reference) for one velocity, or Nones when not stored."""
        for layer in self.layers():
            if abs(float(layer["velocity"]) - float(velocity)) < 1e-6:
                return (
                    self._read(self.directory / layer["generated"], sr),
                    self._read(self.directory / layer["reference"], sr)
                    if layer.get("reference") else None,
                )
        return None, None

    @staticmethod
    def _read(path: Path, sr: int) -> np.ndarray | None:
        if not Path(path).exists():
            return None
        audio, _ = AudioIO.read(path, sr=sr)
        return audio

    def to_row(self) -> dict:
        return {
            "run": self.started, "drum": self.drum, "score": self.total,
            "stft dB": self.stft_loss, "modes": self.modes,
            "seconds": self.elapsed, "device": self.device,
        }


class RunStore:
    """The directory of stored runs, and the writing of new ones."""

    DEFAULT_ROOT = Path("out/runs")

    #: Overrides the root for a whole process. The worker runs as a subprocess
    #: of the app and both have to agree on where runs live; an environment
    #: variable is the one channel they already share. Tests use it to keep
    #: their runs out of the working tree.
    ROOT_VARIABLE = "DRUMSYNTH_RUNS"

    #: Directory names are timestamps, so they sort chronologically as strings.
    STAMP = "%Y-%m-%dT%H-%M-%S"

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(
            root or os.environ.get(RunStore.ROOT_VARIABLE)
            or RunStore.DEFAULT_ROOT
        )

    # -- reading --------------------------------------------------------------

    def list(self, drum: str | None = None) -> list[RunRecord]:
        """Every stored run, newest first. A directory missing `run.json` is
        skipped rather than raised on: an interrupted fit leaves one behind and
        it should not break the list."""
        if not self.root.exists():
            return []
        found: list[RunRecord] = []
        for drum_dir in sorted(self.root.iterdir()):
            if not drum_dir.is_dir() or (drum and drum_dir.name != drum):
                continue
            for run_dir in sorted(drum_dir.iterdir(), reverse=True):
                record = self.read(run_dir)
                if record is not None:
                    found.append(record)
        found.sort(key=lambda record: record.started, reverse=True)
        return found

    def read(self, directory: str | Path) -> RunRecord | None:
        directory = Path(directory)
        summary_path = directory / "run.json"
        if not summary_path.exists():
            return None
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        score_path = directory / "score.json"
        score = {}
        if score_path.exists():
            try:
                score = json.loads(score_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                score = {}
        return RunRecord(
            directory=directory,
            drum=str(summary.get("drum", directory.parent.name)),
            started=str(summary.get("started", directory.name)),
            summary=summary, score=score,
        )

    def drums(self) -> list[str]:
        return sorted({record.drum for record in self.list()})

    # -- writing --------------------------------------------------------------

    def new_directory(self, drum: str, when: datetime | None = None) -> Path:
        stamp = (when or datetime.now()).strftime(RunStore.STAMP)
        directory = self.root / drum / stamp
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def save(
        self,
        drum: str,
        summary: dict,
        params: DrumParams,
        score: dict | None = None,
        layers: list[dict] | None = None,
        sr: int = Audio.DEFAULT_SR,
        when: datetime | None = None,
    ) -> Path:
        """Write one run. `layers` is a list of
        `{"velocity": v, "generated": ndarray, "reference": ndarray | None}`.
        """
        directory = self.new_directory(drum, when)
        started = directory.name

        stored: list[dict] = []
        if layers:
            (directory / "audio").mkdir(exist_ok=True)
            (directory / "reference").mkdir(exist_ok=True)
            for layer in layers:
                velocity = float(layer["velocity"])
                name = f"v{velocity:06.2f}".replace(".", "-") + ".wav"
                entry = {"velocity": velocity, "generated": f"audio/{name}"}
                AudioIO.write(directory / "audio" / name,
                              np.asarray(layer["generated"]), sr)
                if layer.get("reference") is not None:
                    AudioIO.write(directory / "reference" / name,
                                  np.asarray(layer["reference"]), sr)
                    entry["reference"] = f"reference/{name}"
                stored.append(entry)

        payload = dict(summary)
        payload.update({"drum": drum, "started": started,
                        "stored_layers": stored, "sr": int(sr)})
        (directory / "run.json").write_text(
            json.dumps(payload, indent=2, default=RunStore._plain),
            encoding="utf-8")
        params.save(directory / "params.json")
        if score is not None:
            (directory / "score.json").write_text(
                json.dumps(score, indent=2, default=RunStore._plain),
                encoding="utf-8")
        return directory

    def delete(self, directory: str | Path) -> bool:
        """Remove one run. Refuses anything outside the store, because this
        takes a path from a UI and `shutil.rmtree` does not ask twice."""
        directory = Path(directory).resolve()
        root = self.root.resolve()
        if root not in directory.parents or not directory.is_dir():
            return False
        shutil.rmtree(directory)
        return True

    @staticmethod
    def _plain(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        return str(value)
