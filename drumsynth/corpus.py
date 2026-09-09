"""The sample library that ships with the repository, as data.

`data/metadata/library.json` indexes the WAV files under `data/samples`: which
drum, which velocity layer, which round robin. The audio itself is mostly not
in the repository — one file is kept so the tests have something real to fit —
so a corpus entry is a description that may or may not have a file behind it,
and `present()` is how you ask.

This module exists so the fit has something to point at. It holds no opinion
about the model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .core.audio_io import AudioIO


def default_library() -> Path | None:
    """`data/metadata/library.json` beside the package, if the repo is checked out."""
    candidate = Path(__file__).resolve().parent.parent / "data" / "metadata" / "library.json"
    return candidate if candidate.exists() else None


@dataclass(frozen=True)
class Sample:
    """One recorded hit."""

    path: Path
    drum: str
    family: str
    velocity: float
    articulation: str
    round_robin: int

    @property
    def name(self) -> str:
        return f"{self.drum}/{self.path.stem}"

    def present(self) -> bool:
        return self.path.exists()

    def load(self, sr: int | None = None) -> tuple[np.ndarray, int]:
        """Mono float64 audio, at the file's rate unless `sr` says otherwise."""
        return AudioIO.read(self.path, sr=sr)


@dataclass(frozen=True)
class Corpus:
    """Every sample the library indexes, grouped by drum."""

    root: Path
    sample_rate: int
    drums: dict[str, tuple[Sample, ...]]

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Corpus":
        location = Path(path) if path is not None else default_library()
        if location is None:
            raise FileNotFoundError(
                "no sample library found — expected data/metadata/library.json "
                "in the repository"
            )

        data = json.loads(Path(location).read_text(encoding="utf-8"))
        root = Path(location).parent

        drums = {
            drum: tuple(cls._sample(root, drum, entry) for entry in entries)
            for drum, entries in data.get("drums", {}).items()
        }
        return cls(root=root, sample_rate=int(data.get("sr", 44100)), drums=drums)

    @staticmethod
    def _sample(root: Path, drum: str, entry: dict) -> Sample:
        return Sample(
            path=(root / entry["path"]).resolve(),
            drum=entry.get("drum", drum),
            family=entry.get("family", ""),
            velocity=float(entry.get("velocity", 0.0)),
            articulation=entry.get("articulation", ""),
            round_robin=int(entry.get("round_robin", 0)),
        )

    @property
    def names(self) -> list[str]:
        return sorted(self.drums)

    def samples(self, drum: str | None = None, present_only: bool = False) -> list[Sample]:
        """Every sample, or every sample of one drum; optionally only those on disk."""
        if drum is not None:
            found = self.drums.get(drum)
            if found is None:
                raise KeyError(f"no drum named {drum!r}; have {self.names}")
            samples = list(found)
        else:
            samples = [s for entries in self.drums.values() for s in entries]

        return [s for s in samples if s.present()] if present_only else samples
