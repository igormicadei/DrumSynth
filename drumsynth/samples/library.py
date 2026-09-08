"""All drums. A thin index over SampleSets — the fitter never sees this.

It exists so a directory tree can be turned into per-drum sets in one call, and
so `validate()` can be run across a whole session before any fitting starts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from ..core.constants import Audio
from .sample import Sample
from .sample_set import SampleSet


class SampleLibrary:
    """Named `SampleSet`s, one per drum."""

    def __init__(self, sr: int = Audio.DEFAULT_SR) -> None:
        self.sr = int(sr)
        self._sets: dict[str, SampleSet] = {}

    @classmethod
    def scan(
        cls,
        root: str | Path,
        pattern: str | None = None,
        sr: int = Audio.DEFAULT_SR,
        glob: str = "**/*.wav",
    ) -> "SampleLibrary":
        """Walk a directory tree, grouping by parsed drum name.

        Files that do not parse are skipped silently — a session folder holds
        room mics, bounces and notes alongside the hits, and stopping on the
        first one that is not a sample helps nobody.
        """
        library = cls(sr)
        for path in sorted(Path(root).glob(glob)):
            try:
                sample = Sample.from_filename(path, pattern)
            except ValueError:
                continue
            if sample.drum not in library._sets:
                library.add_set(SampleSet(drum=sample.drum, sr=sr))
            sample.sr = sr
            library[sample.drum].add(sample)
        return library

    def add_set(self, sample_set: SampleSet) -> None:
        self._sets[sample_set.drum] = sample_set

    def drums(self) -> list[str]:
        return sorted(self._sets)

    def load_all(self) -> "SampleLibrary":
        for sample_set in self:
            sample_set.load_all()
        return self

    def validate(self) -> dict[str, list[str]]:
        """Every drum's problems, keyed by drum. Run before any fitting."""
        return {drum: self._sets[drum].validate() for drum in self.drums()}

    def summary(self) -> str:
        """Per drum: sample count, velocity range, coverage, warnings."""
        if not self._sets:
            return "SampleLibrary: empty"

        lines = [f"SampleLibrary: {len(self._sets)} drums @ {self.sr} Hz"]
        for drum in self.drums():
            sample_set = self._sets[drum]
            low, high = sample_set.velocity_range()
            lines.append(
                f"  {drum:<20} {len(sample_set):>3} samples  v{low:.0f}-v{high:.0f}  "
                f"coverage {sample_set.velocity_coverage().tolist()}"
            )
            for issue in sample_set.validate():
                lines.append(f"      ! {issue}")
        return "\n".join(lines)

    def save_manifest(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "sr": self.sr,
                    "drums": {
                        drum: [sample.to_dict() for sample in self._sets[drum]]
                        for drum in self.drums()
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load_manifest(cls, path: str | Path) -> "SampleLibrary":
        path = Path(path)
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        library = cls(int(data.get("sr", Audio.DEFAULT_SR)))
        for drum, entries in data.get("drums", {}).items():
            sample_set = SampleSet(drum=drum, sr=library.sr)
            for entry in entries:
                entry = dict(entry)
                sample_path = Path(entry["path"])
                if not sample_path.is_absolute():
                    entry["path"] = str(path.parent / sample_path)
                sample_set.add(Sample.from_dict(entry))
            library.add_set(sample_set)
        return library

    def __contains__(self, drum: str) -> bool:
        return drum in self._sets

    def __len__(self) -> int:
        return len(self._sets)

    def __getitem__(self, drum: str) -> SampleSet:
        if drum not in self._sets:
            raise KeyError(f"no drum named {drum!r}; have {self.drums()}")
        return self._sets[drum]

    def __iter__(self) -> Iterator[SampleSet]:
        return iter(self._sets[drum] for drum in self.drums())
