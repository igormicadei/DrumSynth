"""Where fits are kept after they finish.

A fit is expensive and worth going back to: a drum searched at three targets
is three models to compare, not one to overwrite. So runs are written to a
store outside the working directory — one directory per run, holding
everything that run produced, and a small `run.json` describing it.

    runs/
    ├── instrument/
    │   └── toms-stereo-tom3/
    │       ├── 20260910-004530/     instrument.npz, report.json, sweeps…
    │       └── 20260910-011502/
    └── hit/
        └── rr1-01-tom3-stereo-rr1/
            └── 20260910-002211/     model.npz, reconstruction.wav…

The store is a directory, not a database: a run is readable without this
package, deleting one is `rm -r`, and the index is rebuilt by looking. Nothing
here caches — a listing is a scan, because a stale index of expensive work is
worse than a slow listing of it.
"""

from __future__ import annotations

import json
import os
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

#: Where runs go when nothing says otherwise.
DEFAULT_ROOT = Path("runs")

#: The environment variable that moves the store.
ROOT_ENV = "DRUMSYNTH_RUNS"

#: What a run directory is called: sortable, and readable at a glance.
STAMP = "%Y%m%d-%H%M%S"


@dataclass(frozen=True)
class Run:
    """One finished fit, on disk."""

    path: Path
    kind: str  # "instrument" or "hit"
    name: str
    created: datetime
    metadata: dict

    @classmethod
    def read(cls, path: str | Path) -> "Run":
        path = Path(path)
        data = json.loads((path / "run.json").read_text(encoding="utf-8"))
        return cls(
            path=path,
            kind=data["kind"],
            name=data["name"],
            created=datetime.fromisoformat(data["created"]),
            metadata=data,
        )

    @property
    def label(self) -> str:
        return f"{self.created.strftime('%Y-%m-%d %H:%M')} · {self.summary}"

    @property
    def summary(self) -> str:
        return self.metadata.get("representation", "")

    @property
    def model_path(self) -> Path:
        return self.path / self.metadata["model_file"]

    def load_model(self):
        """The model this run produced, of whichever kind it is."""
        if self.kind == "instrument":
            from .instrument.model import InstrumentModel

            return InstrumentModel.load(self.model_path)

        from .spectral.model import SpectralModel

        return SpectralModel.load(self.model_path)

    def report(self) -> dict:
        """The full report this run wrote, read back."""
        path = self.path / "report.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def files(self) -> list[Path]:
        return sorted(p for p in self.path.iterdir() if p.is_file())

    def n_bytes(self) -> int:
        """Everything this run left on disk."""
        return sum(p.stat().st_size for p in self.path.rglob("*") if p.is_file())


@dataclass(frozen=True)
class RunStore:
    """A directory of runs, grouped by kind and by what was fitted."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    @classmethod
    def default(cls) -> "RunStore":
        return cls(Path(os.environ.get(ROOT_ENV, DEFAULT_ROOT)))

    # -- reading --------------------------------------------------------------

    def names(self, kind: str = "instrument") -> list[str]:
        """What has been fitted, of one kind."""
        base = self.root / kind
        if not base.is_dir():
            return []
        return sorted(p.name for p in base.iterdir() if p.is_dir() and any(p.iterdir()))

    def runs(self, name: str | None = None, kind: str = "instrument") -> list[Run]:
        """Every run of one thing, or of everything of that kind. Newest first."""
        base = self.root / kind
        if not base.is_dir():
            return []

        directories = [base / name] if name is not None else sorted(base.iterdir())
        found = []
        for directory in directories:
            if not directory.is_dir():
                continue
            for run in sorted(directory.iterdir()):
                if (run / "run.json").exists():
                    found.append(Run.read(run))
        # The directory name breaks ties: two runs can land in the same second,
        # and the second one is the newer.
        return sorted(found, key=lambda run: (run.created, run.path.name), reverse=True)

    def latest(self, name: str, kind: str = "instrument") -> Run | None:
        runs = self.runs(name, kind)
        return runs[0] if runs else None

    def find(self, path: str | Path) -> Run:
        """A run by its directory, wherever it is."""
        return Run.read(path)

    # -- writing --------------------------------------------------------------

    def new_run(self, kind: str, name: str, when: datetime | None = None) -> Path:
        """An empty directory for a run that is about to be written."""
        when = when or datetime.now(timezone.utc).astimezone()
        base = self.root / kind / name
        base.mkdir(parents=True, exist_ok=True)

        stamp = when.strftime(STAMP)
        path = base / stamp
        suffix = 1
        while path.exists():
            suffix += 1
            path = base / f"{stamp}-{suffix}"

        path.mkdir(parents=True)
        return path

    def write_index(
        self, path: Path, kind: str, name: str, model_file: str, metadata: dict
    ) -> Run:
        """Describe a run that has been written, so a listing can read it."""
        created = datetime.now(timezone.utc).astimezone().replace(microsecond=0)
        data = {
            "kind": kind,
            "name": name,
            "created": created.isoformat(timespec="seconds"),
            "model_file": model_file,
            **metadata,
        }
        (path / "run.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        return Run(path=path, kind=kind, name=name, created=created, metadata=data)


@contextmanager
def _clean_on_failure(path: Path):
    """A run that could not be written leaves nothing behind to puzzle over."""
    try:
        yield
    except BaseException:
        shutil.rmtree(path, ignore_errors=True)
        raise


def store_instrument_fit(
    result, layers=None, *, store: RunStore | None = None, take: int = 0, plot: bool = True
) -> Run:
    """Write a velocity fit into the store and return the run it became."""
    from .instrument.report import save_instrument_fit

    store = store or RunStore.default()
    path = store.new_run("instrument", result.model.name)
    with _clean_on_failure(path):
        save_instrument_fit(result, path, layers=layers, take=take, plot=plot)

    model = result.model
    return store.write_index(
        path,
        "instrument",
        result.model.name,
        "instrument.npz",
        {
            "representation": model.candidate.label(),
            "target_mse": result.target_mse,
            "target_reached": result.target_reached,
            "reconstruction_mse": result.reconstruction_mse,
            "generalization_mse": result.generalization_mse,
            "interpolation_mse": result.interpolation_mse,
            "n_layers": int(model.velocities.size),
            "velocity_range": list(model.velocity_range),
            "n_recordings": result.n_recordings,
            "encoded_recordings": result.encoded_recordings,
            "averaged_takes": result.averaged,
            "n_scalars": model.n_scalars,
            "bytes": model.n_bytes(),
            "sample_rate": model.sample_rate,
            "duration": model.duration,
            "search_seconds": result.elapsed,
            "candidates_evaluated": len(result.evaluations),
        },
    )


def store_hit_fit(
    result, name: str, *, reference=None, store: RunStore | None = None, plot: bool = True
) -> Run:
    """Write a single-hit fit into the store and return the run it became."""
    from .fitting.report import save_fit

    store = store or RunStore.default()
    path = store.new_run("hit", name)
    with _clean_on_failure(path):
        save_fit(result, path, reference=reference, input_path=name, plot=plot)

    model = result.model
    return store.write_index(
        path,
        "hit",
        name,
        "model.npz",
        {
            "representation": model.candidate.label(),
            "target_mse": result.target_mse,
            "target_reached": result.target_reached,
            "relative_mse": result.quality.relative_mse,
            "snr_db": result.quality.snr_db,
            "n_scalars": model.n_scalars,
            "bytes": model.n_bytes(),
            "sample_rate": model.sample_rate,
            "duration": model.duration,
            "search_seconds": result.elapsed,
            "candidates_evaluated": len(result.evaluations),
        },
    )
