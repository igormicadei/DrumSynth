"""Running a fit in its own process.

A fit takes minutes and Streamlit reruns its whole script on every widget
touch, so the fit cannot live in the app. It runs where the live audio engine
runs — a subprocess on the far end of a pipe, reporting progress as JSON lines
— for the same reasons: the UI cannot stall it, it cannot stall the UI, and if
it dies the app is still there to say so.

    python -m drumsynth.fitting.worker --manifest data/metadata/drums-kick.json
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

from ..core.constants import Audio
from ..live.protocol import Event
from ..synth.params import DrumParams
from .targets import DrumCatalogue, FitTarget, TargetBuilder
from .trainer import DrumTrainer, FitEvaluator, FitResult, TrainingSettings


class FitEvent:
    """Engine -> UI, on top of the shared `Event` encoding."""

    READY = "ready"
    PROGRESS = "progress"
    STAGE = "stage"
    GENERATION = "generation"
    RESULT = "result"
    SCORED = "scored"
    ERROR = "error"
    DONE = "done"


class FitWorker:
    """One fit, start to finish, reporting as it goes."""

    def __init__(self, settings: TrainingSettings, sr: int = Audio.DEFAULT_SR) -> None:
        self.settings = settings
        self.sr = int(sr)

    @staticmethod
    def emit(name: str, **fields) -> None:
        sys.stdout.write(Event.encode(name, **FitWorker._plain(fields)) + "\n")
        sys.stdout.flush()

    @staticmethod
    def _plain(value):
        """JSON cannot carry numpy. Convert rather than fail at the pipe."""
        if isinstance(value, dict):
            return {key: FitWorker._plain(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [FitWorker._plain(item) for item in value]
        if isinstance(value, np.ndarray):
            return FitWorker._plain(value.tolist())
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    # -- the run --------------------------------------------------------------

    def run(self, target: FitTarget, output: Path | None = None) -> int:
        self.emit(
            FitEvent.READY,
            drum=target.drum,
            layers=[
                {"velocity": layer.velocity,
                 "velocity_normalized": layer.velocity_normalized,
                 "source": layer.source}
                for layer in target.layers
            ],
            reference_velocity=target.reference.velocity,
            frequency_velocity=target.frequency_layer().velocity,
            warnings=target.warnings,
            settings=self.settings.to_dict(),
        )

        def progress(event: dict) -> None:
            if "generation" in event:
                self.emit(FitEvent.GENERATION, **event)
            elif "phase" in event:
                self.emit(FitEvent.STAGE, **event)
            else:
                self.emit(FitEvent.PROGRESS, **event)

        try:
            result = DrumTrainer(self.settings, self.sr).run(target, progress=progress)
        except Exception as error:
            self.emit(FitEvent.ERROR, message=f"{type(error).__name__}: {error}",
                      traceback=traceback.format_exc()[-2000:])
            return 1

        self.emit(FitEvent.RESULT, **self._summarize(result))

        evaluator = FitEvaluator(self.sr, self.settings.control_period)
        aggregate, cards = evaluator.score(result, target, progress=progress)
        self.emit(
            FitEvent.SCORED,
            total=aggregate.total,
            stft_loss=aggregate.stft_loss,
            components=[component.to_dict() for component in aggregate.components],
            warnings=aggregate.warnings,
            per_layer=[
                {"velocity": velocity, "total": card.total,
                 "components": {c.name: c.value for c in card.components}}
                for velocity, card in cards
            ],
            report=aggregate.report(),
        )

        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            result.params.save(output)
            self.emit(FitEvent.DONE, path=str(output.resolve()),
                      elapsed=result.elapsed)
        else:
            self.emit(FitEvent.DONE, elapsed=result.elapsed)
        return 0

    def _summarize(self, result: FitResult) -> dict:
        return {
            "drum": result.drum,
            "elapsed": result.elapsed,
            "params": result.params.to_dict(),
            "modes": len(result.modal.modes),
            "tension": {"k": result.modal.tension.k, "tau": result.modal.tension.tau},
            "damping_anchors": result.modal.damping_anchors,
            "notes": result.modal.notes,
            "warnings": result.warnings,
            "inspection": result.inspection.to_dict(),
            "curve": result.curve.to_dict(),
            "shape": result.shape.tolist(),
            "reference_velocity": result.reference_velocity,
            "table": result.velocity_table(),
            "generations": [
                {"index": g.index, "loss": g.loss, "best_loss": g.best_loss,
                 "elapsed": g.elapsed}
                for g in result.generations
            ],
            "mode_table": [
                {"f_static": mode.f_static, "gain": mode.gain, "t60": mode.t60}
                for mode in result.params.sorted_modes()
            ],
        }


class WorkerCLI:
    @staticmethod
    def parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--manifest", type=Path, default=None,
                            help="drums-<slug>.json to fit")
        parser.add_argument("--drum", default=None,
                            help="drum slug; looked up in --metadata-dir")
        parser.add_argument("--metadata-dir", type=Path, default=Path("data/metadata"))
        parser.add_argument("--output", type=Path, default=None,
                            help="where to write the fitted DrumParams JSON")
        parser.add_argument("--sr", type=int, default=44100)
        parser.add_argument("--seconds", type=float, default=2.5)
        parser.add_argument("--max-layers", type=int, default=8)
        parser.add_argument("--max-modes", type=int, default=34)
        parser.add_argument("--generations", type=int, default=24)
        parser.add_argument("--population", type=int, default=12)
        parser.add_argument("--workers", type=int, default=1)
        parser.add_argument("--control-period", type=int, default=64)
        parser.add_argument("--noise-bands", type=int, default=4)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--no-refine", action="store_true",
                            help="skip the per-mode gain polish (much faster)")
        return parser

    @staticmethod
    def main(argv: list[str] | None = None) -> int:
        args = WorkerCLI.parser().parse_args(argv)

        manifest = args.manifest
        if manifest is None and args.drum:
            manifest = Path(args.metadata_dir) / f"drums-{args.drum}.json"
        if manifest is None or not Path(manifest).exists():
            FitWorker.emit(FitEvent.ERROR,
                           message=f"no manifest to fit (looked for {manifest})")
            return 2

        settings = TrainingSettings(
            seconds=args.seconds, max_layers=args.max_layers,
            max_modes=args.max_modes, control_period=args.control_period,
            noise_bands=args.noise_bands, generations=args.generations,
            population=args.population, workers=args.workers,
            refine_gains=not args.no_refine, seed=args.seed,
        )
        worker = FitWorker(settings, args.sr)

        try:
            sample_set = DrumCatalogue.load(manifest, args.sr)
            builder = TargetBuilder(args.sr, args.seconds, args.max_layers)
            target = builder.from_sample_set(
                sample_set,
                progress=lambda event: FitWorker.emit(FitEvent.PROGRESS, **event),
            )
        except Exception as error:
            FitWorker.emit(FitEvent.ERROR,
                           message=f"could not prepare {manifest}: {error}",
                           traceback=traceback.format_exc()[-2000:])
            return 1

        return worker.run(target, args.output)


if __name__ == "__main__":
    sys.exit(WorkerCLI.main())
