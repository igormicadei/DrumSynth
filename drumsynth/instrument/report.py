"""Writing a velocity fit down: the model, what it plays, and how it was judged.

A run leaves behind, besides the model itself, three WAVs meant to be listened
to side by side:

    reference_sweep.wav     the recordings, one after another, quietest first
    model_sweep.wav         the model rendered at those same velocities
    between_sweep.wav       the model rendered halfway between them

The first two are an A/B of what the fit reproduced. The third is the part no
number can settle: velocities that were never played, which is the whole reason
for building a velocity model rather than keeping the WAVs.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ..core.audio_io import AudioIO
from .fit import InstrumentFitResult
from .model import InstrumentModel

#: Silence between hits in a sweep, in seconds.
SWEEP_GAP = 0.15


def sweep(model: InstrumentModel, velocities: np.ndarray, gap: float = SWEEP_GAP) -> np.ndarray:
    """One take of every velocity in turn, with a little silence between."""
    silence = np.zeros(int(round(gap * model.sample_rate)))
    return np.concatenate(
        [piece for velocity in velocities for piece in (model.render(velocity), silence)]
    )


def sweep_slice(
    audio: np.ndarray, index: int, n_samples: int, sample_rate: int, gap: float = SWEEP_GAP
) -> np.ndarray:
    """Pull one hit back out of a sweep, given where it sits in the ladder."""
    stride = n_samples + int(round(gap * sample_rate))
    start = index * stride
    return np.asarray(audio)[start : start + n_samples]


def between(velocities: np.ndarray) -> np.ndarray:
    """The midpoints of a velocity ladder — the hits nobody recorded."""
    velocities = np.asarray(velocities, dtype=np.float64)
    if velocities.size < 2:
        return velocities
    return 0.5 * (velocities[:-1] + velocities[1:])


def report_dict(result: InstrumentFitResult) -> dict:
    model = result.model
    return {
        "instrument": model.name,
        "sample_rate": model.sample_rate,
        "duration": model.duration,
        "n_layers": int(model.velocities.size),
        "n_recordings": result.n_recordings,
        "velocity_range": list(model.velocity_range),
        "velocities": model.velocities.tolist(),
        "donor_velocities": model.donor_velocities.tolist(),
        "averaged_takes": result.averaged,
        "target_relative_mse": result.target_mse,
        "target_reached": result.target_reached,
        "reconstruction_mse": result.reconstruction_mse,
        "worst_layer": result.worst_layer.to_dict(),
        "generalization_mse": result.generalization_mse,
        "interpolation_mse": result.interpolation_mse,
        "search_seconds": result.elapsed,
        "candidates_evaluated": len(result.evaluations),
        "search_space": {
            name: list(getattr(result.space, name))
            for name in result.space.__dataclass_fields__
        },
        "model": model.metadata(),
        "bytes": model.n_bytes(),
        "encoded_recordings": result.encoded_recordings,
        "compression_ratio": model.compression_ratio(result.encoded_recordings),
        "layers": [layer.to_dict() for layer in result.layers],
        "frontier": [evaluation.to_dict() for evaluation in result.frontier],
    }


def save_instrument_fit(
    result: InstrumentFitResult,
    out_dir: str | Path,
    *,
    layers=None,
    take: int = 0,
    plot: bool = True,
) -> dict[str, Path]:
    """Write every artefact of a velocity fit into `out_dir`."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = result.model
    written: dict[str, Path] = {}

    written["model"] = model.save(out / "instrument.npz")
    written["metadata"] = _write_json(out / "instrument.json", model.metadata())
    written["report"] = _write_json(out / "report.json", report_dict(result))
    written["layers"] = _write_layers_csv(out / "layers.csv", result)

    written["model_sweep"] = AudioIO.write(
        out / "model_sweep.wav", sweep(model, model.velocities), model.sample_rate
    )
    written["between_sweep"] = AudioIO.write(
        out / "between_sweep.wav",
        sweep(model, between(model.velocities)),
        model.sample_rate,
    )

    if layers is not None:
        silence = np.zeros(int(round(SWEEP_GAP * model.sample_rate)))
        reference = np.concatenate(
            [
                piece
                for index in range(layers.n_layers)
                for piece in (layers.audio(index, take), silence)
            ]
        )
        written["reference_sweep"] = AudioIO.write(
            out / "reference_sweep.wav", reference, model.sample_rate
        )

    if plot:
        figure = plot_layers(result, out / "velocity.png")
        if figure is not None:
            written["velocity"] = figure
        written.update(_figures(result.model, layers, take, out / "figures"))

    return written


def _figures(model, layers, take: int, out_dir) -> dict[str, Path]:
    """The full set of diagnostic figures, when matplotlib is installed."""
    try:
        from ..plots import save_instrument_figures
    except ImportError:  # pragma: no cover - depends on the environment
        return {}
    return {
        f"figure.{name}": path
        for name, path in save_instrument_figures(model, layers, take, out_dir).items()
    }


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_layers_csv(path: Path, result: InstrumentFitResult) -> Path:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["velocity", "reconstruction_mse", "generalization_mse", "is_donor"],
        )
        writer.writeheader()
        for layer in result.layers:
            writer.writerow(layer.to_dict())
    return path


def plot_layers(result: InstrumentFitResult, path: str | Path) -> Path | None:
    """Error against velocity, and the size/error frontier beside it."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - depends on the environment
        return None

    path = Path(path)
    figure, (left, right) = plt.subplots(1, 2, figsize=(12.0, 4.8))

    velocities = [layer.velocity for layer in result.layers]
    left.semilogy(
        velocities,
        [max(layer.reconstruction_mse, 1e-12) for layer in result.layers],
        marker="o",
        markersize=4,
        label="reconstruction",
    )
    generalization = [layer.generalization_mse for layer in result.layers]
    if any(value is not None for value in generalization):
        left.semilogy(
            velocities,
            [max(value or np.nan, 1e-12) for value in generalization],
            marker="s",
            markersize=4,
            label="generalization",
        )
    for velocity in result.model.donor_velocities:
        left.axvline(velocity, color="grey", alpha=0.25, linewidth=0.8)
    left.axhline(result.target_mse, color="grey", linestyle="--", linewidth=1.0)
    left.set_xlabel("velocity")
    left.set_ylabel("relative error")
    left.set_title(f"{result.model.name} — per layer (grey lines: donors)")
    left.grid(alpha=0.25, which="both")
    left.legend(fontsize=8)

    frontier = result.frontier
    right.loglog(
        [e.n_scalars for e in result.evaluations],
        [max(e.relative_mse, 1e-12) for e in result.evaluations],
        ".",
        markersize=3,
        alpha=0.3,
        label="candidates",
    )
    right.loglog(
        [e.n_scalars for e in frontier],
        [max(e.relative_mse, 1e-12) for e in frontier],
        color="black",
        linewidth=1.2,
        label="frontier",
    )
    right.scatter(
        [result.model.n_scalars],
        [max(result.reconstruction_mse, 1e-12)],
        marker="*",
        s=200,
        color="crimson",
        zorder=5,
        label="chosen",
    )
    right.set_xlabel("model size (stored numbers)")
    right.set_ylabel("mean reconstruction error")
    right.set_title(result.candidate.label(), fontsize=9)
    right.grid(alpha=0.25, which="both")
    right.legend(fontsize=8)

    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path
