"""Writing a fit down: the model, what it sounds like, and what it cost.

A run leaves six files behind:

    model.npz          the model — arrays and metadata, loadable on its own
    model.json         the same metadata, readable without numpy
    reconstruction.wav what the model renders to
    residual.wav       reference minus reconstruction: what the model missed
    report.json        the chosen candidate, the target, the Pareto frontier
    evaluations.csv    every candidate that was measured, one row each
    frontier.png       size against error, if matplotlib is installed

`residual.wav` is there to be listened to. A relative MSE of 1e-5 says the
error is 50 dB down; it does not say whether what is left is broadband hiss or
a missing partial ringing on its own, and those call for different fixes.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ..core.audio_io import AudioIO
from .search import Evaluation, FitResult

_CSV_COLUMNS = [
    "relative_mse",
    "snr_db",
    "correlation",
    "n_scalars",
    "n_frames",
    "n_fft",
    "hop",
    "n_components",
    "codec",
    "codec_param",
    "phase_stride",
]


def _row(evaluation: Evaluation) -> dict:
    candidate = evaluation.candidate
    return {
        "relative_mse": evaluation.quality.relative_mse,
        "snr_db": evaluation.quality.snr_db,
        "correlation": evaluation.quality.correlation,
        "n_scalars": evaluation.n_scalars,
        "n_frames": evaluation.n_frames,
        "n_fft": candidate.n_fft,
        "hop": candidate.hop,
        "n_components": candidate.n_components,
        "codec": candidate.codec,
        "codec_param": candidate.codec_param,
        "phase_stride": candidate.phase_stride,
    }


def report_dict(result: FitResult, input_path: str | Path | None = None) -> dict:
    """The run, as JSON-safe data."""
    return {
        "input": str(input_path) if input_path is not None else None,
        "sample_rate": result.sample_rate,
        "duration": result.model.duration,
        "target_relative_mse": result.target_mse,
        "target_reached": result.target_reached,
        "search_seconds": result.elapsed,
        "candidates_evaluated": len(result.evaluations),
        "search_space": {
            field: list(getattr(result.space, field))
            for field in result.space.__dataclass_fields__
        },
        "model": result.model.metadata(),
        "quality": result.quality.to_dict(),
        "bytes": result.model.n_bytes(),
        "compression_ratio": result.model.compression_ratio(),
        "frontier": [e.to_dict() for e in result.frontier],
    }


def save_fit(
    result: FitResult,
    out_dir: str | Path,
    *,
    reference: np.ndarray | None = None,
    input_path: str | Path | None = None,
    plot: bool = True,
) -> dict[str, Path]:
    """Write every artefact of a fit into `out_dir`, and say where each went."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    reconstruction = result.model.render()
    written["model"] = result.model.save(out / "model.npz")
    written["metadata"] = _write_json(out / "model.json", result.model.metadata())
    written["reconstruction"] = AudioIO.write(
        out / "reconstruction.wav", reconstruction, result.sample_rate
    )

    if reference is not None:
        residual = np.asarray(reference, dtype=np.float64).ravel() - reconstruction
        written["residual"] = AudioIO.write(
            out / "residual.wav", residual, result.sample_rate
        )

    written["report"] = _write_json(
        out / "report.json", report_dict(result, input_path=input_path)
    )
    written["evaluations"] = _write_csv(out / "evaluations.csv", result.evaluations)

    if plot:
        figure = plot_frontier(result, out / "frontier.png")
        if figure is not None:
            written["plot"] = figure

    return written


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_csv(path: Path, evaluations: list[Evaluation]) -> Path:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for evaluation in sorted(evaluations, key=lambda e: e.n_scalars):
            writer.writerow(_row(evaluation))
    return path


def plot_frontier(result: FitResult, path: str | Path) -> Path | None:
    """Scatter every candidate, draw the frontier through it, mark the choice.

    Returns None when matplotlib is not installed — the plot is a convenience,
    never something the rest of the fit depends on.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - depends on the environment
        return None

    path = Path(path)
    frontier = result.frontier
    best = result.quality.relative_mse

    figure, axes = plt.subplots(figsize=(9.0, 5.5))
    by_codec: dict[str, list[Evaluation]] = {}
    for evaluation in result.evaluations:
        by_codec.setdefault(evaluation.candidate.codec, []).append(evaluation)

    for codec, group in sorted(by_codec.items()):
        axes.scatter(
            [e.n_scalars for e in group],
            [max(e.quality.relative_mse, 1e-12) for e in group],
            s=8,
            alpha=0.35,
            label=codec,
        )

    axes.plot(
        [e.n_scalars for e in frontier],
        [max(e.quality.relative_mse, 1e-12) for e in frontier],
        color="black",
        linewidth=1.2,
        label="frontier",
    )
    axes.scatter(
        [result.model.n_scalars],
        [max(best, 1e-12)],
        marker="*",
        s=220,
        color="crimson",
        zorder=5,
        label="chosen",
    )
    axes.axhline(result.target_mse, color="grey", linestyle="--", linewidth=1.0)
    axes.annotate(
        f"target {result.target_mse:.0e}",
        (0.01, result.target_mse),
        xycoords=("axes fraction", "data"),
        va="bottom",
        fontsize=8,
        color="grey",
    )

    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlabel("model size (stored numbers)")
    axes.set_ylabel("relative waveform MSE")
    axes.set_title(f"{len(result.evaluations)} candidates — {result.candidate.label()}")
    axes.grid(alpha=0.25, which="both")
    axes.legend(loc="upper right", fontsize=8)

    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path
