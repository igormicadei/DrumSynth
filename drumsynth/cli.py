"""Command line: fit a sound, decode a model, inspect a model.

    drumsynth fit hit.wav                     # search, write everything to hit_fit/
    drumsynth fit hit.wav --target-mse 1e-6   # ask for 60 dB instead of 50
    drumsynth fit *.wav --jobs 0 -o fits/     # a directory of hits, all cores
    drumsynth decode hit_fit/model.npz out.wav
    drumsynth inspect hit_fit/model.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .core.audio_io import AudioIO
from .fitting.report import save_fit
from .fitting.search import DEFAULT_TARGET_MSE, Progress, SearchSpace, fit
from .spectral.model import SpectralModel

SPACES = {
    "quick": SearchSpace.quick,
    "default": SearchSpace,
    "full": lambda: SearchSpace.full(),
}


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    return args.run(args)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="drumsynth",
        description="Fit a compact spectral model to a sound, and play it back.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    fit_command = commands.add_parser(
        "fit", help="search for the smallest model of a WAV within a quality target"
    )
    fit_command.add_argument("inputs", nargs="+", type=Path, help="input WAV files")
    fit_command.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="output directory (default: <input>_fit next to each input)",
    )
    fit_command.add_argument(
        "--target-mse",
        type=float,
        default=DEFAULT_TARGET_MSE,
        help=f"maximum relative waveform MSE (default: {DEFAULT_TARGET_MSE:g}, 50 dB SNR)",
    )
    fit_command.add_argument(
        "--space",
        choices=sorted(SPACES),
        default="default",
        help="how much of the representation grid to search (default: default)",
    )
    fit_command.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="processes to search with; 0 for one per core (the default)",
    )
    fit_command.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="stop after this many candidates — for a quick look, not for a result",
    )
    fit_command.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="resample the input to this rate before fitting",
    )
    fit_command.add_argument(
        "--no-plot", action="store_true", help="skip frontier.png"
    )
    fit_command.add_argument(
        "-q", "--quiet", action="store_true", help="only print the final summary"
    )
    fit_command.set_defaults(run=_run_fit)

    decode_command = commands.add_parser(
        "decode", help="render a saved model back to a WAV"
    )
    decode_command.add_argument("model", type=Path)
    decode_command.add_argument("output", type=Path, nargs="?", default=None)
    decode_command.set_defaults(run=_run_decode)

    inspect_command = commands.add_parser(
        "inspect", help="print what a saved model contains"
    )
    inspect_command.add_argument("model", type=Path)
    inspect_command.set_defaults(run=_run_inspect)

    return parser


def _run_fit(args) -> int:
    for path in args.inputs:
        signal, sample_rate = AudioIO.read(path, sr=args.sample_rate)
        out_dir = (args.out / path.stem) if args.out else path.with_name(f"{path.stem}_fit")

        print(f"{path}  {len(signal) / sample_rate:.3f} s at {sample_rate} Hz")
        result = fit(
            signal,
            sample_rate,
            target_mse=args.target_mse,
            space=SPACES[args.space](),
            jobs=args.jobs,
            max_candidates=args.max_candidates,
            progress=None if args.quiet else _print_progress,
        )

        written = save_fit(
            result,
            out_dir,
            reference=signal,
            input_path=path,
            plot=not args.no_plot,
        )

        print(result.summary())
        print(f"written to       {out_dir}/")
        for name, target in written.items():
            print(f"  {name:14s} {target.name}")
        print()

        if not result.target_reached:
            print(
                "the target was not reached by any candidate in this space — "
                "try --space full, or a looser --target-mse",
                file=sys.stderr,
            )
    return 0


def _print_progress(progress: Progress) -> None:
    best = progress.best
    percent = 100.0 * progress.done / max(progress.total, 1)
    print(
        f"\r  {progress.done:5d}/{progress.total} ({percent:5.1f}%) "
        f"{progress.elapsed:6.1f}s   best: {best.quality.relative_mse:.3e} "
        f"in {best.n_scalars:7d} numbers  {best.candidate.label():<52}",
        end="",
        flush=True,
    )
    if progress.done >= progress.total:
        print()


def _run_decode(args) -> int:
    model = SpectralModel.load(args.model)
    output = args.output or args.model.with_suffix(".wav")
    AudioIO.write(output, model.render(), model.sample_rate)
    print(f"{output}  {model.duration:.3f} s at {model.sample_rate} Hz")
    return 0


def _run_inspect(args) -> int:
    model = SpectralModel.load(args.model)
    print(f"{args.model}")
    print(f"  representation {model.candidate.label()}")
    print(f"  audio          {model.duration:.3f} s at {model.sample_rate} Hz")
    print(f"  frames         {model.n_frames}")
    print(
        f"  size           {model.n_scalars} numbers, {model.n_bytes() / 1024:.1f} kB, "
        f"{1.0 / model.compression_ratio():.1f}x smaller than 16-bit PCM"
    )
    frequencies = model.candidate.spec.bin_frequencies(model.sample_rate)[model.bins]
    print(
        f"  bins           {model.bins.size} between "
        f"{frequencies.min():.1f} Hz and {frequencies.max():.1f} Hz"
    )
    for name, array in sorted(model.arrays.items()):
        print(f"  {name:<14s} {tuple(array.shape)} {array.dtype}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
