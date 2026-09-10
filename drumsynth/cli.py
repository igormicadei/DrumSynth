"""Command line: fit a sound or a whole drum, play a model back, look inside it.

    drumsynth fit hit.wav                     # one hit: search, write to hit_fit/
    drumsynth fit hit.wav --target-mse 1e-6   # ask for 60 dB instead of 50
    drumsynth fit *.wav --jobs 0 -o fits/     # a directory of hits, all cores

    drumsynth drums                           # what the sample library holds
    drumsynth fit-drum toms-stereo-tom3       # every velocity of one drum, one model
    drumsynth runs                            # every fit that has been kept
    drumsynth play runs/instrument/toms-stereo-tom3/<run>/instrument.npz out.wav -v 96

    drumsynth decode hit_fit/model.npz out.wav
    drumsynth inspect hit_fit/model.npz
    drumsynth bench hit_fit/model.npz         # what it costs to play live

Fits are kept in the run store (`runs/`, or $DRUMSYNTH_RUNS) unless `-o` says
where to put them, so a drum searched at three targets is three runs to compare
rather than one directory overwritten twice.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .core.audio_io import AudioIO
from .corpus import Corpus
from .fitting.report import save_fit
from .fitting.search import DEFAULT_TARGET_MSE, Progress, SearchSpace, fit
from .instrument.fit import DEFAULT_TARGET_MSE as DEFAULT_INSTRUMENT_TARGET_MSE
from .instrument.fit import InstrumentSearchSpace, fit_instrument
from .instrument.layers import VelocityLayers
from .instrument.model import InstrumentModel
from .instrument.report import save_instrument_fit
from .runs import RunStore, store_hit_fit, store_instrument_fit
from .spectral.model import SpectralModel

SPACES = {
    "quick": SearchSpace.quick,
    "default": SearchSpace,
    "full": SearchSpace.full,
}

INSTRUMENT_SPACES = {
    "quick": InstrumentSearchSpace.quick,
    "default": InstrumentSearchSpace,
    "full": InstrumentSearchSpace.full,
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
        help="write here instead of keeping the run in the store",
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
        "--runs", type=Path, default=None, help="a run store other than the default"
    )
    fit_command.add_argument(
        "--no-plot", action="store_true", help="skip every figure"
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

    drum_command = commands.add_parser(
        "fit-drum",
        help="fit every velocity of one drum in the sample library to a single model",
    )
    drum_command.add_argument("drum", help="a name from `drumsynth drums`")
    drum_command.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="write here instead of keeping the run in the store",
    )
    drum_command.add_argument(
        "--target-mse",
        type=float,
        default=DEFAULT_INSTRUMENT_TARGET_MSE,
        help=(
            "maximum mean relative error when reproducing a recorded velocity "
            f"(default: {DEFAULT_INSTRUMENT_TARGET_MSE:g})"
        ),
    )
    drum_command.add_argument(
        "--space", choices=sorted(INSTRUMENT_SPACES), default="default"
    )
    drum_command.add_argument(
        "--take",
        type=int,
        default=0,
        help="which round robin each velocity is built from (default: the first)",
    )
    drum_command.add_argument(
        "--average-takes",
        action="store_true",
        help=(
            "build each velocity from the mean of its round robins rather than "
            "from one of them: closer to the next strike, exactly equal to none"
        ),
    )
    drum_command.add_argument(
        "--donors",
        type=int,
        default=0,
        help=(
            "how many velocities donate their phase; 0, the default, is all of "
            "them. Fewer is a smaller model that plays velocities far from a "
            "donor with the wrong strike's phase"
        ),
    )
    drum_command.add_argument(
        "--library", type=Path, default=None, help="path to a library.json"
    )
    drum_command.add_argument("--sample-rate", type=int, default=None)
    drum_command.add_argument(
        "--max-duration", type=float, default=None, help="trim every recording to this"
    )
    drum_command.add_argument(
        "--runs", type=Path, default=None, help="a run store other than the default"
    )
    drum_command.add_argument("--no-plot", action="store_true")
    drum_command.add_argument("-q", "--quiet", action="store_true")
    drum_command.set_defaults(run=_run_fit_drum)

    drums_command = commands.add_parser(
        "drums", help="list the drums in the sample library and what is on disk"
    )
    drums_command.add_argument("--library", type=Path, default=None)
    drums_command.set_defaults(run=_run_drums)

    play_command = commands.add_parser(
        "play", help="render one hit from a velocity model"
    )
    play_command.add_argument("model", type=Path)
    play_command.add_argument("output", type=Path, nargs="?", default=None)
    play_command.add_argument(
        "-v", "--velocity", type=float, required=True, help="anywhere in the recorded range"
    )
    play_command.set_defaults(run=_run_play)

    runs_command = commands.add_parser(
        "runs", help="list the fits that have been kept"
    )
    runs_command.add_argument(
        "name", nargs="?", default=None, help="only this instrument or hit"
    )
    runs_command.add_argument(
        "--kind", choices=["instrument", "hit"], default="instrument"
    )
    runs_command.add_argument("--runs", dest="root", type=Path, default=None)
    runs_command.set_defaults(run=_run_runs)

    bench_command = commands.add_parser(
        "bench", help="measure what a model costs to play live"
    )
    bench_command.add_argument("model", type=Path)
    bench_command.add_argument(
        "-v", "--velocity", type=float, default=None, help="for a velocity model"
    )
    bench_command.add_argument(
        "--block", type=int, default=256, help="audio callback size (default: 256)"
    )
    bench_command.add_argument(
        "--voices",
        type=int,
        nargs="*",
        default=[1, 4, 8, 16, 32],
        help="polyphony levels to measure",
    )
    bench_command.set_defaults(run=_run_bench)

    inspect_command = commands.add_parser(
        "inspect", help="print what a saved model contains"
    )
    inspect_command.add_argument("model", type=Path)
    inspect_command.set_defaults(run=_run_inspect)

    return parser


def _run_fit(args) -> int:
    for path in args.inputs:
        signal, sample_rate = AudioIO.read(path, sr=args.sample_rate)

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

        if args.out:
            out_dir = args.out / path.stem
            written = save_fit(
                result, out_dir, reference=signal, input_path=path, plot=not args.no_plot
            )
        else:
            run = store_hit_fit(
                result,
                path.stem,
                reference=signal,
                store=RunStore(args.runs) if args.runs else None,
                plot=not args.no_plot,
            )
            out_dir = run.path
            written = {p.name: p for p in run.files()}

        print(result.summary())
        print(f"written to       {out_dir}/")
        print(f"  {len(written)} files" + (", figures/" if not args.no_plot else ""))
        print()

        if not result.target_reached:
            print(
                "the target was not reached by any candidate in this space — "
                "try --space full, or a looser --target-mse",
                file=sys.stderr,
            )
    return 0


def _print_progress(progress: Progress) -> None:
    """One line, rewritten in place on a terminal and appended in a pipe."""
    best = progress.best
    percent = 100.0 * progress.done / max(progress.total, 1)
    line = (
        f"  {progress.done:5d}/{progress.total} ({percent:5.1f}%) "
        f"{progress.elapsed:6.1f}s   best: {best.relative_mse:.3e} "
        f"in {best.n_scalars:7d} numbers  {best.candidate.label()}"
    )

    if sys.stdout.isatty():
        print(f"\r{line:<110}", end="" if progress.done < progress.total else "\n", flush=True)
    elif progress.done >= progress.total or percent % 25 < 100.0 / progress.total:
        print(line, flush=True)


def _run_drums(args) -> int:
    corpus = Corpus.load(args.library)
    print(f"{'drum':28s} {'samples':>8s} {'on disk':>8s} {'velocities':>11s}")
    for name in corpus.names:
        samples = corpus.samples(name)
        present = corpus.samples(name, present_only=True)
        velocities = sorted({s.velocity for s in present}) or sorted(
            {s.velocity for s in samples}
        )
        span = f"{velocities[0]:g}-{velocities[-1]:g}" if velocities else "-"
        print(f"{name:28s} {len(samples):8d} {len(present):8d} {span:>11s}")
    return 0


def _run_fit_drum(args) -> int:
    corpus = Corpus.load(args.library)
    layers = VelocityLayers.from_corpus(
        args.drum,
        corpus,
        sample_rate=args.sample_rate,
        max_duration=args.max_duration,
    )
    print(
        f"{layers.name}: {layers.n_recordings} recordings, {layers.n_layers} velocities, "
        f"{layers.duration:.3f} s at {layers.sample_rate} Hz"
    )

    result = fit_instrument(
        layers,
        target_mse=args.target_mse,
        space=INSTRUMENT_SPACES[args.space](),
        take=args.take,
        average=args.average_takes,
        n_donors=args.donors,
        progress=None if args.quiet else _print_progress,
    )

    if args.out:
        out_dir = args.out
        written = save_instrument_fit(
            result, out_dir, layers=layers, take=args.take, plot=not args.no_plot
        )
    else:
        run = store_instrument_fit(
            result,
            layers,
            store=RunStore(args.runs) if args.runs else None,
            take=args.take,
            plot=not args.no_plot,
        )
        out_dir = run.path
        written = {p.name: p for p in run.files()}

    print(result.summary())
    print(f"written to       {out_dir}/")
    print(f"  {len(written)} files" + (", figures/" if not args.no_plot else ""))
    print()

    if not result.target_reached:
        print(
            "the target was not reached by any candidate in this space — "
            "try --space full, or a looser --target-mse",
            file=sys.stderr,
        )
    return 0


def _run_runs(args) -> int:
    store = RunStore(args.root) if args.root else RunStore.default()
    runs = store.runs(args.name, kind=args.kind)

    if not runs:
        where = f" for {args.name}" if args.name else ""
        print(f"no {args.kind} runs{where} in {store.root}/")
        return 0

    print(f"{'when':17s} {'name':26s} {'error':>10s} {'size':>9s}  representation")
    for run in runs:
        error = run.metadata.get("reconstruction_mse", run.metadata.get("relative_mse"))
        print(
            f"{run.created.strftime('%Y-%m-%d %H:%M'):17s} {run.name:26s} "
            f"{error if error is None else f'{error:10.3e}'} "
            f"{run.metadata.get('bytes', 0) / 1024:8.1f}k  {run.summary}"
        )
    return 0


def _run_bench(args) -> int:
    from .bench import measure_live, measure_polyphony

    model = _load_model(args.model)
    velocity = args.velocity
    if velocity is None and isinstance(model, InstrumentModel):
        low, high = model.velocity_range
        velocity = 0.5 * (low + high)

    cost = measure_live(model, velocity, block=args.block)
    print(cost.report())

    if args.voices:
        print()
        print(
            f"{'voices':>7s} {'mean ms':>9s} {'p95 ms':>9s} {'max ms':>9s} "
            f"{'worst block':>12s}"
        )
        for count in args.voices:
            measured = measure_polyphony(model, velocity, block=args.block, voices=count)
            print(
                f"{count:7d} {measured['mean_ms']:9.3f} {measured['p95_ms']:9.3f} "
                f"{measured['max_ms']:9.3f} {100 * measured['worst_load']:11.1f}%"
            )
    return 0


def _load_model(path: Path):
    """Whichever kind of model is in the file."""
    try:
        return InstrumentModel.load(path)
    except ValueError:
        return SpectralModel.load(path)


def _run_play(args) -> int:
    model = InstrumentModel.load(args.model)
    output = args.output or args.model.with_name(
        f"{args.model.stem}-v{args.velocity:g}.wav"
    )
    AudioIO.write(output, model.render(args.velocity), model.sample_rate)

    low, high = model.velocity_range
    if not low <= args.velocity <= high:
        print(
            f"velocity {args.velocity:g} is outside the recorded range "
            f"{low:g}-{high:g}; the nearest layer was used",
            file=sys.stderr,
        )
    print(f"{output}  velocity {args.velocity:g}  {model.duration:.3f} s")
    return 0


def _run_decode(args) -> int:
    model = SpectralModel.load(args.model)
    output = args.output or args.model.with_suffix(".wav")
    AudioIO.write(output, model.render(), model.sample_rate)
    print(f"{output}  {model.duration:.3f} s at {model.sample_rate} Hz")
    return 0


def _run_inspect(args) -> int:
    try:
        return _inspect_instrument(args)
    except ValueError:
        pass

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


def _inspect_instrument(args) -> int:
    model = InstrumentModel.load(args.model)
    low, high = model.velocity_range

    print(f"{args.model}")
    print(f"  instrument     {model.name}")
    print(f"  representation {model.candidate.label()}")
    print(f"  audio          {model.duration:.3f} s at {model.sample_rate} Hz")
    print(
        f"  velocities     {model.velocities.size} layers from {low:g} to {high:g}, "
        f"{model.donor_velocities.size} of them donating phase"
    )
    print(f"  size           {model.n_scalars} numbers, {model.n_bytes() / 1024:.1f} kB")
    frequencies = model.frequencies()
    print(
        f"  bins           {model.bins.size} between {frequencies.min():.1f} Hz "
        f"and {frequencies.max():.1f} Hz"
    )
    for name, array in sorted(model.field_arrays.items()):
        print(f"  field.{name:<9s} {tuple(array.shape)} {array.dtype}")
    for name, array in sorted(model.donor_arrays[0].items()):
        print(f"  donor.{name:<9s} {tuple(array.shape)} {array.dtype} x {len(model.donor_arrays)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
