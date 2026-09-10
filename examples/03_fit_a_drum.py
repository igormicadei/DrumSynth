#!/usr/bin/env python3
"""Fit one drum across every velocity it was recorded at, and play it back.

    python examples/03_fit_a_drum.py toms-stereo-tom3

Writes `out/<drum>/` — the model, the sweeps to listen to, and the report. The
audio has to be on this machine: the library indexes 3,338 recordings and the
repository ships one of them (see docs/DATA.md), so on a fresh clone this will
find a single velocity and say so.
"""

from __future__ import annotations

import sys

from drumsynth import Corpus
from drumsynth.instrument import (
    InstrumentModel,
    InstrumentSearchSpace,
    VelocityLayers,
    fit_instrument,
    save_instrument_fit,
)

TARGET_MSE = 1e-4


def main() -> None:
    drum = sys.argv[1] if len(sys.argv) > 1 else "toms-stereo-tom3"
    corpus = Corpus.load()

    present = corpus.samples(drum, present_only=True)
    print(f"{drum}: {len(present)} of {len(corpus.samples(drum))} recordings on disk")

    layers = VelocityLayers.from_corpus(drum, corpus)
    print(
        f"  {layers.n_layers} velocities, {layers.velocities.min():g} to "
        f"{layers.velocities.max():g}, {layers.duration:.2f} s each"
    )

    result = fit_instrument(
        layers,
        target_mse=TARGET_MSE,
        space=InstrumentSearchSpace.quick(),
        progress=lambda p: print(
            f"\r  {p.done}/{p.total}  best {p.best.relative_mse:.2e} "
            f"in {p.best.n_scalars} numbers",
            end="",
        ),
    )
    print()
    print(result.summary())

    written = save_instrument_fit(result, f"out/{drum}", layers=layers)
    print(f"\nwritten to out/{drum}: {', '.join(p.name for p in written.values())}")

    # The file is the instrument: every velocity, including the ones between.
    model = InstrumentModel.load(written["model"])
    low, high = model.velocity_range
    print(f"reloaded {model.name}: playable from {low:g} to {high:g}")
    print(f"  a hit at {0.5 * (low + high):g} -> {model.render(0.5 * (low + high)).shape[0]} samples")


if __name__ == "__main__":
    main()
