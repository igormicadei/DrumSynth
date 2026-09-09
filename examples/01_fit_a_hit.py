#!/usr/bin/env python3
"""Fit the one recording that ships with the repository, and save the run.

    python examples/01_fit_a_hit.py

Writes `out/tom/` — the model, what it renders to, what it missed, the report
and the frontier. The same thing `drumsynth fit` does, from the API.
"""

from __future__ import annotations

from drumsynth import Corpus, SpectralModel, save_fit
from drumsynth.fitting import SearchSpace, fit

TARGET_MSE = 1e-5


def main() -> None:
    sample = Corpus.load().samples(present_only=True)[0]
    signal, sample_rate = sample.load()
    print(f"{sample.name}: {len(signal) / sample_rate:.3f} s at {sample_rate} Hz")

    result = fit(
        signal,
        sample_rate,
        target_mse=TARGET_MSE,
        space=SearchSpace.quick(),
        jobs=0,
        progress=lambda p: print(
            f"\r  {p.done}/{p.total}  best {p.best.quality.relative_mse:.2e} "
            f"in {p.best.n_scalars} numbers",
            end="",
        ),
    )
    print()
    print(result.summary())

    written = save_fit(result, "out/tom", reference=signal, input_path=sample.path)
    print(f"\nwritten to out/tom: {', '.join(p.name for p in written.values())}")

    # The file is the model. Nothing from this session is needed to play it.
    again = SpectralModel.load(written["model"])
    print(f"reloaded {again.candidate.label()} -> {again.render().shape[0]} samples")


if __name__ == "__main__":
    main()
