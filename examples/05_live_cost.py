#!/usr/bin/env python3
"""What it costs to play one model live, measured across the things that could matter.

    python examples/05_live_cost.py

Four questions, four tables:

    1. does the number of partials cost anything to play?
    2. does the transform size?
    3. what does the audio block size change?
    4. how many voices fit in one core?

The answers are in docs/LIVE.md. Rerun this after touching the transform or
the streaming path — it is the measurement those numbers come from.
"""

from __future__ import annotations

import numpy as np

from drumsynth import AudioIO, Corpus
from drumsynth.bench import measure_live, measure_polyphony
from drumsynth.spectral import Candidate, encode

BLOCK = 256


def load_hit() -> tuple[np.ndarray, int]:
    """The recording that ships with the repository, or a synthetic stand-in."""
    try:
        sample = Corpus.load().samples(present_only=True)[0]
        return sample.load()
    except (FileNotFoundError, IndexError):  # pragma: no cover - depends on the checkout
        sr = 44100
        t = np.arange(int(3.5 * sr)) / sr
        signal = sum(
            a * np.exp(-t / d) * np.sin(2 * np.pi * f * t)
            for f, a, d in [(92.5, 1.0, 0.6), (147.4, 0.5, 0.4), (612.0, 0.2, 0.2)]
        )
        return signal, sr


def row(label: str, cost) -> None:
    print(
        f"  {label:22s} {cost.trigger_ms:8.2f} {cost.block_mean_ms:10.3f} "
        f"{cost.block_max_ms:9.3f} {100 * cost.load:9.2f}% {cost.voices:8.0f} "
        f"{cost.resident_bytes / 1024:9.0f} {cost.voice_bytes / 1024:9.0f}"
    )


def header(title: str) -> None:
    print(f"\n{title}")
    print(
        f"  {'':22s} {'trigger':>8s} {'per block':>10s} {'worst':>9s} "
        f"{'of a core':>10s} {'voices':>8s} {'model kB':>9s} {'voice kB':>9s}"
    )


def main() -> None:
    signal, sample_rate = load_hit()
    print(f"{len(signal) / sample_rate:.2f} s at {sample_rate} Hz, block {BLOCK} samples")

    header("partials kept (n_fft 2048)")
    for k in (32, 64, 128, 256, 512):
        model = encode(signal, sample_rate, Candidate(2048, 1024, k, "lowrank", 16))
        row(f"k = {k}", measure_live(model, block=BLOCK, repeats=3))

    header("transform size (k = 128)")
    for n_fft in (512, 1024, 2048, 4096, 8192):
        model = encode(signal, sample_rate, Candidate(n_fft, n_fft // 2, 128, "lowrank", 16))
        row(f"n_fft = {n_fft}", measure_live(model, block=BLOCK, repeats=3))

    header("audio block (n_fft 2048, k 128)")
    model = encode(signal, sample_rate, Candidate(2048, 1024, 128, "lowrank", 16))
    for block in (64, 128, 256, 512, 1024):
        row(f"block = {block}", measure_live(model, block=block, repeats=3))

    print("\npolyphony (n_fft 2048, k 128, block 256)")
    print(f"  {'voices':>8s} {'mean ms':>9s} {'p95 ms':>9s} {'max ms':>9s} {'worst block':>12s}")
    for voices in (1, 2, 4, 8, 16, 32, 64):
        measured = measure_polyphony(model, block=BLOCK, voices=voices)
        print(
            f"  {voices:8d} {measured['mean_ms']:9.3f} {measured['p95_ms']:9.3f} "
            f"{measured['max_ms']:9.3f} {100 * measured['worst_load']:11.1f}%"
        )

    quality = np.sum((signal - model.render()) ** 2) / np.sum(signal**2)
    print(f"\n(the model being played is {quality:.1e} relative error, {model.n_bytes() / 1024:.0f} kB)")
    AudioIO.write("/tmp/live_cost_check.wav", model.render(), sample_rate)


if __name__ == "__main__":
    main()
