#!/usr/bin/env python3
"""Why the search exists: no codec wins on every sound.

    python examples/02_compare_codecs.py

Takes one tonal hit and one noise burst, holds the analysis fixed, and prints
what each codec costs to reach a given error on each. This is the measurement
behind docs/FINDINGS.md — rerun it after touching a codec.
"""

from __future__ import annotations

import numpy as np

from drumsynth.spectral.bins import select_bins
from drumsynth.spectral.components import CODECS, codec_for, to_components
from drumsynth.spectral.stft import StftSpec, analyze

SR = 44100
SPEC = StftSpec(2048, 1024)
COMPONENTS = 128
TARGETS = (1e-2, 1e-3, 1e-4)


def tonal_hit() -> np.ndarray:
    """A struck drum: a few partials, each decaying at its own rate."""
    t = np.arange(int(1.5 * SR)) / SR
    return sum(
        a * np.exp(-t / d) * np.sin(2 * np.pi * f * t + p)
        for f, a, d, p in [
            (92.5, 1.00, 0.62, 0.0),
            (147.4, 0.55, 0.44, 1.1),
            (197.0, 0.30, 0.30, 2.4),
            (612.0, 0.12, 0.18, 0.7),
        ]
    )


def noise_burst() -> np.ndarray:
    """A cymbal-like wash: broadband, decaying, no low-rank structure."""
    t = np.arange(int(1.5 * SR)) / SR
    rng = np.random.default_rng(11)
    return np.exp(-t / 0.5) * rng.standard_normal(t.size)


def block_error(reference: np.ndarray, estimate: np.ndarray) -> float:
    return float(
        np.sum(np.abs(reference - estimate) ** 2) / np.sum(np.abs(reference) ** 2)
    )


def cheapest(codec, block: np.ndarray, target: float) -> tuple[int, str] | None:
    """Fewest scalars this codec needs to hold `block` within `target`."""
    k, frames = block.shape
    prepared = codec.prepare(block)
    best: tuple[int, str] | None = None

    params = [0] if codec.name == "raw" else [1, 2, 4, 8, 16, 24, 32, 48, 64]
    strides = [1, 2, 4, 8] if codec.uses_phase else [1]

    for param in params:
        for stride in strides:
            if block_error(block, codec.decode(codec.encode(prepared, param, stride), frames)) > target:
                continue
            cost = codec.n_scalars(k, frames, param, stride)
            if best is None or cost < best[0]:
                best = (cost, f"param={param} stride={stride}")
    return best


def main() -> None:
    for name, signal in [("tonal hit", tonal_hit()), ("noise burst", noise_burst())]:
        spectrogram = analyze(signal, SPEC)
        bins = select_bins(spectrogram, COMPONENTS)
        block = to_components(spectrogram, bins, SPEC)

        codecs = sorted(CODECS)
        print(f"\n{name}: {block.shape[0]} bins x {block.shape[1]} frames")
        print("  scalars needed to hold the block within a relative error")
        print(f"  {'target':>8s}" + "".join(f"{codec:>28s}" for codec in codecs))

        for target in TARGETS:
            cells = []
            for codec_name in codecs:
                found = cheapest(codec_for(codec_name), block, target)
                cells.append("out of reach" if found is None else f"{found[0]:6d}  {found[1]}")
            print(f"  {target:8.0e}" + "".join(f"{cell:>28s}" for cell in cells))


if __name__ == "__main__":
    main()
