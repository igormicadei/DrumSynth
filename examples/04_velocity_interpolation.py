#!/usr/bin/env python3
"""What the velocity axis costs, and what it is worth.

    python examples/04_velocity_interpolation.py

Builds a synthetic drum — 26 velocities, two strikes each, whose spectrum
brightens and lengthens with velocity the way a real one does — and asks three
questions of it:

    1. interpolating magnitude between two velocities: linearly, or in dB?
    2. is interpolating better than just taking the nearest recorded layer?
    3. how do either of those compare with the difference between two real
       strikes at the *same* velocity?

The third is the one that matters. A velocity the model invents is not required
to be perfect; it is required to be no further from the truth than one strike of
a drum is from the next.

The library's own recordings are not in the repository (docs/DATA.md), so this
is a synthetic instrument with a known velocity trend. It is the measurement
behind docs/FINDINGS.md §8 — rerun it after touching the field.
"""

from __future__ import annotations

import numpy as np

from drumsynth.instrument import VelocityLayers
from drumsynth.instrument.fit import magnitude_error
from drumsynth.spectral.stft import StftSpec

SR = 44100
SPEC = StftSpec(2048, 1024)
COMPONENTS = 256

MODES = [
    (92.5, 0.62), (147.4, 0.44), (197.0, 0.33), (232.0, 0.28), (311.0, 0.22),
    (415.0, 0.17), (612.0, 0.13), (902.0, 0.10), (1320.0, 0.07), (1900.0, 0.05),
]


def synthetic_drum(n_velocities: int = 26, takes: int = 2, seconds: float = 3.5) -> VelocityLayers:
    """A drum that gets louder, brighter, longer and slightly flatter when hit harder."""
    rng = np.random.default_rng(5)
    t = np.arange(int(seconds * SR)) / SR

    def strike(velocity: float) -> np.ndarray:
        loudness = velocity / 127.0
        signal = np.zeros(t.size)
        for index, (frequency, decay) in enumerate(MODES):
            amplitude = (
                (loudness**1.3) * (0.82**index) * (loudness ** (0.35 * index))
                * rng.uniform(0.9, 1.1)
            )
            signal += amplitude * np.exp(-t / (decay * (0.8 + 0.4 * loudness))) * np.sin(
                2 * np.pi * frequency * (1 - 0.008 * loudness) * t + rng.uniform(0, 2 * np.pi)
            )
        signal += 0.04 * loudness * np.exp(-t / (0.01 + 0.02 * loudness)) * rng.standard_normal(t.size)
        return signal

    velocities = np.linspace(2.5, 110.0, n_velocities)
    return VelocityLayers.from_audio(
        "synthetic-tom", SR, {float(v): [strike(v) for _ in range(takes)] for v in velocities}
    )


def main() -> None:
    layers = synthetic_drum()
    bins = layers.select_bins(SPEC, COMPONENTS)
    magnitudes, _ = layers.analyse(SPEC, bins, take=0)
    other_take, _ = layers.analyse(SPEC, bins, take=1)
    velocities = layers.velocities

    print(f"{layers.n_layers} velocities, {magnitudes.shape[1]} bins, {magnitudes.shape[2]} frames")

    kept = np.arange(0, layers.n_layers, 2)
    held_out = [
        index
        for index in range(layers.n_layers)
        if index not in kept and kept.min() < index < kept.max()
    ]
    floor = 1e-6 * magnitudes.max()

    linear, decibels, nearest = [], [], []
    for index in held_out:
        low = kept[kept < index].max()
        high = kept[kept > index].min()
        position = (velocities[index] - velocities[low]) / (velocities[high] - velocities[low])

        linear.append(
            magnitude_error(
                magnitudes[index],
                (1 - position) * magnitudes[low] + position * magnitudes[high],
            )
        )
        decibels.append(
            magnitude_error(
                magnitudes[index],
                np.exp(
                    (1 - position) * np.log(magnitudes[low] + floor)
                    + position * np.log(magnitudes[high] + floor)
                )
                - floor,
            )
        )
        nearest.append(
            magnitude_error(magnitudes[index], magnitudes[low if position < 0.5 else high])
        )

    strikes = [
        magnitude_error(magnitudes[index], other_take[index])
        for index in range(layers.n_layers)
    ]

    print(f"\npredicting {len(held_out)} velocities held out of the fit:")
    print(f"  interpolated linearly       {np.mean(linear):.4f}")
    print(f"  interpolated in dB          {np.mean(decibels):.4f}")
    print(f"  nearest recorded layer      {np.mean(nearest):.4f}")
    print("\nfor scale, at velocities that were recorded:")
    print(f"  one strike against another  {np.mean(strikes):.4f}")


if __name__ == "__main__":
    main()
