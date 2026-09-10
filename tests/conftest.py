"""Shared fixtures. Deliberately small — most tests build the signal they need."""

from __future__ import annotations

import numpy as np
import pytest

from drumsynth.corpus import Corpus
from drumsynth.instrument import VelocityLayers

SR = 22050


@pytest.fixture(scope="session")
def sr() -> int:
    return SR


@pytest.fixture(scope="session")
def partials() -> list[tuple[float, float, float]]:
    """(frequency, amplitude, decay time) of a signal with known content."""
    return [
        (183.0, 1.00, 0.42),
        (297.5, 0.55, 0.30),
        (612.0, 0.28, 0.16),
        (1290.0, 0.11, 0.09),
    ]


@pytest.fixture(scope="session")
def tonal_hit(partials) -> np.ndarray:
    """A struck tonal drum: a few decaying partials, no noise.

    Short on purpose. Every test that runs a search runs it on this, and a
    search evaluates hundreds of candidates end to end.
    """
    n = int(0.3 * SR)
    t = np.arange(n) / SR
    signal = sum(
        amplitude * np.exp(-t / decay) * np.sin(2 * np.pi * frequency * t + 0.3)
        for frequency, amplitude, decay in partials
    )
    return np.asarray(signal, dtype=np.float64)


@pytest.fixture(scope="session")
def noisy_hit() -> np.ndarray:
    """A decaying noise burst: the case low-rank structure cannot help with."""
    n = int(0.3 * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(7)
    return np.exp(-t / 0.12) * rng.standard_normal(n)


@pytest.fixture(scope="session")
def velocity_layers() -> VelocityLayers:
    """A synthetic drum recorded at six velocities, twice each.

    Louder hits are brighter, ring longer and sit a little flatter — the three
    things a velocity model has to follow. Every strike gets its own phases,
    including the first: two hits of a real drum are never in phase with each
    other, and a fixture that pretends otherwise would make borrowing phase
    across velocities look like it works.
    """
    rng = np.random.default_rng(5)
    modes = [(96.0, 0.30), (151.0, 0.22), (228.0, 0.16), (415.0, 0.11), (902.0, 0.07)]
    n = int(0.25 * SR)
    t = np.arange(n) / SR

    def strike(velocity: float, take: int) -> np.ndarray:
        loudness = velocity / 127.0
        signal = np.zeros(n)
        for index, (frequency, decay) in enumerate(modes):
            amplitude = (loudness**1.3) * (0.85**index) * (loudness ** (0.4 * index))
            signal += (
                amplitude
                * np.exp(-t / (decay * (0.8 + 0.4 * loudness)))
                * np.sin(
                    2 * np.pi * frequency * (1 - 0.01 * loudness) * t
                    + rng.uniform(0, 2 * np.pi)
                )
            )
        signal += 0.03 * loudness * np.exp(-t / 0.02) * rng.standard_normal(n)
        return signal

    velocities = [10.0, 30.0, 50.0, 75.0, 100.0, 125.0]
    return VelocityLayers.from_audio(
        "synthetic-drum", SR, {v: [strike(v, take) for take in range(2)] for v in velocities}
    )


@pytest.fixture(scope="session")
def recorded_hit() -> tuple[np.ndarray, int]:
    """The one real recording kept in the repository, trimmed to half a second."""
    corpus = Corpus.load()
    present = corpus.samples(present_only=True)
    if not present:
        import pytest as _pytest

        _pytest.skip("no sample audio checked out")

    signal, rate = present[0].load()
    return signal[: rate // 2], rate
