"""Shared fixtures. Kept small on purpose — most tests build their own signal."""

from __future__ import annotations

import numpy as np
import pytest

from drumsynth import DrumPresets, DrumVoice
from drumsynth.core import Decay

SR = 44100


@pytest.fixture(scope="session")
def sr() -> int:
    return SR


@pytest.fixture(scope="session")
def tom_params():
    return DrumPresets.tom()


@pytest.fixture
def voice(tom_params):
    return DrumVoice(tom_params, SR, control_period=64, seed=1)


@pytest.fixture(scope="session")
def synthetic_modes():
    """(freq, amplitude, t60) of a signal with exactly known content."""
    return [
        (92.5, 1.00, 2.30),
        (88.0, 0.70, 2.20),  # deliberate close pair, 86 cents apart
        (147.4, 0.45, 1.10),
        (197.0, 0.30, 0.90),
        (600.0, 0.12, 0.55),
        (1900.0, 0.05, 0.31),
    ]


@pytest.fixture(scope="session")
def synthetic_signal(synthetic_modes):
    """A sum of exponentially decaying sinusoids with a -100 dB noise floor."""
    n = int(2.5 * SR)
    t = np.arange(n) / SR
    signal = sum(
        amplitude * np.exp(-Decay.t60_to_damping(t60) * t) * np.cos(2 * np.pi * freq * t)
        for freq, amplitude, t60 in synthetic_modes
    )
    return signal + 1e-5 * np.random.default_rng(0).standard_normal(n)
