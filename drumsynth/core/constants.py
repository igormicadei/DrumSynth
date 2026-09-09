"""Audio conventions and the decibel conversions built on them.

Static methods on classes rather than loose functions, so the unit system has
one obvious home: if a number changes meaning — amplitude dB against power dB
is the trap — there is exactly one place it can be converted, and mixing two
conventions becomes hard to do by accident.
"""

from __future__ import annotations

import math

import numpy as np


def _scalarize(value: np.ndarray):
    """Return a Python float for 0-d results, the array otherwise."""
    arr = np.asarray(value)
    return float(arr) if arr.ndim == 0 else arr


class Audio:
    """Default audio conventions.

    Everything in this project is mono float64 at a single sample rate. Stereo
    files are summed on load; there is no channel concept anywhere downstream.
    """

    DEFAULT_SR: int = 44100
    MIN_SR: int = 8000
    MAX_SR: int = 192000

    #: Amplitude below which a signal is treated as silence in dB conversions.
    EPS: float = 1e-12

    #: dB value substituted for a true zero amplitude.
    SILENCE_DB: float = -300.0


class Decibels:
    """Amplitude <-> dB conversions.

    Amplitude ratios throughout (20 * log10), never power ratios. Mixing the
    two silently halves or doubles every reported error.
    """

    EPS: float = Audio.EPS
    FLOOR_DB: float = Audio.SILENCE_DB

    @staticmethod
    def from_amplitude(amplitude):
        """20 * log10(|amplitude|), floored at FLOOR_DB rather than -inf."""
        a = np.abs(np.asarray(amplitude, dtype=float))
        db = 20.0 * np.log10(np.maximum(a, Decibels.EPS))
        return _scalarize(np.maximum(db, Decibels.FLOOR_DB))

    @staticmethod
    def to_amplitude(db):
        return _scalarize(10.0 ** (np.asarray(db, dtype=float) / 20.0))

    @staticmethod
    def from_power(power):
        """10 * log10(power). Use only where the input is genuinely a power."""
        p = np.maximum(np.asarray(power, dtype=float), Decibels.EPS**2)
        return _scalarize(np.maximum(10.0 * np.log10(p), Decibels.FLOOR_DB))

    @staticmethod
    def to_power(db):
        return _scalarize(10.0 ** (np.asarray(db, dtype=float) / 10.0))


# Module-level aliases. Convenient, and the only loose names in the package.
DEFAULT_SR: int = Audio.DEFAULT_SR
TWO_PI: float = 2.0 * math.pi
