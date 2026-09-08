"""Physical and numerical constants, and the unit conversions built on them.

Everything here is a class of static methods rather than loose functions so the
unit system has one obvious home: if a number changes meaning (amplitude vs
power dB, cents vs Hz, t60 vs decay coefficient) there is exactly one place it
can be converted, and mixing two conventions becomes hard to do by accident.
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


class Cents:
    """Musical interval conversions.

    Frequency errors are always reported in cents. A 1 Hz error means something
    completely different at 90 Hz than at 9 kHz; a cents error does not.
    """

    @staticmethod
    def between(f_a, f_b):
        """Interval from f_b up to f_a, in cents. NaN if either is non-positive."""
        a = np.asarray(f_a, dtype=float)
        b = np.asarray(f_b, dtype=float)
        valid = (a > 0) & (b > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(valid, a / np.where(valid, b, 1.0), np.nan)
            out = 1200.0 * np.log2(ratio)
        return _scalarize(out)

    @staticmethod
    def to_ratio(cents):
        return _scalarize(2.0 ** (np.asarray(cents, dtype=float) / 1200.0))

    @staticmethod
    def from_ratio(ratio):
        return _scalarize(1200.0 * np.log2(np.asarray(ratio, dtype=float)))

    @staticmethod
    def to_semitones(cents):
        return _scalarize(np.asarray(cents, dtype=float) / 100.0)

    @staticmethod
    def from_semitones(semitones):
        return _scalarize(np.asarray(semitones, dtype=float) * 100.0)


class Decay:
    """t60 <-> per-sample decay coefficient, and the equivalent slope in dB/s.

    A mode's amplitude is multiplied by `r` once per sample. Reaching -60 dB
    after t60 seconds means r ** (t60 * sr) == 1/1000, hence

        r = exp(-ln(1000) / (t60 * sr))
    """

    #: ln(1000). The single constant tying t60 to every other decay quantity.
    LN1000: float = 6.907755278982137

    @staticmethod
    def t60_to_coef(t60, sr: int):
        """Per-sample amplitude multiplier reaching -60 dB at `t60` seconds."""
        t = np.asarray(t60, dtype=float)
        if np.any(t <= 0.0):
            raise ValueError(f"t60 must be positive, got {t60!r}")
        return _scalarize(np.exp(-Decay.LN1000 / (t * float(sr))))

    @staticmethod
    def coef_to_t60(r, sr: int):
        """Inverse of `t60_to_coef`. Returns inf for r >= 1 (no decay)."""
        rr = np.asarray(r, dtype=float)
        if np.any(rr <= 0.0):
            raise ValueError(f"decay coefficient must be positive, got {r!r}")
        with np.errstate(divide="ignore"):
            out = np.where(rr >= 1.0, np.inf, -Decay.LN1000 / (float(sr) * np.log(rr)))
        return _scalarize(out)

    @staticmethod
    def t60_to_slope_db_s(t60):
        """Decay as a straight line in dB: -60 / t60 dB per second."""
        t = np.asarray(t60, dtype=float)
        with np.errstate(divide="ignore"):
            return _scalarize(np.where(t > 0, -60.0 / np.where(t > 0, t, 1.0), -np.inf))

    @staticmethod
    def slope_db_s_to_t60(slope_db_s):
        """Inverse. A non-negative slope is not a decay and returns inf."""
        s = np.asarray(slope_db_s, dtype=float)
        with np.errstate(divide="ignore"):
            return _scalarize(np.where(s < 0, -60.0 / np.where(s < 0, s, -1.0), np.inf))

    @staticmethod
    def damping_to_t60(alpha):
        """Continuous-time damping (envelope exp(-alpha * t)) to t60."""
        a = np.asarray(alpha, dtype=float)
        with np.errstate(divide="ignore"):
            return _scalarize(np.where(a > 0, Decay.LN1000 / np.where(a > 0, a, 1.0), np.inf))

    @staticmethod
    def t60_to_damping(t60):
        t = np.asarray(t60, dtype=float)
        with np.errstate(divide="ignore"):
            return _scalarize(np.where(t > 0, Decay.LN1000 / np.where(t > 0, t, 1.0), 0.0))


# Module-level aliases. Convenient, and the only loose names in the package.
DEFAULT_SR: int = Audio.DEFAULT_SR
LN1000: float = Decay.LN1000
TWO_PI: float = 2.0 * math.pi
