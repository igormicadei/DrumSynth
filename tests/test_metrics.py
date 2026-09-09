"""The measures the whole search is ranked by."""

from __future__ import annotations

import math

import numpy as np
import pytest

from drumsynth.fitting.metrics import (
    Quality,
    correlation,
    peak,
    relative_mse,
    rms,
    snr_db,
)


def test_an_exact_reconstruction_scores_zero_error():
    signal = np.array([0.5, -0.25, 0.125])

    assert relative_mse(signal, signal) == 0.0
    assert correlation(signal, signal) == pytest.approx(1.0)
    assert snr_db(0.0) == math.inf


def test_silence_in_the_reference_is_not_a_division_by_zero():
    assert relative_mse(np.zeros(16), np.ones(16)) == 0.0
    assert correlation(np.zeros(16), np.ones(16)) == 0.0


def test_relative_error_is_the_ratio_of_energies():
    signal = np.array([1.0, 1.0, 1.0, 1.0])
    estimate = np.array([1.0, 1.0, 1.0, 0.0])

    assert relative_mse(signal, estimate) == pytest.approx(0.25)
    assert snr_db(0.25) == pytest.approx(6.0206, abs=1e-3)


def test_relative_error_does_not_depend_on_level():
    reference = np.random.default_rng(0).standard_normal(512)
    error = 0.01 * np.random.default_rng(1).standard_normal(512)

    quiet = relative_mse(reference, reference + error)
    loud = relative_mse(100 * reference, 100 * (reference + error))

    assert quiet == pytest.approx(loud)


def test_quality_reports_level_as_well_as_shape():
    reference = np.array([1.0, -1.0, 1.0, -1.0])

    quality = Quality.measure(reference, 0.5 * reference)

    assert quality.correlation == pytest.approx(1.0)
    assert quality.rms_ratio == pytest.approx(0.5)
    assert quality.peak_ratio == pytest.approx(0.5)
    assert quality.relative_mse == pytest.approx(0.25)


def test_quality_survives_a_round_trip_through_a_dict():
    quality = Quality.measure(np.array([1.0, 0.0]), np.array([0.9, 0.1]))

    data = quality.to_dict()

    assert data["relative_mse"] == quality.relative_mse
    assert data["rms_ratio"] == quality.rms_ratio


def test_rms_and_peak_of_nothing_are_zero():
    empty = np.zeros(0)

    assert rms(empty) == 0.0
    assert peak(empty) == 0.0
