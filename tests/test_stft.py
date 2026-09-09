"""The transform has to be exact, or no error measured through it means anything."""

from __future__ import annotations

import numpy as np
import pytest

from drumsynth.spectral.stft import StftSpec, analyze, synthesize


@pytest.mark.parametrize(
    "n_fft,overlap", [(256, 2), (256, 4), (1024, 2), (1024, 4), (1024, 8)]
)
@pytest.mark.parametrize("n_samples", [1, 255, 4096, 9137])
def test_unmodified_spectrogram_round_trips_to_machine_precision(n_fft, overlap, n_samples):
    spec = StftSpec(n_fft, n_fft // overlap)
    signal = np.random.default_rng(0).standard_normal(n_samples)

    recovered = synthesize(analyze(signal, spec), spec, n_samples)

    assert recovered.shape == signal.shape
    assert np.max(np.abs(recovered - signal)) < 1e-10


def test_frame_count_and_shape():
    spec = StftSpec(512, 128)
    signal = np.zeros(4000)

    spectrogram = analyze(signal, spec)

    assert spectrogram.shape == (spec.n_bins, spec.n_frames(4000))
    assert spec.n_frames(4000) == 1 + 4000 // 128


def test_hop_must_divide_the_transform_size():
    """The overlap-add is written as block adds; a fractional overlap would break it."""
    with pytest.raises(ValueError, match="multiple of hop"):
        StftSpec(1024, 384)


def test_rejects_odd_or_empty_transform():
    with pytest.raises(ValueError, match="positive and even"):
        StftSpec(1023, 1)
    with pytest.raises(ValueError, match="positive and even"):
        StftSpec(0, 1)


def test_window_is_periodic_not_symmetric():
    spec = StftSpec(8, 4)
    window = spec.window

    assert window[0] == pytest.approx(0.0)
    assert window[-1] != pytest.approx(0.0)
    assert np.allclose(window[1:], window[1:][::-1])


def test_bin_frequencies_span_dc_to_nyquist():
    spec = StftSpec(1024, 256)
    frequencies = spec.bin_frequencies(44100)

    assert frequencies[0] == 0.0
    assert frequencies[-1] == pytest.approx(22050.0)
    assert frequencies.size == spec.n_bins
