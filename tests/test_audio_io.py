"""Reading and writing has to be transparent, or a measured error is the writer's."""

from __future__ import annotations

import numpy as np
import pytest

from drumsynth import AudioIO


def test_a_float_wav_round_trips_without_rescaling(tmp_path):
    signal = np.linspace(-0.8, 0.8, 1000)

    path = AudioIO.write(tmp_path / "x.wav", signal, 44100)
    recovered, rate = AudioIO.read(path)

    assert rate == 44100
    assert np.max(np.abs(recovered - signal)) < 1e-7


def test_writing_does_not_normalize_unless_asked(tmp_path):
    signal = 0.02 * np.sin(np.linspace(0, 40, 800))

    quiet, _ = AudioIO.read(AudioIO.write(tmp_path / "quiet.wav", signal, 22050))
    loud, _ = AudioIO.read(
        AudioIO.write(tmp_path / "loud.wav", signal, 22050, normalize=True)
    )

    assert np.max(np.abs(quiet)) == pytest.approx(0.02, rel=1e-3)
    assert np.max(np.abs(loud)) == pytest.approx(10 ** (-1.0 / 20.0), rel=1e-3)


def test_a_float_wav_keeps_samples_past_full_scale(tmp_path):
    """A reconstruction can overshoot; clipping it would hide the model's error."""
    signal = np.array([1.5, -1.5, 0.0, 0.5])

    recovered, _ = AudioIO.read(AudioIO.write(tmp_path / "hot.wav", signal, 8000))

    assert np.allclose(recovered, signal, atol=1e-6)


def test_an_integer_wav_is_clipped_because_it_has_to_be(tmp_path):
    signal = np.array([1.5, -1.5, 0.0])

    recovered, _ = AudioIO.read(
        AudioIO.write(tmp_path / "pcm.wav", signal, 8000, subtype="PCM_24")
    )

    assert np.max(np.abs(recovered)) <= 1.0


def test_stereo_is_averaged_to_mono():
    stereo = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])

    assert np.allclose(AudioIO.to_mono(stereo), [0.5, 0.5, 0.5])


def test_resampling_preserves_a_tone(tmp_path):
    t = np.arange(8000) / 8000.0
    signal = np.sin(2 * np.pi * 200 * t)
    path = AudioIO.write(tmp_path / "tone.wav", signal, 8000)

    resampled, rate = AudioIO.read(path, sr=16000)

    assert rate == 16000
    assert resampled.size == pytest.approx(2 * signal.size, rel=0.01)
    assert np.max(np.abs(resampled)) == pytest.approx(1.0, abs=0.02)


def test_a_missing_file_says_so(tmp_path):
    with pytest.raises(FileNotFoundError):
        AudioIO.read(tmp_path / "nope.wav")
