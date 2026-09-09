"""The whole path on real audio: a recorded tom in, a model file out, audio back."""

from __future__ import annotations

import numpy as np

from drumsynth import AudioIO, SpectralModel, save_fit
from drumsynth.fitting import SearchSpace, fit
from drumsynth.fitting.metrics import relative_mse

SPACE = SearchSpace(
    n_ffts=(1024, 2048),
    overlaps=(2,),
    components=(64, 128, 256),
    ranks=(8, 16, 32),
    phase_strides=(1, 2),
)


def test_a_recorded_hit_fits_compresses_and_plays_back(recorded_hit, tmp_path):
    signal, sr = recorded_hit

    result = fit(signal, sr, target_mse=1e-4, space=SPACE)
    paths = save_fit(result, tmp_path, reference=signal, plot=False)
    played = SpectralModel.load(paths["model"]).render()

    assert result.target_reached
    assert relative_mse(signal, played) <= 1e-4
    assert result.model.compression_ratio() < 1.0  # smaller than the 16-bit PCM
    assert np.array_equal(played, result.model.render())


def test_the_written_reconstruction_is_the_one_that_was_measured(recorded_hit, tmp_path):
    signal, sr = recorded_hit

    result = fit(signal, sr, target_mse=1e-3, space=SPACE)
    paths = save_fit(result, tmp_path, reference=signal, plot=False)
    written, rate = AudioIO.read(paths["reconstruction"])

    assert rate == sr
    assert relative_mse(signal, written) <= 1e-3


def test_a_looser_target_buys_a_smaller_model(recorded_hit):
    signal, sr = recorded_hit

    tight = fit(signal, sr, target_mse=1e-4, space=SPACE)
    loose = fit(signal, sr, target_mse=1e-2, space=SPACE)

    assert loose.model.n_bytes() < tight.model.n_bytes()
    assert loose.target_reached and tight.target_reached
