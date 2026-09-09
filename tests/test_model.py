"""A model is what a file will contain: encode, count, save, load, render."""

from __future__ import annotations

import numpy as np
import pytest

from drumsynth.fitting.metrics import relative_mse
from drumsynth.spectral import Candidate, SpectralModel, encode

CANDIDATES = [
    Candidate(512, 128, 32, "raw"),
    Candidate(512, 128, 32, "raw", phase_stride=4),
    Candidate(512, 128, 32, "shared", 8),
    Candidate(512, 128, 32, "shared", 4, phase_stride=2),
    Candidate(512, 128, 32, "lowrank", 8),
]


@pytest.mark.parametrize("candidate", CANDIDATES, ids=lambda c: c.label())
def test_encode_render_is_the_right_length_and_rate(candidate, tonal_hit, sr):
    model = encode(tonal_hit, sr, candidate)

    audio = model.render()

    assert audio.shape == tonal_hit.shape
    assert model.sample_rate == sr
    assert model.duration == pytest.approx(len(tonal_hit) / sr)


@pytest.mark.parametrize("candidate", CANDIDATES, ids=lambda c: c.label())
def test_predicted_size_matches_the_stored_arrays(candidate, tonal_hit, sr):
    model = encode(tonal_hit, sr, candidate)

    assert candidate.n_scalars(model.n_frames) == model.n_scalars


@pytest.mark.parametrize("candidate", CANDIDATES, ids=lambda c: c.label())
def test_a_saved_model_renders_identically(candidate, tonal_hit, sr, tmp_path):
    model = encode(tonal_hit, sr, candidate)

    reloaded = SpectralModel.load(model.save(tmp_path / "model.npz"))

    assert reloaded.candidate == candidate
    assert np.array_equal(reloaded.render(), model.render())


def test_the_file_is_self_contained(tonal_hit, sr, tmp_path):
    """One .npz — nothing else is needed to play a model back."""
    path = encode(tonal_hit, sr, CANDIDATES[-1]).save(tmp_path / "model.npz")

    assert SpectralModel.load(path).n_samples == len(tonal_hit)
    assert list(tmp_path.iterdir()) == [path]


def test_loading_refuses_an_unknown_format(tonal_hit, sr, tmp_path):
    path = tmp_path / "wrong.npz"
    np.savez_compressed(path, meta=np.asarray('{"format": "something/else"}'))

    with pytest.raises(ValueError, match="expected format"):
        SpectralModel.load(path)


def test_keeping_every_bin_reproduces_the_signal(tonal_hit, sr):
    """With nothing discarded the model is the transform, which is exact."""
    candidate = Candidate(512, 128, 257, "raw")

    model = encode(tonal_hit, sr, candidate)

    assert relative_mse(tonal_hit, model.render()) < 1e-12


def test_more_bins_never_measures_worse(tonal_hit, sr):
    errors = [
        relative_mse(tonal_hit, encode(tonal_hit, sr, Candidate(512, 128, k, "raw")).render())
        for k in (8, 16, 32, 64, 128)
    ]

    assert errors == sorted(errors, reverse=True)


def test_a_codec_that_carries_its_own_phase_rejects_a_stride():
    with pytest.raises(ValueError, match="stores its own phase"):
        Candidate(512, 128, 32, "lowrank", 8, phase_stride=2)


def test_label_names_the_settings_that_apply():
    assert "rank=8" in Candidate(512, 128, 32, "lowrank", 8).label()
    assert "stride" not in Candidate(512, 128, 32, "lowrank", 8).label()
    assert "stride=4" in Candidate(512, 128, 32, "raw", phase_stride=4).label()


def test_frequencies_are_the_kept_bin_centres(tonal_hit, sr):
    model = encode(tonal_hit, sr, Candidate(512, 128, 32, "raw"))

    frequencies = model.frequencies()

    assert frequencies.size == model.bins.size
    assert np.all(np.diff(frequencies) > 0)
    assert frequencies.max() <= sr / 2
