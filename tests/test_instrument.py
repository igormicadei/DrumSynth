"""One drum across every velocity: the layers, the field, the donors, the fit."""

from __future__ import annotations

import numpy as np
import pytest

from drumsynth.instrument import (
    DONOR_CODECS,
    FIELD_CODECS,
    InstrumentCandidate,
    InstrumentModel,
    InstrumentSearchSpace,
    MagnitudeField,
    VelocityLayers,
    fit_instrument,
    interpolation_error,
    magnitude_error,
)
from drumsynth.instrument import donors as donor_codecs
from drumsynth.instrument import field as field_codecs
from drumsynth.instrument.layers import PRE_ROLL, align, onset
from drumsynth.spectral.stft import StftSpec

SPEC = StftSpec(512, 256)

SMALL = InstrumentSearchSpace(
    n_ffts=(512,),
    overlaps=(2,),
    components=(32, 64),
    field_ranks=(2, 4),
    pattern_ranks=(8,),
    donor_ranks=(8,),
)


# -- layers -------------------------------------------------------------------


def test_recordings_are_aligned_on_their_onsets(velocity_layers):
    onsets = {onset(take) for group in velocity_layers.hits for take in group}

    assert onsets == {int(round(PRE_ROLL * velocity_layers.sample_rate))}


@pytest.mark.parametrize(
    "lead_in", [0, 3, 500], ids=["starts-immediately", "inside-the-pre-roll", "late"]
)
def test_alignment_moves_the_onset_to_one_place(lead_in):
    signal = np.concatenate([np.zeros(lead_in), np.ones(100)])

    aligned = align(signal, 1000, 300)

    assert aligned.size == 300
    assert onset(aligned) == int(round(PRE_ROLL * 1000))


def test_layers_are_grouped_by_velocity_and_ascending(velocity_layers):
    assert velocity_layers.n_layers == 6
    assert velocity_layers.n_recordings == 12
    assert list(velocity_layers.velocities) == sorted(velocity_layers.velocities)
    assert all(velocity_layers.takes(index) == 2 for index in range(6))


def test_every_recording_has_the_same_length(velocity_layers):
    lengths = {take.size for group in velocity_layers.hits for take in group}

    assert lengths == {velocity_layers.n_samples}


def test_bin_selection_is_shared_by_the_whole_instrument(velocity_layers):
    bins = velocity_layers.select_bins(SPEC, 24)

    assert bins.size == 24
    assert list(bins) == sorted(set(bins))
    assert bins.max() < SPEC.n_bins


def test_an_instrument_with_no_audio_on_disk_says_so():
    from drumsynth.corpus import Corpus

    corpus = Corpus.load()
    absent = next(
        name for name in corpus.names if not corpus.samples(name, present_only=True)
    )

    with pytest.raises(FileNotFoundError, match="none of its"):
        VelocityLayers.from_corpus(absent, corpus)


# -- the magnitude field ------------------------------------------------------


@pytest.fixture(scope="module")
def tensor(velocity_layers):
    bins = velocity_layers.select_bins(SPEC, 32)
    magnitudes, _ = velocity_layers.analyse(SPEC, bins)
    return velocity_layers.velocities, magnitudes


def field_error(velocities, magnitudes, codec, rank, pattern_rank):
    implementation = field_codecs.codec_for(codec)
    arrays = implementation.encode(
        implementation.prepare(magnitudes), rank, pattern_rank
    )
    field = MagnitudeField.from_arrays(velocities, codec, arrays)
    return magnitude_error(magnitudes, field.layers())


def test_a_full_field_keeps_every_layer(tensor):
    assert field_error(*tensor, "full", 0, 0) < 1e-12


@pytest.mark.parametrize("codec", ["velocity", "separable"])
def test_more_rank_never_measures_worse(codec, tensor):
    errors = [field_error(*tensor, codec, rank, 16) for rank in (1, 2, 3, 4)]

    assert errors == sorted(errors, reverse=True)


def test_a_field_reproduces_the_velocity_it_was_given(tensor):
    velocities, magnitudes = tensor
    implementation = field_codecs.codec_for("full")
    field = MagnitudeField.from_arrays(
        velocities, "full", implementation.encode(magnitudes, 0, 0)
    )

    assert np.allclose(field.at(velocities[2]), magnitudes[2], atol=1e-5)


def test_between_two_layers_a_field_sits_between_them(tensor):
    velocities, magnitudes = tensor
    implementation = field_codecs.codec_for("full")
    field = MagnitudeField.from_arrays(
        velocities, "full", implementation.encode(magnitudes, 0, 0)
    )

    middle = field.at(0.5 * (velocities[0] + velocities[1]))
    expected = 0.5 * (magnitudes[0] + magnitudes[1])

    assert np.allclose(middle, expected, atol=1e-5)


def test_a_field_clamps_outside_the_recorded_range(tensor):
    velocities, magnitudes = tensor
    implementation = field_codecs.codec_for("velocity")
    field = MagnitudeField.from_arrays(
        velocities, "velocity", implementation.encode(implementation.prepare(magnitudes), 3, 0)
    )

    assert np.allclose(field.at(velocities[-1] + 50), field.at(velocities[-1]))
    assert np.allclose(field.at(0.0), field.at(velocities[0]))


@pytest.mark.parametrize("codec", sorted(FIELD_CODECS))
def test_field_scalar_counts_match_the_arrays(codec, tensor):
    velocities, magnitudes = tensor
    implementation = field_codecs.codec_for(codec)
    rank, pattern_rank = implementation.clamp(magnitudes.shape, 3, 8)

    arrays = implementation.encode(implementation.prepare(magnitudes), rank, pattern_rank)
    stored = sum(array.size for array in arrays.values())

    assert implementation.n_scalars(magnitudes.shape, rank, pattern_rank) == stored


def test_unknown_field_codec_names_itself():
    with pytest.raises(ValueError, match="unknown field codec"):
        field_codecs.codec_for("nope")


# -- donors -------------------------------------------------------------------


@pytest.fixture(scope="module")
def block(velocity_layers):
    bins = velocity_layers.select_bins(SPEC, 32)
    _, blocks = velocity_layers.analyse(SPEC, bins)
    return blocks[3]


def test_an_exact_donor_keeps_the_phase_it_was_given(block):
    codec = donor_codecs.ExactDonor

    decoded = codec.decode(codec.encode(codec.prepare(block), 0))

    assert np.allclose(np.exp(1j * decoded), np.exp(1j * np.angle(block)), atol=1e-5)


@pytest.mark.parametrize("codec", sorted(DONOR_CODECS))
def test_donor_scalar_counts_match_the_arrays(codec, block):
    implementation = donor_codecs.codec_for(codec)
    bins, frames = block.shape

    arrays = implementation.encode(implementation.prepare(block), 8)
    stored = sum(a.size * (2 if np.iscomplexobj(a) else 1) for a in arrays.values())

    assert implementation.n_scalars(bins, frames, 8) == stored


def test_donor_choice_spreads_over_the_range():
    velocities = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])

    assert list(donor_codecs.choose(velocities, 0)) == [0, 1, 2, 3, 4, 5]
    assert list(donor_codecs.choose(velocities, 99)) == [0, 1, 2, 3, 4, 5]
    assert donor_codecs.choose(velocities, 3)[0] == 0
    assert donor_codecs.choose(velocities, 3)[-1] == 5


def test_a_rendered_velocity_borrows_from_the_nearest_donor():
    donors = np.array([10.0, 60.0, 120.0])

    assert donor_codecs.nearest(donors, 12.0) == 0
    assert donor_codecs.nearest(donors, 59.0) == 1
    assert donor_codecs.nearest(donors, 400.0) == 2


# -- the model ----------------------------------------------------------------


@pytest.fixture(scope="module")
def model(velocity_layers):
    bins = velocity_layers.select_bins(SPEC, 32)
    magnitudes, blocks = velocity_layers.analyse(SPEC, bins)
    return InstrumentModel.build(
        velocity_layers.name,
        InstrumentCandidate(SPEC.n_fft, SPEC.hop, 32, "velocity", 3, 0, "lowrank", 8, 0),
        sample_rate=velocity_layers.sample_rate,
        n_samples=velocity_layers.n_samples,
        velocities=velocity_layers.velocities,
        bins=bins,
        magnitudes=magnitudes,
        blocks=blocks,
    )


def test_a_model_renders_at_any_velocity_in_range(model, velocity_layers):
    low, high = model.velocity_range

    for velocity in (low, 0.5 * (low + high), high):
        rendered = model.render(velocity)
        assert rendered.shape == (velocity_layers.n_samples,)
        assert np.all(np.isfinite(rendered))


def test_louder_velocities_render_louder(model):
    low, high = model.velocity_range

    quiet = np.sqrt(np.mean(model.render(low) ** 2))
    loud = np.sqrt(np.mean(model.render(high) ** 2))

    assert loud > 4 * quiet


def test_predicted_size_matches_the_stored_arrays(model):
    predicted = model.candidate.n_scalars(
        model.velocities.size, model.n_frames, len(model.donor_arrays)
    )

    assert predicted == model.n_scalars


def test_a_saved_instrument_renders_identically(model, tmp_path):
    reloaded = InstrumentModel.load(model.save(tmp_path / "instrument.npz"))

    assert reloaded.candidate == model.candidate
    assert np.array_equal(reloaded.velocities, model.velocities)
    for velocity in (12.0, 47.5, 110.0):
        assert np.array_equal(reloaded.render(velocity), model.render(velocity))


def test_the_instrument_file_is_self_contained(model, tmp_path):
    path = model.save(tmp_path / "instrument.npz")

    assert list(tmp_path.iterdir()) == [path]
    assert InstrumentModel.load(path).name == model.name


def test_loading_refuses_a_single_hit_model(tonal_hit, sr, tmp_path):
    from drumsynth.spectral import Candidate, encode

    path = encode(tonal_hit, sr, Candidate(512, 256, 16, "lowrank", 4)).save(
        tmp_path / "hit.npz"
    )

    with pytest.raises(ValueError, match="expected format"):
        InstrumentModel.load(path)


# -- the fit ------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted(velocity_layers):
    return fit_instrument(velocity_layers, target_mse=1e-2, space=SMALL)


def test_the_fit_meets_the_target_it_was_given(fitted):
    assert fitted.target_reached
    assert fitted.reconstruction_mse <= 1e-2


def test_every_recorded_velocity_is_measured(fitted, velocity_layers):
    assert len(fitted.layers) == velocity_layers.n_layers
    assert [layer.velocity for layer in fitted.layers] == list(velocity_layers.velocities)
    assert all(layer.is_donor for layer in fitted.layers)


def test_nothing_smaller_in_the_search_also_met_the_target(fitted):
    smaller_and_good = [
        e
        for e in fitted.evaluations
        if e.n_scalars < fitted.model.n_scalars and e.relative_mse <= fitted.target_mse
    ]

    assert smaller_and_good == []


def test_generalization_is_measured_against_takes_the_model_never_saw(fitted):
    assert fitted.generalization_mse is not None
    assert all(layer.generalization_mse is not None for layer in fitted.layers)
    assert fitted.generalization_mse > fitted.reconstruction_mse


def test_interpolation_is_measured_at_held_out_velocities(velocity_layers, fitted):
    error = interpolation_error(velocity_layers, fitted.candidate)

    assert error is not None
    assert 0.0 < error < 1.0


def test_a_velocity_with_no_donor_of_its_own_is_a_different_signal(velocity_layers):
    """Borrowed phase does not blend, and by this objective it does not degrade.

    A velocity whose phase comes from another strike is uncorrelated with the
    recording at that velocity, which is a relative error near two — worse than
    rendering silence. That is what the waveform objective says about it; what
    it sounds like is a question the objective cannot answer.
    """
    every = fit_instrument(velocity_layers, target_mse=1e-2, space=SMALL, n_donors=0)
    sparse = fit_instrument(velocity_layers, target_mse=1e-2, space=SMALL, n_donors=2)

    borrowed = [layer for layer in sparse.layers if not layer.is_donor]
    donated = [layer for layer in sparse.layers if layer.is_donor]

    assert borrowed and donated
    borrowed_error = np.mean([layer.reconstruction_mse for layer in borrowed])
    donated_error = np.mean([layer.reconstruction_mse for layer in donated])

    assert borrowed_error > 1.0
    assert donated_error < borrowed_error / 5
    assert sparse.reconstruction_mse > every.reconstruction_mse


def test_a_tighter_target_costs_more(velocity_layers):
    loose = fit_instrument(velocity_layers, target_mse=1e-1, space=SMALL)
    tight = fit_instrument(velocity_layers, target_mse=1e-3, space=SMALL)

    assert loose.model.n_scalars < tight.model.n_scalars


def test_an_unreachable_target_is_reported_not_hidden(velocity_layers):
    result = fit_instrument(velocity_layers, target_mse=1e-30, space=SMALL)

    assert not result.target_reached
    assert result.reconstruction_mse > 1e-30


def test_a_second_take_can_be_fitted_instead(velocity_layers):
    first = fit_instrument(velocity_layers, target_mse=1e-2, space=SMALL, take=0)
    second = fit_instrument(velocity_layers, target_mse=1e-2, space=SMALL, take=1)

    assert second.target_reached
    assert not np.array_equal(second.model.render(50.0), first.model.render(50.0))


def test_the_search_space_never_offers_a_donor_count_it_was_not_given(velocity_layers):
    for candidate in SMALL.candidates(velocity_layers, n_donors=3):
        assert candidate.n_donors == 3


# -- what a run leaves behind -------------------------------------------------


def test_a_run_writes_the_model_the_report_and_three_sweeps(velocity_layers, tmp_path):
    from drumsynth import AudioIO
    from drumsynth.instrument import between, save_instrument_fit, sweep

    result = fit_instrument(velocity_layers, target_mse=1e-2, space=SMALL)
    paths = save_instrument_fit(result, tmp_path, layers=velocity_layers, plot=False)

    assert set(paths) >= {
        "model",
        "metadata",
        "report",
        "layers",
        "model_sweep",
        "between_sweep",
        "reference_sweep",
    }
    for path in paths.values():
        assert path.exists() and path.stat().st_size > 0

    rendered, rate = AudioIO.read(paths["model_sweep"])
    assert rate == velocity_layers.sample_rate
    assert rendered.size == sweep(result.model, result.model.velocities).size

    played, _ = AudioIO.read(paths["between_sweep"])
    assert played.size < rendered.size  # one fewer hit, between the recorded ones
    assert between(result.model.velocities).size == result.model.velocities.size - 1


def test_the_report_records_all_three_measurements(velocity_layers, tmp_path):
    import json

    from drumsynth.instrument import save_instrument_fit

    result = fit_instrument(velocity_layers, target_mse=1e-2, space=SMALL)
    paths = save_instrument_fit(result, tmp_path, layers=velocity_layers, plot=False)

    report = json.loads(paths["report"].read_text())

    assert report["reconstruction_mse"] == result.reconstruction_mse
    assert report["generalization_mse"] == result.generalization_mse
    assert report["interpolation_mse"] == result.interpolation_mse
    assert len(report["layers"]) == velocity_layers.n_layers
    assert report["velocities"] == list(velocity_layers.velocities)


# -- one take, or the average of them -----------------------------------------


def test_averaging_takes_predicts_the_next_strike_better(velocity_layers):
    """The trade the `average` flag makes, measured on both sides of it."""
    one = fit_instrument(velocity_layers, target_mse=1e-2, space=SMALL)
    averaged = fit_instrument(velocity_layers, target_mse=1e-2, space=SMALL, average=True)

    assert averaged.generalization_mse < one.generalization_mse
    assert one.reconstruction_mse < averaged.reconstruction_mse
    assert averaged.averaged and not one.averaged


def test_averaging_uses_every_recording(velocity_layers):
    bins = velocity_layers.select_bins(SPEC, 32)
    first, _ = velocity_layers.analyse(SPEC, bins, take=0)
    second, _ = velocity_layers.analyse(SPEC, bins, take=1)
    averaged, blocks = velocity_layers.analyse(SPEC, bins, take=0, average=True)

    assert np.allclose(averaged, 0.5 * (first + second))
    assert np.allclose(np.abs(blocks), first)  # phase still comes from one strike
