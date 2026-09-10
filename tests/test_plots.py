"""Figures: every one of them draws, on both kinds of model, without warnings."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from drumsynth import plots
from drumsynth.instrument import InstrumentCandidate, InstrumentModel
from drumsynth.spectral import Candidate, encode
from drumsynth.spectral.stft import StftSpec

SPEC = StftSpec(512, 256)


@pytest.fixture(scope="module")
def hit(tonal_hit, sr):
    model = encode(tonal_hit, sr, Candidate(512, 256, 32, "lowrank", 8))
    return model, tonal_hit, model.render()


@pytest.fixture(scope="module")
def drum(velocity_layers):
    bins = velocity_layers.select_bins(SPEC, 32)
    magnitudes, blocks = velocity_layers.analyse(SPEC, bins)
    return InstrumentModel.build(
        "drum",
        InstrumentCandidate(512, 256, 32, "velocity", 3, 0, "lowrank", 8, 0),
        sample_rate=velocity_layers.sample_rate,
        n_samples=velocity_layers.n_samples,
        velocities=velocity_layers.velocities,
        bins=bins,
        magnitudes=magnitudes,
        blocks=blocks,
    )


def drawn(figure) -> bytes:
    """Render it, the way anything looking at it would. Warnings are failures."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return plots.to_png(figure, dpi=60)


def test_every_comparison_figure_draws(hit):
    model, reference, estimate = hit
    rate = model.sample_rate

    for figure in (
        plots.waveform(reference, estimate, rate),
        plots.spectrogram_panel(reference, estimate, rate),
        plots.spectrum(reference, estimate, rate),
        plots.error_over_time(reference, estimate, rate),
        plots.error_by_frequency(reference, estimate, rate),
    ):
        assert drawn(figure)


def test_every_anatomy_figure_draws(hit):
    model, reference, _ = hit
    components = model.components()
    frequencies = model.frequencies()
    times = np.arange(model.n_frames) * model.candidate.hop / model.sample_rate

    for figure in (
        plots.bin_map(model, reference, components),
        plots.envelope_heatmap(components, frequencies, times),
        plots.envelope_traces(components, frequencies, times),
        plots.decay_times(components, frequencies, times),
        plots.envelope_spectrum(components, model.sample_rate / model.candidate.hop),
    ):
        assert drawn(figure)


def test_every_velocity_figure_draws(drum):
    for figure in (
        plots.velocity_curves(drum),
        plots.velocity_response(drum),
        plots.velocity_spectrum(drum),
    ):
        assert drawn(figure)


def test_a_dc_bin_does_not_break_the_log_axes(hit):
    """The model keeps DC when it carries energy; a log frequency axis cannot."""
    model, _, _ = hit
    components = model.components()
    frequencies = model.frequencies().copy()
    frequencies[0] = 0.0
    times = np.arange(model.n_frames) * model.candidate.hop / model.sample_rate

    assert drawn(plots.decay_times(components, frequencies, times))


def test_a_whole_set_lands_in_a_directory(hit, tmp_path):
    model, reference, _ = hit

    written = plots.save_hit_figures(model, reference, tmp_path)

    assert len(written) == 10
    for path in written.values():
        assert path.exists() and path.stat().st_size > 0


def test_an_instrument_set_covers_the_velocity_axis(drum, velocity_layers, tmp_path):
    written = plots.save_instrument_figures(drum, velocity_layers, 0, tmp_path)

    assert "velocity_curves" in written
    assert "waveform" in written  # layers were given, so it could compare
    assert all(path.exists() for path in written.values())


def test_an_instrument_set_without_recordings_skips_the_comparisons(drum, tmp_path):
    written = plots.save_instrument_figures(drum, None, 0, tmp_path)

    assert "velocity_spectrum" in written
    assert "waveform" not in written


def test_db_floors_rather_than_reaching_minus_infinity():
    values = plots.db(np.array([1.0, 0.0, 1e-12]))

    assert values[0] == pytest.approx(0.0)
    assert values[1] == plots.FLOOR_DB
    assert np.all(np.isfinite(values))
