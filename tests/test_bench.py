"""The live-cost measurements: shaped right, and consistent with each other."""

from __future__ import annotations

import pytest

from drumsynth.bench import measure_live, measure_polyphony, resident_bytes
from drumsynth.spectral import Candidate, encode


@pytest.fixture(scope="module")
def model(tonal_hit, sr):
    return encode(tonal_hit, sr, Candidate(1024, 512, 64, "lowrank", 8))


def test_a_cost_is_measured_end_to_end(model):
    cost = measure_live(model, block=256, repeats=2)

    assert cost.trigger_ms > 0
    assert cost.block_mean_ms > 0
    assert cost.block_max_ms >= cost.block_p95_ms >= cost.block_mean_ms
    assert cost.render_ms > 0


def test_the_budget_is_the_block_in_milliseconds(model):
    cost = measure_live(model, block=256, repeats=1)

    assert cost.budget_ms == pytest.approx(1000 * 256 / model.sample_rate)
    assert cost.load == pytest.approx(cost.block_mean_ms / cost.budget_ms)
    assert cost.voices == pytest.approx(cost.budget_ms / cost.block_max_ms)


def test_rendering_a_hit_beats_real_time(model):
    """If it did not, nothing else in the report would matter."""
    cost = measure_live(model, block=256, repeats=2)

    assert cost.realtime_factor > 1.0


def test_streaming_a_whole_hit_costs_about_what_rendering_it_does(model):
    """Same arithmetic, spread out — within the noise of a small measurement."""
    cost = measure_live(model, block=256, repeats=3)
    blocks = model.n_samples / 256

    assert cost.block_mean_ms * blocks < 6 * cost.render_ms


def test_memory_is_counted_from_the_arrays_that_exist(model):
    resident = resident_bytes(model)

    assert resident >= sum(a.nbytes for a in model.arrays.values())
    assert resident < 50 * model.n_bytes()


def test_polyphony_costs_more_than_one_voice(model):
    one = measure_polyphony(model, block=256, voices=1)
    many = measure_polyphony(model, block=256, voices=8)

    assert many["mean_ms"] > one["mean_ms"]
    assert many["load"] == pytest.approx(many["mean_ms"] / many["budget_ms"])


def test_the_report_says_what_was_measured(model):
    printed = measure_live(model, block=128, repeats=1).report()

    assert "trigger" in printed
    assert "per block" in printed
    assert "voices" in printed
