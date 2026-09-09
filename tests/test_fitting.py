"""The search: does it find the smallest model that actually meets the target."""

from __future__ import annotations

import numpy as np
import pytest

from drumsynth.fitting import (
    Evaluation,
    Quality,
    SearchSpace,
    choose_best,
    fit,
    pareto_frontier,
)
from drumsynth.fitting.metrics import relative_mse, snr_db
from drumsynth.spectral import Candidate, encode

SMALL = SearchSpace(
    n_ffts=(256, 512),
    overlaps=(2,),
    components=(16, 32, 64),
    ranks=(2, 8, 16),
    phase_strides=(1, 2),
)


@pytest.fixture(scope="module")
def tonal_fit(tonal_hit, sr):
    return fit(tonal_hit, sr, target_mse=1e-4, space=SMALL)


def test_the_fit_meets_the_target_it_was_given(tonal_fit):
    assert tonal_fit.target_reached
    assert tonal_fit.quality.relative_mse <= 1e-4


def test_the_reported_quality_is_the_saved_model_s_quality(tonal_fit, tonal_hit):
    """What the search reports has to be what the file plays back, not an estimate."""
    rendered = tonal_fit.model.render()

    assert relative_mse(tonal_hit, rendered) == pytest.approx(
        tonal_fit.quality.relative_mse, rel=1e-12
    )


def test_the_search_measures_what_it_would_store(tonal_fit, tonal_hit, sr):
    """Re-encoding the winning candidate reproduces the winning evaluation exactly."""
    winner = min(
        (e for e in tonal_fit.evaluations if e.candidate == tonal_fit.candidate),
        key=lambda e: e.quality.relative_mse,
    )
    model = encode(tonal_hit, sr, tonal_fit.candidate)

    assert relative_mse(tonal_hit, model.render()) == pytest.approx(
        winner.quality.relative_mse, rel=1e-12
    )


def test_nothing_smaller_in_the_search_also_met_the_target(tonal_fit):
    """The point of the whole thing: smallest first, accuracy only as a tie-break."""
    chosen = tonal_fit.model.n_scalars

    smaller_and_good = [
        e
        for e in tonal_fit.evaluations
        if e.n_scalars < chosen and e.quality.relative_mse <= tonal_fit.target_mse
    ]

    assert smaller_and_good == []


def test_a_tighter_target_costs_more(tonal_hit, sr):
    loose = fit(tonal_hit, sr, target_mse=1e-2, space=SMALL)
    tight = fit(tonal_hit, sr, target_mse=1e-5, space=SMALL)

    assert loose.model.n_scalars < tight.model.n_scalars
    assert loose.quality.relative_mse > tight.quality.relative_mse


def test_an_unreachable_target_is_reported_not_hidden(tonal_hit, sr):
    result = fit(tonal_hit, sr, target_mse=1e-30, space=SMALL)

    assert not result.target_reached
    assert result.quality.relative_mse == min(
        e.quality.relative_mse for e in result.evaluations
    )


def test_the_frontier_is_ordered_and_non_dominated(tonal_fit):
    frontier = tonal_fit.frontier

    sizes = [e.n_scalars for e in frontier]
    errors = [e.quality.relative_mse for e in frontier]

    assert sizes == sorted(sizes)
    assert errors == sorted(errors, reverse=True)
    for evaluation in tonal_fit.evaluations:
        assert any(
            f.n_scalars <= evaluation.n_scalars
            and f.quality.relative_mse <= evaluation.quality.relative_mse
            for f in frontier
        )


def test_progress_is_reported_once_per_group_and_ends_complete(tonal_hit, sr):
    seen = []

    result = fit(tonal_hit, sr, space=SMALL, progress=seen.append)

    assert seen
    assert seen[-1].done == len(result.evaluations)
    assert all(p.total == seen[-1].total for p in seen)
    assert [p.done for p in seen] == sorted(p.done for p in seen)


def test_parallel_and_serial_searches_agree(tonal_hit, sr):
    serial = fit(tonal_hit, sr, space=SMALL, jobs=1)
    parallel = fit(tonal_hit, sr, space=SMALL, jobs=2)

    assert parallel.candidate == serial.candidate
    assert len(parallel.evaluations) == len(serial.evaluations)
    assert parallel.quality.relative_mse == pytest.approx(serial.quality.relative_mse)


def test_the_space_never_offers_a_model_bigger_than_the_waveform(sr):
    space = SearchSpace()
    n_samples = int(0.2 * sr)

    for candidate in space.candidates(n_samples):
        assert candidate.n_scalars(candidate.spec.n_frames(n_samples)) <= n_samples


def test_the_space_only_strides_phase_where_that_means_something():
    for candidate in SearchSpace().candidates(4096):
        if not candidate.codec_class.uses_phase:
            assert candidate.phase_stride == 1


def test_max_candidates_limits_the_work(tonal_hit, sr):
    result = fit(tonal_hit, sr, space=SMALL, max_candidates=5)

    assert len(result.evaluations) == 5


def test_an_empty_signal_is_refused(sr):
    with pytest.raises(ValueError, match="empty signal"):
        fit(np.zeros(0), sr)


# -- the selection policy, on synthetic evaluations ---------------------------


def evaluation(n_scalars: int, error: float) -> Evaluation:
    return Evaluation(
        candidate=Candidate(512, 256, 8),
        quality=Quality(error, snr_db(error), 1.0, 1.0, 1.0, 1.0, 1.0),
        n_scalars=n_scalars,
        n_frames=10,
    )


def test_the_smallest_model_meeting_the_target_wins():
    candidates = [evaluation(100, 1e-3), evaluation(50, 9e-6), evaluation(500, 1e-9)]

    assert choose_best(candidates, 1e-5) is candidates[1]


def test_accuracy_decides_when_nothing_meets_the_target():
    candidates = [evaluation(10, 1e-2), evaluation(1000, 1e-4)]

    assert choose_best(candidates, 1e-9) is candidates[1]


def test_a_dominated_candidate_is_dropped_from_the_frontier():
    good, dominated = evaluation(100, 1e-6), evaluation(200, 1e-5)

    assert pareto_frontier([good, dominated]) == [good]


def test_the_frontier_of_nothing_is_nothing():
    assert pareto_frontier([]) == []
    with pytest.raises(ValueError, match="no candidates"):
        choose_best([], 1e-5)
