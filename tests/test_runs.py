"""The store: runs are kept, listed and read back without being overwritten."""

from __future__ import annotations

import json
from datetime import timedelta

import pytest

from drumsynth.fitting import SearchSpace, fit
from drumsynth.instrument import InstrumentSearchSpace, fit_instrument
from drumsynth.runs import Run, RunStore, store_hit_fit, store_instrument_fit

SMALL_HIT = SearchSpace(
    n_ffts=(512,), overlaps=(2,), components=(16, 32), ranks=(4,), phase_strides=(1,)
)
SMALL_DRUM = InstrumentSearchSpace(
    n_ffts=(512,), overlaps=(2,), components=(32,), field_ranks=(4,),
    pattern_ranks=(8,), donor_ranks=(8,),
)


@pytest.fixture
def store(tmp_path) -> RunStore:
    return RunStore(tmp_path / "runs")


@pytest.fixture
def drum_run(velocity_layers, store):
    result = fit_instrument(velocity_layers, target_mse=1e-2, space=SMALL_DRUM)
    return store_instrument_fit(result, velocity_layers, store=store, plot=False), result


def test_a_stored_run_holds_everything_the_fit_wrote(drum_run, store):
    run, result = drum_run

    assert run.path.parent.name == result.model.name
    assert (run.path / "instrument.npz").exists()
    assert (run.path / "report.json").exists()
    assert run.model_path.exists()
    assert run.n_bytes() > 0


def test_a_run_reads_back_its_model(drum_run):
    run, result = drum_run

    model = run.load_model()

    assert model.name == result.model.name
    assert model.candidate == result.candidate


def test_fitting_the_same_drum_twice_keeps_both(velocity_layers, store):
    first = store_instrument_fit(
        fit_instrument(velocity_layers, target_mse=1e-1, space=SMALL_DRUM),
        store=store,
        plot=False,
    )
    second = store_instrument_fit(
        fit_instrument(velocity_layers, target_mse=1e-3, space=SMALL_DRUM),
        store=store,
        plot=False,
    )

    assert first.path != second.path
    assert len(store.runs(first.name)) == 2
    assert store.latest(first.name).path == max(first.path, second.path)


def test_runs_come_back_newest_first(store, drum_run):
    run, _ = drum_run
    older = store.new_run("instrument", run.name)
    store.write_index(older, "instrument", run.name, "instrument.npz", {})
    (older / "run.json").write_text(
        json.dumps(
            {
                "kind": "instrument",
                "name": run.name,
                "created": (run.created - timedelta(days=1)).isoformat(),
                "model_file": "instrument.npz",
            }
        )
    )

    listed = store.runs(run.name)

    assert [r.created for r in listed] == sorted((r.created for r in listed), reverse=True)


def test_names_lists_what_has_been_fitted(store, drum_run):
    run, _ = drum_run

    assert store.names("instrument") == [run.name]
    assert store.names("hit") == []


def test_hits_and_drums_are_kept_apart(tonal_hit, sr, velocity_layers, store):
    store_instrument_fit(
        fit_instrument(velocity_layers, target_mse=1e-2, space=SMALL_DRUM),
        store=store,
        plot=False,
    )
    hit = store_hit_fit(
        fit(tonal_hit, sr, target_mse=1e-2, space=SMALL_HIT),
        "a-hit",
        reference=tonal_hit,
        store=store,
        plot=False,
    )

    assert store.names("hit") == ["a-hit"]
    assert hit.kind == "hit"
    assert store.runs("a-hit", kind="hit")[0].load_model().n_samples == tonal_hit.size


def test_an_index_carries_what_a_listing_needs(drum_run):
    run, result = drum_run

    assert run.metadata["representation"] == result.candidate.label()
    assert run.metadata["reconstruction_mse"] == result.reconstruction_mse
    assert run.metadata["n_layers"] == result.model.velocities.size
    assert run.summary in run.label


def test_a_run_that_fails_to_write_leaves_nothing_behind(velocity_layers, store, monkeypatch):
    import drumsynth.instrument.report as report

    monkeypatch.setattr(
        report, "save_instrument_fit", lambda *a, **k: (_ for _ in ()).throw(OSError("disk"))
    )
    result = fit_instrument(velocity_layers, target_mse=1e-2, space=SMALL_DRUM)

    with pytest.raises(OSError):
        store_instrument_fit(result, store=store, plot=False)

    assert store.runs(result.model.name) == []


def test_an_empty_store_lists_nothing(tmp_path):
    store = RunStore(tmp_path / "nowhere")

    assert store.names() == []
    assert store.runs() == []
    assert store.latest("anything") is None


def test_the_store_can_be_moved_by_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("DRUMSYNTH_RUNS", str(tmp_path / "elsewhere"))

    assert RunStore.default().root == tmp_path / "elsewhere"


def test_a_run_directory_can_be_opened_directly(drum_run):
    run, _ = drum_run

    assert Run.read(run.path).created == run.created
