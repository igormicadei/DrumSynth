"""The Streamlit pages, driven headlessly. They are thin, and stay thin."""

from __future__ import annotations

from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = st_testing.AppTest

TIMEOUT = 120
PAGES = Path(__file__).resolve().parent.parent / "app_pages"


def open_page(name: str) -> "AppTest":
    return AppTest.from_file(str(PAGES / name), default_timeout=TIMEOUT)


def run(name: str) -> "AppTest":
    app = open_page(name)
    app.run()
    return app


def test_the_fit_page_offers_the_shipped_samples():
    app = run("fit.py")

    assert not app.exception
    assert app.title[0].value == "Fit"
    assert app.sidebar.selectbox[0].options  # at least one sample is checked out


def test_fitting_a_sample_reports_a_model_and_a_reconstruction():
    app = run("fit.py")

    app.sidebar.select_slider[0].set_value(1e-2).run()
    app.sidebar.button[0].click().run()

    assert not app.exception
    assert "relative MSE" in app.code[0].value
    assert len(app.metric) == 2  # error and size
    assert app.dataframe  # the frontier
    assert app.download_button


def test_the_report_page_waits_for_a_model():
    app = run("report.py")

    assert not app.exception
    assert app.info


def test_the_report_page_opens_a_model_file(tonal_hit, sr, tmp_path):
    from drumsynth.spectral import Candidate, encode

    path = encode(tonal_hit, sr, Candidate(512, 256, 32, "lowrank", 8)).save(
        tmp_path / "model.npz"
    )

    app = open_page("report.py")
    app.session_state["run_path"] = str(path)
    app.run()

    assert not app.exception
    assert not app.error
    assert len(app.metric) >= 8  # what it is, and what it costs to play
    assert app.get("image")  # the figures


def test_the_report_page_opens_a_stored_run(velocity_layers, tmp_path, monkeypatch):
    from drumsynth.instrument import InstrumentSearchSpace, fit_instrument
    from drumsynth.runs import RunStore, store_instrument_fit

    space = InstrumentSearchSpace(
        n_ffts=(512,), overlaps=(2,), components=(32,), field_ranks=(4,),
        pattern_ranks=(8,), donor_ranks=(8,),
    )
    store = RunStore(tmp_path / "runs")
    kept = store_instrument_fit(
        fit_instrument(velocity_layers, target_mse=1e-2, space=space),
        velocity_layers,
        store=store,
        plot=False,
    )
    monkeypatch.setenv("DRUMSYNTH_RUNS", str(store.root))

    app = open_page("report.py")
    app.session_state["run_path"] = str(kept.path)
    app.run()

    assert not app.exception
    assert app.slider  # a velocity to play it at
    assert len(app.tabs) == 3


def test_the_instruments_page_lists_trainings(velocity_layers, tmp_path, monkeypatch):
    from drumsynth.instrument import InstrumentSearchSpace, fit_instrument
    from drumsynth.runs import RunStore, store_instrument_fit

    space = InstrumentSearchSpace(
        n_ffts=(512,), overlaps=(2,), components=(32,), field_ranks=(4,),
        pattern_ranks=(8,), donor_ranks=(8,),
    )
    store = RunStore(tmp_path / "runs")
    for target in (1e-1, 1e-3):
        store_instrument_fit(
            fit_instrument(velocity_layers, target_mse=target, space=space),
            store=store,
            plot=False,
        )
    monkeypatch.setenv("DRUMSYNTH_RUNS", str(store.root))

    app = open_page("instruments.py")
    app.run()

    assert not app.exception
    assert app.selectbox[0].value == "synthetic-drum"
    assert len(app.dataframe) == 2  # the catalogue, and this one's history
    assert "Open the full report" in [button.label for button in app.button]


def test_the_instrument_page_lists_drums_whose_audio_is_here():
    app = run("fit_drum.py")

    assert not app.exception
    assert app.title[0].value == "Instrument"
    assert any("toms-stereo-tom3" in option for option in app.sidebar.selectbox[0].options)


def test_fitting_a_drum_from_the_page_reports_and_plays_it():
    app = run("fit_drum.py")

    app.sidebar.select_slider[0].set_value(1e-2).run()
    app.sidebar.button[0].click().run()

    assert not app.exception
    assert len(app.metric) == 3  # reconstruction, size, interpolation
    assert "velocities" in app.code[0].value
    assert app.dataframe  # the per-velocity table
