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


def test_the_model_page_says_so_when_there_is_no_file():
    app = run("model.py")

    assert not app.exception
    assert app.error


def test_the_model_page_opens_a_saved_model(tonal_hit, sr, tmp_path):
    from drumsynth.spectral import Candidate, encode

    path = encode(tonal_hit, sr, Candidate(512, 256, 32, "lowrank", 8)).save(
        tmp_path / "model.npz"
    )

    app = open_page("model.py")
    app.run()
    app.text_input[0].set_value(str(path)).run()

    assert not app.exception
    assert not app.error
    assert len(app.metric) == 3  # representation, size, audio
    assert len(app.dataframe) == 2  # the arrays, and the kept bins
