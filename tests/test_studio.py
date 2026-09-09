"""The Streamlit app, driven headlessly with AppTest.

No browser and no server: `AppTest` runs the script in-process, so these are
ordinary fast tests. They cover the parts that are easy to break by accident —
unit conversion in the strips, and the rule that solo and mute are monitoring
only and must never touch a parameter.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

#: Absolute, because AppTest resolves a relative path against the caller's
#: directory and pytest runs from wherever it was invoked.
APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")
TIMEOUT = 120


@pytest.fixture
def app():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, at.exception[0].message if at.exception else ""
    return at


class TestItRuns:
    def test_no_exception_and_no_warnings(self, app):
        assert not app.exception
        assert len(app.warning) == 0

    def test_the_three_parameter_groups_are_all_present(self, app):
        """One set per mode, one per transient, one overall — the whole point."""
        labels = " ".join(tab.label for tab in app.tabs if tab.label)
        assert "Master" in labels
        assert "Modes" in labels
        assert "Transients" in labels

    def test_it_starts_from_the_tom_preset(self, app):
        assert len(app.session_state.params.modes) == 31
        assert app.session_state.params.fundamental() == pytest.approx(88.07, abs=0.1)

    def test_the_engine_is_not_started_automatically(self, app):
        """Spawning an audio process on page load would be rude, and would
        respawn on every browser reconnect."""
        assert app.session_state.pushed is None


class TestModeStrips:
    def test_frequency_edits_reach_the_parameters(self, app):
        app.number_input(key="mode_0_f").set_value(101.5).run()
        assert app.session_state.params.modes[0].f_static == 101.5

    def test_gain_is_edited_in_db(self, app):
        app.number_input(key="mode_0_gain").set_value(-6.0).run()
        assert app.session_state.params.modes[0].gain == pytest.approx(
            10 ** (-6 / 20), rel=1e-6
        )

    def test_t60_edits_reach_the_parameters(self, app):
        app.number_input(key="mode_1_t60").set_value(0.75).run()
        assert app.session_state.params.modes[1].t60 == 0.75

    def test_editing_one_mode_leaves_the_others_alone(self, app):
        before = [m.f_static for m in app.session_state.params.modes]
        app.number_input(key="mode_2_f").set_value(999.0).run()
        after = [m.f_static for m in app.session_state.params.modes]
        assert after[2] == 999.0
        assert after[:2] + after[3:] == before[:2] + before[3:]


class TestMonitoringIsNotAParameter:
    """Solo and mute are a listening aid. If either ever edits `gain`, a sound
    designed while soloing would be silently wrong when saved."""

    def test_solo_does_not_touch_the_gains(self, app):
        before = [m.gain for m in app.session_state.params.modes]
        app.button(key="mode_3_solo").click().run()
        assert app.session_state.solo == {3}
        assert [m.gain for m in app.session_state.params.modes] == before

    def test_mute_does_not_touch_the_gains(self, app):
        before = [m.gain for m in app.session_state.params.modes]
        app.button(key="mode_1_mute").click().run()
        assert app.session_state.muted == {1}
        assert [m.gain for m in app.session_state.params.modes] == before

    def test_solo_and_mute_are_not_serialized(self, app):
        app.button(key="mode_2_solo").click().run()
        payload = app.session_state.params.to_dict()
        assert "solo" not in payload and "muted" not in payload
        assert "monitor" not in str(payload)

    def test_nothing_is_soloed_or_muted_to_begin_with(self, app):
        assert app.session_state.solo == set()
        assert app.session_state.muted == set()

    def test_solo_wins_over_mute(self, app):
        app.button(key="mode_0_mute").click().run()
        app.button(key="mode_4_solo").click().run()
        assert app.session_state.solo == {4}
        assert app.session_state.muted == {0}

    def test_clearing_restores_everything(self, app):
        app.button(key="mode_4_solo").click().run()
        for button in app.button:
            if button.label == "Clear solo/mute":
                button.click().run()
                break
        assert not app.session_state.solo and not app.session_state.muted


class TestMaster:
    def test_glide_slider_sets_k(self, app):
        app.slider(key="master_glide").set_value(1.0).run()
        assert app.session_state.params.tension.k == pytest.approx(2 ** (1 / 12) - 1)

    def test_tau_slider_sets_tau(self, app):
        app.slider(key="master_tau").set_value(0.3).run()
        assert app.session_state.params.tension.tau == pytest.approx(0.3)

    def test_output_gain_is_edited_in_db(self, app):
        app.number_input(key="master_gain").set_value(-6.0).run()
        assert app.session_state.params.output_gain == pytest.approx(
            10 ** (-6 / 20), rel=1e-6
        )

    def test_tune_scales_every_mode_and_resets_itself(self, app):
        before = [m.f_static for m in app.session_state.params.modes]
        app.slider(key="master_tune").set_value(12.0).run()
        after = [m.f_static for m in app.session_state.params.modes]
        assert all(b * 2 == pytest.approx(a) for a, b in zip(after, before))
        assert app.session_state.master_tune == 0.0, "the transform should be one-shot"

    def test_damping_scales_every_t60(self, app):
        before = [m.t60 for m in app.session_state.params.modes]
        app.slider(key="master_damp").set_value(0.5).run()
        after = [m.t60 for m in app.session_state.params.modes]
        assert all(b * 0.5 == pytest.approx(a) for a, b in zip(after, before))


class TestBandStrips:
    def test_level_is_edited_in_db(self, app):
        app.number_input(key="band_0_level").set_value(-26.0).run()
        assert app.session_state.params.noise[0].level == pytest.approx(
            10 ** (-26 / 20), rel=1e-6
        )

    def test_an_inverted_band_is_corrected_rather_than_rejected(self, app):
        """Dragging the low edge past the high one is a normal thing to do while
        editing; it should not raise."""
        high = app.session_state.params.noise[0].f_high
        app.number_input(key="band_0_low").set_value(high + 500.0).run()
        band = app.session_state.params.noise[0]
        assert band.f_low < band.f_high


class TestPresets:
    def test_loading_a_preset_replaces_the_parameters(self, app):
        app.selectbox(key="preset_pick").set_value("Kick").run()
        for button in app.button:
            if button.label == "Load preset":
                button.click().run()
                break
        assert app.session_state.params.name == "test_kick"
        assert app.session_state.params.fundamental() < 60.0

    def test_loading_clears_stale_widget_state(self, app):
        app.number_input(key="mode_0_f").set_value(500.0).run()
        app.selectbox(key="preset_pick").set_value("Kick").run()
        for button in app.button:
            if button.label == "Load preset":
                button.click().run()
                break
        assert app.session_state.params.modes[0].f_static != 500.0


# =============================================================================
# The training page
# =============================================================================

TRAINING = str(Path(__file__).resolve().parents[1] / "app_pages" / "training.py")


@pytest.fixture
def training():
    at = AppTest.from_file(TRAINING, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, at.exception[0].message if at.exception else ""
    return at


class TestTrainingPage:
    def test_it_runs(self, training):
        assert [t.value for t in training.title] == ["Training"]

    def test_one_drum_at_a_time(self, training):
        """§8: f_static and t60 belong to a specific physical drum. The page
        offers a dropdown, not a multiselect, on purpose."""
        assert [s.label for s in training.selectbox] == ["Drum"]
        assert not training.multiselect

    def test_cymbals_are_not_offered(self, training):
        """§9: a struck cymbal moves energy from low modes into high ones over
        the first few hundred milliseconds. A linear modal bank cannot do that
        at any setting, so fitting one is not a tuning problem."""
        options = training.selectbox[0].options
        assert options
        assert not any(
            word in option.lower()
            for option in options
            for word in ("cymbal", "crash", "ride", "hihat", "hi-hat", "splash")
        )

    def test_starts_idle(self, training):
        assert [b.label for b in training.button] == ["Start training"]


class _FinishedRun:
    """A `TrainingRun` that has already finished, so the result views can be
    driven without a subprocess or a 3.5 GB sample library."""

    def __init__(self, ready: dict, result: dict, scored: dict) -> None:
        self.ready, self.result, self.scored = ready, result, scored
        self.done = {"elapsed": result["elapsed"]}
        self.error = None
        self.steps: list[str] = ["stage 1", "stage 2", "stage 5"]
        self.generations = [
            {"generation": g["index"], "loss": g["loss"],
             "best_loss": g["best_loss"], "elapsed": g["elapsed"]}
            for g in result["generations"]
        ]

    is_running = False
    finished = True
    phase = "done"

    def progress_fraction(self) -> float:
        return 1.0

    def stop(self) -> None:
        pass

    def stderr_tail(self, lines: int = 12) -> str:
        return ""


@pytest.fixture(scope="module")
def finished_fit():
    """A real fit of a synthetic drum, serialized exactly the way the worker
    serializes one."""
    from tests.test_fitting import SR, known_drum, velocity_layers
    from drumsynth.fitting import (
        DrumTrainer, FitEvaluator, TargetBuilder, TrainingSettings,
    )
    from drumsynth.fitting.worker import FitWorker

    hits = velocity_layers(known_drum(), velocities=(45, 90, 127))
    target = TargetBuilder(SR, 0.8).from_audio("synthetic", hits)
    settings = TrainingSettings(seconds=0.8, max_layers=3, max_modes=10,
                                generations=2, population=6, noise_bands=2)
    result = DrumTrainer(settings, SR).run(target)

    worker = FitWorker(settings, SR)
    aggregate, cards = FitEvaluator(SR).score(result, target)
    ready = {
        "drum": target.drum,
        "layers": [
            {"velocity": layer.velocity,
             "velocity_normalized": layer.velocity_normalized,
             "source": layer.source}
            for layer in target.layers
        ],
    }
    scored = {
        "total": aggregate.total,
        "stft_loss": aggregate.stft_loss,
        "components": [c.to_dict() for c in aggregate.components],
        "warnings": aggregate.warnings,
        "per_layer": [{"velocity": v, "total": card.total} for v, card in cards],
        "report": aggregate.report(),
    }
    return ready, worker._summarize(result), scored


@pytest.fixture
def finished(monkeypatch, finished_fit):
    import streamlit as st
    import drumsynth.fitting as fitting

    monkeypatch.setattr(
        fitting, "TrainingRun", lambda *a, **k: _FinishedRun(*finished_fit)
    )
    # The page holds its handle in `st.cache_resource`, which outlives an
    # AppTest run — without this it would keep the real subprocess handle an
    # earlier test cached and the patch would do nothing.
    st.cache_resource.clear()
    at = AppTest.from_file(TRAINING, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, at.exception[0].message if at.exception else ""
    yield at
    st.cache_resource.clear()


class TestTrainingReport:
    def test_every_result_view_renders(self, finished):
        """The report, the velocity table, the audio comparison and the handoff
        into the live synth — the whole tail of the page, which no unit test
        reaches because it only exists once a fit has finished."""
        headers = [s.value for s in finished.subheader]
        assert "Report" in headers
        assert "Velocity table" in headers
        assert "Generated against the sample" in headers
        assert "Try it" in headers

    def test_setup_is_replaced_by_the_result(self, finished):
        assert "Start training" not in [b.label for b in finished.button]
        assert "Fit another drum" in [b.label for b in finished.button]

    def test_stage_3_verdict_is_shown_either_way(self, finished):
        text = " ".join([m.value for m in finished.success]
                        + [m.value for m in finished.warning])
        assert "Stage 3" in text

    def test_the_fitted_drum_can_be_loaded_into_the_live_synth(self, finished):
        for button in finished.button:
            if button.label == "Load into the live synth":
                button.click().run()
                break
        else:
            pytest.fail("no handoff button")
        assert not finished.exception
        assert finished.session_state.params.modes
