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
