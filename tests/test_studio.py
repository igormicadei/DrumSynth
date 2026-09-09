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


class TestOutOfRangeParameters:
    def test_a_long_transient_decay_does_not_crash_the_strip(self, app):
        """A transient band used to be a click, and the strip capped its `t60`
        at one second. The fit now measures each band's decay on the residual —
        a real tom's 200-800 Hz band came back at 1.7 s — and loading that fit
        raised StreamlitValueAboveMaxError before the page drew anything."""
        from drumsynth import DrumParams, Mode, NoiseBand, Tension

        app.session_state.params = DrumParams(
            modes=[Mode(90.0, 1.0, 1.8)],
            noise=[NoiseBand(200.0, 800.0, 0.05, 1.72)],
            tension=Tension(k=0.0, tau=0.12), name="fitted",
        )
        app.run()
        assert not app.exception, (
            app.exception[0].message if app.exception else "")

    def test_values_past_the_editor_range_are_clamped_not_raised(self, app):
        """These numbers can arrive from a hand-edited JSON as easily as from a
        fit, and a widget that raises takes the whole page with it."""
        from drumsynth import DrumParams, Mode, NoiseBand, Tension

        app.session_state.params = DrumParams(
            modes=[Mode(90.0, 1.0, 1.8)],
            noise=[NoiseBand(200.0, 800.0, 40.0, 90.0)],
            tension=Tension(k=0.0, tau=0.12), name="absurd",
        )
        app.run()
        assert not app.exception, (
            app.exception[0].message if app.exception else "")


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
PREVIOUS_RUNS = str(
    Path(__file__).resolve().parents[1] / "app_pages" / "previous_runs.py"
)
RUN_PAGE = str(Path(__file__).resolve().parents[1] / "app_pages" / "run.py")


@pytest.fixture
def empty_store(monkeypatch, tmp_path):
    """Point the run store somewhere empty, so a developer's own `out/runs`
    cannot change what these tests see."""
    from drumsynth.fitting.runs import RunStore

    monkeypatch.setenv(RunStore.ROOT_VARIABLE, str(tmp_path / "runs"))
    return tmp_path / "runs"


@pytest.fixture
def training(empty_store):
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
        assert training.selectbox[0].label == "Drum"
        assert not training.multiselect

    def test_the_device_picker_offers_what_exists(self, training):
        """Stage 5 is the only stage that can move to a GPU, and the picker
        says so rather than offering a device that is not there."""
        from drumsynth.fitting import DeviceChoice

        picker = [s for s in training.selectbox if s.label == "Stage 5 device"]
        assert picker, [s.label for s in training.selectbox]
        assert len(picker[0].options) == len(DeviceChoice.options())

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
        assert "Start training" in [b.label for b in training.button]

    def test_an_empty_store_says_so_rather_than_erroring(self, training):
        assert any(b.label == "Previous runs" for b in training.button)

    def test_previous_runs_page_handles_empty_store(self, empty_store):
        at = AppTest.from_file(PREVIOUS_RUNS, default_timeout=TIMEOUT)
        at.run()
        assert not at.exception
        assert any("No runs stored" in i.value for i in at.info)


# =============================================================================
# The report, over a stored run
# =============================================================================


@pytest.fixture(scope="module")
def fitted_run(tmp_path_factory):
    """A real fit of a synthetic drum, written to a real run directory.

    The page reads finished runs back from disk rather than from the event
    stream, so a stored run is what these tests have to provide — and testing
    the read-back path is worth more than testing a stub of it."""
    from tests.test_fitting import SR, known_drum, velocity_layers
    from drumsynth.fitting import (
        DrumTrainer, FitEvaluator, TargetBuilder, TrainingSettings,
    )
    from drumsynth.fitting.runs import RunStore
    from drumsynth.fitting.worker import FitWorker

    root = tmp_path_factory.mktemp("runs")
    hits = velocity_layers(known_drum(), velocities=(45, 90, 127))
    target = TargetBuilder(SR, 0.8).from_audio("synthetic", hits)
    settings = TrainingSettings(seconds=0.8, max_layers=3, max_modes=10,
                                generations=2, population=6, noise_bands=2,
                                device="cpu")
    result = DrumTrainer(settings, SR).run(target)

    evaluator = FitEvaluator(SR)
    aggregate, cards = evaluator.score(result, target)
    layers = [
        {"velocity": layer.velocity,
         "generated": evaluator.render(result, layer.velocity_normalized, 0.8),
         "reference": layer.audio}
        for layer in target.layers
    ]
    summary = FitWorker(settings, SR)._summarize(result)
    summary["settings"] = settings.to_dict()
    score = {
        "total": aggregate.total, "stft_loss": aggregate.stft_loss,
        "components": [c.to_dict() for c in aggregate.components],
        "warnings": aggregate.warnings,
        "per_layer": [{"velocity": v, "total": c.total} for v, c in cards],
        "report": aggregate.report(),
    }
    directory = RunStore(root).save(
        "synthetic", FitWorker._plain(summary), result.params,
        FitWorker._plain(score), layers, SR)
    return root, directory


@pytest.fixture
def stored(monkeypatch, fitted_run):
    """The page with one stored run and nothing running."""
    import streamlit as st
    from drumsynth.fitting.runs import RunStore

    root, directory = fitted_run
    monkeypatch.setenv(RunStore.ROOT_VARIABLE, str(root))
    st.cache_resource.clear()
    st.cache_data.clear()
    at = AppTest.from_file(RUN_PAGE, default_timeout=TIMEOUT)
    at.query_params["run"] = str(directory)
    at.run()
    assert not at.exception, at.exception[0].message if at.exception else ""
    yield at
    st.cache_resource.clear()
    st.cache_data.clear()


class TestTrainingReport:
    def test_a_stored_run_is_openable(self, stored):
        """Every fit is kept. A run takes minutes and produces a drum you
        cannot judge in one listen, so the comparison that matters is against
        the previous run — which needs the previous run to still exist."""
        assert [t.value for t in stored.title] == ["Run"]

    def test_every_result_view_renders(self, stored):
        headers = [s.value for s in stored.subheader]
        for expected in ("Velocity table", "Report", "Where the time went",
                         "Generated against the sample", "Try it"):
            assert expected in headers, headers

    def test_the_comparison_shows_both_signals(self, stored):
        """Two players and the four views. A report that only plays the
        generated drum is not a comparison."""
        # AppTest has no accessor for st.audio, so the players are checked by
        # their headings; the tabs are what the four views hang off.
        markdown = [m.value for m in stored.markdown]
        assert "**Sample**" in markdown and "**Generated**" in markdown
        labels = [t.label for t in stored.tabs]
        for expected in ("Waveform", "Spectrum", "Decay", "Spectrogram"):
            assert expected in labels, labels

    def test_stage_3_verdict_is_shown_either_way(self, stored):
        text = " ".join([m.value for m in stored.success]
                        + [m.value for m in stored.warning])
        assert "Stage 3" in text

    def test_the_fitted_drum_can_be_loaded_into_the_live_synth(self, stored):
        for button in stored.button:
            if button.label == "Load into the live synth":
                button.click().run()
                break
        else:
            pytest.fail(f"no handoff button in {[b.label for b in stored.button]}")
        assert not stored.exception
        assert stored.session_state.params.modes


class TestTrainedDrumsInTheSidebar:
    def test_a_stored_fit_is_loadable_from_any_page(self, monkeypatch,
                                                   fitted_run):
        """The runs are on disk, not in this session's memory, so a session
        that never ran a fit can still play one."""
        import streamlit as st
        from drumsynth.fitting.runs import RunStore

        root, _ = fitted_run
        monkeypatch.setenv(RunStore.ROOT_VARIABLE, str(root))
        st.cache_resource.clear()

        at = AppTest.from_file(APP, default_timeout=TIMEOUT)
        at.run()
        assert not at.exception, at.exception[0].message if at.exception else ""

        picker = [s for s in at.sidebar.selectbox if s.label == "Fit"]
        assert picker, [s.label for s in at.sidebar.selectbox]

        for button in at.sidebar.button:
            if button.label == "Load trained drum":
                button.click().run()
                break
        else:
            pytest.fail("no load button in the sidebar")
        assert not at.exception
        assert at.session_state.params.modes
        st.cache_resource.clear()
