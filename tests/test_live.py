"""The live engine: a synth that never stops, driven over a pipe.

Every test here runs against the `null` sink, which renders on a timer and
discards the result. That is the whole reason the sink is split out — the
command path, live parameter updates, monitoring and telemetry are all
exercised without a sound card, so this suite runs in CI and in a container.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from drumsynth import DrumPresets, Mode, NoiseBand, Tension
from drumsynth.live import Command, EngineSettings, Event, LiveSynth, NullSink, WavSink
from drumsynth.live.engine import LiveEngine

SR = 44100


def settle(synth: LiveSynth, seconds: float = 0.35) -> None:
    """Wait long enough for the engine to render and report."""
    time.sleep(seconds)


@pytest.fixture
def synth():
    live = LiveSynth(EngineSettings(sink="null", block_size=256, status_interval=0.05))
    live.start()
    yield live
    live.stop()


class TestProtocol:
    def test_commands_round_trip(self):
        line = Command.encode(Command.STRIKE, amplitude=0.5)
        assert Command.decode(line) == {"cmd": "strike", "amplitude": 0.5}

    def test_garbage_decodes_to_none(self):
        for line in ("", "   ", "not json", '{"no":"cmd"}', "[1,2,3]"):
            assert Command.decode(line) is None
            assert Event.decode(line) is None

    def test_events_round_trip(self):
        assert Event.decode(Event.encode(Event.STATUS, peak=0.5)) == {
            "ev": "status", "peak": 0.5
        }


class TestEngineProcess:
    def test_starts_and_reports_ready(self, synth):
        assert synth.is_running
        assert synth.ready["sr"] == SR
        assert synth.ready["sink"] == "null"
        assert synth.ready["latency_ms"] == pytest.approx(256 / SR * 1000, rel=0.01)

    def test_telemetry_arrives(self, synth):
        settle(synth)
        assert synth.status.blocks > 0
        assert synth.status.seconds > 0

    def test_silent_until_struck(self, synth):
        settle(synth)
        assert synth.status.peak == pytest.approx(0.0, abs=1e-9)

    def test_a_strike_makes_sound(self, synth):
        synth.strike(1.0)
        settle(synth)
        assert synth.status.strikes == 1
        assert synth.status.peak > 0.01

    def test_it_keeps_running_between_strikes(self, synth):
        synth.strike(1.0)
        settle(synth)
        first = synth.status.blocks
        settle(synth)
        assert synth.status.blocks > first, "the engine stopped rendering"

    def test_renders_faster_than_realtime(self, synth):
        synth.strike(1.0)
        settle(synth)
        assert 0.0 < synth.status.load < 1.0, "cannot keep up with the clock"

    def test_stops_cleanly(self, synth):
        synth.stop()
        assert not synth.is_running

    def test_a_bad_command_does_not_kill_the_engine(self, synth):
        synth._send("nonsense-command")
        settle(synth)
        assert synth.is_running
        assert synth.last_error and "unknown command" in synth.last_error
        synth.strike(1.0)
        settle(synth)
        assert synth.status.strikes == 1


class TestPlayingLive:
    def test_a_second_strike_lands_on_a_ringing_bank(self, synth):
        """The point of a persistent engine: strikes superpose rather than
        retriggering a fresh voice."""
        synth.strike(1.0)
        settle(synth, 0.5)
        decayed = synth.status.energy
        synth.strike(1.0)
        settle(synth, 0.15)
        assert synth.status.energy > decayed
        assert synth.status.strikes == 2

    def test_the_glide_deepens_on_a_harder_hit(self, synth):
        synth.strike(0.15)
        settle(synth, 0.15)
        soft = synth.status.frequency_ratio
        synth.panic()
        settle(synth, 0.15)
        synth.strike(1.0)
        settle(synth, 0.15)
        hard = synth.status.frequency_ratio
        assert hard > soft, "energy feedback is not driving the glide"

    def test_panic_silences_what_is_ringing(self, synth):
        synth.strike(1.0)
        settle(synth, 0.15)
        assert synth.status.peak > 0.01
        synth.panic()
        settle(synth, 0.2)
        assert synth.status.peak < 1e-6


class TestLiveParameterChanges:
    def test_parameters_change_without_restarting_the_ring(self, synth):
        synth.strike(1.0)
        settle(synth, 0.2)
        before = synth.status.energy
        assert before > 0

        params = DrumPresets.tom()
        params.modes = [mode.scaled(t60_ratio=0.2) for mode in params.modes]
        synth.load(params)
        settle(synth, 0.05)

        # Still ringing (not reset to zero), but decaying faster than it was.
        assert 0 < synth.status.energy
        settle(synth, 0.3)
        assert synth.status.energy < before

    def test_turning_tension_off_stops_the_glide(self, synth):
        synth.strike(1.0)
        settle(synth, 0.15)
        assert synth.status.frequency_ratio > 1.001

        params = DrumPresets.tom()
        params.tension = Tension(k=0.0)
        synth.load(params)
        settle(synth, 0.2)
        assert synth.status.frequency_ratio == pytest.approx(1.0)

    def test_output_gain_takes_effect_immediately(self, synth):
        synth.strike(1.0)
        settle(synth, 0.1)
        loud = synth.status.peak
        synth.set_output_gain(0.001)
        settle(synth, 0.1)
        assert synth.status.peak < loud * 0.5

    def test_adding_and_removing_modes_while_ringing(self, synth):
        synth.strike(1.0)
        params = DrumPresets.tom()
        params.modes = params.modes + [Mode(1234.0, 0.05, 0.3)]
        synth.load(params)
        settle(synth, 0.1)
        assert synth.is_running

        params.modes = params.modes[:6]
        synth.load(params)
        settle(synth, 0.1)
        assert synth.is_running and synth.last_error is None

    def test_changing_noise_bands_while_ringing(self, synth):
        synth.strike(1.0)
        params = DrumPresets.tom()
        params.noise = [NoiseBand(150.0, 900.0, 0.08, 0.05)]
        synth.load(params)
        settle(synth, 0.15)
        assert synth.is_running and synth.last_error is None


class TestMonitoring:
    def test_solo_silences_the_other_partials_live(self, synth):
        synth.strike(1.0)
        settle(synth, 0.15)
        full = synth.status.peak

        count = len(DrumPresets.tom().modes)
        synth.set_monitor([1.0 if index == 0 else 0.0 for index in range(count)])
        settle(synth, 0.15)
        assert synth.status.peak < full

    def test_clearing_the_mask_restores_everything(self, synth):
        count = len(DrumPresets.tom().modes)
        synth.set_monitor([0.0] * count)
        synth.strike(1.0)
        settle(synth, 0.15)
        muted = synth.status.peak

        synth.set_monitor(None)
        synth.strike(1.0)
        settle(synth, 0.15)
        assert synth.status.peak > muted

    def test_monitoring_does_not_change_the_energy_that_drives_the_glide(self, synth):
        """The mask is applied at the mix, so the bank still rings underneath —
        soloing must not change the physics, only what you hear."""
        synth.strike(1.0)
        settle(synth, 0.15)
        energy = synth.status.energy

        count = len(DrumPresets.tom().modes)
        synth.set_monitor([1.0 if index == 0 else 0.0 for index in range(count)])
        settle(synth, 0.05)
        assert synth.status.energy == pytest.approx(energy, rel=0.5)


class TestRecording:
    def test_captures_a_take_to_disk(self, tmp_path):
        live = LiveSynth(EngineSettings(sink="null", status_interval=0.05))
        live.start()
        try:
            path = tmp_path / "take.wav"
            live.record(path, seconds=0.3)
            live.strike(1.0)
            deadline = time.monotonic() + 10.0
            while live.last_recording is None and time.monotonic() < deadline:
                time.sleep(0.05)
            assert live.last_recording is not None, "no recorded event"
            assert path.exists()
            assert live.last_recording["seconds"] == pytest.approx(0.3, rel=0.1)
            assert live.last_recording["peak"] > 0.0
        finally:
            live.stop()


class TestSinks:
    def test_null_sink_runs_without_audio_hardware(self):
        rendered = []
        sink = NullSink(SR, 128, realtime=False)
        sink.start(lambda n: rendered.append(n) or np.zeros(n))
        time.sleep(0.05)
        sink.stop()
        assert len(rendered) > 0

    def test_wav_sink_keeps_what_it_rendered(self, tmp_path):
        sink = WavSink(SR, 128, str(tmp_path / "o.wav"), realtime=False)
        sink.start(lambda n: np.full(n, 0.25))
        time.sleep(0.05)
        sink.stop()
        assert sink.audio().size > 0
        assert np.allclose(sink.audio(), 0.25)


class TestEngineDirectly:
    """Without a subprocess, so a failure points at the engine rather than IPC."""

    def test_render_applies_queued_commands(self):
        engine = LiveEngine(EngineSettings(sink="null"))
        engine._pending.append({"cmd": Command.STRIKE, "amplitude": 1.0})
        block = engine.render(512)
        assert np.max(np.abs(block)) > 0.0
        assert engine._strikes == 1

    def test_render_survives_a_broken_command(self):
        engine = LiveEngine(EngineSettings(sink="null"))
        engine._pending.append({"cmd": Command.PARAMS, "params": {"modes": "nonsense"}})
        block = engine.render(256)
        assert np.all(np.isfinite(block))
