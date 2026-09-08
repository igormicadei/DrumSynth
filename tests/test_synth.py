"""The synthesizer: superposition, decay, tension, and the fast path."""

from __future__ import annotations

import numpy as np
import pytest

from drumsynth import (
    DampingCurve,
    Decay,
    DrumKit,
    DrumParams,
    DrumPresets,
    DrumSequencer,
    DrumVoice,
    ModalBank,
    Mode,
    ModeResonator,
    NoiseBand,
    Tension,
    TensionTracker,
)

SR = 44100


class TestDecayConversions:
    def test_round_trip(self):
        for t60 in (0.005, 0.05, 0.5, 2.3, 10.0):
            coef = Decay.t60_to_coef(t60, SR)
            assert Decay.coef_to_t60(coef, SR) == pytest.approx(t60, rel=1e-9)

    def test_reaches_minus_60_db_at_t60(self):
        t60 = 1.7
        coef = Decay.t60_to_coef(t60, SR)
        assert coef ** (t60 * SR) == pytest.approx(0.001, rel=1e-9)

    def test_rejects_non_positive(self):
        with pytest.raises(ValueError):
            Decay.t60_to_coef(0.0, SR)


class TestModeResonator:
    """The `+=` in excite() is load-bearing and easy to 'simplify' into `=`."""

    def test_excite_superposes_rather_than_replacing(self):
        resonator = ModeResonator(Mode(100.0, 1.0, 1.0), SR)
        resonator.excite(1.0)
        first = resonator.x
        resonator.excite(1.0)
        assert resonator.x == pytest.approx(2.0 * first)

    def test_second_strike_on_a_ringing_mode_adds_energy(self):
        resonator = ModeResonator(Mode(100.0, 1.0, 2.0), SR)
        resonator.excite(1.0)
        resonator.process(1000)
        before = resonator.energy
        resonator.excite(1.0)
        assert resonator.energy > before

    def test_decays_to_minus_60_db_at_t60(self):
        t60 = 0.5
        resonator = ModeResonator(Mode(200.0, 1.0, t60), SR)
        resonator.excite(1.0)
        out = resonator.process(int(t60 * SR))
        envelope_end = resonator.amplitude
        assert envelope_end == pytest.approx(0.001, rel=0.02)
        assert np.max(np.abs(out)) == pytest.approx(1.0, rel=0.02)

    def test_frequency_is_recoverable_from_state(self):
        resonator = ModeResonator(Mode(437.0, 1.0, 1.0), SR)
        assert resonator.frequency == pytest.approx(437.0, rel=1e-9)
        resonator.set_frequency_ratio(1.25)
        assert resonator.frequency == pytest.approx(437.0 * 1.25, rel=1e-9)

    def test_rejects_a_mode_above_nyquist(self):
        with pytest.raises(ValueError):
            ModeResonator(Mode(30000.0, 1.0, 1.0), SR)


class TestModalBankMatchesReference:
    """ModalBank's closed form must equal ModeResonator's per-sample loop.

    This is the test that protects the optimization: if the reference ever
    changes, the fast path has to change with it.
    """

    MODES = [
        Mode(92.5, 1.0, 2.3),
        Mode(88.0, 0.7, 2.2),
        Mode(147.4, 0.5, 0.9),
        Mode(3900.0, 0.05, 0.12),
    ]

    def _reference(self, n, ratio=1.0):
        resonators = [ModeResonator(mode, SR) for mode in self.MODES]
        for resonator in resonators:
            resonator.excite(1.0)
        return sum(resonator.process(n, ratio) for resonator in resonators), resonators

    def test_single_block_matches(self):
        expected, _ = self._reference(4000)
        bank = ModalBank(self.MODES, SR)
        bank.excite(1.0)
        assert np.allclose(bank.process(4000, 1.0), expected, atol=1e-10)

    def test_blocked_render_matches(self):
        expected, resonators = self._reference(4000)
        bank = ModalBank(self.MODES, SR)
        bank.excite(1.0)
        chunks, position = [], 0
        while position < 4000:
            span = min(64, 4000 - position)
            chunks.append(bank.process(span, 1.0))
            position += span
        assert np.allclose(np.concatenate(chunks), expected, atol=1e-10)
        assert np.allclose(bank.x, [r.x for r in resonators], atol=1e-10)
        assert np.allclose(bank.y, [r.y for r in resonators], atol=1e-10)

    def test_matches_across_a_frequency_change(self):
        resonators = [ModeResonator(mode, SR) for mode in self.MODES]
        for resonator in resonators:
            resonator.excite(1.0)
        expected = np.concatenate([
            sum(r.process(1000, 1.0) for r in resonators),
            sum(r.process(1000, 1.05) for r in resonators),
        ])

        bank = ModalBank(self.MODES, SR)
        bank.excite(1.0)
        actual = np.concatenate([bank.process(1000, 1.0), bank.process(1000, 1.05)])
        assert np.allclose(actual, expected, atol=1e-10)

    def test_short_blocks_take_the_direct_path_and_still_match(self):
        expected, _ = self._reference(12)
        bank = ModalBank(self.MODES, SR)
        bank.excite(1.0)
        actual = np.concatenate([bank.process(2, 1.0) for _ in range(6)])
        assert np.allclose(actual, expected, atol=1e-12)


class TestTension:
    def test_inactive_when_k_is_zero(self):
        assert not Tension(k=0.0).is_active
        assert Tension(k=0.1).is_active

    def test_smoothing_coefficient_scales_with_the_update_period(self):
        tension = Tension(k=0.1, tau=0.1)
        fast = tension.smoothing_coef(SR, 1)
        slow = tension.smoothing_coef(SR, 64)
        # 64 updates at the fast rate should land close to one at the slow rate.
        assert (1 - (1 - fast) ** 64) == pytest.approx(slow, rel=1e-9)

    def test_instant_attack_follows_a_rise_immediately(self):
        tracker = TensionTracker(Tension(k=1.0, tau=0.1), SR, 64)
        assert tracker.update(1.0) == pytest.approx(2.0)

    def test_symmetric_mode_ramps_instead(self):
        tracker = TensionTracker(Tension(k=1.0, tau=0.1, instant_attack=False), SR, 64)
        assert tracker.update(1.0) < 1.02

    def test_ratio_is_clamped(self):
        tracker = TensionTracker(Tension(k=1e6, tau=0.1, max_ratio=1.5), SR)
        assert tracker.update(1.0) == pytest.approx(1.5)


class TestDrumVoice:
    def test_render_length_and_finiteness(self, voice):
        out = voice.render_hit(1.0)
        assert len(out) == SR
        assert np.all(np.isfinite(out))

    def test_unit_strike_peaks_where_the_preset_says(self, voice):
        assert np.max(np.abs(voice.render_hit(1.0))) == pytest.approx(0.9, rel=0.02)

    def test_amplitude_scales_the_output(self, voice):
        quiet = np.max(np.abs(voice.render_hit(1.0, 0.25)))
        loud = np.max(np.abs(voice.render_hit(1.0, 1.0)))
        assert loud / quiet == pytest.approx(4.0, rel=0.05)

    def test_seeded_renders_are_identical(self, tom_params):
        a = DrumVoice(tom_params, SR, 64, seed=7).render_hit(0.5)
        b = DrumVoice(tom_params, SR, 64, seed=7).render_hit(0.5)
        assert np.array_equal(a, b)

    def test_unseeded_renders_differ_only_in_the_noise(self, tom_params):
        a = DrumVoice(tom_params, SR, 64, seed=1).render_hit(0.5)
        b = DrumVoice(tom_params, SR, 64, seed=2).render_hit(0.5)
        assert not np.array_equal(a, b)
        assert np.corrcoef(a, b)[0, 1] > 0.95  # modal part dominates and is identical

    def test_control_period_is_inaudible(self, tom_params):
        """Compared as a trajectory, not as a waveform.

        The two settings put the fundamental within a few cents of each other
        everywhere, which is inaudible — and yet their waveform correlation is
        only 0.988, because a 3-cent difference decorrelates phase within a
        couple of seconds. That is the same effect the scoring module refuses
        to null-test on, demonstrated on the engine's own output.
        """
        fine = DrumVoice(tom_params, SR, 1, seed=3)
        coarse = DrumVoice(tom_params, SR, 64, seed=3)
        fine_times, fine_freqs = fine.glide_trace(2.0)
        coarse_times, coarse_freqs = coarse.glide_trace(2.0)

        sampled = np.interp(coarse_times, fine_times, fine_freqs)
        cents = np.abs(1200 * np.log2(sampled / coarse_freqs))
        assert np.max(cents) < 10.0

    def test_control_period_does_not_change_the_noise_stream(self, tom_params):
        params = tom_params.copy()
        params.tension = Tension(k=0.0)
        params.modes = params.modes[:1]
        fine = DrumVoice(params, SR, 1, seed=3).render_hit(0.3)
        coarse = DrumVoice(params, SR, 64, seed=3).render_hit(0.3)
        assert np.allclose(fine, coarse)

    def test_reset_clears_state(self, voice):
        voice.strike(1.0)
        voice.render(1000)
        assert voice.energy > 0
        voice.reset()
        assert voice.energy == pytest.approx(0.0)
        assert voice.frequency_ratio == pytest.approx(1.0)

    def test_is_silent_is_a_method_not_a_property(self, voice):
        # The skeleton declared this as a property taking an argument, which
        # Python cannot express.
        assert voice.is_silent(-120.0) is True
        voice.strike(1.0)
        assert voice.is_silent(-120.0) is False
        assert isinstance(voice.silent, bool)

    def test_glide_falls_and_settles(self, voice):
        times, freqs = voice.glide_trace(2.0)
        assert freqs[0] > freqs[-1]
        depth = 12 * np.log2(freqs.max() / freqs[-1])
        assert 1.5 < depth < 2.5  # the reference measured 2.07 semitones
        assert np.all(np.diff(freqs) <= 1e-9)  # monotone, no initial rise

    def test_no_glide_when_tension_is_off(self, tom_params):
        params = tom_params.copy()
        params.tension = Tension(k=0.0)
        _, freqs = DrumVoice(params, SR, 64, seed=1).glide_trace(1.0)
        assert np.allclose(freqs, freqs[0])

    def test_a_harder_hit_glides_deeper_with_no_velocity_term(self, tom_params):
        """The falsifiable claim: k is velocity-invariant."""
        voice = DrumVoice(tom_params, SR, 64, seed=1)
        _, soft = voice.glide_trace(2.0, amplitude=0.2)
        _, hard = voice.glide_trace(2.0, amplitude=1.0)
        soft_depth = 12 * np.log2(soft.max() / soft[-1])
        hard_depth = 12 * np.log2(hard.max() / hard[-1])
        assert hard_depth > soft_depth * 2


class TestSuperposition:
    def test_two_hits_superpose_in_the_tail(self, voice):
        single = voice.render_sequence([(0.0, 1.0)], 2.0)
        double = voice.render_sequence([(0.0, 1.0), (0.5, 1.0)], 2.0)
        tail = slice(int(0.6 * SR), int(1.0 * SR))
        assert np.sqrt(np.mean(double[tail] ** 2)) > np.sqrt(np.mean(single[tail] ** 2))

    def test_a_flam_needs_no_special_case(self, tom_params):
        kit = DrumKit(SR)
        kit.add("tom", tom_params, 64, seed=1)
        sequencer = DrumSequencer(kit).add_flam(0.1, "tom")
        assert len(sequencer) == 2
        assert np.all(np.isfinite(sequencer.render(1.0)))


class TestDrumParams:
    def test_serialization_round_trip(self, tom_params, tmp_path):
        path = tom_params.save(tmp_path / "tom.json")
        loaded = DrumParams.load(path)
        assert len(loaded.modes) == len(tom_params.modes)
        assert loaded.modes[0].f_static == pytest.approx(tom_params.modes[0].f_static)
        assert loaded.tension.k == pytest.approx(tom_params.tension.k)
        assert loaded.output_gain == pytest.approx(tom_params.output_gain)

    def test_validate_rejects_a_mode_above_nyquist(self):
        params = DrumParams(modes=[Mode(30000.0, 1.0, 1.0)])
        with pytest.raises(ValueError, match="Nyquist"):
            params.validate(SR)

    def test_validate_rejects_an_empty_bank(self):
        with pytest.raises(ValueError, match="no modes"):
            DrumParams().validate(SR)

    def test_tuning_is_one_multiplier(self, tom_params):
        tuned = tom_params.tuned(1.5)
        assert tuned.fundamental() == pytest.approx(tom_params.fundamental() * 1.5)
        assert tuned.modes[0].t60 == pytest.approx(tom_params.modes[0].t60)

    def test_muting_only_touches_t60(self, tom_params):
        muted = tom_params.muted(0.4)
        assert muted.modes[0].t60 == pytest.approx(tom_params.modes[0].t60 * 0.4)
        assert muted.modes[0].f_static == pytest.approx(tom_params.modes[0].f_static)

    def test_noise_band_validation(self):
        with pytest.raises(ValueError, match="f_low < f_high"):
            NoiseBand(800.0, 200.0, 0.1, 0.05).validate(SR)


class TestPresets:
    def test_damping_curve_hits_its_anchors(self):
        curve = DampingCurve()
        for freq, t60 in DampingCurve.REFERENCE_ANCHORS:
            assert curve(freq) == pytest.approx(t60, rel=1e-6)

    def test_damping_falls_with_frequency(self):
        curve = DampingCurve()
        assert curve(92.5) > curve(300) > curve(600) > curve(2000)

    def test_layout_is_inharmonic(self):
        """Membrane modes are not integer multiples; assuming they are is the
        fastest way to make a modal drum sound like a synthesizer."""
        freqs = DrumPresets.tom().mode_frequencies()
        ratios = freqs / freqs.min()
        assert not np.any(np.isclose(ratios, 2.0, atol=0.02))
        assert not np.any(np.isclose(ratios, 3.0, atol=0.02))

    def test_close_pair_exists(self):
        freqs = np.sort(DrumPresets.tom().mode_frequencies())
        gaps_cents = 1200 * np.log2(freqs[1:] / freqs[:-1])
        assert np.min(gaps_cents) < 120  # the deliberate warble pair

    def test_gains_are_normalized_to_unit_energy(self):
        params = DrumPresets.tom()
        assert np.sum(params.mode_gains() ** 2) == pytest.approx(1.0)

    def test_every_preset_validates_and_renders(self):
        for name, params in DrumPresets.all(SR).items():
            params.validate(SR)
            out = DrumVoice(params, SR, 64, seed=1).render_hit(0.5)
            assert np.all(np.isfinite(out)), name
            assert np.max(np.abs(out)) > 0.1, name
