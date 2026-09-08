"""The scoring chain: analysis accuracy, matching, and discrimination."""

from __future__ import annotations

import numpy as np
import pytest

from drumsynth import (
    Analyzer,
    BandDecayAnalyzer,
    Comparator,
    Decay,
    DrumScorer,
    DrumVoice,
    EnvelopeAnalyzer,
    GlideAnalyzer,
    ModalAnalyzer,
    ModeEstimate,
    ModeMatcher,
    MultiResolutionSTFTLoss,
    NoiseAnalyzer,
    ScoreReport,
    SignalPrep,
    Tension,
)

SR = 44100


class TestSignalPrep:
    def test_finds_the_onset_after_padded_silence(self):
        t = np.arange(SR) / SR
        hit = np.exp(-t / 0.3) * np.sin(2 * np.pi * 100 * t)
        padded = np.concatenate([np.zeros(1000), hit])
        assert SignalPrep(SR).find_onset(padded) == pytest.approx(1000, abs=5)

    def test_trim_makes_two_offsets_agree(self):
        t = np.arange(SR) / SR
        hit = np.exp(-t / 0.3) * np.sin(2 * np.pi * 100 * t)
        prep = SignalPrep(SR)
        a = prep.trim_to_onset(np.concatenate([np.zeros(1000), hit]))
        b = prep.trim_to_onset(np.concatenate([np.zeros(7000), hit]))
        assert np.allclose(a[:2000], b[:2000], atol=1e-9)

    def test_rms_normalization_is_not_hostage_to_one_sample(self):
        prep = SignalPrep(SR)
        signal = np.random.default_rng(0).standard_normal(SR)
        spiked = signal.copy()
        spiked[100] = 50.0
        assert np.sqrt(np.mean(prep.normalize(signal) ** 2)) == pytest.approx(1.0)
        # A peak-normalized version would move by ~34 dB; RMS barely moves.
        ratio = np.std(prep.normalize(spiked)) / np.std(prep.normalize(signal))
        assert 0.9 < ratio < 1.1

    def test_noise_floor_tracks_the_actual_floor(self):
        t = np.arange(3 * SR) / SR
        signal = np.exp(-t / 0.1) * np.sin(2 * np.pi * 300 * t)
        signal += 1e-4 * np.random.default_rng(0).standard_normal(len(t))
        floor = SignalPrep(SR).estimate_noise_floor(signal)
        assert -90 < floor < -60


class TestBandDecayAnalyzer:
    @pytest.mark.parametrize("t60", [0.31, 0.55, 2.30])
    def test_recovers_a_known_t60(self, t60):
        t = np.arange(4 * SR) / SR
        signal = np.exp(-Decay.t60_to_damping(t60) * t) * np.sin(2 * np.pi * 92.5 * t)
        signal += 1e-7 * np.random.default_rng(0).standard_normal(len(t))
        band = BandDecayAnalyzer(SR).analyze(signal, -120.0)[0]
        assert band.t60 == pytest.approx(t60, rel=0.05)
        assert band.r_squared > 0.98

    def test_two_modes_bend_the_fit_and_bias_the_t60(self):
        """Curvature in dB is more than one mode, never one non-exponential
        mode. This is why there is no decay_shape parameter: a shape parameter
        would let a fitter hide the second mode behind a fake curve.

        The effect on r_squared is small for two well-separated exponentials —
        the sum converges onto the slower one's line and most of a 60 dB fit
        range sits in that regime — but it is there, and the fitted t60 lands
        between the two and matches neither, which is the part that costs you.
        """
        t = np.arange(4 * SR) / SR
        one = np.exp(-Decay.t60_to_damping(2.0) * t) * np.sin(2 * np.pi * 300 * t)
        two = one + np.exp(-Decay.t60_to_damping(1.0) * t) * np.sin(2 * np.pi * 340 * t)

        analyzer = BandDecayAnalyzer(SR, bands=[(230, 400)])
        single = analyzer.analyze(one, -120.0)[0]
        double = analyzer.analyze(two, -120.0)[0]

        assert single.t60 == pytest.approx(2.0, rel=0.02)
        assert single.r_squared > 0.999
        # The second mode bends the fit and drags the single fitted t60 toward
        # itself. Neither number is either mode's t60 any more.
        assert double.r_squared < single.r_squared
        assert 1.0 < double.t60 < single.t60

    def test_a_dense_band_of_a_real_drum_fits_visibly_curved(self, tom_params):
        """On a real bank, the low band holds many modes across a wide t60
        spread, and the straight-line fit stops being straight."""
        audio = DrumVoice(tom_params, SR, 64, seed=1).render_hit(4.0)
        bands = BandDecayAnalyzer(SR).analyze(audio, -110.0)
        valid = [band for band in bands if band.is_valid]
        assert valid
        assert min(band.r_squared for band in valid) < 0.95


class TestGlideAnalyzer:
    def test_tracks_the_engine_within_a_few_cents(self, tom_params):
        """The engine's own glide_trace is the ground truth. Without the close
        pair, the analyzer should agree with it closely."""
        params = tom_params.copy()
        fundamental = params.fundamental()
        params.modes = [m for m in params.modes if m.f_static > fundamental + 1.0]
        params.noise = []

        voice = DrumVoice(params, SR, 64, seed=1)
        audio = voice.render_hit(3.0)
        times, freqs = voice.glide_trace(3.0)

        track = GlideAnalyzer(SR, (60.0, 140.0)).analyze(audio)
        measured_times, measured_freqs = track.reliable_track()
        expected = np.interp(measured_times, times, freqs)
        mask = measured_times > 0.15  # first window cannot see before its center
        cents = np.abs(1200 * np.log2(measured_freqs[mask] / expected[mask]))
        assert np.max(cents) < 10.0

    def test_reports_no_glide_when_there_is_none(self):
        t = np.arange(2 * SR) / SR
        static = np.exp(-t / 0.8) * np.sin(2 * np.pi * 92.5 * t)
        track = GlideAnalyzer(SR, (60.0, 140.0)).analyze(static)
        assert abs(track.depth_cents) < 10.0

    def test_two_static_modes_are_flagged_as_not_a_glide(self):
        t = np.arange(2 * SR) / SR
        pair = np.exp(-t / 1.5) * np.sin(2 * np.pi * 88.0 * t)
        pair += np.exp(-t / 0.4) * np.sin(2 * np.pi * 130.0 * t)
        assert not GlideAnalyzer(SR, (60.0, 140.0)).is_genuine_glide(pair)


class TestModalAnalyzer:
    def test_recovers_known_modes(self, synthetic_signal, synthetic_modes):
        found = ModalAnalyzer(SR).analyze(synthetic_signal)
        for freq, amplitude, t60 in synthetic_modes:
            nearest = min(found, key=lambda mode: abs(mode.freq - freq))
            assert abs(1200 * np.log2(nearest.freq / freq)) < 5.0, f"{freq} Hz"
            assert nearest.t60 == pytest.approx(t60, rel=0.10), f"{freq} Hz"
            assert nearest.amplitude == pytest.approx(amplitude, rel=0.25), f"{freq} Hz"

    def test_resolves_a_close_pair(self, synthetic_signal):
        """88.0 and 92.5 Hz are 86 cents apart and 4.5 Hz apart — far closer
        than an FFT of any usable length resolves."""
        found = ModalAnalyzer(SR).analyze(synthetic_signal)
        low = [m for m in found if 87.0 < m.freq < 89.0]
        high = [m for m in found if 91.5 < m.freq < 93.5]
        assert len(low) == 1 and len(high) == 1

    def test_finds_few_spurious_modes(self, synthetic_signal):
        found = ModalAnalyzer(SR).analyze(synthetic_signal)
        assert len(found) <= 9  # six real modes, a little slack

    def test_prune_drops_growing_and_inaudible_poles(self):
        analyzer = ModalAnalyzer(SR)
        modes = [
            ModeEstimate(freq=100.0, amplitude=1.0, t60=1.0),
            ModeEstimate(freq=200.0, amplitude=1e-9, t60=1.0),   # inaudible
            ModeEstimate(freq=300.0, amplitude=0.5, t60=1e-6),   # not a mode
            ModeEstimate(freq=99999.0, amplitude=0.5, t60=1.0),  # out of range
        ]
        kept = analyzer.prune(modes)
        assert [round(mode.freq) for mode in kept] == [100]


class TestModeMatcher:
    def test_a_missing_mode_does_not_shift_the_rest(self):
        """Sort-and-zip breaks here; assignment does not."""
        reference = [
            ModeEstimate(88.0, 0.7, 2.2),
            ModeEstimate(92.5, 1.0, 2.3),
            ModeEstimate(147.0, 0.45, 1.1),
            ModeEstimate(197.0, 0.30, 0.9),
        ]
        generated = [
            ModeEstimate(92.6, 1.0, 2.4),
            ModeEstimate(148.0, 0.4, 1.05),
            ModeEstimate(196.0, 0.3, 0.85),
        ]
        matches = ModeMatcher().match(reference, generated)
        assert sum(match.is_missing for match in matches) == 1
        assert sum(match.is_spurious for match in matches) == 0
        for match in matches:
            if match.is_matched:
                assert abs(match.freq_error_cents) < 20.0

    def test_a_distant_pair_is_refused(self):
        matches = ModeMatcher(max_distance_cents=50.0).match(
            [ModeEstimate(100.0, 1.0, 1.0)], [ModeEstimate(200.0, 1.0, 1.0)]
        )
        assert len(matches) == 2
        assert any(match.is_missing for match in matches)
        assert any(match.is_spurious for match in matches)

    def test_empty_sides(self):
        assert ModeMatcher().match([], []) == []
        assert len(ModeMatcher().match([ModeEstimate(100.0, 1.0, 1.0)], [])) == 1


class TestDrumScorer:
    def test_a_signal_scores_one_against_itself(self, voice):
        audio = voice.render_hit(3.0)
        card = DrumScorer.for_fundamental(92.5, SR).score(audio, audio)
        assert card.total == pytest.approx(1.0, abs=1e-9)
        assert card.stft_loss == pytest.approx(0.0, abs=1e-9)

    def test_a_different_noise_seed_barely_moves_the_score(self, tom_params):
        """Noise is compared statistically. Two realizations of the same process
        must not read as a failure."""
        reference = DrumVoice(tom_params, SR, 64, seed=1).render_hit(3.0)
        generated = DrumVoice(tom_params, SR, 64, seed=99).render_hit(3.0)
        assert DrumScorer.for_fundamental(92.5, SR).score(
            reference, generated
        ).total > 0.95

    @pytest.mark.parametrize(
        "label,mutate,expected_component",
        [
            ("detuned", lambda p: setattr(
                p, "modes", [m.scaled(freq_ratio=2 ** (60 / 1200)) for m in p.modes]
            ), "mode_frequency"),
            ("over-damped", lambda p: setattr(
                p, "modes", [m.scaled(t60_ratio=0.4) for m in p.modes]
            ), "band_decay"),
            ("no tension", lambda p: setattr(p, "tension", Tension(k=0.0)), "glide"),
        ],
    )
    def test_catches_a_deliberate_error(
        self, tom_params, label, mutate, expected_component
    ):
        reference = DrumVoice(tom_params, SR, 64, seed=1).render_hit(3.0)
        broken = tom_params.copy()
        mutate(broken)
        generated = DrumVoice(broken, SR, 64, seed=1).render_hit(3.0)

        card = DrumScorer.for_fundamental(92.5, SR).score(reference, generated)
        assert card.total < 0.8, label
        assert card.by_name(expected_component).value < 0.5, label

    def test_the_score_is_monotone_in_the_error(self, tom_params):
        reference = DrumVoice(tom_params, SR, 64, seed=1).render_hit(3.0)
        scorer = DrumScorer.for_fundamental(92.5, SR)

        totals = []
        for cents in (5.0, 30.0, 100.0):
            broken = tom_params.copy()
            broken.modes = [
                m.scaled(freq_ratio=2 ** (cents / 1200)) for m in broken.modes
            ]
            generated = DrumVoice(broken, SR, 64, seed=1).render_hit(3.0)
            totals.append(scorer.score(reference, generated).total)
        assert totals[0] > totals[1] > totals[2]

    def test_soft_normalization_keeps_a_gradient_where_linear_saturates(self):
        """Linear is the architecture's contract: zero at the tolerance and
        nothing below. That is right for reading a card, and useless for an
        optimizer, which is why 'soft' exists."""
        reference = [ModeEstimate(100.0, 1.0, 1.0), ModeEstimate(200.0, 0.5, 1.0)]
        tolerance = Comparator.DEFAULT_TOLERANCES["mode_frequency_cents"]

        def frequency_score(normalization, cents):
            shifted = [
                ModeEstimate(mode.freq * 2 ** (cents / 1200), mode.amplitude, mode.t60)
                for mode in reference
            ]
            comparator = Comparator(normalization=normalization)
            return comparator.score_mode_frequency(
                ModeMatcher().match(reference, shifted)
            ).value

        # Three times the tolerance, well inside the matcher's 150-cent gate.
        error = 3 * tolerance
        assert frequency_score("linear", error) == pytest.approx(0.0)
        assert 0.0 < frequency_score("soft", error) < 0.2
        assert frequency_score("soft", error) < frequency_score("soft", tolerance)

    def test_aggregate_reports_the_worst_hit_not_the_mean(self, tom_params):
        reference = DrumVoice(tom_params, SR, 64, seed=1).render_hit(2.0)
        broken = tom_params.copy()
        broken.modes = [m.scaled(freq_ratio=2 ** (80 / 1200)) for m in broken.modes]
        bad = DrumVoice(broken, SR, 64, seed=1).render_hit(2.0)

        scorer = DrumScorer.for_fundamental(92.5, SR)
        cards = scorer.score_set([(reference, reference), (reference, bad)])
        aggregate = scorer.aggregate(cards)
        assert aggregate.total == pytest.approx(min(card.total for card in cards),
                                                abs=0.05)
        assert aggregate.total < np.mean([card.total for card in cards])

    def test_report_renders(self, tom_params):
        reference = DrumVoice(tom_params, SR, 64, seed=1).render_hit(2.0)
        card = DrumScorer.for_fundamental(92.5, SR).score(reference, reference)
        text = card.report()
        assert "TOTAL" in text and "mode_frequency" in text
        assert "STFT loss" in text

    def test_score_files_round_trip(self, tom_params, tmp_path):
        from drumsynth import AudioIO

        audio = DrumVoice(tom_params, SR, 64, seed=1).render_hit(2.0)
        AudioIO.write(tmp_path / "a.wav", audio, SR)
        AudioIO.write(tmp_path / "b.wav", audio, SR)
        card = DrumScorer.for_fundamental(92.5, SR).score_files(
            tmp_path / "a.wav", tmp_path / "b.wav"
        )
        assert card.total > 0.99


class TestAttribution:
    def test_names_the_mode_and_the_direction(self, tom_params):
        reference = DrumVoice(tom_params, SR, 64, seed=1).render_hit(3.0)
        broken = tom_params.copy()
        broken.modes[3] = broken.modes[3].scaled(t60_ratio=2.5)
        generated = DrumVoice(broken, SR, 64, seed=1).render_hit(3.0)

        scorer = DrumScorer.for_fundamental(92.5, SR)
        card, edits = scorer.score_and_attribute(reference, generated, broken)
        assert edits
        assert any(".t60" in edit.target for edit in edits)
        assert ScoreReport.suggestions(edits)

    def test_a_perfect_match_suggests_nothing(self, tom_params):
        audio = DrumVoice(tom_params, SR, 64, seed=1).render_hit(2.0)
        scorer = DrumScorer.for_fundamental(92.5, SR)
        _, edits = scorer.score_and_attribute(audio, audio, tom_params)
        assert edits == []

    def test_missing_tension_is_attributed_to_k(self, tom_params):
        reference = DrumVoice(tom_params, SR, 64, seed=1).render_hit(3.0)
        broken = tom_params.copy()
        broken.tension = Tension(k=0.0, tau=0.12)
        generated = DrumVoice(broken, SR, 64, seed=1).render_hit(3.0)

        scorer = DrumScorer.for_fundamental(92.5, SR)
        _, edits = scorer.score_and_attribute(reference, generated, broken)
        tension_edits = [edit for edit in edits if edit.target == "tension.k"]
        assert tension_edits
        assert tension_edits[0].suggested > 0.0


class TestSTFTLoss:
    def test_zero_against_itself(self, synthetic_signal):
        loss = MultiResolutionSTFTLoss(SR)
        assert loss(synthetic_signal, synthetic_signal) == pytest.approx(0.0, abs=1e-9)

    def test_grows_with_the_error(self, synthetic_signal):
        loss = MultiResolutionSTFTLoss(SR)
        small = loss(synthetic_signal, synthetic_signal * 1.01)
        large = loss(synthetic_signal, synthetic_signal * 2.0)
        assert 0 < small < large

    def test_reports_per_resolution(self, synthetic_signal):
        per = MultiResolutionSTFTLoss(SR).per_resolution(
            synthetic_signal, synthetic_signal
        )
        assert set(per) == {256, 1024, 4096, 16384}


class TestNoiseIsNeverComparedSampleWise:
    def test_two_noise_realizations_have_similar_statistics(self):
        rng_a = np.random.default_rng(1).standard_normal(SR)
        rng_b = np.random.default_rng(2).standard_normal(SR)
        analyzer = NoiseAnalyzer(SR)
        a, b = analyzer.analyze(rng_a), analyzer.analyze(rng_b)
        for band in a.band_energy_db:
            assert abs(a.band_energy_db[band] - b.band_energy_db[band]) < 1.0
        # ... while being essentially uncorrelated sample-wise.
        assert abs(np.corrcoef(rng_a, rng_b)[0, 1]) < 0.05


class TestEnvelopeAnalyzer:
    def test_measures_a_known_rise(self):
        n = SR
        ramp = np.linspace(0.0, 1.0, int(0.02 * SR))
        envelope = np.concatenate([ramp, np.exp(-np.arange(n - len(ramp)) / (0.3 * SR))])
        signal = envelope * np.sin(2 * np.pi * 200 * np.arange(n) / SR)
        curve = EnvelopeAnalyzer(SR, 0.001).analyze(signal)
        assert curve.peak_time == pytest.approx(0.02, abs=0.004)
        assert curve.rise_10_90 == pytest.approx(0.016, abs=0.005)
