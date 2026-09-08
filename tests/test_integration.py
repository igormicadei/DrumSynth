"""The three subsystems together, on the path a user actually takes.

Synthesize -> write -> read -> analyze -> score -> attribute. If the round trip
loses something, it shows up here rather than in a unit test that never touched
a file.
"""

from __future__ import annotations

import numpy as np
import pytest

from drumsynth import (
    Analyzer,
    AudioIO,
    DrumKit,
    DrumParams,
    DrumPresets,
    DrumScorer,
    DrumSequencer,
    DrumVoice,
    SampleLibrary,
    Tension,
)

SR = 44100


class TestSynthMatchesTheReferenceMeasurements:
    """The preset is the architecture's claims turned into parameters. If the
    engine is wired correctly the render reproduces the measurements it came
    from — these are the numbers from docs/ARCHITECTURE.md §2."""

    @pytest.fixture(scope="class")
    @staticmethod
    def rendered():
        return DrumVoice(DrumPresets.tom(), SR, 64, seed=1).render_hit(4.0)

    @pytest.mark.parametrize(
        "band,expected_t60",
        [((40, 130), 2.33), ((230, 400), 0.75), ((400, 900), 0.55)],
    )
    def test_band_decay_matches_the_reference_table(self, rendered, band, expected_t60):
        from drumsynth import BandDecayAnalyzer

        analyzer = BandDecayAnalyzer(SR, bands=[band])
        measured = analyzer.analyze(rendered, -110.0)[0]
        assert measured.t60 == pytest.approx(expected_t60, rel=0.20)

    def test_glide_depth_matches_the_reference(self, rendered):
        from drumsynth import GlideAnalyzer

        track = GlideAnalyzer(SR, (60.0, 140.0)).analyze(rendered)
        assert track.depth_cents == pytest.approx(207.0, abs=60.0)
        assert track.f_asymptote == pytest.approx(92.5, rel=0.06)

    def test_the_attack_does_not_match_and_that_is_known(self, rendered):
        """Documented in docs/FINDINGS.md.

        The architecture claims the ~10 ms envelope peak emerges from summing
        modes at different frequencies. It cannot: every mode is impulse-excited
        at phase zero, so the sum is at its maximum on the first sample and can
        only fall. This test pins the disagreement so it cannot be mistaken for
        a regression, and will fail loudly if an excitation model is ever added.
        """
        from drumsynth import EnvelopeAnalyzer

        envelope = EnvelopeAnalyzer(SR, 0.001).analyze(rendered)
        assert envelope.peak_time < 0.005  # reference measured 0.0098 s
        assert envelope.rise_10_90 == pytest.approx(0.0, abs=0.001)


class TestFileRoundTrip:
    def test_write_read_score(self, tom_params, tmp_path):
        audio = DrumVoice(tom_params, SR, 64, seed=1).render_hit(3.0)
        path = AudioIO.write(tmp_path / "hit.wav", audio, SR)
        reloaded, sr = AudioIO.read(path, sr=SR)

        assert sr == SR
        assert len(reloaded) == pytest.approx(len(audio), rel=0.01)

        # 24-bit PCM plus peak normalization: the shapes must still match.
        card = DrumScorer.for_fundamental(92.5, SR).score(audio, reloaded)
        assert card.total > 0.95

    def test_params_survive_a_save_load_render_cycle(self, tom_params, tmp_path):
        path = tom_params.save(tmp_path / "tom.json")
        reloaded = DrumParams.load(path)
        a = DrumVoice(tom_params, SR, 64, seed=5).render_hit(1.0)
        b = DrumVoice(reloaded, SR, 64, seed=5).render_hit(1.0)
        assert np.allclose(a, b)


class TestFullPipeline:
    def test_synthetic_session_to_scorecard(self, tom_params, tmp_path):
        """Render a session, scan it as a library, validate it, then score the
        reference sample against a fresh render of the same parameters."""
        rng = np.random.default_rng(0)
        for velocity in (40, 60, 80, 100, 120):
            hit = DrumVoice(tom_params, SR, 64, seed=velocity).render_hit(
                3.5, amplitude=(velocity / 127.0) ** 1.6
            )
            padded = np.concatenate([np.zeros(500), hit])
            padded += 3e-5 * rng.standard_normal(len(padded))
            AudioIO.write(
                tmp_path / f"floor_tom_16_v{velocity}.wav", padded, SR, normalize=False
            )

        library = SampleLibrary.scan(tmp_path, sr=SR)
        assert library.drums() == ["floor_tom_16"]

        sample_set = library["floor_tom_16"]
        issues = sample_set.validate()
        assert not any("unusable" in issue for issue in issues)
        assert not any("NOT monotone" in issue for issue in issues)

        reference = sample_set.reference_sample()
        assert reference.velocity_normalized is not None

        generated = DrumVoice(tom_params, SR, 64, seed=int(reference.velocity)).render_hit(
            3.5, amplitude=(reference.velocity / 127.0) ** 1.6
        )
        card, edits = DrumScorer.for_fundamental(92.5, SR).score_and_attribute(
            reference.audio, generated, tom_params
        )
        assert card.total > 0.85
        assert isinstance(edits, list)
        assert "TOTAL" in card.report()

    def test_a_kit_pattern_renders(self):
        kit = DrumKit(SR)
        for name, params in DrumPresets.all(SR).items():
            kit.add(name, params, control_period=64, seed=1)

        pattern = DrumSequencer.from_grid(
            kit,
            {"kick": "x..x..x.", "snare_shell": "....x...", "rack_tom": "......o."},
            bpm=100.0,
        )
        audio = pattern.render(tail=1.0)
        assert np.all(np.isfinite(audio))
        assert np.max(np.abs(audio)) > 0.1

    def test_attribution_points_at_the_parameter_that_was_broken(self, tom_params):
        reference = DrumVoice(tom_params, SR, 64, seed=1).render_hit(3.0)

        broken = tom_params.copy()
        broken.tension = Tension(k=tom_params.tension.k * 0.35, tau=0.12)
        generated = DrumVoice(broken, SR, 64, seed=1).render_hit(3.0)

        card, edits = DrumScorer.for_fundamental(92.5, SR).score_and_attribute(
            reference, generated, broken
        )
        tension_edits = [edit for edit in edits if edit.target == "tension.k"]
        assert tension_edits, "the glide is wrong and nothing pointed at tension.k"
        # The suggestion should move k back toward the value that produced the
        # reference, not just report that something is off.
        suggested = tension_edits[0].suggested
        assert suggested > broken.tension.k
        assert suggested == pytest.approx(tom_params.tension.k, rel=0.6)


class TestGuardRails:
    def test_noise_is_never_compared_sample_wise(self, tom_params):
        """Two renders differing only in their noise realization must not read
        as a failure, or every fit will chase a stochastic target."""
        a = DrumVoice(tom_params, SR, 64, seed=11).render_hit(3.0)
        b = DrumVoice(tom_params, SR, 64, seed=22).render_hit(3.0)
        assert DrumScorer.for_fundamental(92.5, SR).score(a, b).total > 0.95

    def test_scoring_stops_above_the_reference_noise_floor(self, tom_params):
        """The floor is the REFERENCE's, applied to both sides.

        A synthesized hit decays into digital silence and can be measured 100 dB
        down; the reference flattens at its own floor 60 dB down. Measuring each
        over its own range makes the recording's limit look like the
        synthesizer's error — here, a 900-2000 Hz band decay reported 3.7x too
        long purely because the clean side could see further.
        """
        clean = DrumVoice(tom_params, SR, 64, seed=1).render_hit(3.0)
        noisy = clean + 1e-3 * np.random.default_rng(0).standard_normal(len(clean))

        card = DrumScorer.for_fundamental(92.5, SR).score(noisy, clean)
        ratios = [
            value
            for value in card.by_name("band_decay").detail.values()
            if isinstance(value, float)
        ]
        assert ratios
        assert all(0.9 < ratio < 1.1 for ratio in ratios), ratios
        assert card.by_name("band_decay").value > 0.9
        assert any("floored at the reference" in w for w in card.warnings)

    def test_stft_loss_is_reported_but_not_in_the_total(self, tom_params):
        """A single scalar says 'worse', not 'which of the 109 numbers'."""
        reference = DrumVoice(tom_params, SR, 64, seed=1).render_hit(2.0)
        broken = tom_params.copy()
        broken.modes = [m.scaled(freq_ratio=1.02) for m in broken.modes]
        generated = DrumVoice(broken, SR, 64, seed=1).render_hit(2.0)

        card = DrumScorer.for_fundamental(92.5, SR).score(reference, generated)
        assert card.stft_loss > 0.0
        assert all(component.name != "stft_loss" for component in card.components)

        from drumsynth import Comparator

        assert card.total == pytest.approx(
            Comparator().total(card.components), abs=1e-9
        )
