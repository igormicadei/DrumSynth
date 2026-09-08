"""Training data: quality gates, velocity calibration, sets and splits."""

from __future__ import annotations

import numpy as np
import pytest

from drumsynth import (
    AudioIO,
    DrumVoice,
    QualityChecker,
    Sample,
    SampleLibrary,
    SampleSet,
    VelocityCalibration,
)

SR = 44100


@pytest.fixture
def session(tmp_path, tom_params):
    """A synthetic recording session: one drum, six velocities, one WAV each.

    Renders through the synthesizer so the ground truth is known — the whole
    point of a synthetic session is that anything the checks report is either
    real or a bug in the checks.
    """
    rng = np.random.default_rng(0)
    for velocity in (40, 55, 70, 85, 100, 115):
        hit = DrumVoice(tom_params, SR, 64, seed=velocity).render_hit(
            3.0, amplitude=(velocity / 127.0) ** 1.6
        )
        padded = np.concatenate([np.zeros(500), hit])
        padded += 3e-5 * rng.standard_normal(len(padded))
        AudioIO.write(
            tmp_path / f"floor_tom_16_v{velocity}.wav", padded, SR, normalize=False
        )
    return tmp_path


class TestFilenameParsing:
    def test_default_grammar(self):
        sample = Sample.from_filename("floor_tom_16_v100_take2.wav")
        assert sample.drum == "floor_tom_16"
        assert sample.velocity == 100.0
        assert sample.take == 2

    def test_without_a_take(self):
        sample = Sample.from_filename("snare_v64.wav")
        assert (sample.drum, sample.velocity, sample.take) == ("snare", 64.0, 0)

    def test_unparseable_name_raises(self):
        with pytest.raises(ValueError, match="cannot parse"):
            Sample.from_filename("room_mic_bounce.wav")

    def test_metadata_round_trip(self):
        sample = Sample.from_filename("snare_v64.wav")
        sample.tuning = "session A"
        restored = Sample.from_dict(sample.to_dict())
        assert restored.tuning == "session A"
        assert restored.velocity == sample.velocity


class TestQuality:
    def test_a_clean_hit_is_usable(self, tom_params):
        hit = DrumVoice(tom_params, SR, 64, seed=1).render_hit(3.0, 0.5)
        padded = np.concatenate([np.zeros(500), hit])
        padded += 3e-5 * np.random.default_rng(0).standard_normal(len(padded))
        quality = QualityChecker(SR).check(padded)
        assert quality.is_usable
        assert quality.onset_count == 1

    def test_digital_silence_before_the_onset_is_reported(self, tom_params):
        """True digital zeros mean the file was edited, so the onset is where
        an editor put it rather than where the hit was."""
        hit = DrumVoice(tom_params, SR, 64, seed=1).render_hit(1.0, 0.5)
        assert QualityChecker(SR).check(
            np.concatenate([np.zeros(500), hit])
        ).has_pre_onset_silence
        assert not QualityChecker(SR).check(hit).has_pre_onset_silence

    def test_beating_between_close_modes_is_not_a_second_onset(self, tom_params):
        """A level-crossing detector reports every tom with a close pair as a
        double hit. Onsets are fast rises, not level crossings."""
        hit = DrumVoice(tom_params, SR, 64, seed=1).render_hit(3.0, 0.5)
        assert QualityChecker(SR).check(hit).onset_count == 1

    def test_two_hits_are_disqualifying(self, voice):
        two_hits = voice.render_sequence([(0.0, 0.6), (0.9, 0.8)], 3.0)
        quality = QualityChecker(SR).check(two_hits)
        assert quality.onset_count == 2
        assert not quality.is_usable

    def test_a_flam_reads_as_one_hit(self, voice):
        flam = voice.render_sequence([(0.0, 0.3), (0.025, 0.9)], 3.0)
        assert QualityChecker(SR).check(flam).onset_count == 1

    def test_truncation_is_detected_by_the_tail_slope(self, tom_params):
        """A file that reached its noise floor has a flat tail; one that was cut
        is still descending. The end LEVEL cannot tell them apart, because the
        floor estimate follows the truncation down."""
        hit = DrumVoice(tom_params, SR, 64, seed=1).render_hit(6.0, 0.5)
        hit += 2e-4 * np.random.default_rng(0).standard_normal(len(hit))

        full = QualityChecker(SR).check(hit)
        assert not full.tail_truncated
        assert full.tail_slope_db_s > -6.0  # settled into the noise

        cut = QualityChecker(SR).check(hit[: int(0.5 * SR)])
        assert cut.tail_truncated
        assert cut.tail_slope_db_s < -10.0  # still falling when the file ended
        assert any("truncated" in warning for warning in cut.warnings())

    def test_clipping_is_disqualifying(self, tom_params):
        hit = DrumVoice(tom_params, SR, 64, seed=1).render_hit(1.0, 1.0)
        clipped = np.clip(hit * 4.0, -1.0, 1.0)
        quality = QualityChecker(SR).check(clipped)
        assert quality.is_clipped
        assert not quality.is_usable

    def test_truncation_alone_is_not_disqualifying(self, tom_params):
        """It bounds usable_duration; a fit restricted to that window is still
        valid evidence. Clipping and a second hit are different — both corrupt
        the signal inside the window, and no restriction recovers them."""
        hit = DrumVoice(tom_params, SR, 64, seed=1).render_hit(3.0, 0.5)
        quality = QualityChecker(SR).check(hit[: int(0.5 * SR)])
        assert quality.tail_truncated
        assert quality.is_usable


class TestVelocityCalibration:
    def test_a_compressive_controller_curve_is_recovered(self, session):
        sample_set = SampleSet.from_directory(session, "floor_tom_16")
        calibration = sample_set.calibrate()
        assert calibration.is_monotone()
        assert calibration.normalized[0] == pytest.approx(0.0)
        assert calibration.normalized[-1] == pytest.approx(1.0)

        # The whole reason this class exists: equal label steps are NOT equal
        # energy steps, so the normalized scale is not the label scale.
        steps = np.diff(calibration.measured_db)
        assert steps[0] > steps[-1] * 1.5

    def test_normalized_velocity_lands_on_every_sample(self, session):
        sample_set = SampleSet.from_directory(session, "floor_tom_16")
        sample_set.calibrate()
        assert all(
            sample.velocity_normalized is not None for sample in sample_set
        )

    def test_a_non_monotone_session_is_flagged(self, session):
        """A harder hit that measured quieter is a mislabeled take, a moved mic
        or a retuned drum — never something to fit around."""
        sample_set = SampleSet.from_directory(session, "floor_tom_16")
        sample_set.load_all()
        # Swap two samples' labels, as a mislabeled take would.
        ordered = sample_set.by_velocity()
        ordered[1].velocity, ordered[4].velocity = (
            ordered[4].velocity,
            ordered[1].velocity,
        )
        calibration = VelocityCalibration("energy").fit(sample_set.samples)
        assert not calibration.is_monotone()
        assert calibration.inversions()
        assert "NOT MONOTONE" in calibration.report()

    def test_identity_mode_ignores_the_audio(self, session):
        sample_set = SampleSet.from_directory(session, "floor_tom_16")
        sample_set.load_all()
        calibration = VelocityCalibration("identity").fit(sample_set.samples)
        assert calibration.normalize(40.0) == pytest.approx(0.0)
        assert calibration.normalize(115.0) == pytest.approx(1.0)


class TestSampleSet:
    def test_one_set_is_one_drum(self):
        sample_set = SampleSet(drum="floor_tom_16")
        with pytest.raises(ValueError, match="One SampleSet per drum"):
            sample_set.add(Sample.from_filename("snare_v64.wav"))

    def test_reference_sample_is_not_the_loudest(self, session):
        """Picked for fit quality: best SNR and longest usable tail, normally
        mid-to-upper velocity. The loudest hit is the most nonlinear and the
        most likely to be clipped."""
        sample_set = SampleSet.from_directory(session, "floor_tom_16")
        sample_set.load_all()
        reference = sample_set.reference_sample()
        loudest = sample_set.by_velocity()[-1]
        assert reference.velocity < loudest.velocity
        assert reference.velocity > sample_set.velocity_range()[0]

    def test_split_holds_out_interior_velocities(self, session):
        """Holding out the endpoints tests extrapolation, which is a different
        and much easier-to-fail question."""
        sample_set = SampleSet.from_directory(session, "floor_tom_16")
        train, holdout = sample_set.split(0.34)
        assert len(holdout) >= 1
        low, high = sample_set.velocity_range()
        for sample in holdout:
            assert low < sample.velocity < high

    def test_explicit_holdout_labels(self, session):
        sample_set = SampleSet.from_directory(session, "floor_tom_16")
        train, holdout = sample_set.split([70.0])
        assert [sample.velocity for sample in holdout] == [70.0]
        assert 70.0 not in [sample.velocity for sample in train]

    def test_leave_one_out_covers_everything(self, session):
        sample_set = SampleSet.from_directory(session, "floor_tom_16")
        held = [held.velocity for _, held in sample_set.leave_one_out()]
        assert sorted(held) == sorted(sample.velocity for sample in sample_set)

    def test_validate_reports_a_clean_session_cleanly(self, session):
        sample_set = SampleSet.from_directory(session, "floor_tom_16")
        issues = sample_set.validate()
        assert not any("unusable" in issue for issue in issues)
        assert not any("NOT monotone" in issue for issue in issues)

    def test_validate_catches_a_short_library(self, session, tom_params):
        """The reference's slowest mode is t60 ~2.3 s. A set whose longest
        usable window is under that cannot fit it, and will not complain."""
        short = SampleSet(drum="floor_tom_16", sr=SR)
        for velocity in (60, 90):
            hit = DrumVoice(tom_params, SR, 64, seed=velocity).render_hit(0.4, 0.5)
            path = session / f"short_v{velocity}.wav"
            AudioIO.write(path, hit, SR, normalize=False)
            sample = Sample.from_filename(path)
            sample.drum = "floor_tom_16"
            short.add(sample)
        assert any("slowest decay" in issue for issue in short.validate())

    def test_manifest_round_trip(self, session, tmp_path):
        sample_set = SampleSet.from_directory(session, "floor_tom_16")
        sample_set.load_all()
        restored = SampleSet.from_manifest(
            sample_set.save_manifest(tmp_path / "manifest.json")
        )
        assert len(restored) == len(sample_set)
        assert restored.drum == sample_set.drum
        assert restored[0].quality is not None

    def test_audio_window_defaults_to_the_usable_region(self, session):
        sample_set = SampleSet.from_directory(session, "floor_tom_16")
        sample_set.load_all()
        sample = sample_set[0]
        assert len(sample.audio_window()) <= len(sample.audio)
        assert len(sample.audio_window()) == pytest.approx(
            sample.quality.usable_duration * SR, rel=0.01
        )


class TestSampleLibrary:
    def test_scan_groups_by_drum(self, session):
        library = SampleLibrary.scan(session)
        assert library.drums() == ["floor_tom_16"]
        assert len(library["floor_tom_16"]) == 6

    def test_unparseable_files_are_skipped(self, session):
        AudioIO.write(session / "room_mic_bounce.wav", np.zeros(1000), SR)
        assert SampleLibrary.scan(session).drums() == ["floor_tom_16"]

    def test_summary_renders(self, session):
        text = SampleLibrary.scan(session).summary()
        assert "floor_tom_16" in text and "coverage" in text

    def test_manifest_round_trip(self, session, tmp_path):
        library = SampleLibrary.scan(session).load_all()
        restored = SampleLibrary.load_manifest(
            library.save_manifest(tmp_path / "library.json")
        )
        assert restored.drums() == library.drums()
        assert len(restored["floor_tom_16"]) == 6
