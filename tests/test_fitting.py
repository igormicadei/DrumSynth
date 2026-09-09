"""The fitter: recovering a drum from its own sound.

Every test here works against a synthetic drum whose parameters are known
exactly, because that is the only way to ask whether a fit is RIGHT rather
than merely low-loss. A fit that matches a recording it was fitted to proves
nothing; a fit that puts the modes back where they were does.

The samples themselves are not in the repository (§ data/, 3.5 GB of WAV), so
these are the strongest claims that can be made in CI.
"""

from __future__ import annotations

import numpy as np
import pytest

from drumsynth import DrumPresets, DrumVoice, Mode, NoiseBand, Tension
from drumsynth.core.constants import Cents
from drumsynth.synth.params import DrumParams
from drumsynth.fitting import (
    BatchLoss,
    Device,
    DeviceChoice,
    DrumTrainer,
    ExcitationStage,
    ExcitationTiltModel,
    FitEvaluator,
    InspectionStage,
    JointStage,
    LinearVoiceBasis,
    LossBackend,
    ModalStage,
    SpectralTarget,
    TargetBuilder,
    TensionFit,
    TensionStage,
    TrainingSettings,
    VelocityCurve,
    VelocityCurveStage,
)
from drumsynth.fitting.backend import TorchBatchLoss
from drumsynth.fitting.objective import LevelMatch
from drumsynth.fitting.stages import ExcitationFit

SR = 44100
SECONDS = 1.2


# =============================================================================
# The drum every test fits
# =============================================================================


def known_drum(k: float = 0.0, n_modes: int = 8) -> DrumParams:
    """A small drum with exactly known parameters. Few modes on purpose: these
    tests are about whether the machinery recovers what is there, and eight
    modes make that visible without costing a minute a test."""
    f0 = 140.0
    ratios = [1.0, 1.59, 2.14, 2.30, 2.65, 3.16, 3.60, 4.15][:n_modes]
    modes = [
        Mode(f0 * ratio, 1.0 / (1.0 + (f0 * ratio / 800.0) ** 1.2),
             1.6 * (1.0 / ratio) ** 0.7)
        for ratio in ratios
    ]
    return DrumParams(
        modes=modes,
        noise=[NoiseBand(200, 900, 0.03, 0.06), NoiseBand(900, 4000, 0.012, 0.04)],
        tension=Tension(k=k, tau=0.09),
        output_gain=1.0,
        name="known",
    )


def render(params: DrumParams, seconds: float = SECONDS, seed: int = 0,
           amplitude: float = 1.0) -> np.ndarray:
    return DrumVoice(params, SR, 64, seed=seed).render_hit(seconds, amplitude)


def velocity_layers(params: DrumParams, velocities=(35, 70, 105, 127)):
    """The same drum struck at several strengths, the way §7.1 says velocity
    works: a shorter contact (so a brighter excitation) and a harder strike.
    Nothing else moves."""
    hits = []
    for velocity in velocities:
        vn = velocity / 127.0
        contact = 0.0022 * max(vn, 0.05) ** -0.45
        cut = 1.0 / (np.pi * contact)
        strength = vn ** 1.3
        shaped = DrumParams(
            modes=[
                Mode(m.f_static, m.gain * strength / (1.0 + (m.f_static / cut) ** 2),
                     m.t60)
                for m in params.modes
            ],
            noise=[NoiseBand(b.f_low, b.f_high, b.level * strength, b.t60)
                   for b in params.noise],
            tension=params.tension,
            output_gain=1.0,
        )
        hits.append((float(velocity), render(shaped, seed=int(velocity))))
    return hits


@pytest.fixture(scope="module")
def layers():
    return velocity_layers(known_drum())


@pytest.fixture(scope="module")
def target(layers):
    return TargetBuilder(SR, SECONDS).from_audio("known", layers)


# =============================================================================
# The objective
# =============================================================================


class TestSpectralTarget:
    def test_identical_audio_scores_near_zero(self):
        audio = render(known_drum())
        assert SpectralTarget(audio, SR).distance(audio) == pytest.approx(0.0, abs=1e-9)

    def test_noise_seed_does_not_dominate(self):
        """§6.5: two realizations of the same noise process are uncorrelated, so
        a bin-by-bin loss compares them sample by sample and reports a large
        distance for identical parameters. Band aggregation is what makes the
        loss a measure of the DRUM rather than of the seed. Measured at 11.7 dB
        bin-wise on this synthesizer; the band loss has to be far below that or
        the optimizer spends its budget chasing one noise realization."""
        params = known_drum()
        first, second = render(params, seed=1), render(params, seed=2)
        assert SpectralTarget(first, SR).distance(second) < 3.0

    def test_a_wrong_drum_scores_far_worse_than_a_reseed(self):
        params = known_drum()
        target = SpectralTarget(render(params, seed=1), SR)
        floor = target.distance(render(params, seed=2))
        detuned = DrumParams(
            modes=[Mode(m.f_static * 1.25, m.gain, m.t60) for m in params.modes],
            noise=params.noise, tension=params.tension,
        )
        assert target.distance(render(detuned, seed=2)) > floor + 3.0


class TestLinearVoiceBasis:
    @staticmethod
    def _relative_error(rebuilt: np.ndarray, direct: np.ndarray) -> float:
        return float(np.sqrt(np.mean((rebuilt - direct) ** 2))
                     / np.sqrt(np.mean(direct**2)))

    def test_matrix_product_reproduces_the_engine(self):
        """The render is linear in the gains and the noise levels, which is what
        lets a fit evaluate thousands of candidates as one matrix product
        instead of thousands of renders.

        Not bit-exact: each noise band is rendered on its own to build the
        basis, so the bands draw from the generator in a different order than
        they do when the voice renders them together. The modal part carries
        the drum and it is the modal part that has to match."""
        params = known_drum()
        n = int(SECONDS * SR)
        basis = LinearVoiceBasis.build(params, n, SR, 64)
        gains = np.array([m.gain for m in params.modes])
        levels = np.array([b.level for b in params.noise])

        rebuilt = basis.render(gains, levels, params.output_gain)
        direct = render(params, seconds=n / SR, seed=0)
        assert rebuilt.shape == direct.shape
        assert self._relative_error(rebuilt, direct) < 0.02

    def test_holds_with_the_glide_on(self):
        """The trajectory is captured from a real render, so the basis rings at
        the frequencies the engine would — glide included."""
        params = known_drum(k=0.12)
        n = int(SECONDS * SR)
        basis = LinearVoiceBasis.build(params, n, SR, 64)
        gains = np.array([m.gain for m in params.modes])
        levels = np.array([b.level for b in params.noise])
        direct = render(params, seconds=n / SR, seed=0)
        assert self._relative_error(
            basis.render(gains, levels, 1.0), direct) < 0.05


# =============================================================================
# Stage 1 — measurement, not search
# =============================================================================


class TestModalStage:
    @staticmethod
    @pytest.fixture(scope="class")
    def fit():
        params = known_drum()
        return ModalStage(SR, max_modes=14).run(render(params), render(params))

    def test_frequencies_land_on_the_real_partials(self, fit):
        truth = np.array([m.f_static for m in known_drum().modes])
        found = np.array(sorted(m.f_static for m in fit.modes))
        nearest = np.array([found[np.argmin(np.abs(found - f))] for f in truth])
        cents = np.abs(Cents.between(nearest, truth))
        # ESPRIT on a drum at rest; a quarter-tone is 50 cents.
        assert np.median(cents) < 25.0

    def test_decays_are_in_the_right_order_of_magnitude(self, fit):
        truth = {m.f_static: m.t60 for m in known_drum().modes}
        errors = []
        for mode in fit.modes:
            closest = min(truth, key=lambda f: abs(f - mode.f_static))
            errors.append(abs(mode.t60 - truth[closest]) / truth[closest])
        assert np.median(errors) < 0.45

    def test_does_not_fit_tension(self, fit):
        """Stage 1 measures the drum at rest. k depends on the strike energy,
        which stage 2 has not established yet, so fitting it here would be
        fitting it against an arbitrary scale."""
        assert fit.tension.k == 0.0


# =============================================================================
# Stage 2 — excitation
# =============================================================================


class TestExcitationTiltModel:
    def test_shorter_contact_is_brighter(self):
        freqs = np.array([100.0, 1000.0, 5000.0])
        short = ExcitationTiltModel.tilt(freqs, 0.0005)
        long = ExcitationTiltModel.tilt(freqs, 0.004)
        assert short[-1] / short[0] > long[-1] / long[0]

    def test_brightness_reads_as_a_slope(self):
        """A blunt companion to the contact-time model: fit a line through the
        gains against log frequency. A shorter contact tilts it upward."""
        freqs = np.geomspace(100.0, 8000.0, 12)
        short = ExcitationTiltModel.brightness_db(
            freqs, ExcitationTiltModel.tilt(freqs, 0.0005))
        long = ExcitationTiltModel.brightness_db(
            freqs, ExcitationTiltModel.tilt(freqs, 0.004))
        assert short > long


class TestExcitationStage:
    def test_recovers_gains_it_was_given(self):
        """With the modes correct, the gain fit is a linear problem and should
        get close. This is the number every later stage inherits."""
        params = known_drum()
        audio = render(params)
        modal = ModalStage(SR, max_modes=14).run(audio, audio)
        fit = ExcitationStage(SR, 64).run(
            modal, audio, velocity=100.0, velocity_normalized=1.0,
            noise_bands=list(params.noise),
        )
        assert fit.loss < 3.0
        assert fit.gains.shape == (len(modal.modes),)
        assert np.all(fit.gains >= 0.0)


# =============================================================================
# Stage 2b — the glide
# =============================================================================


class TestTensionStage:
    @staticmethod
    @pytest.fixture(scope="class")
    def setup():
        params = known_drum(k=0.14)
        audio = render(params)
        modal = ModalStage(SR, max_modes=14).run(render(known_drum()), audio)
        fit = ExcitationStage(SR, 64).run(
            modal, audio, 100.0, 1.0, list(params.noise))
        return modal, fit, audio

    def test_peak_energy_is_what_k_multiplies(self, setup):
        modal, fit, audio = setup
        peak = TensionStage(SR, 64).peak_energy(modal.modes, fit.gains, len(audio))
        assert peak > 0
        # It is close to the summed squared gains, but not equal — the bank is
        # struck over a control block, not instantaneously.
        assert peak == pytest.approx(float(np.sum(fit.gains**2)), rel=0.5)

    def test_finds_a_glide_that_is_there(self, setup):
        modal, fit, audio = setup
        result = TensionStage(SR, 64).fit_layer(
            modal.modes, fit.gains, fit.output_gain, audio)
        assert result.is_evidence
        assert result.improvement > 0.05

    def test_reports_no_glide_when_there_is_none(self):
        params = known_drum(k=0.0)
        audio = render(params)
        modal = ModalStage(SR, max_modes=14).run(audio, audio)
        fit = ExcitationStage(SR, 64).run(
            modal, audio, 100.0, 1.0, list(params.noise))
        result = TensionStage(SR, 64).fit_layer(
            modal.modes, fit.gains, fit.output_gain, audio)
        assert not result.is_evidence

    def test_the_grid_cannot_propose_an_octave(self):
        """`Tension.max_ratio` allows a doubling, and a search left to itself
        will take it: a huge k with a very short tau pins every mode at the
        clamp for a few milliseconds, smearing the attack across the coarse STFT
        frames. It scores well and is not a drum."""
        assert TensionStage.MAX_RATIO_SPAN <= 0.35

    def test_unmeasurable_is_not_zero(self):
        """"This layer says nothing" and "this drum does not glide" are
        different claims. Only the second is a measurement."""
        assert not TensionFit.unmeasurable().is_evidence
        assert TensionFit.unmeasurable().improvement == float("-inf")


# =============================================================================
# Stage 3 — the experiment
# =============================================================================


def _fit(velocity, vn, amplitude, contact, brightness) -> ExcitationFit:
    return ExcitationFit(
        velocity=velocity, velocity_normalized=vn, amplitude=amplitude,
        contact_time=contact, gains=np.array([amplitude]), levels=np.array([0.0]),
        output_gain=1.0, loss=1.0, brightness_db=brightness,
    )


class TestInspectionStage:
    def test_a_physical_table_passes(self):
        fits = [
            _fit(30, 0.0, 0.2, 0.0030, -44.0),
            _fit(60, 0.33, 0.5, 0.0022, -41.0),
            _fit(95, 0.66, 0.8, 0.0016, -38.0),
            _fit(127, 1.0, 1.0, 0.0011, -35.0),
        ]
        result = InspectionStage().run(fits, [0.10, 0.11, 0.105, 0.10])
        assert result.passed, result.findings

    def test_darkening_with_velocity_fails(self):
        """Harder hits are brighter. The reverse is the mechanism backwards, and
        no amount of per-velocity fitting makes it right."""
        fits = [
            _fit(30, 0.0, 0.2, 0.0011, -35.0),
            _fit(60, 0.33, 0.5, 0.0016, -38.0),
            _fit(95, 0.66, 0.8, 0.0022, -41.0),
            _fit(127, 1.0, 1.0, 0.0030, -44.0),
        ]
        assert not InspectionStage().run(fits, [0.1] * 4).passed

    def test_non_monotone_amplitude_is_flagged_as_calibration(self):
        fits = [
            _fit(30, 0.0, 0.9, 0.0030, -44.0),
            _fit(60, 0.33, 0.3, 0.0022, -41.0),
            _fit(95, 0.66, 0.8, 0.0016, -38.0),
            _fit(127, 1.0, 0.4, 0.0011, -35.0),
        ]
        result = InspectionStage().run(fits, [0.1] * 4)
        assert not result.amplitude_monotone
        assert "calibration" in result.verdict

    def test_k_varying_as_one_over_amplitude_squared_blames_the_amplitude(self):
        """§8.1 says a velocity-dependent k means the energy feedback is wrong.
        It only means that once a measurement error is ruled out: k is fitted
        against each layer's bank energy, which goes as amplitude squared, so an
        amplitude that is off by a factor comes back as a k off by its square.
        When `k * amplitude^2` is constant the mechanism is fine."""
        fits = [
            _fit(30, 0.0, 0.2, 0.0030, -44.0),
            _fit(60, 0.33, 0.5, 0.0022, -41.0),
            _fit(95, 0.66, 0.8, 0.0016, -38.0),
            _fit(127, 1.0, 1.0, 0.0011, -35.0),
        ]
        constant = 0.10
        per_layer = [constant / fit.amplitude**2 for fit in fits]
        result = InspectionStage().run(fits, per_layer)
        assert result.tension_invariant
        assert any("ENERGY FEEDBACK IS FINE" in f for f in result.findings)

    def test_nan_layers_are_dropped_not_counted_as_zero(self):
        fits = [
            _fit(30, 0.0, 0.2, 0.0030, -44.0),
            _fit(60, 0.33, 0.5, 0.0022, -41.0),
            _fit(95, 0.66, 0.8, 0.0016, -38.0),
            _fit(127, 1.0, 1.0, 0.0011, -35.0),
        ]
        result = InspectionStage().run(
            fits, [float("nan"), float("nan"), 0.10, 0.10])
        assert result.tension_invariant
        assert any("could not run" in f for f in result.findings)


# =============================================================================
# Stage 4 — velocity becomes continuous
# =============================================================================


class TestVelocityCurve:
    def test_softest_layer_is_not_silent(self):
        """A power law through velocity_normalized puts the softest layer at
        exactly zero, and a drum that makes no sound at its lowest sampled
        velocity is a bug, not a model."""
        curve = VelocityCurve(
            amplitude_scale=1.0, amplitude_exponent=1.3,
            contact_scale=0.002, contact_exponent=-0.4,
            level_scale=np.array([0.02]), level_exponent=1.2,
        )
        assert curve.strength(0.0) >= VelocityCurve.MIN_STRENGTH
        assert curve.strength(0.0) < curve.strength(1.0)

    def test_fits_a_power_law_it_was_given(self):
        fits = [
            _fit(30, 0.0, 0.12, 0.0045, -44.0),
            _fit(60, 0.33, 0.31, 0.0031, -41.0),
            _fit(95, 0.66, 0.62, 0.0024, -38.0),
            _fit(127, 1.0, 1.00, 0.0020, -35.0),
        ]
        curve = VelocityCurveStage().run(fits)
        assert curve.contact_exponent < 0.0        # shorter with velocity
        assert curve.amplitude_exponent > 0.0      # harder with velocity
        assert curve.contact_time(1.0) < curve.contact_time(0.0)
        assert curve.amplitude(1.0) > curve.amplitude(0.0)


# =============================================================================
# A manifest and its audio are separate things
# =============================================================================


class TestMissingAudio:
    @staticmethod
    def _set(tmp_path, present: int, named: int):
        """A SampleSet naming `named` files of which only `present` exist."""
        from drumsynth.core.audio_io import AudioIO
        from drumsynth.samples.sample import Sample
        from drumsynth.samples.sample_set import SampleSet

        params = known_drum()
        sample_set = SampleSet(drum="partial", sr=SR)
        for index in range(named):
            path = tmp_path / f"hit{index}.wav"
            if index < present:
                AudioIO.write(path, render(params, seconds=0.4,
                                           seed=index, amplitude=0.3 + index * 0.2),
                              SR)
            sample_set.add(Sample(
                path=str(path), drum="partial",
                velocity=float(20 + index * 25), sr=SR))
        return sample_set

    def test_one_absent_wav_does_not_kill_the_load(self, tmp_path):
        """The manifests are committed and 3.5 GB of WAV is not, so a partial
        checkout is ordinary. Raising on the first missing file turns "three of
        104 are missing" into a stack trace and no fit at all."""
        sample_set = self._set(tmp_path, present=3, named=5)
        sample_set.load_all(skip_missing=True)
        assert len(sample_set) == 3
        assert len(sample_set.missing) == 2

    def test_it_still_raises_when_not_asked_to_skip(self, tmp_path):
        sample_set = self._set(tmp_path, present=1, named=3)
        with pytest.raises(FileNotFoundError):
            sample_set.load_all()

    def test_the_target_says_what_was_left_out(self, tmp_path):
        sample_set = self._set(tmp_path, present=4, named=6)
        target = TargetBuilder(SR, 0.4).from_sample_set(sample_set)
        assert len(target.layers) == 4
        assert any("not on disk" in w for w in target.warnings)

    def test_too_few_layers_is_called_out(self, tmp_path):
        """§8.2: never fit to a single hit. Per-velocity fitting always
        succeeds, so one layer produces a confident, meaningless answer."""
        sample_set = self._set(tmp_path, present=1, named=4)
        target = TargetBuilder(SR, 0.4).from_sample_set(sample_set)
        assert any("§8.2" in w for w in target.warnings)


# =============================================================================
# Where stage 5 runs
# =============================================================================


class TestDeviceChoice:
    def test_asking_for_cuda_that_is_not_there_is_an_error(self):
        """Silently falling back is the wrong answer for an explicit request:
        someone who picked the GPU wants to know it did not happen. "auto" is
        the setting that falls back."""
        if DeviceChoice.cuda_detail() is not None:
            pytest.skip("this machine has CUDA")
        with pytest.raises(RuntimeError, match="CUDA"):
            DeviceChoice.resolve(DeviceChoice.CUDA)

    def test_auto_always_resolves(self):
        device = DeviceChoice.resolve(DeviceChoice.AUTO)
        assert device.kind in ("cpu", "cuda")
        assert device.backend in ("numpy", "torch")
        assert device.detail

    def test_cpu_is_the_numpy_backend(self):
        assert DeviceChoice.resolve(DeviceChoice.CPU).backend == "numpy"

    def test_the_ui_is_told_what_is_actually_available(self):
        values = [value for value, _ in DeviceChoice.options()]
        assert values == ["auto", "cuda", "cpu"]
        labels = dict(DeviceChoice.options())
        assert ("not available" in labels["cuda"]) == (
            DeviceChoice.cuda_detail() is None)


@pytest.fixture(scope="module")
def batch_case():
    """A basis, a target, and a population of candidates to score against it."""
    params = known_drum()
    n = int(SECONDS * SR)
    reference = render(params, seconds=SECONDS, seed=3)
    basis = LinearVoiceBasis.build(params, n, SR, 64)
    target = SpectralTarget(reference, SR)

    rng = np.random.default_rng(0)
    gains = np.abs(rng.normal(0.5, 0.3, (6, len(params.modes))))
    levels = np.abs(rng.normal(0.02, 0.01, (6, len(params.noise))))
    return basis, target, reference, gains, levels


def _one_at_a_time(basis, target, reference, gains, levels) -> np.ndarray:
    out = []
    for row_gains, row_levels in zip(gains, levels):
        candidate = basis.render(row_gains, row_levels, 1.0)
        out.append(target.distance(
            candidate * LevelMatch.match_rms(candidate, reference)))
    return np.array(out)


def _flat_curve(fit) -> VelocityCurve:
    """A curve through one fitted layer. Stage 4 needs several velocities to
    fit exponents; these tests only need something for stage 5 to start from."""
    return VelocityCurve(
        amplitude_scale=float(np.linalg.norm(fit.gains)) or 1.0,
        amplitude_exponent=1.0,
        contact_scale=float(fit.contact_time),
        contact_exponent=-0.3,
        level_scale=np.asarray(fit.levels, dtype=float),
        level_exponent=1.2,
    )


class TestBatchLoss:
    def test_the_batch_is_the_same_number_as_one_at_a_time(self, batch_case):
        """The whole point of the backend is speed, not a different answer.
        If these diverge, every baseline in TRAINING.md stops being comparable
        with what stage 5 reports."""
        basis, target, reference, gains, levels = batch_case
        batched = BatchLoss(basis, target, reference).losses(gains, levels)
        assert batched == pytest.approx(
            _one_at_a_time(*batch_case), abs=1e-9)

    def test_chunking_does_not_change_the_answer(self, batch_case):
        basis, target, reference, gains, levels = batch_case
        small = BatchLoss(basis, target, reference, chunk=1).losses(gains, levels)
        large = BatchLoss(basis, target, reference, chunk=64).losses(gains, levels)
        assert small == pytest.approx(large, abs=1e-9)

    def test_a_silent_candidate_does_not_divide_by_zero(self, batch_case):
        basis, target, reference, gains, levels = batch_case
        losses = BatchLoss(basis, target, reference).losses(
            np.zeros_like(gains), np.zeros_like(levels))
        assert np.all(np.isfinite(losses))


class TestTorchBackend:
    @staticmethod
    @pytest.fixture(scope="class")
    def torch_device():
        if DeviceChoice.torch() is None:
            pytest.skip("torch is not installed")
        # Deliberately the torch backend on the CPU. It is not a device anyone
        # would pick, but it is the same code path CUDA takes with the dtype
        # and the transfers left in, which is the part worth testing where
        # there is no GPU.
        return Device("cpu", "torch", "torch on the CPU, for testing")

    def test_it_agrees_with_numpy(self, batch_case, torch_device):
        basis, target, reference, gains, levels = batch_case
        torched = TorchBatchLoss(
            basis, target, reference, torch_device).losses(gains, levels)
        assert torched == pytest.approx(
            _one_at_a_time(*batch_case), abs=1e-9)

    def test_the_backend_factory_picks_it_up(self, batch_case, torch_device):
        basis, target, reference, gains, levels = batch_case
        built = LossBackend.build(basis, target, reference, torch_device)
        assert isinstance(built, TorchBatchLoss)
        assert isinstance(
            LossBackend.build(basis, target, reference,
                              DeviceChoice.resolve("cpu")), BatchLoss)

    def test_stage_5_runs_on_it(self, batch_case, torch_device):
        params = known_drum()
        audio = render(params)
        modal = ModalStage(SR, max_modes=12).run(audio, audio)
        fit = ExcitationStage(SR, 64).run(
            modal, audio, 100.0, 1.0, list(params.noise))
        curve = _flat_curve(fit)
        shape = fit.gains / (np.linalg.norm(fit.gains) or 1.0)

        refined, generations = JointStage(SR, 64).run(
            modal, shape, curve, [(1.0, audio)], list(params.noise),
            generations=2, population=4, device=torch_device,
        )
        assert generations
        assert np.isfinite(refined.loss)


class TestStageFiveUsesNoProcesses:
    def test_the_population_is_evaluated_in_one_call(self):
        """The regression this pins is a real crash, not a slowdown.

        scipy's `workers=N` puts each candidate in its own process, and the
        objective is a closure: under `spawn` — which is what Windows uses —
        pickling it raises `Can't get local object`. The fix is not a picklable
        objective but a vectorized one, so nothing is ever sent to a process.
        """
        import multiprocessing

        params = known_drum()
        audio = render(params, seconds=0.6)
        modal = ModalStage(SR, max_modes=10).run(audio, audio)
        fit = ExcitationStage(SR, 64).run(
            modal, audio, 100.0, 1.0, list(params.noise), seconds=0.6)
        curve = _flat_curve(fit)
        shape = fit.gains / (np.linalg.norm(fit.gains) or 1.0)

        original = multiprocessing.Pool

        def forbidden(*args, **kwargs):
            raise AssertionError(
                "stage 5 started a worker pool; the objective is a closure and "
                "this is exactly the crash on Windows")

        multiprocessing.Pool = forbidden
        try:
            _, generations = JointStage(SR, 64).run(
                modal, shape, curve, [(1.0, audio[: int(0.6 * SR)])],
                list(params.noise), generations=2, population=4,
            )
        finally:
            multiprocessing.Pool = original
        assert generations


# =============================================================================
# The wire between the fit and the UI
# =============================================================================


class TestWorkerSerialization:
    def test_an_unmeasurable_k_survives_the_pipe_as_null(self):
        """A NaN k is meaningful: the layer's glide was too small to measure.
        Python's json will happily write a bare `NaN`, which is not JSON and
        which a reader cannot distinguish from a number until it tries to use
        it. Send null, which every reader can test."""
        from drumsynth.fitting.worker import FitWorker

        plain = FitWorker._plain(
            {"per_layer": [0.12, float("nan"), np.float64("nan"),
                           np.float64(0.13), float("inf")]}
        )
        assert plain["per_layer"] == [0.12, None, None, pytest.approx(0.13), None]

    def test_numpy_does_not_reach_the_pipe(self):
        from drumsynth.fitting.worker import FitWorker

        plain = FitWorker._plain(
            {"gains": np.array([1.0, 2.0]), "modes": np.int64(7)})
        assert plain == {"gains": [1.0, 2.0], "modes": 7}
        assert isinstance(plain["modes"], int)


# =============================================================================
# End to end
# =============================================================================


class TestDrumTrainer:
    @staticmethod
    @pytest.fixture(scope="class")
    def result(target):
        settings = TrainingSettings(
            seconds=SECONDS, max_layers=4, max_modes=14,
            generations=3, population=6, noise_bands=2,
        )
        return DrumTrainer(settings, SR).run(target)

    def test_produces_a_playable_drum_at_any_velocity(self, result):
        for vn in (0.0, 0.5, 1.0):
            params = result.params_at(vn)
            audio = render(params, seconds=0.5)
            assert np.all(np.isfinite(audio))
            assert np.max(np.abs(audio)) > 0.0

    def test_louder_velocity_is_louder(self, result):
        quiet = np.sqrt(np.mean(render(result.params_at(0.0), 0.5) ** 2))
        loud = np.sqrt(np.mean(render(result.params_at(1.0), 0.5) ** 2))
        assert loud > quiet

    def test_modes_do_not_move_with_velocity(self, result):
        """§7.1: velocity changes the excitation and nothing else. A fit that
        moves f_static between velocities has stopped modelling a drum."""
        first = [m.f_static for m in result.params_at(0.2).modes]
        second = [m.f_static for m in result.params_at(0.9).modes]
        assert first == pytest.approx(second)

    def test_recovers_the_fundamental(self, result):
        truth = min(m.f_static for m in known_drum().modes)
        found = min(m.f_static for m in result.modal.modes)
        assert abs(Cents.between(np.array([found]), truth)[0]) < 60.0

    def test_reports_progress_for_a_ui_to_follow(self, target):
        seen = []
        settings = TrainingSettings(
            seconds=SECONDS, max_layers=4, max_modes=10,
            generations=2, population=6, noise_bands=2,
        )
        DrumTrainer(settings, SR).run(target, progress=seen.append)
        phases = {record.get("phase") for record in seen}
        assert {"start", "stage1", "stage3", "stage4"} <= phases
        assert any(record.get("stage") == "tension" for record in seen)

    def test_generations_are_recorded_in_order(self, result):
        losses = [generation.index for generation in result.generations]
        assert losses == sorted(losses)

    def test_the_scorer_is_the_real_one(self, result, target):
        """The report the UI shows is the project's own DrumScorer, not a
        second opinion invented for the fitter. The aggregate is the WORST
        layer, per §8.2: a fit that nails one velocity and misses the rest has
        not fitted the drum."""
        aggregate, cards = FitEvaluator(SR).score(result, target)
        assert cards
        assert 0.0 <= aggregate.total <= 1.0
        # Worst-COMPONENT aggregation, not worst card: the aggregate takes the
        # weakest value of each of the 109 numbers across every layer, so it
        # can only sit at or below the best-scoring layer.
        assert aggregate.total <= min(card.total for _, card in cards) + 1e-9
