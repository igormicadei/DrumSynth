"""The five stages of ARCHITECTURE.md §8, each freezing what the last settled.

    1. f_static and t60, from one mid-velocity sample. Frozen permanently.
    2. Excitation only, per velocity, independently.
    3. Inspect the table. THIS is the experiment.
    4. A smooth 2-3 parameter curve through the per-velocity values.
    5. Joint refinement with the mapping in place.

Stage 1 is not a search. `ModalAnalyzer` measures f_static and t60 directly out
of the audio to within a cent and a few percent, and §6.1 says as much: the
descriptors ARE the parameters. Searching for numbers that can be measured is
wasted machinery and a worse answer.

The search only starts at stage 2, and even there the gains have a closed-form
starting point — modes sit at different frequencies, so total spectral power is
close to the sum of the per-mode parts and non-negative least squares solves
for them directly. What is left after that is a polish.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import differential_evolution, nnls

from ..core.constants import Audio, Cents, Decibels
from ..scoring.analyzer import Analyzer
from ..scoring.descriptors import SoundDescriptors
from ..scoring.bands import BandDecayAnalyzer
from ..scoring.glide import GlideAnalyzer
from ..scoring.prep import SignalPrep
from ..synth.presets import DampingCurve
from ..synth.params import DrumParams, Mode, NoiseBand, Tension
from ..synth.voice import DrumVoice
from .backend import Device, DeviceChoice, LossBackend
from .telemetry import TorchMemory
from .objective import LevelMatch, LinearVoiceBasis, SpectralTarget

Progress = Callable[[dict], None]


def _noop(_: dict) -> None:
    pass


# =============================================================================
# Stage 1 — the drum itself
# =============================================================================


class ExcitationTiltModel:
    """The gain tilt a finite contact time imposes on the excitation.

    §4: `contact_time` is the width of the force pulse, and a wider pulse is a
    lower-cut lowpass on what reaches the modes — short contact is bright, long
    contact is dull. That is the mechanism behind the whole velocity model, so
    the fit parameterizes the gains through it rather than treating 31 numbers
    as independent.
    """

    #: Contact times outside this are not a stick on a drum head (§4).
    RANGE = (0.0002, 0.005)

    @staticmethod
    def cutoff(contact_time: float) -> float:
        """First-order corner of a pulse that wide, in Hz."""
        return 1.0 / (np.pi * max(contact_time, 1e-6))

    @staticmethod
    def tilt(freqs: np.ndarray, contact_time: float) -> np.ndarray:
        cut = ExcitationTiltModel.cutoff(contact_time)
        return 1.0 / (1.0 + (np.asarray(freqs, dtype=float) / cut) ** 2)

    #: Below this a mode has been switched off by the gain solve rather than
    #: fitted quiet. Ten octaves under the softest gain any real fit produces.
    LIVE_GAIN: float = 1e-8

    #: Contact time assigned to the reference layer. The gain ratio between two
    #: layers determines their contact times only RELATIVE to each other, so one
    #: of them has to be pinned; the middle of §4's 0.3-3 ms range is the
    #: honest place to pin it.
    REFERENCE_CONTACT: float = 0.0015

    @staticmethod
    def relative_contact_time(
        freqs: np.ndarray,
        gains: np.ndarray,
        reference_gains: np.ndarray,
        reference_contact: float = REFERENCE_CONTACT,
    ) -> float:
        """Contact time of one layer, from its gain tilt RELATIVE to another.

        Fitting a lowpass to a layer's gains on their own does not work, and the
        reason is that the gains are the drum's own excitation shape TIMES the
        contact-time tilt. A single fit cannot separate them, so it hands the
        whole downward slope to contact time and pins it at the bound.

        The ratio between two layers cancels the shape, because the shape does
        not depend on velocity — that is what makes it the shape. What is left
        is the brightness difference, which is exactly what contact time is
        supposed to explain, and it is what stage 3 then tests for a trend.
        """
        freqs = np.asarray(freqs, dtype=float)
        ratio_db = Decibels.from_amplitude(
            np.maximum(gains, 1e-12) / np.maximum(reference_gains, 1e-12)
        )
        ratio_db = ratio_db - np.median(ratio_db)  # level is not brightness

        candidates = np.geomspace(*ExcitationTiltModel.RANGE, 128)
        reference_tilt = Decibels.from_amplitude(
            ExcitationTiltModel.tilt(freqs, reference_contact)
        )
        errors = []
        for contact in candidates:
            model = Decibels.from_amplitude(
                ExcitationTiltModel.tilt(freqs, contact)
            ) - reference_tilt
            model = model - np.median(model)
            errors.append(float(np.mean((ratio_db - model) ** 2)))
        return float(candidates[int(np.argmin(errors))])

    @staticmethod
    def brightness_db(freqs: np.ndarray, gains: np.ndarray) -> float:
        """Slope of gain against log frequency, in dB per decade.

        A blunt, assumption-free companion to `relative_contact_time`: if the
        contact-time model is right these two agree, and stage 3 says so when
        they do not.
        """
        freqs = np.asarray(freqs, dtype=float)
        gains = np.asarray(gains, dtype=float)

        # A mode the gain solve switched off is not a quiet mode, it is a mode
        # that is not there — and at the 1e-9 floor it reads as -180 dB, which
        # a straight line through log-frequency cannot ignore. Measured on a
        # real fit: three dead modes clustered at the bottom of the range turned
        # a genuine -40 dB/decade tilt into +70, and stage 3 read that as the
        # excitation getting BRIGHTER with frequency. Fit the line through the
        # modes that are actually sounding.
        alive = gains > ExcitationTiltModel.LIVE_GAIN
        if alive.sum() < 3:
            return 0.0
        freqs, gains = freqs[alive], gains[alive]
        return float(
            np.polyfit(np.log10(freqs), Decibels.from_amplitude(gains), 1)[0]
        )


@dataclass
class ModalFit:
    """What stage 1 established. Frozen from here on."""

    modes: list[Mode]
    tension: Tension
    descriptors: SoundDescriptors
    notes: list[str] = field(default_factory=list)
    damping_anchors: list[tuple[float, float]] = field(default_factory=list)

    def frozen_frequencies(self) -> np.ndarray:
        return np.array([mode.f_static for mode in self.modes])


class ModalStage:
    """Stage 1: measure f_static, t60 and the glide. Then freeze all of it.

    Three measurements, each taken where it is actually measurable — which is
    the one place this deviates from §8's "one mid-velocity reference sample",
    and the reason is in the definition of the parameter being measured.

    **f_static is the frequency AT REST**, with zero tension offset. A loud hit
    is the drum at its tightest, not at rest: the tension feedback holds every
    mode up to two semitones sharp through the first second, and a subspace
    method fitting a stationary exponential to that returns the average of a
    moving frequency. Measured on this synthesizer, mode frequencies come back
    63 cents out from a full-strength hit and 2.3 cents out from a soft one.
    So frequencies are measured from a SOFT layer, where the drum is at rest by
    construction.

    **t60 comes from band decays, not from per-mode damping.** Per-mode t60 out
    of the subspace fit lands ~56% off with outliers past 90%; a damping curve
    interpolated through the measured band decays lands at 13% with the worst
    at 14%. Band decays fit a straight line in dB over 60 dB of range with
    r² above 0.99, and §2.1 says the trend is most of the realism — so the
    robust measurement sets the law and the sharp one sets the frequencies.

    The glide is not fitted here at all — see `TensionStage` for why it has to
    wait for stage 2.
    """

    #: Modes quieter than this against the loudest are below what the analysis
    #: can separate from its own noise, and adding them adds only free
    #: parameters.
    MIN_MODE_DB: float = -55.0

    #: A membrane drum's partials. Above this the ear stops resolving them and
    #: §3.1 hands the region to the noise bank.
    MAX_MODE_HZ: float = 4000.0

    #: Which layer measures frequencies: this far up the velocity range. Soft
    #: enough that the glide has not lifted the modes, loud enough to be above
    #: the noise.
    FREQUENCY_LAYER_POSITION: float = 0.25

    #: Band fits flatter than this are noise, not decay, and are left out of
    #: the damping curve.
    MIN_BAND_R2: float = 0.4

    def __init__(self, sr: int = Audio.DEFAULT_SR, max_modes: int = 34) -> None:
        self.sr = int(sr)
        self.max_modes = int(max_modes)

    def run(self, frequency_audio: np.ndarray, damping_audio: np.ndarray,
            progress: Progress = _noop) -> ModalFit:
        """`frequency_audio` should be a soft hit, `damping_audio` a loud one."""
        progress({"stage": 1, "step": "measuring mode frequencies (soft hit)"})
        prep = SignalPrep(self.sr)
        soft = prep.prepare(frequency_audio)
        loud = prep.prepare(damping_audio)

        analyzer = Analyzer(self.sr)
        descriptors = analyzer.analyze(soft)
        notes: list[str] = []

        frequencies, amplitudes = self._frequencies_from(descriptors, notes)
        progress({"stage": 1, "step": f"{len(frequencies)} partials recovered"})

        progress({"stage": 1, "step": "fitting the damping curve (band decays)"})
        damping, anchors = self._damping_from(loud, notes)
        t60s = np.asarray(damping(frequencies), dtype=float)

        # The amplitudes ESPRIT returns are a by-product, not a measurement of
        # the excitation, and stage 2 replaces every one of them. They are
        # normalized to unit bank energy here anyway, because until stage 2 runs
        # they are the only gains the modes carry — and anything that builds a
        # tension trajectory from them gets `ratio = 1 + k * energy` evaluated
        # at an energy two orders of magnitude too large. Measured: raw ESPRIT
        # amplitudes summed to 211 against the true 4.8, which pinned the glide
        # at `Tension.max_ratio` (an octave) for the whole hit. A basis built on
        # that rings in the wrong place and no parameter can fix it.
        amplitudes = np.asarray(amplitudes, dtype=float)
        energy = float(np.sqrt(np.sum(amplitudes**2))) or 1.0
        amplitudes = amplitudes / energy

        modes = [
            Mode(float(freq), float(max(amp, 1e-9)), float(np.clip(t60, 0.005, 12.0)))
            for freq, amp, t60 in zip(frequencies, amplitudes, t60s)
        ]

        # Tension is NOT fitted here. `ratio = 1 + k * energy` means k is only
        # meaningful together with the absolute energy a strike puts into the
        # bank, and that is not known until stage 2 has fitted the gains. See
        # TensionStage.
        return ModalFit(modes=modes, tension=Tension(k=0.0, tau=0.12),
                        descriptors=descriptors, notes=notes,
                        damping_anchors=anchors)

    # -- frequencies ----------------------------------------------------------

    def _frequencies_from(self, descriptors: SoundDescriptors,
                          notes: list[str]) -> tuple[np.ndarray, np.ndarray]:
        estimates = [
            estimate
            for estimate in descriptors.modes
            if estimate.freq <= ModalStage.MAX_MODE_HZ and estimate.amplitude > 0
        ]
        if not estimates:
            raise ValueError(
                "modal analysis found no partials — check the sample is a drum "
                "hit and not silence or a truncated file"
            )

        loudest = max(estimate.amplitude for estimate in estimates)
        threshold = loudest * Decibels.to_amplitude(ModalStage.MIN_MODE_DB)
        survivors = [item for item in estimates if item.amplitude >= threshold]

        merged, splits = ModalStage._merge_close(survivors)
        if splits:
            notes.append(
                f"{splits} partial{'s' if splits != 1 else ''} had been split "
                f"into two or more estimates less than {ModalStage.MERGE_CENTS:.0f} "
                "cents apart and were merged. A subspace estimator does this when "
                "a mode is not quite a pure exponential, and each copy costs a "
                "slot the bank could have spent on a partial that is really there"
            )

        kept = sorted(merged, key=lambda item: -item[1])[: self.max_modes]
        kept.sort(key=lambda item: item[0])

        if len(kept) < 8:
            notes.append(
                f"only {len(kept)} partials rose above the noise. The modal bank "
                "will be thin and the tail will sound synthetic — a cleaner or "
                "longer reference is the fix, not more free parameters"
            )
        elif len(kept) < 20:
            notes.append(
                f"{len(kept)} partials recovered, against the 25-35 §3.1 "
                "assumes. Everything above the highest is the noise bank's job"
            )

        return (
            np.array([freq for freq, _ in kept]),
            np.array([amplitude for _, amplitude in kept]),
        )

    #: Two estimates closer than this are one partial the estimator split.
    #: Well under the 86 cents of the closest genuine pair in the test fixtures,
    #: and well over the 15-30 cents a split produces.
    MERGE_CENTS: float = 40.0

    @staticmethod
    def _merge_close(estimates) -> tuple[list[tuple[float, float]], int]:
        """Collapse runs of near-identical frequencies into one partial each.

        Measured on a real fit: four estimates at 92.97, 93.94, 94.80 and
        96.04 Hz — one partial, reported four times, spending four of the
        thirty-odd slots the bank has. Three of the four then came back from the
        gain solve at the 1e-9 floor, which is the solve saying the same thing.

        The survivor sits at the amplitude-weighted centroid and carries the
        summed amplitude, because the split shared one partial's energy out
        between the copies.
        """
        ordered = sorted(estimates, key=lambda item: item.freq)
        groups: list[list] = []
        for item in ordered:
            if groups and abs(Cents.between(
                    np.array([item.freq]), groups[-1][-1].freq)[0]
            ) < ModalStage.MERGE_CENTS:
                groups[-1].append(item)
            else:
                groups.append([item])

        out: list[tuple[float, float]] = []
        for group in groups:
            weights = np.array([item.amplitude for item in group], dtype=float)
            freqs = np.array([item.freq for item in group], dtype=float)
            total = float(weights.sum())
            centre = float(np.dot(freqs, weights) / total) if total > 0 else freqs[0]
            out.append((centre, total))
        return out, sum(1 for group in groups if len(group) > 1)

    # -- damping --------------------------------------------------------------

    def _damping_from(self, audio: np.ndarray,
                      notes: list[str]) -> tuple[DampingCurve, list[tuple[float, float]]]:
        floor = SignalPrep(self.sr).estimate_noise_floor(audio)
        bands = [
            band
            for band in BandDecayAnalyzer(self.sr).analyze(audio, floor)
            if band.is_valid and band.r_squared >= ModalStage.MIN_BAND_R2
        ]
        if len(bands) < 2:
            notes.append(
                "fewer than two bands gave a usable decay fit; falling back to "
                "the reference drum's damping curve, which is a guess about "
                "this drum rather than a measurement of it"
            )
            return DampingCurve(), list(DampingCurve.REFERENCE_ANCHORS)

        anchors = sorted(
            (float(np.sqrt(band.f_low * band.f_high)), float(band.t60))
            for band in bands
        )
        curve = DampingCurve(tuple(anchors))

        slope = np.polyfit(
            np.log([f for f, _ in anchors]), np.log([t for _, t in anchors]), 1
        )[0]
        if slope > -0.2:
            notes.append(
                f"t60 barely falls with frequency (log-log slope {slope:+.2f}, "
                "the reference measured about -1). Either the tail was cut "
                "before the low modes decayed, or the fit is in the noise floor"
            )
        return curve, anchors


# =============================================================================
# Stage 2 — excitation, per velocity
# =============================================================================


@dataclass
class ExcitationFit:
    """The excitation of ONE hit. ~35 numbers, with the hard part already done."""

    velocity: float
    velocity_normalized: float
    amplitude: float                      # overall strike strength
    contact_time: float                   # s, relative to the reference layer
    gains: np.ndarray                     # linear, per mode
    levels: np.ndarray                    # linear, per noise band
    output_gain: float
    loss: float
    brightness_db: float = 0.0            # gain slope per decade; stage 3 uses it
    iterations: int = 0

    def to_row(self) -> dict:
        return {
            "velocity": self.velocity,
            "velocity_normalized": self.velocity_normalized,
            "amplitude": self.amplitude,
            "contact_time_ms": self.contact_time * 1000.0,
            "brightness_db_per_decade": self.brightness_db,
            "noise_db": float(Decibels.from_amplitude(np.max(self.levels)))
            if len(self.levels) else -np.inf,
            "loss_db": self.loss,
        }


class ExcitationStage:
    """Stage 2: fit `gain[]`, noise `level[]` and `contact_time` for one hit.

    Two modes, and which one runs is the difference between a well-posed fit
    and a pile of numbers.

    **The reference layer fits every gain.** 31 free parameters, with a
    closed-form start from non-negative least squares on spectral power — modes
    sit at different frequencies, so total power is close to the sum of the
    per-mode parts — and coordinate descent to polish. Coordinate descent
    because each gain moves mostly its own region of the spectrum; the problem
    is nearly separable and exploiting that beats searching 31 dimensions blind.

    **Every other layer fits ONE**: contact time, with the per-mode shape frozen
    from the reference and the strike strength measured from the layer's own
    level. This is §4 taken literally — "at a single fixed velocity [contact_time] is absorbed into the
    fitted gain values; when velocity is added, contact_time is reintroduced and
    the gains are REFIT AGAINST IT".

    Letting every layer fit 31 free gains instead looks more general and is
    much worse. The loss is nearly flat in any individual gain, so many gain
    vectors score the same, and the contact time implied afterwards comes out
    as noise — measured on a synthetic reference with a known, strictly
    monotone contact-time law, free per-layer gains produced contact times of
    0.20, 0.77, 0.73, 1.48, 0.21, 0.20 ms. Stage 3 cannot test a mechanism
    against numbers like that, and stage 3 is the whole experiment.
    """

    #: Each sweep halves its search span. A wide first pass finds the right
    #: neighbourhood, narrow later ones settle in it, and the total evaluation
    #: count stays small.
    SWEEPS: int = 3
    PROBES: int = 5
    SPAN_DB: float = 10.0

    #: Grid over the one searched parameter of a non-reference layer.
    CONTACT_PROBES: int = 32

    def __init__(self, sr: int = Audio.DEFAULT_SR, control_period: int = 64) -> None:
        self.sr = int(sr)
        self.control_period = int(control_period)

    def run(
        self,
        modal: ModalFit,
        reference: np.ndarray,
        velocity: float,
        velocity_normalized: float,
        noise_bands: Sequence[NoiseBand],
        seconds: float | None = None,
        progress: Progress = _noop,
        free_gains: bool = True,
        initial_gains: np.ndarray | None = None,
        shape: np.ndarray | None = None,
        amplitude: float | None = None,
        output_gain: float | None = None,
    ) -> ExcitationFit:
        n_samples = len(reference) if seconds is None else int(seconds * self.sr)
        n_samples = min(n_samples, len(reference))
        reference = reference[:n_samples]
        target = SpectralTarget(reference, self.sr)
        freqs = np.array([mode.f_static for mode in modal.modes])

        # RELINEARIZE. The basis freezes a frequency trajectory, and that
        # trajectory comes from the bank's energy, which comes from the gains
        # the basis is about to be used to solve for. Building it once from a
        # unit-gain probe gets the glide completely wrong — the bank's energy
        # is then hundreds of times too large and the ratio pins at its ceiling
        # for the whole hit. Measured that way, even the TRUE excitation scored
        # 11.7 dB against a 1.1 dB floor.
        #
        # So: solve once with the tension off, where there is no trajectory to
        # get wrong, then rebuild the basis using those gains and solve again.
        gains = initial_gains
        levels = np.zeros(len(noise_bands))
        fixed_output_gain = output_gain
        output_gain, contact = 1.0, ExcitationTiltModel.REFERENCE_CONTACT
        best = float("inf")
        basis = None
        passes = 2 if modal.tension.is_active else 1

        for relinearization in range(passes):
            tension = (
                Tension(k=0.0, tau=modal.tension.tau)
                if relinearization == 0 and passes > 1
                else modal.tension
            )
            probe_gains = (
                gains if gains is not None
                else np.array([mode.gain for mode in modal.modes])
            )
            probe = DrumParams(
                modes=[
                    Mode(mode.f_static, float(max(gain, 1e-12)), mode.t60)
                    for mode, gain in zip(modal.modes, probe_gains)
                ],
                noise=list(noise_bands), tension=tension, output_gain=1.0,
            )
            basis = LinearVoiceBasis.build(
                probe, n_samples, self.sr, self.control_period
            )

            if shape is None:
                gains = self._initial_gains(basis, target)
            else:
                gains, contact = self._fit_contact_time(
                    basis, target, freqs, shape, float(amplitude or 1.0),
                    fixed_output_gain,
                )
            levels = self._initial_levels(basis, target, gains)
            output_gain = (
                fixed_output_gain
                if fixed_output_gain is not None
                else LevelMatch.match_rms(basis.render(gains, levels, 1.0), reference)
            )
            best = target.distance(basis.render(gains, levels, output_gain))

        progress({"stage": 2, "velocity": velocity, "iteration": 0, "loss": best})

        iterations = 0
        if free_gains and shape is None:
            for sweep in range(ExcitationStage.SWEEPS):
                span = ExcitationStage.SPAN_DB / (2**sweep)
                gains, levels, output_gain, best = self._sweep(
                    basis, target, reference, gains, levels, output_gain, best, span
                )
                iterations += 1
                progress({"stage": 2, "velocity": velocity,
                          "iteration": sweep + 1, "loss": best})
        elif shape is not None:
            levels, output_gain, best = self._sweep_levels(
                basis, target, reference, gains, levels, output_gain, best
            )
            iterations = 1

        return ExcitationFit(
            velocity=velocity,
            velocity_normalized=velocity_normalized,
            amplitude=float(np.sqrt(np.sum(gains**2))),
            contact_time=float(contact),
            gains=gains,
            levels=levels,
            output_gain=float(output_gain),
            loss=float(best),
            brightness_db=ExcitationTiltModel.brightness_db(freqs, gains),
            iterations=iterations,
        )

    # -- the two-parameter path -----------------------------------------------

    def _fit_contact_time(self, basis, target, freqs, shape, amplitude,
                          output_gain):
        """Contact time, with the shape frozen and the strike strength MEASURED.

        Amplitude is not searched, and the reason is that it cannot be. The
        spectral loss is evaluated after a level match, so scaling a candidate
        up and scaling `output_gain` down leaves it unchanged — the two are the
        same parameter seen twice, and a grid over both picks a winner at
        random. Fitted that way on a synthetic reference with a strictly
        increasing strike strength, the recovered amplitudes came out 0.63,
        1.00, 0.10, 0.02, 0.04, 0.03: anti-monotone noise.

        So `output_gain` is fixed once, from the reference layer, because it is
        a property of the drum rather than of a hit; and each layer's strike
        strength comes from its measured level relative to that reference,
        because that is what a louder recording of the same drum means. What is
        left to search is one number — the tilt — which is exactly what §4 says
        velocity changes.
        """
        contacts = np.geomspace(*ExcitationTiltModel.RANGE,
                                ExcitationStage.CONTACT_PROBES)
        zero_levels = np.zeros(basis.n_bands)
        gain = output_gain if output_gain is not None else 1.0

        best, best_loss = (shape.copy(), ExcitationTiltModel.REFERENCE_CONTACT), np.inf
        for contact in contacts:
            tilted = shape * ExcitationTiltModel.tilt(freqs, contact)
            norm = float(np.sqrt(np.sum(tilted**2))) or 1.0
            candidate = tilted / norm * amplitude
            loss = target.distance(basis.render(candidate, zero_levels, gain))
            if loss < best_loss:
                best_loss, best = loss, (candidate, float(contact))
        return best[0], best[1]

    def _sweep_levels(self, basis, target, reference, gains, levels,
                      output_gain, best):
        """Noise levels only. The modal part is already pinned by the shape."""
        probes = np.linspace(-9.0, 9.0, 7)
        for index in range(len(levels)):
            current = levels[index]
            for offset in probes:
                trial = levels.copy()
                trial[index] = max(current, 1e-9) * Decibels.to_amplitude(offset)
                loss = target.distance(basis.render(gains, trial, output_gain))
                if loss < best:
                    best, levels, current = loss, trial.copy(), trial[index]
        output_gain = LevelMatch.match_rms(basis.render(gains, levels, 1.0), reference)
        return levels, output_gain, min(best, target.distance(
            basis.render(gains, levels, output_gain)))

    # -- closed-form start ----------------------------------------------------

    @staticmethod
    def _initial_gains(basis: LinearVoiceBasis, target: SpectralTarget,
                       n_fft: int = 2048) -> np.ndarray:
        """Non-negative least squares on spectral power.

        Solves `power_ref ~= sum_i (g_i**2) * power_i`, which is exact to the
        extent the modes do not overlap in frequency — and membrane partials
        are inharmonic and well separated, so it is close.
        """
        mode_power = basis.mode_power(n_fft)
        if mode_power.size == 0:
            return np.zeros(basis.n_modes)
        reference_power = target.reference_power(n_fft)
        rows = min(mode_power.shape[1], reference_power.shape[0])
        try:
            squared, _ = nnls(mode_power[:, :rows].T, reference_power[:rows])
        except Exception:
            squared = np.ones(basis.n_modes)
        gains = np.sqrt(np.maximum(squared, 0.0))
        if not np.any(gains > 0):
            gains = np.ones(basis.n_modes)
        return gains

    @staticmethod
    def _initial_levels(basis: LinearVoiceBasis, target: SpectralTarget,
                        gains: np.ndarray) -> np.ndarray:
        """Noise levels from the residual the modes could not explain."""
        if basis.n_bands == 0:
            return np.zeros(0)
        residual = target.raw[: basis.n_samples] - gains @ basis.modal
        levels = []
        for row in basis.noise:
            scale = LevelMatch.best_gain(row, residual)
            levels.append(max(scale, 0.0))
        return np.array(levels)

    # -- polish ---------------------------------------------------------------

    def _sweep(self, basis, target, reference, gains, levels, output_gain, best,
               span: float):
        """One pass of coordinate descent over every gain and level, in dB."""
        probes = np.linspace(-span, span, ExcitationStage.PROBES)

        for index in range(len(gains)):
            current = gains[index]
            if current <= 0:
                continue
            trial = gains.copy()
            for offset in probes:
                trial[index] = current * Decibels.to_amplitude(offset)
                candidate = basis.render(trial, levels, output_gain)
                loss = target.distance(candidate)
                if loss < best:
                    best, gains = loss, trial.copy()
                    current = trial[index]
            gains[index] = current

        for index in range(len(levels)):
            current = levels[index]
            trial = levels.copy()
            for offset in probes:
                trial[index] = max(current, 1e-9) * Decibels.to_amplitude(offset)
                loss = target.distance(basis.render(gains, trial, output_gain))
                if loss < best:
                    best, levels = loss, trial.copy()
                    current = trial[index]
            levels[index] = current

        # Level is arithmetic, not search.
        output_gain = LevelMatch.match_rms(basis.render(gains, levels, 1.0), reference)
        best = min(best, target.distance(basis.render(gains, levels, output_gain)))
        return gains, levels, output_gain, best


# =============================================================================
# Stage 3 — the experiment
# =============================================================================


@dataclass
class Inspection:
    """Stage 3's verdict. This is the part that can say the model is wrong."""

    brightness_rises: bool
    brightness_spearman: float
    brightness_span_db: float
    amplitude_monotone: bool
    amplitude_spearman: float
    tension_invariant: bool
    tension_spread: float
    findings: list[str] = field(default_factory=list)
    verdict: str = ""

    @property
    def passed(self) -> bool:
        return self.brightness_rises and self.amplitude_monotone and self.tension_invariant

    def to_dict(self) -> dict:
        return {
            "brightness_rises": self.brightness_rises,
            "brightness_spearman": self.brightness_spearman,
            "brightness_span_db": self.brightness_span_db,
            "amplitude_monotone": self.amplitude_monotone,
            "amplitude_spearman": self.amplitude_spearman,
            "tension_invariant": self.tension_invariant,
            "tension_spread": self.tension_spread,
            "findings": list(self.findings),
            "verdict": self.verdict,
            "passed": self.passed,
        }


class InspectionStage:
    """Stage 3: does the per-velocity table vary smoothly and in the right
    direction?

    §8: per-velocity fitting ALWAYS succeeds, which is why it tells you nothing
    on its own. The evidence that the velocity model is right is that the fitted
    numbers move smoothly and physically — a harder hit is a shorter contact and
    a brighter excitation. Values that jump around, or brightness going the
    wrong way, mean the model is missing a mechanism, and no interpolation in
    stage 4 will paper over it.

    §8.1's falsifiable check lives here too: `tension.k` must be
    velocity-invariant. The glide deepens on hard hits by itself because more
    energy enters the bank, so a fit that wants a different k per velocity is
    telling you the energy feedback is wrong — not that k needs a velocity term.
    """

    #: Rank correlation past this is a trend rather than scatter.
    TREND_LIMIT: float = 0.6

    #: Brightness that moves less than this across the whole velocity range is
    #: not moving.
    MIN_BRIGHTNESS_SPAN_DB: float = 1.5

    #: k values spread wider than this fraction of their mean are not one k.
    TENSION_SPREAD_LIMIT: float = 0.35

    def run(self, fits: Sequence[ExcitationFit],
            tension_per_layer: Sequence[float] | None = None,
            progress: Progress = _noop) -> Inspection:
        progress({"stage": 3, "step": "inspecting the velocity table"})
        findings: list[str] = []

        if len(fits) < 3:
            return Inspection(
                brightness_rises=True, brightness_spearman=0.0,
                brightness_span_db=0.0, amplitude_monotone=True,
                amplitude_spearman=0.0, tension_invariant=True, tension_spread=0.0,
                findings=[f"only {len(fits)} velocity layers — too few to judge "
                          "whether the velocity model holds"],
                verdict="inconclusive",
            )

        ordered = sorted(fits, key=lambda fit: fit.velocity_normalized)
        velocities = np.array([fit.velocity_normalized for fit in ordered])
        brightness = np.array([fit.brightness_db for fit in ordered])
        amplitudes = np.array([fit.amplitude for fit in ordered])

        # Brightness is a dB-per-decade slope: less negative means brighter.
        spearman = InspectionStage._rank_correlation(velocities, brightness)
        span = float(np.ptp(brightness))
        rises = spearman >= InspectionStage.TREND_LIMIT and span >= InspectionStage.MIN_BRIGHTNESS_SPAN_DB

        if rises:
            findings.append(
                f"the excitation brightens with velocity (rank correlation "
                f"{spearman:+.2f}, {brightness[0]:+.1f} to {brightness[-1]:+.1f} "
                "dB/decade). Harder hits put more energy into the high modes, "
                "which is the shorter contact time the model claims"
            )
        elif span < InspectionStage.MIN_BRIGHTNESS_SPAN_DB:
            findings.append(
                f"brightness barely moves across the whole velocity range "
                f"({span:.2f} dB/decade of spread). The excitation is scaling in "
                "level and not changing shape, so contact_time explains nothing "
                "here — expected from a synthetic reference with no contact "
                "model, a red flag on real samples"
            )
        elif spearman <= -InspectionStage.TREND_LIMIT:
            findings.append(
                f"the excitation DARKENS with velocity (rank correlation "
                f"{spearman:+.2f}). That is the mechanism backwards. "
                "Per-velocity fitting still succeeded, as it always does — but "
                "fitting a curve through these numbers would hide the problem "
                "rather than fix it"
            )
        else:
            findings.append(
                f"brightness varies but not monotonically (rank correlation "
                f"{spearman:+.2f}, {span:.2f} dB/decade of spread). Values that "
                "jump around mean the model is missing a mechanism"
            )

        amplitude_rank = InspectionStage._rank_correlation(velocities, amplitudes)
        amplitude_monotone = amplitude_rank >= 0.8
        if not amplitude_monotone:
            findings.append(
                f"strike amplitude is not monotone in velocity (rank correlation "
                f"{amplitude_rank:+.2f}) — check the velocity calibration before "
                "reading anything else here"
            )

        # NaN means the layer's glide was too small to measure, not that k is
        # zero there. Those layers are dropped: §8.1 asks whether one k explains
        # every hit that HAS a glide, and a soft hit that barely moves would
        # fail that test on a drum where the mechanism is working perfectly.
        tension_invariant, spread = True, 0.0
        if tension_per_layer is not None:
            values = np.array([k for k in tension_per_layer if np.isfinite(k) and k > 0])
            if values.size < 3:
                findings.append(
                    f"§8.1's invariance check on k could not run: only "
                    f"{values.size} of {len(tension_per_layer)} layers glided "
                    "enough to measure. Not a failure, but not evidence either"
                )
            if values.size >= 3:
                spread = float(np.std(values) / np.mean(values))
                tension_invariant = spread <= InspectionStage.TENSION_SPREAD_LIMIT
                if tension_invariant:
                    findings.append(
                        f"tension k is velocity-invariant (spread {spread:.0%} "
                        f"around {np.mean(values):.4f}). The glide deepens on "
                        "hard hits from the energy feedback alone, exactly as "
                        "§8.1 requires"
                    )
                else:
                    # Before blaming the mechanism, rule out the measurement.
                    # k is fitted against each layer's own bank energy, which
                    # goes as amplitude squared, so an amplitude that is off by
                    # a factor comes back as a k off by its square — and
                    # `k * amplitude^2` is then the constant that k should have
                    # been. If that product is tight while k is not, the energy
                    # feedback is doing exactly what §8.1 says it does and the
                    # amplitudes are what need fixing.
                    paired = [
                        (k, fit.amplitude)
                        for k, fit in zip(tension_per_layer, fits)
                        if np.isfinite(k) and k > 0 and fit.amplitude > 0
                    ]
                    products = np.array([k * amp**2 for k, amp in paired])
                    product_spread = (
                        float(np.std(products) / np.mean(products))
                        if products.size >= 3 and np.mean(products) > 0
                        else float("inf")
                    )
                    if product_spread < spread * 0.5:
                        tension_invariant = True
                        findings.append(
                            f"tension k varies {spread:.0%} across velocity, but "
                            f"k * amplitude^2 varies only {product_spread:.0%}. "
                            "That product is the glide the strike actually "
                            "produces, so the ENERGY FEEDBACK IS FINE — what "
                            "varies is the measured strike amplitude, and k is "
                            "absorbing the error. §8.1 is not violated; the "
                            "velocity calibration is what to look at"
                        )
                    else:
                        findings.append(
                            f"tension k varies {spread:.0%} across velocity "
                            f"({values.min():.4f} to {values.max():.4f}), and "
                            f"k * amplitude^2 still varies {product_spread:.0%}, "
                            "so this is not an amplitude error. §8.1 is explicit "
                            "that it means the ENERGY FEEDBACK is wrong, not "
                            "that k needs a velocity term — do not add one"
                        )

        if rises and amplitude_monotone and tension_invariant:
            verdict = "the velocity model holds"
        elif not amplitude_monotone:
            verdict = "the velocity calibration is suspect"
        elif not tension_invariant:
            verdict = "the energy feedback does not explain the glide"
        elif span < InspectionStage.MIN_BRIGHTNESS_SPAN_DB:
            verdict = "the excitation does not vary with velocity beyond level"
        else:
            verdict = "the velocity model does not hold on this data"

        return Inspection(
            brightness_rises=rises,
            brightness_spearman=float(spearman),
            brightness_span_db=span,
            amplitude_monotone=amplitude_monotone,
            amplitude_spearman=float(amplitude_rank),
            tension_invariant=tension_invariant,
            tension_spread=spread,
            findings=findings,
            verdict=verdict,
        )

    @staticmethod
    def _rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) < 3:
            return 0.0
        rank_a = np.argsort(np.argsort(a)).astype(float)
        rank_b = np.argsort(np.argsort(b)).astype(float)
        if np.std(rank_a) == 0 or np.std(rank_b) == 0:
            return 0.0
        return float(np.corrcoef(rank_a, rank_b)[0, 1])


@dataclass
class NoiseDecayStage:
    """The transient bank's decays, measured from what the modes leave behind.

    §4 fixes the noise bank's SHAPE — four bands, fixed edges — and the fit was
    only ever allowed to move their levels. Their `t60` values came from the
    architecture's defaults: 55, 28, 14 and 7 ms. That is a click, and a click
    is not what a drum's unresolved content sounds like.

    Measured on a real tom: subtract the modal bank's own render from the
    sample, and the 200-800 Hz residual decays over **1645 ms with an r-squared
    of 0.98**, at only 28 dB below the sample's peak. That is a large, clean,
    well-determined signal the modal bank cannot represent — twenty-odd
    resolved partials do not cover a membrane's dense high-order content — and
    the band that should have carried it was pinned at 55 ms. With a decay 30
    times too short it cannot help at any level, so the gain solve switched it
    off: `level` came back at 5.9e-09, which is silence.

    So the decay is measured rather than assumed, on the residual rather than
    on the sample, because the noise bank's job is exactly what the modal bank
    could not do.

    It is measured and then CHECKED. Above 800 Hz that same tom's residual
    reports decays of 2.9 to 5.3 seconds — at 66 to 76 dB below peak, with
    r-squared around 0.55. Those are not decays; they are a flat noise floor
    fitted with a confident straight line. A band that fails the check keeps
    the architecture's default, and the fit says which ones did.
    """

    #: Straightness of the log-level regression. A real decay is a line; a
    #: noise floor is flat with a slope the fit invents.
    MIN_R_SQUARED: float = 0.80

    #: How far above the recording's own floor a band has to sit before its
    #: decay is a measurement of the drum.
    MIN_LEVEL_DB: float = -55.0

    #: Nothing shorter than the architecture's own default, and nothing longer
    #: than a drum: a band claiming to ring for four seconds is measuring a room.
    RANGE: tuple[float, float] = (0.005, 2.5)

    def __init__(self, sr: int = Audio.DEFAULT_SR, control_period: int = 64) -> None:
        self.sr = int(sr)
        self.control_period = int(control_period)

    def run(self, modes: list[Mode], gains: np.ndarray, tension: Tension,
            bands: Sequence[NoiseBand], reference: np.ndarray,
            progress: Progress = _noop) -> tuple[list[NoiseBand], list[str]]:
        """Bands with measured decays where the evidence supports one."""
        if not len(bands):
            return list(bands), []

        progress({"stage": "noise", "step": "measuring the transient decays"})
        residual = self.residual(modes, gains, tension, reference)

        peak = max(
            (decay.level_db for decay in BandDecayAnalyzer(self.sr).analyze(reference)),
            default=0.0,
        )
        measured = BandDecayAnalyzer(
            self.sr, bands=[(band.f_low, band.f_high) for band in bands]
        ).analyze(residual)

        out: list[NoiseBand] = []
        notes: list[str] = []
        for band, decay in zip(bands, measured):
            trusted = (
                decay.is_valid
                and decay.r_squared >= NoiseDecayStage.MIN_R_SQUARED
                and (decay.level_db - peak) >= NoiseDecayStage.MIN_LEVEL_DB
                and NoiseDecayStage.RANGE[0] <= decay.t60 <= NoiseDecayStage.RANGE[1]
            )
            if trusted:
                out.append(NoiseBand(band.f_low, band.f_high, band.level,
                                     float(decay.t60)))
                notes.append(
                    f"{band.f_low:.0f}-{band.f_high:.0f} Hz: the residual decays "
                    f"over {decay.t60 * 1000:.0f} ms (r^2 {decay.r_squared:.2f}, "
                    f"{decay.level_db - peak:.0f} dB below peak) — measured, "
                    f"against the {band.t60 * 1000:.0f} ms default"
                )
            else:
                out.append(band)
                notes.append(
                    f"{band.f_low:.0f}-{band.f_high:.0f} Hz: keeping the "
                    f"{band.t60 * 1000:.0f} ms default — the residual there is "
                    f"{decay.level_db - peak:.0f} dB below peak with r^2 "
                    f"{decay.r_squared:.2f}, which is a noise floor, not a decay"
                )
        progress({"stage": "noise", "decays": [band.t60 for band in out],
                  "notes": notes})
        return out, notes

    def residual(self, modes: list[Mode], gains: np.ndarray, tension: Tension,
                 reference: np.ndarray) -> np.ndarray:
        """The reference minus the modal bank's best least-squares fit to it.

        Least squares rather than RMS matching, because what is wanted here is
        the part of the signal the modes cannot reach — and that is what is left
        after removing as much of them as possible.
        """
        params = DrumParams(
            modes=[
                Mode(mode.f_static, float(max(gain, 1e-12)), mode.t60)
                for mode, gain in zip(modes, gains)
            ],
            noise=[], tension=tension, output_gain=1.0,
        )
        modal = DrumVoice(params, self.sr, self.control_period, seed=0).render_hit(
            len(reference) / self.sr)
        length = min(len(modal), len(reference))
        modal, reference = modal[:length], reference[:length]
        scale = float(np.dot(modal, reference)) / max(float(np.dot(modal, modal)), 1e-30)
        return reference - modal * scale


@dataclass
class TensionFit:
    """One layer's glide, and whether the data supports it at all.

    The distinction matters more than the number. `ratio = 1 + k * energy`, so
    a soft hit barely glides — and a glide too small to move the spectrum is a
    k that could be anything. Averaging those in with the loud layers drags the
    estimate to nonsense, and feeding them to §8.1's invariance check fails it
    on every drum, for a reason that has nothing to do with the mechanism.
    """

    k: float
    tau: float
    loss: float                # band dB with the fitted glide
    baseline: float            # band dB with no glide at all

    @classmethod
    def unmeasurable(cls) -> "TensionFit":
        return cls(0.0, 0.12, float("inf"), float("inf"))

    @property
    def improvement(self) -> float:
        """How much band error the glide removes. Negative means it added some."""
        if not np.isfinite(self.baseline) or not np.isfinite(self.loss):
            return float("-inf")
        return self.baseline - self.loss

    @property
    def is_evidence(self) -> bool:
        """True when this layer actually says something about the glide."""
        return self.k > 0 and self.improvement > TensionStage.ACCEPT_MARGIN_DB


class TensionStage:
    """Fitting the glide, once the excitation scale is known.

    This runs AFTER stage 2, not with stage 1, and the reason is arithmetic:
    `ratio = 1 + k * energy`, so k is only meaningful together with the
    absolute energy the strike puts into the bank. Fitting k against a
    unit-energy probe and then playing the drum at a tenth of that energy gets
    a tenth of the glide.

    Running it here also makes §8.1's falsifiable check possible, because k can
    be fitted independently at every velocity and the spread inspected. That
    check is the whole reason the glide is energy-driven rather than a pitch
    envelope, so it is worth the ordering.
    """

    #: The coarse grid, as a fraction of MAX_RATIO_SPAN. `fit_layer` refines
    #: inside whichever bracket wins, so the steps only have to bracket it.
    GRID_K: tuple[float, ...] = (0.0, 0.03, 0.07, 0.13, 0.22, 0.35, 0.52, 0.74, 1.0)
    GRID_TAU: tuple[float, ...] = (0.04, 0.08, 0.14, 0.24)

    #: The largest glide the grid may propose, as a frequency ratio above rest.
    #: Three semitones. `Tension.max_ratio` allows a full octave, and left to
    #: itself the grid will take it: a huge k with a very short tau pins every
    #: mode at the clamp for the first few milliseconds, which is a cheap way to
    #: smear the attack across the low-resolution STFT frames and score better
    #: than the true glide does. That candidate is not a drum. Bounding the
    #: search in musical units rather than in k removes the corner outright.
    MAX_RATIO_SPAN: float = 0.30

    def __init__(self, sr: int = Audio.DEFAULT_SR, control_period: int = 64) -> None:
        self.sr = int(sr)
        self.control_period = int(control_period)

    def peak_energy(self, modes: list[Mode], gains: np.ndarray,
                    n_samples: int) -> float:
        """The largest bank energy this strike reaches, with the glide off.

        This is the quantity k multiplies, so it is what turns a k into a
        musical interval. Measured rather than assumed: it is close to the sum
        of the squared gains, but only close, and the grid's bounds are only
        meaningful if it is exact.
        """
        probe = [
            Mode(mode.f_static, float(max(gain, 1e-12)), mode.t60)
            for mode, gain in zip(modes, gains)
        ]
        params = DrumParams(modes=probe, noise=[], tension=Tension(k=0.0, tau=0.12))
        voice = DrumVoice(params, self.sr, self.control_period, seed=0)
        voice.strike(1.0)
        peak, position = 0.0, 0
        while position < n_samples:
            span = min(self.control_period, n_samples - position)
            peak = max(peak, float(voice.modes.energy))
            voice.modes.process(span, 1.0)
            position += span
        return peak

    #: How far the refinement narrows around the coarse winner, and how often.
    REFINE_ROUNDS: int = 3
    REFINE_PROBES: int = 4

    def fit_layer(self, modes: list[Mode], gains: np.ndarray, output_gain: float,
                  reference: np.ndarray) -> "TensionFit":
        """The glide of one hit, fitted on the SPECTRUM.

        The obvious way to fit a glide is to track the fundamental in both
        signals and match the curves, and that is what this did first. It does
        not work well enough. The tracked fundamental of a drum is a short,
        noisy, partly-masked thing, the comparison has to be made in cents
        relative to each track's own asymptote (an absolute offset would swamp
        the shape), and what survives all that is a proxy an optimizer can win
        against while getting k wrong by a factor of six — which then ruins
        every gain refitted afterwards, because the partials end up in the
        wrong place.

        Measured on a drum with a known glide, the band loss has a sharp,
        well-behaved minimum at the true k (1.8 dB, against 4.0 dB for no glide
        at all) at a point where the pitch track returned zero. The glide moves
        energy between bands, so the loss the fit is judged on can see it
        directly. Fit it there.
        """
        target = SpectralTarget(reference, self.sr)
        energy = float(np.sum(np.asarray(gains, dtype=float) ** 2))
        if energy <= 0:
            return TensionFit.unmeasurable()

        # `ratio = 1 + k * smoothed_energy`, so k alone means nothing without
        # the energy this particular strike puts into the bank. Measure that
        # once, and the grid is over the RATIO the glide reaches — the same
        # range for every drum, whatever its gains happen to be scaled to.
        peak = self.peak_energy(modes, gains, len(reference))
        if peak <= 0:
            return TensionFit.unmeasurable()
        k_scale = TensionStage.MAX_RATIO_SPAN / peak

        def loss(k: float, tau: float) -> float:
            return self.spectral_loss(
                modes, gains, reference, Tension(k=float(k), tau=float(tau)), target)

        # The no-glide loss is the baseline every candidate has to beat, and
        # it is also grid point zero, so it costs nothing extra.
        baseline = loss(0.0, TensionStage.GRID_TAU[0])

        best_fraction, best_tau, best = 0.0, 0.12, baseline
        for fraction in TensionStage.GRID_K[1:]:
            for tau in TensionStage.GRID_TAU:
                value = loss(fraction * k_scale, tau)
                if value < best:
                    best, best_fraction, best_tau = value, fraction, tau
        if not np.isfinite(best) or best_fraction == 0.0:
            return TensionFit(0.0, 0.12, baseline, baseline)

        # Refine k inside the bracket the coarse grid left, holding tau: the
        # grid steps are wide enough that the minimum sits between them.
        grid = TensionStage.GRID_K
        position = grid.index(best_fraction)
        low = grid[position - 1] if position > 0 else 0.0
        high = grid[position + 1] if position + 1 < len(grid) else grid[-1] * 1.5
        for _ in range(TensionStage.REFINE_ROUNDS):
            probes = np.linspace(low, high, TensionStage.REFINE_PROBES + 2)[1:-1]
            for fraction in probes:
                value = loss(fraction * k_scale, best_tau)
                if value < best:
                    best, best_fraction = value, float(fraction)
            span = (high - low) * 0.25
            low, high = max(0.0, best_fraction - span), best_fraction + span

        return TensionFit(best_fraction * k_scale, best_tau, best, baseline)

    def spectral_loss(self, modes: list[Mode], gains: np.ndarray,
                      reference: np.ndarray, tension: Tension,
                      target: SpectralTarget | None = None) -> float:
        """Band-loss of the unit-gain bank under one tension setting.

        Level is re-matched every time so this measures the SPECTRUM, not the
        loudness: a glide moves energy between bands, and that is the part
        worth accepting or rejecting on.
        """
        target = target or SpectralTarget(reference, self.sr)
        probe = [
            Mode(mode.f_static, float(max(gain, 1e-12)), mode.t60)
            for mode, gain in zip(modes, gains)
        ]
        params = DrumParams(modes=probe, noise=[], tension=tension)
        audio = DrumVoice(params, self.sr, self.control_period,
                          seed=0).render_hit(len(reference) / self.sr, 1.0)
        return target.distance(audio * LevelMatch.match_rms(audio, reference))

    #: A glide has to pay for itself by this much band-dB before it counts as
    #: evidence. Below this the layer is saying nothing either way.
    ACCEPT_MARGIN_DB: float = 0.05

    def run(self, modes: list[Mode], fits: Sequence[ExcitationFit],
            layers: Sequence[np.ndarray], progress: Progress = _noop,
            ) -> tuple[Tension, list[float], list[str]]:
        """One glide for the drum, plus the per-layer k values §8.1 inspects.

        A layer whose glide the spectrum cannot see comes back as NaN rather
        than as a number, and stage 3 drops it. Reporting a fabricated k there
        would fail the invariance check on every drum ever fitted.
        """
        results: list[TensionFit] = []
        notes: list[str] = []

        for index, (fit, audio) in enumerate(zip(fits, layers)):
            progress({"stage": "tension", "layer": index + 1, "of": len(fits),
                      "velocity": fit.velocity})
            results.append(
                self.fit_layer(modes, fit.gains, fit.output_gain, audio))

        # NaN, not zero: "this layer says nothing" and "this layer says the
        # drum does not glide" are different claims, and only the first is true
        # of a hit too soft to move the pitch at all.
        per_layer = [
            result.k if result.is_evidence else float("nan") for result in results
        ]
        evidenced = [result for result in results if result.is_evidence]

        if not evidenced:
            notes.append(
                "no layer's spectrum preferred a glide to no glide, so tension "
                "is off (k = 0). Either this drum genuinely does not glide, or "
                "the modes are far enough out that moving them cannot help — "
                "check GlideAnalyzer.is_genuine_glide on the reference before "
                "concluding the first"
            )
            progress({"stage": "tension", "k": 0.0, "tau": 0.12,
                      "per_layer": per_layer, "evidenced": 0})
            return Tension(k=0.0, tau=0.12), per_layer, notes

        # §8.1: ONE k for the drum, and it comes from the LOUDEST layer that
        # carries evidence — not from a median over all of them.
        #
        # The reason is `_canonicalize`: the gain scale is fixed by convention
        # on the loudest layer, which therefore has an exact amplitude by
        # definition, while every other layer's amplitude is measured and
        # carries error. `ratio = 1 + k * energy` and energy goes as amplitude
        # squared, so a layer whose amplitude is low by a factor of two comes
        # back with a k four times too big — it is compensating, and it fits
        # its own audio perfectly while doing so. Averaging that in moves the
        # one number that is right.
        loudest = max(
            range(len(results)),
            key=lambda i: (results[i].is_evidence, fits[i].amplitude),
        )
        chosen = results[loudest]
        k, tau = chosen.k, chosen.tau

        if fits[loudest].amplitude < max(fit.amplitude for fit in fits):
            notes.append(
                "the loudest layer's glide was not measurable, so k comes from "
                "a quieter one whose amplitude is measured rather than fixed by "
                "convention. Treat its absolute value with suspicion"
            )

        worst = max(result.loss for result in evidenced)
        if worst > 8.0:
            notes.append(
                f"the best glide still leaves up to {worst:.1f} dB of band error "
                "on some layer. That is the fit as a whole, not the tension "
                "alone, but if stage 5 does not close it, check "
                "GlideAnalyzer.is_genuine_glide on the reference: energy-driven "
                "tension may be the wrong explanation for this drum"
            )
        if len(evidenced) < 3 and len(results) >= 3:
            notes.append(
                f"only {len(evidenced)} of {len(results)} layers glided enough to "
                "measure, so §8.1's velocity-invariance check on k is weak here. "
                "It is not a failure — soft hits are supposed to barely glide"
            )
        progress({"stage": "tension", "k": k, "tau": tau,
                  "per_layer": per_layer, "evidenced": len(evidenced),
                  "improvement_db": float(np.median(
                      [result.improvement for result in evidenced]))})
        return Tension(k=k, tau=tau), per_layer, notes


# =============================================================================
# Stages 4 and 5 — velocity becomes continuous
# =============================================================================


@dataclass
class VelocityCurve:
    """Excitation as a function of normalized velocity. Two numbers each.

        amplitude(v)    = amplitude_scale * v ** amplitude_exponent
        contact_time(v) = contact_scale * v ** contact_exponent
        noise level(v)  = level_scale[j] * v ** level_exponent
    """

    amplitude_scale: float = 1.0
    amplitude_exponent: float = 1.0
    contact_scale: float = 0.001
    contact_exponent: float = -0.3
    level_scale: np.ndarray = field(default_factory=lambda: np.zeros(0))
    level_exponent: float = 1.2
    loss: float = float("inf")

    #: The softest recorded hit is not a zero-strength strike, but velocity
    #: calibration maps it to 0 — it is the bottom of the observed range, not
    #: the bottom of the physical one. Feeding 0 to a power law makes the
    #: quietest layer silent and, worse, drags log(0) into the log-log fit and
    #: corrupts the exponent for every other layer. Strength is therefore the
    #: normalized velocity remapped onto a strictly positive interval.
    MIN_STRENGTH: float = 0.08

    @staticmethod
    def strength(velocity) -> np.ndarray:
        v = np.clip(np.asarray(velocity, dtype=float), 0.0, 1.0)
        return VelocityCurve.MIN_STRENGTH + (1.0 - VelocityCurve.MIN_STRENGTH) * v

    def amplitude(self, velocity) -> np.ndarray:
        return self.amplitude_scale * self.strength(velocity) ** self.amplitude_exponent

    def contact_time(self, velocity) -> np.ndarray:
        return np.clip(
            self.contact_scale * self.strength(velocity) ** self.contact_exponent,
            *ExcitationTiltModel.RANGE,
        )

    def levels(self, velocity) -> np.ndarray:
        return self.level_scale * float(
            self.strength(velocity) ** self.level_exponent
        )

    def gains(self, freqs: np.ndarray, shape: np.ndarray, velocity: float) -> np.ndarray:
        """Mode gains at a velocity: the frozen per-mode shape, tilted by the
        contact time that velocity implies, scaled to that velocity's strength."""
        tilt = ExcitationTiltModel.tilt(freqs, float(self.contact_time(velocity)))
        raw = shape * tilt
        norm = np.sqrt(np.sum(raw**2))
        if norm <= 0:
            return raw
        return raw / norm * float(self.amplitude(velocity))

    @staticmethod
    def batch_gains(vectors: np.ndarray, freqs: np.ndarray, shape: np.ndarray,
                    velocity: float) -> np.ndarray:
        """`(S, 6)` candidate vectors -> `(S, modes)` gains, in one pass.

        The same arithmetic as `gains`, done for a whole population at once.
        Worth having as its own method rather than a loop: stage 5 asks for this
        once per layer per generation, and a Python loop over sixty candidates
        times six layers is 360 small numpy calls per generation, which on a GPU
        run is dead time the device spends waiting for the interpreter.
        """
        vectors = np.atleast_2d(np.asarray(vectors, dtype=float))
        strength = VelocityCurve.strength(velocity)

        contact = np.clip(
            vectors[:, 2] * strength ** vectors[:, 3], *ExcitationTiltModel.RANGE)
        amplitude = vectors[:, 0] * strength ** vectors[:, 1]

        cut = 1.0 / (np.pi * np.maximum(contact, 1e-6))            # (S,)
        tilt = 1.0 / (1.0 + (np.asarray(freqs, float)[None, :] / cut[:, None]) ** 2)
        raw = np.asarray(shape, float)[None, :] * tilt             # (S, modes)
        norm = np.sqrt(np.sum(raw**2, axis=1))
        scale = np.where(norm > 0, amplitude / np.maximum(norm, 1e-30), 1.0)
        return raw * scale[:, None]

    @staticmethod
    def batch_levels(vectors: np.ndarray, band_shape: np.ndarray,
                     velocity: float) -> np.ndarray:
        """`(S, 6)` -> `(S, bands)` noise levels."""
        vectors = np.atleast_2d(np.asarray(vectors, dtype=float))
        strength = VelocityCurve.strength(velocity)
        scale = vectors[:, 4] * strength ** vectors[:, 5]
        return np.asarray(band_shape, float)[None, :] * scale[:, None]

    def to_dict(self) -> dict:
        return {
            "amplitude_scale": self.amplitude_scale,
            "amplitude_exponent": self.amplitude_exponent,
            "contact_scale": self.contact_scale,
            "contact_exponent": self.contact_exponent,
            "level_scale": np.asarray(self.level_scale).tolist(),
            "level_exponent": self.level_exponent,
            "loss": self.loss,
        }


class VelocityCurveStage:
    """Stage 4: a smooth two-parameter curve through each per-velocity value."""

    def run(self, fits: Sequence[ExcitationFit], progress: Progress = _noop) -> VelocityCurve:
        progress({"stage": 4, "step": "fitting velocity curves"})
        ordered = sorted(fits, key=lambda fit: fit.velocity_normalized)
        velocities = VelocityCurve.strength(
            np.array([fit.velocity_normalized for fit in ordered])
        )
        amplitudes = np.array([fit.amplitude for fit in ordered])
        contacts = np.array([fit.contact_time for fit in ordered])
        levels = np.array([fit.levels for fit in ordered])

        amplitude_scale, amplitude_exponent = self._power_law(velocities, amplitudes)
        contact_scale, contact_exponent = self._power_law(velocities, contacts)

        if levels.size:
            peak = levels[-1]
            level_scale, level_exponent = peak, self._power_law(
                velocities, np.maximum(levels.max(axis=1), 1e-9)
            )[1]
        else:
            level_scale, level_exponent = np.zeros(0), 1.0

        return VelocityCurve(
            amplitude_scale=amplitude_scale,
            amplitude_exponent=amplitude_exponent,
            contact_scale=contact_scale,
            contact_exponent=contact_exponent,
            level_scale=level_scale,
            level_exponent=level_exponent,
        )

    @staticmethod
    def _power_law(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """Least squares in log-log, which is a straight line for y = a x**b."""
        valid = (x > 0) & (y > 0) & np.isfinite(y)
        if valid.sum() < 2:
            return (float(np.mean(y)) if len(y) else 1.0, 0.0)
        exponent, intercept = np.polyfit(np.log(x[valid]), np.log(y[valid]), 1)
        return float(np.exp(intercept)), float(exponent)


@dataclass
class Generation:
    """One step of the joint refinement, for the progress display."""

    index: int
    loss: float
    best_loss: float
    convergence: float = 0.0
    elapsed: float = 0.0
    parameters: dict = field(default_factory=dict)


class JointStage:
    """Stage 5: refine the velocity mapping across every hit at once.

    Six numbers, evaluated against the WHOLE set rather than any single hit —
    §8.2 is explicit that a parameter set tuned against one sample matches it
    and generalizes to nothing. The aggregate here is the worst hit, not the
    mean, for the same reason `DrumScorer.aggregate` reports the worst: a value
    that is right for three velocities and wrong for the fourth is broken, and
    averaging hides exactly that.

    Differential evolution because the loss is not differentiable (the analysis
    chain has an SVD and an assignment problem in it) and six dimensions is
    where a population method is at its best.
    """

    def __init__(self, sr: int = Audio.DEFAULT_SR, control_period: int = 64) -> None:
        self.sr = int(sr)
        self.control_period = int(control_period)

    def run(
        self,
        modal: ModalFit,
        shape: np.ndarray,
        curve: VelocityCurve,
        layers: Sequence[tuple[float, np.ndarray]],
        noise_bands: Sequence[NoiseBand],
        generations: int = 24,
        population: int = 12,
        device: Device | None = None,
        progress: Progress = _noop,
        seed: int = 0,
    ) -> tuple[VelocityCurve, list[Generation]]:
        import time

        device = device or DeviceChoice.resolve(DeviceChoice.CPU)
        freqs = np.array([mode.f_static for mode in modal.modes])

        def linearize(velocity: float, reference: np.ndarray,
                      around: VelocityCurve) -> LinearVoiceBasis:
            """The basis for one layer, linearized around a real excitation.

            This is not a detail. The basis is unit-gain per mode, so the gains
            it is built with matter for exactly one thing — the tension
            trajectory, which `LinearVoiceBasis` captures from a real render and
            then freezes. `ratio = 1 + k * bank_energy`, so building it around
            the wrong energy rings every mode at the wrong frequency for the
            whole hit, and no combination of the six parameters being searched
            can move it back.

            It was built around `modal.modes` before, whose gains come out of
            stage 1 and mean nothing. Measured on a synthetic drum: that pinned
            the trajectory at `Tension.max_ratio` — an octave — and stage 5 sat
            flat at 8.3 dB while stage 2 had already reached 4.5 dB on the same
            layers. The same signature, flat at 11.1 dB, is what a real run
            reported.
            """
            gains = np.asarray(around.gains(freqs, shape, velocity), dtype=float)
            levels = np.atleast_1d(around.levels(velocity))
            params = DrumParams(
                modes=[
                    Mode(mode.f_static, float(max(gain, 1e-12)), mode.t60)
                    for mode, gain in zip(modal.modes, gains)
                ],
                noise=[
                    NoiseBand(band.f_low, band.f_high, float(max(level, 0.0)),
                              band.t60)
                    for band, level in zip(noise_bands, levels)
                ],
                tension=modal.tension,
                output_gain=1.0,
            )
            return LinearVoiceBasis.build(
                params, len(reference), self.sr, self.control_period)

        def prepare(around: VelocityCurve) -> list:
            return [
                (velocity,
                 LossBackend.build(linearize(velocity, reference, around),
                                   SpectralTarget(reference, self.sr),
                                   reference, device))
                for velocity, reference in layers
            ]

        prepared = prepare(curve)
        TorchMemory.reset()
        progress({"stage": 5, "step": f"stage 5 on {device}",
                  "device": device.kind, "backend": device.backend,
                  "device_detail": device.detail})

        start = np.array([
            curve.amplitude_scale, curve.amplitude_exponent,
            curve.contact_scale, curve.contact_exponent,
            float(np.max(curve.level_scale)) if np.size(curve.level_scale) else 1e-6,
            curve.level_exponent,
        ])
        bounds = [
            (start[0] * 0.25, start[0] * 4.0),
            (0.2, 3.5),
            ExcitationTiltModel.RANGE,
            (-2.0, 0.5),
            (1e-9, max(start[4] * 8.0, 1e-6)),
            (0.2, 4.0),
        ]
        start = np.array([
            float(np.clip(value, low, high))
            for value, (low, high) in zip(start, bounds)
        ])
        band_shape = (
            curve.level_scale / np.max(curve.level_scale)
            if np.size(curve.level_scale) and np.max(curve.level_scale) > 0
            else np.zeros(len(noise_bands))
        )

        def population_loss(vectors: np.ndarray) -> np.ndarray:
            """(S, 6) candidates -> (S,) losses, the WORST layer for each.

            Every candidate in the generation is evaluated together: one
            `(S, modes) @ (modes, samples)` product and one batched STFT per
            layer, rather than S of each. That is what makes a GPU worth
            anything here, and on the CPU it is still the faster arrangement.
            """
            worst = np.zeros(len(vectors))
            for velocity, loss in prepared:
                gains = VelocityCurve.batch_gains(vectors, freqs, shape, velocity)
                levels = VelocityCurve.batch_levels(vectors, band_shape, velocity)
                worst = np.maximum(worst, loss.losses(gains, levels))
            return worst

        def objective(vector: np.ndarray) -> float:
            return float(population_loss(np.atleast_2d(vector))[0])

        history: list[Generation] = []
        began = time.perf_counter()
        best = {"loss": float("inf")}

        def on_generation(vector, convergence=0.0) -> bool:
            loss = objective(vector)
            best["loss"] = min(best["loss"], loss)
            record = Generation(
                index=len(history) + 1,
                loss=float(loss),
                best_loss=float(best["loss"]),
                convergence=float(convergence),
                elapsed=time.perf_counter() - began,
                parameters=JointStage._curve_from(vector, band_shape).to_dict(),
            )
            history.append(record)
            progress({"stage": 5, "generation": record.index, "loss": record.loss,
                      "best_loss": record.best_loss,
                      "convergence": record.convergence,
                      "elapsed": record.elapsed})
            return False

        def search(x0: np.ndarray, budget: int):
            # `vectorized`, never `workers`. Handing scipy `workers=N` puts each
            # candidate in its own process, which on Windows raises outright --
            # `spawn` cannot pickle a closure -- and everywhere else pickles
            # tens of megabytes of basis per task to save a few milliseconds of
            # arithmetic. A vectorized objective evaluates the generation in one
            # call instead, which is where the batching (and the GPU) pays off.
            return differential_evolution(
                lambda x: population_loss(np.atleast_2d(x.T)),
                bounds=bounds,
                maxiter=max(1, budget),
                popsize=max(4, population),
                tol=1e-6,
                seed=seed,
                polish=True,
                vectorized=True,
                updating="deferred",
                callback=on_generation,
                x0=x0,
            )

        # Two passes, for the same reason stage 2 runs two. The basis freezes a
        # tension trajectory, and the trajectory depends on the excitation the
        # search is changing; one relinearization around the answer the first
        # pass found puts the modes where the refined drum actually rings. The
        # second pass is cheap because it starts where the first one stopped.
        first = max(1, int(generations))
        second = max(1, first // 2)

        # What stage 4 handed over, measured the same way the result will be.
        # Stage 5 is a refinement and is not allowed to be a regression.
        incoming = self.true_loss(modal, shape, curve, layers, noise_bands)
        progress({"stage": 5, "step": "pass 1 of 2", "pass": 1,
                  "generations": first, "incoming_loss": incoming})
        result = search(start, first)

        if modal.tension.k > 0:
            refined = JointStage._curve_from(result.x, band_shape)
            progress({"stage": 5, "step": "relinearizing around the refined "
                                          "excitation", "pass": 2,
                      "generations": second})
            prepared = prepare(refined)
            result = search(np.array([
                float(np.clip(value, low, high))
                for value, (low, high) in zip(result.x, bounds)
            ]), second)

        refined = JointStage._curve_from(result.x, band_shape)

        # The number reported is measured on a REAL render, not on the basis.
        # The basis is an approximation with a frozen glide, and letting it
        # report its own opinion of itself is how a fit gets to look better
        # than it is: on one run the basis said 4.99 dB where a real render of
        # the same parameters said 5.18.
        refined.loss = self.true_loss(modal, shape, refined, layers, noise_bands)

        # And having measured both honestly, keep the better one. A search over
        # a six-parameter curve, scored through an approximate basis, can land
        # somewhere worse than it began; handing that back as "refined" would
        # make stage 5 a coin flip that the user pays a minute for.
        if incoming <= refined.loss:
            progress({"stage": 5, "step": "keeping stage 4's curve",
                      "loss": incoming, "refined_loss": refined.loss,
                      **self._device_report(device)})
            kept = curve
            kept.loss = incoming
            return kept, history

        progress({"stage": 5, "step": "stage 5 done", "loss": refined.loss,
                  "basis_loss": float(result.fun), "incoming_loss": incoming,
                  "improvement_db": incoming - refined.loss,
                  **self._device_report(device)})
        return refined, history

    @staticmethod
    def _device_report(device: Device) -> dict:
        """What the run ACTUALLY used, not what it was asked to use.

        A peak allocation of zero after stage 5 means no tensor ever reached
        the GPU, whatever the picker said — which is the one fact that
        separates "CUDA is working and the device is simply idle between
        kernels" from "this ran on the CPU".
        """
        peak = TorchMemory.peak_bytes()
        return {
            "device": device.kind,
            "device_detail": device.detail,
            "peak_vram_bytes": peak,
            "used_cuda": bool(device.is_cuda and peak > 0),
        }

    def true_loss(self, modal: ModalFit, shape: np.ndarray, curve: VelocityCurve,
                  layers: Sequence[tuple[float, np.ndarray]],
                  noise_bands: Sequence[NoiseBand]) -> float:
        """The worst layer's band distance, from a full render of each layer."""
        freqs = np.array([mode.f_static for mode in modal.modes])
        worst = 0.0
        for velocity, reference in layers:
            gains = np.asarray(curve.gains(freqs, shape, velocity), dtype=float)
            levels = np.atleast_1d(curve.levels(velocity))
            params = DrumParams(
                modes=[
                    Mode(mode.f_static, float(max(gain, 1e-12)), mode.t60)
                    for mode, gain in zip(modal.modes, gains)
                ],
                noise=[
                    NoiseBand(band.f_low, band.f_high, float(max(level, 0.0)),
                              band.t60)
                    for band, level in zip(noise_bands, levels)
                ],
                tension=modal.tension, output_gain=1.0,
            )
            audio = DrumVoice(params, self.sr, self.control_period,
                              seed=0).render_hit(len(reference) / self.sr)
            scale = LevelMatch.match_rms(audio, reference)
            worst = max(worst, SpectralTarget(reference, self.sr).distance(
                audio * scale))
        return float(worst)

    @staticmethod
    def _curve_from(vector: np.ndarray, band_shape: np.ndarray) -> VelocityCurve:
        return VelocityCurve(
            amplitude_scale=float(vector[0]),
            amplitude_exponent=float(vector[1]),
            contact_scale=float(vector[2]),
            contact_exponent=float(vector[3]),
            level_scale=np.asarray(band_shape) * float(vector[4]),
            level_exponent=float(vector[5]),
        )
