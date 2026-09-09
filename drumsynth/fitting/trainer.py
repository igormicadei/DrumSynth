"""Running the five stages end to end for one drum.

One drum at a time, deliberately. `f_static` and `t60` are properties of a
specific physical drum, so there is nothing a second drum could contribute to
this fit except a way to get them confused.

The trainer owns the staging and the freezing; the stages own the numerics. It
emits a progress record for every meaningful step, so a UI can follow along
without knowing anything about how the fit works.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from ..core.constants import Audio, Decibels
from ..scoring.comparator import ScoreCard
from ..scoring.scorer import DrumScorer
from ..synth.params import DrumParams, Mode, NoiseBand, Tension
from ..synth.voice import DrumVoice
from .objective import LevelMatch
from .stages import (
    ExcitationFit,
    ExcitationStage,
    ExcitationTiltModel,
    Generation,
    Inspection,
    InspectionStage,
    JointStage,
    ModalFit,
    ModalStage,
    TensionStage,
    VelocityCurve,
    VelocityCurveStage,
)
from .targets import FitTarget

Progress = Callable[[dict], None]


def _noop(_: dict) -> None:
    pass


@dataclass
class TrainingSettings:
    """Everything a run can be tuned by. Defaults are sized for a few minutes."""

    seconds: float = 2.5
    max_layers: int = 8
    max_modes: int = 34
    control_period: int = 64
    noise_bands: int = 4
    generations: int = 24
    population: int = 12
    workers: int = 1
    refine_gains: bool = True
    seed: int = 0

    def to_dict(self) -> dict:
        return {
            "seconds": self.seconds, "max_layers": self.max_layers,
            "max_modes": self.max_modes, "control_period": self.control_period,
            "noise_bands": self.noise_bands, "generations": self.generations,
            "population": self.population, "workers": self.workers,
            "refine_gains": self.refine_gains, "seed": self.seed,
        }


@dataclass
class FitResult:
    """A fitted drum, and everything needed to judge it."""

    drum: str
    params: DrumParams                      # at the reference velocity
    modal: ModalFit
    excitation: list[ExcitationFit]
    inspection: Inspection
    curve: VelocityCurve
    generations: list[Generation]
    shape: np.ndarray                       # frozen per-mode gain shape
    reference_velocity: float
    elapsed: float = 0.0
    settings: TrainingSettings = field(default_factory=TrainingSettings)
    warnings: list[str] = field(default_factory=list)

    def params_at(self, velocity_normalized: float,
                  sr: int = Audio.DEFAULT_SR) -> DrumParams:
        """The drum as it plays at one velocity.

        Velocity varies excitation only — the modes and the tension are the
        same object at every strength, which is the whole claim of §7.1.
        """
        freqs = np.array([mode.f_static for mode in self.modal.modes])
        gains = self.curve.gains(freqs, self.shape, velocity_normalized)
        levels = self.curve.levels(velocity_normalized)

        modes = [
            Mode(mode.f_static, float(max(gain, 1e-9)), mode.t60)
            for mode, gain in zip(self.modal.modes, gains)
        ]
        noise = [
            NoiseBand(band.f_low, band.f_high, float(max(level, 0.0)), band.t60)
            for band, level in zip(self.params.noise, levels)
        ] if len(levels) else list(self.params.noise)

        return DrumParams(
            modes=modes, noise=noise, tension=self.modal.tension,
            output_gain=self.params.output_gain,
            name=f"{self.drum}@v{velocity_normalized:.2f}",
        )

    def velocity_table(self) -> list[dict]:
        return [fit.to_row() for fit in self.excitation]


class DrumTrainer:
    """The staged fit of ARCHITECTURE.md §8, for one drum."""

    #: Noise bands are not measured from scratch — the architecture fixes their
    #: shape (§4) and only their levels are fitted.
    DEFAULT_BANDS: tuple[tuple[float, float, float], ...] = (
        (200.0, 800.0, 0.055),
        (800.0, 2000.0, 0.028),
        (2000.0, 6000.0, 0.014),
        (6000.0, 15000.0, 0.007),
    )

    def __init__(self, settings: TrainingSettings | None = None,
                 sr: int = Audio.DEFAULT_SR) -> None:
        self.settings = settings or TrainingSettings()
        self.sr = int(sr)

    # -- the run --------------------------------------------------------------

    def run(self, target: FitTarget, progress: Progress = _noop) -> FitResult:
        started = time.perf_counter()
        settings = self.settings
        warnings = list(target.warnings)

        layers = target.layers[: settings.max_layers]
        if len(layers) < len(target.layers):
            warnings.append(
                f"fitting {len(layers)} of {len(target.layers)} velocity layers"
            )
        reference = target.layers[target.reference_index]
        frequency_layer = target.frequency_layer()

        progress({"phase": "start", "drum": target.drum, "layers": len(layers),
                  "reference_velocity": reference.velocity,
                  "frequency_velocity": frequency_layer.velocity})

        # --- stage 1: the drum, frozen -------------------------------------
        # Frequencies from a soft hit (the drum at rest, which is what
        # f_static means); damping from the loud one, where every band is
        # above the noise long enough to fit a slope.
        modal = ModalStage(self.sr, settings.max_modes).run(
            frequency_layer.audio, reference.audio, progress=progress
        )
        warnings.extend(modal.notes)
        progress({"phase": "stage1", "modes": len(modal.modes),
                  "f0": modal.modes[0].f_static if modal.modes else 0.0,
                  "tension_k": modal.tension.k, "tension_tau": modal.tension.tau,
                  "notes": modal.notes})

        bands = self._noise_bands(settings.noise_bands)

        # --- stage 2: excitation, per velocity, independently ---------------
        # Two passes. The first runs with the tension off, because a linearized
        # basis needs a frequency trajectory and there is not one yet; the
        # second re-runs with the fitted glide in place. The trajectory barely
        # moves once the gains are roughly right, so two passes is enough.
        stage2 = ExcitationStage(self.sr, settings.control_period)

        reference_position = min(target.reference_index, len(layers) - 1)

        def fit_layers(current: ModalFit, label: str,
                       shape: np.ndarray | None = None,
                       seed_fits: list[ExcitationFit] | None = None) -> list[ExcitationFit]:
            """The reference layer fits every gain; the rest fit two numbers.

            §4: at one fixed velocity `contact_time` is absorbed into the gains,
            and when velocity is added it is reintroduced and the gains are
            refit against it. That is exactly this split, and it is what makes
            stage 3's brightness trend a measurement rather than noise.
            """
            out: list[ExcitationFit] = []
            reference_shape = shape
            order = [reference_position] + [
                index for index in range(len(layers)) if index != reference_position
            ]
            results: dict[int, ExcitationFit] = {}

            # Strike strength is measured, not searched: a louder recording of
            # the same drum is a harder hit, and the level ratio says by how
            # much. See ExcitationStage._fit_contact_time.
            levels_rms = np.array([
                float(np.sqrt(np.mean(layer.audio**2))) for layer in layers
            ])
            reference_rms = max(levels_rms[reference_position], 1e-12)
            reference_amplitude = 1.0
            shared_output_gain = None

            for step, index in enumerate(order):
                layer = layers[index]
                progress({"phase": "stage2", "pass": label, "layer": step + 1,
                          "of": len(layers), "velocity": layer.velocity,
                          "free_gains": index == reference_position})
                fit = stage2.run(
                    current, layer.audio, layer.velocity,
                    layer.velocity_normalized, bands,
                    seconds=settings.seconds, progress=progress,
                    free_gains=settings.refine_gains,
                    initial_gains=seed_fits[index].gains if seed_fits else None,
                    shape=None if index == reference_position else reference_shape,
                    amplitude=(
                        None if index == reference_position
                        else reference_amplitude * levels_rms[index] / reference_rms
                    ),
                    output_gain=(
                        None if index == reference_position else shared_output_gain
                    ),
                )
                results[index] = fit
                if index == reference_position:
                    reference_amplitude = fit.amplitude
                    # output_gain is the drum's, not the hit's: one value for
                    # every velocity, so a layer's level error shows up in its
                    # loss instead of being normalized away.
                    shared_output_gain = fit.output_gain
                if index == reference_position and reference_shape is None:
                    # The reference layer defines the shape every other layer
                    # is re-tilted from. Divide out its own contact tilt so the
                    # shape is the part velocity does NOT explain.
                    freqs = np.array([mode.f_static for mode in current.modes])
                    raw = fit.gains / np.maximum(
                        ExcitationTiltModel.tilt(
                            freqs, ExcitationTiltModel.REFERENCE_CONTACT
                        ),
                        1e-9,
                    )
                    norm = float(np.sqrt(np.sum(raw**2))) or 1.0
                    reference_shape = raw / norm

            out = [results[index] for index in range(len(layers))]
            return out, reference_shape

        fits, shape = fit_layers(modal, "first")
        self._canonicalize(fits)

        # --- the glide, now that the excitation scale is known ---------------
        tension_stage = TensionStage(self.sr, settings.control_period)
        tension, per_layer_k, tension_notes = tension_stage.run(
            modal.modes, fits, [layer.audio for layer in layers], progress=progress
        )
        warnings.extend(tension_notes)
        modal = ModalFit(
            modes=modal.modes, tension=tension, descriptors=modal.descriptors,
            notes=modal.notes, damping_anchors=modal.damping_anchors,
        )
        progress({"phase": "tension", "k": tension.k, "tau": tension.tau,
                  "per_layer": per_layer_k,
                  "glide_semitones": 12.0 * np.log2(1.0 + tension.k)
                  if tension.k > 0 else 0.0})

        if tension.k > 0:
            fits, shape = fit_layers(modal, "second", seed_fits=fits)
            self._canonicalize(fits)

        progress({"phase": "stage2_done", "table": [fit.to_row() for fit in fits]})

        # --- stage 3: the experiment ----------------------------------------
        inspection = InspectionStage().run(fits, per_layer_k, progress=progress)
        progress({"phase": "stage3", **inspection.to_dict()})
        if not inspection.passed:
            warnings.append(
                "stage 3 did not pass: " + inspection.verdict + ". The curve "
                "below still fits, because per-velocity fitting always does — "
                "read the findings before trusting it"
            )

        # --- stage 4: velocity becomes continuous ---------------------------
        curve = VelocityCurveStage().run(fits, progress=progress)
        progress({"phase": "stage4", "curve": curve.to_dict()})

        # `shape` came out of the reference layer's free fit above: the part of
        # the excitation that velocity does NOT explain.

        # --- stage 5: joint refinement --------------------------------------
        joint = JointStage(self.sr, settings.control_period)
        curve, generations = joint.run(
            modal, shape, curve,
            [(layer.velocity_normalized, layer.audio) for layer in layers],
            bands,
            generations=settings.generations,
            population=settings.population,
            workers=settings.workers,
            progress=progress,
            seed=settings.seed,
        )
        progress({"phase": "stage5_done", "generations": len(generations),
                  "loss": curve.loss})

        params = self._assemble(modal, shape, curve, bands, reference, target.drum)
        result = FitResult(
            drum=target.drum, params=params, modal=modal, excitation=fits,
            inspection=inspection, curve=curve, generations=generations,
            shape=shape, reference_velocity=reference.velocity_normalized,
            elapsed=time.perf_counter() - started, settings=settings,
            warnings=warnings,
        )
        progress({"phase": "done", "elapsed": result.elapsed})
        return result

    # -- pieces ---------------------------------------------------------------

    @staticmethod
    def _canonicalize(fits: Sequence[ExcitationFit]) -> None:
        """Put the gains on a canonical energy scale, in place.

        `ratio = 1 + k * energy` means k and the absolute size of the gains are
        not separately identifiable from audio: a recording's level is a mic
        preamp setting, and only the PRODUCT of k and the bank's energy shows up
        in the glide. `output_gain` sits after the bank and cannot fix it,
        because it does not change the energy the tension block sees.

        So the scale is fixed by convention instead of by measurement: the
        loudest layer gets unit bank energy, every layer is scaled by the same
        factor so their relative strengths survive, and `output_gain` takes up
        the slack. k is then comparable across drums and reads as the peak
        frequency ratio minus one — the same convention the presets use.
        """
        if not fits:
            return
        peak = max(float(np.sum(fit.gains**2)) for fit in fits)
        if peak <= 0:
            return
        scale = 1.0 / np.sqrt(peak)
        for fit in fits:
            fit.gains = fit.gains * scale
            fit.levels = fit.levels * scale
            fit.output_gain = fit.output_gain / scale
            fit.amplitude = float(np.sqrt(np.sum(fit.gains**2)))

    def _noise_bands(self, count: int) -> list[NoiseBand]:
        chosen = DrumTrainer.DEFAULT_BANDS[: max(0, count)]
        return [
            NoiseBand(low, high, 0.01, t60) for low, high, t60 in chosen
        ]

    def _assemble(self, modal, shape, curve, bands, reference, drum) -> DrumParams:
        freqs = np.array([mode.f_static for mode in modal.modes])
        velocity = reference.velocity_normalized
        gains = curve.gains(freqs, shape, velocity)
        levels = curve.levels(velocity)

        params = DrumParams(
            modes=[
                Mode(mode.f_static, float(max(gain, 1e-9)), mode.t60)
                for mode, gain in zip(modal.modes, gains)
            ],
            noise=[
                NoiseBand(band.f_low, band.f_high, float(max(level, 0.0)), band.t60)
                for band, level in zip(bands, levels)
            ],
            tension=modal.tension,
            output_gain=1.0,
            name=drum,
        )
        # One render fixes the absolute level; there is no reason to search for
        # a number that arithmetic gives exactly.
        rendered = DrumVoice(params, self.sr, self.settings.control_period,
                             seed=0).render_hit(self.settings.seconds)
        params.output_gain = float(
            LevelMatch.match_rms(rendered, reference.audio[: len(rendered)])
        )
        params.validate(self.sr)
        return params


class FitEvaluator:
    """Scores a fit the way the architecture says to: on the whole set, worst
    component first, with the real ScoreCard rather than the training loss.

    §6.5 is explicit that a single scalar says "worse", not "which of the 109
    numbers". The scalar is what the optimizer descends; this is what tells you
    where you actually are.
    """

    def __init__(self, sr: int = Audio.DEFAULT_SR, control_period: int = 64) -> None:
        self.sr = int(sr)
        self.control_period = int(control_period)

    def render(self, result: FitResult, velocity_normalized: float,
               seconds: float) -> np.ndarray:
        params = result.params_at(velocity_normalized, self.sr)
        return DrumVoice(params, self.sr, self.control_period, seed=0).render_hit(seconds)

    def score(self, result: FitResult, target: FitTarget,
              progress: Progress = _noop) -> tuple[ScoreCard, list[tuple[float, ScoreCard]]]:
        f0 = result.modal.modes[0].f_static if result.modal.modes else 100.0
        scorer = DrumScorer.for_fundamental(f0, self.sr)

        cards: list[tuple[float, ScoreCard]] = []
        for index, layer in enumerate(target.layers[: result.settings.max_layers]):
            progress({"phase": "scoring", "layer": index + 1,
                      "of": min(len(target.layers), result.settings.max_layers),
                      "velocity": layer.velocity})
            generated = self.render(
                result, layer.velocity_normalized, len(layer.audio) / self.sr
            )
            cards.append((layer.velocity, scorer.score(layer.audio, generated)))

        aggregate = scorer.aggregate([card for _, card in cards])
        progress({"phase": "scored", "total": aggregate.total})
        return aggregate, cards
