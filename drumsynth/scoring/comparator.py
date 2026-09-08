"""Turning two descriptor sets into a ScoreCard.

Weights are the honest part of this module — they encode what you think
matters, and the defaults below are a starting guess, not a result. Modal
frequency and band decay carry the most perceptual weight; noise statistics the
least, because the ear is far more tolerant of a wrong noise spectrum than of a
wrong pitch or a wrong decay time.

Every component reports its error in its own units alongside the normalized
0-1 value, because "0.62" is not actionable and "modes average 34 cents sharp"
is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.constants import Cents, Decibels
from .descriptors import SoundDescriptors
from .matching import ModeMatch, ModeMatcher


@dataclass
class ComponentScore:
    """Score for one aspect of the sound. Native units plus a normalized value."""

    name: str
    value: float  # normalized, 1.0 = perfect
    raw_error: float  # in the component's own units
    unit: str
    detail: dict = field(default_factory=dict)
    parameters: list[str] = field(default_factory=list)  # which synth params to touch

    #: False when there was nothing to compare — an empty component is excluded
    #: from the total rather than counted as zero, which would punish a signal
    #: for a descriptor the analysis could not extract from either side.
    available: bool = True

    def __str__(self) -> str:
        if not self.available:
            return f"{self.name:<18} {'--':>7}   (no data)"
        return (
            f"{self.name:<18} {self.value:7.3f}   "
            f"{self.raw_error:9.3f} {self.unit}"
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "raw_error": self.raw_error,
            "unit": self.unit,
            "available": self.available,
            "parameters": list(self.parameters),
            "detail": {
                key: (float(val) if isinstance(val, (int, float, np.floating)) else val)
                for key, val in self.detail.items()
            },
        }


@dataclass
class ScoreCard:
    """Full result. Per-component so you know WHERE to look, plus one total."""

    total: float
    components: list[ComponentScore]
    mode_matches: list[ModeMatch] = field(default_factory=list)
    stft_loss: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def worst(self, n: int = 3) -> list[ComponentScore]:
        """The n components most responsible for the shortfall. Start here."""
        available = [component for component in self.components if component.available]
        return sorted(available, key=lambda component: component.value)[:n]

    def by_name(self, name: str) -> ComponentScore:
        for component in self.components:
            if component.name == name:
                return component
        raise KeyError(f"no component named {name!r}; have {[c.name for c in self.components]}")

    def missing_modes(self) -> list[ModeMatch]:
        return [match for match in self.mode_matches if match.is_missing]

    def spurious_modes(self) -> list[ModeMatch]:
        return [match for match in self.mode_matches if match.is_spurious]

    def report(self) -> str:
        """Human-readable table."""
        from .report import ScoreReport

        return ScoreReport(self).text()

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "stft_loss": self.stft_loss,
            "components": [component.to_dict() for component in self.components],
            "modes": {
                "matched": sum(1 for m in self.mode_matches if m.is_matched),
                "missing": len(self.missing_modes()),
                "spurious": len(self.spurious_modes()),
            },
            "warnings": list(self.warnings),
        }


class Comparator:
    """Scores one `SoundDescriptors` against another."""

    DEFAULT_WEIGHTS: dict[str, float] = {
        "mode_frequency": 0.25,
        "mode_t60": 0.20,
        "mode_amplitude": 0.10,
        "mode_completeness": 0.10,  # missing / spurious modes
        "band_decay": 0.15,
        "glide": 0.10,
        "envelope": 0.05,
        "noise": 0.05,
    }

    #: Error at or above these values scores 0 for that component.
    DEFAULT_TOLERANCES: dict[str, float] = {
        "mode_frequency_cents": 20.0,
        "mode_t60_ratio": 0.15,
        "mode_amplitude_db": 3.0,
        "band_slope_pct": 10.0,
        "glide_cents": 10.0,
        "envelope_db": 1.0,
        "noise_db": 6.0,
        "noise_slope_pct": 30.0,
    }

    #: How a native-unit error maps onto 0-1.
    #:
    #: "linear" is the architecture's contract: 1.0 at zero error, 0.0 at the
    #: tolerance, nothing below. It is the right default because the tolerances
    #: are acceptance thresholds — past them the component is simply wrong, and
    #: how wrong is a question for `raw_error`, which is reported in native
    #: units alongside every component.
    #:
    #: "soft" is exp(-error / tolerance): 0.37 at the tolerance, asymptotic to
    #: zero, never flat. Use it when something automated is following the
    #: gradient, because a component pinned at 0.0 gives an optimizer nothing
    #: to descend.
    NORMALIZATIONS: tuple[str, ...] = ("linear", "soft")

    #: A missing mode costs this much more than a spurious one. Adding a partial
    #: that should not be there is audible; leaving one out is worse, because
    #: nothing else in the model can stand in for it.
    MISSING_PENALTY: float = 2.0

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        tolerances: dict[str, float] | None = None,
        matcher: ModeMatcher | None = None,
        normalization: str = "linear",
    ) -> None:
        if normalization not in Comparator.NORMALIZATIONS:
            raise ValueError(
                f"normalization must be one of {Comparator.NORMALIZATIONS}, "
                f"got {normalization!r}"
            )
        self.normalization = normalization
        self.weights = dict(Comparator.DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        self.tolerances = dict(Comparator.DEFAULT_TOLERANCES)
        if tolerances:
            self.tolerances.update(tolerances)
        self.matcher = matcher or ModeMatcher()

    # -- top level ------------------------------------------------------------

    def compare(
        self, reference: SoundDescriptors, generated: SoundDescriptors
    ) -> ScoreCard:
        matches = self.matcher.match(reference.modes, generated.modes)
        components = [
            self.score_mode_frequency(matches),
            self.score_mode_t60(matches),
            self.score_mode_amplitude(matches),
            self.score_mode_completeness(matches),
            self.score_band_decay(reference, generated),
            self.score_glide(reference, generated),
            self.score_envelope(reference, generated),
            self.score_noise(reference, generated),
        ]

        warnings = self._warnings(reference, generated, components)
        return ScoreCard(
            total=self.total(components),
            components=components,
            mode_matches=matches,
            warnings=warnings,
        )

    def total(self, components: list[ComponentScore]) -> float:
        """Weighted mean over the components that had data.

        Unavailable components are dropped and the remaining weights are
        renormalized, rather than scored as zero: a reference whose glide could
        not be tracked should not drag the total down for a synthesis that may
        well be right.
        """
        available = [c for c in components if c.available and np.isfinite(c.value)]
        if not available:
            return 0.0
        weights = np.array([self.weights.get(c.name, 0.0) for c in available])
        values = np.array([c.value for c in available])
        return float(np.sum(weights * values) / np.sum(weights)) if weights.sum() > 0 else 0.0

    # -- normalization --------------------------------------------------------

    def _normalize(self, error, tolerance: float):
        """Map a native-unit error onto 0-1. See NORMALIZATIONS."""
        magnitude = np.abs(np.asarray(error, dtype=float))
        if self.normalization == "soft":
            return np.exp(-magnitude / tolerance)
        return np.clip(1.0 - magnitude / tolerance, 0.0, 1.0)

    @staticmethod
    def _weighted(values: np.ndarray, weights: np.ndarray) -> float:
        total = float(np.sum(weights))
        return float(np.sum(values * weights) / total) if total > 0 else 0.0

    # -- individual components ------------------------------------------------

    def score_mode_frequency(self, matches: list[ModeMatch]) -> ComponentScore:
        """Amplitude-weighted mean frequency error in cents.

        Weighting by amplitude is the whole point: a 50-cent error on a -55 dB
        partial is inaudible, and the same error on the fundamental is not.
        """
        paired = [match for match in matches if match.is_matched]
        if not paired:
            return ComponentScore("mode_frequency", 0.0, np.nan, "cents",
                                  available=False, parameters=["modes[].f_static"])

        errors = np.array([match.freq_error_cents for match in paired])
        weights = np.array([match.weight for match in paired])
        tolerance = self.tolerances["mode_frequency_cents"]
        value = self._weighted(self._normalize(errors, tolerance), weights)

        worst = paired[int(np.argmax(np.abs(errors) * weights))]
        return ComponentScore(
            name="mode_frequency",
            value=value,
            raw_error=self._weighted(np.abs(errors), weights),
            unit="cents",
            detail={
                "max_error_cents": float(np.max(np.abs(errors))),
                "worst_mode_hz": worst.freq,
                "n_modes": len(paired),
            },
            parameters=["modes[].f_static", "tuning"],
        )

    def score_mode_t60(self, matches: list[ModeMatch]) -> ComponentScore:
        paired = [match for match in matches if match.is_matched]
        if not paired:
            return ComponentScore("mode_t60", 0.0, np.nan, "ratio",
                                  available=False, parameters=["modes[].t60"])

        ratios = np.array([match.t60_error_ratio for match in paired])
        errors = np.abs(ratios - 1.0)
        weights = np.array([match.weight for match in paired])
        value = self._weighted(self._normalize(errors, self.tolerances["mode_t60_ratio"]),
                               weights)

        worst = paired[int(np.argmax(errors * weights))]
        return ComponentScore(
            name="mode_t60",
            value=value,
            raw_error=self._weighted(errors, weights),
            unit="ratio",
            detail={
                "max_error_ratio": float(np.max(errors)),
                "worst_mode_hz": worst.freq,
                "worst_ratio": float(worst.t60_error_ratio),
            },
            parameters=["modes[].t60"],
        )

    def score_mode_amplitude(self, matches: list[ModeMatch]) -> ComponentScore:
        paired = [match for match in matches if match.is_matched]
        if not paired:
            return ComponentScore("mode_amplitude", 0.0, np.nan, "dB",
                                  available=False, parameters=["modes[].gain"])

        errors = np.array([match.amplitude_error_db for match in paired])
        weights = np.array([match.weight for match in paired])
        value = self._weighted(self._normalize(errors, self.tolerances["mode_amplitude_db"]),
                               weights)
        return ComponentScore(
            name="mode_amplitude",
            value=value,
            raw_error=self._weighted(np.abs(errors), weights),
            unit="dB",
            detail={
                "mean_offset_db": float(np.mean(errors)),
                "max_error_db": float(np.max(np.abs(errors))),
            },
            parameters=["modes[].gain", "output_gain"],
        )

    def score_mode_completeness(self, matches: list[ModeMatch]) -> ComponentScore:
        """Penalize missing and spurious modes. Missing hurts more."""
        if not matches:
            return ComponentScore("mode_completeness", 0.0, np.nan, "modes",
                                  available=False, parameters=["modes"])

        missing = [match for match in matches if match.is_missing]
        spurious = [match for match in matches if match.is_spurious]
        n_reference = sum(1 for match in matches if match.reference is not None)

        # Weight by amplitude: a missing -60 dB partial is not the same failure
        # as a missing fundamental.
        missing_weight = sum(match.weight for match in missing)
        spurious_weight = sum(match.weight for match in spurious)
        reference_weight = sum(
            match.weight for match in matches if match.reference is not None
        )

        if reference_weight <= 0:
            return ComponentScore("mode_completeness", 0.0, float(len(spurious)), "modes",
                                  available=False, parameters=["modes"])

        penalty = (
            Comparator.MISSING_PENALTY * missing_weight + spurious_weight
        ) / (Comparator.MISSING_PENALTY * reference_weight)

        return ComponentScore(
            name="mode_completeness",
            value=float(np.clip(1.0 - penalty, 0.0, 1.0)),
            raw_error=float(len(missing) + len(spurious)),
            unit="modes",
            detail={
                "missing": len(missing),
                "spurious": len(spurious),
                "reference_modes": n_reference,
                "missing_hz": [round(match.freq, 1) for match in missing[:8]],
                "spurious_hz": [round(match.freq, 1) for match in spurious[:8]],
            },
            parameters=["modes"],
        )

    def score_band_decay(
        self, reference: SoundDescriptors, generated: SoundDescriptors
    ) -> ComponentScore:
        """Per-band decay slope, compared as a percentage error.

        The robust half of the decay score. Where modal extraction is
        unreliable — dense high bands, poor SNR — this still works, and a band
        slope that matches is strong evidence the modes inside it do too.
        """
        pairs = []
        for reference_band in reference.bands:
            generated_band = generated.band(reference_band.f_low, reference_band.f_high)
            if generated_band is None:
                continue
            if not (reference_band.is_valid and generated_band.is_valid):
                continue
            pairs.append((reference_band, generated_band))

        if not pairs:
            return ComponentScore("band_decay", 0.0, np.nan, "%",
                                  available=False, parameters=["modes[].t60"])

        errors, weights, detail = [], [], {}
        for reference_band, generated_band in pairs:
            error_pct = 100.0 * abs(
                generated_band.slope_db_s / reference_band.slope_db_s - 1.0
            )
            errors.append(error_pct)
            # Louder bands matter more, and a straight reference fit is more
            # trustworthy evidence than a curved one.
            weights.append(
                Decibels.to_amplitude(reference_band.level_db)
                * max(reference_band.r_squared, 0.1)
            )
            detail[f"{reference_band.f_low:.0f}-{reference_band.f_high:.0f}Hz"] = round(
                generated_band.t60 / reference_band.t60, 3
            ) if reference_band.t60 > 0 else np.nan

        errors = np.array(errors)
        weights = np.array(weights)
        value = self._weighted(self._normalize(errors, self.tolerances["band_slope_pct"]),
                               weights)
        return ComponentScore(
            name="band_decay",
            value=value,
            raw_error=self._weighted(errors, weights),
            unit="%",
            detail=detail,
            parameters=["modes[].t60", "noise[].t60"],
        )

    def score_glide(
        self, reference: SoundDescriptors, generated: SoundDescriptors
    ) -> ComponentScore:
        """Compare the full f0(t) trajectory in cents, not just the depth.

        Depth alone is satisfiable by the wrong mechanism — a fixed pitch
        envelope reproduces the depth and gets the shape wrong — and comparing
        the trajectory is what separates `tension.k` (depth) from `tension.tau`
        (settle time).
        """
        if reference.glide is None or generated.glide is None:
            return ComponentScore("glide", 0.0, np.nan, "cents",
                                  available=False, parameters=["tension.k", "tension.tau"])

        ref_times, ref_freqs = reference.glide.reliable_track()
        gen_times, _ = generated.glide.reliable_track()
        if ref_times.size < 2 or gen_times.size < 2:
            return ComponentScore("glide", 0.0, np.nan, "cents",
                                  available=False, parameters=["tension.k", "tension.tau"])

        # Common grid over the span both tracks actually cover.
        start = max(ref_times[0], gen_times[0])
        stop = min(ref_times[-1], gen_times[-1])
        if stop <= start:
            return ComponentScore("glide", 0.0, np.nan, "cents",
                                  available=False, parameters=["tension.k", "tension.tau"])

        grid = np.linspace(start, stop, 64)
        errors = Cents.between(generated.glide.at(grid), reference.glide.at(grid))
        errors = errors[np.isfinite(errors)]
        if errors.size == 0:
            return ComponentScore("glide", 0.0, np.nan, "cents",
                                  available=False, parameters=["tension.k", "tension.tau"])

        rms_cents = float(np.sqrt(np.mean(errors**2)))
        value = float(self._normalize(rms_cents, self.tolerances["glide_cents"]))
        return ComponentScore(
            name="glide",
            value=value,
            raw_error=rms_cents,
            unit="cents",
            detail={
                "reference_depth_cents": reference.glide.depth_cents,
                "generated_depth_cents": generated.glide.depth_cents,
                "depth_error_cents": generated.glide.depth_cents - reference.glide.depth_cents,
                "reference_settle_s": reference.glide.settle_time,
                "generated_settle_s": generated.glide.settle_time,
                "max_error_cents": float(np.max(np.abs(errors))),
            },
            parameters=["tension.k", "tension.tau"],
        )

    def score_envelope(
        self, reference: SoundDescriptors, generated: SoundDescriptors
    ) -> ComponentScore:
        if reference.envelope is None or generated.envelope is None:
            return ComponentScore("envelope", 0.0, np.nan, "dB",
                                  available=False, parameters=["modes[].gain", "noise[].level"])

        ref_env, gen_env = reference.envelope, generated.envelope
        floor = ref_env.peak_db() + min(reference.noise_floor_db, -6.0)
        mask = ref_env.above(floor)
        times = ref_env.times[mask]
        if times.size < 4:
            return ComponentScore("envelope", 0.0, np.nan, "dB",
                                  available=False, parameters=["modes[].gain"])

        # Both curves are re-referenced to their own peaks: the level offset
        # belongs to mode_amplitude, and counting it twice double-penalizes it.
        errors = (gen_env.at(times) - gen_env.peak_db()) - (
            ref_env.db[mask] - ref_env.peak_db()
        )
        errors = errors[np.isfinite(errors)]
        rms_db = float(np.sqrt(np.mean(errors**2))) if errors.size else np.nan
        value = float(self._normalize(rms_db, self.tolerances["envelope_db"]))
        return ComponentScore(
            name="envelope",
            value=value,
            raw_error=rms_db,
            unit="dB",
            detail={
                "reference_peak_time_ms": ref_env.peak_time * 1000.0,
                "generated_peak_time_ms": gen_env.peak_time * 1000.0,
                "reference_rise_ms": ref_env.rise_10_90 * 1000.0,
                "generated_rise_ms": gen_env.rise_10_90 * 1000.0,
                "crest_error_db": gen_env.crest_factor_db - ref_env.crest_factor_db,
            },
            parameters=["modes[].gain", "noise[].level", "contact_time"],
        )

    def score_noise(
        self, reference: SoundDescriptors, generated: SoundDescriptors
    ) -> ComponentScore:
        """Statistics only: band energies and band decay rates. Never sample-wise."""
        if reference.noise is None or generated.noise is None:
            return ComponentScore("noise", 0.0, np.nan, "dB",
                                  available=False, parameters=["noise[]"])

        shared = [
            band for band in reference.noise.band_energy_db
            if band in generated.noise.band_energy_db
        ]
        if not shared:
            return ComponentScore("noise", 0.0, np.nan, "dB",
                                  available=False, parameters=["noise[]"])

        energy_errors, slope_errors, detail = [], [], {}
        for band in shared:
            energy_error = (
                generated.noise.band_energy_db[band] - reference.noise.band_energy_db[band]
            )
            energy_errors.append(energy_error)
            detail[f"{band[0]:.0f}-{band[1]:.0f}Hz"] = round(float(energy_error), 2)

            ref_slope = reference.noise.band_slope_db_s.get(band, np.nan)
            gen_slope = generated.noise.band_slope_db_s.get(band, np.nan)
            if np.isfinite(ref_slope) and np.isfinite(gen_slope) and ref_slope != 0:
                slope_errors.append(100.0 * abs(gen_slope / ref_slope - 1.0))

        energy_errors = np.array(energy_errors)
        energy_value = float(
            np.mean(self._normalize(energy_errors, self.tolerances["noise_db"]))
        )
        if slope_errors:
            slope_value = float(
                np.mean(self._normalize(np.array(slope_errors),
                                        self.tolerances["noise_slope_pct"]))
            )
            value = 0.6 * energy_value + 0.4 * slope_value
        else:
            value = energy_value

        return ComponentScore(
            name="noise",
            value=value,
            raw_error=float(np.mean(np.abs(energy_errors))),
            unit="dB",
            detail=detail,
            parameters=["noise[].level", "noise[].t60"],
        )

    # -- guard rails ----------------------------------------------------------

    def _warnings(
        self,
        reference: SoundDescriptors,
        generated: SoundDescriptors,
        components: list[ComponentScore],
    ) -> list[str]:
        warnings = list(reference.warnings) + [
            f"generated: {warning}" for warning in generated.warnings
        ]

        if reference.sr != generated.sr:
            warnings.append(
                f"sample rates differ ({reference.sr} vs {generated.sr}); every "
                "frequency comparison below is suspect"
            )
        if generated.duration < 0.5 * reference.duration:
            warnings.append(
                f"generated signal is {generated.duration:.2f}s against the "
                f"reference's {reference.duration:.2f}s — render a longer tail "
                "before trusting any decay score"
            )
        if np.isfinite(reference.noise_floor_db) and reference.noise_floor_db > -50.0:
            warnings.append(
                f"reference noise floor is high ({reference.noise_floor_db:.1f} dB "
                "below peak); decay fits stop early and t60 estimates are the "
                "least trustworthy number on this card"
            )
        unavailable = [c.name for c in components if not c.available]
        if unavailable:
            warnings.append(
                "no data for: " + ", ".join(unavailable) + " (excluded from the total)"
            )
        return warnings
