"""Mapping ScoreCard entries back onto DrumParams fields.

This is the point of the whole scoring module. Without `DrumParams` the
ScoreCard still says which COMPONENT is wrong; with it you get the index of the
specific mode and a numeric correction:

    "mode near 197 Hz has t60 40% too long"  ->  modes[7].t60 = 0.52

Frequency and t60 corrections are direct, because the measured reference value
IS the target — the descriptors were extracted with the same chain that would
be used to verify the fix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.constants import Cents, Decibels
from .comparator import ComponentScore, ScoreCard
from .matching import ModeMatch


@dataclass
class ParameterSuggestion:
    """A concrete edit to make."""

    target: str  # e.g. "modes[7].t60", "tension.k", "noise[2].level"
    current: float | None
    suggested: float | None
    reason: str
    confidence: float
    priority: float  # how much of the total shortfall this accounts for

    def __str__(self) -> str:
        if self.current is None or self.suggested is None:
            return f"{self.target:<24} {self.reason}"
        return (
            f"{self.target:<24} {self.current:>10.4g} -> {self.suggested:<10.4g}  "
            f"{self.reason}"
        )

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "current": self.current,
            "suggested": self.suggested,
            "reason": self.reason,
            "confidence": self.confidence,
            "priority": self.priority,
        }


class Attributor:
    """Turns component scores into edits against a specific `DrumParams`."""

    #: Errors below these are not worth an edit — inside them, the fit is as
    #: good as the measurement.
    DEFAULT_TOLERANCES: dict[str, float] = {
        "mode_frequency_cents": 10.0,
        "mode_t60_ratio": 0.10,
        "mode_amplitude_db": 2.0,
        "glide_cents": 8.0,
        "noise_db": 3.0,
    }

    def __init__(self, tolerances: dict[str, float] | None = None) -> None:
        self.tolerances = dict(Attributor.DEFAULT_TOLERANCES)
        if tolerances:
            self.tolerances.update(tolerances)

    def attribute(self, card: ScoreCard, params) -> list[ParameterSuggestion]:
        """Every edit worth making, most important first.

        `params` is a `synth.DrumParams` — the parameters that produced the
        generated side. Passing the reference's parameters (if you somehow have
        them) produces confident nonsense.
        """
        suggestions = self.suggest_mode_edits(card.mode_matches, params)

        for name, builder in (
            ("glide", self.suggest_tension_edits),
            ("noise", self.suggest_noise_edits),
            ("envelope", self.suggest_envelope_edits),
        ):
            try:
                component = card.by_name(name)
            except KeyError:
                continue
            if component.available:
                suggestions.extend(builder(component, params))

        suggestions.sort(key=lambda suggestion: -suggestion.priority)
        return suggestions

    # -- modes ----------------------------------------------------------------

    def suggest_mode_edits(
        self, matches: list[ModeMatch], params
    ) -> list[ParameterSuggestion]:
        """Frequency and t60 corrections are direct — the measured value IS the
        target. Missing modes become 'add a Mode at f=..., t60=..., gain=...'."""
        suggestions: list[ParameterSuggestion] = []
        loudest = max((match.weight for match in matches), default=1.0) or 1.0

        for match in matches:
            weight = match.weight / loudest

            if match.is_missing:
                mode = match.reference
                suggestions.append(
                    ParameterSuggestion(
                        target=f"modes[+] add at {mode.freq:.1f} Hz",
                        current=None,
                        suggested=mode.freq,
                        reason=(
                            f"reference has a partial at {mode.freq:.1f} Hz "
                            f"({mode.amplitude_db():.1f} dB, t60 {mode.t60:.3f} s) "
                            "with no counterpart"
                        ),
                        confidence=mode.confidence,
                        priority=weight * 1.5,
                    )
                )
                continue

            if match.is_spurious:
                mode = match.generated
                index = params.nearest_mode_index(mode.freq)
                suggestions.append(
                    ParameterSuggestion(
                        target=f"modes[{index}] remove" if index is not None else "modes[-]",
                        current=mode.freq,
                        suggested=None,
                        reason=(
                            f"generated partial at {mode.freq:.1f} Hz "
                            f"({mode.amplitude_db():.1f} dB) has no reference counterpart"
                        ),
                        confidence=mode.confidence,
                        priority=weight * 0.75,
                    )
                )
                continue

            index = params.nearest_mode_index(match.generated.freq)
            if index is None:
                continue
            current = params.modes[index]

            if abs(match.freq_error_cents) > self.tolerances["mode_frequency_cents"]:
                ratio = Cents.to_ratio(-match.freq_error_cents)
                suggestions.append(
                    ParameterSuggestion(
                        target=f"modes[{index}].f_static",
                        current=current.f_static,
                        suggested=current.f_static * ratio,
                        reason=f"{match.freq_error_cents:+.1f} cents at {match.freq:.1f} Hz",
                        confidence=match.generated.confidence,
                        priority=weight * abs(match.freq_error_cents) / 100.0,
                    )
                )

            t60_error = match.t60_error_ratio - 1.0
            if abs(t60_error) > self.tolerances["mode_t60_ratio"]:
                suggestions.append(
                    ParameterSuggestion(
                        target=f"modes[{index}].t60",
                        current=current.t60,
                        suggested=current.t60 / match.t60_error_ratio,
                        reason=(
                            f"t60 is {t60_error * 100:+.0f}% at {match.freq:.1f} Hz "
                            f"(reference {match.reference.t60:.3f} s)"
                        ),
                        confidence=match.generated.confidence,
                        priority=weight * abs(t60_error),
                    )
                )

            if abs(match.amplitude_error_db) > self.tolerances["mode_amplitude_db"]:
                suggestions.append(
                    ParameterSuggestion(
                        target=f"modes[{index}].gain",
                        current=current.gain,
                        suggested=current.gain
                        * Decibels.to_amplitude(-match.amplitude_error_db),
                        reason=f"{match.amplitude_error_db:+.1f} dB at {match.freq:.1f} Hz",
                        confidence=match.generated.confidence,
                        priority=weight * abs(match.amplitude_error_db) / 12.0,
                    )
                )

        return suggestions

    # -- tension --------------------------------------------------------------

    def suggest_tension_edits(
        self, glide: ComponentScore, params
    ) -> list[ParameterSuggestion]:
        """Glide depth wrong -> tension.k. Settle time wrong -> tension.tau.

        These are independent, which is why the glide is scored as a trajectory
        rather than a single depth number: a depth-only score cannot tell them
        apart, and adjusting k to fix a tau problem gets both wrong.
        """
        suggestions: list[ParameterSuggestion] = []
        detail = glide.detail
        reference_depth = detail.get("reference_depth_cents")
        generated_depth = detail.get("generated_depth_cents")

        if reference_depth and generated_depth and abs(
            generated_depth - reference_depth
        ) > self.tolerances["glide_cents"]:
            # ratio - 1 is proportional to k, so the depths scale with k
            # directly — except when k is already zero, where there is nothing
            # to scale. Then fall back to the ratio the reference implies,
            # which is exact when the mode gains carry unit energy
            # (see ExcitationTilt.normalized) and the right order of magnitude
            # otherwise.
            reference_ratio = Cents.to_ratio(reference_depth) - 1.0
            generated_ratio = Cents.to_ratio(generated_depth) - 1.0
            scalable = params.tension.k > 0.0 and generated_ratio > 1e-6

            if scalable:
                suggested = params.tension.k * (reference_ratio / generated_ratio)
                reason = (
                    f"glide depth {generated_depth:.0f} cents against the "
                    f"reference's {reference_depth:.0f}"
                )
                confidence = 0.8
            elif reference_ratio > 1e-6:
                suggested = reference_ratio
                reason = (
                    f"reference glides {reference_depth:.0f} cents and the "
                    "generated hit barely moves; k cannot be scaled up from "
                    "zero, so this is the ratio the reference implies"
                )
                confidence = 0.5
            else:
                suggested = 0.0
                reason = (
                    f"generated hit glides {generated_depth:.0f} cents and the "
                    "reference does not; turn the tension feedback off"
                )
                confidence = 0.7

            suggestions.append(
                ParameterSuggestion(
                    target="tension.k",
                    current=params.tension.k,
                    suggested=float(suggested),
                    reason=reason,
                    confidence=confidence,
                    priority=1.0 - glide.value,
                )
            )

        reference_settle = detail.get("reference_settle_s")
        generated_settle = detail.get("generated_settle_s")
        if (
            reference_settle
            and generated_settle
            and reference_settle > 0
            and abs(generated_settle / reference_settle - 1.0) > 0.25
        ):
            scale = reference_settle / generated_settle
            suggestions.append(
                ParameterSuggestion(
                    target="tension.tau",
                    current=params.tension.tau,
                    suggested=params.tension.tau * scale,
                    reason=(
                        f"glide settles in {generated_settle:.2f}s against the "
                        f"reference's {reference_settle:.2f}s"
                    ),
                    confidence=0.6,
                    priority=0.5 * (1.0 - glide.value),
                )
            )
        return suggestions

    # -- noise ----------------------------------------------------------------

    def suggest_noise_edits(
        self, noise: ComponentScore, params
    ) -> list[ParameterSuggestion]:
        """Band energy error -> that band's level. Slope error -> its t60."""
        suggestions: list[ParameterSuggestion] = []
        for label, error_db in noise.detail.items():
            if not isinstance(error_db, (int, float)):
                continue
            if abs(error_db) <= self.tolerances["noise_db"]:
                continue
            index = self._nearest_noise_band(label, params)
            if index is None:
                continue
            band = params.noise[index]
            suggestions.append(
                ParameterSuggestion(
                    target=f"noise[{index}].level",
                    current=band.level,
                    suggested=band.level * Decibels.to_amplitude(-float(error_db)),
                    reason=f"{label} band is {error_db:+.1f} dB against the reference",
                    confidence=0.5,
                    priority=0.4 * min(abs(float(error_db)) / 12.0, 1.0),
                )
            )
        return suggestions

    @staticmethod
    def _nearest_noise_band(label: str, params) -> int | None:
        """Map a '200-800Hz' detail key onto the closest configured band."""
        try:
            low_text, high_text = label.replace("Hz", "").split("-")
            center = 0.5 * (float(low_text) + float(high_text))
        except (ValueError, AttributeError):
            return None
        if not params.noise:
            return None
        centers = np.array([0.5 * (b.f_low + b.f_high) for b in params.noise])
        return int(np.argmin(np.abs(centers - center)))

    # -- envelope -------------------------------------------------------------

    def suggest_envelope_edits(
        self, envelope: ComponentScore, params
    ) -> list[ParameterSuggestion]:
        """The envelope score does not map onto a single parameter.

        Reported as a note rather than a number: an attack mismatch means the
        excitation model is wrong, and the fix is `contact_time` — which does
        not exist yet. Scaling gains to chase it would be fitting around a
        missing mechanism.
        """
        detail = envelope.detail
        reference_peak = detail.get("reference_peak_time_ms")
        generated_peak = detail.get("generated_peak_time_ms")
        if reference_peak is None or generated_peak is None:
            return []
        if abs(generated_peak - reference_peak) < 3.0:
            return []
        return [
            ParameterSuggestion(
                target="excitation.contact_time",
                current=None,
                suggested=None,
                reason=(
                    f"envelope peaks at {generated_peak:.1f} ms against the "
                    f"reference's {reference_peak:.1f} ms; impulse excitation "
                    "cannot move this peak, it needs a finite force pulse"
                ),
                confidence=0.9,
                priority=0.6 * (1.0 - envelope.value),
            )
        ]
