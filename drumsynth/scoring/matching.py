"""Pairing reference modes with generated modes.

This is an assignment problem, not a sort-and-zip. Sorting both lists by
frequency and pairing by index breaks catastrophically the first time a mode is
missing: every subsequent pair shifts by one and all the errors look enormous,
which tells you nothing about which mode is actually wrong.

Hungarian assignment on a cents-distance cost matrix, with a gate above which a
pair is refused and both sides are reported unmatched. Unmatched modes are not
a failure of the matcher — missing and spurious modes are their own score
component, and they are among the most actionable things the scorer reports.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..core.constants import Cents
from .descriptors import ModeEstimate


@dataclass
class ModeMatch:
    """One reference mode paired with its generated counterpart, or unpaired."""

    reference: ModeEstimate | None
    generated: ModeEstimate | None
    freq_error_cents: float = 0.0
    amplitude_error_db: float = 0.0
    t60_error_ratio: float = 1.0  # generated / reference

    @property
    def is_matched(self) -> bool:
        return self.reference is not None and self.generated is not None

    @property
    def is_missing(self) -> bool:
        """Reference mode with no generated counterpart — add a mode."""
        return self.reference is not None and self.generated is None

    @property
    def is_spurious(self) -> bool:
        """Generated mode with no reference counterpart — remove a mode."""
        return self.reference is None and self.generated is not None

    @property
    def weight(self) -> float:
        """Perceptual weight: the reference amplitude, or the generated one if
        the reference has no mode there.

        A 50-cent error on a -55 dB partial is inaudible; the same error on the
        fundamental is not, and an unweighted mean treats them identically.
        """
        if self.reference is not None:
            return float(self.reference.amplitude)
        return float(self.generated.amplitude) if self.generated is not None else 0.0

    @property
    def freq(self) -> float:
        """Where on the frequency axis this match sits, for reporting."""
        side = self.reference or self.generated
        return float(side.freq) if side is not None else float("nan")

    def __str__(self) -> str:
        if self.is_missing:
            return f"{self.freq:8.2f} Hz  MISSING   (reference only)"
        if self.is_spurious:
            return f"{self.freq:8.2f} Hz  SPURIOUS  (generated only)"
        return (
            f"{self.freq:8.2f} Hz  {self.freq_error_cents:+7.1f} cents  "
            f"{self.amplitude_error_db:+6.1f} dB  t60 x{self.t60_error_ratio:.3f}"
        )


class ModeMatcher:
    """Hungarian assignment on a cents-distance cost matrix."""

    def __init__(
        self, max_distance_cents: float = 150.0, amplitude_weight: float = 0.3
    ) -> None:
        """
        max_distance_cents: pairs farther apart than this are refused. Set it
            wider than the frequency tolerance being scored — a mode 100 cents
            out is a badly wrong mode, but it is still the same mode, and
            calling it "missing plus spurious" hides that.
        amplitude_weight: cents of cost per dB of amplitude difference. Keeps a
            loud reference mode from being matched to an inaudible generated one
            that happens to sit nearby.
        """
        self.max_distance_cents = float(max_distance_cents)
        self.amplitude_weight = float(amplitude_weight)

    def cost_matrix(
        self, reference: list[ModeEstimate], generated: list[ModeEstimate]
    ) -> np.ndarray:
        """Pairwise cost in cents-equivalent units."""
        if not reference or not generated:
            return np.zeros((len(reference), len(generated)), dtype=np.float64)

        ref_freqs = np.array([mode.freq for mode in reference])[:, None]
        gen_freqs = np.array([mode.freq for mode in generated])[None, :]
        ref_db = np.array([mode.amplitude_db() for mode in reference])[:, None]
        gen_db = np.array([mode.amplitude_db() for mode in generated])[None, :]

        distance = np.abs(Cents.between(gen_freqs, ref_freqs))
        return distance + self.amplitude_weight * np.abs(gen_db - ref_db)

    def match(
        self, reference: list[ModeEstimate], generated: list[ModeEstimate]
    ) -> list[ModeMatch]:
        """Pair the two lists, then report every unpaired mode on both sides."""
        matches: list[ModeMatch] = []
        if not reference and not generated:
            return matches

        paired_ref: set[int] = set()
        paired_gen: set[int] = set()

        if reference and generated:
            cost = self.cost_matrix(reference, generated)
            # Refused pairs get a cost the assignment will avoid unless forced,
            # and are filtered out afterwards.
            gated = np.where(cost <= self.max_distance_cents, cost, 1e6)
            rows, columns = linear_sum_assignment(gated)
            for row, column in zip(rows, columns):
                if gated[row, column] >= 1e6:
                    continue
                paired_ref.add(int(row))
                paired_gen.add(int(column))
                matches.append(self._build(reference[row], generated[column]))

        matches.extend(
            ModeMatch(reference=mode, generated=None)
            for index, mode in enumerate(reference)
            if index not in paired_ref
        )
        matches.extend(
            ModeMatch(reference=None, generated=mode)
            for index, mode in enumerate(generated)
            if index not in paired_gen
        )
        matches.sort(key=lambda match: match.freq)
        return matches

    @staticmethod
    def _build(reference: ModeEstimate, generated: ModeEstimate) -> ModeMatch:
        ratio = generated.t60 / reference.t60 if reference.t60 > 0 else np.inf
        return ModeMatch(
            reference=reference,
            generated=generated,
            freq_error_cents=float(Cents.between(generated.freq, reference.freq)),
            amplitude_error_db=float(generated.amplitude_db() - reference.amplitude_db()),
            t60_error_ratio=float(ratio),
        )
