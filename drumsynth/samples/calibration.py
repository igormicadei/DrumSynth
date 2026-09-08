"""Mapping nominal velocity labels onto a physical 0-1 scale.

MIDI velocity is a controller value, not a unit of energy. Two consequences for
the fit:

  * the label spacing is not the energy spacing, so fitting excitation
    parameters against raw labels bakes the controller's response curve into
    the drum model
  * the curve differs per drum and per session

Fit a monotone curve of measured energy against label, then express every
sample on a normalized scale. If the relationship is NOT monotone, the session
has a problem — mislabeled takes, a moved mic, a drum retuned partway — and the
fitter should not paper over it.
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

from ..core.constants import Decibels


class VelocityCalibration:
    """Label -> physical 0-1 velocity, fitted from measured energy."""

    MODES: tuple[str, ...] = ("energy", "peak", "linear", "identity")

    #: A drop this large between consecutive labels is a real inversion rather
    #: than measurement scatter. In dB of measured energy.
    MONOTONE_TOLERANCE_DB: float = 1.0

    def __init__(
        self, mode: Literal["energy", "peak", "linear", "identity"] = "energy"
    ) -> None:
        if mode not in VelocityCalibration.MODES:
            raise ValueError(
                f"mode must be one of {VelocityCalibration.MODES}, got {mode!r}"
            )
        self.mode = mode
        self.labels: np.ndarray = np.zeros(0)
        self.measured_db: np.ndarray = np.zeros(0)
        self.normalized: np.ndarray = np.zeros(0)
        self.is_fitted: bool = False

    # -- fitting --------------------------------------------------------------

    def fit(self, samples: Sequence) -> "VelocityCalibration":
        """Build the label -> normalized-velocity map from measured energies."""
        if not samples:
            raise ValueError("cannot calibrate an empty sample set")

        labels = np.array([sample.velocity for sample in samples], dtype=np.float64)
        measured = np.array([self._measure(sample) for sample in samples])

        # Average duplicate labels (multiple takes) before fitting; otherwise
        # take-to-take scatter reads as non-monotonicity.
        unique_labels = np.unique(labels)
        averaged = np.array(
            [float(np.mean(measured[labels == label])) for label in unique_labels]
        )

        self.labels = unique_labels
        self.measured_db = averaged
        self.normalized = self._to_unit_scale(unique_labels, averaged)
        self.is_fitted = True
        return self

    def _measure(self, sample) -> float:
        """The physical quantity this calibration maps against, in dB."""
        if self.mode == "identity":
            return float(sample.velocity)
        if self.mode == "linear":
            return float(sample.velocity)
        if self.mode == "peak":
            return float(sample.peak_db())
        return float(Decibels.from_power(sample.energy()))

    def _to_unit_scale(self, labels: np.ndarray, measured: np.ndarray) -> np.ndarray:
        """Place each label on 0-1.

        `identity` and `linear` divide the label range; the measurement-driven
        modes place labels by where their energy sits between the softest and
        loudest hit, which is what makes the spacing physical.
        """
        if self.mode in ("identity", "linear"):
            span = labels.max() - labels.min()
            return (labels - labels.min()) / span if span > 0 else np.ones_like(labels)

        span = measured.max() - measured.min()
        if span <= 0:
            return np.ones_like(measured)
        return (measured - measured.min()) / span

    def apply(self, samples: Sequence) -> None:
        """Set velocity_normalized on each sample in place."""
        if not self.is_fitted:
            self.fit(samples)
        for sample in samples:
            sample.velocity_normalized = self.normalize(sample.velocity)

    # -- the map --------------------------------------------------------------

    def normalize(self, velocity: float) -> float:
        """Label -> 0-1. Interpolated between fitted labels, clamped outside."""
        if not self.is_fitted:
            raise RuntimeError("call fit() before normalize()")
        if self.labels.size == 1:
            return float(self.normalized[0])
        return float(np.interp(velocity, self.labels, self.normalized))

    def denormalize(self, velocity_normalized: float) -> float:
        """0-1 -> label. Only meaningful where the map is monotone."""
        if not self.is_fitted:
            raise RuntimeError("call fit() before denormalize()")
        if self.labels.size == 1:
            return float(self.labels[0])
        order = np.argsort(self.normalized)
        return float(
            np.interp(velocity_normalized, self.normalized[order], self.labels[order])
        )

    def is_monotone(self) -> bool:
        """False means the labels and the audio disagree. Investigate, don't fit.

        A harder hit that measured quieter is not noise to be smoothed over: it
        is a mislabeled take, a moved mic, or a drum that was retuned partway
        through the session, and every one of those invalidates the assumption
        that one `DrumParams` describes the whole set.
        """
        if not self.is_fitted or self.measured_db.size < 2:
            return True
        if self.mode in ("identity", "linear"):
            return True
        drops = np.diff(self.measured_db)
        return bool(np.all(drops > -VelocityCalibration.MONOTONE_TOLERANCE_DB))

    def inversions(self) -> list[tuple[float, float, float]]:
        """(lower label, higher label, dB drop) for every backwards step."""
        if not self.is_fitted or self.measured_db.size < 2:
            return []
        out = []
        for index in range(len(self.labels) - 1):
            drop = self.measured_db[index + 1] - self.measured_db[index]
            if drop <= -VelocityCalibration.MONOTONE_TOLERANCE_DB:
                out.append(
                    (float(self.labels[index]), float(self.labels[index + 1]), float(drop))
                )
        return out

    # -- reporting ------------------------------------------------------------

    def report(self) -> str:
        """Table of label | measured energy dB | normalized. Read before fitting."""
        if not self.is_fitted:
            return "VelocityCalibration (not fitted)"

        unit = "dB" if self.mode in ("energy", "peak") else "label"
        lines = [
            f"VelocityCalibration mode={self.mode}  "
            f"{'MONOTONE' if self.is_monotone() else '*** NOT MONOTONE ***'}",
            f"  {'label':>8} {'measured':>12} {'step':>8} {'normalized':>11}",
            "  " + "-" * 43,
        ]
        previous = None
        for label, measured, normalized in zip(
            self.labels, self.measured_db, self.normalized
        ):
            step = "" if previous is None else f"{measured - previous:+8.2f}"
            lines.append(
                f"  {label:8.1f} {measured:9.2f} {unit:<3}{step:>8} {normalized:11.4f}"
            )
            previous = measured

        for low, high, drop in self.inversions():
            lines.append(
                f"  ! v{low:.0f} -> v{high:.0f} DROPS {abs(drop):.1f} dB — "
                "harder hit measured quieter; check the take, the mic and the tuning"
            )
        return "\n".join(lines)

    def __str__(self) -> str:
        state = "fitted" if self.is_fitted else "unfitted"
        return f"VelocityCalibration({self.mode}, {state}, {self.labels.size} labels)"
