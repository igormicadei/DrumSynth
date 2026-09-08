"""The static description of a drum: ~109 numbers, fully serializable.

`DrumParams` is the drum's identity. It holds no runtime state, so the same
instance can back several `DrumVoice`s, be saved to JSON, diffed against a
fitted set, or edited by hand between renders.

Nothing here knows about sample rate except where it must (converting a t60 to
a per-sample coefficient), which is why every such method takes `sr` rather
than storing it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from typing import ClassVar
from pathlib import Path

import numpy as np

from ..core.constants import Audio, Decay
from ..core.dsp import FilterDesign


@dataclass
class Mode:
    """One resonant partial of the membrane.

    A drum uses ~25-35 of these. Place close pairs deliberately (e.g. 88.0 and
    92.8 Hz) — the beating between them is a large part of why real drums sound
    alive and synthesized ones sound dead. Cost is one extra resonator.

    There is no phase parameter. Impulse excitation starts every mode in phase,
    which is physically correct and a large part of why a hit sounds like a hit.
    """

    f_static: float  # Hz, frequency at rest (zero tension offset)
    gain: float  # linear, initial amplitude imparted by a unit strike
    t60: float  # s, time to decay -60 dB

    def decay_coef(self, sr: int) -> float:
        """Per-sample amplitude multiplier `r` reaching -60 dB at t60."""
        return Decay.t60_to_coef(self.t60, sr)

    def angular_frequency(self, sr: int) -> float:
        """Normalized radian frequency, 2*pi*f/sr."""
        return 2.0 * np.pi * self.f_static / sr

    def epsilon(self, sr: int, ratio: float = 1.0) -> float:
        """Coupled-form frequency coefficient, eps = 2*sin(pi*f/sr)."""
        return 2.0 * np.sin(np.pi * self.f_static * ratio / sr)

    def validate(self, sr: int = Audio.DEFAULT_SR) -> None:
        """Raise ValueError for anything unplayable at this sample rate."""
        if not np.isfinite(self.f_static) or self.f_static <= 0.0:
            raise ValueError(f"mode f_static must be positive, got {self.f_static}")
        if self.f_static >= 0.5 * sr:
            raise ValueError(
                f"mode at {self.f_static} Hz is at or above Nyquist for sr={sr}"
            )
        if not np.isfinite(self.gain) or self.gain <= 0.0:
            raise ValueError(f"mode gain must be positive, got {self.gain}")
        if not np.isfinite(self.t60) or self.t60 <= 0.0:
            raise ValueError(f"mode t60 must be positive, got {self.t60}")

    def scaled(self, freq_ratio: float = 1.0, gain_ratio: float = 1.0,
               t60_ratio: float = 1.0) -> "Mode":
        """A copy with any of the three numbers multiplied. Tuning, strike
        position and muting are all expressed through this."""
        return Mode(
            f_static=self.f_static * freq_ratio,
            gain=self.gain * gain_ratio,
            t60=self.t60 * t60_ratio,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Mode":
        return cls(
            f_static=float(data["f_static"]),
            gain=float(data["gain"]),
            t60=float(data["t60"]),
        )


@dataclass
class NoiseBand:
    """One band of the contact-noise burst.

    A drum uses 3-4 of these, e.g. (200, 800), (800, 2000), (2000, 6000),
    (6000, 15000). t60 is short — typically 0.005 to 0.15 s — and shorter for
    higher bands.

    No attack. Level starts at `level` and decays immediately. The ~10 ms
    envelope peak heard in a real drum comes from the modal bank summing up,
    not from anything here.
    """

    f_low: float  # Hz
    f_high: float  # Hz
    level: float  # linear, initial amplitude
    t60: float  # s

    def decay_coef(self, sr: int) -> float:
        return Decay.t60_to_coef(self.t60, sr)

    def center_and_q(self) -> tuple[float, float]:
        """Geometric center frequency and Q, for bandpass coefficient design."""
        return FilterDesign.center_and_q(self.f_low, self.f_high)

    def validate(self, sr: int = Audio.DEFAULT_SR) -> None:
        if not (0.0 < self.f_low < self.f_high):
            raise ValueError(
                f"noise band must satisfy 0 < f_low < f_high, got "
                f"{self.f_low}-{self.f_high} Hz"
            )
        if self.f_low >= 0.5 * sr:
            raise ValueError(f"noise band {self.f_low} Hz is above Nyquist for sr={sr}")
        if not np.isfinite(self.level) or self.level < 0.0:
            raise ValueError(f"noise level must be non-negative, got {self.level}")
        if not np.isfinite(self.t60) or self.t60 <= 0.0:
            raise ValueError(f"noise t60 must be positive, got {self.t60}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "NoiseBand":
        return cls(
            f_low=float(data["f_low"]),
            f_high=float(data["f_high"]),
            level=float(data["level"]),
            t60=float(data["t60"]),
        )


@dataclass
class Tension:
    """Global tension nonlinearity — the source of the pitch glide.

    Total modal energy raises effective head tension, which raises every mode
    frequency by the same ratio:

        ratio = 1 + k * smoothed_energy

    This is why a hard hit glides down ~2 semitones as it decays while a soft
    hit barely moves, with no velocity term anywhere in the glide path. Set
    k = 0.0 to disable (linear modal bank).
    """

    k: float = 0.0  # energy -> frequency-ratio scaling
    tau: float = 0.1  # s, smoothing time constant of the energy estimate

    #: Hard ceiling on the ratio. A fit that runs away would otherwise push
    #: modes past Nyquist; clamping fails loudly in the output instead of
    #: producing NaNs deep inside the resonator bank.
    max_ratio: float = 2.0

    #: Follow rising energy instantly and smooth only the fall.
    #:
    #: A symmetric one-pole starting from zero produces a pitch RISE over the
    #: first few tau, because the smoothed energy has to climb to meet the
    #: energy that arrived instantaneously at the strike. The reference hit
    #: glides DOWN monotonically from 10 ms onward, so the symmetric form has
    #: the shape wrong at the start regardless of how k and tau are fitted.
    #:
    #: Physically the head tightens the instant it is displaced; the smoothing
    #: exists to stop the ratio tracking the beat between close mode pairs, not
    #: to model a delay. Setting this False restores the literal symmetric
    #: one-pole, which is worth doing once to hear why it is wrong.
    #:
    #: This is a mode switch, not a fitted number, and is excluded from the
    #: parameter budget.
    instant_attack: bool = True

    @property
    def is_active(self) -> bool:
        """False means the modal bank is linear and can be rendered in one go."""
        return self.k != 0.0

    def smoothing_coef(self, sr: int, period: int = 1) -> float:
        """One-pole coefficient for an update every `period` samples.

        The period is part of the formula, not an afterthought: updating the
        ratio every 64 samples with a per-sample coefficient would stretch the
        glide's settle time by 64x, which is exactly the audible artifact
        `control_period` is supposed to avoid.
        """
        if self.tau <= 0.0:
            return 1.0
        return float(1.0 - np.exp(-period / (self.tau * sr)))

    def ratio_for(self, energy: float) -> float:
        return float(min(1.0 + self.k * energy, self.max_ratio))

    def validate(self, sr: int = Audio.DEFAULT_SR) -> None:
        if not np.isfinite(self.k):
            raise ValueError(f"tension k must be finite, got {self.k}")
        if not np.isfinite(self.tau) or self.tau <= 0.0:
            raise ValueError(f"tension tau must be positive, got {self.tau}")
        if self.max_ratio < 1.0:
            raise ValueError(f"tension max_ratio must be >= 1, got {self.max_ratio}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Tension":
        return cls(
            k=float(data.get("k", 0.0)),
            tau=float(data.get("tau", 0.1)),
            max_ratio=float(data.get("max_ratio", 2.0)),
            instant_attack=bool(data.get("instant_attack", True)),
        )


@dataclass
class DrumParams:
    """Complete static description of one drum. ~109 numbers for a realistic tom.

        30 modes x 3      =  90
         4 noise bands x 4 =  16
        tension            =   2
        output_gain        =   1
                              ---
                              109
    """

    modes: list[Mode] = field(default_factory=list)
    noise: list[NoiseBand] = field(default_factory=list)
    tension: Tension = field(default_factory=Tension)
    output_gain: float = 1.0
    name: str = ""

    FORMAT_VERSION: ClassVar[int] = 1

    # -- validation -----------------------------------------------------------

    def validate(self, sr: int = Audio.DEFAULT_SR) -> None:
        """Validate every mode and band; raise on anything unplayable."""
        if not self.modes:
            raise ValueError(f"drum {self.name!r} has no modes")
        for index, mode in enumerate(self.modes):
            try:
                mode.validate(sr)
            except ValueError as error:
                raise ValueError(f"modes[{index}]: {error}") from error
        for index, band in enumerate(self.noise):
            try:
                band.validate(sr)
            except ValueError as error:
                raise ValueError(f"noise[{index}]: {error}") from error
        self.tension.validate(sr)
        if not np.isfinite(self.output_gain):
            raise ValueError(f"output_gain must be finite, got {self.output_gain}")

    def warnings(self, sr: int = Audio.DEFAULT_SR) -> list[str]:
        """Non-fatal problems worth reading before rendering."""
        issues: list[str] = []
        if len(self.modes) < 10:
            issues.append(
                f"{len(self.modes)} modes: below the 25-35 the architecture assumes; "
                "expect a thin, synthetic tail"
            )
        highest_ratio = self.tension.max_ratio if self.tension.is_active else 1.0
        top = max((m.f_static for m in self.modes), default=0.0) * highest_ratio
        if top >= 0.45 * sr:
            issues.append(
                f"highest mode reaches {top:.0f} Hz under maximum tension, close to "
                f"Nyquist at sr={sr}"
            )
        if self.tension.is_active and self.tension.tau > 0.5:
            issues.append(
                f"tension.tau={self.tension.tau:.2f}s smooths slower than the glide "
                "it is meant to track (~1s total)"
            )
        return issues

    # -- inspection -----------------------------------------------------------

    def sorted_modes(self) -> list[Mode]:
        """Modes ascending by f_static. Useful for inspection and fitting."""
        return sorted(self.modes, key=lambda mode: mode.f_static)

    def mode_frequencies(self) -> np.ndarray:
        return np.array([mode.f_static for mode in self.modes], dtype=np.float64)

    def mode_gains(self) -> np.ndarray:
        return np.array([mode.gain for mode in self.modes], dtype=np.float64)

    def mode_t60s(self) -> np.ndarray:
        return np.array([mode.t60 for mode in self.modes], dtype=np.float64)

    def fundamental(self) -> float:
        """Lowest mode frequency. Not necessarily the loudest."""
        return float(min(mode.f_static for mode in self.modes)) if self.modes else 0.0

    def parameter_count(self) -> int:
        return 3 * len(self.modes) + 4 * len(self.noise) + 2 + 1

    def longest_t60(self) -> float:
        """Slowest decay in the drum. A render shorter than this is truncated."""
        candidates = [m.t60 for m in self.modes] + [b.t60 for b in self.noise]
        return float(max(candidates)) if candidates else 0.0

    def nearest_mode_index(self, freq: float) -> int | None:
        """Index of the mode closest to `freq`. Used by error attribution."""
        if not self.modes:
            return None
        distances = np.abs(self.mode_frequencies() - float(freq))
        return int(np.argmin(distances))

    def summary(self) -> str:
        lines = [
            f"DrumParams {self.name or '<unnamed>'}",
            f"  modes       : {len(self.modes)}  "
            f"({self.fundamental():.1f} Hz .. "
            f"{max(self.mode_frequencies(), default=0.0):.0f} Hz)",
            f"  noise bands : {len(self.noise)}",
            f"  tension     : k={self.tension.k:.6g} tau={self.tension.tau:.3g}s",
            f"  output_gain : {self.output_gain:.4g}",
            f"  parameters  : {self.parameter_count()}",
        ]
        lines.extend(f"  ! {issue}" for issue in self.warnings())
        return "\n".join(lines)

    # -- transforms -----------------------------------------------------------

    def tuned(self, ratio: float) -> "DrumParams":
        """Every mode frequency scaled. Tuning is one multiplier, by design."""
        return replace(self, modes=[m.scaled(freq_ratio=ratio) for m in self.modes],
                       name=f"{self.name}(tuned x{ratio:.4g})")

    def muted(self, t60_ratio: float) -> "DrumParams":
        """Every mode damped. Hand muting is one multiplier on t60."""
        return replace(self, modes=[m.scaled(t60_ratio=t60_ratio) for m in self.modes],
                       name=f"{self.name}(muted x{t60_ratio:.4g})")

    def copy(self) -> "DrumParams":
        return DrumParams(
            modes=[Mode(m.f_static, m.gain, m.t60) for m in self.modes],
            noise=[NoiseBand(b.f_low, b.f_high, b.level, b.t60) for b in self.noise],
            tension=Tension(self.tension.k, self.tension.tau, self.tension.max_ratio),
            output_gain=self.output_gain,
            name=self.name,
        )

    # -- serialization --------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "format_version": self.FORMAT_VERSION,
            "name": self.name,
            "output_gain": self.output_gain,
            "tension": self.tension.to_dict(),
            "modes": [mode.to_dict() for mode in self.modes],
            "noise": [band.to_dict() for band in self.noise],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DrumParams":
        version = int(data.get("format_version", cls.FORMAT_VERSION))
        if version > cls.FORMAT_VERSION:
            raise ValueError(
                f"parameter file format v{version} is newer than this build "
                f"(v{cls.FORMAT_VERSION})"
            )
        return cls(
            modes=[Mode.from_dict(item) for item in data.get("modes", [])],
            noise=[NoiseBand.from_dict(item) for item in data.get("noise", [])],
            tension=Tension.from_dict(data.get("tension", {})),
            output_gain=float(data.get("output_gain", 1.0)),
            name=str(data.get("name", "")),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "DrumParams":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
