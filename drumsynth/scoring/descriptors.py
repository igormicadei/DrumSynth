"""What gets extracted from a signal.

These are the descriptors, and the descriptors *are* the parameters. The
reference is a WAV with nothing attached to it, so the only way to diff
parameter against parameter is to re-derive a parameter set from the reference
audio and compare it to one re-derived, identically, from the generated audio.

Any descriptor that cannot be extracted from raw audio cannot be compared, and
so does not belong here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.constants import Audio, Cents, Decay, Decibels


@dataclass
class ModeEstimate:
    """One partial recovered from audio. Mirrors `synth.Mode`, plus confidence."""

    freq: float  # Hz, at the measurement window's center time
    amplitude: float  # linear, extrapolated back to t=0
    t60: float  # s
    confidence: float = 1.0  # 0-1; low when SNR is poor or the residual is large
    t_window: tuple[float, float] = (0.0, 0.0)  # window the estimate came from

    #: Measured, not modelled. The synthesizer has no phase parameter — impulse
    #: excitation starts every mode in phase — but subtracting estimated modes
    #: from the signal they came out of needs it, so it is carried here.
    phase: float = 0.0

    def amplitude_db(self) -> float:
        return Decibels.from_amplitude(self.amplitude)

    def damping(self) -> float:
        """Continuous-time damping constant, exp(-alpha * t)."""
        return Decay.t60_to_damping(self.t60)

    def cents_from(self, other: "ModeEstimate | float") -> float:
        reference = other.freq if isinstance(other, ModeEstimate) else float(other)
        return Cents.between(self.freq, reference)

    def render(self, n: int, sr: int) -> np.ndarray:
        """This mode alone, as a decaying sinusoid. Used to subtract the modal
        part of a signal before measuring its noise statistics."""
        t = np.arange(int(n), dtype=np.float64) / sr
        envelope = np.exp(-self.damping() * t)
        return self.amplitude * envelope * np.cos(2.0 * np.pi * self.freq * t + self.phase)

    def to_dict(self) -> dict:
        return {
            "freq": self.freq,
            "amplitude": self.amplitude,
            "t60": self.t60,
            "confidence": self.confidence,
            "phase": self.phase,
            "t_window": list(self.t_window),
        }

    def __str__(self) -> str:
        return (
            f"{self.freq:8.2f} Hz  {self.amplitude_db():7.1f} dB  "
            f"t60 {self.t60:6.3f} s  conf {self.confidence:.2f}"
        )


@dataclass
class BandDecay:
    """Aggregate decay of one frequency band. Robust where modal extraction fails."""

    f_low: float
    f_high: float
    level_db: float  # dB at t=0, from the fit's intercept
    slope_db_s: float  # dB/s, fitted over the valid range
    t60: float  # -60 / slope
    valid_range: tuple[float, float]  # s, region above the noise floor
    r_squared: float = 0.0  # straightness of the fit; low => multiple modes

    @property
    def band(self) -> tuple[float, float]:
        return (self.f_low, self.f_high)

    @property
    def is_valid(self) -> bool:
        """False when the band never rose above the noise floor."""
        return np.isfinite(self.slope_db_s) and self.slope_db_s < 0.0

    @property
    def usable_seconds(self) -> float:
        return max(0.0, self.valid_range[1] - self.valid_range[0])

    def __str__(self) -> str:
        return (
            f"{self.f_low:6.0f}-{self.f_high:<6.0f} Hz  {self.level_db:7.1f} dB  "
            f"{self.slope_db_s:8.1f} dB/s  t60 {self.t60:6.3f} s  r2 {self.r_squared:.3f}"
        )


@dataclass
class GlideTrack:
    """Instantaneous frequency of the fundamental over time — the tension signature."""

    times: np.ndarray  # s
    freqs: np.ndarray  # Hz
    f_initial: float  # Hz, earliest reliable estimate
    f_asymptote: float  # Hz, settled value
    depth_cents: float  # 1200 * log2(f_initial / f_asymptote)
    settle_time: float  # s to reach within 2% of asymptote

    #: Frames whose level was too low to trust. Scoring must ignore these, or a
    #: reference that has decayed into its codec floor contributes noise.
    reliable: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

    def as_cents(self) -> np.ndarray:
        """Track expressed in cents relative to f_asymptote."""
        return Cents.between(self.freqs, self.f_asymptote)

    def depth_semitones(self) -> float:
        return self.depth_cents / 100.0

    def reliable_track(self) -> tuple[np.ndarray, np.ndarray]:
        """(times, freqs) with the untrustworthy frames removed."""
        if self.reliable.size != self.times.size:
            return self.times, self.freqs
        return self.times[self.reliable], self.freqs[self.reliable]

    def at(self, seconds) -> np.ndarray:
        """Frequency interpolated onto arbitrary times."""
        times, freqs = self.reliable_track()
        if times.size == 0:
            return np.full(np.shape(seconds), np.nan)
        return np.interp(seconds, times, freqs)

    def __str__(self) -> str:
        return (
            f"glide {self.f_initial:.2f} -> {self.f_asymptote:.2f} Hz  "
            f"({self.depth_semitones():.2f} semitones, settles {self.settle_time:.2f} s)"
        )


@dataclass
class EnvelopeCurve:
    """Broadband RMS envelope in dB."""

    times: np.ndarray
    db: np.ndarray
    peak_time: float  # s, where the envelope maximum sits
    rise_10_90: float  # s
    crest_factor_db: float

    def peak_db(self) -> float:
        return float(np.max(self.db)) if self.db.size else Audio.SILENCE_DB

    def above(self, floor_db: float) -> np.ndarray:
        """Boolean mask of frames above an absolute dB floor."""
        return self.db > floor_db

    def at(self, seconds) -> np.ndarray:
        if self.times.size == 0:
            return np.full(np.shape(seconds), np.nan)
        return np.interp(seconds, self.times, self.db)

    def __str__(self) -> str:
        return (
            f"envelope peak at {self.peak_time * 1000:.1f} ms, "
            f"rise {self.rise_10_90 * 1000:.2f} ms, crest {self.crest_factor_db:.1f} dB"
        )


@dataclass
class NoiseStats:
    """Statistics of the stochastic component. Never compared sample-wise.

    Two realizations of the same noise process have roughly zero correlation, so
    the only matchable quantities are aggregates: how much energy sits in each
    band and how fast each band falls.
    """

    band_energy_db: dict[tuple[float, float], float] = field(default_factory=dict)
    band_slope_db_s: dict[tuple[float, float], float] = field(default_factory=dict)
    energy_fraction_30ms: float = 0.0  # fraction of total energy in the first 30 ms
    energy_fraction_150ms: float = 0.0
    spectral_flux_attack: float = 0.0
    modal_residual_db: float = 0.0  # level of the residual after modes were removed

    def bands(self) -> list[tuple[float, float]]:
        return sorted(self.band_energy_db)

    def __str__(self) -> str:
        parts = [
            f"E(30ms) {self.energy_fraction_30ms * 100:.1f}%",
            f"E(150ms) {self.energy_fraction_150ms * 100:.1f}%",
            f"flux {self.spectral_flux_attack:.3f}",
        ]
        return "noise: " + "  ".join(parts)


@dataclass
class SoundDescriptors:
    """Everything extracted from one signal. Produced identically for both sides."""

    sr: int
    duration: float
    modes: list[ModeEstimate] = field(default_factory=list)
    bands: list[BandDecay] = field(default_factory=list)
    glide: GlideTrack | None = None
    envelope: EnvelopeCurve | None = None
    noise: NoiseStats | None = None
    noise_floor_db: float = -np.inf  # measurement floor; scoring must stop above this

    #: Whatever the analysis chain noticed but could not act on.
    warnings: list[str] = field(default_factory=list)

    def mode_freqs(self) -> np.ndarray:
        return np.array([mode.freq for mode in self.modes], dtype=np.float64)

    def mode_amplitudes(self) -> np.ndarray:
        return np.array([mode.amplitude for mode in self.modes], dtype=np.float64)

    def mode_t60s(self) -> np.ndarray:
        return np.array([mode.t60 for mode in self.modes], dtype=np.float64)

    def sorted_modes(self) -> list[ModeEstimate]:
        return sorted(self.modes, key=lambda mode: mode.freq)

    def loudest_modes(self, n: int = 10) -> list[ModeEstimate]:
        return sorted(self.modes, key=lambda mode: -mode.amplitude)[:n]

    def band(self, f_low: float, f_high: float) -> BandDecay | None:
        for decay in self.bands:
            if decay.f_low == f_low and decay.f_high == f_high:
                return decay
        return None

    def summary(self) -> str:
        lines = [
            f"SoundDescriptors  {self.duration:.2f} s @ {self.sr} Hz  "
            f"(noise floor {self.noise_floor_db:.1f} dB)",
        ]
        if self.envelope is not None:
            lines.append(f"  {self.envelope}")
        if self.glide is not None:
            lines.append(f"  {self.glide}")
        if self.noise is not None:
            lines.append(f"  {self.noise}")
        if self.bands:
            lines.append(f"  bands ({len(self.bands)}):")
            lines.extend(f"    {band}" for band in self.bands)
        if self.modes:
            lines.append(f"  modes ({len(self.modes)}), loudest first:")
            lines.extend(f"    {mode}" for mode in self.loudest_modes(12))
        lines.extend(f"  ! {warning}" for warning in self.warnings)
        return "\n".join(lines)
