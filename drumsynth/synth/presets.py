"""Hand-built drums for sanity-checking the engine before any fitting exists.

Every number here traces back to the reference floor-tom measurements in
docs/ARCHITECTURE.md §2. Nothing was fitted — these are the architecture's own
claims turned into parameters, so that if the engine is wired correctly the
result already sounds like a plausible drum.

    DampingCurve    t60 falls roughly as 1/f, anchored on the measured bands
    ModalLayout     inharmonic mode frequencies from circular-membrane theory
    ExcitationTilt  mode gains rolling off with frequency
    DrumPresets     the three of them assembled into DrumParams
"""

from __future__ import annotations

import numpy as np

from ..core.constants import Audio
from .params import DrumParams, Mode, NoiseBand, Tension


class DampingCurve:
    """Frequency-dependent decay, interpolated in log-log space.

    From the reference hit, fitted T60 per band:

        40-130 Hz    2.33 s
        230-400 Hz   0.75 s
        400-900 Hz   0.55 s
        900-4000 Hz  0.31 s

    A 7.5x spread, and not a single clean power law — the slope between the
    first two anchors is close to 1/f, between the last two much shallower.
    Interpolating log(t60) against log(f) through the measured points keeps the
    real shape instead of forcing an exponent that fits neither end. Most of the
    realism in this synthesizer is in this curve.
    """

    #: (frequency Hz, t60 s). The first anchor is the fundamental itself rather
    #: than its band's center, because that is where the 2.33 s was measured.
    REFERENCE_ANCHORS: tuple[tuple[float, float], ...] = (
        (92.5, 2.33),
        (303.0, 0.75),
        (600.0, 0.55),
        (1897.0, 0.31),
    )

    def __init__(
        self,
        anchors: tuple[tuple[float, float], ...] | None = None,
        f0: float | None = None,
    ) -> None:
        """`f0` rescales the whole curve so a drum tuned elsewhere keeps the
        same damping *shape* relative to its own fundamental."""
        anchors = anchors or DampingCurve.REFERENCE_ANCHORS
        scale = 1.0 if f0 is None else f0 / DampingCurve.REFERENCE_ANCHORS[0][0]
        self.freqs = np.array([f * scale for f, _ in anchors], dtype=np.float64)
        self.t60s = np.array([t for _, t in anchors], dtype=np.float64)
        self._log_f = np.log(self.freqs)
        self._log_t = np.log(self.t60s)

    def __call__(self, freq):
        """t60 at one frequency or an array of them, extrapolating end slopes."""
        log_f = np.log(np.maximum(np.asarray(freq, dtype=np.float64), 1e-6))
        log_t = np.interp(log_f, self._log_f, self._log_t)

        # np.interp clamps outside the anchors; continue the end slopes instead,
        # or a 4 kHz mode inherits the 1897 Hz t60 exactly.
        low_slope = (self._log_t[1] - self._log_t[0]) / (self._log_f[1] - self._log_f[0])
        high_slope = (self._log_t[-1] - self._log_t[-2]) / (self._log_f[-1] - self._log_f[-2])
        below = log_f < self._log_f[0]
        above = log_f > self._log_f[-1]
        log_t = np.where(below, self._log_t[0] + low_slope * (log_f - self._log_f[0]), log_t)
        log_t = np.where(above, self._log_t[-1] + high_slope * (log_f - self._log_f[-1]), log_t)

        out = np.exp(log_t)
        return float(out) if out.ndim == 0 else out

    def scaled(self, factor: float) -> "DampingCurve":
        """The same shape, every t60 multiplied. This is what muting does."""
        anchors = tuple(zip(self.freqs.tolist(), (self.t60s * factor).tolist()))
        return DampingCurve(anchors)


class ModalLayout:
    """Mode frequencies for a circular membrane.

    Ratios are the Bessel zeros j_mn / j_01. Membrane modes are inharmonic —
    the first two partials sit at 1.594 and 2.136, not at 2 and 3 — and
    assuming integer ratios is the single fastest way to make a modal drum
    sound like a synthesizer.

    Real heads deviate from the ideal (the reference measured 1.63 and 2.13
    against theory's 1.594 and 2.136) because of air loading and stiffness, so
    a small per-mode stretch is applied on top.
    """

    #: j_mn / j_01 for a clamped circular membrane, ascending.
    BESSEL_RATIOS: tuple[float, ...] = (
        1.000, 1.594, 2.136, 2.296, 2.653, 2.918, 3.156, 3.501,
        3.598, 3.652, 4.060, 4.154, 4.230, 4.601, 4.643, 4.832,
        4.903, 5.126, 5.412, 5.500, 5.651, 5.906, 6.000, 6.209,
        6.312, 6.529, 6.876, 6.923, 7.243, 7.510,
    )

    #: Multiplicative deviation from ideal, applied to the first few partials.
    #: 1.594 -> 1.63 and 2.136 -> 2.13 reproduces the measured reference.
    HEAD_STRETCH: tuple[float, ...] = (1.0, 1.0226, 0.9972, 1.004, 0.998, 1.003)

    #: Irrational step used to spread the extrapolated ratios deterministically.
    #: Golden-ratio jitter rather than an RNG: presets must be reproducible, and
    #: a perfectly regular sqrt(n) spacing produces an audible metallic comb.
    JITTER_STEP: float = 0.6180339887498949
    JITTER_DEPTH: float = 0.015

    def __init__(
        self,
        f0: float,
        n_modes: int = 30,
        stretch: tuple[float, ...] | None = None,
    ) -> None:
        self.f0 = float(f0)
        self.n_modes = int(n_modes)
        self.stretch = stretch if stretch is not None else ModalLayout.HEAD_STRETCH

    @staticmethod
    def ratios(n_modes: int) -> np.ndarray:
        """`n_modes` ascending ratios, extrapolating past the tabulated zeros.

        Mode density on a membrane grows as f**2, so the n-th ratio grows as
        sqrt(n). Past the table that asymptotic law takes over, with a small
        deterministic jitter — evenly spaced high modes comb-filter audibly.
        """
        table = np.array(ModalLayout.BESSEL_RATIOS, dtype=np.float64)
        if n_modes <= len(table):
            return table[:n_modes]

        last_index = len(table) - 1
        extra = np.arange(len(table), n_modes, dtype=np.float64)
        base = table[last_index] * np.sqrt((extra + 1.0) / (last_index + 1.0))
        jitter = ((extra * ModalLayout.JITTER_STEP) % 1.0) - 0.5
        return np.concatenate([table, base * (1.0 + ModalLayout.JITTER_DEPTH * jitter)])

    def frequencies(self) -> np.ndarray:
        ratios = ModalLayout.ratios(self.n_modes)
        deviations = np.ones_like(ratios)
        n = min(len(self.stretch), len(ratios))
        deviations[:n] = self.stretch[:n]
        return self.f0 * ratios * deviations

    def with_close_pair(
        self, detune_cents: float = -85.0, partner_index: int = 0
    ) -> np.ndarray:
        """Frequencies plus one deliberately detuned twin of `partner_index`.

        The reference showed peaks at 88.0 and 92.8 Hz — a pair 92 cents apart
        beating at ~4.8 Hz. That warble is a large part of why a real drum
        sounds alive, and it costs exactly one extra resonator. It is not a
        `beat_rate` parameter; it emerges.
        """
        freqs = self.frequencies()
        twin = freqs[partner_index] * (2.0 ** (detune_cents / 1200.0))
        return np.sort(np.append(freqs, twin))


class ExcitationTilt:
    """Mode gains as a function of frequency.

    A stick contact of finite width acts as a lowpass on the excitation: short
    contact is bright, long contact is dull. That is `contact_time`, and it is
    deferred until velocity fitting — at a single fixed velocity it is entirely
    absorbed into the fitted gains, which is what this class stands in for.

    The reference hit put 12.5% of its energy in the first 30 ms with a 19.7 dB
    crest factor, which needs real high-frequency content; a curve that rolls
    off much faster than this produces a dull thud with no stick in it.
    """

    def __init__(self, f_cut: float = 420.0, order: float = 1.0, floor: float = 0.004) -> None:
        self.f_cut = float(f_cut)
        self.order = float(order)
        self.floor = float(floor)

    def __call__(self, freqs):
        f = np.asarray(freqs, dtype=np.float64)
        tilt = 1.0 / (1.0 + (f / self.f_cut) ** (2.0 * self.order))
        return np.maximum(tilt, self.floor)

    @staticmethod
    def normalized(gains: np.ndarray) -> np.ndarray:
        """Scale so sum(gain**2) == 1 for a unit strike.

        This makes the bank's energy at t=0 exactly 1.0, which in turn makes
        `Tension.k` read directly as the peak frequency ratio minus one. Without
        it, k silently changes meaning every time a mode is added.
        """
        gains = np.asarray(gains, dtype=np.float64)
        total = float(np.sqrt(np.sum(gains**2)))
        return gains / total if total > 0 else gains


class DrumPresets:
    """Hand-built drums. Starting points and engine checks, not fitted models."""

    #: Reference floor tom, from docs/ARCHITECTURE.md §2.
    REFERENCE_F0: float = 92.5
    REFERENCE_GLIDE_SEMITONES: float = 2.07
    REFERENCE_GLIDE_TAU: float = 0.12

    DEFAULT_NOISE_BANDS: tuple[tuple[float, float, float, float], ...] = (
        # f_low, f_high, level, t60
        (200.0, 800.0, 0.055, 0.055),
        (800.0, 2000.0, 0.042, 0.028),
        (2000.0, 6000.0, 0.030, 0.014),
        (6000.0, 15000.0, 0.016, 0.007),
    )

    #: Peak amplitude a unit strike should produce. Every mode starts in phase,
    #: so the first sample is exactly output_gain * sum(gain) and the peak is
    #: known without rendering anything.
    TARGET_PEAK: float = 0.9

    @staticmethod
    def fit_output_gain(params: DrumParams, target_peak: float = TARGET_PEAK) -> float:
        """output_gain that puts a unit strike at `target_peak`."""
        total = float(np.sum(params.mode_gains()))
        return target_peak / total if total > 0 else 1.0

    @staticmethod
    def tom(
        f0: float = REFERENCE_F0,
        sr: int = Audio.DEFAULT_SR,
        n_modes: int = 30,
        with_noise: bool = True,
        with_tension: bool = True,
        name: str = "test_tom",
    ) -> DrumParams:
        """Low tom matching the analysed reference.

        f0 ~ 92.5 Hz, t60 ~ 2.3 s at the fundamental falling to ~0.3 s by
        2 kHz, a close pair near the fundamental for warble, and ~2 semitones
        of energy-driven glide.

        Build order note: start with `with_noise=False, with_tension=False` and
        listen. A pure linear modal bank should already sound like a plausible
        drum; if it does not, nothing added afterwards will fix it.
        """
        layout = ModalLayout(f0, n_modes)
        freqs = layout.with_close_pair()
        freqs = freqs[freqs < 0.45 * sr]

        damping = DampingCurve(f0=f0)
        t60s = damping(freqs)
        gains = ExcitationTilt.normalized(ExcitationTilt()(freqs))

        modes = [
            Mode(f_static=float(f), gain=float(g), t60=float(t))
            for f, g, t in zip(freqs, gains, t60s)
        ]

        noise = (
            [NoiseBand(*band) for band in DrumPresets.DEFAULT_NOISE_BANDS]
            if with_noise
            else []
        )

        # Gains are normalized to unit energy at a unit strike, so k is exactly
        # the peak frequency ratio minus one.
        tension = (
            Tension(
                k=2.0 ** (DrumPresets.REFERENCE_GLIDE_SEMITONES / 12.0) - 1.0,
                tau=DrumPresets.REFERENCE_GLIDE_TAU,
            )
            if with_tension
            else Tension(k=0.0)
        )

        params = DrumParams(
            modes=modes, noise=noise, tension=tension, output_gain=1.0, name=name
        )
        params.output_gain = DrumPresets.fit_output_gain(params)
        params.validate(sr)
        return params

    @staticmethod
    def kick(sr: int = Audio.DEFAULT_SR, name: str = "test_kick") -> DrumParams:
        """Lower, shorter, duller than the tom, with a heavier beater thump.

        Not a fitted kick — the same curves at a lower f0 with the damping
        pulled in and the excitation tilt darkened.
        """
        params = DrumPresets.tom(f0=55.0, sr=sr, n_modes=24, name=name)
        damping = DampingCurve(f0=55.0).scaled(0.42)
        tilt = ExcitationTilt(f_cut=230.0)
        freqs = np.array([m.f_static for m in params.modes])
        gains = ExcitationTilt.normalized(tilt(freqs))
        params.modes = [
            Mode(float(f), float(g), float(t))
            for f, g, t in zip(freqs, gains, damping(freqs))
        ]
        params.noise = [
            NoiseBand(80.0, 400.0, 0.09, 0.020),
            NoiseBand(400.0, 1500.0, 0.055, 0.012),
            NoiseBand(1500.0, 5000.0, 0.022, 0.006),
        ]
        params.tension = Tension(k=2.0 ** (1.4 / 12.0) - 1.0, tau=0.09)
        params.output_gain = DrumPresets.fit_output_gain(params)
        params.output_gain = DrumPresets.fit_output_gain(params)
        params.name = name
        params.validate(sr)
        return params

    @staticmethod
    def rack_tom(sr: int = Audio.DEFAULT_SR, name: str = "test_rack_tom") -> DrumParams:
        """Higher tom. Same construction, f0 at 165 Hz."""
        return DrumPresets.tom(f0=165.0, sr=sr, n_modes=28, name=name)

    @staticmethod
    def snare_shell(sr: int = Audio.DEFAULT_SR, name: str = "test_snare_shell") -> DrumParams:
        """Snare *shell* only — no wires.

        The wire subsystem is a threshold nonlinearity reading bottom-head
        displacement, and it is not built yet. This is the ~90% of a snare that
        the membrane engine already covers.
        """
        params = DrumPresets.tom(f0=185.0, sr=sr, n_modes=28, name=name)
        damping = DampingCurve(f0=185.0).scaled(0.30)
        freqs = np.array([m.f_static for m in params.modes])
        tilt = ExcitationTilt(f_cut=900.0)
        gains = ExcitationTilt.normalized(tilt(freqs))
        params.modes = [
            Mode(float(f), float(g), float(t))
            for f, g, t in zip(freqs, gains, damping(freqs))
        ]
        params.noise = [
            NoiseBand(300.0, 1200.0, 0.10, 0.035),
            NoiseBand(1200.0, 4000.0, 0.11, 0.045),
            NoiseBand(4000.0, 10000.0, 0.075, 0.030),
            NoiseBand(10000.0, 18000.0, 0.035, 0.015),
        ]
        params.tension = Tension(k=2.0 ** (0.8 / 12.0) - 1.0, tau=0.06)
        params.output_gain = DrumPresets.fit_output_gain(params)
        params.output_gain = DrumPresets.fit_output_gain(params)
        params.name = name
        params.validate(sr)
        return params

    @staticmethod
    def all(sr: int = Audio.DEFAULT_SR) -> dict[str, DrumParams]:
        return {
            "kick": DrumPresets.kick(sr),
            "floor_tom": DrumPresets.tom(sr=sr, name="test_floor_tom"),
            "rack_tom": DrumPresets.rack_tom(sr),
            "snare_shell": DrumPresets.snare_shell(sr),
        }
