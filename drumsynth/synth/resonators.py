"""Modal resonators: one reference implementation and one fast bank.

`ModeResonator` is the architecture written out literally — the coupled
("magic circle") form, one sample at a time, four multiplies and two adds. It
is the definition of what this synthesizer does.

`ModalBank` is what `DrumVoice` actually runs. It produces bit-comparable
output by exploiting a fact about the coupled form: over any span where the
frequency is held constant the mode is linear and time-invariant, its poles sit
at r*exp(+/-j*w), and the whole span therefore has a closed form

    x[k] = r**k * (A*cos(w*k) + B*sin(w*k))

with A and B fixed by the first two steps. That turns a per-sample Python loop
into two array expressions per control block, which is the difference between
a 4-second render taking seconds and taking milliseconds. The equivalence is
covered by a test — if `ModeResonator` ever changes, that test is what catches
`ModalBank` drifting away from it.

Why the coupled form and not a direct-form biquad: frequency is modulated every
control period by the tension feedback, and a direct-form 2-pole jumps in
amplitude when its coefficients change mid-ring. The coupled form holds
amplitude across frequency changes and exposes frequency as a single number.
"""

from __future__ import annotations

import numpy as np

from ..core.constants import Audio, Decay
from .params import Mode


class ModeResonator:
    """Single mode, coupled-form oscillator. The reference implementation.

        eps = 2 * sin(pi * f / sr)
        x  += eps * y
        y  -= eps * x
        x  *= r
        y  *= r

    Output is `x`. Energy is x**2 + y**2.
    """

    __slots__ = ("sr", "f_static", "gain", "t60", "eps", "r", "x", "y")

    #: Frequencies are clamped into (0, MAX_FREQ_FRACTION * sr) before use, so
    #: a runaway tension ratio degrades audibly instead of producing NaNs.
    MAX_FREQ_FRACTION: float = 0.49

    def __init__(self, mode: Mode, sr: int = Audio.DEFAULT_SR) -> None:
        mode.validate(sr)
        self.sr = int(sr)
        self.f_static = float(mode.f_static)
        self.gain = float(mode.gain)
        self.t60 = float(mode.t60)
        self.r = float(Decay.t60_to_coef(mode.t60, sr))
        self.eps = 0.0
        self.x = 0.0
        self.y = 0.0
        self.set_frequency(self.f_static)

    # -- frequency ------------------------------------------------------------

    def set_frequency(self, f: float) -> None:
        """Update eps from an absolute frequency in Hz. Safe to call per sample."""
        limit = ModeResonator.MAX_FREQ_FRACTION * self.sr
        clamped = min(max(float(f), 0.0), limit)
        self.eps = 2.0 * np.sin(np.pi * clamped / self.sr)

    def set_frequency_ratio(self, ratio: float) -> None:
        """set_frequency(f_static * ratio). This is the tension path."""
        self.set_frequency(self.f_static * ratio)

    @property
    def frequency(self) -> float:
        """Current frequency in Hz, recovered from eps."""
        return float(np.arcsin(np.clip(self.eps / 2.0, -1.0, 1.0)) * self.sr / np.pi)

    # -- playing --------------------------------------------------------------

    def excite(self, amplitude: float) -> None:
        """Impulse excitation.

        Adds to state rather than replacing it, so a strike landing on a still-
        ringing mode superposes correctly. This `+=` is what makes flams, ghost
        notes and rolls work with no special-case logic, and it is easy to
        "simplify" into `=` later, which is why it has its own test.
        """
        self.x += amplitude * self.gain

    def step(self) -> float:
        """Advance one sample, return output."""
        x = self.x + self.eps * self.y
        y = self.y - self.eps * x
        self.x = x = x * self.r
        self.y = y * self.r
        return x

    def process(self, n: int, freq_ratio: np.ndarray | float = 1.0) -> np.ndarray:
        """Advance n samples. `freq_ratio` may be scalar or length-n.

        Sample-at-a-time by design — this class is the reference, not the fast
        path. Use `ModalBank` for anything long.
        """
        if n <= 0:
            return np.zeros(0, dtype=np.float64)
        out = np.empty(int(n), dtype=np.float64)
        ratios = np.asarray(freq_ratio, dtype=np.float64)

        if ratios.ndim == 0:
            self.set_frequency_ratio(float(ratios))
            for index in range(int(n)):
                out[index] = self.step()
        else:
            if len(ratios) != n:
                raise ValueError(f"freq_ratio has {len(ratios)} entries, expected {n}")
            for index in range(int(n)):
                self.set_frequency_ratio(float(ratios[index]))
                out[index] = self.step()
        return out

    # -- inspection -----------------------------------------------------------

    @property
    def energy(self) -> float:
        """x**2 + y**2 — proportional to stored energy. Feeds TensionTracker."""
        return self.x * self.x + self.y * self.y

    @property
    def amplitude(self) -> float:
        """sqrt(x**2 + y**2) — current envelope value."""
        return float(np.hypot(self.x, self.y))

    def reset(self) -> None:
        self.x = 0.0
        self.y = 0.0
        self.set_frequency(self.f_static)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"ModeResonator(f={self.f_static:.2f}Hz, gain={self.gain:.4g}, "
            f"t60={self.t60:.3g}s, amp={self.amplitude:.3e})"
        )


class ModalBank:
    """Every mode of one drum, advanced together.

    State is held as arrays rather than as a list of `ModeResonator` objects.
    The tension feedback needs the summed energy of the whole bank before any
    mode advances, which is a bank-level operation, and the closed-form block
    render is only worth writing once.
    """

    #: Blocks shorter than this are stepped directly; setting up the closed
    #: form costs more than a handful of iterations saves.
    DIRECT_LIMIT: int = 3

    #: Decay coefficients are floored here so the closed form never divides by
    #: zero for an absurdly short t60.
    MIN_DECAY_COEF: float = 1e-30

    def __init__(self, modes: list[Mode], sr: int = Audio.DEFAULT_SR) -> None:
        if not modes:
            raise ValueError("a modal bank needs at least one mode")
        for mode in modes:
            mode.validate(sr)

        self.sr = int(sr)
        self.f_static = np.array([m.f_static for m in modes], dtype=np.float64)
        self.gain = np.array([m.gain for m in modes], dtype=np.float64)
        self.t60 = np.array([m.t60 for m in modes], dtype=np.float64)
        self.r = np.maximum(
            np.asarray(Decay.t60_to_coef(self.t60, sr), dtype=np.float64),
            ModalBank.MIN_DECAY_COEF,
        )

        self.x = np.zeros(len(modes), dtype=np.float64)
        self.y = np.zeros(len(modes), dtype=np.float64)

        #: Per-mode output scaling, applied at the mix and nowhere else.
        #:
        #: A LISTENING AID, not a drum parameter. It is deliberately not part of
        #: DrumParams, is never serialized, and never reaches a fit — its only
        #: job is to let you solo one partial while the drum is ringing to hear
        #: which mode a ring belongs to. Muting through `gain` instead would
        #: only take effect on the next strike, because gain is excitation.
        self.monitor_mask = np.ones(len(modes), dtype=np.float64)

        self._ratio = 0.0  # forces the first set_ratio to recompute
        self._w = np.zeros(len(modes), dtype=np.float64)
        self._eps = np.zeros(len(modes), dtype=np.float64)
        self._cos_w = np.ones(len(modes), dtype=np.float64)
        self._sin_w = np.zeros(len(modes), dtype=np.float64)
        self.set_ratio(1.0)

    def __len__(self) -> int:
        return len(self.f_static)

    # -- frequency ------------------------------------------------------------

    def set_ratio(self, ratio: float) -> None:
        """Apply a global frequency ratio. Recomputes trig only when it moves."""
        ratio = float(ratio)
        if ratio == self._ratio:
            return
        self._ratio = ratio
        limit = ModeResonator.MAX_FREQ_FRACTION * self.sr
        freqs = np.clip(self.f_static * ratio, 0.0, limit)
        self._w = 2.0 * np.pi * freqs / self.sr
        self._eps = 2.0 * np.sin(0.5 * self._w)
        self._cos_w = np.cos(self._w)
        self._sin_w = np.sin(self._w)

    def frequencies(self) -> np.ndarray:
        """Current (tension-shifted) frequency of every mode, in Hz."""
        return self._w * self.sr / (2.0 * np.pi)

    def set_params(self, modes: list[Mode]) -> None:
        """Replace the bank's parameters WITHOUT clearing its state.

        This is the live-editing path: turning a mode's t60 knob while the drum
        is ringing has to change how the rest of that ring decays, not restart
        it. Because the coupled form keeps amplitude in (x, y) and frequency in
        a separate coefficient, both can be retuned mid-ring — which is the
        same property that lets the tension feedback modulate frequency every
        control period.

        Mode i keeps mode i's state. Added modes start silent; removed modes
        take their state with them. Reordering the list therefore reassigns
        state, so keep the order stable while playing.
        """
        if not modes:
            raise ValueError("a modal bank needs at least one mode")
        for mode in modes:
            mode.validate(self.sr)

        count = len(modes)
        previous = len(self.f_static)
        self.f_static = np.array([m.f_static for m in modes], dtype=np.float64)
        self.gain = np.array([m.gain for m in modes], dtype=np.float64)
        self.t60 = np.array([m.t60 for m in modes], dtype=np.float64)
        self.r = np.maximum(
            np.asarray(Decay.t60_to_coef(self.t60, self.sr), dtype=np.float64),
            ModalBank.MIN_DECAY_COEF,
        )

        if count != previous:
            keep = min(count, previous)
            x, y = np.zeros(count), np.zeros(count)
            x[:keep], y[:keep] = self.x[:keep], self.y[:keep]
            self.x, self.y = x, y
            mask = np.ones(count)
            mask[:keep] = self.monitor_mask[:keep]
            self.monitor_mask = mask

        ratio, self._ratio = self._ratio, 0.0  # force the trig to recompute
        self.set_ratio(ratio)

    # -- playing --------------------------------------------------------------

    def set_monitor_mask(self, mask: np.ndarray | None) -> None:
        """Solo/mute individual partials without touching the parameters."""
        if mask is None:
            self.monitor_mask = np.ones(len(self), dtype=np.float64)
            return
        values = np.asarray(mask, dtype=np.float64)
        if len(values) != len(self):
            raise ValueError(
                f"monitor mask has {len(values)} entries, expected {len(self)}"
            )
        self.monitor_mask = values

    @property
    def is_monitoring_all(self) -> bool:
        return bool(np.all(self.monitor_mask == 1.0))

    def excite(self, amplitude: float, gain_scale: np.ndarray | None = None) -> None:
        """x += amplitude * gain, for every mode at once.

        `gain_scale` is the seam for strike position: a per-mode multiplier that
        zeroes modes with a node at the strike point. Nothing supplies it yet.
        """
        delta = float(amplitude) * self.gain
        if gain_scale is not None:
            delta = delta * np.asarray(gain_scale, dtype=np.float64)
        self.x += delta

    def step(self) -> float:
        """Advance one sample, return the summed output."""
        x = self.x + self._eps * self.y
        y = self.y - self._eps * x
        self.x = x = x * self.r
        self.y = y * self.r
        return float(np.dot(x, self.monitor_mask))

    def process(self, n: int, ratio: float | None = None) -> np.ndarray:
        """Advance `n` samples at a constant frequency ratio, return the mix.

        The ratio is constant across the whole block — that is what makes the
        closed form valid, and it is why the caller loops over control blocks
        rather than passing a per-sample ratio array.
        """
        if n <= 0:
            return np.zeros(0, dtype=np.float64)
        if ratio is not None:
            self.set_ratio(ratio)

        n = int(n)
        if n <= ModalBank.DIRECT_LIMIT:
            return np.array([self.step() for _ in range(n)], dtype=np.float64)


        r, eps = self.r, self._eps
        cos_w, sin_w = self._cos_w, self._sin_w

        # Two explicit steps pin down the closed form's coefficients.
        one_minus_eps2 = 1.0 - eps * eps
        x1 = r * (self.x + eps * self.y)
        y1 = r * (-eps * self.x + one_minus_eps2 * self.y)
        x2 = r * (x1 + eps * y1)
        y2 = r * (-eps * x1 + one_minus_eps2 * y1)

        safe_sin = np.where(np.abs(sin_w) < 1e-12, 1.0, sin_w)
        a_x, a_y = x1, y1
        b_x = (x2 / r - a_x * cos_w) / safe_sin
        b_y = (y2 / r - a_y * cos_w) / safe_sin

        k = np.arange(n, dtype=np.float64)
        phase = self._w[:, None] * k[None, :]
        decay = np.exp(np.log(r)[:, None] * k[None, :])
        out = decay * (a_x[:, None] * np.cos(phase) + b_x[:, None] * np.sin(phase))

        # Final state is the closed form evaluated at the last sample.
        last = float(n - 1)
        decay_last = np.exp(np.log(r) * last)
        cos_last, sin_last = np.cos(self._w * last), np.sin(self._w * last)
        self.x = decay_last * (a_x * cos_last + b_x * sin_last)
        self.y = decay_last * (a_y * cos_last + b_y * sin_last)

        return (
            out.sum(axis=0)
            if self.is_monitoring_all
            else self.monitor_mask @ out
        )

    def process_modes(self, n: int, ratio: float | None = None) -> np.ndarray:
        """Same as `process` but returns one row per mode, without mixing.

        Diagnostic only: this is how you listen to a single partial to decide
        whether a band's curvature is one mode or three.
        """
        if n <= 0:
            return np.zeros((len(self), 0), dtype=np.float64)
        rows = np.empty((len(self), int(n)), dtype=np.float64)
        saved_x, saved_y = self.x.copy(), self.y.copy()
        for index in range(len(self)):
            self.x = np.zeros_like(saved_x)
            self.y = np.zeros_like(saved_y)
            self.x[index], self.y[index] = saved_x[index], saved_y[index]
            rows[index] = self.process(n, ratio)
        self.x, self.y = saved_x, saved_y
        self.process(n, ratio)
        return rows

    # -- inspection -----------------------------------------------------------

    @property
    def energy(self) -> float:
        """Summed x**2 + y**2 across the bank. Read this BEFORE advancing."""
        return float(np.dot(self.x, self.x) + np.dot(self.y, self.y))

    @property
    def amplitudes(self) -> np.ndarray:
        return np.hypot(self.x, self.y)

    @property
    def peak_amplitude(self) -> float:
        return float(np.max(self.amplitudes)) if len(self) else 0.0

    def reset(self) -> None:
        """Clear the ring. Leaves the monitor mask alone — it is a view setting,
        not state, and losing it on every reset would be maddening."""
        self.x[:] = 0.0
        self.y[:] = 0.0
        self.set_ratio(1.0)


class TensionTracker:
    """One-pole energy follower producing the global frequency ratio.

        ratio = 1 + k * smoothed_energy

    The smoothing coefficient depends on how often `update` is called, so a
    voice running at control_period=64 glides on the same timescale as one
    running at 1.

    Rising energy is followed instantly by default; see `Tension.instant_attack`
    for why a symmetric one-pole gets the start of the glide backwards.
    """

    __slots__ = ("k", "tau", "max_ratio", "instant_attack", "sr", "period",
                 "coef", "smoothed", "ratio")

    def __init__(self, tension, sr: int = Audio.DEFAULT_SR, period: int = 1) -> None:
        tension.validate(sr)
        self.k = float(tension.k)
        self.tau = float(tension.tau)
        self.max_ratio = float(tension.max_ratio)
        self.instant_attack = bool(tension.instant_attack)
        self.sr = int(sr)
        self.period = max(1, int(period))
        self.coef = tension.smoothing_coef(sr, self.period)
        self.smoothed = 0.0
        self.ratio = 1.0

    @property
    def is_active(self) -> bool:
        return self.k != 0.0

    def update(self, energy: float) -> float:
        """Feed raw bank energy, return the frequency ratio to apply."""
        energy = float(energy)
        if self.instant_attack and energy > self.smoothed:
            self.smoothed = energy
        else:
            self.smoothed += (energy - self.smoothed) * self.coef
        self.ratio = min(1.0 + self.k * self.smoothed, self.max_ratio)
        return self.ratio

    def reset(self) -> None:
        self.smoothed = 0.0
        self.ratio = 1.0
