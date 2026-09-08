"""The playable instrument: one drum, and a kit of them.

Per-sample update order is the load-bearing detail:

    1. energy = sum over modes of (x**2 + y**2)     <- BEFORE anything advances
    2. ratio  = tension.update(energy)
    3. modes advance at f_static * ratio
    4. noise bands advance
    5. out = output_gain * (modes + noise)

Reading the energy before the modes advance is what keeps the feedback loop
delayed by one control block. A delayless loop — advancing the modes and then
using their new energy to set the frequency they just ran at — can go unstable
at high `k`, and the failure looks like a tuning problem rather than a
structural one.

`control_period` is how many samples pass between tension updates. The glide
moves on a ~1 s timescale, so 32-64 is inaudible and keeps the frequency
recompute out of the inner loop. It also sets the render block size, so a
larger period is strictly faster.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from ..core.audio_io import AudioIO
from ..core.constants import Audio, Decibels
from .noise import NoiseBank
from .params import DrumParams
from .resonators import ModalBank, TensionTracker


class DrumVoice:
    """One playable drum. Holds all runtime state; `params` stays untouched."""

    #: Block size used when the tension feedback is off and the bank is linear.
    #: Bounded only so a long render does not allocate one giant intermediate.
    LINEAR_CHUNK: int = 8192

    #: Default silence threshold, relative to a unit strike.
    SILENCE_DB: float = -120.0

    def __init__(
        self,
        params: DrumParams,
        sr: int = Audio.DEFAULT_SR,
        control_period: int = 1,
        seed: int | None = None,
    ) -> None:
        """
        control_period: update the tension ratio every N samples instead of
            every sample. Leave at 1 while validating, raise it for real-time.
        seed: fixes the noise bank's random stream. Set it whenever a render is
            going to be scored, so a score difference means a parameter changed
            and not that the noise came out differently.
        """
        params.validate(sr)
        if control_period < 1:
            raise ValueError(f"control_period must be >= 1, got {control_period}")

        self.params = params
        self.sr = int(sr)
        self.control_period = int(control_period)
        self.seed = seed

        self._rng = np.random.default_rng(seed)
        self.modes = ModalBank(params.modes, sr)
        self.noise = NoiseBank(params.noise, sr, self._rng)
        self.tension = TensionTracker(params.tension, sr, self.control_period)
        self.output_gain = float(params.output_gain)

    # -- playing --------------------------------------------------------------

    def strike(
        self, amplitude: float = 1.0, gain_scale: np.ndarray | None = None
    ) -> None:
        """Excite every mode and noise band simultaneously.

        Impulse excitation: `x += amplitude * gain` for every mode, `env +=
        amplitude * level` for every noise band. Additive, not assignment — a
        strike landing on a still-ringing bank superposes, which is the whole
        reason flams, ghost notes and rolls need no special-case code.

        `gain_scale` is the strike-position seam: a per-mode multiplier, one
        entry per mode, where a mode with a node at the strike point receives
        nothing. Nothing computes it yet.

        Velocity mapping (contact_time as a spectral tilt across mode gains,
        disproportionate high-band noise) lands here later without changing
        this signature's meaning.

        Note on the attack: because every mode starts in phase, the summed
        output is at its maximum on the first sample and can only fall. The
        ~10 ms envelope peak measured on the reference does NOT emerge from
        this — see docs/ARCHITECTURE.md §2.2 and the note in docs/FINDINGS.md.
        """
        self.modes.excite(amplitude, gain_scale)
        self.noise.excite(amplitude)

    def step(self) -> float:
        """Advance one sample.

        Correct but slow — it goes through the whole block machinery for a
        single sample. Use `render` for anything longer than a few samples.
        """
        return float(self.render(1)[0])

    def render(self, num_samples: int) -> np.ndarray:
        """Advance num_samples, return a float64 mono buffer."""
        num_samples = int(num_samples)
        if num_samples <= 0:
            return np.zeros(0, dtype=np.float64)

        out = np.empty(num_samples, dtype=np.float64)
        block_size = (
            self.control_period if self.tension.is_active else DrumVoice.LINEAR_CHUNK
        )

        position = 0
        while position < num_samples:
            block = min(block_size, num_samples - position)
            ratio = (
                self.tension.update(self.modes.energy)
                if self.tension.is_active
                else 1.0
            )
            chunk = self.modes.process(block, ratio)
            if len(self.noise):
                chunk = chunk + self.noise.process(block)
            out[position : position + block] = chunk
            position += block

        return out * self.output_gain

    def render_seconds(self, seconds: float) -> np.ndarray:
        return self.render(int(round(seconds * self.sr)))

    def render_hit(self, seconds: float = 4.0, amplitude: float = 1.0) -> np.ndarray:
        """reset() + strike() + render_seconds(). Convenience for A/B against a WAV."""
        self.reset()
        self.strike(amplitude)
        return self.render_seconds(seconds)

    def render_sequence(
        self, hits: list[tuple[float, float]], seconds: float
    ) -> np.ndarray:
        """Render `hits` as (time_seconds, amplitude) onto one timeline.

        Strikes superpose onto whatever is still ringing, with no crossfade and
        no voice stealing — that is the whole point of a resonator bank with
        persistent state. Flams, ghost notes and rolls need no special case.
        """
        self.reset()
        total = int(round(seconds * self.sr))
        out = np.zeros(total, dtype=np.float64)

        schedule = sorted(
            (max(0, int(round(t * self.sr))), float(a)) for t, a in hits
        )
        position = 0
        for onset, amplitude in schedule:
            onset = min(onset, total)
            if onset > position:
                out[position:onset] = self.render(onset - position)
                position = onset
            self.strike(amplitude)
        if position < total:
            out[position:total] = self.render(total - position)
        return out

    def set_monitor(self, mask: np.ndarray | None) -> None:
        """Solo or mute individual modes while they ring. See ModalBank."""
        self.modes.set_monitor_mask(mask)

    def update_params(self, params: DrumParams) -> None:
        """Adopt a new parameter set WITHOUT interrupting what is ringing.

        This is what makes live editing worth having: a knob moved while the
        drum is decaying changes the rest of that decay. Nothing is rebuilt
        unless it actually changed — redesigning a noise band's filter resets
        its state, which clicks, and it should not happen because a level
        slider moved.

        Mode order is identity here: mode i inherits mode i's state, so
        reordering the list while playing reassigns ring state between
        partials. Adding modes is safe (they start silent), removing them is
        safe (their ring goes with them).
        """
        params.validate(self.sr)
        previous, self.params = self.params, params

        if params.modes != previous.modes:
            self.modes.set_params(params.modes)
        if params.noise != previous.noise:
            self.noise.set_bands(params.noise)
        if params.tension != previous.tension:
            smoothed = self.tension.smoothed
            self.tension = TensionTracker(
                params.tension, self.sr, self.control_period
            )
            # Carry the energy estimate across, or the glide restarts from rest
            # the moment k or tau is touched.
            self.tension.smoothed = smoothed
            self.tension.update(self.modes.energy)
        self.output_gain = float(params.output_gain)

    def reset(self) -> None:
        """Clear all state. Does not touch params."""
        self.modes.reset()
        self.noise.reset()
        self.tension.reset()
        self._rng = np.random.default_rng(self.seed)
        self.noise.reseed(self._rng)

    # -- inspection -----------------------------------------------------------

    @property
    def energy(self) -> float:
        return self.modes.energy

    @property
    def frequency_ratio(self) -> float:
        """Current tension-induced ratio. 1.0 at rest."""
        return self.tension.ratio

    @property
    def amplitude(self) -> float:
        """Largest single-mode envelope currently ringing."""
        return max(self.modes.peak_amplitude, self.noise.amplitude)

    def is_silent(self, threshold_db: float = SILENCE_DB) -> bool:
        """True once the whole voice is below `threshold_db`.

        A method, not a property: the skeleton declared this as a property
        taking an argument, which Python cannot express. `silent` below is the
        zero-argument form.
        """
        return Decibels.from_amplitude(self.amplitude * self.output_gain) < threshold_db

    @property
    def silent(self) -> bool:
        """`is_silent()` at the default threshold, for voice-pool bookkeeping."""
        return self.is_silent()

    def mode_frequencies(self) -> np.ndarray:
        """Current (tension-shifted) frequency of every mode, in Hz."""
        return self.modes.frequencies()

    def glide_trace(self, seconds: float = 2.0, amplitude: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """(times, fundamental frequency) over a hit, straight from the engine.

        The ground truth to compare `GlideAnalyzer` against — if the analyzer
        disagrees with this on synthetic audio, the analyzer is wrong.
        """
        self.reset()
        self.strike(amplitude)
        total = int(round(seconds * self.sr))
        block = self.control_period if self.tension.is_active else DrumVoice.LINEAR_CHUNK
        times, freqs = [], []
        fundamental = self.params.fundamental()

        position = 0
        while position < total:
            span = min(block, total - position)
            ratio = (
                self.tension.update(self.modes.energy)
                if self.tension.is_active
                else 1.0
            )
            times.append(position / self.sr)
            freqs.append(fundamental * ratio)
            self.modes.process(span, ratio)
            if len(self.noise):
                self.noise.process(span)
            position += span

        return np.asarray(times), np.asarray(freqs)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"DrumVoice({self.params.name or '<unnamed>'}, "
            f"{len(self.modes)} modes, {len(self.noise)} noise bands, sr={self.sr})"
        )


class DrumKit:
    """Container for several DrumVoice instances, mixed to one output.

    Sympathetic coupling between drums is deliberately absent — every voice is
    independent, and a strike on one does not excite another.
    """

    def __init__(self, sr: int = Audio.DEFAULT_SR) -> None:
        self.sr = int(sr)
        self._voices: dict[str, DrumVoice] = {}

    def add(
        self,
        name: str,
        params: DrumParams,
        control_period: int = 1,
        seed: int | None = None,
    ) -> DrumVoice:
        voice = DrumVoice(params, self.sr, control_period, seed)
        self._voices[name] = voice
        return voice

    def strike(self, name: str, amplitude: float = 1.0) -> None:
        self[name].strike(amplitude)

    def render(self, num_samples: int) -> np.ndarray:
        out = np.zeros(int(max(num_samples, 0)), dtype=np.float64)
        for voice in self._voices.values():
            out += voice.render(num_samples)
        return out

    def render_seconds(self, seconds: float) -> np.ndarray:
        return self.render(int(round(seconds * self.sr)))

    def render_pattern(
        self, hits: list[tuple[float, str, float]], seconds: float
    ) -> np.ndarray:
        """Render (time_seconds, drum_name, amplitude) triples onto one timeline."""
        self.reset()
        total = int(round(seconds * self.sr))
        out = np.zeros(total, dtype=np.float64)

        schedule = sorted(
            (max(0, int(round(t * self.sr))), name, float(a)) for t, name, a in hits
        )
        position = 0
        for onset, name, amplitude in schedule:
            onset = min(onset, total)
            if onset > position:
                out[position:onset] = self.render(onset - position)
                position = onset
            self.strike(name, amplitude)
        if position < total:
            out[position:total] = self.render(total - position)
        return out

    def write(self, path: str, signal: np.ndarray, normalize: bool = True):
        return AudioIO.write(path, signal, self.sr, normalize=normalize)

    def reset(self) -> None:
        for voice in self._voices.values():
            voice.reset()

    def names(self) -> list[str]:
        return list(self._voices)

    def __contains__(self, name: str) -> bool:
        return name in self._voices

    def __len__(self) -> int:
        return len(self._voices)

    def __getitem__(self, name: str) -> DrumVoice:
        if name not in self._voices:
            raise KeyError(
                f"no drum named {name!r} in this kit; have {sorted(self._voices)}"
            )
        return self._voices[name]

    def __iter__(self) -> Iterator[DrumVoice]:
        return iter(self._voices.values())
