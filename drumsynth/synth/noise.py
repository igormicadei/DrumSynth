"""Contact-noise generation: white noise, bandpassed, exponentially decayed.

The noise bank carries the part of the strike that is not resonance — stick on
head, air, the broadband splat in the first few tens of milliseconds. It has no
attack: level starts at maximum and decays immediately. The ~10 ms envelope
peak a real drum shows comes from the modal bank summing up, not from here.

Levels are defined as RMS amplitude at t=0, not as a raw multiplier on the
filter output. Each band divides out its own filter's noise gain, so `level`
means the same thing in a 600 Hz-wide band and a 9 kHz-wide one — otherwise a
fitter comparing two bands is really comparing their bandwidths.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as _signal

from ..core.constants import Audio, Decay
from ..core.dsp import FilterDesign
from .params import NoiseBand


class NoiseVoice:
    """One band-limited noise burst: white -> bandpass -> exponential envelope."""

    __slots__ = ("sr", "band", "level", "t60", "env", "env_coef", "sos", "zi",
                 "_rng", "_noise_gain")

    #: Order of the bandpass. Higher isolates the band better but rings longer,
    #: which matters when the envelope is 5 ms.
    FILTER_ORDER: int = 2

    def __init__(
        self,
        band: NoiseBand,
        sr: int = Audio.DEFAULT_SR,
        rng: np.random.Generator | None = None,
    ) -> None:
        band.validate(sr)
        self.sr = int(sr)
        self.band = band
        self.level = float(band.level)
        self.t60 = float(band.t60)
        self.env = 0.0
        self.env_coef = float(Decay.t60_to_coef(band.t60, sr))
        self.sos = FilterDesign.bandpass(
            band.f_low, band.f_high, sr, order=NoiseVoice.FILTER_ORDER
        )
        self.zi = np.zeros((self.sos.shape[0], 2), dtype=np.float64)
        self._rng = rng if rng is not None else np.random.default_rng()
        self._noise_gain = self._measure_noise_gain()

    def _measure_noise_gain(self) -> float:
        """RMS output of this filter for unit-variance white noise in.

        Computed from the frequency response rather than by generating noise,
        so it is deterministic and costs nothing.
        """
        freqs = np.linspace(0.0, 0.5 * self.sr, 4096)
        response = FilterDesign.response_at(self.sos, freqs, self.sr)
        gain = float(np.sqrt(np.mean(response**2)))
        return max(gain, 1e-9)

    # -- playing --------------------------------------------------------------

    def excite(self, amplitude: float) -> None:
        """env += amplitude * level — additive, for the same reason modes are."""
        self.env += float(amplitude) * self.level

    def step(self) -> float:
        return float(self.process(1)[0])

    def process(self, n: int) -> np.ndarray:
        """Advance n samples. Frequency never changes, so blocks can be any size."""
        if n <= 0:
            return np.zeros(0, dtype=np.float64)
        n = int(n)
        if self.env <= 0.0:
            # Still advance the filter state so a later burst is not preceded by
            # a stale tail, but skip the work when there is nothing to shape.
            self.zi[:] = 0.0
            return np.zeros(n, dtype=np.float64)

        white = self._rng.standard_normal(n)
        filtered, self.zi = _signal.sosfilt(self.sos, white, zi=self.zi)
        filtered /= self._noise_gain

        envelope = self.env * self.env_coef ** np.arange(n, dtype=np.float64)
        self.env *= self.env_coef**n
        if self.env < 1e-12:
            self.env = 0.0
        return filtered * envelope

    # -- inspection -----------------------------------------------------------

    @property
    def amplitude(self) -> float:
        return self.env

    def reseed(self, rng: np.random.Generator) -> None:
        """Swap the random stream. Used by `DrumVoice.reset` so a reset voice
        replays the exact same noise for a seeded render."""
        self._rng = rng

    def reset(self) -> None:
        self.env = 0.0
        self.zi[:] = 0.0

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"NoiseVoice({self.band.f_low:.0f}-{self.band.f_high:.0f}Hz, "
            f"level={self.level:.4g}, t60={self.t60:.4g}s)"
        )


class NoiseBank:
    """Every noise band of one drum, mixed.

    Deliberately thin — noise bands do not interact, and nothing modulates them.
    It exists so `DrumVoice` has one object to talk to on each side of the mix.
    """

    def __init__(
        self,
        bands: list[NoiseBand],
        sr: int = Audio.DEFAULT_SR,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.sr = int(sr)
        self._rng = rng if rng is not None else np.random.default_rng()
        self.voices = [NoiseVoice(band, sr, self._rng) for band in bands]

    def __len__(self) -> int:
        return len(self.voices)

    def __iter__(self):
        return iter(self.voices)

    def __getitem__(self, index: int) -> NoiseVoice:
        return self.voices[index]

    def set_bands(self, bands: list[NoiseBand]) -> None:
        """Replace the bands, carrying each voice's envelope across.

        Only rebuilds a voice whose band edges actually moved — redesigning a
        bandpass resets its filter state, which is an audible click if it
        happens on every touch of a level slider.
        """
        voices: list[NoiseVoice] = []
        for index, band in enumerate(bands):
            existing = self.voices[index] if index < len(self.voices) else None
            if (
                existing is not None
                and existing.band.f_low == band.f_low
                and existing.band.f_high == band.f_high
            ):
                existing.band = band
                existing.level = float(band.level)
                existing.t60 = float(band.t60)
                existing.env_coef = float(Decay.t60_to_coef(band.t60, existing.sr))
                voices.append(existing)
                continue

            voice = NoiseVoice(band, self.sr, self._rng)
            if existing is not None:
                voice.env = existing.env  # keep the burst that is still running
            voices.append(voice)
        self.voices = voices

    def excite(self, amplitude: float) -> None:
        for voice in self.voices:
            voice.excite(amplitude)

    def process(self, n: int) -> np.ndarray:
        out = np.zeros(int(max(n, 0)), dtype=np.float64)
        for voice in self.voices:
            out += voice.process(n)
        return out

    def step(self) -> float:
        return float(sum(voice.step() for voice in self.voices))

    @property
    def amplitude(self) -> float:
        """Largest band envelope still running."""
        return max((voice.amplitude for voice in self.voices), default=0.0)

    @property
    def is_silent(self) -> bool:
        return all(voice.env <= 0.0 for voice in self.voices)

    def reseed(self, rng: np.random.Generator) -> None:
        self._rng = rng
        for voice in self.voices:
            voice.reseed(rng)

    def reset(self) -> None:
        for voice in self.voices:
            voice.reset()
