"""The generated drum beside the sample, in the four views that show something.

The ScoreCard says *how far off* and *which parameter*. This says *where* — in
time, in frequency, and in both at once — which is what you need to decide
whether a fit is wrong or merely different.

Every function here returns arrays, not pictures. The plotting belongs to
whatever is displaying them; what belongs to the library is the decision about
what to plot and on what axes, because those decisions are the ones that make a
comparison honest:

* **Envelopes are in dB against a shared floor.** A linear waveform overlay of
  two drums shows the attack and nothing else — everything after the first
  30 ms is a flat line at zero on a scale set by the peak.
* **Spectra are averaged over the whole hit, not one frame.** A single frame is
  a sample of the noise as much as of the drum, and §6.5 is explicit that
  comparing two noise realizations bin by bin measures the seed.
* **Both signals are level-matched first.** The absolute level is a mic preamp
  setting; a comparison that lets it in reports the gain staging.
* **Spectrograms share one colour range.** Two images auto-scaled to their own
  maxima look similar no matter how different they are.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.constants import Audio, Decibels
from ..scoring.bands import BandDecayAnalyzer
from .objective import LevelMatch, SpectralTarget


def _stft(signal: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """(frames, bins) magnitude."""
    signal = np.asarray(signal, dtype=np.float64)
    if len(signal) < n_fft:
        signal = np.pad(signal, (0, n_fft - len(signal)))
    window = np.hanning(n_fft)
    frames = np.lib.stride_tricks.sliding_window_view(signal, n_fft)[::hop]
    return np.abs(np.fft.rfft(frames * window, axis=-1))


@dataclass
class Envelopes:
    """Amplitude against time, in dB, for both signals."""

    times: np.ndarray
    reference_db: np.ndarray
    generated_db: np.ndarray
    floor_db: float

    #: Peak-to-RMS, the number the architecture's §2 measurements report and
    #: the one that catches an attack that is too sharp.
    reference_crest_db: float = 0.0
    generated_crest_db: float = 0.0

    #: Where each signal reaches its maximum. §2 measured ~10 ms on the
    #: reference floor tom; a modal bank struck at cosine phase peaks on the
    #: first sample, and this is where that shows.
    reference_peak_ms: float = 0.0
    generated_peak_ms: float = 0.0


@dataclass
class Spectra:
    """Long-term average magnitude, in dB, for both signals."""

    freqs: np.ndarray
    reference_db: np.ndarray
    generated_db: np.ndarray


@dataclass
class Spectrograms:
    """Both signals as (frames, bands) dB images on one shared scale."""

    times: np.ndarray
    freqs: np.ndarray
    reference_db: np.ndarray
    generated_db: np.ndarray
    difference_db: np.ndarray
    vmin: float
    vmax: float


@dataclass
class BandDecays:
    """Per-band t60, the quantity stage 1 fits the damping curve through."""

    centres: np.ndarray
    reference_t60: np.ndarray
    generated_t60: np.ndarray


class Comparison:
    """Everything the report shows, computed from two signals.

    Construction level-matches the candidate to the reference once, and every
    view below uses that matched copy.
    """

    #: The envelope follower's window. Long enough not to track individual
    #: cycles of the fundamental, short enough to show the attack.
    ENVELOPE_MS: float = 3.0

    #: Anything this far below the reference's peak is the recording, not the
    #: drum (§6.5).
    FLOOR_DB: float = -80.0

    def __init__(self, reference: np.ndarray, generated: np.ndarray,
                 sr: int = Audio.DEFAULT_SR) -> None:
        self.sr = int(sr)
        reference = np.asarray(reference, dtype=np.float64)
        generated = np.asarray(generated, dtype=np.float64)

        length = min(len(reference), len(generated))
        self.reference = reference[:length]
        self.generated = generated[:length] * LevelMatch.match_rms(
            generated[:length], reference[:length])

    # -- time -----------------------------------------------------------------

    def envelopes(self) -> Envelopes:
        window = max(4, int(Comparison.ENVELOPE_MS * 1e-3 * self.sr))

        def follow(signal: np.ndarray) -> np.ndarray:
            padded = np.pad(signal**2, (0, window - len(signal) % window or 0))
            frames = padded[: len(padded) // window * window].reshape(-1, window)
            return np.sqrt(frames.mean(axis=1))

        reference = follow(self.reference)
        generated = follow(self.generated)
        peak = max(reference.max(), generated.max(), 1e-12)
        times = np.arange(len(reference)) * window / self.sr

        def crest(signal: np.ndarray) -> float:
            rms = float(np.sqrt(np.mean(signal**2)))
            top = float(np.max(np.abs(signal)))
            return Decibels.from_amplitude(top / rms) if rms > 0 else 0.0

        return Envelopes(
            times=times,
            reference_db=np.maximum(
                Decibels.from_amplitude(reference / peak), Comparison.FLOOR_DB),
            generated_db=np.maximum(
                Decibels.from_amplitude(generated / peak), Comparison.FLOOR_DB),
            floor_db=Comparison.FLOOR_DB,
            reference_crest_db=crest(self.reference),
            generated_crest_db=crest(self.generated),
            reference_peak_ms=float(np.argmax(reference) * window / self.sr * 1000),
            generated_peak_ms=float(np.argmax(generated) * window / self.sr * 1000),
        )

    def waveforms(self, points: int = 1200) -> dict:
        """Min/max per pixel column — the shape a DAW draws, not a decimation
        that aliases the peaks away."""
        length = len(self.reference)
        step = max(1, length // points)
        usable = length // step * step

        def envelope(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            blocks = signal[:usable].reshape(-1, step)
            return blocks.min(axis=1), blocks.max(axis=1)

        reference_low, reference_high = envelope(self.reference)
        generated_low, generated_high = envelope(self.generated)
        return {
            "times": np.arange(len(reference_low)) * step / self.sr,
            "reference_low": reference_low, "reference_high": reference_high,
            "generated_low": generated_low, "generated_high": generated_high,
        }

    # -- frequency ------------------------------------------------------------

    def spectra(self, n_fft: int = 8192) -> Spectra:
        """Magnitude averaged over the whole hit."""
        freqs = np.fft.rfftfreq(n_fft, 1.0 / self.sr)

        def average(signal: np.ndarray) -> np.ndarray:
            magnitude = _stft(signal, n_fft, n_fft // 4).mean(axis=0)
            return Decibels.from_amplitude(np.maximum(magnitude, 1e-12))

        reference = average(self.reference)
        generated = average(self.generated)
        top = max(reference.max(), generated.max())
        alive = freqs >= 20.0
        return Spectra(
            freqs=freqs[alive],
            reference_db=np.maximum(reference[alive] - top, Comparison.FLOOR_DB),
            generated_db=np.maximum(generated[alive] - top, Comparison.FLOOR_DB),
        )

    # -- both -----------------------------------------------------------------

    def spectrograms(self, n_fft: int = 1024, n_bands: int = 96) -> Spectrograms:
        """Log-frequency spectrograms of both, on one shared dB scale.

        Log-spaced bands rather than raw bins, for the same reason the fitting
        loss uses them: linear bins put nine tenths of the picture above 2 kHz,
        where a drum has almost nothing, and squeeze every mode that matters
        into the bottom rows.
        """
        hop = n_fft // 4
        bins = np.fft.rfftfreq(n_fft, 1.0 / self.sr)
        edges = np.geomspace(30.0, min(16000.0, self.sr / 2 * 0.99), n_bands + 1)
        bank = np.zeros((len(bins), n_bands))
        for index in range(n_bands):
            inside = (bins >= edges[index]) & (bins < edges[index + 1])
            if not inside.any():
                inside = np.zeros_like(bins, dtype=bool)
                inside[int(np.argmin(np.abs(bins - edges[index])))] = True
            bank[inside, index] = 1.0

        def image(signal: np.ndarray) -> np.ndarray:
            power = _stft(signal, n_fft, hop) ** 2
            return 10.0 * np.log10(power @ bank + 1e-12)

        reference = image(self.reference)
        generated = image(self.generated)
        rows = min(len(reference), len(generated))
        reference, generated = reference[:rows], generated[:rows]

        vmax = float(max(reference.max(), generated.max()))
        return Spectrograms(
            times=np.arange(rows) * hop / self.sr,
            freqs=np.sqrt(edges[:-1] * edges[1:]),
            reference_db=reference,
            generated_db=generated,
            difference_db=generated - reference,
            vmin=vmax + Comparison.FLOOR_DB,
            vmax=vmax,
        )

    def band_decays(self) -> BandDecays:
        analyzer = BandDecayAnalyzer(self.sr)
        reference = analyzer.analyze(self.reference)
        generated = analyzer.analyze(self.generated)
        # A band that never rose above the noise floor has no slope to fit, and
        # its t60 is not a measurement. Dropped from both sides together, so
        # the two curves stay on the same axis.
        usable = [
            index for index in range(min(len(reference), len(generated)))
            if reference[index].is_valid and generated[index].is_valid
        ]
        return BandDecays(
            centres=np.array([
                float(np.sqrt(reference[i].f_low * reference[i].f_high))
                for i in usable]),
            reference_t60=np.array([reference[i].t60 for i in usable]),
            generated_t60=np.array([generated[i].t60 for i in usable]),
        )

    # -- one number -----------------------------------------------------------

    def band_distance(self) -> float:
        """The loss stage 5 minimizes, for this pair."""
        return SpectralTarget(self.reference, self.sr).distance(self.generated)
