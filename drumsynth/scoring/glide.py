"""Tracking the fundamental's frequency over time — the tension signature.

Short-window spectral peak location, not just Hilbert instantaneous frequency.
The peak-location test is what distinguishes a genuine glide from two static
modes trading dominance: two resolved sinusoids put the spectral peak at one
frequency or the other, never in between, so a peak landing at 95.6 Hz when the
modes sit at 92.8 and 100.3 Hz is proof of real frequency modulation.

Window length is a direct trade, and the default is 150 ms rather than the
250 ms the skeleton proposed. Two reasons, both measured against the engine's
own `DrumVoice.glide_trace`:

  * A window cannot report anything before its own center, so 250 ms is blind
    until t = 125 ms — and the reference glide falls from 104.3 to 98.4 Hz
    inside the first 160 ms. Most of the depth happens where a 250 ms window
    cannot see it.
  * At 150 ms the tracked frequency lands within ~2 cents of ground truth from
    the second frame onward, against ~10 cents at 250 ms.

It is still far short of the ~500 ms needed to resolve the 88.0/92.8 Hz pair,
which is deliberate: an unresolved pair reports its moving centroid, and that
is what a glide track should follow. Both signals go through this same window,
so whatever bias it introduces is common to both sides of a comparison.
"""

from __future__ import annotations

import numpy as np

from ..core.constants import Audio, Cents, Decibels
from ..core.dsp import SpectralPeak
from .descriptors import GlideTrack


class GlideAnalyzer:
    """Frame-by-frame spectral peak location within a narrow frequency range."""

    #: Zero-padding factor. Costs nothing and makes the parabolic interpolation
    #: work on a smoother curve.
    PAD_FACTOR: int = 8

    #: A frame this far below the loudest frame is not trusted. The reference
    #: decays into its codec floor and the peak location becomes noise there.
    RELIABLE_RANGE_DB: float = 40.0

    #: Fraction of the asymptote a frame must come within to count as settled.
    SETTLE_TOLERANCE: float = 0.02

    #: Frames averaged for f_asymptote. f_initial uses the first frame alone —
    #: averaging across the steepest part of the glide flattens the depth.
    EDGE_FRAMES: int = 3

    def __init__(
        self,
        sr: int = Audio.DEFAULT_SR,
        f_range: tuple[float, float] = (60.0, 140.0),
        window_seconds: float = 0.15,
        hop_seconds: float = 0.025,
    ) -> None:
        self.sr = int(sr)
        self.f_range = (float(f_range[0]), float(f_range[1]))
        self.window = max(16, int(round(window_seconds * sr)))
        self.hop = max(1, int(round(hop_seconds * sr)))
        self.n_fft = int(2 ** np.ceil(np.log2(self.window * GlideAnalyzer.PAD_FACTOR)))
        self._window_fn = np.hanning(self.window)
        self._freqs = np.fft.rfftfreq(self.n_fft, 1.0 / self.sr)
        self._band = np.flatnonzero(
            (self._freqs >= self.f_range[0]) & (self._freqs <= self.f_range[1])
        )

    # -- framing --------------------------------------------------------------

    def _frames(self, x: np.ndarray):
        """Yield (start_sample, windowed frame) for every full frame."""
        x = np.asarray(x, dtype=np.float64)
        for start in range(0, max(0, len(x) - self.window + 1), self.hop):
            yield start, x[start : start + self.window] * self._window_fn

    def _spectrum(self, frame: np.ndarray) -> np.ndarray:
        return np.abs(np.fft.rfft(frame, self.n_fft))

    # -- analysis -------------------------------------------------------------

    def analyze(self, x: np.ndarray) -> GlideTrack:
        x = np.asarray(x, dtype=np.float64)
        if self._band.size < 3 or len(x) < self.window:
            return GlideTrack(np.zeros(0), np.zeros(0), np.nan, np.nan, 0.0, 0.0,
                              np.zeros(0, dtype=bool))

        times, freqs, levels = [], [], []
        bin_hz = self.sr / self.n_fft
        for start, frame in self._frames(x):
            spectrum = self._spectrum(frame)
            band = spectrum[self._band]
            local_peak = int(np.argmax(band))
            refined, magnitude = SpectralPeak.refine(band, local_peak)
            times.append((start + 0.5 * self.window) / self.sr)
            freqs.append((self._band[0] + refined) * bin_hz)
            levels.append(magnitude)

        times = np.asarray(times)
        freqs = np.asarray(freqs)
        levels_db = Decibels.from_amplitude(np.asarray(levels))
        reliable = levels_db > (np.max(levels_db) - GlideAnalyzer.RELIABLE_RANGE_DB)

        return self._summarize(times, freqs, reliable)

    def _summarize(
        self, times: np.ndarray, freqs: np.ndarray, reliable: np.ndarray
    ) -> GlideTrack:
        good = np.flatnonzero(reliable)
        if good.size == 0:
            return GlideTrack(times, freqs, np.nan, np.nan, 0.0, 0.0, reliable)

        edge = GlideAnalyzer.EDGE_FRAMES
        f_initial = float(freqs[good[0]])
        f_asymptote = float(np.median(freqs[good[-min(edge * 2, good.size):]]))
        depth_cents = float(Cents.between(f_initial, f_asymptote))

        settle_time = float(times[good[-1]])
        tolerance = GlideAnalyzer.SETTLE_TOLERANCE * f_asymptote
        for index in good:
            # First frame from which everything that follows stays settled.
            rest = good[good >= index]
            if np.all(np.abs(freqs[rest] - f_asymptote) <= tolerance):
                settle_time = float(times[index])
                break

        return GlideTrack(
            times=times,
            freqs=freqs,
            f_initial=f_initial,
            f_asymptote=f_asymptote,
            depth_cents=depth_cents,
            settle_time=settle_time,
            reliable=reliable,
        )

    # -- diagnostics ----------------------------------------------------------

    def count_peaks_per_window(
        self, x: np.ndarray, relative_threshold_db: float = -12.0
    ) -> list[int]:
        """Peaks inside the tracked range, per frame.

        More than one peak in a window means the glide model is the wrong
        explanation for that region and you are looking at separate modes. Run
        this before believing any glide result on unfamiliar material.
        """
        counts = []
        for _, frame in self._frames(x):
            band = self._spectrum(frame)[self._band]
            counts.append(len(SpectralPeak.find_all(band, relative_threshold_db)))
        return counts

    def is_genuine_glide(self, x: np.ndarray, min_single_peak_fraction: float = 0.8) -> bool:
        """True when most windows hold a single sliding lobe.

        The falsification test from the architecture: a sum of two resolved
        sinusoids peaks at one frequency or the other, never in between, so a
        single lobe that moves is real frequency modulation.

        Expect False on a drum carrying a deliberate close pair near the
        fundamental — two modes 92 cents apart are two modes, and this
        correctly says so. That is the diagnostic working, not failing: it
        means the glide track in that region is a centroid of two partials and
        should not be read as one mode's frequency.
        """
        counts = self.count_peaks_per_window(x)
        if not counts:
            return False
        return float(np.mean(np.asarray(counts) <= 1)) > min_single_peak_fraction
