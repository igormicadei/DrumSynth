"""What the fitter optimizes, and the trick that makes it cheap.

A full render costs ~0.2 s, which would put a population search into the hours.
It does not have to. For a FIXED tension trajectory and a fixed noise seed, the
output is **linear** in the excitation parameters:

    audio = mode_gains @ modal_basis + noise_levels @ noise_basis

because `excite()` adds `amplitude * gain` to a mode's state and everything
downstream of that is a linear, time-invariant response. Precompute the two
bases once and a candidate render becomes one matrix product — microseconds.

The tension feedback is what makes this approximate rather than exact: bank
energy depends on the gains, so changing them changes the trajectory that the
basis was built with. That is handled by relinearizing — build the basis from
the current best estimate, solve, re-render for real, repeat. Two or three
passes is enough because the trajectory moves very little once the gains are in
the right neighbourhood.

The loss is the multi-resolution log-magnitude STFT distance. §6.2 of the
architecture argues for exactly this shape and §6.5 warns that a single scalar
is useless for hand-tuning — which is right, and is why the ScoreCard is what
gets reported while this is only what gets minimized.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.constants import Audio, Decibels
from ..synth.noise import NoiseBank
from ..synth.params import DrumParams, Mode
from ..synth.resonators import ModalBank, TensionTracker
from ..synth.voice import DrumVoice


class TensionTrajectory:
    """The sequence of (block length, frequency ratio) a hit produces.

    Captured from a real render so the linearized basis rings at the same
    frequencies the true engine would, glide and all.
    """

    def __init__(self, params: DrumParams, n_samples: int, sr: int,
                 control_period: int = 64, amplitude: float = 1.0) -> None:
        self.blocks: list[tuple[int, float]] = []
        voice = DrumVoice(params, sr, control_period, seed=0)
        voice.strike(amplitude)

        block_size = control_period if voice.tension.is_active else DrumVoice.LINEAR_CHUNK
        position = 0
        while position < n_samples:
            span = min(block_size, n_samples - position)
            ratio = (
                voice.tension.update(voice.modes.energy)
                if voice.tension.is_active
                else 1.0
            )
            self.blocks.append((span, ratio))
            voice.modes.process(span, ratio)
            if len(voice.noise):
                voice.noise.process(span)
            position += span

    @property
    def ratios(self) -> np.ndarray:
        return np.array([ratio for _, ratio in self.blocks])


@dataclass
class LinearVoiceBasis:
    """Unit-gain response of every mode and every noise band.

    `modal[i]` is what mode i contributes to a unit strike at gain 1, and
    `noise[j]` the same for noise band j at level 1. Any candidate is then

        audio = output_gain * (gains @ modal + levels @ noise)
    """

    modal: np.ndarray            # (n_modes, n_samples)
    noise: np.ndarray            # (n_bands, n_samples)
    sr: int
    trajectory: TensionTrajectory

    @classmethod
    def build(
        cls,
        params: DrumParams,
        n_samples: int,
        sr: int = Audio.DEFAULT_SR,
        control_period: int = 64,
        amplitude: float = 1.0,
        seed: int = 0,
    ) -> "LinearVoiceBasis":
        trajectory = TensionTrajectory(params, n_samples, sr, control_period, amplitude)

        unit_modes = [Mode(mode.f_static, 1.0, mode.t60) for mode in params.modes]
        bank = ModalBank(unit_modes, sr)
        bank.excite(amplitude)
        modal = np.concatenate(
            [bank.process_modes(span, ratio) for span, ratio in trajectory.blocks],
            axis=1,
        )

        if params.noise:
            unit_bands = [
                type(band)(band.f_low, band.f_high, 1.0, band.t60)
                for band in params.noise
            ]
            rows = []
            for index, band in enumerate(unit_bands):
                # One voice per band, each with its own stream, so a level
                # change never reshuffles which noise the other bands get.
                single = NoiseBank([band], sr, np.random.default_rng(seed + index))
                single.excite(amplitude)
                rows.append(
                    np.concatenate(
                        [single.process(span) for span, _ in trajectory.blocks]
                    )
                )
            noise = np.stack(rows)
        else:
            noise = np.zeros((0, n_samples), dtype=np.float64)

        return cls(modal=modal, noise=noise, sr=sr, trajectory=trajectory)

    # -- rendering ------------------------------------------------------------

    @property
    def n_modes(self) -> int:
        return self.modal.shape[0]

    @property
    def n_bands(self) -> int:
        return self.noise.shape[0]

    @property
    def n_samples(self) -> int:
        return self.modal.shape[1]

    def render(self, gains: np.ndarray, levels: np.ndarray,
               output_gain: float = 1.0) -> np.ndarray:
        out = np.asarray(gains, dtype=np.float64) @ self.modal
        if self.n_bands and levels is not None and len(levels):
            out = out + np.asarray(levels, dtype=np.float64) @ self.noise
        return out * output_gain

    def render_batch(self, gains: np.ndarray, levels: np.ndarray | None = None,
                     output_gain: float = 1.0) -> np.ndarray:
        """One row of audio per candidate. (P, M) @ (M, N) in a single BLAS call."""
        out = np.asarray(gains, dtype=np.float64) @ self.modal
        if self.n_bands and levels is not None and np.size(levels):
            out = out + np.asarray(levels, dtype=np.float64) @ self.noise
        return out * output_gain

    def mode_power(self, n_fft: int = 2048) -> np.ndarray:
        """Per-mode spectrogram power, summed over time.

        The modes sit at different frequencies, so their spectrograms barely
        overlap and total power is close to the sum of the parts. That makes
        `power = (gains**2) @ mode_power` an accurate enough model to solve for
        the gains in closed form, which is how the fit gets its starting point
        instead of searching for one.
        """
        window = np.hanning(n_fft)
        hop = n_fft // 4
        rows = []
        for signal in self.modal:
            frames = _frame(signal, n_fft, hop)
            spectrum = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
            rows.append(spectrum.sum(axis=0))
        return np.stack(rows) if rows else np.zeros((0, n_fft // 2 + 1))


def _frame(signal: np.ndarray, size: int, hop: int) -> np.ndarray:
    if len(signal) < size:
        signal = np.pad(signal, (0, size - len(signal)))
    starts = range(0, len(signal) - size + 1, hop)
    return np.stack([signal[start : start + size] for start in starts])


class SpectralTarget:
    """The reference, pre-transformed once, in LOG-SPACED BANDS.

    Two decisions, and the second one is the whole reason this class is not
    just `MultiResolutionSTFTLoss` with the reference cached.

    **Cached.** Half the cost of a multi-resolution loss is transforming the
    reference, and it never changes during a fit.

    **Band-aggregated.** A bin-by-bin log-magnitude distance compares the noise
    bank sample by sample, which §6.5 forbids outright: two realizations of the
    same noise process have roughly zero correlation. Measured on this
    synthesizer, two renders of *identical parameters* with different noise
    seeds sit 11.7 dB apart under a bin-wise loss — a floor far above any error
    the fit is trying to remove, and an optimizer handed that will spend its
    budget chasing a particular noise realization.

    Summing power into log-spaced bands first averages the variance away. The
    modes still separate, because log-spaced bands are narrow exactly where the
    partials are, and the noise becomes what it should have been all along: a
    spectral envelope over time.
    """

    #: Bands from the lowest mode a membrane drum has to the top of hearing.
    BAND_RANGE: tuple[float, float] = (30.0, 16000.0)
    N_BANDS: int = 64

    #: Bands this far below the loudest one carry no information about the drum,
    #: only about the recording (§6.5).
    FLOOR_DB: float = -70.0

    def __init__(
        self,
        reference: np.ndarray,
        sr: int = Audio.DEFAULT_SR,
        fft_sizes: tuple[int, ...] = (256, 1024, 4096),
        n_bands: int = N_BANDS,
        floor_db: float = FLOOR_DB,
        log_eps: float = 1e-12,
    ) -> None:
        self.sr = int(sr)
        self.fft_sizes = tuple(fft_sizes)
        self.floor_db = float(floor_db)
        self.log_eps = float(log_eps)
        self.raw = np.asarray(reference, dtype=np.float64)
        self.length = len(self.raw)

        self._windows = {size: np.hanning(size) for size in self.fft_sizes}
        self._banks = {
            size: self._band_matrix(size, n_bands) for size in self.fft_sizes
        }
        self._reference = {
            size: self._band_spectrogram(self.raw, size) for size in self.fft_sizes
        }
        self._weights = {}
        for size, spectrogram in self._reference.items():
            alive = spectrogram > (np.max(spectrogram) + self.floor_db)
            self._weights[size] = alive.astype(np.float64)

    def _band_matrix(self, n_fft: int, n_bands: int) -> np.ndarray:
        """(bins, bands) summing matrix over log-spaced edges."""
        freqs = np.fft.rfftfreq(n_fft, 1.0 / self.sr)
        low, high = SpectralTarget.BAND_RANGE
        edges = np.geomspace(low, min(high, self.sr / 2 * 0.99), n_bands + 1)
        matrix = np.zeros((len(freqs), n_bands))
        for index in range(n_bands):
            inside = (freqs >= edges[index]) & (freqs < edges[index + 1])
            if not inside.any():
                # Narrower than one bin at this resolution: take the nearest.
                inside = np.zeros_like(freqs, dtype=bool)
                inside[int(np.argmin(np.abs(freqs - edges[index])))] = True
            matrix[inside, index] = 1.0
        return matrix

    def _band_spectrogram(self, signal: np.ndarray, size: int) -> np.ndarray:
        frames = _frame(np.asarray(signal, dtype=np.float64), size, size // 4)
        power = np.abs(np.fft.rfft(frames * self._windows[size], axis=1)) ** 2
        banded = power @ self._banks[size]
        return 10.0 * np.log10(banded + self.log_eps)

    # -- what a batched backend needs -----------------------------------------
    #
    # The transform is defined here and nowhere else. A GPU backend has to
    # reproduce it exactly or its losses are not comparable with the ones every
    # other stage reports, so it borrows these rather than reimplementing them.

    HOP_DIVISOR: int = 4

    @staticmethod
    def hop(n_fft: int) -> int:
        return n_fft // SpectralTarget.HOP_DIVISOR

    def window(self, n_fft: int) -> np.ndarray:
        return self._windows[n_fft]

    def band_matrix(self, n_fft: int) -> np.ndarray:
        """(bins, bands), summing power into log-spaced bands."""
        return self._banks[n_fft]

    def reference_bands(self, n_fft: int) -> np.ndarray:
        """(frames, bands) of the reference, in dB."""
        return self._reference[n_fft]

    def band_weights(self, n_fft: int) -> np.ndarray:
        """(frames, bands), 1 where the reference is above the floor."""
        return self._weights[n_fft]

    def distance(self, candidate: np.ndarray) -> float:
        """Weighted mean absolute band-level distance, in dB, over all
        resolutions."""
        total, weight_total = 0.0, 0.0
        for size in self.fft_sizes:
            reference = self._reference[size]
            weights = self._weights[size]
            other = self._band_spectrogram(candidate, size)
            rows = min(len(reference), len(other))
            if rows == 0:
                continue
            error = np.abs(reference[:rows] - other[:rows]) * weights[:rows]
            total += float(error.sum())
            weight_total += float(weights[:rows].sum())
        return total / weight_total if weight_total > 0 else float("inf")

    def distance_batch(self, candidates: np.ndarray) -> np.ndarray:
        return np.array([self.distance(row) for row in candidates])

    def reference_power(self, n_fft: int = 2048) -> np.ndarray:
        """Power spectrum of the reference, summed over time.

        Feeds the closed-form gain solve: modes sit at different frequencies,
        so total power is close to the sum of the per-mode parts.
        """
        window = np.hanning(n_fft)
        frames = _frame(self.raw, n_fft, n_fft // 4)
        return (np.abs(np.fft.rfft(frames * window, axis=1)) ** 2).sum(axis=0)


@dataclass
class FitObjective:
    """Everything a stage needs to score a candidate parameter set."""

    target: SpectralTarget
    basis: LinearVoiceBasis
    reference: np.ndarray = field(repr=False, default=None)

    def loss(self, gains: np.ndarray, levels: np.ndarray, output_gain: float) -> float:
        return self.target.distance(self.basis.render(gains, levels, output_gain))

    def render(self, gains: np.ndarray, levels: np.ndarray,
               output_gain: float) -> np.ndarray:
        return self.basis.render(gains, levels, output_gain)


class LevelMatch:
    """Absolute level, solved rather than searched.

    A single scalar multiplying the whole render is the one parameter with a
    closed-form optimum, and leaving it in the search wastes a dimension on
    something arithmetic can do exactly.
    """

    @staticmethod
    def best_gain(candidate: np.ndarray, reference: np.ndarray) -> float:
        """The scale that minimizes squared error against the reference."""
        denominator = float(np.dot(candidate, candidate))
        if denominator <= 0:
            return 1.0
        return float(np.dot(candidate, reference) / denominator)

    @staticmethod
    def match_rms(candidate: np.ndarray, reference: np.ndarray) -> float:
        """The scale that equalizes RMS. More robust than least squares when
        the two are not phase-aligned, which after a fit they never are."""
        candidate_rms = float(np.sqrt(np.mean(candidate**2)))
        reference_rms = float(np.sqrt(np.mean(reference**2)))
        return reference_rms / candidate_rms if candidate_rms > 0 else 1.0
