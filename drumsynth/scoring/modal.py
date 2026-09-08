"""Recovering individual modes from audio.

Band filtering plus Hilbert envelope fitting fails exactly where it matters:
on close pairs, and on partials sitting under a fundamental 40 dB louder. A
high-resolution subspace method — ESPRIT or matrix pencil — estimates frequency
AND damping jointly, resolves modes closer together than the FFT bin spacing,
and does not need the modes separated in advance.

The working trick is the band decomposition. Filter to a band, take the
analytic signal, heterodyne it to baseband and decimate hard. A 30-150 Hz band
survives decimation by 147, so a 2-second segment becomes 600 complex samples
and the whole subspace decomposition costs almost nothing. Without that step
the Hankel matrix for a 2-second segment at 44.1 kHz is unusable.

This is the last piece to build and the one to trust least. Use band decays as
the proxy until the synth is roughly in the right place.
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
from scipy import linalg as _linalg
from scipy import signal as _signal

from ..core.constants import Audio, Decay, Decibels
from ..core.dsp import FilterDesign
from .descriptors import ModeEstimate


class ModalAnalyzer:
    """Subspace mode extraction, band by band."""

    #: Analysis bands. Narrow at the bottom, where resolving 88.0 from 92.8 Hz
    #: is the whole point, and wide at the top, where the ear stops resolving
    #: individual partials anyway.
    #:
    #: These are ACCEPTANCE ranges and they tile the spectrum exactly. The
    #: bandpass actually applied is wider by FILTER_MARGIN on each side, so
    #: every accepted mode sits well inside the passband where the filter's
    #: gain is near unity. A mode sitting on a filter skirt has its amplitude
    #: divided by a small number, and small errors in that number become large
    #: errors in the reported amplitude.
    DEFAULT_BANDS: tuple[tuple[float, float], ...] = (
        (30.0, 150.0),
        (150.0, 300.0),
        (300.0, 600.0),
        (600.0, 1200.0),
        (1200.0, 2400.0),
        (2400.0, 4000.0),
    )

    #: Longest segment fed to the decomposition. Beyond a couple of seconds the
    #: tail is in the noise floor and only degrades the estimate.
    MAX_SEGMENT_SECONDS: float = 2.0

    #: Decimated segment length cap. Bounds the SVD; the Hankel matrix is
    #: roughly (N - M) x M and the cost grows fast.
    MAX_DECIMATED_SAMPLES: int = 1400

    #: Hankel column count, as a fraction of the segment, and its cap.
    PENCIL_FRACTION: float = 1.0 / 3.0
    MAX_PENCIL: int = 220

    #: Range below the band peak that is still worth fitting.
    SEGMENT_RANGE_DB: float = 55.0

    #: How far past its acceptance range each analysis bandpass extends.
    FILTER_MARGIN: float = 0.15

    #: Safety factor on the "faster than the filter itself" rejection.
    SETTLE_MARGIN: float = 2.0

    #: Singular values below this fraction of the largest are noise, and the
    #: model order is truncated there.
    ORDER_THRESHOLD: float = 0.008

    #: How much of its own ringdown the analysis bandpass is allowed to leave in
    #: the segment. The filter's transient is itself a decaying sinusoid at the
    #: band center, and the subspace method cannot tell it from a real mode — it
    #: comes back as a loud pole with an implausibly short t60. Skipping the
    #: filter's ringdown removes the cause; rejecting short-t60 poles afterwards
    #: catches what leaks through.
    FILTER_SETTLE_FRACTION: float = 1.0

    #: Amplitudes are extrapolated back to t=0 across the skipped region. This
    #: caps that extrapolation, so a fast-decaying pole cannot be projected back
    #: into an absurd amplitude — a pole needing more than this much boost has a
    #: t60 close to the filter's own settle time and is barely a mode anyway.
    MAX_EXTRAPOLATION_DB: float = 12.0

    def __init__(
        self,
        sr: int = Audio.DEFAULT_SR,
        max_modes: int = 40,
        f_range: tuple[float, float] = (30.0, 4000.0),
        method: Literal["esprit", "matrix_pencil", "peak_fit"] = "esprit",
        bands: Sequence[tuple[float, float]] | None = None,
        modes_per_band: int = 10,
    ) -> None:
        self.sr = int(sr)
        self.max_modes = int(max_modes)
        self.f_range = (float(f_range[0]), float(f_range[1]))
        self.method = method
        self.modes_per_band = int(modes_per_band)
        self.bands = tuple(
            band
            for band in (bands if bands is not None else ModalAnalyzer.DEFAULT_BANDS)
            if band[1] > self.f_range[0] and band[0] < min(self.f_range[1], 0.5 * self.sr)
        )
        self._filters: dict[tuple[float, float], np.ndarray] = {}
        self._settle: dict[tuple[float, float], float] = {}

    # -- the analysis filters -------------------------------------------------

    def _band_filter(self, f_low: float, f_high: float) -> tuple[np.ndarray, float]:
        """Bandpass for a band, plus how long it rings on its own, in seconds.

        The ringdown is measured from the filter's own impulse response rather
        than estimated from its bandwidth, because a 4th-order Butterworth has
        two pole pairs and the narrower one rings roughly twice as long as the
        bandwidth alone suggests.
        """
        key = (float(f_low), float(f_high))
        if key not in self._filters:
            margin = ModalAnalyzer.FILTER_MARGIN
            sos = FilterDesign.bandpass(
                f_low * (1.0 - margin), f_high * (1.0 + margin), self.sr
            )
            # Half a second is long enough for the narrowest band here to ring
            # out; measuring over too short a window reports a settle time
            # shorter than the filter's, which is worse than not measuring.
            impulse = np.zeros(max(2048, self.sr // 2))
            impulse[0] = 1.0
            response = np.abs(_signal.hilbert(FilterDesign.apply(sos, impulse)))
            peak = int(np.argmax(response))
            below = np.flatnonzero(response[peak:] < response[peak] * 1e-3)
            settle = (peak + int(below[0])) / self.sr if below.size else len(impulse) / self.sr
            self._filters[key] = sos
            self._settle[key] = float(settle)
        return self._filters[key], self._settle[key]

    # -- top level ------------------------------------------------------------

    def analyze(self, x: np.ndarray) -> list[ModeEstimate]:
        """Every mode found across every band, pruned and capped at max_modes."""
        x = np.asarray(x, dtype=np.float64)
        if x.size < 64:
            return []

        found: list[ModeEstimate] = []
        for f_low, f_high in self.bands:
            found.extend(self.analyze_band(x, f_low, f_high, self.modes_per_band))

        pruned = self.prune(found)
        pruned.sort(key=lambda mode: -mode.amplitude * mode.confidence)
        return sorted(pruned[: self.max_modes], key=lambda mode: mode.freq)

    def analyze_band(
        self, x: np.ndarray, f_low: float, f_high: float, n_modes: int
    ) -> list[ModeEstimate]:
        """Decimate to the band, run the subspace method, map frequencies back."""
        f_high = min(f_high, 0.49 * self.sr)
        if f_low >= f_high:
            return []

        sos, settle = self._band_filter(f_low, f_high)
        filtered = FilterDesign.apply(sos, np.asarray(x, dtype=np.float64))
        segment, offset = self._segment(filtered, settle)
        if segment.size < 64:
            return []

        margin = ModalAnalyzer.FILTER_MARGIN
        baseband, sr_dec, f_center, decim_sos = self._to_baseband(
            segment, f_low * (1.0 - margin), f_high * (1.0 + margin)
        )
        if baseband.size < 24:
            return []

        if self.method == "peak_fit":
            poles = self._peak_fit_poles(baseband, sr_dec, n_modes)
        else:
            poles = self._subspace_poles(baseband, n_modes)
        if poles.size == 0:
            return []

        amplitudes, residual = self._fit_amplitudes(baseband, poles)
        return self._to_estimates(
            poles, amplitudes, residual, sr_dec, f_center, f_low, f_high,
            sos, decim_sos, offset, segment.size, settle
        )

    # -- preparation ----------------------------------------------------------

    def _segment(self, filtered: np.ndarray, settle: float = 0.0) -> tuple[np.ndarray, int]:
        """The stretch of a filtered band worth fitting: peak to floor, capped.

        Fitting past the point where the band has fallen into the noise measures
        the recording. Fitting from before the band's own peak measures the
        filter's transient.
        """
        magnitude = np.abs(filtered)
        if not np.any(magnitude > 0):
            return np.zeros(0), 0

        peak_index = int(np.argmax(magnitude))
        peak_db = Decibels.from_amplitude(magnitude[peak_index])
        start = peak_index + int(round(settle * ModalAnalyzer.FILTER_SETTLE_FRACTION * self.sr))
        if start >= len(filtered) - 64:
            start = peak_index

        # Smooth before thresholding, or a single zero crossing ends the segment.
        smoothed = np.convolve(
            magnitude, np.ones(max(4, self.sr // 1000)) / max(4, self.sr // 1000), "same"
        )
        floor = Decibels.to_amplitude(peak_db - ModalAnalyzer.SEGMENT_RANGE_DB)
        tail = smoothed[peak_index:]
        below = np.flatnonzero(tail < floor)
        end = peak_index + (int(below[0]) if below.size else len(tail))
        end = min(end, start + int(ModalAnalyzer.MAX_SEGMENT_SECONDS * self.sr))
        if end <= start + 64:
            return np.zeros(0), start
        return filtered[start:end], start

    def _to_baseband(
        self, segment: np.ndarray, f_low: float, f_high: float
    ) -> tuple[np.ndarray, float, float, np.ndarray | None]:
        """Analytic signal, heterodyned to DC and decimated.

        Returns (complex baseband, decimated rate, center frequency, the
        anti-alias filter used) — the filter comes back so its gain can be
        divided out of the amplitudes later.
        """
        analytic = _signal.hilbert(segment)
        f_center = 0.5 * (f_low + f_high)
        bandwidth = f_high - f_low

        t = np.arange(len(segment), dtype=np.float64) / self.sr
        shifted = analytic * np.exp(-2j * np.pi * f_center * t)

        decimation = max(1, int(self.sr // (2.5 * bandwidth)))
        decim_sos = None
        if decimation > 1:
            decim_sos = FilterDesign.lowpass(0.5 * bandwidth * 1.2, self.sr, order=6)
            shifted = _signal.sosfilt(decim_sos, shifted)
            shifted = shifted[::decimation]

        sr_dec = self.sr / decimation
        if len(shifted) > ModalAnalyzer.MAX_DECIMATED_SAMPLES:
            shifted = shifted[: ModalAnalyzer.MAX_DECIMATED_SAMPLES]
        return shifted, sr_dec, f_center, decim_sos

    # -- the decomposition ----------------------------------------------------

    def _subspace_poles(self, z: np.ndarray, n_modes: int) -> np.ndarray:
        """ESPRIT (or matrix pencil) poles of a complex exponential sum.

        Both methods build the same Hankel matrix and exploit the same rotational
        invariance; they differ in whether the shift is taken on the left or the
        right singular subspace. On well-conditioned data they agree closely.
        """
        n = len(z)
        pencil = int(min(ModalAnalyzer.MAX_PENCIL, max(4, n * ModalAnalyzer.PENCIL_FRACTION)))
        rows = n - pencil + 1
        if rows < 4 or pencil < 4:
            return np.zeros(0, dtype=complex)

        hankel = _linalg.hankel(z[:rows], z[rows - 1 :])
        try:
            u, singular, vh = np.linalg.svd(hankel, full_matrices=False)
        except np.linalg.LinAlgError:  # pragma: no cover - ill-conditioned input
            return np.zeros(0, dtype=complex)

        order = self._model_order(singular, n_modes)
        if order < 1:
            return np.zeros(0, dtype=complex)

        if self.method == "matrix_pencil":
            basis = vh[:order].conj().T
        else:
            basis = u[:, :order]

        upper, lower = basis[:-1], basis[1:]
        try:
            psi = np.linalg.pinv(upper) @ lower
            return np.linalg.eigvals(psi)
        except np.linalg.LinAlgError:  # pragma: no cover
            return np.zeros(0, dtype=complex)

    def _model_order(self, singular: np.ndarray, n_modes: int) -> int:
        """How many singular values are signal rather than noise.

        Asking for more modes than the band holds is the normal case, and an
        over-specified order produces poles that `prune` then discards — but
        truncating here is cheaper and keeps the spurious poles from stealing
        amplitude from the real ones in the least-squares fit.
        """
        if singular.size == 0 or singular[0] <= 0:
            return 0
        significant = int(np.sum(singular / singular[0] > ModalAnalyzer.ORDER_THRESHOLD))
        return int(np.clip(significant, 1, min(n_modes, len(singular) - 1)))

    def _peak_fit_poles(self, z: np.ndarray, sr_dec: float, n_modes: int) -> np.ndarray:
        """Cheap fallback: FFT peaks for frequency, envelope slope for damping.

        Keeps `ModalAnalyzer` usable when the subspace path is misbehaving, and
        makes it obvious when a result depends on the subspace method rather
        than on the data.
        """
        n_fft = int(2 ** np.ceil(np.log2(max(64, len(z) * 4))))
        spectrum = np.abs(np.fft.fft(z, n_fft))
        freqs = np.fft.fftfreq(n_fft, 1.0 / sr_dec)

        from ..core.dsp import SpectralPeak

        order = np.argsort(-spectrum)
        picked, poles = [], []
        min_separation = max(1, n_fft // 200)
        for index in order:
            if len(picked) >= n_modes:
                break
            if any(abs(int(index) - other) < min_separation for other in picked):
                continue
            picked.append(int(index))
            refined, _ = SpectralPeak.refine(spectrum, int(index))
            frequency = float(np.interp(refined, np.arange(n_fft), np.fft.fftshift(freqs)
                                        if False else freqs))
            # Damping from the overall envelope: crude, and deliberately so.
            envelope = np.abs(z)
            slope = np.polyfit(np.arange(len(envelope)) / sr_dec,
                               np.log(np.maximum(envelope, 1e-30)), 1)[0]
            poles.append(np.exp((slope + 2j * np.pi * frequency) / sr_dec))
        return np.asarray(poles, dtype=complex)

    @staticmethod
    def _fit_amplitudes(z: np.ndarray, poles: np.ndarray) -> tuple[np.ndarray, float]:
        """Least-squares complex amplitudes, plus the relative residual.

        The residual is the honest confidence signal: a band modelled by the
        wrong number of modes leaves structure behind, and that shows up here
        before it shows up anywhere else.
        """
        n = len(z)
        exponents = np.arange(n, dtype=np.float64)[:, None]
        with np.errstate(over="ignore", invalid="ignore"):
            vandermonde = poles[None, :] ** exponents
        if not np.all(np.isfinite(vandermonde)):
            finite = np.all(np.isfinite(vandermonde), axis=0)
            vandermonde = vandermonde[:, finite]
            poles = poles[finite]
            if poles.size == 0:
                return np.zeros(0, dtype=complex), 1.0

        amplitudes, *_ = np.linalg.lstsq(vandermonde, z, rcond=None)
        reconstruction = vandermonde @ amplitudes
        energy = float(np.sum(np.abs(z) ** 2))
        residual = (
            float(np.sum(np.abs(z - reconstruction) ** 2) / energy) if energy > 0 else 1.0
        )
        return amplitudes, residual

    # -- interpretation -------------------------------------------------------

    def _to_estimates(
        self,
        poles: np.ndarray,
        amplitudes: np.ndarray,
        residual: float,
        sr_dec: float,
        f_center: float,
        f_low: float,
        f_high: float,
        band_sos: np.ndarray,
        decim_sos: np.ndarray | None,
        offset: int,
        segment_length: int,
        settle: float,
    ) -> list[ModeEstimate]:
        if poles.size == 0 or amplitudes.size == 0:
            return []

        magnitude = np.abs(poles)
        with np.errstate(divide="ignore"):
            damping = -np.log(np.maximum(magnitude, 1e-300)) * sr_dec
        frequencies = f_center + np.angle(poles) * sr_dec / (2.0 * np.pi)

        confidence = float(np.clip(1.0 - np.sqrt(max(residual, 0.0)), 0.0, 1.0))
        window = (offset / self.sr, (offset + segment_length) / self.sr)

        # Undo the gain the analysis filters applied at each estimated frequency,
        # otherwise every mode near a band edge is reported quieter than it is.
        band_gain = FilterDesign.response_at(band_sos, np.abs(frequencies), self.sr)
        gain = np.maximum(band_gain, 1e-3)
        if decim_sos is not None:
            baseband_gain = FilterDesign.response_at(
                decim_sos, np.abs(frequencies - f_center), self.sr
            )
            gain = gain * np.maximum(baseband_gain, 1e-3)

        # The segment starts after the filter settled, so every amplitude is
        # measured `offset` seconds late and has to be projected back to t=0.
        offset_seconds = offset / self.sr
        boost = np.minimum(
            np.exp(np.maximum(damping, 0.0) * offset_seconds),
            Decibels.to_amplitude(ModalAnalyzer.MAX_EXTRAPOLATION_DB),
        )

        estimates = []
        for index, pole in enumerate(poles):
            frequency = float(frequencies[index])
            if not (f_low <= frequency <= f_high):
                continue
            t60 = float(Decay.damping_to_t60(float(damping[index])))
            # A pole decaying faster than the analysis filter's own ringdown
            # cannot be distinguished from that ringdown.
            if t60 < settle * ModalAnalyzer.SETTLE_MARGIN:
                continue
            estimates.append(
                ModeEstimate(
                    freq=frequency,
                    amplitude=float(np.abs(amplitudes[index]) * boost[index] / gain[index]),
                    t60=t60,
                    confidence=confidence,
                    t_window=window,
                    phase=float(np.angle(amplitudes[index])),
                )
            )
        return estimates

    def prune(
        self, modes: list[ModeEstimate], min_amplitude_db: float = -60.0
    ) -> list[ModeEstimate]:
        """Drop numerical artifacts: negative damping, out-of-band, inaudible.

        Also merges near-duplicates, which appear where two analysis bands
        overlap at their skirts and both recover the same partial.
        """
        if not modes:
            return []

        alive = [
            mode
            for mode in modes
            if np.isfinite(mode.freq)
            and self.f_range[0] <= mode.freq <= self.f_range[1]
            and np.isfinite(mode.t60)
            and 0.001 < mode.t60 < 60.0
            and np.isfinite(mode.amplitude)
            and mode.amplitude > 0.0
        ]
        if not alive:
            return []

        loudest = max(mode.amplitude for mode in alive)
        threshold = loudest * Decibels.to_amplitude(min_amplitude_db)
        alive = [mode for mode in alive if mode.amplitude >= threshold]
        return self._merge_duplicates(alive)

    @staticmethod
    def _merge_duplicates(
        modes: list[ModeEstimate], tolerance_cents: float = 12.0
    ) -> list[ModeEstimate]:
        """Keep the loudest of any group of modes within `tolerance_cents`."""
        ordered = sorted(modes, key=lambda mode: mode.freq)
        kept: list[ModeEstimate] = []
        for mode in ordered:
            if kept and abs(1200.0 * np.log2(mode.freq / kept[-1].freq)) < tolerance_cents:
                if mode.amplitude > kept[-1].amplitude:
                    kept[-1] = mode
            else:
                kept.append(mode)
        return kept
