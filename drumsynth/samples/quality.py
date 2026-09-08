"""Everything that can quietly invalidate a fit.

Nothing here is fatal by itself — `usable_duration` is the point. Fit only
within it and a truncated or noisy sample still contributes what it has.

The one that costs real time if missed is tail truncation. A WAV cut before the
resonance finished decaying fits a shorter t60 than the real one, and nothing
about the result looks wrong: the fit converges, the residual is small, the
number is simply too small. It is checked here, not assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from ..core.constants import Audio, Decibels
from ..core.dsp import EnvelopeFollower, LinearRegression


@dataclass
class SampleQuality:
    """Computed once on load. Cheap; run it on everything."""

    noise_floor_db: float = -np.inf  # where signal stops and the floor begins
    peak_db: float = 0.0
    is_clipped: bool = False  # samples at or near full scale
    clipped_count: int = 0
    dc_offset: float = 0.0

    tail_truncated: bool = False  # file ends while the signal is still falling
    end_level_db: float = -np.inf  # level at the last sample, relative to peak
    tail_slope_db_s: float = 0.0  # decay rate over the final stretch
    usable_duration: float = 0.0  # s from onset until the floor is reached

    has_pre_onset_silence: bool = False  # digital zeros => edited, not raw capture
    multiple_onsets: bool = False  # a second hit in the file invalidates decay fits
    onset_count: int = 1

    snr_db: float = 0.0
    duration: float = 0.0

    #: Truncation is judged by the SLOPE at the end of the file, not by the
    #: end level against a noise floor.
    #:
    #: The floor is estimated as a low quantile of the frame levels, and in a
    #: truncated file the quietest frames ARE the cut-off tail — so the floor
    #: estimate follows the truncation down and the file looks like it reached
    #: it. A recording that genuinely decayed into its noise floor has a flat
    #: tail; one that was cut is still descending at close to the drum's own
    #: decay rate. The slope tells them apart and the level cannot.
    TRUNCATION_SLOPE_DB_S: ClassVar[float] = -6.0

    #: Fraction of the file used to measure that final slope.
    TAIL_FRACTION: ClassVar[float] = 0.20
    MIN_TAIL_SECONDS: ClassVar[float] = 0.30

    #: Samples within this of full scale count as clipped.
    CLIP_THRESHOLD: ClassVar[float] = 0.999

    #: A DC offset above this fraction of full scale skews every energy measure.
    DC_LIMIT: ClassVar[float] = 0.01

    @property
    def is_usable(self) -> bool:
        """False only for the disqualifying cases: clipping, or more than one onset.

        Truncation and a poor noise floor are NOT disqualifying — they bound
        `usable_duration`, and a fit restricted to that window is still valid
        evidence. Clipping and a second hit are different: both corrupt the
        signal inside the window, and no restriction recovers them.
        """
        return not self.is_clipped and not self.multiple_onsets

    def warnings(self) -> list[str]:
        issues: list[str] = []
        if self.is_clipped:
            issues.append(
                f"clipped: {self.clipped_count} samples at full scale — the peak "
                "is flattened and every excitation gain fitted from it is wrong"
            )
        if self.multiple_onsets:
            issues.append(
                f"{self.onset_count} onsets in one file — a second hit restarts the "
                "decay and invalidates every t60 measured across it"
            )
        if self.tail_truncated:
            issues.append(
                f"tail truncated: the file ends at {self.end_level_db:.1f} dB still "
                f"falling at {self.tail_slope_db_s:.0f} dB/s; t60 will fit short "
                "and nothing about the fit will look wrong"
            )
        if self.snr_db < 30.0:
            issues.append(
                f"low SNR ({self.snr_db:.1f} dB): usable duration is only "
                f"{self.usable_duration:.2f}s"
            )
        if abs(self.dc_offset) > SampleQuality.DC_LIMIT:
            issues.append(f"DC offset {self.dc_offset:+.4f} — high-pass before fitting")
        if not self.has_pre_onset_silence:
            issues.append(
                "no digital silence before the onset; the hit may start before "
                "the file does"
            )
        return issues

    def summary(self) -> str:
        state = "usable" if self.is_usable else "UNUSABLE"
        return (
            f"{state}: peak {self.peak_db:.1f} dB, floor {self.noise_floor_db:.1f} dB, "
            f"SNR {self.snr_db:.1f} dB, usable {self.usable_duration:.2f}s of "
            f"{self.duration:.2f}s"
        )


class QualityChecker:
    """Measures a `SampleQuality` from raw audio.

    Separate from the dataclass so the measurement policy — what counts as an
    onset, how the floor is estimated — lives in one place and can be tuned
    without touching the record it produces.
    """

    #: Level, relative to peak, at which a hit is considered to have started.
    ONSET_DB: float = -40.0

    #: An onset is a FAST RISE, not a level crossing. A drum carrying a close
    #: mode pair beats by 6-8 dB well into its decay, and any threshold low
    #: enough to catch a ghost note is crossed by that beating several times per
    #: second — level-crossing detection reports every tom in the library as a
    #: double hit.
    #:
    #: Rise required, in dB, over RISE_WINDOW seconds.
    RISE_DB: float = 12.0
    RISE_WINDOW: float = 0.020

    #: An onset must also reach this level, relative to the file's peak.
    ONSET_LEVEL_DB: float = -35.0

    #: Minimum gap between two onsets. Closer than this is one hit's own attack.
    MIN_ONSET_GAP: float = 0.050

    def __init__(self, sr: int = Audio.DEFAULT_SR) -> None:
        self.sr = int(sr)
        self._follower = EnvelopeFollower(self.sr, 0.002, 0.008)

    def check(self, x: np.ndarray) -> SampleQuality:
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return SampleQuality()

        peak = float(np.max(np.abs(x)))
        clipped = int(np.sum(np.abs(x) >= SampleQuality.CLIP_THRESHOLD))
        times, rms = self._follower.rms(x)
        peak_rms = float(np.max(rms)) if rms.size else 0.0

        floor_db = self._noise_floor_db(rms, peak_rms)
        onsets = self.find_onsets(rms, times, peak_rms)
        usable = self._usable_duration(times, rms, peak_rms, floor_db)
        end_db = float(Decibels.from_amplitude(rms[-1] / peak_rms)) if peak_rms > 0 else -np.inf
        tail_slope = self._tail_slope(times, rms, peak_rms)

        return SampleQuality(
            noise_floor_db=floor_db,
            peak_db=float(Decibels.from_amplitude(peak)),
            is_clipped=clipped > 2,
            clipped_count=clipped,
            dc_offset=float(np.mean(x)),
            tail_truncated=tail_slope < SampleQuality.TRUNCATION_SLOPE_DB_S,
            end_level_db=end_db,
            tail_slope_db_s=tail_slope,
            usable_duration=usable,
            has_pre_onset_silence=self._has_pre_onset_silence(x, onsets),
            multiple_onsets=len(onsets) > 1,
            onset_count=max(1, len(onsets)),
            snr_db=float(-floor_db) if np.isfinite(floor_db) else 0.0,
            duration=len(x) / self.sr,
        )

    # -- internals ------------------------------------------------------------

    @staticmethod
    def _noise_floor_db(rms: np.ndarray, peak_rms: float) -> float:
        alive = rms[rms > 0.0]
        if alive.size == 0 or peak_rms <= 0.0:
            return -np.inf
        return float(Decibels.from_amplitude(np.quantile(alive, 0.05) / peak_rms))

    def find_onsets(
        self, rms: np.ndarray, times: np.ndarray, peak_rms: float
    ) -> list[float]:
        """Onset times in seconds.

        A frame is an onset when the envelope climbs RISE_DB above its local
        minimum over the preceding RISE_WINDOW, and lands above ONSET_LEVEL_DB.
        A real strike rises 20 dB or more in a few milliseconds; beating between
        close modes rises a few dB over tens of milliseconds and is rejected.
        """
        if peak_rms <= 0.0 or rms.size == 0:
            return []

        db = Decibels.from_amplitude(rms / peak_rms)
        lookback = max(1, int(round(QualityChecker.RISE_WINDOW / max(
            times[1] - times[0] if times.size > 1 else 1.0, 1e-9))))

        onsets: list[float] = []
        # A file that starts already loud has its first onset at or before
        # sample zero — there is no rise to detect because the lead-in was
        # trimmed off. Without this, a file with no pre-onset silence loses its
        # first hit and a double hit reads as a single one.
        if db[0] > QualityChecker.ONSET_LEVEL_DB:
            onsets.append(float(times[0]))

        for index in range(len(db)):
            if db[index] < QualityChecker.ONSET_LEVEL_DB:
                continue
            window_start = max(0, index - lookback)
            local_min = float(np.min(db[window_start : index + 1]))
            if db[index] - local_min < QualityChecker.RISE_DB:
                continue
            if onsets and times[index] - onsets[-1] < QualityChecker.MIN_ONSET_GAP:
                continue
            onsets.append(float(times[index]))
        return onsets

    def _tail_slope(
        self, times: np.ndarray, rms: np.ndarray, peak_rms: float
    ) -> float:
        """dB/s over the final stretch of the file.

        Near zero means the signal reached whatever floor the recording has.
        Steeply negative means it was still decaying when the file ended.
        """
        if peak_rms <= 0.0 or times.size < 8:
            return 0.0
        duration = float(times[-1])
        span = max(SampleQuality.TAIL_FRACTION * duration,
                   SampleQuality.MIN_TAIL_SECONDS)
        start = np.searchsorted(times, max(0.0, duration - span))
        if times.size - start < 5:
            start = max(0, times.size - 5)

        db = Decibels.from_amplitude(rms[start:] / peak_rms)
        fit = LinearRegression.fit(times[start:], db)
        return float(fit.slope) if fit.is_valid else 0.0

    def _usable_duration(
        self, times: np.ndarray, rms: np.ndarray, peak_rms: float, floor_db: float
    ) -> float:
        if peak_rms <= 0.0 or times.size == 0:
            return 0.0
        db = Decibels.from_amplitude(rms / peak_rms)
        limit = floor_db + 6.0 if np.isfinite(floor_db) else -np.inf
        above = np.flatnonzero(db > limit)
        return float(times[above[-1]]) if above.size else 0.0

    def _has_pre_onset_silence(self, x: np.ndarray, onsets: list[float]) -> bool:
        """True when the file starts with digital zeros — an edited file, which
        means the onset is where the editor put it and not where the hit was."""
        if not onsets:
            return False
        lead = x[: max(1, int(onsets[0] * self.sr))]
        return bool(lead.size > 8 and np.all(lead == 0.0))
