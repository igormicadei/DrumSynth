"""Load, align and normalize before anything else.

Onset alignment and gain normalization have to happen first, or every
downstream comparison inherits a constant offset — a 3 ms misalignment makes
the attack score meaningless and a 2 dB level difference makes every band level
wrong by the same 2 dB.

Alignment is on the envelope onset, not the file start: a reference WAV
normally has digital silence padded in front of it, and a rendered hit does not.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from ..core.audio_io import AudioIO
from ..core.constants import Audio, Decibels
from ..core.dsp import EnvelopeFollower


class SignalPrep:
    """Everything that happens to a signal before it is analysed."""

    #: Level, relative to peak, at which the signal is considered to have started.
    DEFAULT_ONSET_DB: float = -40.0

    #: Frame grid used for onset and noise-floor detection.
    FRAME_SECONDS: float = 0.001

    #: Quantile of frame levels taken as the noise floor.
    FLOOR_QUANTILE: float = 0.05

    def __init__(self, sr: int = Audio.DEFAULT_SR) -> None:
        self.sr = int(sr)
        self._follower = EnvelopeFollower(self.sr, SignalPrep.FRAME_SECONDS,
                                          SignalPrep.FRAME_SECONDS * 4)

    # -- loading --------------------------------------------------------------

    def load(self, path: str | Path) -> np.ndarray:
        """Load WAV, sum to mono, resample to self.sr."""
        signal, _ = AudioIO.read(path, sr=self.sr)
        return signal

    # -- alignment ------------------------------------------------------------

    def find_onset(self, x: np.ndarray, threshold_db: float = DEFAULT_ONSET_DB) -> int:
        """Index of the first sample at which the signal crosses the threshold.

        Backtracks from the first frame above threshold to the last sample below
        it, so a hit that rises in 0.1 ms is not reported 1 ms late.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return 0

        peak = float(np.max(np.abs(x)))
        if peak <= 0.0:
            return 0

        times, rms = self._follower.rms(x)
        threshold = peak * (10.0 ** (threshold_db / 20.0))
        above = np.flatnonzero(rms >= threshold)
        if above.size == 0:
            return int(np.argmax(np.abs(x)))

        frame_start = max(0, int(times[above[0]] * self.sr) - self._follower.window)
        window = np.abs(x[frame_start : frame_start + 2 * self._follower.window])
        crossings = np.flatnonzero(window >= threshold)
        return int(frame_start + crossings[0]) if crossings.size else frame_start

    def trim_to_onset(
        self, x: np.ndarray, pre_roll: float = 0.0,
        threshold_db: float = DEFAULT_ONSET_DB
    ) -> np.ndarray:
        """Drop everything before the onset, keeping `pre_roll` seconds of it."""
        onset = self.find_onset(x, threshold_db)
        start = max(0, onset - int(round(pre_roll * self.sr)))
        return np.asarray(x, dtype=np.float64)[start:].copy()

    # -- level ----------------------------------------------------------------

    def normalize(
        self, x: np.ndarray, mode: Literal["peak", "rms", "none"] = "rms"
    ) -> np.ndarray:
        """RMS by default.

        Peak normalization is hostage to a single transient sample: one clipped
        or one unusually tall cycle moves the whole signal's level, and every
        band level downstream moves with it.
        """
        x = np.asarray(x, dtype=np.float64)
        if mode == "none" or x.size == 0:
            return x.copy()
        if mode == "peak":
            reference = float(np.max(np.abs(x)))
        elif mode == "rms":
            reference = float(np.sqrt(np.mean(x**2)))
        else:
            raise ValueError(f"unknown normalization mode {mode!r}")
        return x / reference if reference > 0 else x.copy()

    def estimate_noise_floor(self, x: np.ndarray) -> float:
        """dB level, relative to peak, at which the recording stops being signal.

        Critical for the reference: a lossy-encoded file flattens out at the
        codec floor, and fitting decay slopes into that region produces
        nonsense. Everything below this level must be excluded from scoring.

        Taken as a low quantile of the frame levels rather than the minimum —
        the minimum is one unlucky frame, and a decaying signal spends most of
        its length near the floor, so the low quantile lands squarely on it.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return -np.inf

        peak = float(np.max(np.abs(x)))
        if peak <= 0.0:
            return -np.inf

        _, rms = self._follower.rms(x)
        rms = rms[rms > 0.0]
        if rms.size == 0:
            return -np.inf

        floor = float(np.quantile(rms, SignalPrep.FLOOR_QUANTILE))
        return float(Decibels.from_amplitude(floor / peak))

    def usable_duration(
        self, x: np.ndarray, floor_db: float | None = None, margin_db: float = 6.0
    ) -> float:
        """Seconds from onset until the envelope reaches the noise floor.

        `margin_db` keeps the measurement above the floor rather than in it. A
        decay slope fitted through the last 6 dB before the floor is measuring
        the transition, not the drum.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return 0.0
        floor_db = self.estimate_noise_floor(x) if floor_db is None else floor_db
        if not np.isfinite(floor_db):
            return len(x) / self.sr

        times, rms = self._follower.rms(x)
        peak = float(np.max(rms)) if rms.size else 0.0
        if peak <= 0.0:
            return 0.0

        db = Decibels.from_amplitude(rms / peak)
        above = np.flatnonzero(db > floor_db + margin_db)
        return float(times[above[-1]]) if above.size else 0.0

    # -- the whole chain ------------------------------------------------------

    def prepare(
        self,
        x_or_path: np.ndarray | str | Path,
        normalization: Literal["peak", "rms", "none"] = "rms",
        pre_roll: float = 0.0,
    ) -> np.ndarray:
        """load -> trim_to_onset -> normalize.

        Both signals must go through this, with the same arguments, or the
        comparison is measuring the preparation.
        """
        signal = (
            self.load(x_or_path)
            if isinstance(x_or_path, (str, Path))
            else np.asarray(x_or_path, dtype=np.float64)
        )
        return self.normalize(self.trim_to_onset(signal, pre_roll), normalization)
