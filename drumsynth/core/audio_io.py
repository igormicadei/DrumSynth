"""Reading and writing WAV files, and the resampling that goes with it.

One class, static methods, so that every entry point into the project reads
audio the same way: mono float64 at a known sample rate. `soundfile` is used
when installed (it handles 24-bit, float and compressed WAV variants); the
stdlib `wave` module is the fallback so the package still works without it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .constants import Audio

try:  # pragma: no cover - exercised implicitly by whichever backend is present
    import soundfile as _soundfile
except ImportError:  # pragma: no cover
    _soundfile = None


class AudioIO:
    """Load, resample and write mono audio."""

    #: Written files are normalized to this peak level, leaving a little headroom.
    WRITE_PEAK_DBFS: float = -1.0

    # -- reading --------------------------------------------------------------

    @staticmethod
    def read(path: str | Path, sr: int | None = None) -> tuple[np.ndarray, int]:
        """Read a WAV file as mono float64.

        Returns (signal, sample_rate). If `sr` is given the signal is resampled
        to it, otherwise the file's native rate is reported back unchanged.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"audio file not found: {path}")

        if _soundfile is not None:
            data, file_sr = _soundfile.read(str(path), dtype="float64", always_2d=True)
            signal = AudioIO.to_mono(data)
        else:
            signal, file_sr = AudioIO._read_with_wave_module(path)

        if sr is not None and sr != file_sr:
            signal = AudioIO.resample(signal, file_sr, sr)
            file_sr = sr
        return signal, int(file_sr)

    @staticmethod
    def _read_with_wave_module(path: Path) -> tuple[np.ndarray, int]:
        """PCM-only fallback for when soundfile is unavailable."""
        import wave

        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            file_sr = handle.getframerate()
            raw = handle.readframes(handle.getnframes())

        if width == 1:  # unsigned 8-bit
            data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
        elif width == 2:
            data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
        elif width == 3:
            packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
            ints = packed[:, 0] | (packed[:, 1] << 8) | (packed[:, 2] << 16)
            ints = np.where(ints >= 1 << 23, ints - (1 << 24), ints)
            data = ints.astype(np.float64) / float(1 << 23)
        elif width == 4:
            data = np.frombuffer(raw, dtype="<i4").astype(np.float64) / float(1 << 31)
        else:
            raise ValueError(f"unsupported sample width {width * 8} bit in {path}")

        return AudioIO.to_mono(data.reshape(-1, channels)), file_sr

    # -- conversion -----------------------------------------------------------

    @staticmethod
    def to_mono(data: np.ndarray) -> np.ndarray:
        """Average channels to mono. Averaging, not summing — summing can clip."""
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim == 1:
            return arr
        if arr.ndim != 2:
            raise ValueError(f"expected 1-D or 2-D audio, got shape {arr.shape}")
        return arr.mean(axis=1)

    @staticmethod
    def resample(signal: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
        """Polyphase resample. Falls back to linear interpolation without scipy."""
        if sr_from == sr_to:
            return np.asarray(signal, dtype=np.float64)

        from math import gcd

        divisor = gcd(int(sr_from), int(sr_to))
        up, down = int(sr_to) // divisor, int(sr_from) // divisor
        try:
            from scipy.signal import resample_poly

            return np.asarray(resample_poly(signal, up, down), dtype=np.float64)
        except ImportError:  # pragma: no cover
            n_out = int(round(len(signal) * sr_to / sr_from))
            src = np.arange(len(signal), dtype=np.float64)
            dst = np.linspace(0.0, len(signal) - 1.0, n_out)
            return np.interp(dst, src, np.asarray(signal, dtype=np.float64))

    # -- writing --------------------------------------------------------------

    @staticmethod
    def write(
        path: str | Path,
        signal: np.ndarray,
        sr: int = Audio.DEFAULT_SR,
        normalize: bool = True,
        subtype: str = "PCM_24",
    ) -> Path:
        """Write mono audio, normalized to WRITE_PEAK_DBFS by default.

        Normalization is on by default because rendered hits carry the synth's
        absolute gain, which is rarely near full scale. Pass normalize=False
        when the absolute level is the thing you are inspecting.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        out = np.asarray(signal, dtype=np.float64).ravel()

        if normalize:
            out = AudioIO.normalize_peak(out, AudioIO.WRITE_PEAK_DBFS)
        out = np.clip(out, -1.0, 1.0)

        if _soundfile is not None:
            _soundfile.write(str(path), out, int(sr), subtype=subtype)
        else:
            AudioIO._write_with_wave_module(path, out, int(sr))
        return path

    @staticmethod
    def _write_with_wave_module(path: Path, signal: np.ndarray, sr: int) -> None:
        """16-bit PCM fallback."""
        import wave

        pcm = np.round(signal * 32767.0).astype("<i2")
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sr)
            handle.writeframes(pcm.tobytes())

    @staticmethod
    def normalize_peak(signal: np.ndarray, peak_dbfs: float = -1.0) -> np.ndarray:
        """Scale so the largest absolute sample sits at `peak_dbfs`."""
        out = np.asarray(signal, dtype=np.float64)
        peak = float(np.max(np.abs(out))) if out.size else 0.0
        if peak <= 0.0:
            return out.copy()
        target = 10.0 ** (peak_dbfs / 20.0)
        return out * (target / peak)
