"""Where the rendered audio goes.

Three of them, and the point of the split is that only one needs a sound card.
`NullSink` runs the entire engine — command handling, live parameter updates,
telemetry — with no audio hardware at all, which is what makes the live path
testable in CI and on a machine that has no output device.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

Renderer = Callable[[int], np.ndarray]


class AudioSink:
    """Pulls blocks from a renderer until stopped."""

    name = "base"

    def __init__(self, sr: int, block_size: int) -> None:
        self.sr = int(sr)
        self.block_size = int(block_size)
        self.underruns = 0

    @property
    def description(self) -> str:
        return self.name

    def start(self, render: Renderer) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    @property
    def is_running(self) -> bool:
        raise NotImplementedError


class NullSink(AudioSink):
    """Renders on a timer and discards the result.

    Paced to wall clock by default so the engine behaves as it would with a
    device — telemetry arrives at the same rate, strikes land at plausible
    times, and a parameter change is heard (or rather, measured) after the same
    delay.
    """

    name = "null"

    def __init__(self, sr: int, block_size: int, realtime: bool = True) -> None:
        super().__init__(sr, block_size)
        self.realtime = realtime
        self._running = False
        self._thread = None

    @property
    def description(self) -> str:
        return f"no output ({self.sr} Hz, {self.block_size} frames)"

    def start(self, render: Renderer) -> None:
        import threading

        self._running = True
        period = self.block_size / self.sr

        def loop() -> None:
            deadline = time.perf_counter()
            while self._running:
                render(self.block_size)
                if not self.realtime:
                    continue
                deadline += period
                delay = deadline - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    self.underruns += 1
                    deadline = time.perf_counter()

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    @property
    def is_running(self) -> bool:
        return self._running


class WavSink(NullSink):
    """Renders on the same timer and accumulates into a buffer.

    For rendering a take without a device, and for tests that need to look at
    what the engine actually produced rather than trust that it produced
    something.
    """

    name = "wav"

    def __init__(
        self, sr: int, block_size: int, path: str, realtime: bool = False,
        max_seconds: float = 120.0,
    ) -> None:
        super().__init__(sr, block_size, realtime)
        self.path = path
        self.max_samples = int(max_seconds * sr)
        self._chunks: list[np.ndarray] = []
        self._length = 0

    def start(self, render: Renderer) -> None:
        def capture(frames: int) -> np.ndarray:
            block = render(frames)
            if self._length < self.max_samples:
                self._chunks.append(block.copy())
                self._length += len(block)
            return block

        super().start(capture)

    @property
    def description(self) -> str:
        return f"writing {self.path}"

    def audio(self) -> np.ndarray:
        return (
            np.concatenate(self._chunks) if self._chunks else np.zeros(0, dtype=np.float64)
        )

    def write(self) -> str:
        from ..core.audio_io import AudioIO

        AudioIO.write(self.path, self.audio(), self.sr, normalize=False)
        return self.path


class DeviceSink(AudioSink):
    """Real output through PortAudio.

    The renderer runs inside the audio callback, so it must not block, allocate
    unboundedly, or take a lock the UI thread might hold. The engine satisfies
    that by draining a deque of commands at the top of each block and doing all
    its mutation there — the audio thread owns the synthesizer outright.
    """

    name = "device"

    def __init__(
        self, sr: int, block_size: int, device: str | int | None = None
    ) -> None:
        super().__init__(sr, block_size)
        self.device = device
        self._stream = None

    @staticmethod
    def available() -> bool:
        try:
            import sounddevice  # noqa: F401
        except Exception:
            return False
        return True

    @staticmethod
    def devices() -> list[dict]:
        """Output devices, or an empty list when PortAudio is unavailable."""
        try:
            import sounddevice as sd

            return [
                {"index": index, "name": item["name"],
                 "channels": item["max_output_channels"],
                 "default_sr": item["default_samplerate"]}
                for index, item in enumerate(sd.query_devices())
                if item["max_output_channels"] > 0
            ]
        except Exception:
            return []

    @property
    def description(self) -> str:
        try:
            import sounddevice as sd

            info = sd.query_devices(self.device, "output")
            return f"{info['name']} ({self.sr} Hz, {self.block_size} frames)"
        except Exception:
            return f"device {self.device!r}"

    def start(self, render: Renderer) -> None:
        import sounddevice as sd

        def callback(outdata, frames, time_info, status) -> None:
            if status:
                self.underruns += 1
            outdata[:, 0] = render(frames)

        self._stream = sd.OutputStream(
            samplerate=self.sr,
            blocksize=self.block_size,
            channels=1,
            dtype="float32",
            device=self.device,
            callback=callback,
            latency="low",
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    @property
    def is_running(self) -> bool:
        return self._stream is not None and self._stream.active


class SinkFactory:
    """Builds a sink from an `EngineSettings.sink` string."""

    @staticmethod
    def build(settings, path: str | None = None) -> AudioSink:
        if settings.sink == "device":
            return DeviceSink(settings.sr, settings.block_size, settings.device)
        if settings.sink == "wav":
            return WavSink(settings.sr, settings.block_size, path or "live.wav")
        if settings.sink == "null":
            return NullSink(
                settings.sr, settings.block_size,
                realtime=bool(settings.extra.get("realtime", True)),
            )
        raise ValueError(f"unknown sink {settings.sink!r}; expected device, null or wav")
