"""Driving the audio process from another process.

`LiveSynth` owns the subprocess, writes commands to its stdin, and keeps a
background reader draining its stdout so the pipe never fills and blocks the
audio thread. The UI never waits on audio and audio never waits on the UI.

    synth = LiveSynth().start()
    synth.load(DrumPresets.tom())
    synth.strike(0.8)
    print(synth.status.peak)
    synth.stop()
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

from ..synth.params import DrumParams
from .protocol import Command, EngineSettings, Event, Status


class LiveSynth:
    """A handle on a running `drumsynth.live.engine` process."""

    #: Events kept for display. The engine talks continuously; only the recent
    #: past is interesting and an unbounded log is a slow memory leak.
    LOG_SIZE: int = 200

    START_TIMEOUT: float = 10.0

    def __init__(
        self,
        settings: EngineSettings | None = None,
        python: str | None = None,
        cwd: str | Path | None = None,
    ) -> None:
        self.settings = settings or EngineSettings()
        self.python = python or sys.executable
        self.cwd = str(cwd) if cwd else None

        self.process: subprocess.Popen | None = None
        self.status = Status()
        self.ready: dict | None = None
        self.log: deque[dict] = deque(maxlen=LiveSynth.LOG_SIZE)
        self.last_error: str | None = None
        self.last_recording: dict | None = None

        self._reader: threading.Thread | None = None
        self._stderr_reader: threading.Thread | None = None
        self._stderr: deque[str] = deque(maxlen=50)

    # -- lifecycle ------------------------------------------------------------

    def command_line(self) -> list[str]:
        settings = self.settings
        argv = [
            self.python, "-m", "drumsynth.live.engine",
            "--sr", str(settings.sr),
            "--block-size", str(settings.block_size),
            "--control-period", str(settings.control_period),
            "--sink", settings.sink,
            "--status-interval", str(settings.status_interval),
            "--seed", str(-1 if settings.seed is None else settings.seed),
        ]
        if settings.device is not None:
            argv += ["--device", str(settings.device)]
        if settings.sink == "null" and not settings.extra.get("realtime", True):
            argv.append("--no-realtime")
        return argv

    def start(self, params: DrumParams | None = None) -> "LiveSynth":
        if self.is_running:
            return self

        self.process = subprocess.Popen(
            self.command_line(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=self.cwd,
        )
        self._reader = threading.Thread(target=self._read_events, daemon=True)
        self._reader.start()
        self._stderr_reader = threading.Thread(target=self._read_stderr, daemon=True)
        self._stderr_reader.start()

        if not self._await_ready():
            message = self.last_error or "".join(self._stderr) or "no response"
            self.stop()
            raise RuntimeError(f"audio engine did not start: {message.strip()}")

        if params is not None:
            self.load(params)
        return self

    def _await_ready(self) -> bool:
        deadline = time.monotonic() + LiveSynth.START_TIMEOUT
        while time.monotonic() < deadline:
            if self.ready is not None:
                return True
            if self.process is not None and self.process.poll() is not None:
                return False
            time.sleep(0.02)
        return False

    def stop(self, timeout: float = 3.0) -> None:
        if self.process is None:
            return
        try:
            self._send(Command.QUIT)
            self.process.wait(timeout=timeout)
        except Exception:
            self.process.kill()
        finally:
            for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except Exception:
                    pass
            self.process = None
            self.ready = None

    def restart(self, params: DrumParams | None = None) -> "LiveSynth":
        self.stop()
        return self.start(params)

    @property
    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    # -- commands -------------------------------------------------------------

    def _send(self, name: str, **fields) -> bool:
        if self.process is None or self.process.stdin is None:
            return False
        try:
            self.process.stdin.write(Command.encode(name, **fields) + "\n")
            self.process.stdin.flush()
            return True
        except (BrokenPipeError, ValueError):
            # The engine died. Surface it rather than raising into a UI redraw.
            self.last_error = "audio engine is gone (broken pipe)"
            return False

    def strike(self, amplitude: float = 1.0) -> bool:
        return self._send(Command.STRIKE, amplitude=float(amplitude))

    def load(self, params: DrumParams) -> bool:
        """Push a whole parameter set. The engine applies it without resetting."""
        return self._send(Command.PARAMS, params=params.to_dict())

    def set_output_gain(self, value: float) -> bool:
        return self._send(Command.GAIN, value=float(value))

    def set_monitor(self, mask: list[float] | None) -> bool:
        """Solo/mute individual modes. A listening aid — not a parameter."""
        return self._send(Command.MONITOR, mask=list(mask) if mask else None)

    def reset(self) -> bool:
        return self._send(Command.RESET)

    def panic(self) -> bool:
        return self._send(Command.PANIC)

    def record(self, path: str | Path, seconds: float = 4.0, normalize: bool = False) -> bool:
        return self._send(
            Command.RECORD, path=str(path), seconds=float(seconds), normalize=normalize
        )

    def ping(self) -> bool:
        return self._send(Command.PING)

    # -- telemetry ------------------------------------------------------------

    def _read_events(self) -> None:
        stream = self.process.stdout if self.process else None
        if stream is None:
            return
        for line in stream:
            message = Event.decode(line)
            if message is None:
                continue
            self.log.append(message)
            kind = message.get("ev")
            if kind == Event.STATUS:
                self.status = Status.from_dict(message)
            elif kind == Event.READY:
                self.ready = message
            elif kind == Event.ERROR:
                self.last_error = str(message.get("message", ""))
            elif kind == Event.RECORDED:
                self.last_recording = message

    def _read_stderr(self) -> None:
        stream = self.process.stderr if self.process else None
        if stream is None:
            return
        for line in stream:
            self._stderr.append(line)

    def stderr_tail(self, lines: int = 10) -> str:
        return "".join(list(self._stderr)[-lines:])

    def recent(self, kinds: tuple[str, ...] = (Event.ERROR, Event.RECORDED),
               limit: int = 10) -> list[dict]:
        return [item for item in self.log if item.get("ev") in kinds][-limit:]

    def describe(self) -> str:
        if not self.is_running:
            return "stopped"
        if self.ready is None:
            return "starting"
        return f"{self.ready['device']} · {self.ready['latency_ms']:.1f} ms blocks"

    def __enter__(self) -> "LiveSynth":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
