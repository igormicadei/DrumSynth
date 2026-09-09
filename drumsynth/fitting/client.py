"""Driving a fit from the UI process.

The same shape as `live.LiveSynth`: own the subprocess, drain its stdout in a
background thread so the pipe never fills, and expose what has arrived so far
as plain state a Streamlit rerun can read.

A fit runs for minutes. Nothing here blocks on it.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from collections import deque
from pathlib import Path

from ..live.protocol import Event
from .worker import FitEvent


class TrainingRun:
    """A handle on one `drumsynth.fitting.worker` process."""

    LOG_SIZE: int = 4000

    def __init__(self, python: str | None = None, cwd: str | Path | None = None) -> None:
        self.python = python or sys.executable
        self.cwd = str(cwd) if cwd else None

        self.process: subprocess.Popen | None = None
        self.ready: dict | None = None
        self.result: dict | None = None
        self.scored: dict | None = None
        self.error: dict | None = None
        self.done: dict | None = None

        self.stage: dict = {}
        self.generations: list[dict] = []
        self.steps: deque[str] = deque(maxlen=200)
        self.log: deque[dict] = deque(maxlen=TrainingRun.LOG_SIZE)
        self._stderr: deque[str] = deque(maxlen=80)
        self._reader: threading.Thread | None = None

    # -- lifecycle ------------------------------------------------------------

    def command_line(self, manifest: str | Path, output: str | Path,
                     settings: dict) -> list[str]:
        argv = [
            self.python, "-m", "drumsynth.fitting.worker",
            "--manifest", str(manifest),
            "--output", str(output),
            "--seconds", str(settings.get("seconds", 2.5)),
            "--max-layers", str(settings.get("max_layers", 8)),
            "--max-modes", str(settings.get("max_modes", 34)),
            "--generations", str(settings.get("generations", 24)),
            "--population", str(settings.get("population", 12)),
            "--device", str(settings.get("device", "auto")),
            "--control-period", str(settings.get("control_period", 64)),
            "--noise-bands", str(settings.get("noise_bands", 4)),
            "--seed", str(settings.get("seed", 0)),
        ]
        if not settings.get("refine_gains", True):
            argv.append("--no-refine")
        return argv

    def start(self, manifest: str | Path, output: str | Path,
              settings: dict) -> "TrainingRun":
        if self.is_running:
            return self
        self.__init__(self.python, self.cwd)  # a run is single-use
        self.process = subprocess.Popen(
            self.command_line(manifest, output, settings),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=self.cwd,
        )
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        threading.Thread(target=self._read_stderr, daemon=True).start()
        return self

    def stop(self) -> None:
        if self.process is None:
            return
        try:
            self.process.terminate()
            self.process.wait(timeout=5)
        except Exception:
            self.process.kill()
        finally:
            self.process = None

    @property
    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    @property
    def finished(self) -> bool:
        return self.done is not None or self.error is not None

    # -- reading --------------------------------------------------------------

    def _read(self) -> None:
        stream = self.process.stdout if self.process else None
        if stream is None:
            return
        for line in stream:
            message = Event.decode(line)
            if message is None:
                continue
            self.log.append(message)
            kind = message.get("ev")
            if kind == FitEvent.READY:
                self.ready = message
            elif kind == FitEvent.GENERATION:
                self.generations.append(message)
            elif kind == FitEvent.STAGE:
                self.stage = message
                self.steps.append(TrainingRun._describe(message))
            elif kind == FitEvent.PROGRESS:
                described = TrainingRun._describe(message)
                if described:
                    self.steps.append(described)
            elif kind == FitEvent.RESULT:
                self.result = message
            elif kind == FitEvent.SCORED:
                self.scored = message
            elif kind == FitEvent.ERROR:
                self.error = message
            elif kind == FitEvent.DONE:
                self.done = message

    def _read_stderr(self) -> None:
        stream = self.process.stderr if self.process else None
        if stream is None:
            return
        for line in stream:
            self._stderr.append(line)

    @staticmethod
    def _describe(message: dict) -> str:
        phase = message.get("phase")
        if phase == "start":
            return f"fitting {message.get('drum')} — {message.get('layers')} layers"
        if phase == "stage1":
            return f"stage 1: {message.get('modes')} modes frozen"
        if phase == "stage2":
            which = message.get("pass", "")
            return (f"stage 2{' (' + which + ' pass)' if which else ''}: layer "
                    f"{message.get('layer')}/{message.get('of')} "
                    f"v{message.get('velocity', 0):.0f}")
        if phase == "tension":
            return f"glide: k = {message.get('k', 0):.4g}"
        if phase == "stage3":
            return f"stage 3: {message.get('verdict', '')}"
        if phase == "stage4":
            return "stage 4: velocity curves fitted"
        if phase == "stage5_done":
            return f"stage 5: {message.get('generations')} generations"
        if phase == "scoring":
            return (f"scoring layer {message.get('layer')}/{message.get('of')} "
                    f"v{message.get('velocity', 0):.0f}")
        if phase == "done":
            return f"finished in {message.get('elapsed', 0):.0f} s"
        if "step" in message:
            return str(message["step"])
        return ""

    # -- what the UI reads ----------------------------------------------------

    @property
    def phase(self) -> str:
        if self.error is not None:
            return "error"
        if self.done is not None:
            return "done"
        if self.scored is not None:
            return "scored"
        if self.result is not None:
            return "scoring"
        return str(self.stage.get("phase", "starting"))

    def progress_fraction(self) -> float:
        """Rough completion, for a progress bar. Stage weights are measured
        rather than guessed — stage 2 dominates."""
        weights = {
            "starting": 0.0, "start": 0.02, "stage1": 0.10, "stage2": 0.20,
            "tension": 0.55, "stage3": 0.70, "stage4": 0.72, "stage5": 0.75,
            "stage5_done": 0.90, "scoring": 0.92, "scored": 0.98,
            "done": 1.0, "error": 1.0,
        }
        base = weights.get(self.phase, 0.5)
        if self.phase == "stage5" and self.generations:
            base = 0.75 + 0.15 * min(len(self.generations) / 24.0, 1.0)
        return float(min(max(base, 0.0), 1.0))

    def stderr_tail(self, lines: int = 12) -> str:
        return "".join(list(self._stderr)[-lines:])
