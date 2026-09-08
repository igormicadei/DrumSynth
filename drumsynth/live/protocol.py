"""The wire format between the UI and the audio process.

Newline-delimited JSON over stdin/stdout. Not because JSON is fast — it is the
slowest reasonable choice — but because the two processes are decoupled by a
pipe rather than by shared memory, and a pipe means a crash in the audio
process cannot take the UI with it, and a stalled UI cannot glitch the audio.

The traffic is tiny: a strike is 40 bytes, a full parameter set a few kB, and
neither happens at audio rate.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


class Command:
    """UI -> engine. Names are the wire values; do not rename casually."""

    STRIKE = "strike"
    PARAMS = "params"
    RESET = "reset"
    PANIC = "panic"
    RECORD = "record"
    GAIN = "gain"
    MONITOR = "monitor"
    PING = "ping"
    QUIT = "quit"

    @staticmethod
    def encode(name: str, **fields: Any) -> str:
        return json.dumps({"cmd": name, **fields}, separators=(",", ":"))

    @staticmethod
    def decode(line: str) -> dict | None:
        line = line.strip()
        if not line:
            return None
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            return None
        return message if isinstance(message, dict) and "cmd" in message else None


class Event:
    """Engine -> UI."""

    READY = "ready"
    STATUS = "status"
    RECORDED = "recorded"
    ERROR = "error"
    STOPPED = "stopped"

    @staticmethod
    def encode(name: str, **fields: Any) -> str:
        return json.dumps({"ev": name, **fields}, separators=(",", ":"))

    @staticmethod
    def decode(line: str) -> dict | None:
        line = line.strip()
        if not line:
            return None
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            return None
        return message if isinstance(message, dict) and "ev" in message else None


@dataclass
class Status:
    """One telemetry frame. Everything the UI needs to draw a meter."""

    peak: float = 0.0            # linear, since the last frame
    rms: float = 0.0
    energy: float = 0.0          # modal bank energy, drives the tension ratio
    frequency_ratio: float = 1.0
    blocks: int = 0
    strikes: int = 0
    underruns: int = 0           # audio callback missed its deadline
    clipped: bool = False
    load: float = 0.0            # fraction of the block period spent rendering
    seconds: float = 0.0         # engine uptime

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Status":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class EngineSettings:
    """How the audio process is configured. Fixed for the life of a process."""

    sr: int = 44100
    block_size: int = 256
    control_period: int = 64
    sink: str = "device"          # device | null | wav
    device: str | None = None
    seed: int | None = 0
    status_interval: float = 0.1  # seconds between telemetry frames
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
