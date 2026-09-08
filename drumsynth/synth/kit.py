"""Turning a list of hits into audio.

Separate from `DrumKit` because scheduling is not synthesis: the sequencer owns
time, the kit owns voices. It exists mainly to make the superposition claim easy
to hear — a flam is two hits 25 ms apart, and it works because the resonator
bank has persistent state, not because anything here handles it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.audio_io import AudioIO
from ..core.constants import Audio
from .voice import DrumKit


@dataclass
class Hit:
    """One scheduled strike."""

    time: float  # s from the start of the pattern
    drum: str
    amplitude: float = 1.0

    def shifted(self, seconds: float) -> "Hit":
        return Hit(self.time + seconds, self.drum, self.amplitude)


class DrumSequencer:
    """Schedule hits onto a kit and render the result.

    Times are in seconds, not steps. A grid is a convenience the caller can
    build with `from_grid`; nothing downstream knows about tempo.
    """

    def __init__(self, kit: DrumKit, sr: int | None = None) -> None:
        self.kit = kit
        self.sr = int(sr or kit.sr)
        self.hits: list[Hit] = []

    # -- building -------------------------------------------------------------

    def add(self, time: float, drum: str, amplitude: float = 1.0) -> "DrumSequencer":
        if drum not in self.kit:
            raise KeyError(f"no drum named {drum!r} in this kit; have {self.kit.names()}")
        self.hits.append(Hit(float(time), drum, float(amplitude)))
        return self

    def add_flam(
        self, time: float, drum: str, amplitude: float = 1.0,
        spacing: float = 0.025, grace_ratio: float = 0.45,
    ) -> "DrumSequencer":
        """A grace note followed by the main hit.

        The point of this method is that it needs no support anywhere else: the
        second strike lands on a bank that is still ringing and superposes.
        """
        self.add(time - spacing, drum, amplitude * grace_ratio)
        return self.add(time, drum, amplitude)

    def add_roll(
        self, start: float, drum: str, count: int, interval: float,
        amplitude: float = 0.6, accent_every: int = 0, accent: float = 1.0,
    ) -> "DrumSequencer":
        for index in range(count):
            level = (
                accent
                if accent_every and index % accent_every == 0
                else amplitude
            )
            self.add(start + index * interval, drum, level)
        return self

    @classmethod
    def from_grid(
        cls,
        kit: DrumKit,
        pattern: dict[str, str],
        bpm: float = 100.0,
        steps_per_beat: int = 4,
        amplitude: float = 1.0,
        ghost: float = 0.35,
    ) -> "DrumSequencer":
        """Build from a step grid: 'x' is a hit, 'o' a ghost note, '.' a rest.

            DrumSequencer.from_grid(kit, {"kick": "x..x..x.", "snare": "..o.x..o"})
        """
        sequencer = cls(kit)
        step_seconds = 60.0 / bpm / steps_per_beat
        for drum, row in pattern.items():
            for index, symbol in enumerate(row):
                if symbol in "xX":
                    sequencer.add(index * step_seconds, drum, amplitude)
                elif symbol in "oO":
                    sequencer.add(index * step_seconds, drum, amplitude * ghost)
        return sequencer

    # -- rendering ------------------------------------------------------------

    def duration(self, tail: float = 2.0) -> float:
        """Pattern length plus enough tail for the slowest decay to finish."""
        last = max((hit.time for hit in self.hits), default=0.0)
        return last + tail

    def render(self, seconds: float | None = None, tail: float = 2.0) -> np.ndarray:
        total = seconds if seconds is not None else self.duration(tail)
        triples = [(hit.time, hit.drum, hit.amplitude) for hit in self.hits]
        return self.kit.render_pattern(triples, total)

    def write(self, path: str, seconds: float | None = None, tail: float = 2.0):
        return AudioIO.write(path, self.render(seconds, tail), self.sr)

    def clear(self) -> "DrumSequencer":
        self.hits.clear()
        return self

    def __len__(self) -> int:
        return len(self.hits)
