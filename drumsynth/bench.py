"""How expensive a model is to play, measured rather than estimated.

Three numbers decide whether a model can be an instrument rather than a file
format:

    trigger     how long between asking for a hit and the first sample of it
    per block   how long each audio callback spends on one sounding voice
    polyphony   how many voices that leaves room for

The unit that matters is the callback's budget: at 44.1 kHz with a 256-sample
block, everything the process does has to fit in 5.8 ms. A cost is quoted as a
fraction of that budget, because "1.2 ms per block" means nothing on its own
and "20% of one core" means everything.

Both playback strategies are measured. A sampler can render the whole hit when
it is triggered and read out of the buffer afterwards, which is simple and puts
every millisecond in one callback; or it can stream (`drumsynth.streaming`),
which spreads the same work over the hit. They are the same total arithmetic
and completely different to schedule.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np

from .streaming import voice_for

#: Blocks timed per measurement before the numbers are trusted.
WARMUP = 3


def _time(call, repeats: int) -> list[float]:
    """Milliseconds per call, after a warm-up that is thrown away."""
    for _ in range(WARMUP):
        call()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        samples.append((time.perf_counter() - started) * 1000.0)
    return samples


def resident_bytes(model) -> int:
    """What the model's arrays occupy in memory, decoded and ready to play."""
    arrays = list(model._stored().values())
    if hasattr(model, "field_arrays"):
        arrays += [model.field.weights, model.field.patterns]
    return sum(int(np.asarray(a).nbytes) for a in arrays)


@dataclass(frozen=True)
class LiveCost:
    """What one model costs to play, at one block size."""

    name: str
    representation: str
    sample_rate: int
    duration: float
    block: int
    n_components: int
    n_frames: int

    trigger_ms: float
    first_block_ms: float
    block_mean_ms: float
    block_p95_ms: float
    block_max_ms: float
    render_ms: float

    file_bytes: int
    resident_bytes: int
    voice_bytes: int

    @property
    def budget_ms(self) -> float:
        """How long the audio callback has, at this block size."""
        return 1000.0 * self.block / self.sample_rate

    @property
    def load(self) -> float:
        """Fraction of one core a sounding voice takes, on average."""
        return self.block_mean_ms / self.budget_ms

    @property
    def worst_load(self) -> float:
        return self.block_max_ms / self.budget_ms

    @property
    def voices(self) -> float:
        """How many voices fit in one core, judged by the worst block."""
        return self.budget_ms / self.block_max_ms if self.block_max_ms else float("inf")

    @property
    def realtime_factor(self) -> float:
        """Seconds of audio produced per second of CPU, rendering in one go."""
        return 1000.0 * self.duration / self.render_ms if self.render_ms else float("inf")

    def to_dict(self) -> dict:
        data = asdict(self)
        data.update(
            budget_ms=self.budget_ms,
            load=self.load,
            worst_load=self.worst_load,
            voices=self.voices,
            realtime_factor=self.realtime_factor,
        )
        return data

    def report(self) -> str:
        return "\n".join(
            [
                f"model            {self.name} — {self.representation}",
                f"audio            {self.duration:.3f} s at {self.sample_rate} Hz, "
                f"{self.n_components} bins x {self.n_frames} frames",
                f"in memory        {self.resident_bytes / 1024:.1f} kB "
                f"({self.file_bytes / 1024:.1f} kB on disk), "
                f"+{self.voice_bytes / 1024:.1f} kB per sounding voice",
                "",
                f"block            {self.block} samples, "
                f"{self.budget_ms:.2f} ms of budget",
                f"trigger          {self.trigger_ms:.3f} ms "
                f"({100 * self.trigger_ms / self.budget_ms:.0f}% of one block)",
                f"first block      {self.first_block_ms:.3f} ms",
                f"per block        {self.block_mean_ms:.3f} ms mean, "
                f"{self.block_p95_ms:.3f} p95, {self.block_max_ms:.3f} max",
                f"one voice costs  {100 * self.load:.1f}% of a core "
                f"({100 * self.worst_load:.1f}% at its worst block)",
                f"room for         {self.voices:.0f} voices in one core",
                "",
                f"rendered in one go: {self.render_ms:.1f} ms for {self.duration:.2f} s "
                f"of audio ({self.realtime_factor:.0f}x real time), "
                f"{self.render_ms / self.budget_ms:.0f} blocks' worth in one callback",
            ]
        )


def measure_live(
    model, velocity: float | None = None, *, block: int = 256, repeats: int = 3
) -> LiveCost:
    """Time triggering, streaming and one-shot rendering of one model."""
    trigger = _time(lambda: voice_for(model, velocity), repeats)

    first = []
    per_block: list[float] = []
    voice_bytes = 0
    for _ in range(repeats):
        voice = voice_for(model, velocity)
        started = time.perf_counter()
        voice.read(block)
        first.append((time.perf_counter() - started) * 1000.0)

        while not voice.finished:
            started = time.perf_counter()
            voice.read(block)
            per_block.append((time.perf_counter() - started) * 1000.0)
        voice_bytes = max(voice_bytes, int(voice.rows.nbytes))

    render = _time(
        (lambda: model.render()) if velocity is None else (lambda: model.render(velocity)),
        repeats,
    )
    blocks = np.asarray(per_block)

    return LiveCost(
        name=getattr(model, "name", "hit"),
        representation=model.candidate.label(),
        sample_rate=model.sample_rate,
        duration=model.duration,
        block=block,
        n_components=int(model.bins.size),
        n_frames=int(model.n_frames),
        trigger_ms=float(np.median(trigger)),
        first_block_ms=float(np.median(first)),
        block_mean_ms=float(blocks.mean()),
        block_p95_ms=float(np.percentile(blocks, 95)),
        block_max_ms=float(blocks.max()),
        render_ms=float(np.median(render)),
        file_bytes=model.n_bytes(),
        resident_bytes=resident_bytes(model),
        voice_bytes=voice_bytes,
    )


def measure_polyphony(
    model, velocity: float | None = None, *, block: int = 256, voices: int = 8
) -> dict:
    """Time a block with `voices` hits sounding at once, mixed.

    Not the same as multiplying the single-voice cost: voices land on different
    frames, so their expensive blocks do not line up, and the mix itself costs
    something.
    """
    budget = 1000.0 * block / model.sample_rate
    playing = [voice_for(model, velocity) for _ in range(voices)]

    # Stagger them, the way hits in a bar would be.
    spread = max(1, int(model.n_samples / max(voices, 1) / block))
    for index, voice in enumerate(playing):
        for _ in range(index * spread):
            voice.read(block)

    mix = np.zeros(block, dtype=np.float64)
    costs = []
    for _ in range(200):
        started = time.perf_counter()
        mix[:] = 0.0
        for voice in playing:
            mix += voice.read(block)
        costs.append((time.perf_counter() - started) * 1000.0)
        playing = [v if not v.finished else voice_for(model, velocity) for v in playing]

    measured = np.asarray(costs)
    return {
        "voices": voices,
        "block": block,
        "budget_ms": budget,
        "mean_ms": float(measured.mean()),
        "p95_ms": float(np.percentile(measured, 95)),
        "max_ms": float(measured.max()),
        "load": float(measured.mean() / budget),
        "worst_load": float(measured.max() / budget),
    }
