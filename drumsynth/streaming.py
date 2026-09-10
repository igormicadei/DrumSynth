"""Playing a model the way a sampler would: triggered, then a block at a time.

`model.render()` produces a whole hit at once. That is right for measuring a
fit and wrong for playing one: at 48 kHz with a 256-sample buffer, the audio
callback has 5.3 ms to fill each block, and a 3.5 second tom rendered in one
go takes far longer than that no matter how fast the transform is.

A :class:`Voice` spreads the same work across the hit. Triggering decodes the
component block — one matrix product — and every block after that inverse
transforms only the frames that overlap it. The total work is the same; what
changes is that none of it lands in one callback.

The output is sample-for-sample identical to `render()`. It has to be: two
paths that disagree would mean the thing being measured is not the thing being
heard.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spectral.stft import StftSpec, _window_power


@dataclass
class Voice:
    """One triggered hit, read out in blocks.

    `rows` are the spectrogram rows of the kept bins, already carrying the
    bin's own rotation — everything the model had to decide is decided by the
    time a voice exists.
    """

    rows: np.ndarray
    bins: np.ndarray
    spec: StftSpec
    n_samples: int

    def __post_init__(self) -> None:
        self._frames = int(self.rows.shape[1])
        self._power = _window_power(self.spec, self._frames)
        self._window = self.spec.window

        # The accumulator holds the samples that frames are still landing in.
        # `origin` is the absolute index of its first element; the first
        # `spec.pad` samples are the analysis padding and are never emitted.
        self._buffer = np.zeros(self.spec.n_fft, dtype=np.float64)
        self._origin = 0
        self._next_frame = 0
        self._position = self.spec.pad
        self._spectrum = np.zeros(self.spec.n_bins, dtype=np.complex128)

    @property
    def finished(self) -> bool:
        """True once every sample of the hit has been read."""
        return self._position - self.spec.pad >= self.n_samples

    @property
    def position(self) -> int:
        """How many samples have been read."""
        return self._position - self.spec.pad

    def read(self, n: int) -> np.ndarray:
        """The next `n` samples, zero-padded once the hit has run out."""
        out = np.zeros(n, dtype=np.float64)
        remaining = min(n, max(0, self.n_samples - self.position))
        if remaining == 0:
            self._position += n
            return out

        end = self._position + remaining
        self._add_frames_through(end - 1)
        self._grow_to(end)

        start = self._position - self._origin
        stop = start + remaining
        with np.errstate(invalid="ignore", divide="ignore"):
            power = self._power[self._position : end]
            out[:remaining] = np.divide(
                self._buffer[start:stop],
                power,
                out=np.zeros(remaining, dtype=np.float64),
                where=power > 1e-12,
            )

        self._position += n
        self._discard_before(self._position)
        return out

    # -- the accumulator ------------------------------------------------------

    def _add_frames_through(self, sample: int) -> None:
        """Add every frame that can still reach `sample`."""
        last = min(sample // self.spec.hop, self._frames - 1)
        while self._next_frame <= last:
            start = self._next_frame * self.spec.hop
            self._grow_to(start + self.spec.n_fft)

            self._spectrum[:] = 0.0
            self._spectrum[self.bins] = self.rows[:, self._next_frame]
            frame = np.fft.irfft(self._spectrum, n=self.spec.n_fft) * self._window

            offset = start - self._origin
            self._buffer[offset : offset + self.spec.n_fft] += frame
            self._next_frame += 1

    def _grow_to(self, sample: int) -> None:
        """Make sure the accumulator reaches `sample`."""
        needed = sample - self._origin
        if needed > self._buffer.size:
            self._buffer = np.concatenate(
                [self._buffer, np.zeros(needed - self._buffer.size + self.spec.n_fft)]
            )

    def _discard_before(self, sample: int) -> None:
        """Drop what has been read and can no longer be written to."""
        drop = sample - self._origin
        if drop >= self.spec.n_fft:
            keep = drop - (drop % self.spec.n_fft)
            self._buffer = self._buffer[keep:].copy()
            self._origin += keep

    def render_rest(self) -> np.ndarray:
        """Everything left, in one call. The same samples, read differently."""
        return self.read(max(0, self.n_samples - self.position))


def voice_for(model, velocity: float | None = None) -> Voice:
    """Trigger a hit from either kind of model.

    This is the moment a sampler would care about: everything the model has to
    decode happens here, and every block after it is transform and add.
    """
    from .instrument.model import InstrumentModel

    if isinstance(model, InstrumentModel):
        if velocity is None:
            raise ValueError("an instrument needs a velocity to be triggered")
        low, high = model.velocity_range
        rows = model.rows(float(np.clip(velocity, low, high)))
    else:
        rows = model.rows()

    return Voice(
        rows=rows,
        bins=np.asarray(model.bins, dtype=np.int64),
        spec=model.candidate.spec,
        n_samples=model.n_samples,
    )


def stream(model, velocity: float | None = None, block: int = 256) -> np.ndarray:
    """Play a whole hit through the streaming path. Mostly for checking it."""
    voice = voice_for(model, velocity)
    pieces = []
    while not voice.finished:
        pieces.append(voice.read(block))
    return np.concatenate(pieces)[: model.n_samples]
