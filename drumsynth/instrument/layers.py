"""Turning a drum's recordings into velocity layers a model can be fitted to.

A drum in the library is a grid: 26 velocity bands for tom 3, four round robins
in each, 104 separate recordings of the same physical object hit harder and
harder. What a velocity model needs from that is one aligned block per
velocity, and the recordings do not arrive aligned:

* they start at slightly different offsets, and averaging or interpolating
  two hits whose transients are 2 ms apart smears both;
* they are not the same length, and a quiet hit is genuinely shorter;
* several round robins share a velocity, and each is a different strike
  position on the same head.

So every recording is shifted to put its onset at the same place and padded to
a common length. What happens to the round robins after that is a choice the
fit exposes rather than makes:

* by default one take per velocity is the anchor, and the others are what
  generalization is measured against. The model then reproduces recordings
  that exist, and every number it reports can go to zero.
* with `average=True` the magnitudes of all the takes are averaged instead.
  That predicts the *next* strike measurably better — a mean is closer to an
  unseen take than any single take is (docs/FINDINGS.md §11) — at the cost of
  no longer reproducing any particular recording exactly, because the phase
  can still only come from one of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field

import numpy as np

from ..corpus import Corpus, Sample
from ..spectral.bins import bin_energy
from ..spectral.components import to_components
from ..spectral.stft import StftSpec, analyze

#: A sample counts as started once it passes this fraction of its own peak.
ONSET_THRESHOLD = 0.02

#: Silence kept in front of the onset, in seconds.
PRE_ROLL = 0.002


def onset(signal: np.ndarray, threshold: float = ONSET_THRESHOLD) -> int:
    """First sample that passes `threshold` of the signal's own peak."""
    x = np.abs(np.asarray(signal, dtype=np.float64))
    peak = float(x.max()) if x.size else 0.0
    if peak <= 0.0:
        return 0
    return int(np.argmax(x >= threshold * peak))


def align(signal: np.ndarray, sample_rate: int, length: int) -> np.ndarray:
    """Put the onset at `PRE_ROLL` seconds and pad or trim to `length`.

    Shifts either way: a recording with a long lead-in is cut into, and one
    that starts on the first sample is pushed back, so every layer of a drum
    has its transient at the same place whatever the recordings did.
    """
    x = np.asarray(signal, dtype=np.float64)
    start = onset(x) - int(round(PRE_ROLL * sample_rate))

    out = np.zeros(length, dtype=np.float64)
    if start >= 0:
        keep = min(length, x.size - start)
        out[:keep] = x[start : start + keep]
    else:
        keep = min(length + start, x.size)
        out[-start : -start + keep] = x[:keep]
    return out


@dataclass
class VelocityLayers:
    """Every recording of one drum, aligned and grouped by velocity.

    `hits[i]` are the round robins recorded at `velocities[i]`, aligned and all
    of length `n_samples`.
    """

    name: str
    sample_rate: int
    velocities: np.ndarray
    hits: list[list[np.ndarray]]
    _bin_orders: dict = dataclass_field(default_factory=dict, repr=False)

    @property
    def n_layers(self) -> int:
        return int(self.velocities.size)

    @property
    def n_samples(self) -> int:
        return int(self.hits[0][0].size)

    @property
    def n_recordings(self) -> int:
        return sum(len(group) for group in self.hits)

    @property
    def duration(self) -> float:
        return self.n_samples / float(self.sample_rate)

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_samples(
        cls,
        name: str,
        samples: list[Sample],
        *,
        sample_rate: int | None = None,
        max_duration: float | None = None,
    ) -> "VelocityLayers":
        """Load, align and group recordings that are on disk."""
        present = [s for s in samples if s.present()]
        if not present:
            raise FileNotFoundError(
                f"{name}: none of its {len(samples)} recordings are on this machine — "
                "the library indexes the audio but does not ship it, see docs/DATA.md"
            )

        loaded: list[tuple[float, int, np.ndarray]] = []
        rate = sample_rate
        for sample in sorted(present, key=lambda s: (s.velocity, s.round_robin)):
            signal, file_rate = sample.load(sr=rate)
            rate = rate or file_rate
            loaded.append((sample.velocity, sample.round_robin, signal))

        length = max(signal.size for _, _, signal in loaded)
        if max_duration is not None:
            length = min(length, int(round(max_duration * rate)))

        grouped: dict[float, list[np.ndarray]] = {}
        for velocity, _, signal in loaded:
            grouped.setdefault(velocity, []).append(align(signal, rate, length))

        velocities = np.array(sorted(grouped), dtype=np.float64)
        return cls(
            name=name,
            sample_rate=int(rate),
            velocities=velocities,
            hits=[grouped[v] for v in velocities],
        )

    @classmethod
    def from_corpus(
        cls,
        drum: str,
        corpus: Corpus | None = None,
        *,
        sample_rate: int | None = None,
        max_duration: float | None = None,
    ) -> "VelocityLayers":
        corpus = corpus or Corpus.load()
        return cls.from_samples(
            drum,
            corpus.samples(drum),
            sample_rate=sample_rate,
            max_duration=max_duration,
        )

    @classmethod
    def from_audio(
        cls, name: str, sample_rate: int, hits: dict[float, list[np.ndarray]]
    ) -> "VelocityLayers":
        """Build layers straight from audio — for tests, and for a library of one."""
        length = max(h.size for group in hits.values() for h in group)
        velocities = np.array(sorted(hits), dtype=np.float64)
        return cls(
            name=name,
            sample_rate=int(sample_rate),
            velocities=velocities,
            hits=[[align(h, sample_rate, length) for h in hits[v]] for v in velocities],
        )

    # -- analysis -------------------------------------------------------------

    def bin_order(self, spec: StftSpec) -> np.ndarray:
        """Bins ranked over the whole instrument, most useful first.

        Each recording's energy is normalized before the ranking, so a bin that
        only matters to the quiet layers is not buried under the loud ones. The
        model has to serve every velocity, not the ones that happen to carry
        the most energy.

        Cached: the ranking depends on the framing alone, and a search asks for
        it once per bin count per framing.
        """
        cached = self._bin_orders.get(spec)
        if cached is not None:
            return cached

        score = np.zeros(spec.n_bins, dtype=np.float64)
        for group in self.hits:
            for signal in group:
                energy = bin_energy(analyze(signal, spec))
                total = float(energy.sum())
                if total > 0.0:
                    score += energy / total

        order = np.argsort(score, kind="stable")[::-1]
        self._bin_orders[spec] = order
        return order

    def select_bins(self, spec: StftSpec, n_components: int) -> np.ndarray:
        """The `n_components` best bins for this instrument, in ascending order."""
        keep = int(np.clip(n_components, 1, spec.n_bins))
        return np.sort(self.bin_order(spec)[:keep]).astype(np.int32)

    def block_of(
        self, layer: int, take: int, spec: StftSpec, bins: np.ndarray
    ) -> np.ndarray:
        """The component block of one recording, or None where that take is missing."""
        group = self.hits[layer]
        return to_components(analyze(group[take % len(group)], spec), bins, spec)

    def takes(self, layer: int) -> int:
        return len(self.hits[layer])

    def analyse(
        self, spec: StftSpec, bins: np.ndarray, take: int = 0, average: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """(magnitudes, blocks) for every layer.

        `magnitudes` is (layers, bins, frames) and feeds the velocity field;
        `blocks` are complex and are where phase comes from. Phase always comes
        from `take`, since it can only come from one strike; `average` decides
        whether the magnitudes come from that take too, or from all of them.
        """
        n_frames = spec.n_frames(self.n_samples)
        magnitudes = np.zeros((self.n_layers, bins.size, n_frames), dtype=np.float64)
        blocks = np.zeros((self.n_layers, bins.size, n_frames), dtype=np.complex128)

        for layer in range(self.n_layers):
            blocks[layer] = self.block_of(layer, take, spec, bins)
            if average:
                for other in range(self.takes(layer)):
                    magnitudes[layer] += np.abs(
                        blocks[layer]
                        if other == take % self.takes(layer)
                        else self.block_of(layer, other, spec, bins)
                    )
                magnitudes[layer] /= self.takes(layer)
            else:
                magnitudes[layer] = np.abs(blocks[layer])

        return magnitudes, blocks

    def audio(self, layer: int, take: int = 0) -> np.ndarray:
        group = self.hits[layer]
        return group[take % len(group)]
