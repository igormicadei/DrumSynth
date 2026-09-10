"""One drum, every velocity, one file.

An :class:`InstrumentModel` renders a hit at any velocity in the range that was
recorded — including velocities that were not. It holds two things:

    the magnitude field   what the drum sounds like as a function of velocity
    the donors            phase fields borrowed whole from individual strikes

and renders by combining them:

    render(v) = synthesize( field.at(v) · exp(i · phase of nearest donor) )

The split is the whole design. Magnitude varies smoothly with how hard a drum
is hit and can be modelled, compressed and interpolated. Phase does not vary
smoothly with anything — it is set by one strike — so it is stored per layer
and taken, never mixed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Sequence

import numpy as np

from ..spectral.components import drift
from ..spectral.stft import StftSpec, synthesize
from . import donors as donor_codecs
from .field import MagnitudeField
from .field import codec_for as field_codec_for

#: Bumped whenever the on-disk layout changes. Old files are not migrated.
FORMAT = "drumsynth.instrument/1"


@dataclass(frozen=True)
class InstrumentCandidate:
    """The choices that define a velocity model."""

    n_fft: int
    hop: int
    n_components: int
    field_codec: str = "separable"
    field_rank: int = 4
    pattern_rank: int = 16
    donor_codec: str = "lowrank"
    donor_rank: int = 16
    n_donors: int = 0  # 0 means every layer donates

    @property
    def spec(self) -> StftSpec:
        return StftSpec(self.n_fft, self.hop)

    def n_scalars(self, n_layers: int, n_frames: int, n_donors: int) -> int:
        """Stored numbers: the field, the donors, the bin list and the velocities."""
        shape = (n_layers, self.n_components, n_frames)
        field = field_codec_for(self.field_codec).n_scalars(
            shape, self.field_rank, self.pattern_rank
        )
        donor = donor_codecs.codec_for(self.donor_codec).n_scalars(
            self.n_components, n_frames, self.donor_rank
        )
        return field + n_donors * donor + self.n_components + n_layers + n_donors

    def label(self) -> str:
        parts = [f"{self.field_codec}"]
        if self.field_codec != "full":
            parts.append(f"rank={self.field_rank}")
        if self.field_codec == "separable":
            parts.append(f"pattern={self.pattern_rank}")
        parts.append(f"+{self.donor_codec}")
        if self.donor_codec != "exact":
            parts.append(f"rank={self.donor_rank}")
        parts.append(f"donors={'all' if self.n_donors <= 0 else self.n_donors}")
        parts.append(f"n_fft={self.n_fft}")
        parts.append(f"hop={self.hop}")
        parts.append(f"k={self.n_components}")
        return " ".join(parts)

    def to_dict(self) -> dict:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: dict) -> "InstrumentCandidate":
        return cls(**{f: data[f] for f in cls.__dataclass_fields__ if f in data})


@dataclass
class InstrumentModel:
    """A whole drum: every velocity, and everything between them."""

    name: str
    candidate: InstrumentCandidate
    sample_rate: int
    n_samples: int
    n_frames: int
    bins: np.ndarray
    velocities: np.ndarray
    field_arrays: dict[str, np.ndarray]
    donor_velocities: np.ndarray
    donor_arrays: list[dict[str, np.ndarray]] = dataclass_field(default_factory=list)

    # -- playback -------------------------------------------------------------

    @property
    def field(self) -> MagnitudeField:
        """The magnitude field, decoded from the arrays the file holds."""
        if self._field is None:
            self._field = MagnitudeField.from_arrays(
                self.velocities, self.candidate.field_codec, self.field_arrays
            )
        return self._field

    _field: MagnitudeField | None = dataclass_field(default=None, repr=False)

    @property
    def velocity_range(self) -> tuple[float, float]:
        return self.field.range

    @property
    def duration(self) -> float:
        return self.n_samples / float(self.sample_rate)

    def donor_index(self, velocity: float) -> int:
        return donor_codecs.nearest(self.donor_velocities, velocity)

    def phase(self, velocity: float) -> np.ndarray:
        codec = donor_codecs.codec_for(self.candidate.donor_codec)
        return codec.decode(self.donor_arrays[self.donor_index(velocity)])

    def components(self, velocity: float) -> np.ndarray:
        """The component block this drum has at `velocity`."""
        return self.field.at(velocity) * np.exp(1j * self.phase(velocity))

    def rotation(self) -> np.ndarray:
        """exp(i · drift): the bin's own turn, the same for every render."""
        if self._rotation is None:
            self._rotation = np.exp(
                1j * drift(self.bins, self.candidate.spec, self.n_frames)
            )
        return self._rotation

    _rotation: np.ndarray | None = dataclass_field(default=None, repr=False)

    def rows(self, velocity: float) -> np.ndarray:
        """The spectrogram rows of the kept bins at `velocity`.

        One expression, used by rendering, by streaming and by the search, so
        that all three produce the same samples rather than nearly the same
        ones.
        """
        return self.components(velocity) * self.rotation()

    def render(self, velocity: float) -> np.ndarray:
        """Synthesize a hit at `velocity`, clamped to the recorded range."""
        spec = self.candidate.spec
        spectrogram = np.zeros((spec.n_bins, self.n_frames), dtype=np.complex128)
        spectrogram[self.bins] = self.rows(velocity)
        return synthesize(spectrogram, spec, self.n_samples)

    def frequencies(self) -> np.ndarray:
        return self.candidate.spec.bin_frequencies(self.sample_rate)[self.bins]

    # -- size -----------------------------------------------------------------

    @property
    def n_scalars(self) -> int:
        return sum(
            int(np.size(a)) * (2 if np.iscomplexobj(a) else 1)
            for a in self._stored().values()
        )

    def to_bytes(self) -> bytes:
        import io

        buffer = io.BytesIO()
        self._save_to(buffer)
        return buffer.getvalue()

    def n_bytes(self) -> int:
        return len(self.to_bytes())

    def compression_ratio(self, n_recordings: int) -> float:
        """Model bytes against 16-bit PCM of every recording it was fitted to."""
        return self.n_bytes() / (2.0 * self.n_samples * max(n_recordings, 1))

    # -- persistence ----------------------------------------------------------

    def metadata(self) -> dict:
        return {
            "format": FORMAT,
            "name": self.name,
            "candidate": self.candidate.to_dict(),
            "sample_rate": self.sample_rate,
            "n_samples": self.n_samples,
            "n_frames": self.n_frames,
            "duration": self.duration,
            "n_components": int(self.bins.size),
            "n_layers": int(self.velocities.size),
            "velocity_range": list(self.velocity_range),
            "n_donors": len(self.donor_arrays),
            "n_scalars": self.n_scalars,
            "field_arrays": sorted(self.field_arrays),
            "donor_arrays": sorted(self.donor_arrays[0]) if self.donor_arrays else [],
        }

    def _stored(self) -> dict[str, np.ndarray]:
        arrays = {
            "bins": self.bins,
            "velocities": self.velocities.astype(np.float32),
            "donor_velocities": self.donor_velocities.astype(np.float32),
            **{f"field__{k}": v for k, v in self.field_arrays.items()},
        }
        for index, donor in enumerate(self.donor_arrays):
            arrays.update({f"donor{index}__{k}": v for k, v in donor.items()})
        return arrays

    def _save_to(self, target) -> None:
        payload = {k: np.asarray(v) for k, v in self._stored().items()}
        payload["meta"] = np.asarray(json.dumps(self.metadata()))
        np.savez_compressed(target, **payload)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            self._save_to(handle)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "InstrumentModel":
        with np.load(Path(path), allow_pickle=False) as data:
            meta = json.loads(str(data["meta"].item()))
            if meta.get("format") != FORMAT:
                raise ValueError(
                    f"{path}: expected format {FORMAT!r}, found {meta.get('format')!r}"
                )

            candidate = InstrumentCandidate.from_dict(meta["candidate"])
            field_arrays = {
                key[len("field__") :]: data[key]
                for key in data.files
                if key.startswith("field__")
            }
            donor_arrays: list[dict[str, np.ndarray]] = [
                {} for _ in range(int(meta["n_donors"]))
            ]
            for key in data.files:
                if not key.startswith("donor") or "__" not in key:
                    continue
                head, name = key.split("__", 1)
                donor_arrays[int(head[len("donor") :])][name] = data[key]

            return cls(
                name=meta["name"],
                candidate=candidate,
                sample_rate=int(meta["sample_rate"]),
                n_samples=int(meta["n_samples"]),
                n_frames=int(meta["n_frames"]),
                bins=data["bins"],
                velocities=np.asarray(data["velocities"], dtype=np.float64),
                field_arrays=field_arrays,
                donor_velocities=np.asarray(data["donor_velocities"], dtype=np.float64),
                donor_arrays=donor_arrays,
            )

    @classmethod
    def build(
        cls,
        name: str,
        candidate: InstrumentCandidate,
        *,
        sample_rate: int,
        n_samples: int,
        velocities: np.ndarray,
        bins: np.ndarray,
        magnitudes: np.ndarray,
        blocks: np.ndarray,
    ) -> "InstrumentModel":
        """Fit the field and the donors of one candidate to an analysed drum."""
        return InstrumentAnalysis(
            name=name,
            sample_rate=int(sample_rate),
            n_samples=int(n_samples),
            velocities=np.asarray(velocities, dtype=np.float64),
            bins=np.asarray(bins, dtype=np.int32),
            magnitudes=magnitudes,
            blocks=blocks,
        ).build(candidate)


@dataclass
class InstrumentAnalysis:
    """One analysis of a drum, and every model that can be built from it.

    A search asks for hundreds of models over the same analysis, differing only
    in ranks and codecs. The factorizations those ranks slice are the expensive
    part and do not depend on the ranks at all, so they are computed once here
    and kept.
    """

    name: str
    sample_rate: int
    n_samples: int
    velocities: np.ndarray
    bins: np.ndarray
    magnitudes: np.ndarray
    blocks: np.ndarray
    _fields: dict = dataclass_field(default_factory=dict, repr=False)
    _donors: dict = dataclass_field(default_factory=dict, repr=False)

    @property
    def n_layers(self) -> int:
        return int(self.velocities.size)

    @property
    def n_frames(self) -> int:
        return int(self.magnitudes.shape[2])

    def field_state(self, codec: str):
        if codec not in self._fields:
            self._fields[codec] = field_codec_for(codec).prepare(self.magnitudes)
        return self._fields[codec]

    def donor_state(self, codec: str, layer: int):
        key = (codec, layer)
        if key not in self._donors:
            self._donors[key] = donor_codecs.codec_for(codec).prepare(self.blocks[layer])
        return self._donors[key]

    def donors(self, candidate: InstrumentCandidate) -> np.ndarray:
        """Which layers donate phase, for this candidate's donor count."""
        return donor_codecs.choose(self.velocities, candidate.n_donors)

    def magnitudes_at(
        self, field_setting: tuple[str, int, int], velocities: np.ndarray
    ) -> np.ndarray:
        """The magnitude block one field setting gives at each of `velocities`."""
        codec_name, rank, pattern_rank = field_setting
        codec = field_codec_for(codec_name)
        arrays = codec.encode(self.field_state(codec_name), rank, pattern_rank)
        field = MagnitudeField.from_arrays(self.velocities, codec_name, arrays)
        return np.stack([field.at(velocity) for velocity in velocities])

    def phase_units(self, codec_name: str, rank: int, layers: Sequence[int]) -> np.ndarray:
        """exp(i·phase) of one donor setting, at each of `layers`.

        The exponential rather than the phase, because that is what a rendered
        block is multiplied by and a search does it tens of thousands of times.
        """
        codec = donor_codecs.codec_for(codec_name)
        return np.stack(
            [
                np.exp(1j * codec.decode(codec.encode(self.donor_state(codec_name, layer), rank)))
                for layer in layers
            ]
        )

    def build(self, candidate: InstrumentCandidate) -> InstrumentModel:
        """Assemble one candidate's model from the shared factorizations."""
        codec = field_codec_for(candidate.field_codec)
        field_arrays = codec.encode(
            self.field_state(candidate.field_codec),
            candidate.field_rank,
            candidate.pattern_rank,
        )

        chosen = donor_codecs.choose(self.velocities, candidate.n_donors)
        donor_codec = donor_codecs.codec_for(candidate.donor_codec)
        donor_arrays = [
            donor_codec.encode(
                self.donor_state(candidate.donor_codec, int(layer)), candidate.donor_rank
            )
            for layer in chosen
        ]

        return InstrumentModel(
            name=self.name,
            candidate=candidate,
            sample_rate=self.sample_rate,
            n_samples=self.n_samples,
            n_frames=self.n_frames,
            bins=self.bins,
            velocities=self.velocities,
            field_arrays=field_arrays,
            donor_velocities=self.velocities[chosen],
            donor_arrays=donor_arrays,
        )
