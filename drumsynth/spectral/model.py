"""The model itself: what a fitted sound is reduced to, and how it plays back.

A :class:`SpectralModel` is a :class:`Candidate` — the choices that define the
representation — plus the arrays those choices imply. It knows how to render
itself to audio, how big it is, and how to survive a round trip through a file.
It does not know how it was chosen; that is :mod:`drumsynth.fitting`.

Everything is stored at float32 (complex64 where the codec is complex), and
the quantization happens at encode time. So the error a fit reports is the
error of the model as it will be stored, never of a double-precision version
of it that no file ever contains.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .components import codec_for, from_components
from .stft import StftSpec, synthesize

#: Bumped whenever the on-disk layout changes. Old files are not migrated.
FORMAT = "drumsynth.spectral/1"


@dataclass(frozen=True)
class Candidate:
    """One point in the space of representations.

    n_fft, hop      the analysis framing
    n_components    how many FFT bins the model keeps
    codec           how the component block is stored: raw, shared or lowrank
    codec_param     the rank, for shared and lowrank; unused for raw
    phase_stride    store phase every this many frames and interpolate between;
                    meaningless for lowrank, which has the phase inside its
                    factorization, and forced to 1 there
    """

    n_fft: int
    hop: int
    n_components: int
    codec: str = "raw"
    codec_param: int = 0
    phase_stride: int = 1

    def __post_init__(self) -> None:
        if not self.codec_class.uses_phase and self.phase_stride != 1:
            raise ValueError(
                f"codec {self.codec!r} stores its own phase; phase_stride must be 1"
            )

    @property
    def spec(self) -> StftSpec:
        return StftSpec(self.n_fft, self.hop)

    @property
    def codec_class(self):
        return codec_for(self.codec)

    def n_scalars(self, n_frames: int) -> int:
        """Stored numbers: the component block, plus the list of kept bins."""
        return (
            self.codec_class.n_scalars(
                self.n_components, n_frames, self.codec_param, self.phase_stride
            )
            + self.n_components
        )

    def label(self) -> str:
        parts = [self.codec]
        if self.codec_class.param_name != "unused":
            parts.append(f"{self.codec_class.param_name[:4]}={self.codec_param}")
        parts.append(f"n_fft={self.n_fft}")
        parts.append(f"hop={self.hop}")
        parts.append(f"k={self.n_components}")
        if self.codec_class.uses_phase:
            parts.append(f"stride={self.phase_stride}")
        return " ".join(parts)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Candidate":
        return cls(**{f: data[f] for f in cls.__dataclass_fields__ if f in data})


@dataclass
class SpectralModel:
    """A fitted sound: the candidate, the arrays it implies, and the audio they make."""

    candidate: Candidate
    sample_rate: int
    n_samples: int
    n_frames: int
    bins: np.ndarray
    arrays: dict[str, np.ndarray] = field(default_factory=dict)

    # -- playback -------------------------------------------------------------

    def components(self) -> np.ndarray:
        """The decoded component block: k slowly varying complex amplitudes."""
        return self.candidate.codec_class.decode(self.arrays, self.n_frames)

    def magnitudes(self) -> np.ndarray:
        """Amplitude envelope per kept bin, one value per frame."""
        return np.abs(self.components())

    def spectrogram(self) -> np.ndarray:
        """The full complex spectrogram the model stands for — zero off the kept bins."""
        spec = self.candidate.spec
        full = np.zeros((spec.n_bins, self.n_frames), dtype=np.complex128)
        full[self.bins] = from_components(self.components(), self.bins, spec)
        return full

    def render(self) -> np.ndarray:
        """Synthesize the audio, at the original length and level."""
        return synthesize(self.spectrogram(), self.candidate.spec, self.n_samples)

    def frequencies(self) -> np.ndarray:
        """Centre frequency of each kept bin, in Hz."""
        return self.candidate.spec.bin_frequencies(self.sample_rate)[self.bins]

    # -- size -----------------------------------------------------------------

    @property
    def n_scalars(self) -> int:
        """Stored numbers, counted from the arrays actually held."""
        return sum(
            int(np.size(a)) * (2 if np.iscomplexobj(a) else 1)
            for a in self._stored().values()
        )

    @property
    def duration(self) -> float:
        return self.n_samples / float(self.sample_rate)

    def n_bytes(self) -> int:
        """Size of the compressed file this model writes, in bytes."""
        return len(self.to_bytes())

    def compression_ratio(self) -> float:
        """Model bytes against 16-bit PCM of the same audio."""
        return self.n_bytes() / (2.0 * self.n_samples)

    # -- persistence ----------------------------------------------------------

    def metadata(self) -> dict:
        return {
            "format": FORMAT,
            "candidate": self.candidate.to_dict(),
            "sample_rate": self.sample_rate,
            "n_samples": self.n_samples,
            "n_frames": self.n_frames,
            "duration": self.duration,
            "n_components": int(self.bins.size),
            "n_scalars": self.n_scalars,
            "arrays": sorted(self._stored()),
        }

    def _stored(self) -> dict[str, np.ndarray]:
        return {"bins": self.bins, **{f"codec__{k}": v for k, v in self.arrays.items()}}

    def _save_to(self, target) -> None:
        payload = {k: np.asarray(v) for k, v in self._stored().items()}
        payload["meta"] = np.asarray(json.dumps(self.metadata()))
        np.savez_compressed(target, **payload)

    def to_bytes(self) -> bytes:
        """The .npz this model would save, in memory."""
        import io

        buffer = io.BytesIO()
        self._save_to(buffer)
        return buffer.getvalue()

    def save(self, path: str | Path) -> Path:
        """Write the whole model — arrays and metadata — to one .npz."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            self._save_to(handle)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SpectralModel":
        with np.load(Path(path), allow_pickle=False) as data:
            meta = json.loads(str(data["meta"].item()))
            if meta.get("format") != FORMAT:
                raise ValueError(
                    f"{path}: expected format {FORMAT!r}, found {meta.get('format')!r}"
                )
            return cls(
                candidate=Candidate.from_dict(meta["candidate"]),
                sample_rate=int(meta["sample_rate"]),
                n_samples=int(meta["n_samples"]),
                n_frames=int(meta["n_frames"]),
                bins=data["bins"],
                arrays={
                    key[len("codec__") :]: data[key]
                    for key in data.files
                    if key.startswith("codec__")
                },
            )
