"""Where a search does its arithmetic: numpy, or a GPU.

A search spends nearly all of itself in one operation — take a candidate's
component block, put it back into a spectrogram, inverse transform it, overlap
and add, and compare the result against a recording. Tens of thousands of
times. This module is that operation, twice:

    numpy   float64, threads over the batch. Exact: bit for bit what
            `model.render()` produces, which is what makes a searched number
            and a reported number the same number.

    torch   float32 by default, the whole batch as one set of tensors on
            whatever device torch is pointed at. On a GPU that is a batched
            cuFFT, which is the shape of work a GPU is actually good at.

The two agree to about 1e-9 relative — nine orders below the tightest target
anyone sets — and the model that wins a search is always rebuilt and
re-measured in float64 before anything is reported, so the device a search ran
on never reaches the numbers a run is judged by.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field as dataclass_field
from typing import Sequence

import numpy as np

from .parallel import map_workers, resolve_jobs
from .spectral.components import drift
from .spectral.stft import StftSpec, _window_power, synthesize_frames

#: What one batch of renders may occupy, in bytes, before it is split up.
DEFAULT_BUDGET = 512 * 1024 * 1024

#: Columns of what `measure` returns, per render.
STATS = ("relative_mse", "correlation", "rms", "peak")


def available_devices() -> list[str]:
    """Which backends this machine can actually run."""
    devices = ["numpy"]
    try:
        import torch
    except ImportError:
        return devices

    devices.append("cpu")
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


def describe_device(device: str) -> str:
    """One line about what a device is, for a report or a log."""
    if device == "numpy":
        return "numpy, float64"
    try:
        import torch
    except ImportError:  # pragma: no cover - guarded by available_devices
        return f"{device} (torch is not installed)"

    if device.startswith("cuda") and torch.cuda.is_available():
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        return (
            f"{properties.name}, {properties.total_memory / 1e9:.1f} GB, "
            f"{properties.multi_processor_count} SMs, torch {torch.__version__}"
        )
    return f"torch {torch.__version__} on {device}, float32"


@dataclass
class Renderer:
    """Renders batches of component blocks and measures them against recordings.

    `references` is one signal per probe: a single hit has one, a velocity
    model has one per velocity the search listens at. A batch is shaped
    (candidates, probes, bins kept, frames) and comes back (candidates, probes,
    4) — relative error, correlation, rms and peak of each render.
    """

    spec: StftSpec
    bins: np.ndarray
    n_samples: int
    references: np.ndarray
    jobs: int = 1
    budget: int = DEFAULT_BUDGET

    device = "numpy"

    def __post_init__(self) -> None:
        self.references = np.atleast_2d(np.asarray(self.references, dtype=np.float64))
        self.bins = np.asarray(self.bins, dtype=np.int64)
        self._rotation = np.exp(1j * drift(self.bins, self.spec, self.n_frames))
        self._energy = np.sum(self.references**2, axis=1)
        self._local = threading.local()

    @property
    def n_frames(self) -> int:
        return self.spec.n_frames(self.n_samples)

    @property
    def n_probes(self) -> int:
        return int(self.references.shape[0])

    @property
    def batch(self) -> int:
        """How many candidates to hand over at once, given the memory budget."""
        per_render = self.spec.n_bins * self.n_frames * 16 + self.spec.n_fft * self.n_frames * 8
        return max(1, int(self.budget / max(per_render * self.n_probes, 1)))

    # -- the operation --------------------------------------------------------

    def measure(self, blocks: np.ndarray) -> np.ndarray:
        """(candidates, probes, bins, frames) in, (candidates, probes, 4) out."""
        blocks = np.asarray(blocks)
        shape = blocks.shape[:2]
        flat = blocks.reshape(-1, *blocks.shape[2:])
        probes = np.tile(np.arange(shape[1]), shape[0])

        stats = map_workers(
            lambda pair: self._one(pair[0], pair[1]), zip(flat, probes), self.jobs
        )
        return np.asarray(stats, dtype=np.float64).reshape(*shape, len(STATS))

    def render(self, block: np.ndarray) -> np.ndarray:
        """One block to audio — the same samples `model.render()` gives.

        Built frames-major into a buffer this thread keeps: only the kept bins
        are ever written, and they are written every time, so the rest of it
        stays zero without being cleared again.
        """
        spectrogram = self._buffer()
        spectrogram[:, self.bins] = (np.asarray(block) * self._rotation).T
        return synthesize_frames(spectrogram, self.spec, self.n_samples)

    def _buffer(self) -> np.ndarray:
        held = getattr(self._local, "spectrogram", None)
        if held is None:
            held = np.zeros((self.n_frames, self.spec.n_bins), dtype=np.complex128)
            self._local.spectrogram = held
        return held

    def _one(self, block: np.ndarray, probe: int) -> tuple[float, ...]:
        estimate = self.render(block)
        reference = self.references[probe]
        residual = reference - estimate

        energy = self._energy[probe]
        error = float(np.sum(residual**2) / energy) if energy > 0 else 0.0
        return (
            error,
            _correlation(reference, estimate),
            float(np.sqrt(np.mean(estimate**2))),
            float(np.max(np.abs(estimate))) if estimate.size else 0.0,
        )


def _correlation(reference: np.ndarray, estimate: np.ndarray) -> float:
    if reference.size < 2 or np.std(reference) <= 0.0 or np.std(estimate) <= 0.0:
        return 0.0
    return float(np.corrcoef(reference, estimate)[0, 1])


@dataclass
class TorchRenderer(Renderer):
    """The same operation, batched onto a torch device.

    Everything that does not change between candidates — the window, the
    per-bin rotation, the overlap-add normalization, the recordings — is moved
    to the device once and stays there. What crosses per batch is the
    candidates' component blocks going out and four numbers per render coming
    back.
    """

    device: str = "cuda"
    dtype: str = "float32"
    _state: dict = dataclass_field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        import torch

        self._torch = torch
        self._device = torch.device(self.device)
        if self._device.type == "cpu":
            # torch parallelizes inside its own operators; threading the caller
            # as well just oversubscribes the cores.
            torch.set_num_threads(resolve_jobs(self.jobs))
        real = torch.float32 if self.dtype == "float32" else torch.float64
        complex_dtype = torch.complex64 if self.dtype == "float32" else torch.complex128

        def send(array, dtype):
            # A copy, because torch will not take a read-only numpy array.
            return torch.as_tensor(
                np.array(array, copy=True), dtype=dtype, device=self._device
            )

        self._state = {
            "real": real,
            "complex": complex_dtype,
            "rotation": send(self._rotation.T, complex_dtype),  # frames-major
            "bins": send(self.bins, torch.long),
            "window": send(self.spec.window, real),
            "power": send(_window_power(self.spec, self.n_frames), real),
            "references": send(self.references, real),
            "energy": send(self._energy, torch.float64),
        }

    @property
    def batch(self) -> int:
        per_render = (
            self.spec.n_bins * self.n_frames * (8 if self.dtype == "float32" else 16)
            + self.spec.n_fft * self.n_frames * (4 if self.dtype == "float32" else 8)
        ) * 3  # the transform needs room for its own working copies
        return max(1, int(self.budget / max(per_render * self.n_probes, 1)))

    def measure(self, blocks: np.ndarray) -> np.ndarray:
        torch = self._torch
        state = self._state
        spec = self.spec

        blocks = np.asarray(blocks)
        candidates, probes = blocks.shape[:2]
        renders = candidates * probes

        # Frames-major throughout: the inverse transform runs along the last
        # axis, and handing it a transposed view instead costs three times as
        # much in copies as the transform itself does.
        flat = torch.as_tensor(
            np.ascontiguousarray(
                blocks.reshape(renders, *blocks.shape[2:]).transpose(0, 2, 1)
            ),
            dtype=state["complex"],
            device=self._device,
        )

        spectrogram = self._spectrogram(renders)
        spectrogram[:, :, state["bins"]] = flat * state["rotation"]

        frames = torch.fft.irfft(spectrogram, n=spec.n_fft, dim=-1)
        frames *= state["window"]
        signal = _overlap_add_torch(frames, spec, torch)

        estimate = torch.where(
            state["power"] > 1e-12, signal / state["power"], torch.zeros_like(signal)
        )[:, spec.pad : spec.pad + self.n_samples]

        reference = state["references"].repeat(candidates, 1)
        energy = state["energy"].repeat(candidates)
        stats = _stats_torch(reference, estimate, energy, torch)

        return stats.reshape(candidates, probes, len(STATS)).cpu().numpy()

    def _spectrogram(self, renders: int):
        """A zeroed (renders, frames, bins) buffer, kept between batches.

        Only the kept bins are ever written, and they are written every time,
        so the buffer stays correct without being cleared again.
        """
        torch = self._torch
        held = self._state.get("spectrogram")
        if held is None or held.shape[0] < renders:
            held = torch.zeros(
                (renders, self.n_frames, self.spec.n_bins),
                dtype=self._state["complex"],
                device=self._device,
            )
            self._state["spectrogram"] = held
        return held[:renders]

    def render(self, block: np.ndarray) -> np.ndarray:
        """Kept in float64 on the host: one render is not what a device is for."""
        return Renderer.render(self, block)


def _overlap_add_torch(frames, spec: StftSpec, torch):
    """The same block-shifted add the numpy path uses, one batch at a time."""
    batch, n_frames, n_fft = frames.shape
    overlap = spec.n_fft // spec.hop

    pieces = frames.reshape(batch, n_frames, overlap, spec.hop)
    out = torch.zeros(
        (batch, n_frames + overlap - 1, spec.hop), dtype=frames.dtype, device=frames.device
    )
    for shift in range(overlap):
        out[:, shift : shift + n_frames] += pieces[:, :, shift, :]
    return out.reshape(batch, -1)


def _stats_torch(reference, estimate, energy, torch):
    """Relative error, correlation, rms and peak, without leaving the device."""
    residual = reference - estimate
    error = torch.sum(residual * residual, dim=1, dtype=torch.float64) / torch.clamp(
        energy, min=1e-300
    )
    error = torch.where(energy > 0, error, torch.zeros_like(error))

    n = estimate.shape[1]
    reference64 = reference.to(torch.float64)
    estimate64 = estimate.to(torch.float64)
    left = reference64 - reference64.mean(dim=1, keepdim=True)
    right = estimate64 - estimate64.mean(dim=1, keepdim=True)
    denominator = torch.sqrt(torch.sum(left * left, dim=1) * torch.sum(right * right, dim=1))
    correlation = torch.where(
        denominator > 0, torch.sum(left * right, dim=1) / denominator, torch.zeros_like(denominator)
    )

    rms = torch.sqrt(torch.sum(estimate64 * estimate64, dim=1) / n)
    peak = torch.amax(torch.abs(estimate64), dim=1)
    return torch.stack([error, correlation, rms, peak], dim=1)


def renderer(
    spec: StftSpec,
    bins: np.ndarray,
    n_samples: int,
    references: np.ndarray,
    *,
    device: str = "numpy",
    jobs: int = 1,
    budget: int = DEFAULT_BUDGET,
) -> Renderer:
    """A renderer on the requested device, falling back to numpy with a reason.

    `auto` takes a GPU if there is one and numpy otherwise — never torch on the
    CPU, which is faster but not exact, and a fallback should not quietly
    change what a search measures.
    """
    if device == "auto":
        device = "cuda" if "cuda" in available_devices() else "numpy"

    if device == "numpy":
        return Renderer(spec, bins, n_samples, references, jobs=jobs, budget=budget)

    if device not in available_devices():
        raise ValueError(
            f"device {device!r} is not available here (have: {available_devices()}). "
            "Install torch for 'cpu', and a CUDA build of it for 'cuda'."
        )

    return TorchRenderer(
        spec, bins, n_samples, references, jobs=jobs, budget=budget, device=device
    )


def benchmark_devices(
    devices: Sequence[str] | None = None, *, jobs: int = 0, repeats: int = 3
) -> dict[str, float]:
    """Time each device on the same batch of renders. Milliseconds per render.

    The work is a drum-sized problem — 3.5 seconds at 44.1 kHz, 256 bins kept,
    a 2048-point transform — so the answer is about the machine rather than
    about a toy. Run it before committing an afternoon to a search.
    """
    import time

    sample_rate, seconds, k, n_fft = 44100, 3.5, 256, 2048
    spec = StftSpec(n_fft, n_fft // 2)
    n_samples = int(seconds * sample_rate)
    n_frames = spec.n_frames(n_samples)

    rng = np.random.default_rng(0)
    bins = np.sort(rng.choice(spec.n_bins, k, replace=False))
    reference = rng.standard_normal(n_samples)
    blocks = (rng.standard_normal((24, 1, k, n_frames)) + 1j * rng.standard_normal((24, 1, k, n_frames)))

    timings: dict[str, float] = {}
    for device in devices or available_devices():
        engine = renderer(spec, bins, n_samples, reference[None, :], device=device, jobs=jobs)
        batch = blocks[: max(1, min(engine.batch, blocks.shape[0]))]

        engine.measure(batch)
        started = time.perf_counter()
        for _ in range(repeats):
            engine.measure(batch)
        elapsed = (time.perf_counter() - started) / repeats
        timings[device] = elapsed * 1000.0 / batch.shape[0]
    return timings
