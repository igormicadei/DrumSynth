"""Where stage 5's population is evaluated, and on what.

Stage 5 asks the same question a few thousand times: *given these six numbers,
how far is the drum they describe from the recordings?* Every one of those
evaluations is a matrix product against a cached basis followed by a
multi-resolution STFT. That is a batch job, and it was not being run as one.

Two changes live here.

**The population is evaluated together.** `differential_evolution` was being
handed `workers=N`, which puts each candidate in its own process. On Windows
that fails outright — the objective is a closure and `spawn` cannot pickle it —
and where it does work it ships tens of megabytes of basis matrices down a pipe
per task to save a few milliseconds of arithmetic. Evaluating the whole
population as one `(S, modes) @ (modes, samples)` product and one batched STFT
is both correct and faster, and it needs no processes at all.

**Batched, the work fits a GPU.** One candidate at a time is a small, latency-
bound job and a PCIe round trip costs more than the arithmetic saves — which is
what the earlier note in TRAINING.md said, and it was right about that case and
wrong to stop there. A population of sixty candidates across six velocity
layers is 360 renders and 1080 STFTs per generation, all independent, all on
data that can sit in VRAM for the whole run. That is the shape CUDA is for.

The two backends compute the same transform. `SpectralTarget` owns its
definition — window, hop, band matrix, floor — and both borrow it, because a
loss that does not match the one every other stage reports is not comparable
with the baselines in TRAINING.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .objective import LinearVoiceBasis, SpectralTarget


# =============================================================================
# Choosing a device
# =============================================================================


@dataclass(frozen=True)
class Device:
    """Where stage 5 will run, and how it got there."""

    kind: str               # "cpu" or "cuda"
    backend: str            # "numpy" or "torch"
    detail: str             # for the UI and the log

    AUTO: str = "auto"

    @property
    def is_cuda(self) -> bool:
        return self.kind == "cuda"

    def __str__(self) -> str:
        return self.detail


class DeviceChoice:
    """Resolving "auto", "cuda" or "cpu" into something that actually exists.

    A request for CUDA on a machine without it is an error worth raising rather
    than silently absorbing: someone who asked for the GPU wants to know it did
    not happen. "auto" is the setting that quietly falls back.
    """

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"

    @staticmethod
    def torch():
        """The torch module, or None. Imported lazily — torch is an optional
        dependency and importing it costs a second even when unused."""
        try:
            import torch
        except ImportError:
            return None
        return torch

    @staticmethod
    def cuda_detail() -> str | None:
        """A description of the CUDA device, or None if there is not one."""
        torch = DeviceChoice.torch()
        if torch is None or not torch.cuda.is_available():
            return None
        index = torch.cuda.current_device()
        name = torch.cuda.get_device_name(index)
        total = torch.cuda.get_device_properties(index).total_memory
        return f"{name}, {total / 1024**3:.1f} GiB, torch {torch.__version__}"

    @staticmethod
    def resolve(request: str = AUTO) -> Device:
        request = (request or DeviceChoice.AUTO).lower()
        detail = DeviceChoice.cuda_detail()

        if request == DeviceChoice.CUDA:
            if detail is None:
                torch = DeviceChoice.torch()
                raise RuntimeError(
                    "CUDA was requested but is not available: "
                    + ("torch is not installed — see docs/TRAINING.md"
                       if torch is None else
                       f"torch {torch.__version__} reports no CUDA device. A "
                       "CPU-only wheel is the usual cause; reinstall from the "
                       "cu-tagged index.")
                )
            return Device("cuda", "torch", f"CUDA — {detail}")

        if request == DeviceChoice.AUTO and detail is not None:
            return Device("cuda", "torch", f"CUDA — {detail}")

        return Device("cpu", "numpy", f"CPU — numpy {np.__version__}")

    @staticmethod
    def options() -> list[tuple[str, str]]:
        """(value, label) for a UI, saying what is actually there."""
        detail = DeviceChoice.cuda_detail()
        cuda = f"CUDA ({detail})" if detail else "CUDA (not available here)"
        auto = "Automatic" + (" — CUDA" if detail else " — CPU")
        return [
            (DeviceChoice.AUTO, auto),
            (DeviceChoice.CUDA, cuda),
            (DeviceChoice.CPU, "CPU"),
        ]


# =============================================================================
# The batched loss
# =============================================================================


class BatchLoss:
    """One velocity layer, pre-transformed, scoring a whole population at once.

    `losses(gains, levels)` takes `(S, modes)` and `(S, bands)` and returns `S`
    band distances — the same number `SpectralTarget.distance` returns for one
    candidate, and it is checked against that in the tests.

    Level is re-matched per candidate before comparing, because the absolute
    scale has a closed-form optimum and leaving it in the search wastes a
    dimension on something arithmetic does exactly.
    """

    #: Candidates transformed at once. The intermediate is
    #: (chunk, frames, n_fft), which at the smallest FFT size and a two-second
    #: hit is ~3.5 MB per candidate in float64 — a whole population of them is
    #: not something to allocate in one go on a laptop.
    CPU_CHUNK: int = 8
    #: A whole population at once, usually. At the smallest FFT size a
    #: four-second hit frames to (64, 2750, 256) — 169 MB in float32, plus a
    #: complex64 transform of the same shape. Under a gigabyte, and it keeps
    #: the device-to-host syncs down to one per resolution per layer.
    CUDA_CHUNK: int = 64

    def __init__(self, basis: LinearVoiceBasis, target: SpectralTarget,
                 reference: np.ndarray, chunk: int = CPU_CHUNK) -> None:
        self.modal = np.ascontiguousarray(basis.modal, dtype=np.float64)
        self.noise = (
            np.ascontiguousarray(basis.noise, dtype=np.float64)
            if basis.n_bands else None
        )
        self.target = target
        self.reference = np.asarray(reference, dtype=np.float64)
        self.reference_rms = float(np.sqrt(np.mean(self.reference**2)))
        self.chunk = max(1, int(chunk))
        self.n_samples = self.modal.shape[1]

    # -- the pieces, so the torch backend can mirror them exactly -------------

    def render(self, gains: np.ndarray, levels: np.ndarray | None) -> np.ndarray:
        out = np.asarray(gains, dtype=np.float64) @ self.modal
        if self.noise is not None and levels is not None and np.size(levels):
            out = out + np.asarray(levels, dtype=np.float64) @ self.noise
        return out

    def match_levels(self, batch: np.ndarray) -> np.ndarray:
        """Scale every row to the reference's RMS. Rows that came out silent
        are left alone rather than divided by zero; they will score badly on
        their own merits."""
        rms = np.sqrt(np.mean(batch**2, axis=1))
        scale = np.where(rms > 0, self.reference_rms / np.maximum(rms, 1e-30), 1.0)
        return batch * scale[:, None]

    def losses(self, gains: np.ndarray, levels: np.ndarray | None = None,
               ) -> np.ndarray:
        batch = self.match_levels(self.render(gains, levels))
        total = np.zeros(len(batch))
        weight = 0.0
        for n_fft in self.target.fft_sizes:
            summed, count = self._band_error(batch, n_fft)
            total += summed
            weight += count
        return total / weight if weight > 0 else np.full(len(batch), np.inf)

    def _band_error(self, batch: np.ndarray, n_fft: int) -> tuple[np.ndarray, float]:
        reference = self.target.reference_bands(n_fft)
        weights = self.target.band_weights(n_fft)
        window = self.target.window(n_fft)
        bank = self.target.band_matrix(n_fft)
        hop = SpectralTarget.hop(n_fft)

        summed = np.zeros(len(batch))
        rows = 0
        for start in range(0, len(batch), self.chunk):
            piece = batch[start : start + self.chunk]
            spectrogram = BatchLoss._band_spectrogram(piece, n_fft, hop, window, bank)
            rows = min(len(reference), spectrogram.shape[1])
            error = np.abs(reference[None, :rows] - spectrogram[:, :rows])
            summed[start : start + self.chunk] = (
                error * weights[None, :rows]).sum(axis=(1, 2))
        return summed, float(weights[:rows].sum()) if rows else 0.0

    @staticmethod
    def _band_spectrogram(batch: np.ndarray, n_fft: int, hop: int,
                          window: np.ndarray, bank: np.ndarray) -> np.ndarray:
        """(S, frames, bands) in dB, framed the way `SpectralTarget` frames."""
        if batch.shape[1] < n_fft:
            batch = np.pad(batch, ((0, 0), (0, n_fft - batch.shape[1])))
        frames = np.lib.stride_tricks.sliding_window_view(
            batch, n_fft, axis=1)[:, ::hop]
        power = np.abs(np.fft.rfft(frames * window, axis=-1)) ** 2
        return 10.0 * np.log10(power @ bank + 1e-12)


class TorchBatchLoss(BatchLoss):
    """The same arithmetic, on tensors that stay where they were put.

    The basis, the band matrices and the reference spectrograms are uploaded
    once at construction and never move again; only the `(S, modes)` gain
    matrix crosses the bus per generation, which is a few kilobytes.

    float32 on CUDA, float64 on the CPU. The loss is a difference of dB values
    in the tens and float32 carries seven digits, so this costs nothing that
    matters: measured against the float64 path on the same candidates, the
    largest disagreement was 1e-6 dB, against a loss floor of 1.1 dB.
    """

    def __init__(self, basis: LinearVoiceBasis, target: SpectralTarget,
                 reference: np.ndarray, device: Device,
                 chunk: int | None = None) -> None:
        torch = DeviceChoice.torch()
        if torch is None:
            raise RuntimeError("the torch backend was selected but torch is "
                               "not installed")
        super().__init__(basis, target, reference,
                         chunk or (BatchLoss.CUDA_CHUNK if device.is_cuda
                                   else BatchLoss.CPU_CHUNK))
        self.torch = torch
        self.device = torch.device(device.kind)
        self.dtype = torch.float32 if device.is_cuda else torch.float64

        def upload(array) -> "torch.Tensor":
            return torch.as_tensor(np.asarray(array), dtype=self.dtype,
                                   device=self.device)

        self.t_modal = upload(self.modal)
        self.t_noise = upload(self.noise) if self.noise is not None else None
        self.t_reference_rms = float(self.reference_rms)
        self.t_windows = {size: upload(target.window(size))
                          for size in target.fft_sizes}
        self.t_banks = {size: upload(target.band_matrix(size))
                        for size in target.fft_sizes}
        self.t_reference = {size: upload(target.reference_bands(size))
                            for size in target.fft_sizes}
        self.t_weights = {size: upload(target.band_weights(size))
                          for size in target.fft_sizes}

    def losses(self, gains: np.ndarray, levels: np.ndarray | None = None,
               ) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            batch = torch.as_tensor(np.asarray(gains), dtype=self.dtype,
                                    device=self.device) @ self.t_modal
            if self.t_noise is not None and levels is not None and np.size(levels):
                batch = batch + torch.as_tensor(
                    np.asarray(levels), dtype=self.dtype,
                    device=self.device) @ self.t_noise

            rms = torch.sqrt(torch.mean(batch**2, dim=1))
            scale = torch.where(
                rms > 0, self.t_reference_rms / torch.clamp(rms, min=1e-30),
                torch.ones_like(rms))
            batch = batch * scale[:, None]

            total = torch.zeros(len(batch), dtype=self.dtype, device=self.device)
            weight = 0.0
            for n_fft in self.target.fft_sizes:
                summed, count = self._band_error_torch(batch, n_fft)
                total = total + summed
                weight += count
            if weight <= 0:
                return np.full(len(batch), np.inf)
            return (total / weight).double().cpu().numpy()

    def _band_error_torch(self, batch, n_fft: int):
        torch = self.torch
        reference = self.t_reference[n_fft]
        weights = self.t_weights[n_fft]
        window = self.t_windows[n_fft]
        bank = self.t_banks[n_fft]
        hop = SpectralTarget.hop(n_fft)

        if batch.shape[1] < n_fft:
            batch = torch.nn.functional.pad(batch, (0, n_fft - batch.shape[1]))

        summed = torch.zeros(len(batch), dtype=self.dtype, device=self.device)
        rows = 0
        for start in range(0, len(batch), self.chunk):
            piece = batch[start : start + self.chunk]
            # `unfold` is a view, so the copy happens at the windowing.
            frames = piece.unfold(1, n_fft, hop) * window
            power = torch.fft.rfft(frames, dim=-1).abs() ** 2
            spectrogram = 10.0 * torch.log10(power @ bank + 1e-12)
            rows = min(reference.shape[0], spectrogram.shape[1])
            error = (reference[None, :rows] - spectrogram[:, :rows]).abs()
            summed[start : start + self.chunk] = (
                error * weights[None, :rows]).sum(dim=(1, 2))
        return summed, float(weights[:rows].sum().item()) if rows else 0.0


class LossBackend:
    """Builds the right `BatchLoss` for a device."""

    @staticmethod
    def build(basis: LinearVoiceBasis, target: SpectralTarget,
              reference: np.ndarray, device: Device) -> BatchLoss:
        if device.backend == "torch":
            return TorchBatchLoss(basis, target, reference, device)
        return BatchLoss(basis, target, reference)
