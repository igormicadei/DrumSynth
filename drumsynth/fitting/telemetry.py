"""What the GPU is actually doing, read from NVML.

`torch.cuda.is_available()` says a device exists. It does not say the fit is
using it, and it says nothing at all about whether the device is busy — which
is the question someone asks when they have installed a CUDA wheel, started a
run, and watched the GPU sit at 0%.

Two different measurements answer that, and both are here:

* **The device**, from NVML: name, utilization, VRAM, temperature, power. This
  is the whole GPU, every process on it, sampled live.
* **The process**, from torch: how much VRAM this fit has ever allocated. Zero
  means stage 5 never ran on CUDA, whatever the device picker said.

Neither is decoration. A run where the device reads 0% and the process
allocated 300 MB is a run that used the GPU and finished the GPU part quickly —
stages 1 to 4 are CPU work and usually most of the wall clock. A run where the
process allocated nothing used the CPU, and the picker or the wheel is why.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class GpuSample:
    """One reading of one device."""

    index: int
    name: str
    utilization: float          # percent, whole device
    memory_used: float          # bytes
    memory_total: float         # bytes
    temperature: float          # celsius, nan when unavailable
    power: float                # watts, nan when unavailable
    at: float = field(default_factory=time.time)
    utilization_stale: bool = False

    @property
    def memory_fraction(self) -> float:
        return self.memory_used / self.memory_total if self.memory_total else 0.0

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "name": self.name,
            "utilization": self.utilization,
            "utilization_stale": self.utilization_stale,
            "memory_used": self.memory_used,
            "memory_total": self.memory_total,
            "memory_fraction": self.memory_fraction,
            "temperature": self.temperature,
            "power": self.power,
            "at": self.at,
        }


class GpuMonitor:
    """NVML, if it is there.

    Every failure mode ends the same way — `available` is False and `reason`
    says why — because there is nothing a caller can usefully do about the
    difference between "pynvml is not installed" and "the driver is not
    loaded", except show the user which one it was.
    """

    def __init__(self) -> None:
        self._nvml: Any = None
        self.reason = ""
        self._last_utilization: dict[int, float] = {}
        self._start()

    def _start(self) -> None:
        try:
            import pynvml
        except ImportError:
            self.reason = ("pynvml is not installed — `pip install nvidia-ml-py` "
                           "for live GPU readings")
            return
        try:
            pynvml.nvmlInit()
        except Exception as error:                     # NVMLError, and friends
            self.reason = f"NVML did not start: {type(error).__name__}"
            return
        self._nvml = pynvml

    @property
    def available(self) -> bool:
        return self._nvml is not None

    @property
    def count(self) -> int:
        if not self.available:
            return 0
        try:
            return int(self._nvml.nvmlDeviceGetCount())
        except Exception:
            return 0

    def sample(self) -> list[GpuSample]:
        """One reading per device. Empty when NVML is not usable."""
        if not self.available:
            return []
        nvml = self._nvml
        out: list[GpuSample] = []
        for index in range(self.count):
            try:
                handle = nvml.nvmlDeviceGetHandleByIndex(index)
                name = nvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):            # older bindings
                    name = name.decode("utf-8", "replace")
                memory = nvml.nvmlDeviceGetMemoryInfo(handle)
                utilization, utilization_stale = self._utilization(nvml, handle, index)
                out.append(
                    GpuSample(
                        index=index,
                        name=str(name),
                        utilization=utilization,
                        memory_used=float(memory.used),
                        memory_total=float(memory.total),
                        temperature=self._optional(
                            nvml.nvmlDeviceGetTemperature,
                            handle,
                            nvml.NVML_TEMPERATURE_GPU,
                        ),
                        power=self._scaled(
                            nvml.nvmlDeviceGetPowerUsage, handle, 1000.0
                        ),
                        utilization_stale=utilization_stale,
                    )
                )
            except Exception:
                # A device that stops answering mid-run is not worth failing a
                # fit over; it is a readout.
                continue
        return out

    def _utilization(self, nvml, handle, index: int) -> tuple[float, bool]:
        """Read utilization without losing the rest of a device sample."""
        try:
            value = float(nvml.nvmlDeviceGetUtilizationRates(handle).gpu)
        except Exception:
            return self._last_utilization.get(index, float("nan")), True
        self._last_utilization[index] = value
        return value, False

    @staticmethod
    def _optional(call, *args) -> float:
        try:
            return float(call(*args))
        except Exception:
            return float("nan")

    @staticmethod
    def _scaled(call, handle, divisor: float) -> float:
        value = GpuMonitor._optional(call, handle)
        return value / divisor if value == value else value      # nan-safe

    def shutdown(self) -> None:
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml = None


class TorchMemory:
    """What this process allocated on CUDA, from torch itself.

    The device can read 0% while the fit is genuinely using it — stage 5 is a
    small share of the run — but a peak allocation of zero is unambiguous: no
    tensor ever reached the GPU.
    """

    @staticmethod
    def peak_bytes() -> float:
        try:
            import torch
        except ImportError:
            return 0.0
        if not torch.cuda.is_available():
            return 0.0
        try:
            return float(torch.cuda.max_memory_allocated())
        except Exception:
            return 0.0

    @staticmethod
    def reset() -> None:
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
