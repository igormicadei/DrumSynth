"""Rendering many glides at once.

Stage 2b asks the same question forty-odd times per velocity layer: *if the
tension were this k and this tau, what would the drum sound like?* Each answer
is a full render, and a render with the glide on cannot be a single matrix
product — `ratio = 1 + k * bank_energy` makes the frequency at block N depend on
the state at block N-1, so the engine walks the hit one control block at a time.

Profiled on a 23-mode fit of a real tom, one layer of the grid took 7.7 seconds
and **89% of it was inside `ModalBank._advance`, called 77,549 times**. That is
1,723 calls per render — one per 64-sample control block — each doing a closed
form over a 23 by 64 array. The arithmetic is nothing; the cost is asking numpy
to do it seventy-seven thousand times.

The candidates cannot be made independent of their own history, but they are
independent of *each other*. So they are stepped in lockstep: one array of
states shaped `(candidates, modes)`, one block at a time, every candidate
advancing together. The number of numpy calls stays at 1,723 and each one does
forty-five times the work.

That is the same move stage 5 made, for the same reason, and it has the same
second benefit: a batched block render is a shape a GPU can take. The win on the
CPU comes first and is the larger one here.
"""

from __future__ import annotations

import numpy as np

from ..core.constants import Audio
from ..synth.params import Mode, Tension
from ..synth.resonators import ModeResonator
from .backend import Device, DeviceChoice


class BatchedGlideRender:
    """One modal bank per candidate, all advanced together.

    Every candidate shares the modes and the gains and differs only in its
    tension. Construction sets up the shared per-mode constants once; `render`
    walks the hit and returns `(candidates, samples)`.

    The arithmetic is `ModalBank._advance`'s, generalized by one axis, and the
    tests hold it to matching `DrumVoice.render_hit` sample for sample.
    """

    def __init__(self, modes: list[Mode], gains: np.ndarray,
                 tensions: list[Tension], sr: int = Audio.DEFAULT_SR,
                 control_period: int = 64, device: "Device | None" = None) -> None:
        self.device = device
        self.torch = None
        if device is not None and device.backend == "torch":
            self.torch = DeviceChoice.torch()
        self.sr = int(sr)
        self.control_period = max(1, int(control_period))
        self.n_candidates = len(tensions)
        self.n_modes = len(modes)

        self.f_static = np.array([mode.f_static for mode in modes], dtype=float)
        self.gains = np.asarray(gains, dtype=float)

        # `r` is per mode and does not move with the ratio, so it is computed
        # once. `log(r)` too — it is inside the inner loop otherwise.
        t60 = np.array([mode.t60 for mode in modes], dtype=float)
        self.r = np.exp(-np.log(1000.0) / np.maximum(t60, 1e-9) / self.sr)
        self.log_r = np.log(self.r)

        self.k = np.array([tension.k for tension in tensions], dtype=float)
        self.max_ratio = np.array(
            [tension.max_ratio for tension in tensions], dtype=float)
        self.instant_attack = np.array(
            [bool(tension.instant_attack) for tension in tensions], dtype=bool)
        self.coef = np.array(
            [tension.smoothing_coef(self.sr, self.control_period)
             for tension in tensions], dtype=float)

    # -- the render -----------------------------------------------------------

    def render(self, n_samples: int, amplitude: float = 1.0) -> np.ndarray:
        """`(candidates, n_samples)`. Every candidate struck at the same
        amplitude with the same gains, differing only in its glide."""
        if self.torch is not None:
            return self._render_torch(n_samples, amplitude)
        candidates, modes = self.n_candidates, self.n_modes
        if candidates == 0 or modes == 0 or n_samples <= 0:
            return np.zeros((candidates, max(0, n_samples)))

        # §3.2: a strike adds `amplitude * gain` to x, leaving y alone.
        x = np.tile(self.gains * float(amplitude), (candidates, 1))
        y = np.zeros((candidates, modes))

        smoothed = np.zeros(candidates)
        out = np.zeros((candidates, n_samples))
        limit = ModeResonator.MAX_FREQ_FRACTION * self.sr

        # `decay` depends only on `r` and the offset into the block, so it is
        # the same array for every block and every candidate. Recomputing it in
        # the loop was a third of the transcendental work.
        full = self._decay_table(min(self.control_period, n_samples))

        position = 0
        while position < n_samples:
            span = min(self.control_period, n_samples - position)
            decay = full if span == full.shape[1] else self._decay_table(span)

            # The ratio is read from the state BEFORE anything advances, which
            # is what keeps the feedback loop stable (see voice.py).
            energy = np.sum(x * x + y * y, axis=1)
            rising = self.instant_attack & (energy > smoothed)
            smoothed = np.where(
                rising, energy, smoothed + (energy - smoothed) * self.coef)
            ratio = np.minimum(1.0 + self.k * smoothed, self.max_ratio)

            block, x, y = self._advance(x, y, ratio, span, limit, decay)
            out[:, position : position + span] = block
            position += span
        return out

    def _decay_table(self, n: int) -> np.ndarray:
        """`(modes, n)` of `r ** step`, and the last column separately."""
        steps = np.arange(n, dtype=np.float64)
        return np.exp(self.log_r[:, None] * steps[None, :])

    def _advance(self, x: np.ndarray, y: np.ndarray, ratio: np.ndarray,
                 n: int, limit: float, decay: np.ndarray):
        """`ModalBank._advance`, with a candidate axis in front.

        Shapes: `x`, `y` and every per-candidate-per-mode quantity are
        `(candidates, modes)`; the output block is `(candidates, n)`.
        """
        freqs = np.clip(self.f_static[None, :] * ratio[:, None], 0.0, limit)
        w = 2.0 * np.pi * freqs / self.sr
        eps = 2.0 * np.sin(0.5 * w)
        cos_w, sin_w = np.cos(w), np.sin(w)

        r = self.r[None, :]
        one_minus_eps2 = 1.0 - eps * eps
        x1 = r * (x + eps * y)
        y1 = r * (-eps * x + one_minus_eps2 * y)
        x2 = r * (x1 + eps * y1)
        y2 = r * (-eps * x1 + one_minus_eps2 * y1)

        safe_sin = np.where(np.abs(sin_w) < 1e-12, 1.0, sin_w)
        a_x, a_y = x1, y1
        b_x = (x2 / r - a_x * cos_w) / safe_sin
        b_y = (y2 / r - a_y * cos_w) / safe_sin

        steps = np.arange(n, dtype=np.float64)
        phase = w[:, :, None] * steps[None, None, :]
        block = decay[None, :, :] * (a_x[:, :, None] * np.cos(phase)
                                     + b_x[:, :, None] * np.sin(phase))

        last = float(n - 1)
        decay_last = decay[None, :, -1]
        cos_last, sin_last = np.cos(w * last), np.sin(w * last)
        return (
            block.sum(axis=1),
            decay_last * (a_x * cos_last + b_x * sin_last),
            decay_last * (a_y * cos_last + b_y * sin_last),
        )

    # -- the same walk, on tensors --------------------------------------------

    def _render_torch(self, n_samples: int, amplitude: float = 1.0) -> np.ndarray:
        """The identical arithmetic on a device.

        Worth having and worth being precise about why. The loop is a feedback
        loop and stays sequential — 431 blocks at a 256-sample control period —
        so this is not the embarrassingly-parallel shape stage 5 has. What IS
        parallel is inside each block: measured on a real tom, one layer of the
        grid is 48 candidates by 23 modes by 110250 samples, which is 244
        million cosines and as many sines. That is the work, it does not shrink
        with the block size (measured: identical time at 64 and 512), and it is
        exactly what a GPU is for.

        Before the batching there was nothing here a GPU could take at all: 45
        separate sequential renders, each one 23 by 64 at a time.
        """
        torch = self.torch
        device = torch.device(self.device.kind)
        dtype = torch.float32 if self.device.is_cuda else torch.float64
        candidates, modes = self.n_candidates, self.n_modes
        if candidates == 0 or modes == 0 or n_samples <= 0:
            return np.zeros((candidates, max(0, n_samples)))

        def send(array):
            return torch.as_tensor(np.asarray(array), dtype=dtype, device=device)

        f_static, r, log_r = send(self.f_static), send(self.r), send(self.log_r)
        k, max_ratio, coef = send(self.k), send(self.max_ratio), send(self.coef)
        instant = torch.as_tensor(self.instant_attack, device=device)

        with torch.no_grad():
            x = send(self.gains * float(amplitude)).repeat(candidates, 1)
            y = torch.zeros((candidates, modes), dtype=dtype, device=device)
            smoothed = torch.zeros(candidates, dtype=dtype, device=device)
            out = torch.zeros((candidates, n_samples), dtype=dtype, device=device)
            limit = ModeResonator.MAX_FREQ_FRACTION * self.sr

            position = 0
            while position < n_samples:
                span = min(self.control_period, n_samples - position)
                steps = torch.arange(span, dtype=dtype, device=device)
                decay = torch.exp(log_r[:, None] * steps[None, :])

                energy = torch.sum(x * x + y * y, dim=1)
                rising = instant & (energy > smoothed)
                smoothed = torch.where(
                    rising, energy, smoothed + (energy - smoothed) * coef)
                ratio = torch.minimum(1.0 + k * smoothed, max_ratio)

                w = (2.0 * np.pi
                     * torch.clamp(f_static[None, :] * ratio[:, None], 0.0, limit)
                     / self.sr)
                eps = 2.0 * torch.sin(0.5 * w)
                cos_w, sin_w = torch.cos(w), torch.sin(w)

                row = r[None, :]
                one_minus = 1.0 - eps * eps
                x1 = row * (x + eps * y)
                y1 = row * (-eps * x + one_minus * y)
                x2 = row * (x1 + eps * y1)
                y2 = row * (-eps * x1 + one_minus * y1)

                safe = torch.where(torch.abs(sin_w) < 1e-12,
                                   torch.ones_like(sin_w), sin_w)
                b_x = (x2 / row - x1 * cos_w) / safe
                b_y = (y2 / row - y1 * cos_w) / safe

                phase = w[:, :, None] * steps[None, None, :]
                block = decay[None, :, :] * (x1[:, :, None] * torch.cos(phase)
                                             + b_x[:, :, None] * torch.sin(phase))
                out[:, position : position + span] = block.sum(dim=1)

                last = float(span - 1)
                decay_last = decay[None, :, -1]
                cos_last, sin_last = torch.cos(w * last), torch.sin(w * last)
                x = decay_last * (x1 * cos_last + b_x * sin_last)
                y = decay_last * (y1 * cos_last + b_y * sin_last)
                position += span

            return out.double().cpu().numpy()
