"""Storing the phase of a component block, at or below the frame rate.

The phase handled here is already drift-compensated (see
:mod:`drumsynth.spectral.components`), so it is the slowly turning part: its
slope is a partial's detuning from its bin centre, in radians per frame, not
the partial's absolute frequency. That is what makes it worth unwrapping and,
sometimes, worth sampling at less than one value per frame.

Sometimes. Interpolated phase costs quality steeply — every stride above 1
measured in this project lands near 1e-2 relative error no matter how much is
spent on it — so a fit with a tight target will refuse the option. It stays in
the search because for noisy sounds in the lossy regime it does win, and
because a knob that is measured and rejected is worth more than a knob that
was never tried.
"""

from __future__ import annotations

import numpy as np


def stored_frames(n_frames: int, stride: int) -> np.ndarray:
    """Frame indices kept at `stride` spacing, always including the last frame."""
    if stride < 1:
        raise ValueError(f"phase stride must be at least 1, got {stride}")

    indices = np.arange(0, n_frames, stride, dtype=np.int32)
    if indices[-1] != n_frames - 1:
        indices = np.append(indices, np.int32(n_frames - 1))
    return indices


def unwrap(phase: np.ndarray) -> np.ndarray:
    """Undo the ±π wrapping along time, so the trajectory can be interpolated."""
    return np.unwrap(np.asarray(phase, dtype=np.float64), axis=1)


def sample(unwrapped: np.ndarray, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """(kept frame indices, unwrapped phase at those frames)."""
    indices = stored_frames(unwrapped.shape[1], stride)
    return indices, unwrapped[:, indices].astype(np.float32)


def decode(indices: np.ndarray, values: np.ndarray, n_frames: int) -> np.ndarray:
    """Linearly interpolate the kept phase back to every frame."""
    kept = np.asarray(indices, dtype=np.float64)
    stored = np.asarray(values, dtype=np.float64)
    if kept.size == n_frames:
        return stored

    targets = np.arange(n_frames, dtype=np.float64)
    phase = np.empty((stored.shape[0], n_frames), dtype=np.float64)
    for component in range(stored.shape[0]):
        phase[component] = np.interp(targets, kept, stored[component])
    return phase


def n_scalars(n_components: int, n_frames: int, stride: int) -> int:
    """Stored phase numbers: one per component per kept frame, plus the index list."""
    kept = int(stored_frames(n_frames, stride).size)
    return n_components * kept + kept
