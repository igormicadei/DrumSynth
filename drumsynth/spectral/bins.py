"""Choosing which FFT bins the model keeps.

Bins are ranked by the energy they carry over the whole file and the top `k`
are kept. Energy, not summed magnitude: the objective the fit is measured
against is a waveform mean-squared error, and by Parseval the bins that
dominate that error are the ones with the most energy in them.

Nothing here looks for peaks. A windowed partial spreads over several bins,
and its skirts carry real energy — dropping them to keep only the centre bin
puts the error back in as a ringing artefact. Ranking by energy keeps the
skirts of a loud partial ahead of the centre of a quiet one, which is the
right order for this objective.
"""

from __future__ import annotations

import numpy as np


def bin_energy(spectrogram: np.ndarray) -> np.ndarray:
    """Total energy per bin across all frames."""
    return np.sum(np.abs(spectrogram) ** 2, axis=1)


def energy_order(spectrogram: np.ndarray) -> np.ndarray:
    """Bin indices, most energetic first."""
    return np.argsort(bin_energy(spectrogram), kind="stable")[::-1]


def select_bins(spectrogram: np.ndarray, n_components: int) -> np.ndarray:
    """The `n_components` most energetic bins, in ascending bin order."""
    n_bins = spectrogram.shape[0]
    keep = int(np.clip(n_components, 1, n_bins))
    return np.sort(energy_order(spectrogram)[:keep]).astype(np.int32)
