"""The fit: search the space of representations, measure every one, choose.

    drumsynth.fitting.metrics   what "close enough" means, numerically
    drumsynth.fitting.search    the candidate grid and the measured search
    drumsynth.fitting.pareto    the size/error frontier and the choice policy
    drumsynth.fitting.report    writing a run to disk
"""

from .metrics import Quality, correlation, peak, relative_mse, rms, snr_db
from .pareto import choose_best, pareto_frontier, preferred
from .report import plot_frontier, report_dict, save_fit
from .search import (
    DEFAULT_TARGET_MSE,
    Evaluation,
    FitResult,
    Progress,
    SearchSpace,
    fit,
)

__all__ = [
    "DEFAULT_TARGET_MSE",
    "Evaluation",
    "FitResult",
    "Progress",
    "Quality",
    "SearchSpace",
    "choose_best",
    "correlation",
    "fit",
    "pareto_frontier",
    "peak",
    "plot_frontier",
    "preferred",
    "relative_mse",
    "report_dict",
    "rms",
    "save_fit",
    "snr_db",
]
