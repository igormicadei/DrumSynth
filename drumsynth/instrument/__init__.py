"""One drum across every velocity it was recorded at, as one model.

    drumsynth.instrument.layers   recordings in, aligned velocity layers out
    drumsynth.instrument.field    how magnitude changes with velocity
    drumsynth.instrument.donors   where a rendered hit borrows its phase
    drumsynth.instrument.model    InstrumentCandidate and InstrumentModel
    drumsynth.instrument.fit      the search, and the three things it measures
    drumsynth.instrument.report   writing a velocity fit to disk

The split that shapes all of it: magnitude varies smoothly with how hard a
drum is hit, so it is modelled and interpolated; phase is set by one strike,
so it is stored and borrowed, never mixed.
"""

from .donors import CODECS as DONOR_CODECS
from .field import CODECS as FIELD_CODECS
from .field import MagnitudeField
from .fit import (
    DEFAULT_TARGET_MSE,
    InstrumentEvaluation,
    InstrumentFitResult,
    InstrumentSearchSpace,
    LayerReport,
    fit_instrument,
    interpolation_error,
    magnitude_error,
    measure_layers,
)
from .layers import VelocityLayers
from .report import between, plot_layers, save_instrument_fit, sweep
from .model import FORMAT, InstrumentCandidate, InstrumentModel

__all__ = [
    "DEFAULT_TARGET_MSE",
    "DONOR_CODECS",
    "FIELD_CODECS",
    "FORMAT",
    "InstrumentCandidate",
    "InstrumentEvaluation",
    "InstrumentFitResult",
    "InstrumentModel",
    "InstrumentSearchSpace",
    "LayerReport",
    "MagnitudeField",
    "VelocityLayers",
    "between",
    "fit_instrument",
    "interpolation_error",
    "magnitude_error",
    "measure_layers",
    "plot_layers",
    "save_instrument_fit",
    "sweep",
]
