"""Spectral modelling for drum hits: analyse a sound, fit it, play it back.

A hit becomes a short list of FFT bins and the complex amplitude of each one
over time — the component block — stored under whichever coding survives
measurement for that particular sound. Nothing about the representation is
decided in advance: the fit encodes, quantizes and *renders* every candidate,
and compares the audio it gets back against the audio it was given.

    drumsynth.spectral   the representation — audio + Candidate -> SpectralModel
    drumsynth.fitting    the search        — audio -> the smallest model that fits
    drumsynth.corpus     the sample library that ships with the repository
    drumsynth.core       audio I/O and units

Quick start
-----------
    from drumsynth import AudioIO, fit, save_fit

    signal, sr = AudioIO.read("hit.wav")
    result = fit(signal, sr, target_mse=1e-5)
    print(result.summary())
    save_fit(result, "hit_fit", reference=signal)

and later, from the file alone:

    from drumsynth import SpectralModel

    model = SpectralModel.load("hit_fit/model.npz")
    AudioIO.write("again.wav", model.render(), model.sample_rate)

See docs/ARCHITECTURE.md for why the model is shaped this way, and
docs/FINDINGS.md for the places where measurement contradicted the design.
"""

from .core import DEFAULT_SR, TWO_PI, Audio, AudioIO, Decibels
from .corpus import Corpus, Sample
from .fitting import (
    DEFAULT_TARGET_MSE,
    Evaluation,
    FitResult,
    Progress,
    Quality,
    SearchSpace,
    fit,
    pareto_frontier,
    relative_mse,
    save_fit,
)
from .spectral import Candidate, SpectralModel, StftSpec, analyze, encode, synthesize

__version__ = "0.2.0"

__all__ = [
    "DEFAULT_SR",
    "DEFAULT_TARGET_MSE",
    "TWO_PI",
    "Audio",
    "AudioIO",
    "Candidate",
    "Corpus",
    "Decibels",
    "Evaluation",
    "FitResult",
    "Progress",
    "Quality",
    "Sample",
    "SearchSpace",
    "SpectralModel",
    "StftSpec",
    "__version__",
    "analyze",
    "encode",
    "fit",
    "pareto_frontier",
    "relative_mse",
    "save_fit",
    "synthesize",
]
