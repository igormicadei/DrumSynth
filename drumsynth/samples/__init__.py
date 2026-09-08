"""Training data for modal drum fitting.

One Sample = one recorded hit. A SampleSet = every hit of ONE drum across
velocities, which is the unit the fitter consumes:

    SampleSet (one drum)
      |-- shared:   DrumParams        <- f_static, t60, tension   (fit once, frozen)
      \\-- per-hit:  excitation params <- gain[], noise level[], contact_time

Two things worth knowing before populating this:

  * The velocity LABEL is nominal, not physical. A MIDI 100 hit and a MIDI 110
    hit from the same session are not reliably 10 units apart in energy, and
    the mapping is neither linear nor consistent across drums. Store the label,
    but calibrate against measured energy and fit against the calibrated value.

  * Tail truncation silently corrupts every t60 estimate. A WAV cut before the
    resonance finished decaying fits a shorter t60 than the real one, and
    nothing about the result looks wrong. This is checked, not assumed.
"""

from .calibration import VelocityCalibration
from .library import SampleLibrary
from .quality import QualityChecker, SampleQuality
from .sample import Sample
from .sample_set import SampleSet

__all__ = [
    "Sample",
    "SampleQuality",
    "QualityChecker",
    "VelocityCalibration",
    "SampleSet",
    "SampleLibrary",
]
