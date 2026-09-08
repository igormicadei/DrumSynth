"""Modal drum synthesis — membrane drums (kick, toms, snare shell).

    strike --> excitation --> [ modal bank ] --> mix --> out
                          \\-> [ noise bank ] -/
                                    ^
                              tension feedback
                              (bank energy -> mode frequency)

Two parameter groups, cleanly separated:

  * DrumParams  -- the drum's identity. Static, serializable, fittable.
  * DrumVoice   -- runtime state. One instance = one playable drum.
"""

from .kit import DrumSequencer
from .noise import NoiseBank, NoiseVoice
from .params import DrumParams, Mode, NoiseBand, Tension
from .presets import DampingCurve, DrumPresets, ExcitationTilt, ModalLayout
from .resonators import ModalBank, ModeResonator, TensionTracker
from .voice import DrumKit, DrumVoice

__all__ = [
    "DrumParams",
    "Mode",
    "NoiseBand",
    "Tension",
    "ModeResonator",
    "ModalBank",
    "TensionTracker",
    "NoiseVoice",
    "NoiseBank",
    "DrumVoice",
    "DrumKit",
    "DrumSequencer",
    "DrumPresets",
    "DampingCurve",
    "ModalLayout",
    "ExcitationTilt",
]
