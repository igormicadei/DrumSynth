"""Scoring for modal drum synthesis — how close is the generated hit to the sample?

Input contract
--------------
Audio in, always. Both signals go through the SAME analysis chain and produce
the same descriptor set. This is non-negotiable: the reference is a WAV with no
parameters attached, so any descriptor you cannot extract from raw audio is a
descriptor you cannot compare.

`DrumParams` is an OPTIONAL second input. It is not needed to compute the score
— it is used to attribute an error back to specific parameters ("mode 7 t60 is
40% too long") instead of just reporting that something is off.

    reference.wav  --\\
                      >-- Analyzer --> SoundDescriptors --\\
    generated.wav  --/                                     >-- Comparator --> ScoreCard
                                                          /
    DrumParams (optional) ----------------------------------  (attribution only)
"""

from .analyzer import Analyzer
from .attribution import Attributor, ParameterSuggestion
from .bands import BandDecayAnalyzer
from .comparator import Comparator, ComponentScore, ScoreCard
from .descriptors import (
    BandDecay,
    EnvelopeCurve,
    GlideTrack,
    ModeEstimate,
    NoiseStats,
    SoundDescriptors,
)
from .envelope import EnvelopeAnalyzer
from .glide import GlideAnalyzer
from .matching import ModeMatch, ModeMatcher
from .modal import ModalAnalyzer
from .noise import NoiseAnalyzer
from .prep import SignalPrep
from .report import ScoreReport
from .scorer import DrumScorer
from .stft import MultiResolutionSTFTLoss

__all__ = [
    "ModeEstimate",
    "BandDecay",
    "GlideTrack",
    "EnvelopeCurve",
    "NoiseStats",
    "SoundDescriptors",
    "SignalPrep",
    "ModalAnalyzer",
    "BandDecayAnalyzer",
    "GlideAnalyzer",
    "EnvelopeAnalyzer",
    "NoiseAnalyzer",
    "Analyzer",
    "ModeMatch",
    "ModeMatcher",
    "ComponentScore",
    "ScoreCard",
    "Comparator",
    "MultiResolutionSTFTLoss",
    "ParameterSuggestion",
    "Attributor",
    "DrumScorer",
    "ScoreReport",
]
