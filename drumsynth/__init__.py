"""Modal drum synthesis, scoring and sample handling for membrane drums.

Three subsystems, deliberately separable:

    drumsynth.synth     the synthesizer   — DrumParams in, audio out
    drumsynth.scoring   the metric        — two signals in, a ScoreCard out
    drumsynth.samples   the training data — a directory in, validated sets out

The synthesizer does not import the scorer and the scorer does not import the
synthesizer. Scoring takes AUDIO, always, on both sides: the reference is a WAV
with no parameters attached, so any descriptor that cannot be extracted from
raw audio is a descriptor that cannot be compared. `DrumParams` is an optional
extra input used only to attribute an error back to a specific number.

Quick start
-----------
    from drumsynth import DrumPresets, DrumVoice, DrumScorer, AudioIO

    params = DrumPresets.tom()
    voice = DrumVoice(params, control_period=64, seed=0)
    AudioIO.write("tom.wav", voice.render_hit(4.0))

    card = DrumScorer.for_fundamental(92.5).score(reference_audio, voice.render_hit(4.0))
    print(card.report())

See docs/ARCHITECTURE.md for why the model is shaped this way, and
docs/FINDINGS.md for where measurement disagreed with the design.
"""

from .core import (
    DEFAULT_SR,
    LN1000,
    Audio,
    AudioIO,
    Cents,
    Decay,
    Decibels,
    EnvelopeFollower,
    FilterDesign,
    LinearRegression,
    SpectralPeak,
)
from .samples import (
    QualityChecker,
    Sample,
    SampleLibrary,
    SampleQuality,
    SampleSet,
    VelocityCalibration,
)
from .scoring import (
    Analyzer,
    Attributor,
    BandDecay,
    BandDecayAnalyzer,
    Comparator,
    ComponentScore,
    DrumScorer,
    EnvelopeAnalyzer,
    EnvelopeCurve,
    GlideAnalyzer,
    GlideTrack,
    ModalAnalyzer,
    ModeEstimate,
    ModeMatch,
    ModeMatcher,
    MultiResolutionSTFTLoss,
    NoiseAnalyzer,
    NoiseStats,
    ParameterSuggestion,
    ScoreCard,
    ScoreReport,
    SignalPrep,
    SoundDescriptors,
)
from .synth import (
    DampingCurve,
    DrumKit,
    DrumParams,
    DrumPresets,
    DrumSequencer,
    DrumVoice,
    ExcitationTilt,
    ModalBank,
    ModalLayout,
    Mode,
    ModeResonator,
    NoiseBand,
    NoiseBank,
    NoiseVoice,
    Tension,
    TensionTracker,
)

__version__ = "0.1.0"

__all__ = [
    # core
    "Audio",
    "AudioIO",
    "Cents",
    "Decay",
    "Decibels",
    "EnvelopeFollower",
    "FilterDesign",
    "LinearRegression",
    "SpectralPeak",
    "DEFAULT_SR",
    "LN1000",
    # synth
    "Mode",
    "NoiseBand",
    "Tension",
    "DrumParams",
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
    # scoring
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
    # samples
    "Sample",
    "SampleQuality",
    "QualityChecker",
    "VelocityCalibration",
    "SampleSet",
    "SampleLibrary",
    "__version__",
]
