"""Playing the synthesizer live, from a separate process.

The audio runs in its own process and never stops. Strikes land on a bank that
is still ringing, and parameter edits change the rest of the current decay
rather than restarting it — which is what a resonator bank with persistent
state is for, and what a render-on-demand call cannot do.

    from drumsynth.live import LiveSynth

    with LiveSynth() as synth:
        synth.strike(0.8)
        synth.load(params)     # applied mid-ring
        synth.strike(0.5)      # superposes onto what is still decaying
"""

from .client import LiveSynth
from .engine import EngineCLI, LiveEngine
from .protocol import Command, EngineSettings, Event, Status
from .sinks import AudioSink, DeviceSink, NullSink, SinkFactory, WavSink

__all__ = [
    "LiveSynth",
    "LiveEngine",
    "EngineCLI",
    "Command",
    "Event",
    "Status",
    "EngineSettings",
    "AudioSink",
    "NullSink",
    "WavSink",
    "DeviceSink",
    "SinkFactory",
]
