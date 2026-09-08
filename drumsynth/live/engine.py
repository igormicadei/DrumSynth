"""The audio process.

Runs one `DrumVoice` continuously and never stops. Strikes land on a bank that
is still ringing, and parameter edits change the rest of the decay rather than
restarting it — which is the whole reason this is a persistent process and not
a render-on-demand call.

    python -m drumsynth.live.engine --sink device

Threading, and why it is arranged this way:

    stdin thread   parses commands, appends to a deque, and does nothing else
    audio thread   drains the deque at the top of every block, applies every
                   change, renders, writes telemetry to a slot
    stdout thread  publishes the telemetry slot on a timer

All mutation of the synthesizer happens on the audio thread. There is no lock
anywhere near the render call, because a lock held by the UI side is a dropout
on the audio side. `deque.append` and `popleft` are atomic under the GIL, which
is the only synchronization this needs.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from ..core.audio_io import AudioIO
from ..synth.params import DrumParams
from ..synth.presets import DrumPresets
from ..synth.voice import DrumVoice
from .protocol import Command, EngineSettings, Event, Status
from .sinks import SinkFactory, WavSink


class LiveEngine:
    """One always-on voice, driven by commands from stdin."""

    #: Beyond this many queued commands the UI is further ahead than the audio
    #: thread can consume. Dropping the oldest is better than growing without
    #: bound; it only happens if something upstream is in a loop.
    MAX_PENDING: int = 512

    def __init__(self, settings: EngineSettings, params: DrumParams | None = None) -> None:
        self.settings = settings
        self.params = params or DrumPresets.tom(sr=settings.sr)
        self.voice = DrumVoice(
            self.params, settings.sr, settings.control_period, settings.seed
        )

        self._pending: deque[dict] = deque()
        self._status = Status()
        self._running = threading.Event()
        self._blocks = 0
        self._strikes = 0
        self._started_at = 0.0
        self._recorder: dict | None = None
        self._block_period = settings.block_size / settings.sr

        self.sink = SinkFactory.build(settings)

    # -- output ---------------------------------------------------------------

    @staticmethod
    def emit(name: str, **fields) -> None:
        """One JSON line to stdout, flushed. The UI is reading a pipe."""
        sys.stdout.write(Event.encode(name, **fields) + "\n")
        sys.stdout.flush()

    # -- the audio thread -----------------------------------------------------

    def render(self, frames: int) -> np.ndarray:
        """Called by the sink for every block. Owns all mutation."""
        started = time.perf_counter()
        self._drain()

        block = self.voice.render(frames)
        self._blocks += 1

        if self._recorder is not None:
            self._capture(block)

        peak = float(np.max(np.abs(block))) if block.size else 0.0
        self._status = Status(
            peak=peak,
            rms=float(np.sqrt(np.mean(block**2))) if block.size else 0.0,
            energy=self.voice.energy,
            frequency_ratio=self.voice.frequency_ratio,
            blocks=self._blocks,
            strikes=self._strikes,
            underruns=self.sink.underruns,
            clipped=peak > 1.0,
            load=(time.perf_counter() - started) / self._block_period,
            seconds=time.perf_counter() - self._started_at,
        )
        return block

    def _drain(self) -> None:
        while self._pending:
            try:
                self.apply(self._pending.popleft())
            except Exception as error:  # never let one bad command kill the audio
                self.emit(Event.ERROR, message=f"{type(error).__name__}: {error}")

    def apply(self, message: dict) -> None:
        """Act on one command. Runs on the audio thread."""
        name = message.get("cmd")

        if name == Command.STRIKE:
            self.voice.strike(float(message.get("amplitude", 1.0)))
            self._strikes += 1

        elif name == Command.PARAMS:
            # Not reset(): the point of a live engine is that a knob changes the
            # rest of the current ring instead of restarting it.
            self.params = DrumParams.from_dict(message["params"])
            self.voice.update_params(self.params)

        elif name == Command.GAIN:
            self.params.output_gain = float(message["value"])
            self.voice.output_gain = self.params.output_gain

        elif name == Command.MONITOR:
            mask = message.get("mask")
            self.voice.set_monitor(np.asarray(mask, dtype=float) if mask else None)

        elif name == Command.RESET:
            self.voice.reset()

        elif name == Command.PANIC:
            self.voice.reset()
            self._pending.clear()

        elif name == Command.RECORD:
            self._start_recording(message)

        elif name == Command.PING:
            self.emit(Event.STATUS, **self._status.to_dict())

        elif name == Command.QUIT:
            self._running.clear()

        else:
            self.emit(Event.ERROR, message=f"unknown command {name!r}")

    # -- recording ------------------------------------------------------------

    def _start_recording(self, message: dict) -> None:
        seconds = float(message.get("seconds", 4.0))
        self._recorder = {
            "path": str(message.get("path", "live-take.wav")),
            "remaining": int(seconds * self.settings.sr),
            "chunks": [],
            "normalize": bool(message.get("normalize", False)),
        }

    def _capture(self, block: np.ndarray) -> None:
        recorder = self._recorder
        take = block[: recorder["remaining"]]
        recorder["chunks"].append(take.copy())
        recorder["remaining"] -= len(take)
        if recorder["remaining"] > 0:
            return

        self._recorder = None
        audio = np.concatenate(recorder["chunks"])
        try:
            AudioIO.write(
                recorder["path"], audio, self.settings.sr,
                normalize=recorder["normalize"],
            )
            self.emit(
                Event.RECORDED,
                path=str(Path(recorder["path"]).resolve()),
                seconds=len(audio) / self.settings.sr,
                peak=float(np.max(np.abs(audio))) if audio.size else 0.0,
            )
        except Exception as error:
            self.emit(Event.ERROR, message=f"could not write take: {error}")

    # -- the command thread ---------------------------------------------------

    def _read_stdin(self) -> None:
        for line in sys.stdin:
            message = Command.decode(line)
            if message is None:
                continue
            if len(self._pending) >= LiveEngine.MAX_PENDING:
                self._pending.popleft()
            self._pending.append(message)
            if message.get("cmd") == Command.QUIT:
                break
        self._running.clear()

    # -- lifecycle ------------------------------------------------------------

    def run(self) -> int:
        self._started_at = time.perf_counter()
        self._running.set()

        try:
            self.sink.start(self.render)
        except Exception as error:
            self.emit(Event.ERROR, message=f"could not open audio output: {error}")
            return 1

        self.emit(
            Event.READY,
            sr=self.settings.sr,
            block_size=self.settings.block_size,
            control_period=self.settings.control_period,
            sink=self.sink.name,
            device=self.sink.description,
            latency_ms=1000.0 * self.settings.block_size / self.settings.sr,
            modes=len(self.params.modes),
            noise_bands=len(self.params.noise),
        )

        threading.Thread(target=self._read_stdin, daemon=True).start()

        interval = self.settings.status_interval
        try:
            while self._running.is_set():
                time.sleep(interval)
                self.emit(Event.STATUS, **self._status.to_dict())
        except KeyboardInterrupt:
            pass
        finally:
            self.sink.stop()
            if isinstance(self.sink, WavSink):
                self.sink.write()
            self.emit(Event.STOPPED, blocks=self._blocks, strikes=self._strikes)
        return 0


class EngineCLI:
    """Argument parsing, kept out of the engine so tests can build one directly."""

    @staticmethod
    def parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--sr", type=int, default=44100)
        parser.add_argument(
            "--block-size", type=int, default=256,
            help="frames per audio callback; sets the strike latency",
        )
        parser.add_argument(
            "--control-period", type=int, default=64,
            help="samples between tension updates (1 while validating)",
        )
        parser.add_argument("--sink", choices=("device", "null", "wav"), default="device")
        parser.add_argument("--device", default=None, help="output device name or index")
        parser.add_argument("--seed", type=int, default=0, help="noise seed; -1 for random")
        parser.add_argument(
            "--params", type=Path, default=None,
            help="DrumParams JSON to start from (default: the tom preset)",
        )
        parser.add_argument("--status-interval", type=float, default=0.1)
        parser.add_argument(
            "--no-realtime", action="store_true",
            help="null sink only: render as fast as possible instead of pacing",
        )
        parser.add_argument(
            "--list-devices", action="store_true", help="print output devices and exit"
        )
        return parser

    @staticmethod
    def main(argv: list[str] | None = None) -> int:
        args = EngineCLI.parser().parse_args(argv)

        if args.list_devices:
            from .sinks import DeviceSink

            devices = DeviceSink.devices()
            if not devices:
                print("no output devices (is sounddevice installed?)")
                return 1
            for item in devices:
                print(f"[{item['index']}] {item['name']}  "
                      f"{item['channels']}ch @ {item['default_sr']:.0f} Hz")
            return 0

        device: str | int | None = args.device
        if isinstance(device, str) and device.isdigit():
            device = int(device)

        settings = EngineSettings(
            sr=args.sr,
            block_size=args.block_size,
            control_period=args.control_period,
            sink=args.sink,
            device=device,
            seed=None if args.seed < 0 else args.seed,
            status_interval=args.status_interval,
            extra={"realtime": not args.no_realtime},
        )
        params = DrumParams.load(args.params) if args.params else None
        return LiveEngine(settings, params).run()


if __name__ == "__main__":
    sys.exit(EngineCLI.main())
