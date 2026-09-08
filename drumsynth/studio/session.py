"""State that has to survive a Streamlit rerun.

Streamlit reruns the whole script on every widget change, so the audio process
cannot live in ordinary module state or it would be respawned on every slider
move. It lives in `st.cache_resource`, which is the one place Streamlit keeps a
long-lived, unserializable object.

The parameters live in `st.session_state`, and the only thing that ever gets
sent to the engine is a whole `DrumParams` — the engine diffs it and rebuilds
only what changed, so pushing the lot on every edit is both simple and quiet.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import streamlit as st

from ..core.constants import Decibels
from ..live.client import LiveSynth
from ..live.protocol import EngineSettings
from ..live.sinks import DeviceSink
from ..synth.params import DrumParams, Mode, NoiseBand, Tension
from ..synth.presets import DrumPresets


class Studio:
    """The app's whole model: parameters, the engine handle, and the bridge."""

    PRESETS = {
        "Floor tom (reference)": DrumPresets.tom,
        "Kick": DrumPresets.kick,
        "Rack tom": DrumPresets.rack_tom,
        "Snare shell": DrumPresets.snare_shell,
    }

    #: Gain and level are edited in dB, because that is how a mixer works and
    #: because a linear 0-1 slider gives almost all its travel to the top 6 dB.
    GAIN_RANGE_DB = (-72.0, 12.0)
    LEVEL_RANGE_DB = (-72.0, 6.0)

    # -- construction ---------------------------------------------------------

    @staticmethod
    def bootstrap() -> "Studio":
        """Initialize session state once, then return a handle on it."""
        state = st.session_state
        state.setdefault("params", DrumPresets.tom())
        state.setdefault("preset_name", "Floor tom (reference)")
        state.setdefault("pushed", None)
        state.setdefault("solo", set())
        state.setdefault("muted", set())
        state.setdefault("pushed_mask", None)
        state.setdefault("strike_amplitude", 0.85)
        state.setdefault("engine_error", None)
        state.setdefault(
            "engine_settings",
            EngineSettings(
                sink="device" if DeviceSink.available() else "null",
                block_size=256,
                control_period=64,
            ),
        )
        return Studio()

    # -- parameters -----------------------------------------------------------

    @property
    def params(self) -> DrumParams:
        return st.session_state.params

    @params.setter
    def params(self, value: DrumParams) -> None:
        st.session_state.params = value

    def load_preset(self, name: str) -> None:
        st.session_state.params = Studio.PRESETS[name]()
        st.session_state.preset_name = name
        self.clear_widget_state()
        self.reset_monitor()

    def load_file(self, data: bytes, name: str) -> None:
        import json

        st.session_state.params = DrumParams.from_dict(json.loads(data))
        st.session_state.preset_name = name
        self.clear_widget_state()
        self.reset_monitor()

    def clear_widget_state(self) -> None:
        """Drop every mixer widget key.

        Widget values are sticky by key, so a freshly loaded preset would be
        overwritten by the sliders left over from the previous one.
        """
        for key in [k for k in st.session_state if k.startswith(("mode_", "band_", "master_"))]:
            del st.session_state[key]

    # -- the engine -----------------------------------------------------------

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def _engine(signature: str, settings_dict: dict) -> LiveSynth:
        """One audio process per settings signature.

        Keyed on the settings so changing the device or block size gives a new
        process — those cannot be changed on a running PortAudio stream — while
        every other rerun reuses the same one.
        """
        settings = EngineSettings(**settings_dict)
        return LiveSynth(settings, cwd=str(Path.cwd()))

    def engine(self) -> LiveSynth | None:
        """The running engine, or None if it is stopped or failed to start."""
        settings = st.session_state.engine_settings
        signature = repr(sorted(settings.to_dict().items()))
        synth = Studio._engine(signature, settings.to_dict())
        if synth.is_running:
            return synth
        return None

    def start_engine(self) -> bool:
        settings = st.session_state.engine_settings
        signature = repr(sorted(settings.to_dict().items()))
        synth = Studio._engine(signature, settings.to_dict())
        if synth.is_running:
            return True
        try:
            synth.start(self.params)
            st.session_state.engine_error = None
            st.session_state.pushed = self.params.to_dict()
            return True
        except Exception as error:
            st.session_state.engine_error = str(error)
            return False

    def stop_engine(self) -> None:
        settings = st.session_state.engine_settings
        signature = repr(sorted(settings.to_dict().items()))
        Studio._engine(signature, settings.to_dict()).stop()
        st.session_state.pushed = None
        st.session_state.pushed_mask = None

    def apply_settings(self, **fields) -> None:
        """Change engine settings. Stops the current process, since block size
        and device are fixed for the life of a stream."""
        current: EngineSettings = st.session_state.engine_settings
        if all(getattr(current, key) == value for key, value in fields.items()):
            return
        self.stop_engine()
        st.session_state.engine_settings = replace(current, **fields)

    # -- pushing changes ------------------------------------------------------

    def sync(self) -> bool:
        """Send the parameters and monitor mask if either changed.

        Diffed here rather than in the engine so an idle app is silent on the
        pipe; the engine diffs again to decide what to rebuild.
        """
        synth = self.engine()
        if synth is None:
            return False

        sent = False
        payload = self.params.to_dict()
        if payload != st.session_state.pushed:
            synth.load(self.params)
            st.session_state.pushed = payload
            sent = True

        mask = self.monitor_mask()
        if mask != st.session_state.pushed_mask:
            synth.set_monitor(mask)
            st.session_state.pushed_mask = mask
            sent = True
        return sent

    # -- monitoring -----------------------------------------------------------

    def monitor_mask(self) -> list[float] | None:
        """None when everything is audible, else one multiplier per mode."""
        solo: set[int] = st.session_state.solo
        muted: set[int] = st.session_state.muted
        count = len(self.params.modes)
        if not solo and not muted:
            return None
        if solo:
            return [1.0 if index in solo else 0.0 for index in range(count)]
        return [0.0 if index in muted else 1.0 for index in range(count)]

    def toggle_solo(self, index: int) -> None:
        solo: set[int] = st.session_state.solo
        solo.symmetric_difference_update({index})

    def toggle_mute(self, index: int) -> None:
        muted: set[int] = st.session_state.muted
        muted.symmetric_difference_update({index})

    def reset_monitor(self) -> None:
        st.session_state.solo = set()
        st.session_state.muted = set()

    def is_audible(self, index: int) -> bool:
        solo: set[int] = st.session_state.solo
        if solo:
            return index in solo
        return index not in st.session_state.muted

    # -- editing helpers ------------------------------------------------------

    def replace_mode(self, index: int, **fields) -> None:
        modes = list(self.params.modes)
        current = modes[index]
        modes[index] = Mode(
            f_static=float(fields.get("f_static", current.f_static)),
            gain=float(fields.get("gain", current.gain)),
            t60=float(fields.get("t60", current.t60)),
        )
        self.params.modes = modes

    def replace_band(self, index: int, **fields) -> None:
        bands = list(self.params.noise)
        current = bands[index]
        bands[index] = NoiseBand(
            f_low=float(fields.get("f_low", current.f_low)),
            f_high=float(fields.get("f_high", current.f_high)),
            level=float(fields.get("level", current.level)),
            t60=float(fields.get("t60", current.t60)),
        )
        self.params.noise = bands

    def add_mode(self) -> None:
        highest = max((mode.f_static for mode in self.params.modes), default=100.0)
        self.params.modes = list(self.params.modes) + [
            Mode(f_static=highest * 1.13, gain=0.02, t60=0.25)
        ]

    def remove_mode(self, index: int) -> None:
        if len(self.params.modes) <= 1:
            return
        self.params.modes = [
            mode for position, mode in enumerate(self.params.modes) if position != index
        ]
        self.reset_monitor()
        self.clear_widget_state()

    def add_band(self) -> None:
        highest = max((band.f_high for band in self.params.noise), default=200.0)
        low = min(highest, 15000.0)
        self.params.noise = list(self.params.noise) + [
            NoiseBand(f_low=low, f_high=min(low * 2.5, 20000.0), level=0.02, t60=0.02)
        ]

    def remove_band(self, index: int) -> None:
        self.params.noise = [
            band for position, band in enumerate(self.params.noise) if position != index
        ]
        self.clear_widget_state()

    def set_tension(self, k: float, tau: float) -> None:
        self.params.tension = Tension(
            k=k, tau=tau,
            max_ratio=self.params.tension.max_ratio,
            instant_attack=self.params.tension.instant_attack,
        )

    # -- units ----------------------------------------------------------------

    @staticmethod
    def to_db(value: float) -> float:
        return float(Decibels.from_amplitude(max(value, 1e-9)))

    @staticmethod
    def from_db(value: float) -> float:
        return float(Decibels.to_amplitude(value))

    # -- derived views --------------------------------------------------------

    def mode_table(self):
        import pandas as pd

        return pd.DataFrame(
            {
                "audible": [self.is_audible(i) for i in range(len(self.params.modes))],
                "f_static": [m.f_static for m in self.params.modes],
                "gain_db": [Studio.to_db(m.gain) for m in self.params.modes],
                "t60": [m.t60 for m in self.params.modes],
            }
        )

    def apply_mode_table(self, frame) -> None:
        modes, solo, muted = [], set(), set()
        for index, row in enumerate(frame.itertuples(index=False)):
            modes.append(
                Mode(
                    f_static=float(row.f_static),
                    gain=Studio.from_db(float(row.gain_db)),
                    t60=float(row.t60),
                )
            )
            if not bool(row.audible):
                muted.add(index)
        self.params.modes = modes
        st.session_state.solo = solo
        st.session_state.muted = muted

    def summary(self) -> dict:
        params = self.params
        freqs = params.mode_frequencies()
        return {
            "modes": len(params.modes),
            "bands": len(params.noise),
            "parameters": params.parameter_count(),
            "f0": params.fundamental(),
            "top": float(np.max(freqs)) if len(freqs) else 0.0,
            "longest_t60": params.longest_t60(),
            "glide_semitones": 12.0 * np.log2(1.0 + params.tension.k)
            if params.tension.k > 0
            else 0.0,
        }
