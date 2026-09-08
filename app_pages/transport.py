"""Sidebar: the audio process, the strike, and preset in/out.

Everything here is app-level — which engine is running, what it is playing, and
how to hit it — so it stays out of the mixer's way.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from drumsynth.live.sinks import DeviceSink
from drumsynth.studio import Meter, Studio


def render_transport(studio: Studio) -> None:
    """Order matters: striking is the primary action, so it sits above the fold.

    Engine setup is a once-per-session task and lives in a collapsed expander
    below the controls you actually reach for while designing a sound.
    """
    st.markdown("### DrumSynth studio")
    _engine_status(studio)
    _play_controls(studio)
    st.divider()
    _engine_settings(studio)
    st.divider()
    _preset_controls(studio)


# -- engine -------------------------------------------------------------------


def _engine_status(studio: Studio) -> None:
    synth = studio.engine()
    running = synth is not None

    with st.container(horizontal=True, wrap=False):
        if running:
            st.button(
                "Stop", icon=":material/stop:", width="stretch",
                on_click=studio.stop_engine,
            )
        else:
            st.button(
                "Start audio", icon=":material/play_arrow:", type="primary",
                width="stretch", on_click=studio.start_engine,
            )
        st.button(
            "Panic", icon=":material/block:", width="stretch",
            help="Silence everything that is ringing",
            disabled=not running,
            on_click=lambda: synth and synth.panic(),
        )

    if st.session_state.engine_error:
        st.error(st.session_state.engine_error, icon=":material/error:")
    st.caption(
        f":green[●] {synth.describe()}" if running else ":gray[○ stopped]"
    )


def _engine_settings(studio: Studio) -> None:
    settings = st.session_state.engine_settings
    running = studio.engine() is not None

    with st.expander("Audio settings", expanded=False):
        devices = DeviceSink.devices()
        options = ["device", "null"] if devices else ["null"]
        if not devices:
            st.caption(
                "No sound card here, so only `null` is offered — it runs the "
                "whole engine and its meters without audio output. Locally, "
                "`pip install -e \".[live]\"` enables device output."
            )

        sink = st.segmented_control(
            "Output", options, default=settings.sink if settings.sink in options else options[0],
            key="cfg_sink",
        )
        chosen = None
        if sink == "device" and devices:
            names = {f"[{d['index']}] {d['name']}": d["index"] for d in devices}
            label = st.selectbox("Device", list(names), key="cfg_device")
            chosen = names[label]

        block = st.select_slider(
            "Block size", options=[64, 128, 256, 512, 1024],
            value=settings.block_size, key="cfg_block",
            help="Frames per callback. Smaller is tighter; too small drops out.",
        )
        st.caption(f"{1000 * block / settings.sr:.1f} ms strike latency")

        control = st.select_slider(
            "Control period", options=[1, 8, 16, 32, 64, 128],
            value=settings.control_period, key="cfg_control",
            help=(
                "Samples between tension updates. 64 is inaudible and much "
                "cheaper; 1 is for validating."
            ),
        )
        st.button(
            "Apply and restart" if running else "Apply",
            icon=":material/restart_alt:", width="stretch",
            on_click=studio.apply_settings,
            kwargs={
                "sink": sink, "device": chosen,
                "block_size": int(block), "control_period": int(control),
            },
        )


# -- playing ------------------------------------------------------------------


def _play_controls(studio: Studio) -> None:
    synth = studio.engine()
    st.markdown("**Play**")

    # No `value=` here: the key already holds it (Studio.bootstrap seeds it),
    # and passing both makes Streamlit warn about conflicting sources.
    amplitude = st.slider(
        "Strike strength", min_value=0.02, max_value=1.5, step=0.01,
        key="strike_amplitude",
        help=(
            "One number. The glide deepens on hard hits by itself, because more "
            "energy enters the bank — there is no velocity term anywhere."
        ),
    )

    st.button(
        "Strike", icon=":material/radio_button_checked:", type="primary",
        width="stretch", disabled=synth is None,
        on_click=lambda: synth and synth.strike(amplitude),
    )
    with st.container(horizontal=True, wrap=False):
        for label, level in (("ghost", 0.15), ("mid", 0.5), ("hard", 1.0)):
            st.button(
                label, key=f"hit_{label}", width="stretch", disabled=synth is None,
                on_click=lambda level=level: synth and synth.strike(level),
            )
    st.button(
        "Flam", icon=":material/repeat:", width="stretch", disabled=synth is None,
        help="Two strikes 25 ms apart — they superpose, with no special case",
        on_click=lambda: _flam(studio, amplitude),
    )

    _live_meter(studio)


def _flam(studio: Studio, amplitude: float) -> None:
    import time

    synth = studio.engine()
    if synth is None:
        return
    synth.strike(amplitude * 0.45)
    time.sleep(0.025)
    synth.strike(amplitude)


@st.fragment(run_every="0.4s")
def _live_meter(studio: Studio) -> None:
    Meter.render(studio)


# -- presets ------------------------------------------------------------------


def _preset_controls(studio: Studio) -> None:
    st.markdown("**Parameters**")

    st.selectbox(
        "Preset", list(Studio.PRESETS), key="preset_pick",
        index=list(Studio.PRESETS).index(st.session_state.preset_name)
        if st.session_state.preset_name in Studio.PRESETS else 0,
    )
    st.button(
        "Load preset", icon=":material/download:", width="stretch",
        on_click=lambda: studio.load_preset(st.session_state.preset_pick),
    )

    uploaded = st.file_uploader("Open DrumParams JSON", type="json", key="preset_upload")
    if uploaded is not None and st.button(
        "Load file", icon=":material/folder_open:", width="stretch"
    ):
        studio.load_file(uploaded.getvalue(), Path(uploaded.name).stem)
        st.rerun()

    st.download_button(
        "Save DrumParams JSON",
        data=json.dumps(studio.params.to_dict(), indent=2),
        file_name=f"{studio.params.name or 'drum'}.json",
        mime="application/json",
        icon=":material/save:",
        width="stretch",
    )

    synth = studio.engine()
    with st.container(horizontal=True, wrap=False):
        seconds = st.number_input(
            "Take (s)", min_value=0.5, max_value=30.0, value=4.0, step=0.5,
            key="record_seconds", label_visibility="collapsed",
        )
        st.button(
            "Record", icon=":material/fiber_manual_record:", width="stretch",
            disabled=synth is None,
            help="Capture what the engine plays next, straight to a WAV",
            on_click=lambda: synth and synth.record("out/live-take.wav", seconds),
        )
    if synth is not None and synth.last_recording:
        st.caption(f":green[saved] {synth.last_recording['path']}")
