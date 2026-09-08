"""The mixer: one strip per mode, one per noise band, one master.

Every strip edits the live `DrumParams`. The changes go to the running engine
at the end of the script, and the engine applies them without resetting — so a
knob moved while the drum is ringing changes the rest of that ring.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from drumsynth.studio import BandStrip, MasterStrip, ModeStrip, Studio

studio = Studio.bootstrap()
summary = studio.summary()

st.title("Mixer")
with st.container(horizontal=True, gap="medium"):
    st.metric("Modes", summary["modes"])
    st.metric("Noise bands", summary["bands"])
    st.metric("Fundamental", f"{summary['f0']:.1f} Hz")
    st.metric("Top mode", f"{summary['top']:.0f} Hz")
    st.metric("Longest t60", f"{summary['longest_t60']:.2f} s")
    st.metric("Glide", f"{summary['glide_semitones']:.2f} st")

warnings = studio.params.warnings()
if warnings:
    for warning in warnings:
        st.warning(warning, icon=":material/warning:")

master, modal, transient = st.tabs(
    [
        f":material/speed: Master",
        f":material/graphic_eq: Modes ({summary['modes']})",
        f":material/bolt: Transients ({summary['bands']})",
    ]
)


# -- master -------------------------------------------------------------------

with master:
    left, right = st.columns([2, 3], gap="medium")
    with left:
        MasterStrip.render(studio)
    with right:
        with st.container(border=True):
            st.markdown("**What the parameters say**")
            st.caption(
                "Nothing here is fitted. These are the numbers the engine is "
                "playing right now, read back."
            )
            frequencies = np.sort(studio.params.mode_frequencies())
            ratios = frequencies / frequencies[0] if len(frequencies) else frequencies
            st.dataframe(
                {
                    "Hz": np.round(frequencies, 2),
                    "ratio to f0": np.round(ratios, 4),
                    "t60 s": np.round(
                        [m.t60 for m in studio.params.sorted_modes()], 3
                    ),
                    "gain dB": np.round(
                        [Studio.to_db(m.gain) for m in studio.params.sorted_modes()], 1
                    ),
                },
                height=380,
                width="stretch",
            )


# -- modes --------------------------------------------------------------------

PER_PAGE = 8


@st.fragment
def _mode_mixer() -> None:
    modes = studio.params.modes
    fundamental = studio.params.fundamental()

    view = st.segmented_control(
        "View", ["Strips", "Table"], default="Strips", key="mode_view",
        label_visibility="collapsed",
    )

    if view == "Table":
        edited = st.data_editor(
            studio.mode_table(),
            key="mode_editor",
            width="stretch",
            height=520,
            num_rows="dynamic",
            column_config={
                "audible": st.column_config.CheckboxColumn("on", width="small"),
                "f_static": st.column_config.NumberColumn(
                    "Hz", min_value=10.0, max_value=18000.0, step=0.5, format="%.2f"
                ),
                "gain_db": st.column_config.NumberColumn(
                    "gain dB", min_value=-72.0, max_value=12.0, step=0.5, format="%.1f"
                ),
                "t60": st.column_config.NumberColumn(
                    "t60 s", min_value=0.005, max_value=12.0, step=0.01, format="%.3f"
                ),
            },
        )
        if st.button("Apply table", icon=":material/check:", type="primary"):
            studio.apply_mode_table(edited)
            studio.clear_widget_state()
            st.rerun()
        st.caption(
            "Sorting the table does not reorder the bank. Mode order is "
            "identity for the live state — mode 3 keeps mode 3's ring."
        )
        return

    with st.container(horizontal=True, gap="small"):
        st.button(
            "Add mode", icon=":material/add:", on_click=studio.add_mode
        )
        st.button(
            "Clear solo/mute", icon=":material/hearing:",
            on_click=studio.reset_monitor,
            disabled=not (st.session_state.solo or st.session_state.muted),
        )
        if st.session_state.solo:
            st.badge(
                f"soloing {len(st.session_state.solo)}", color="orange",
                icon=":material/hearing:",
            )
        if st.session_state.muted:
            st.badge(f"{len(st.session_state.muted)} muted", color="grey")

    pages = max(1, -(-len(modes) // PER_PAGE))
    page = 1
    if pages > 1:
        with st.container(horizontal_alignment="center"):
            page = st.pagination(pages, key="mode_page")

    start = (page - 1) * PER_PAGE
    shown = list(range(start, min(start + PER_PAGE, len(modes))))

    columns = st.columns(len(shown), gap="small", wrap=False)
    for column, index in zip(columns, shown):
        with column:
            ModeStrip.render(studio, index, fundamental)

    st.caption(
        f"Showing modes {shown[0]}-{shown[-1]} of {len(modes)}. "
        "S solos a partial and M mutes it, both live and both only affect "
        "monitoring — neither is a parameter and neither is saved."
    )


with modal:
    _mode_mixer()


# -- transients ---------------------------------------------------------------

with transient:
    st.caption(
        "Contact noise: white through a bandpass, level starting at maximum and "
        "decaying immediately. There is no attack here — the ~10 ms envelope "
        "peak a real drum shows comes from the modal bank summing up."
    )
    bands = studio.params.noise
    if bands:
        columns = st.columns(min(len(bands), 4), gap="small", wrap=False)
        for column, index in zip(columns, range(len(bands))):
            with column:
                BandStrip.render(studio, index)
    else:
        st.info("No noise bands. The modal bank alone is a valid drum.",
                icon=":material/info:")
    st.button("Add band", icon=":material/add:", on_click=studio.add_band)


# -- push ---------------------------------------------------------------------

studio.sync()
