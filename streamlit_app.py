"""DrumSynth studio — edit a drum by ear while it is ringing.

    streamlit run streamlit_app.py

The audio runs in a separate process that never stops (`drumsynth.live.engine`).
Striking while the bank is still decaying superposes, and moving a slider
changes the rest of the current decay rather than restarting it.
"""

from __future__ import annotations

import streamlit as st

from drumsynth.studio import Studio

st.set_page_config(
    page_title="DrumSynth studio",
    page_icon=":material/graphic_eq:",
    layout="wide",
    initial_sidebar_state="expanded",
)

studio = Studio.bootstrap()

from app_pages.transport import render_transport  # noqa: E402  (needs page config first)

with st.sidebar:
    render_transport(studio)

navigation = st.navigation(
    [
        st.Page(
            "app_pages/mixer.py", title="Mixer", icon=":material/tune:", default=True
        ),
        st.Page(
            "app_pages/analysis.py", title="Analysis", icon=":material/monitoring:"
        ),
        st.Page("app_pages/training.py", title="Training", icon=":material/school:"),
        st.Page(
            "app_pages/previous_runs.py",
            title="Previous runs",
            icon=":material/history:",
        ),
        st.Page("app_pages/run.py", title="Run", icon=":material/description:"),
    ]
)
navigation.run()
