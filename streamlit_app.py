"""DrumSynth studio — fit a sound, then listen to what the model kept.

    streamlit run streamlit_app.py

Four pages. **Hit** and **Drum** run the two kinds of fit; **Instruments**
lists what has been fitted and every training each one has had; **Report**
opens one of those trainings and shows everything measurable about it, down to
what it costs to play live. All four are thin: everything they do is in the
library, and nothing here is needed to run a fit — the CLI does the same job.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="DrumSynth",
    page_icon=":material/graphic_eq:",
    layout="wide",
    initial_sidebar_state="expanded",
)

navigation = st.navigation(
    [
        st.Page(
            "app_pages/instruments.py",
            title="Instruments",
            icon=":material/library_music:",
            default=True,
        ),
        st.Page("app_pages/report.py", title="Report", icon=":material/monitoring:"),
        st.Page("app_pages/fit.py", title="Hit", icon=":material/search:"),
        st.Page("app_pages/fit_drum.py", title="Drum", icon=":material/graphic_eq:"),
    ]
)
navigation.run()
