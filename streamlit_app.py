"""DrumSynth studio — fit a sound, then listen to what the model kept.

    streamlit run streamlit_app.py

Two pages. **Fit** runs a search on a sample and shows the size/error frontier
it found; **Model** opens a model file on its own and plays it back. Both are
thin: everything they do is in `drumsynth.fitting` and `drumsynth.spectral`,
and nothing here is needed to run a fit — the CLI does the same job.
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
        st.Page("app_pages/fit.py", title="Fit", icon=":material/search:", default=True),
        st.Page("app_pages/model.py", title="Model", icon=":material/piano:"),
    ]
)
navigation.run()
