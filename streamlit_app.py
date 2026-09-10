"""DrumSynth studio — fit a sound, then listen to what the model kept.

    streamlit run streamlit_app.py

Three pages. **Fit** runs a search on one sample and shows the size/error
frontier it found; **Instrument** fits a whole drum across its velocities and
plays it back at any of them; **Model** opens a model file on its own. All
three are thin: everything they do is in the library, and nothing here is
needed to run a fit — the CLI does the same job.
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
        st.Page(
            "app_pages/instrument.py", title="Instrument", icon=":material/graphic_eq:"
        ),
        st.Page("app_pages/model.py", title="Model", icon=":material/piano:"),
    ]
)
navigation.run()
