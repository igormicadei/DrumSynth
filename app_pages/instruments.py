"""Every instrument that has been fitted, and every training it has had."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from drumsynth import Corpus
from drumsynth.runs import RunStore

KINDS = {"instrument": "Drums", "hit": "Single hits"}


def _error(run) -> float | None:
    """Whichever error this kind of run reports."""
    return run.metadata.get("reconstruction_mse", run.metadata.get("relative_mse"))


@st.cache_data(show_spinner=False)
def _library() -> dict[str, tuple[int, int]]:
    """What the sample library holds: indexed recordings, and how many are here."""
    try:
        corpus = Corpus.load()
    except FileNotFoundError:
        return {}
    return {
        name: (len(corpus.samples(name)), len(corpus.samples(name, present_only=True)))
        for name in corpus.names
    }


st.title("Instruments")

store = RunStore(st.sidebar.text_input("Run store", value=str(RunStore.default().root)))
kind = st.sidebar.radio("Kind", list(KINDS), format_func=KINDS.get, horizontal=True)

fitted = store.names(kind)
library = _library() if kind == "instrument" else {}

if not fitted and not library:
    st.info(
        f"Nothing fitted yet, and no sample library found. "
        f"`drumsynth fit-drum <name>` puts a run in `{store.root}/`."
    )
    st.stop()

st.caption(
    f"{len(fitted)} fitted · runs kept in `{store.root}/` · "
    "a fit is expensive, so none of them overwrite each other"
)

# -- the catalogue ------------------------------------------------------------

rows = []
for name in sorted(set(fitted) | set(library)):
    runs = store.runs(name, kind)
    indexed, present = library.get(name, (0, 0))
    rows.append(
        {
            "instrument": name,
            "trainings": len(runs),
            "last trained": runs[0].created.strftime("%Y-%m-%d %H:%M") if runs else "—",
            "best error": min((_error(run) for run in runs), default=None),
            "recordings": indexed,
            "on disk": present,
        }
    )

st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

# -- one instrument's history -------------------------------------------------

# Fitted instruments first: they are the ones with something to show.
choices = sorted(fitted) + sorted(set(library) - set(fitted))
selected = st.selectbox("Instrument", choices, index=0 if choices else None)

runs = store.runs(selected, kind)
st.subheader(f"{selected} — {len(runs)} training{'' if len(runs) == 1 else 's'}")

if not runs:
    indexed, present = library.get(selected, (0, 0))
    if present:
        st.info(f"Never fitted. {present} of its {indexed} recordings are on this machine.")
    else:
        st.warning(
            f"Never fitted, and none of its {indexed} recordings are on this machine "
            "— see docs/DATA.md."
        )
    st.stop()

st.dataframe(
    pd.DataFrame(
        [
            {
                "when": run.created.strftime("%Y-%m-%d %H:%M"),
                "error": _error(run),
                "target": run.metadata.get("target_mse"),
                "met": run.metadata.get("target_reached"),
                "kB": round(run.metadata.get("bytes", 0) / 1024, 1),
                "layers": run.metadata.get("n_layers"),
                "search s": round(run.metadata.get("search_seconds", 0.0), 1),
                "representation": run.summary,
            }
            for run in runs
        ]
    ),
    hide_index=True,
    width="stretch",
)

chosen = st.selectbox(
    "Open a training", runs, format_func=lambda run: run.label, index=0
)

left, right = st.columns(2)
if left.button("Open the full report", type="primary", width="stretch"):
    st.session_state["run_path"] = str(chosen.path)
    st.switch_page("app_pages/report.py")

right.write(f"`{chosen.path}` · {chosen.n_bytes() / 1024:.0f} kB on disk")
