"""Manage and open stored training runs."""

from __future__ import annotations

import streamlit as st
import pandas as pd

from drumsynth.fitting.runs import RunStore

st.title("Previous runs")

store = RunStore()
runs = store.list()

if st.button("New run", icon=":material/add:", type="primary", width="content"):
    st.switch_page("app_pages/training.py")

if not runs:
    st.info(
        "No runs stored yet. Start a training run to see it here.",
        icon=":material/info:",
    )
    st.stop()

st.caption(f"{len(runs)} stored run{'s' if len(runs) != 1 else ''}.")
st.dataframe(
    pd.DataFrame([record.to_row() for record in runs]),
    width="stretch",
    hide_index=True,
    column_config={
        "score": st.column_config.NumberColumn(format="%.3f"),
        "stft dB": st.column_config.NumberColumn(format="%.2f"),
        "seconds": st.column_config.NumberColumn(format="%.0f s"),
    },
)

labels = {record.label: record for record in runs}
selected_label = st.selectbox("Run", list(labels), key="previous_run_pick")
selected = labels[selected_label]

with st.container(horizontal=True, gap="small"):
    if st.button("Open run", icon=":material/open_in_new:", type="primary"):
        st.query_params["run"] = str(selected.directory)
        st.switch_page("app_pages/run.py")

    delete_key = f"delete_{selected.started}"
    if st.button("Delete run", icon=":material/delete:", key=delete_key):
        st.session_state["confirm_delete_run"] = str(selected.directory)

if st.session_state.get("confirm_delete_run") == str(selected.directory):
    st.warning(
        f"Delete {selected.label}? This cannot be undone.", icon=":material/warning:"
    )
    with st.container(horizontal=True, gap="small"):
        if st.button("Confirm delete", type="primary", key="confirm_delete"):
            store.delete(selected.directory)
            st.session_state.pop("confirm_delete_run", None)
            st.rerun()
        if st.button("Cancel", key="cancel_delete"):
            st.session_state.pop("confirm_delete_run", None)
            st.rerun()
