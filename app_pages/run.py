"""Full report for one selected stored training run."""

from __future__ import annotations

import streamlit as st

from drumsynth.fitting.runs import RunStore

from app_pages.run_report import render

st.title("Run")

run_path = st.query_params.get("run")
record = RunStore().read(run_path) if run_path else None

with st.container(horizontal=True, gap="small"):
    if st.button("Previous runs", icon=":material/history:"):
        st.switch_page("app_pages/previous_runs.py")
    if st.button("New run", icon=":material/add:"):
        st.switch_page("app_pages/training.py")

if record is None:
    st.info(
        "Select a run from Previous runs to see its full report.",
        icon=":material/info:",
    )
else:
    st.subheader(f"{record.drum} · {record.started.replace('T', ' ')}")
    render(record)
