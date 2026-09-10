"""Fit a whole drum across its velocities, then play it at any of them."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from drumsynth import Corpus
from drumsynth.backend import available_devices
from drumsynth.parallel import resolve_jobs
from drumsynth.instrument import (
    InstrumentSearchSpace,
    VelocityLayers,
    between,
    fit_instrument,
    save_instrument_fit,
    sweep,
)

SPACES = {
    "quick": InstrumentSearchSpace.quick,
    "default": InstrumentSearchSpace,
    "full": InstrumentSearchSpace.full,
}


@st.cache_data(show_spinner=False)
def _playable_drums() -> dict[str, int]:
    """Drums whose audio is actually on this machine, and how much of it."""
    try:
        corpus = Corpus.load()
    except FileNotFoundError:
        return {}
    return {
        name: len(corpus.samples(name, present_only=True))
        for name in corpus.names
        if corpus.samples(name, present_only=True)
    }


@st.cache_resource(show_spinner="loading recordings")
def _layers(drum: str, max_duration: float | None) -> VelocityLayers:
    return VelocityLayers.from_corpus(drum, max_duration=max_duration)


st.title("Instrument")
st.caption(
    "One drum, every velocity it was recorded at, one model — and the velocities "
    "in between, which is the part no recording can give you."
)

drums = _playable_drums()
if not drums:
    st.info(
        "No sample audio on this machine. The library indexes the recordings but "
        "does not ship them — see docs/DATA.md."
    )
    st.stop()

drum = st.sidebar.selectbox(
    "Drum", sorted(drums), format_func=lambda name: f"{name} ({drums[name]} recordings)"
)
target = st.sidebar.select_slider(
    "Target relative MSE",
    options=[1e-2, 1e-3, 1e-4, 1e-5],
    value=1e-4,
    format_func=lambda v: f"{v:.0e}  ({-10 * np.log10(v):.0f} dB)",
)
space = st.sidebar.radio("Search space", list(SPACES), index=0, horizontal=True)
donors = st.sidebar.number_input(
    "Velocities donating phase (0 = all)", min_value=0, max_value=64, value=0
)
jobs = st.sidebar.slider("Threads", 1, 32, resolve_jobs(0))
device = st.sidebar.selectbox(
    "Device", available_devices(), help="numpy is exact; torch devices are float32"
)
max_duration = st.sidebar.number_input(
    "Trim recordings to (s, 0 = full length)", min_value=0.0, max_value=10.0, value=0.0
)

layers = _layers(drum, max_duration or None)
st.write(
    f"**{layers.n_recordings}** recordings across **{layers.n_layers}** velocities, "
    f"{layers.velocities.min():g} to {layers.velocities.max():g}, "
    f"{layers.duration:.2f} s each at {layers.sample_rate} Hz"
)

if st.sidebar.button("Fit", type="primary", width="stretch"):
    bar = st.progress(0.0, text="searching")
    st.session_state["instrument_fit"] = fit_instrument(
        layers,
        target_mse=target,
        space=SPACES[space](),
        n_donors=int(donors),
        jobs=jobs,
        device=device,
        progress=lambda p: bar.progress(
            p.done / max(p.total, 1),
            text=f"{p.done}/{p.total} — best {p.best.relative_mse:.2e} "
            f"in {p.best.n_scalars} numbers",
        ),
    )
    bar.empty()

result = st.session_state.get("instrument_fit")
if result is None or result.model.name != drum:
    st.info("Press Fit to search for a model of this drum.")
    st.stop()

model = result.model
left, middle, right = st.columns(3)
left.metric("Reconstruction", f"{result.reconstruction_mse:.2e}", f"target {result.target_mse:.0e}")
middle.metric(
    "Model",
    f"{model.n_bytes() / 1024:.0f} kB",
    f"{1.0 / model.compression_ratio(result.encoded_recordings):.1f}x smaller "
    f"than the {result.encoded_recordings} recordings it holds",
)
right.metric(
    "Interpolation",
    "n/a" if result.interpolation_mse is None else f"{result.interpolation_mse:.2e}",
    "at velocities held out of the fit",
)

if not result.target_reached:
    st.warning("No candidate in this space reached the target. Try a wider space.")

st.code(result.summary(), language="text")

st.subheader("Play it")
low, high = model.velocity_range
if high > low:
    velocity = st.slider("Velocity", float(low), float(high), float(0.5 * (low + high)))
else:
    # A fresh clone has one recording of one drum, so this is the usual case here.
    velocity = float(low)
    st.caption(f"only one velocity was recorded ({velocity:g}), so there is nothing to slide")
st.audio(model.render(velocity), sample_rate=model.sample_rate)

nearest = float(model.velocities[int(np.argmin(np.abs(model.velocities - velocity)))])
st.caption(
    f"nearest recorded velocity {nearest:g}; phase borrowed from "
    f"{model.donor_velocities[model.donor_index(velocity)]:g}"
)

with st.expander("Sweeps"):
    st.write("The model at every recorded velocity")
    st.audio(sweep(model, model.velocities), sample_rate=model.sample_rate)
    st.write("And halfway between them — hits nobody played")
    st.audio(sweep(model, between(model.velocities)), sample_rate=model.sample_rate)

st.subheader("Per velocity")
st.dataframe(
    pd.DataFrame([layer.to_dict() for layer in result.layers]),
    hide_index=True,
    width="stretch",
)

st.download_button(
    "Download instrument.npz",
    model.to_bytes(),
    file_name=f"{drum}.npz",
    width="stretch",
)
directory = st.text_input("Save the whole run to", value=f"out/{drum}")
if st.button("Save run", width="stretch"):
    written = save_instrument_fit(result, directory, layers=layers)
    st.success("  ".join(str(path) for path in written.values()))
