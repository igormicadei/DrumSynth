"""Open a saved model on its own: what it holds, and what it sounds like."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from drumsynth import SpectralModel

st.title("Model")
st.caption("One .npz holds the arrays and the metadata. Nothing else is needed to play it.")

path = st.text_input("Path to a model.npz", value="out/model.npz")
if not path:
    st.stop()

try:
    model = SpectralModel.load(path)
except (FileNotFoundError, ValueError, OSError) as error:
    st.error(str(error))
    st.stop()

left, middle, right = st.columns(3)
left.metric("Representation", model.candidate.codec, model.candidate.label())
middle.metric("Size", f"{model.n_bytes() / 1024:.1f} kB", f"{model.n_scalars} numbers")
right.metric("Audio", f"{model.duration:.3f} s", f"{model.sample_rate} Hz")

st.audio(model.render(), sample_rate=model.sample_rate)

st.subheader("What it stores")
st.dataframe(
    pd.DataFrame(
        [
            {"array": name, "shape": str(tuple(array.shape)), "dtype": str(array.dtype)}
            for name, array in sorted({"bins": model.bins, **model.arrays}.items())
        ]
    ),
    hide_index=True,
    width="stretch",
)

st.subheader("Kept bins")
magnitudes = model.magnitudes()
st.dataframe(
    pd.DataFrame(
        {
            "Hz": np.round(model.frequencies(), 1),
            "peak": magnitudes.max(axis=1),
            "energy": np.sum(magnitudes**2, axis=1),
        }
    ).sort_values("energy", ascending=False),
    hide_index=True,
    width="stretch",
    height=320,
)
