"""Run a search on one sound and look at what it cost."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from drumsynth import AudioIO, Corpus
from drumsynth.fitting import SearchSpace, fit, save_fit

SPACES = {"quick": SearchSpace.quick, "default": SearchSpace, "full": SearchSpace.full}


@st.cache_data(show_spinner=False)
def _corpus_samples() -> dict[str, str]:
    try:
        corpus = Corpus.load()
    except FileNotFoundError:
        return {}
    return {sample.name: str(sample.path) for sample in corpus.samples(present_only=True)}


def _load_input() -> tuple[np.ndarray, int, str] | None:
    """A sample from the shipped corpus, or a WAV the user drops in."""
    samples = _corpus_samples()
    uploaded = st.sidebar.file_uploader("A WAV file", type=["wav"])

    if uploaded is not None:
        signal, rate = AudioIO.read_bytes(uploaded.getvalue())
        return signal, rate, uploaded.name

    if not samples:
        st.info("Drop a WAV file in the sidebar to fit it.")
        return None

    chosen = st.sidebar.selectbox("or a sample from the library", sorted(samples))
    signal, rate = AudioIO.read(samples[chosen])
    return signal, rate, chosen


def _frontier_frame(result) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "numbers": e.n_scalars,
                "relative MSE": e.quality.relative_mse,
                "SNR dB": e.quality.snr_db,
                "representation": e.candidate.label(),
            }
            for e in result.frontier
        ]
    )


st.title("Fit")
st.caption(
    "Every candidate is encoded, quantized, rendered back to audio and compared "
    "against the input. The chosen one is the smallest that stayed inside the target."
)

loaded = _load_input()
if loaded is None:
    st.stop()

signal, sample_rate, name = loaded

target = st.sidebar.select_slider(
    "Target relative MSE",
    options=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
    value=1e-4,
    format_func=lambda v: f"{v:.0e}  ({-10 * np.log10(v):.0f} dB)",
)
space = st.sidebar.radio("Search space", list(SPACES), index=0, horizontal=True)
jobs = st.sidebar.slider("Processes", 1, 8, 4)

st.sidebar.write(f"{len(SPACES[space]().candidates(signal.size))} candidates")
st.audio(signal, sample_rate=sample_rate)

if not st.sidebar.button("Fit", type="primary", width="stretch"):
    st.stop()

bar = st.progress(0.0, text="searching")
result = fit(
    signal,
    sample_rate,
    target_mse=target,
    space=SPACES[space](),
    jobs=jobs,
    progress=lambda p: bar.progress(
        p.done / max(p.total, 1),
        text=f"{p.done}/{p.total} — best {p.best.quality.relative_mse:.2e} "
        f"in {p.best.n_scalars} numbers",
    ),
)
bar.empty()

reconstruction = result.model.render()
left, right = st.columns(2)
left.metric("Relative MSE", f"{result.quality.relative_mse:.2e}", f"{result.quality.snr_db:.1f} dB SNR")
right.metric(
    "Model",
    f"{result.model.n_bytes() / 1024:.1f} kB",
    f"{1.0 / result.model.compression_ratio():.1f}x smaller than 16-bit PCM",
)

if not result.target_reached:
    st.warning("No candidate in this space reached the target. Try a wider space.")

st.code(result.summary(), language="text")

st.subheader("Reconstruction")
st.audio(reconstruction, sample_rate=sample_rate)
st.caption("Residual — what the model missed")
st.audio(signal - reconstruction, sample_rate=sample_rate)

st.subheader("Frontier")
frame = _frontier_frame(result)
st.scatter_chart(frame, x="numbers", y="relative MSE", size=None)
st.dataframe(frame, hide_index=True, width="stretch")

st.download_button(
    "Download model.npz",
    result.model.to_bytes(),
    file_name=f"{Path(name).stem}.npz",
    width="stretch",
)

directory = st.text_input("Save the whole run to", value=f"out/{Path(name).stem}")
if st.button("Save run", width="stretch"):
    written = save_fit(result, directory, reference=signal, input_path=name)
    st.success("  ".join(str(path) for path in written.values()))
