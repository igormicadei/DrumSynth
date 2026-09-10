"""Everything about one model: what it kept, what it sounds like, what it costs."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from drumsynth import AudioIO, plots
from drumsynth.bench import measure_live, measure_polyphony, resident_bytes
from drumsynth.instrument.model import InstrumentModel
from drumsynth.instrument.report import SWEEP_GAP, sweep_slice
from drumsynth.runs import Run, RunStore
from drumsynth.spectral.model import SpectralModel
from drumsynth.streaming import voice_for

st.title("Report")


# -- loading ------------------------------------------------------------------


@st.cache_resource(show_spinner="loading the model")
def _model(path: str):
    try:
        return InstrumentModel.load(path)
    except ValueError:
        return SpectralModel.load(path)


@st.cache_data(show_spinner=False)
def _reference(run_path: str, index: int, n_samples: int, sample_rate: int):
    """The recording this model was measured against, out of what the run saved."""
    run = Path(run_path)
    if (run / "reconstruction.wav").exists() and (run / "residual.wav").exists():
        reconstruction, _ = AudioIO.read(run / "reconstruction.wav")
        residual, _ = AudioIO.read(run / "residual.wav")
        return reconstruction + residual
    if (run / "reference_sweep.wav").exists():
        sweep, _ = AudioIO.read(run / "reference_sweep.wav")
        return sweep_slice(sweep, index, n_samples, sample_rate, SWEEP_GAP)
    return None


@st.cache_data(show_spinner=False, max_entries=64)
def _figure(_builder, key: str) -> bytes:
    """A figure as PNG bytes, so the cache holds something picklable."""
    return plots.to_png(_builder())


def show(name: str, builder, key: str) -> None:
    st.image(_figure(builder, key), caption=name, width="stretch")


# -- which model --------------------------------------------------------------

store = RunStore.default()
default_path = st.session_state.get("run_path", "")
path = st.sidebar.text_input("Run directory or model file", value=default_path)

if not path:
    st.info("Pick a training on the Instruments page, or type a path here.")
    st.stop()

target = Path(path)
run: Run | None = None
if (target / "run.json").exists():
    run = store.find(target)
    model_path = run.model_path
elif target.is_file():
    model_path = target
else:
    st.error(f"{target} is neither a run directory nor a model file.")
    st.stop()

model = _model(str(model_path))
instrument = isinstance(model, InstrumentModel)
st.caption(f"`{model_path}`")

# -- what it is ---------------------------------------------------------------

memory = resident_bytes(model)
columns = st.columns(4)
columns[0].metric("In memory", f"{memory / 1024:.0f} kB", f"{model.n_bytes() / 1024:.0f} kB on disk")
columns[1].metric("Stored numbers", f"{model.n_scalars:,}", model.candidate.codec if not instrument else model.candidate.field_codec)
columns[2].metric("Partials", f"{model.bins.size}", f"{model.n_frames} frames")
columns[3].metric("Audio", f"{model.duration:.2f} s", f"{model.sample_rate} Hz")

st.code(model.candidate.label(), language="text")

if run is not None:
    metrics = {
        key: value
        for key, value in run.metadata.items()
        if key.endswith("_mse") or key in {"snr_db", "target_reached", "search_seconds"}
    }
    st.dataframe(pd.DataFrame([metrics]), hide_index=True, width="stretch")

# -- live ---------------------------------------------------------------------

st.header("Live")

velocity = None
layer_index = 0
if instrument:
    low, high = model.velocity_range
    if high > low:
        velocity = st.slider("Velocity", float(low), float(high), float(high))
    else:
        velocity = float(low)
        st.caption(f"one velocity was recorded ({velocity:g})")
    layer_index = int(np.argmin(np.abs(model.velocities - velocity)))

partials = st.slider(
    "Resonators", 1, int(model.bins.size), int(model.bins.size),
    help="Keep only the loudest N partials — the model rebuilding itself as you add them.",
)


def loudest(block: np.ndarray) -> np.ndarray:
    """Which rows survive the resonator slider."""
    if partials >= block.shape[0]:
        return np.arange(block.shape[0])
    return np.argsort(np.abs(block).max(axis=1))[::-1][:partials]


def components_at() -> np.ndarray:
    """The component block as the slider leaves it."""
    block = model.components(velocity) if instrument else model.components()
    if partials >= block.shape[0]:
        return block
    masked = np.zeros_like(block)
    kept = loudest(block)
    masked[kept] = block[kept]
    return masked


def trigger() -> tuple[np.ndarray, float, float]:
    """Strike it, and time the two halves separately, the way audio would.

    Triggering decodes; reading is transform and add. A sampler pays the first
    once and the second once per block, which is the whole reason they are
    timed apart.
    """
    started = time.perf_counter()
    voice = voice_for(model, velocity)
    if partials < voice.rows.shape[0]:
        kept = loudest(voice.rows)
        rows = np.zeros_like(voice.rows)
        rows[kept] = voice.rows[kept]
        voice.rows = rows
    triggered = time.perf_counter()

    audio = voice.render_rest()
    return audio, (triggered - started) * 1000.0, (time.perf_counter() - triggered) * 1000.0


audio, trigger_ms, read_ms = trigger()
st.audio(audio, sample_rate=model.sample_rate)

block = st.select_slider("Audio block", options=[64, 128, 256, 512, 1024], value=256)


@st.cache_data(show_spinner="measuring", max_entries=32)
def _cost(_model, key: str, block: int):
    return measure_live(_model, velocity, block=block, repeats=3)


@st.cache_data(show_spinner="measuring", max_entries=32)
def _polyphony(_model, key: str, block: int, voices: int):
    return measure_polyphony(_model, velocity, block=block, voices=voices)


cost = _cost(model, f"{model_path}:{velocity}", block)

live = st.columns(4)
live[0].metric("Trigger", f"{trigger_ms:.2f} ms", f"{cost.trigger_ms:.2f} ms median")
live[1].metric("Whole hit", f"{read_ms:.1f} ms", f"{cost.realtime_factor:.0f}x real time")
live[2].metric("Per block", f"{cost.block_mean_ms:.3f} ms", f"budget {cost.budget_ms:.2f} ms")
live[3].metric("One voice", f"{100 * cost.load:.1f}% of a core", f"room for {cost.voices:.0f}")

with st.expander("Polyphony"):
    st.dataframe(
        pd.DataFrame(
            [
                _polyphony(model, f"{model_path}:{velocity}", block, count)
                for count in (1, 4, 8, 16, 32)
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Voices are staggered, the way hits in a bar are: their expensive blocks do "
        "not line up. `worst_load` is the fraction of one callback's budget the "
        "worst block took."
    )

st.code(cost.report(), language="text")

# -- figures ------------------------------------------------------------------

st.header("Anatomy")

components = components_at()
frequencies = model.frequencies()
times = np.arange(model.n_frames) * model.candidate.hop / model.sample_rate
frame_rate = model.sample_rate / model.candidate.hop
key = f"{model_path}:{velocity}:{partials}"

reference = (
    _reference(str(run.path), layer_index, model.n_samples, model.sample_rate)
    if run is not None
    else None
)

anatomy, comparison, velocity_tab = st.tabs(["Partials", "Against the recording", "Velocity"])

with anatomy:
    show("Kept bins", lambda: plots.bin_map(model, reference, components), f"bins:{key}")
    show(
        "Envelopes",
        lambda: plots.envelope_heatmap(components, frequencies, times),
        f"env:{key}",
    )
    show(
        "The loudest partials",
        lambda: plots.envelope_traces(components, frequencies, times),
        f"traces:{key}",
    )
    show(
        "Decay per partial",
        lambda: plots.decay_times(components, frequencies, times),
        f"decay:{key}",
    )
    show(
        "Spectrum of the envelopes",
        lambda: plots.envelope_spectrum(components, frame_rate),
        f"envspec:{key}",
    )

with comparison:
    if reference is None:
        st.info("This run did not keep the recording, so there is nothing to compare against.")
    else:
        estimate = model.render(velocity) if instrument else model.render()
        error = float(np.sum((reference - estimate) ** 2) / max(np.sum(reference**2), 1e-30))
        st.metric("Relative error at this velocity", f"{error:.3e}", f"{-10 * np.log10(max(error, 1e-30)):.1f} dB SNR")

        st.audio(reference, sample_rate=model.sample_rate)
        st.caption("the recording")
        st.audio(reference - estimate, sample_rate=model.sample_rate)
        st.caption("what the model missed")

        show("Waveform", lambda: plots.waveform(reference, estimate, model.sample_rate), f"wave:{key}")
        show(
            "Spectrogram",
            lambda: plots.spectrogram_panel(reference, estimate, model.sample_rate),
            f"spectrogram:{key}",
        )
        show("Spectrum", lambda: plots.spectrum(reference, estimate, model.sample_rate), f"spectrum:{key}")
        show(
            "Error through the hit",
            lambda: plots.error_over_time(reference, estimate, model.sample_rate),
            f"errtime:{key}",
        )
        show(
            "Error by band",
            lambda: plots.error_by_frequency(reference, estimate, model.sample_rate),
            f"errfreq:{key}",
        )

with velocity_tab:
    if not instrument:
        st.info("A single hit has no velocity axis. Fit a drum to get one.")
    else:
        show("Velocity curves", lambda: plots.velocity_curves(model), f"curves:{model_path}")
        show("Response to velocity", lambda: plots.velocity_response(model), f"response:{model_path}")
        show("Spectrum against velocity", lambda: plots.velocity_spectrum(model), f"velspec:{model_path}")
        if run is not None:
            layers = pd.DataFrame(run.report().get("layers", []))
            if not layers.empty:
                st.dataframe(layers, hide_index=True, width="stretch")
