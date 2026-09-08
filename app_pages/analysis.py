"""What the drum you just built actually measures.

Renders a hit offline — the live engine keeps playing, untouched — and runs the
same analysis chain the scorer uses. Metrics say where to look; ears remain the
acceptance test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from drumsynth import (
    Analyzer,
    AudioIO,
    BandDecayAnalyzer,
    DrumVoice,
    EnvelopeAnalyzer,
    GlideAnalyzer,
)
from drumsynth.studio import Studio

studio = Studio.bootstrap()
params = studio.params

st.title("Analysis")
st.caption(
    "Rendered offline from the current parameters. The live engine is not "
    "touched, so you can keep playing while this runs."
)

controls = st.container(horizontal=True, gap="medium")
with controls:
    seconds = st.slider("Length (s)", 0.5, 8.0, 4.0, 0.5, key="an_seconds")
    amplitude = st.slider("Strike", 0.05, 1.5, 1.0, 0.05, key="an_amplitude")


@st.cache_data(show_spinner="Rendering…", max_entries=8, ttl="10m")
def render(params_dict: dict, seconds: float, amplitude: float, sr: int) -> np.ndarray:
    from drumsynth import DrumParams

    voice = DrumVoice(DrumParams.from_dict(params_dict), sr, control_period=64, seed=0)
    return voice.render_hit(seconds, amplitude)


sr = st.session_state.engine_settings.sr
audio = render(params.to_dict(), seconds, amplitude, sr)

st.audio(AudioIO.normalize_peak(audio, -1.0), sample_rate=sr)

envelope = EnvelopeAnalyzer(sr, 0.001).analyze(audio)
glide = GlideAnalyzer(sr, (params.fundamental() * 0.6, params.fundamental() * 1.6)).analyze(audio)

with st.container(horizontal=True, gap="medium"):
    st.metric("Peak", f"{np.max(np.abs(audio)):.3f}")
    st.metric("Envelope peak", f"{envelope.peak_time * 1000:.1f} ms")
    st.metric("Crest", f"{envelope.crest_factor_db:.1f} dB")
    st.metric("Glide depth", f"{glide.depth_cents / 100:+.2f} st")
    st.metric("f0 settles at", f"{glide.f_asymptote:.1f} Hz")

reference, generated = st.tabs(
    [":material/show_chart: Decay", ":material/timeline: Glide and envelope"]
)

with reference:
    bands = BandDecayAnalyzer(sr).analyze(audio, -110.0)
    rows = [
        {
            "band": f"{band.f_low:.0f}-{band.f_high:.0f} Hz",
            "t60 s": round(band.t60, 3),
            "slope dB/s": round(band.slope_db_s, 1),
            "level dB": round(band.level_db, 1),
            "r²": round(band.r_squared, 3),
        }
        for band in bands
        if band.is_valid
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption(
        "A low r² is informative, not a failure: it means the band holds "
        "several modes with different t60, which is why there is no "
        "decay_shape parameter. The reference floor tom measured 2.33 s at "
        "40-130 Hz falling to 0.31 s by 2 kHz."
    )

with generated:
    times, freqs = glide.reliable_track()
    if times.size:
        st.line_chart(
            pd.DataFrame({"f0 (Hz)": freqs}, index=pd.Index(times, name="time (s)")),
            height=240,
        )
    else:
        st.info("No reliable glide frames in range.", icon=":material/info:")

    step = max(1, len(envelope.times) // 900)
    st.line_chart(
        pd.DataFrame(
            {"envelope (dB)": envelope.db[::step]},
            index=pd.Index(envelope.times[::step], name="time (s)"),
        ),
        height=240,
    )
    st.caption(
        "The envelope peaks on the first sample: every mode is impulse-excited "
        "at phase zero, so the sum starts at its maximum and can only fall. The "
        "reference hit peaks at 9.8 ms — see docs/FINDINGS.md."
    )
