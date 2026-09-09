"""Charts for the training report.

Altair rather than matplotlib for anything with axes, because the app is a
Streamlit app and Vega charts are interactive, themed and readable at any width
without a figure size guess. The one exception is the spectrogram: it is an
image, and 96 bands by 250 frames is 24 000 rectangles that Vega would draw one
at a time — so it is rendered straight to RGB and shown with `st.image`.

Every chart here takes a `Comparison` and draws both signals on one pair of
axes. Two charts side by side, each auto-scaled, is the most common way to make
a bad fit look fine.
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

from drumsynth.fitting.comparison import Comparison

#: The sample is the thing being matched, so it is the neutral colour and the
#: generated drum is the one that stands out against it.
REFERENCE = "#8c8c8c"
GENERATED = "#ff6a2b"
SERIES = ["sample", "generated"]
SCALE = alt.Scale(domain=SERIES, range=[REFERENCE, GENERATED])


def _long(x: np.ndarray, reference: np.ndarray, generated: np.ndarray,
          x_name: str, y_name: str) -> pd.DataFrame:
    return pd.DataFrame({
        x_name: np.concatenate([x, x]),
        y_name: np.concatenate([reference, generated]),
        "signal": ["sample"] * len(x) + ["generated"] * len(x),
    })


def envelope_chart(comparison: Comparison, height: int = 220) -> alt.Chart:
    """Amplitude against time, in dB.

    dB, not linear: on a linear axis everything after the first thirty
    milliseconds is a flat line at zero, and the decay is the part of a drum
    that takes two seconds.
    """
    envelopes = comparison.envelopes()
    frame = _long(envelopes.times, envelopes.reference_db,
                  envelopes.generated_db, "seconds", "dB")
    return (
        alt.Chart(frame)
        .mark_line(strokeWidth=1.6)
        .encode(
            x=alt.X("seconds:Q", title="seconds"),
            y=alt.Y("dB:Q", title="level (dB)",
                    scale=alt.Scale(domain=[envelopes.floor_db, 2])),
            color=alt.Color("signal:N", scale=SCALE, title=None),
            tooltip=["signal:N", alt.Tooltip("seconds:Q", format=".3f"),
                     alt.Tooltip("dB:Q", format=".1f")],
        )
        .properties(height=height)
    )


def waveform_chart(comparison: Comparison, height: int = 200) -> alt.Chart:
    """The min/max envelope a DAW draws, both signals stacked."""
    waves = comparison.waveforms()
    frames = []
    for name, low, high in (
        ("sample", waves["reference_low"], waves["reference_high"]),
        ("generated", waves["generated_low"], waves["generated_high"]),
    ):
        frames.append(pd.DataFrame({
            "seconds": waves["times"], "low": low, "high": high,
            "signal": name,
        }))
    frame = pd.concat(frames, ignore_index=True)
    return (
        alt.Chart(frame)
        .mark_area(opacity=0.85)
        .encode(
            x=alt.X("seconds:Q", title="seconds"),
            y=alt.Y("low:Q", title="amplitude"),
            y2="high:Q",
            color=alt.Color("signal:N", scale=SCALE, title=None),
            row=alt.Row("signal:N", title=None,
                        header=alt.Header(labelFontWeight="bold")),
        )
        .properties(height=height // 2)
    )


def spectrum_chart(comparison: Comparison, height: int = 260) -> alt.Chart:
    """Long-term average magnitude on a log frequency axis.

    Averaged over the whole hit rather than taken from one frame: a single
    frame samples the noise as much as the drum, and §6.5 is explicit that
    comparing two noise realizations measures the seed.
    """
    spectra = comparison.spectra()
    frame = _long(spectra.freqs, spectra.reference_db, spectra.generated_db,
                  "Hz", "dB")
    frame = frame[frame["Hz"] >= 25.0]
    return (
        alt.Chart(frame)
        .mark_line(strokeWidth=1.2, opacity=0.9)
        .encode(
            x=alt.X("Hz:Q", title="frequency (Hz)",
                    scale=alt.Scale(type="log", domain=[25, 20000])),
            y=alt.Y("dB:Q", title="magnitude (dB)",
                    scale=alt.Scale(domain=[-80, 2])),
            color=alt.Color("signal:N", scale=SCALE, title=None),
            tooltip=["signal:N", alt.Tooltip("Hz:Q", format=".0f"),
                     alt.Tooltip("dB:Q", format=".1f")],
        )
        .properties(height=height)
    )


def band_decay_chart(comparison: Comparison, height: int = 220) -> alt.Chart:
    """t60 per band — the quantity stage 1 fits its damping curve through."""
    decays = comparison.band_decays()
    if not len(decays.centres):
        return alt.Chart(pd.DataFrame({"Hz": [], "t60": [], "signal": []}))
    frame = _long(decays.centres, decays.reference_t60, decays.generated_t60,
                  "Hz", "t60")
    return (
        alt.Chart(frame)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("Hz:Q", title="band centre (Hz)",
                    scale=alt.Scale(type="log")),
            y=alt.Y("t60:Q", title="t60 (s)"),
            color=alt.Color("signal:N", scale=SCALE, title=None),
            tooltip=["signal:N", alt.Tooltip("Hz:Q", format=".0f"),
                     alt.Tooltip("t60:Q", format=".2f")],
        )
        .properties(height=height)
    )


def generation_chart(generations: list[dict], height: int = 240) -> alt.Chart:
    """Loss per generation, with the running best.

    A flat line here is not automatically bad news and not automatically good
    news, which is why both series are drawn: `best` flat while `loss` moves
    means the search is exploring and finding nothing better.
    """
    frame = pd.concat([
        pd.DataFrame({
            "generation": [g["generation"] for g in generations],
            "dB": [g["loss"] for g in generations], "series": "this generation",
        }),
        pd.DataFrame({
            "generation": [g["generation"] for g in generations],
            "dB": [g["best_loss"] for g in generations], "series": "best so far",
        }),
    ], ignore_index=True)
    return (
        alt.Chart(frame)
        .mark_line(point=False, strokeWidth=2)
        .encode(
            x=alt.X("generation:Q", title="generation",
                    scale=alt.Scale(nice=False)),
            y=alt.Y("dB:Q", title="band loss (dB)",
                    scale=alt.Scale(zero=False)),
            color=alt.Color(
                "series:N", title=None,
                scale=alt.Scale(domain=["this generation", "best so far"],
                                range=["#8c8c8c", "#ff6a2b"])),
            tooltip=["series:N", "generation:Q", alt.Tooltip("dB:Q", format=".3f")],
        )
        .properties(height=height)
    )


# =============================================================================
# The spectrogram
# =============================================================================


#: A perceptually-ordered ramp, dark to bright. Written out rather than pulled
#: from matplotlib so the report does not need a plotting dependency.
_MAGMA = np.array([
    (0.001, 0.000, 0.014), (0.078, 0.045, 0.163), (0.208, 0.072, 0.354),
    (0.351, 0.081, 0.430), (0.487, 0.134, 0.428), (0.629, 0.184, 0.398),
    (0.767, 0.246, 0.335), (0.882, 0.343, 0.257), (0.958, 0.489, 0.243),
    (0.994, 0.653, 0.375), (0.997, 0.812, 0.567), (0.987, 0.991, 0.750),
])


def _colorize(image: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """(frames, bands) dB -> (bands, frames, 3) uint8, low frequency at the
    bottom and time running left to right."""
    span = max(vmax - vmin, 1e-9)
    normalized = np.clip((image - vmin) / span, 0.0, 1.0)
    position = normalized * (len(_MAGMA) - 1)
    low = np.floor(position).astype(int)
    high = np.minimum(low + 1, len(_MAGMA) - 1)
    blend = (position - low)[..., None]
    rgb = _MAGMA[low] * (1 - blend) + _MAGMA[high] * blend
    return (np.flipud(rgb.transpose(1, 0, 2)) * 255).astype(np.uint8)


def spectrograms(comparison: Comparison) -> dict:
    """Both spectrograms and their difference, as RGB arrays on ONE scale.

    The shared scale is the whole point. Two images each auto-scaled to their
    own maximum look alike however different they are, which is exactly the
    mistake this report exists to avoid.
    """
    grams = comparison.spectrograms()
    difference = np.clip(grams.difference_db, -24, 24)
    return {
        "sample": _colorize(grams.reference_db, grams.vmin, grams.vmax),
        "generated": _colorize(grams.generated_db, grams.vmin, grams.vmax),
        "difference": _colorize(difference, -24, 24),
        "vmin": grams.vmin, "vmax": grams.vmax,
        "seconds": float(grams.times[-1]) if len(grams.times) else 0.0,
        "low_hz": float(grams.freqs[0]) if len(grams.freqs) else 0.0,
        "high_hz": float(grams.freqs[-1]) if len(grams.freqs) else 0.0,
    }
