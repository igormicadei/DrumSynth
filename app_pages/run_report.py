"""Shared rendering for one stored training run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from drumsynth import AudioIO, DrumParams
from drumsynth.fitting.comparison import Comparison
from drumsynth.fitting.runs import RunStore
from drumsynth.studio import Studio

from app_pages import plots

studio = Studio.bootstrap()
store = RunStore()


def velocity_table(result: dict) -> None:
    table = result.get("table", [])
    if not table:
        return
    st.dataframe(
        pd.DataFrame(table).rename(
            columns={
                "velocity": "v",
                "velocity_normalized": "vn",
                "amplitude": "amplitude",
                "contact_time_ms": "contact ms",
                "brightness_db_per_decade": "brightness dB/dec",
                "noise_db": "noise dB",
                "loss_db": "loss dB",
            }
        ),
        width="stretch",
        hide_index=True,
    )


def inspection(result: dict) -> None:
    data = result.get("inspection", {})
    passed = data.get("passed", False)
    verdict = data.get("verdict", "")
    (st.success if passed else st.warning)(
        f"**Stage 3 — {verdict}**", icon=":material/science:"
    )
    st.caption(
        "Stage 3 is the real experiment. Per-velocity fitting ALWAYS succeeds, "
        "which is why it tells you nothing on its own; the evidence is whether "
        "the fitted numbers move smoothly and in the physical direction."
    )
    for finding in data.get("findings", []):
        st.markdown(f"- {finding}")


def score_report(result: dict, scored: dict) -> None:
    st.subheader("Report")
    with st.container(horizontal=True, gap="medium"):
        st.metric("Score", f"{scored['total']:.3f}")
        st.metric("STFT loss", f"{scored['stft_loss']:.2f} dB")
        st.metric("Modes", result["modes"])
        st.metric(
            "Glide",
            (
                f"{12 * np.log2(1 + result['tension']['k']):.2f} st"
                if result["tension"]["k"] > 0
                else "off"
            ),
        )
        st.metric("Fitted in", f"{result['elapsed']:.0f} s")
    st.caption(
        "The score aggregates the WORST component across every velocity, not "
        "the mean — a parameter right for three hits and wrong for the fourth "
        "is broken, and averaging hides exactly that (§8.2)."
    )
    components = pd.DataFrame(
        [
            {
                "component": item["name"],
                "score": item["value"] if item["available"] else None,
                "error": item["raw_error"] if item["available"] else None,
                "unit": item["unit"],
                "parameters": ", ".join(item["parameters"]),
            }
            for item in scored["components"]
        ]
    )
    st.dataframe(components, width="stretch", hide_index=True)
    per_layer = scored.get("per_layer", [])
    if per_layer:
        st.markdown("**Per velocity**")
        st.line_chart(
            pd.DataFrame(
                {"total": [row["total"] for row in per_layer]},
                index=pd.Index([row["velocity"] for row in per_layer], name="velocity"),
            ),
            height=200,
        )
    with st.expander("Full ScoreCard"):
        st.code(scored.get("report", ""), language="text")
    for warning in result.get("warnings", []):
        st.warning(warning, icon=":material/warning:")


def stage_five(result: dict) -> None:
    st.subheader("Stage 5 — the search")
    generations = result.get("generations") or []
    if not generations:
        st.caption(":gray[stage 5 did not run]")
        return
    rows = [
        {"generation": g["index"], "loss": g["loss"], "best_loss": g["best_loss"]}
        for g in generations
    ]
    first, last = rows[0]["best_loss"], rows[-1]["best_loss"]
    with st.container(horizontal=True, gap="medium"):
        st.metric("Generations", len(rows), border=True)
        st.metric("Started at", f"{first:.3f} dB", border=True)
        st.metric(
            "Ended at",
            f"{last:.3f} dB",
            delta=f"{last - first:+.3f} dB",
            delta_color="inverse",
            border=True,
        )
        st.metric(
            "Reported loss",
            f"{result.get('stage5_loss', last):.3f} dB",
            border=True,
            help="Measured on a real render of the final parameters.",
        )
    st.altair_chart(plots.generation_chart(rows), width="stretch")
    if abs(last - first) < 1e-3:
        st.caption(
            ":gray[Flat. Stage 5 searches the velocity mapping with the "
            "modes and per-mode shape frozen.]"
        )


def timings(result: dict) -> None:
    data = result.get("timings") or {}
    if not data:
        return
    total = sum(data.values()) or 1.0
    frame = pd.DataFrame(
        [
            {"stage": name, "seconds": seconds, "share": 100.0 * seconds / total}
            for name, seconds in data.items()
        ]
    )
    st.dataframe(
        frame,
        width="stretch",
        hide_index=True,
        column_config={
            "seconds": st.column_config.NumberColumn("seconds", format="%.1f s"),
            "share": st.column_config.ProgressColumn(
                "share of the run", min_value=0.0, max_value=100.0, format="%.0f%%"
            ),
        },
    )
    device = str(result.get("device", "cpu"))
    peak = float(result.get("peak_vram_bytes", 0.0) or 0.0)
    stage5 = data.get("stage 5 — joint refinement", 0.0) / total
    if device == "cuda" and peak > 0:
        st.success(
            f"Stage 5 ran on CUDA and allocated {peak / 1024**2:.0f} MiB "
            f"of VRAM. It was {stage5:.0%} of the run.",
            icon=":material/memory:",
        )
    elif device == "cuda":
        st.warning(
            "The device was reported as CUDA but no tensor was allocated "
            "on it. Stage 5 ran on the CPU.",
            icon=":material/warning:",
        )
    else:
        st.info(
            f"Stage 5 ran on the CPU and was {stage5:.0%} of the run. Only "
            "stage 5 can move to a GPU.",
            icon=":material/info:",
        )


@st.cache_data(show_spinner=False, max_entries=32)
def model(result: dict) -> None:
    """The fitted drum itself: every resonator, every transient band, and what
    each of them does in time.

    The ScoreCard says how far off the fit is. This says WHAT IT IS — the
    numbers a person would have to type into the Mixer to reproduce it, and the
    two or three of them that explain most of what you hear.
    """
    st.subheader("The fitted model")
    params = result.get("params") or {}
    modes = params.get("modes", [])
    bands = params.get("noise", [])
    if not modes:
        st.caption(":gray[this run stored no parameters]")
        return

    live = [mode for mode in modes if mode["gain"] > 1e-8]
    silenced = len(modes) - len(live)
    loud = max(live, key=lambda mode: mode["gain"]) if live else None
    tension = params.get("tension", {})

    with st.container(horizontal=True, gap="medium"):
        st.metric("Resonators", f"{len(live)}", border=True,
                  delta=f"{silenced} silenced" if silenced else None,
                  delta_color="off",
                  help="A mode the gain solve pinned at the 1e-9 floor is a "
                       "slot stage 1 spent on a partial that was not there.")
        if loud:
            st.metric("Fundamental", f"{loud['f_static']:.1f} Hz", border=True,
                      help="The loudest resonator, not the lowest.")
            st.metric("Longest ring", f"{max(m['t60'] for m in live):.2f} s",
                      border=True)
        st.metric("Transient bands",
                  f"{sum(1 for band in bands if band['level'] > 1e-8)} of {len(bands)}",
                  border=True,
                  help="Bands the level solve kept. A band whose decay is far "
                       "too short cannot help at any level, so it gets "
                       "switched off — which is what used to happen to all of "
                       "them.")
        st.metric("Glide",
                  f"{12 * np.log2(1 + tension.get('k', 0.0)):.2f} st"
                  if tension.get("k", 0.0) > 0 else "off", border=True)
        st.metric("Output gain", f"{params.get('output_gain', 1.0):.3f}",
                  border=True)

    bank, transients, envelopes, raw = st.tabs(
        ["Resonator bank", "Transients", "Envelopes", "JSON"])

    with bank:
        st.altair_chart(plots.mode_chart(modes), width="stretch")
        st.caption(
            "One bar per resonator: frequency on a log axis, `gain` as height, "
            "`t60` as colour. Faded bars are modes the gain solve switched off."
        )
        st.markdown("**The damping curve**")
        st.altair_chart(
            plots.damping_chart(modes, result.get("damping_anchors")),
            width="stretch")
        st.caption(
            "Stage 1 does not fit a `t60` per mode — per-mode subspace damping "
            "measured 56% wrong. It fits ONE curve through measured band decays "
            "(the circles) and reads every mode off it."
        )
        st.dataframe(
            pd.DataFrame([
                {"f_static (Hz)": mode["f_static"], "gain": mode["gain"],
                 "gain (dB)": 20 * np.log10(max(mode["gain"], 1e-12)),
                 "t60 (s)": mode["t60"],
                 "": "silenced" if mode["gain"] <= 1e-8 else ""}
                for mode in sorted(modes, key=lambda m: m["f_static"])
            ]),
            width="stretch", hide_index=True,
            column_config={
                "f_static (Hz)": st.column_config.NumberColumn(format="%.2f"),
                "gain": st.column_config.NumberColumn(format="%.3e"),
                "gain (dB)": st.column_config.NumberColumn(format="%.1f"),
                "t60 (s)": st.column_config.NumberColumn(format="%.3f"),
            },
        )

    with transients:
        st.caption(
            "§4 fixes the band edges; the fit moves each band's level, and its "
            "decay where the residual — the reference minus the modal bank — "
            "actually shows one."
        )
        st.dataframe(
            pd.DataFrame([
                {"band": f"{band['f_low']:.0f}-{band['f_high']:.0f} Hz",
                 "level": band["level"],
                 "level (dB)": 20 * np.log10(max(band["level"], 1e-12)),
                 "t60 (ms)": band["t60"] * 1000.0,
                 "": "off" if band["level"] <= 1e-8 else ""}
                for band in bands
            ]),
            width="stretch", hide_index=True,
            column_config={
                "level": st.column_config.NumberColumn(format="%.3e"),
                "level (dB)": st.column_config.NumberColumn(format="%.1f"),
                "t60 (ms)": st.column_config.NumberColumn(format="%.0f"),
            },
        )
        for note in result.get("noise_notes", []):
            st.markdown(f"- {note}")

    st.markdown("**Stage 1, checked against the residual**")
    st.caption(
        "Stage 1 is the ceiling on everything after it — no later stage moves "
        "`f_static` — so the residual, which is by construction everything the "
        "model cannot produce, is where a missed partial shows up. Peaks that "
        "ring are proposed as modes and kept only if re-fitting earns it on "
        "the layers that were not free to fit."
    )
    for note in result.get("residual_notes", []) or ["(not recorded for this run)"]:
        st.markdown(f"- {note}")

    with envelopes:
        st.altair_chart(
            plots.bank_envelope_chart(params, result.get("seconds", 2.5)),
            width="stretch")
        st.caption(
            "Every part's own decay, before anything is summed — the drum as a "
            "set of exponentials. A transient band whose line falls off the "
            "bottom in the first few milliseconds while the resonators ring for "
            "two seconds is a click, not a strike."
        )

    with raw:
        st.code(json.dumps(params, indent=2), language="json")


def comparison_data(directory: str, velocity: float, sr: int) -> dict | None:
    record = RunStore().read(directory)
    if record is None:
        return None
    generated, reference = record.audio(velocity, sr)
    if generated is None or reference is None:
        return None
    comparison = Comparison(reference, generated, sr)
    envelopes = comparison.envelopes()
    return {
        "generated": generated,
        "reference": reference,
        "envelope": plots.envelope_chart(comparison),
        "waveform": plots.waveform_chart(comparison),
        "spectrum": plots.spectrum_chart(comparison),
        "band_decay": plots.band_decay_chart(comparison),
        "spectrograms": plots.spectrograms(comparison),
        "band_distance": comparison.band_distance(),
        "crest": (envelopes.reference_crest_db, envelopes.generated_crest_db),
        "peak_ms": (envelopes.reference_peak_ms, envelopes.generated_peak_ms),
    }


def comparison(record) -> None:
    st.subheader("Generated against the sample")
    st.caption(
        "Both signals are level-matched first — absolute level is a mic "
        "preamp setting — and every pair is drawn on one pair of axes."
    )
    layers = record.layers()
    if not layers:
        st.caption(":gray[this run stored no audio]")
        return
    sr = int(
        record.summary.get("sr")
        or getattr(st.session_state.get("engine_settings"), "sr", 44100)
    )
    labels = {f"v{layer['velocity']:.0f}": layer["velocity"] for layer in layers}
    picked = labels[
        st.select_slider(
            "Velocity",
            list(labels),
            key=f"cmp_v_{record.started}",
            value=list(labels)[len(labels) // 2],
        )
    ]
    data = comparison_data(str(record.directory), float(picked), sr)
    if data is None:
        st.caption(":gray[audio for this layer is not on disk]")
        return
    with st.container(horizontal=True, gap="medium"):
        st.metric("Band distance", f"{data['band_distance']:.2f} dB", border=True)
        st.metric("Crest (sample)", f"{data['crest'][0]:.1f} dB", border=True)
        st.metric("Crest (generated)", f"{data['crest'][1]:.1f} dB", border=True)
        st.metric("Peak at (sample)", f"{data['peak_ms'][0]:.1f} ms", border=True)
        st.metric("Peak at (generated)", f"{data['peak_ms'][1]:.1f} ms", border=True)
    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown("**Sample**")
        st.audio(AudioIO.normalize_peak(data["reference"], -1.0), sample_rate=sr)
    with right:
        st.markdown("**Generated**")
        st.audio(AudioIO.normalize_peak(data["generated"], -1.0), sample_rate=sr)
    waveform, spectrum, decay, spectrogram = st.tabs(
        ["Waveform", "Spectrum", "Decay", "Spectrogram"]
    )
    with waveform:
        st.altair_chart(data["waveform"], width="stretch")
        st.markdown("**Envelope**")
        st.altair_chart(data["envelope"], width="stretch")
    with spectrum:
        st.altair_chart(data["spectrum"], width="stretch")
    with decay:
        st.altair_chart(data["band_decay"], width="stretch")
    with spectrogram:
        grams = data["spectrograms"]
        st.caption(
            f"Log frequency, {grams['low_hz']:.0f} Hz to "
            f"{grams['high_hz'] / 1000:.0f} kHz, over {grams['seconds']:.2f} s."
        )
        first, second = st.columns(2, gap="medium")
        with first:
            st.markdown("**Sample**")
            st.image(grams["sample"], width="stretch")
        with second:
            st.markdown("**Generated**")
            st.image(grams["generated"], width="stretch")
        st.markdown("**Generated minus sample**")
        st.image(grams["difference"], width=900)


def handoff(record) -> None:
    st.subheader("Try it")
    st.caption(
        "Loads the fitted parameters into the live engine. Strike it, edit "
        "it on the Mixer page while it rings, and save it when it is right."
    )
    try:
        params = record.params()
    except (OSError, ValueError) as error:
        st.error(
            f"could not read this run's parameters: {error}", icon=":material/error:"
        )
        return
    with st.container(horizontal=True, gap="small"):
        if st.button(
            "Load into the live synth",
            icon=":material/graphic_eq:",
            type="primary",
            key=f"load_{record.started}",
        ):
            if "engine_settings" not in st.session_state:
                Studio.bootstrap()
            studio.params = params
            studio.clear_widget_state()
            studio.reset_monitor()
            st.session_state.preset_name = params.name or record.drum
            studio.sync()
            st.success("Loaded — press Strike in the sidebar.", icon=":material/check:")
        st.download_button(
            "Download DrumParams JSON",
            data=json.dumps(params.to_dict(), indent=2),
            file_name=f"{record.drum}.json",
            mime="application/json",
            icon=":material/save:",
            key=f"dl_{record.started}",
        )
    st.caption(f":gray[stored at {record.directory}]")


def render(record) -> None:
    result, scored = record.summary, record.score
    inspection(result)
    st.divider()
    st.subheader("Velocity table")
    st.caption(
        "Stage 2 fits the excitation at each velocity independently — "
        "with f_static and t60 already frozen."
    )
    velocity_table(result)
    st.divider()
    model(result)
    if scored:
        st.divider()
        score_report(result, scored)
    st.divider()
    stage_five(result)
    st.divider()
    st.subheader("Where the time went")
    timings(result)
    st.divider()
    comparison(record)
    st.divider()
    handoff(record)
