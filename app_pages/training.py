"""Fitting one drum to its samples, with the run visible while it happens.

One drum at a time, chosen from the dropdown. `f_static` and `t60` are
properties of a specific physical drum, so there is nothing a second drum could
contribute to a fit except a way to get them confused.

The fit runs in a subprocess (`drumsynth.fitting.worker`) and reports as it
goes, so this page can redraw as often as it likes without touching it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from drumsynth import AudioIO, DrumParams, DrumVoice
from drumsynth.fitting import DrumCatalogue, TrainingRun
from drumsynth.studio import Studio

studio = Studio.bootstrap()
st.session_state.setdefault("fit_output_dir", Path("out/fits"))

st.title("Training")

catalogue = DrumCatalogue.available()
if not catalogue:
    st.warning(
        "No drum manifests under `data/metadata/`. Run "
        "`python tools/import_samples.py <DrumModalSynth/data>` first.",
        icon=":material/folder_off:",
    )
    st.stop()


@st.cache_resource(show_spinner=False)
def _run_handle() -> TrainingRun:
    """One training process for the session, kept across reruns."""
    return TrainingRun(cwd=str(Path.cwd()))


run = _run_handle()


# -- setup --------------------------------------------------------------------

def _setup() -> None:
    st.caption(
        "Cymbals are not offered. A struck cymbal cascades energy from low "
        "modes into high ones over the first few hundred milliseconds, which a "
        "linear modal bank cannot do at any setting — see ARCHITECTURE.md §9."
    )

    labels = {
        f"{item['drum']}  ·  {item['samples']} samples, {item['bands']} velocity bands": item
        for item in catalogue
    }
    choice = st.selectbox("Drum", list(labels), key="fit_drum")
    entry = labels[choice]

    left, right = st.columns(2, gap="medium")
    with left:
        layers = st.slider(
            "Velocity layers", 3, 12, 6, key="fit_layers",
            help=(
                "One representative hit per layer, spread across the range. "
                "§8.2: never fit to a single hit — a parameter set tuned "
                "against one sample matches it and generalizes to nothing."
            ),
        )
        seconds = st.slider(
            "Seconds per hit", 1.0, 4.0, 2.5, 0.25, key="fit_seconds",
            help="The reference tom's slowest mode measured t60 ≈ 2.3 s.",
        )
        modes = st.slider("Max modes", 12, 40, 30, key="fit_modes",
                          help="§3.1 assumes 25-35 resolved partials.")
    with right:
        generations = st.slider(
            "Generations", 4, 60, 20, key="fit_generations",
            help="Stage 5 only: how long the joint refinement runs.",
        )
        population = st.slider("Population", 6, 24, 10, key="fit_population")
        workers = st.slider(
            "Worker processes", 1, 16, 4, key="fit_workers",
            help=(
                "Stage 5 evaluates a population per generation, which is "
                "embarrassingly parallel. Set this to your core count."
            ),
        )

    refine = st.toggle(
        "Polish every mode gain", value=True, key="fit_refine",
        help=(
            "Stage 2 already has a closed-form start from non-negative least "
            "squares. This adds coordinate descent on top — better, and most "
            "of the run time."
        ),
    )

    st.info(
        f"**{entry['drum']}** — {entry['samples']} samples across "
        f"{entry['bands']} velocity bands, v{entry['velocity_low']:.0f}"
        f"-v{entry['velocity_high']:.0f}. "
        "Stage 1 measures the frequencies from a soft layer (the drum at rest) "
        "and the damping from the loud one, then freezes both permanently.",
        icon=":material/info:",
    )

    settings = {
        "seconds": seconds, "max_layers": layers, "max_modes": modes,
        "generations": generations, "population": population, "workers": workers,
        "control_period": 64, "noise_bands": 4, "seed": 0, "refine_gains": refine,
    }

    output = Path(st.session_state.fit_output_dir) / f"{entry['drum']}.json"
    disabled = run.is_running
    if st.button(
        "Start training", icon=":material/play_arrow:", type="primary",
        disabled=disabled, width="content",
    ):
        run.start(entry["manifest"], output, settings)
        st.rerun()

    samples_root = Path("data/samples")
    if not samples_root.exists():
        st.error(
            "`data/samples/` is missing — the WAVs are gitignored and have to "
            "be regenerated locally with `python tools/import_samples.py "
            "<DrumModalSynth/data>`. Training needs the audio, not just the "
            "manifests.",
            icon=":material/error:",
        )


# -- live progress ------------------------------------------------------------

@st.fragment(run_every="1s")
def _progress() -> None:
    if run.ready is None and not run.is_running and run.error is None:
        return

    with st.container(border=True):
        header = st.container(horizontal=True, gap="medium")
        with header:
            st.markdown(f"**{(run.ready or {}).get('drum', '…')}**")
            st.markdown(f":gray[{run.phase}]")
            if run.is_running:
                st.button("Stop", icon=":material/stop:", on_click=run.stop,
                          key="fit_stop")

        st.progress(run.progress_fraction(), text=" ")

        if run.error is not None:
            st.error(run.error.get("message", "the fit failed"),
                     icon=":material/error:")
            with st.expander("Traceback"):
                st.code(
                    run.error.get("traceback", "") or run.stderr_tail(100),
                    language="text",
                )
            return

        steps = list(run.steps)[-6:]
        if steps:
            st.caption(" · ".join(steps[-3:]))

        if run.generations:
            frame = pd.DataFrame(
                {
                    "loss (dB)": [g["loss"] for g in run.generations],
                    "best (dB)": [g["best_loss"] for g in run.generations],
                },
                index=pd.Index(
                    [g["generation"] for g in run.generations], name="generation"
                ),
            )
            st.line_chart(frame, height=200)
            latest = run.generations[-1]
            with st.container(horizontal=True, gap="medium"):
                st.metric("Generation", latest["generation"])
                st.metric("Best loss", f"{latest['best_loss']:.2f} dB")
                st.metric("Elapsed", f"{latest.get('elapsed', 0):.0f} s")
        elif run.is_running:
            st.caption(
                ":gray[stage 5 has not started — stages 1 and 2 are direct "
                "measurement and a closed-form solve, not a search]"
            )


# -- results ------------------------------------------------------------------

def _velocity_table(result: dict) -> None:
    table = result.get("table", [])
    if not table:
        return
    frame = pd.DataFrame(table)
    st.dataframe(
        frame.rename(columns={
            "velocity": "v", "velocity_normalized": "vn",
            "amplitude": "amplitude", "contact_time_ms": "contact ms",
            "brightness_db_per_decade": "brightness dB/dec",
            "noise_db": "noise dB", "loss_db": "loss dB",
        }),
        width="stretch", hide_index=True,
    )


def _inspection(result: dict) -> None:
    inspection = result.get("inspection", {})
    passed = inspection.get("passed", False)
    verdict = inspection.get("verdict", "")

    (st.success if passed else st.warning)(
        f"**Stage 3 — {verdict}**", icon=":material/science:"
    )
    st.caption(
        "Stage 3 is the real experiment. Per-velocity fitting ALWAYS succeeds, "
        "which is why it tells you nothing on its own; the evidence is whether "
        "the fitted numbers move smoothly and in the physical direction."
    )
    for finding in inspection.get("findings", []):
        st.markdown(f"- {finding}")


def _report(result: dict, scored: dict) -> None:
    st.subheader("Report")
    with st.container(horizontal=True, gap="medium"):
        st.metric("Score", f"{scored['total']:.3f}")
        st.metric("STFT loss", f"{scored['stft_loss']:.2f} dB")
        st.metric("Modes", result["modes"])
        st.metric("Glide", f"{12 * np.log2(1 + result['tension']['k']):.2f} st"
                  if result["tension"]["k"] > 0 else "off")
        st.metric("Fitted in", f"{result['elapsed']:.0f} s")

    st.caption(
        "The score aggregates the WORST component across every velocity, not "
        "the mean — a parameter right for three hits and wrong for the fourth "
        "is broken, and averaging hides exactly that (§8.2)."
    )

    components = pd.DataFrame([
        {
            "component": item["name"],
            "score": item["value"] if item["available"] else None,
            "error": item["raw_error"] if item["available"] else None,
            "unit": item["unit"],
            "parameters": ", ".join(item["parameters"]),
        }
        for item in scored["components"]
    ])
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


def _comparison(result: dict) -> None:
    """The generated sound beside the sample it was fitted to."""
    st.subheader("Generated against the sample")
    params = DrumParams.from_dict(result["params"])
    sr = st.session_state.engine_settings.sr

    layers = (run.ready or {}).get("layers", [])
    if not layers:
        return
    labels = {f"v{layer['velocity']:.0f}": layer for layer in layers}
    picked = st.select_slider("Velocity", list(labels), key="fit_compare_v")
    layer = labels[picked]

    from drumsynth.fitting.trainer import FitResult  # noqa: F401  (typing only)

    generated = DrumVoice(params, sr, 64, seed=0).render_hit(
        result.get("seconds", 2.5) if isinstance(result.get("seconds"), float) else 2.5
    )
    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown("**Generated**")
        st.audio(AudioIO.normalize_peak(generated, -1.0), sample_rate=sr)
    with right:
        st.markdown("**Sample**")
        source = Path("data/samples")
        st.caption(layer.get("source", ""))
        found = list(source.rglob(layer.get("source", "___nope___")))
        if found:
            audio, file_sr = AudioIO.read(found[0], sr=sr)
            st.audio(AudioIO.normalize_peak(audio, -1.0), sample_rate=sr)
        else:
            st.caption(":gray[sample audio not found on disk]")


def _handoff(result: dict) -> None:
    st.subheader("Try it")
    st.caption(
        "Loads the fitted parameters into the live engine. Strike it, edit it "
        "on the Mixer page while it rings, and save it when it is right."
    )
    params = DrumParams.from_dict(result["params"])
    with st.container(horizontal=True, gap="small"):
        if st.button("Load into the live synth", icon=":material/graphic_eq:",
                     type="primary"):
            studio.params = params
            studio.clear_widget_state()
            studio.reset_monitor()
            st.session_state.preset_name = params.name or "fitted"
            studio.sync()
            st.success("Loaded — open the Mixer page and press Strike.",
                       icon=":material/check:")
        st.download_button(
            "Download DrumParams JSON",
            data=json.dumps(result["params"], indent=2),
            file_name=f"{result['drum']}.json",
            mime="application/json",
            icon=":material/save:",
        )
    if run.done and run.done.get("path"):
        st.caption(f":gray[saved to {run.done['path']}]")


# -- page ---------------------------------------------------------------------

if run.result is None:
    _setup()

_progress()

if run.result is not None:
    st.divider()
    _inspection(run.result)
    st.divider()
    st.subheader("Velocity table")
    st.caption(
        "Stage 2 fits the excitation at each velocity independently — "
        "`gain[]`, noise `level[]` and the contact time — with `f_static` and "
        "`t60` already frozen."
    )
    _velocity_table(run.result)

    if run.scored is not None:
        st.divider()
        _report(run.result, run.scored)
        st.divider()
        _comparison(run.result)
        st.divider()
        _handoff(run.result)

    st.divider()
    if st.button("Fit another drum", icon=":material/refresh:"):
        _run_handle.clear()
        st.rerun()
