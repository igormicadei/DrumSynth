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
from drumsynth.fitting import DeviceChoice, DrumCatalogue, TrainingRun
from drumsynth.fitting.comparison import Comparison
from drumsynth.fitting.runs import RunStore
from drumsynth.fitting.telemetry import GpuMonitor
from drumsynth.studio import Studio

from app_pages import plots

studio = Studio.bootstrap()
st.session_state.setdefault("fit_output_dir", Path("out/fits"))

st.title("Training")

store = RunStore()


@st.cache_resource(show_spinner=False)
def _gpu() -> GpuMonitor:
    """One NVML handle for the session. Opening it per rerun would be a
    driver call every second."""
    return GpuMonitor()

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

        options = DeviceChoice.options()
        labels = {label: value for value, label in options}
        device = labels[st.selectbox(
            "Stage 5 device", list(labels), key="fit_device",
            help=(
                "Stage 5 evaluates a whole generation at once — one matrix "
                "product and one batched STFT per velocity layer. On CUDA the "
                "bases stay in VRAM for the run and only the candidate gains "
                "cross the bus. Stages 1-4 are measurement and closed-form "
                "solves and run on the CPU either way."
            ),
        )]

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
        "generations": generations, "population": population, "device": device,
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

def _gpu_panel() -> None:
    """What the GPU is doing right now, from NVML.

    This exists because "I installed the CUDA wheel and the GPU sits at 0%" is
    not answerable from inside the fit. Two different things can be true — the
    device is idle because stage 5 has not started, or because stage 5 is
    running on the CPU — and only a live reading separates them. The report
    below adds the other half: how much VRAM the fit actually allocated.
    """
    monitor = _gpu()
    if not monitor.available:
        st.caption(f":gray[GPU telemetry unavailable — {monitor.reason}]")
        return

    samples = monitor.sample()
    if not samples:
        st.caption(":gray[NVML started but reported no devices]")
        return

    for sample in samples:
        with st.container(border=True):
            st.markdown(f"**{sample.name}**")
            with st.container(horizontal=True, gap="medium"):
                st.metric("GPU", f"{sample.utilization:.0f}%")
                st.metric(
                    "VRAM",
                    f"{sample.memory_used / 1024**3:.1f} / "
                    f"{sample.memory_total / 1024**3:.1f} GiB",
                )
                if sample.temperature == sample.temperature:
                    st.metric("Temp", f"{sample.temperature:.0f} °C")
                if sample.power == sample.power:
                    st.metric("Power", f"{sample.power:.0f} W")
            st.progress(min(sample.memory_fraction, 1.0),
                        text=f"VRAM {sample.memory_fraction:.0%}")


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

        left, right = st.columns([3, 2], gap="medium")

        with left:
            if run.generations:
                latest = run.generations[-1]
                best = min(run.generations, key=lambda g: g["best_loss"])
                first = run.generations[0]["best_loss"]
                gained = first - latest["best_loss"]

                with st.container(horizontal=True, gap="medium"):
                    st.metric("Generation", latest["generation"], border=True)
                    st.metric(
                        "Best loss", f"{latest['best_loss']:.3f} dB",
                        delta=f"{-gained:.3f} dB" if gained else None,
                        delta_color="inverse", border=True,
                        help="Lower is closer. The floor is about 1.1 dB — two "
                             "renders of identical parameters with different "
                             "noise seeds sit that far apart.",
                    )
                    st.metric("Best at", f"gen {best['generation']}", border=True,
                              help="Where the best candidate was found. If this "
                                   "stops moving, the search has converged.")
                    st.metric("Stage 5 time", f"{latest.get('elapsed', 0):.0f} s",
                              border=True)

                st.altair_chart(plots.generation_chart(run.generations),
                                width="stretch")
                if gained <= 1e-3 and len(run.generations) > 4:
                    st.caption(
                        ":gray[Flat. Stage 5 only moves the velocity mapping — "
                        "six numbers — with the modes and the per-mode shape "
                        "frozen. A flat chart means the remaining error is "
                        "somewhere it cannot reach, not that the fit is good.]"
                    )
            elif run.is_running:
                st.caption(
                    ":gray[stage 5 has not started — stages 1 and 2 are direct "
                    "measurement and a closed-form solve, not a search]"
                )

        with right:
            _gpu_panel()


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


def _stage_five(result: dict) -> None:
    """The search, after the fact."""
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
        st.metric("Ended at", f"{last:.3f} dB",
                  delta=f"{last - first:+.3f} dB", delta_color="inverse",
                  border=True)
        st.metric("Reported loss", f"{result.get('stage5_loss', last):.3f} dB",
                  border=True,
                  help="Measured on a REAL render of the final parameters, not "
                       "on the linearized basis the search used. The basis is "
                       "an approximation, and letting it grade itself is how a "
                       "fit comes to look better than it is.")

    st.altair_chart(plots.generation_chart(rows), width="stretch")
    if abs(last - first) < 1e-3:
        st.caption(
            ":gray[Flat. Stage 5 searches SIX numbers — the velocity mapping — "
            "with the modes, the damping and the per-mode gain shape frozen by "
            "the stages before it. A flat chart means the remaining error is "
            "somewhere those six numbers cannot reach, not that the fit is "
            "good. The per-layer losses in the velocity table are where to "
            "look next.]"
        )


def _timings(result: dict) -> None:
    """Where the run's time went, per stage.

    The reason this is in the report rather than a debug log: only stage 5 can
    use a GPU, and "why is my GPU idle" is answered by the share of the run
    stage 5 occupies together with the VRAM it actually allocated. Both are
    here.
    """
    timings = result.get("timings") or {}
    if not timings:
        return

    total = sum(timings.values()) or 1.0
    frame = pd.DataFrame([
        {"stage": name, "seconds": seconds, "share": 100.0 * seconds / total}
        for name, seconds in timings.items()
    ])
    st.dataframe(
        frame, width="stretch", hide_index=True,
        column_config={
            "seconds": st.column_config.NumberColumn("seconds", format="%.1f s"),
            "share": st.column_config.ProgressColumn(
                "share of the run", min_value=0.0, max_value=100.0,
                format="%.0f%%"),
        },
    )

    device = str(result.get("device", "cpu"))
    peak = float(result.get("peak_vram_bytes", 0.0) or 0.0)
    stage5 = timings.get("stage 5 — joint refinement", 0.0) / total

    if device == "cuda" and peak > 0:
        st.success(
            f"Stage 5 ran on CUDA and allocated {peak / 1024**2:.0f} MiB of "
            f"VRAM. It was {stage5:.0%} of the run; the rest is stages 1-4, "
            "which are LAPACK on small matrices and stay on the CPU.",
            icon=":material/memory:",
        )
    elif device == "cuda":
        st.warning(
            "The device was reported as CUDA but no tensor was ever allocated "
            "on it. Stage 5 ran on the CPU. Check that the training subprocess "
            "uses the same interpreter as this app and that `torch` there is a "
            "cu-tagged wheel.",
            icon=":material/warning:",
        )
    else:
        st.info(
            f"Stage 5 ran on the CPU and was {stage5:.0%} of the run. Only "
            "stage 5 can move to a GPU — stages 1-4 are ESPRIT, band-decay "
            "regressions and an NNLS solve.",
            icon=":material/info:",
        )


# =============================================================================
# The comparison
# =============================================================================


@st.cache_data(show_spinner=False, max_entries=32)
def _comparison_data(directory: str, velocity: float, sr: int) -> dict | None:
    """Everything the comparison draws, from a stored run.

    Cached on the run directory rather than on the audio, so flipping between
    velocities and between runs is instant and re-reading a two-second WAV and
    running four analyses is not repeated on every rerun.
    """
    record = RunStore().read(directory)
    if record is None:
        return None
    generated, reference = record.audio(velocity, sr)
    if generated is None or reference is None:
        return None

    comparison = Comparison(reference, generated, sr)
    envelopes = comparison.envelopes()
    return {
        "generated": generated, "reference": reference,
        "envelope": plots.envelope_chart(comparison),
        "waveform": plots.waveform_chart(comparison),
        "spectrum": plots.spectrum_chart(comparison),
        "band_decay": plots.band_decay_chart(comparison),
        "spectrograms": plots.spectrograms(comparison),
        "band_distance": comparison.band_distance(),
        "crest": (envelopes.reference_crest_db, envelopes.generated_crest_db),
        "peak_ms": (envelopes.reference_peak_ms, envelopes.generated_peak_ms),
    }


def _comparison(record) -> None:
    """The generated sound beside the sample, in every view that shows
    something the ScoreCard cannot."""
    st.subheader("Generated against the sample")
    st.caption(
        "Both signals are level-matched first — absolute level is a mic preamp "
        "setting — and every pair is drawn on ONE pair of axes. Two charts side "
        "by side, each auto-scaled, is the most common way to make a bad fit "
        "look fine."
    )

    layers = record.layers()
    if not layers:
        st.caption(":gray[this run stored no audio]")
        return

    sr = int(record.summary.get("sr", st.session_state.engine_settings.sr))
    labels = {f"v{layer['velocity']:.0f}": layer["velocity"] for layer in layers}
    picked = labels[st.select_slider(
        "Velocity", list(labels), key=f"cmp_v_{record.started}",
        value=list(labels)[len(labels) // 2],
    )]

    data = _comparison_data(str(record.directory), float(picked), sr)
    if data is None:
        st.caption(":gray[audio for this layer is not on disk]")
        return

    with st.container(horizontal=True, gap="medium"):
        st.metric("Band distance", f"{data['band_distance']:.2f} dB", border=True,
                  help="The loss stage 5 minimizes, for this layer. The floor "
                       "is about 1.1 dB.")
        st.metric("Crest (sample)", f"{data['crest'][0]:.1f} dB", border=True)
        st.metric("Crest (generated)", f"{data['crest'][1]:.1f} dB", border=True,
                  help="Peak over RMS. A modal bank struck at cosine phase "
                       "peaks on the first sample, so this runs high — "
                       "finding #1 in docs/FINDINGS.md.")
        st.metric("Peak at (sample)", f"{data['peak_ms'][0]:.1f} ms", border=True)
        st.metric("Peak at (generated)", f"{data['peak_ms'][1]:.1f} ms",
                  border=True)

    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown("**Sample**")
        st.audio(AudioIO.normalize_peak(data["reference"], -1.0), sample_rate=sr)
    with right:
        st.markdown("**Generated**")
        st.audio(AudioIO.normalize_peak(data["generated"], -1.0), sample_rate=sr)

    waveform, spectrum, decay, spectrogram = st.tabs(
        ["Waveform", "Spectrum", "Decay", "Spectrogram"])

    with waveform:
        st.altair_chart(data["waveform"], width="stretch")
        st.markdown("**Envelope**")
        st.altair_chart(data["envelope"], width="stretch")
        st.caption(
            "The envelope is in dB. On a linear axis everything past the first "
            "thirty milliseconds is a flat line at zero, and the decay is the "
            "part of a drum that lasts two seconds."
        )

    with spectrum:
        st.altair_chart(data["spectrum"], width="stretch")
        st.caption(
            "Averaged over the whole hit, not one frame: a single frame samples "
            "the noise as much as the drum (§6.5). Peaks that line up are modes "
            "the fit found; peaks in the sample with nothing under them are "
            "modes it missed."
        )

    with decay:
        st.altair_chart(data["band_decay"], width="stretch")
        st.caption(
            "t60 per band — the quantity stage 1 fits its damping curve "
            "through, and the one that decides whether the drum rings for the "
            "right length of time at each frequency."
        )

    with spectrogram:
        grams = data["spectrograms"]
        st.caption(
            f"Log frequency, {grams['low_hz']:.0f} Hz to "
            f"{grams['high_hz'] / 1000:.0f} kHz, over {grams['seconds']:.2f} s. "
            f"Both images share one dB scale ({grams['vmin']:.0f} to "
            f"{grams['vmax']:.0f} dB) — auto-scaling each one separately makes "
            "any two drums look alike."
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
        st.caption(
            ":gray[Dark is quieter than the sample, bright is louder, ±24 dB. "
            "A horizontal bright line is a mode the fit put in the wrong place "
            "or left ringing too long. Fine vertical striping is the noise "
            "bank running on a different seed and is not a defect — §6.5.]"
        )


def _handoff(record) -> None:
    st.subheader("Try it")
    st.caption(
        "Loads the fitted parameters into the live engine. Strike it, edit it "
        "on the Mixer page while it rings, and save it when it is right. Every "
        "stored fit is also in the sidebar's **Trained drums** list, from any "
        "page and in any later session."
    )
    try:
        params = record.params()
    except (OSError, ValueError) as error:
        st.error(f"could not read this run's parameters: {error}",
                 icon=":material/error:")
        return

    with st.container(horizontal=True, gap="small"):
        if st.button("Load into the live synth", icon=":material/graphic_eq:",
                     type="primary", key=f"load_{record.started}"):
            studio.params = params
            studio.clear_widget_state()
            studio.reset_monitor()
            st.session_state.preset_name = params.name or record.drum
            studio.sync()
            st.success("Loaded — press Strike in the sidebar.",
                       icon=":material/check:")
        st.download_button(
            "Download DrumParams JSON",
            data=json.dumps(params.to_dict(), indent=2),
            file_name=f"{record.drum}.json",
            mime="application/json",
            icon=":material/save:",
            key=f"dl_{record.started}",
        )
    st.caption(f":gray[stored at {record.directory}]")


def _run_report(record) -> None:
    """The whole report for one stored run — the same view whether the fit
    just finished or happened last week."""
    result = record.summary
    scored = record.score

    _inspection(result)
    st.divider()
    st.subheader("Velocity table")
    st.caption(
        "Stage 2 fits the excitation at each velocity independently — "
        "`gain[]`, noise `level[]` and the contact time — with `f_static` and "
        "`t60` already frozen."
    )
    _velocity_table(result)

    if scored:
        st.divider()
        _report(result, scored)

    st.divider()
    _stage_five(result)

    st.divider()
    st.subheader("Where the time went")
    _timings(result)

    st.divider()
    _comparison(record)
    st.divider()
    _handoff(record)


def _history() -> None:
    """Every stored run, so two fits of the same drum can be compared."""
    runs = store.list()
    if not runs:
        st.caption(":gray[no runs stored yet]")
        return

    st.caption(
        f"{len(runs)} stored. Every fit is kept — a run takes minutes and "
        "produces a drum you cannot judge in one listen, so the useful "
        "comparison is against the previous one."
    )
    st.dataframe(
        pd.DataFrame([record.to_row() for record in runs]),
        width="stretch", hide_index=True,
        column_config={
            "score": st.column_config.NumberColumn(format="%.3f"),
            "stft dB": st.column_config.NumberColumn(format="%.2f"),
            "seconds": st.column_config.NumberColumn(format="%.0f s"),
        },
    )

    labels = {record.label: record for record in runs}
    chosen = labels[st.selectbox("Open a run", list(labels), key="run_pick")]
    st.divider()
    _run_report(chosen)


# -- page ---------------------------------------------------------------------

if run.result is None:
    _setup()

_progress()

# A finished run is read back from its stored directory rather than from the
# event stream, so this is the same code path as opening an old run — and the
# report cannot drift between "just finished" and "opened later".
finished = None
if run.done and run.done.get("run_directory"):
    finished = store.read(run.done["run_directory"])

if finished is not None:
    st.divider()
    _run_report(finished)
    st.divider()
    if st.button("Fit another drum", icon=":material/refresh:"):
        _run_handle.clear()
        st.rerun()
elif run.result is not None:
    st.divider()
    st.info("The fit finished but its run directory was not written — the "
            "report below is from this session's events only.",
            icon=":material/info:")
    _inspection(run.result)
    _velocity_table(run.result)
    if run.scored is not None:
        _report(run.result, run.scored)

st.divider()
with st.expander("Past runs", expanded=finished is None and run.result is None):
    _history()
