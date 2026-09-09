"""The mixer widgets.

Three strip kinds, matching the three parameter groups the synthesizer actually
has: one per mode, one per noise band, and one master. Every strip writes
straight into the `DrumParams` held in session state; nothing is buffered, so
what you see is what the engine is playing.

Gains and levels are edited in dB. A linear 0-1 fader spends almost all of its
travel in the top 6 dB, which makes the quiet partials — the ones that decide
whether a drum sounds alive — impossible to set.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from ..core.constants import Cents
from .session import Studio


class ModeStrip:
    """One resonant partial: frequency, gain, decay, and solo/mute."""

    #: Frequency is edited as a ratio to the fundamental as well as in Hz.
    #: Membrane modes are inharmonic and their ratios are the physically
    #: meaningful number — 1.594 and 2.136 mean something, 147.4 Hz does not.
    F_RANGE = (10.0, 18000.0)
    T60_RANGE = (0.005, 12.0)

    @staticmethod
    def render(studio: Studio, index: int, fundamental: float) -> None:
        mode = studio.params.modes[index]
        audible = studio.is_audible(index)
        soloed = index in st.session_state.solo
        muted = index in st.session_state.muted

        with st.container(border=True, gap=None):
            ratio = mode.f_static / fundamental if fundamental > 0 else 1.0
            st.markdown(
                f"**{index}** · {'`ring`' if audible else '`off`'}  \n"
                f":gray[×{ratio:.3f}]"
            )

            # Every value below is CLAMPED into its widget's range rather
            # than trusted. These numbers arrive from a fit or from a
            # hand-edited JSON, and `st.number_input` raises on an
            # out-of-range value — which takes the whole page down before it
            # draws anything. A fitted mode 73 dB below the loudest is a real
            # thing the solver produces; crashing on it is not.
            st.number_input(
                "Hz",
                min_value=ModeStrip.F_RANGE[0],
                max_value=ModeStrip.F_RANGE[1],
                value=float(np.clip(mode.f_static, *ModeStrip.F_RANGE)),
                step=0.5,
                format="%.2f",
                key=f"mode_{index}_f",
                on_change=ModeStrip._commit,
                args=(studio, index),
            )
            st.number_input(
                "dB",
                min_value=Studio.GAIN_RANGE_DB[0],
                max_value=Studio.GAIN_RANGE_DB[1],
                value=float(np.clip(Studio.to_db(mode.gain),
                                    *Studio.GAIN_RANGE_DB)),
                step=0.5,
                format="%.1f",
                key=f"mode_{index}_gain",
                on_change=ModeStrip._commit,
                args=(studio, index),
            )
            st.number_input(
                "t60 s",
                min_value=ModeStrip.T60_RANGE[0],
                max_value=ModeStrip.T60_RANGE[1],
                value=float(np.clip(mode.t60, *ModeStrip.T60_RANGE)),
                step=0.01,
                format="%.3f",
                key=f"mode_{index}_t60",
                on_change=ModeStrip._commit,
                args=(studio, index),
            )

            with st.container(horizontal=True, gap=None, wrap=False):
                st.button(
                    "S",
                    key=f"mode_{index}_solo",
                    type="primary" if soloed else "secondary",
                    help="Solo — hear this partial alone, live",
                    on_click=studio.toggle_solo,
                    args=(index,),
                    width="stretch",
                )
                st.button(
                    "M",
                    key=f"mode_{index}_mute",
                    type="primary" if muted else "secondary",
                    help="Mute this partial, live",
                    on_click=studio.toggle_mute,
                    args=(index,),
                    width="stretch",
                )
                st.button(
                    "",
                    key=f"mode_{index}_remove",
                    icon=":material/delete:",
                    help="Remove this mode",
                    disabled=len(studio.params.modes) <= 1,
                    on_click=studio.remove_mode,
                    args=(index,),
                    width="stretch",
                )

    @staticmethod
    def _commit(studio: Studio, index: int) -> None:
        state = st.session_state
        studio.replace_mode(
            index,
            f_static=state[f"mode_{index}_f"],
            gain=Studio.from_db(state[f"mode_{index}_gain"]),
            t60=state[f"mode_{index}_t60"],
        )


class BandStrip:
    """One contact-noise band: edges, level, decay.

    These are the transient. No attack — level starts at maximum and decays
    immediately; the envelope peak a real drum shows comes from the modal bank
    summing, not from here.
    """

    F_RANGE = (20.0, 20000.0)

    #: A transient band used to be a click and this capped at one second. The
    #: fit now MEASURES each band's decay on the residual, and a real tom's
    #: 200-800 Hz residual came back at 1.7 s — loading that fit crashed the
    #: strip. The ceiling covers what `NoiseDecayStage` can produce, with room
    #: to hand-edit past it.
    T60_RANGE = (0.001, 4.0)

    @staticmethod
    def render(studio: Studio, index: int) -> None:
        band = studio.params.noise[index]
        with st.container(border=True, gap=None):
            centre, quality = band.center_and_q()
            st.markdown(f"**band {index}**  \n:gray[{centre:.0f} Hz · Q {quality:.2f}]")

            st.number_input(
                "low Hz", min_value=BandStrip.F_RANGE[0], max_value=BandStrip.F_RANGE[1],
                value=float(np.clip(band.f_low, *BandStrip.F_RANGE)),
                step=10.0, format="%.0f",
                key=f"band_{index}_low",
                on_change=BandStrip._commit, args=(studio, index),
            )
            st.number_input(
                "high Hz", min_value=BandStrip.F_RANGE[0], max_value=BandStrip.F_RANGE[1],
                value=float(np.clip(band.f_high, *BandStrip.F_RANGE)),
                step=10.0, format="%.0f",
                key=f"band_{index}_high",
                on_change=BandStrip._commit, args=(studio, index),
            )
            st.number_input(
                "dB", min_value=Studio.LEVEL_RANGE_DB[0], max_value=Studio.LEVEL_RANGE_DB[1],
                value=float(np.clip(Studio.to_db(band.level), *Studio.LEVEL_RANGE_DB)),
                step=0.5, format="%.1f",
                key=f"band_{index}_level",
                on_change=BandStrip._commit, args=(studio, index),
            )
            st.number_input(
                "t60 s", min_value=BandStrip.T60_RANGE[0], max_value=BandStrip.T60_RANGE[1],
                # Clamped, not trusted: these numbers can arrive from a fit or
                # from a hand-edited JSON, and a widget that raises on an
                # out-of-range value takes the whole page down with it.
                value=float(np.clip(band.t60, *BandStrip.T60_RANGE)),
                step=0.002, format="%.3f",
                key=f"band_{index}_t60",
                on_change=BandStrip._commit, args=(studio, index),
            )
            st.button(
                "Remove", key=f"band_{index}_remove", width="stretch",
                icon=":material/delete:",
                on_click=studio.remove_band, args=(index,),
            )

    @staticmethod
    def _commit(studio: Studio, index: int) -> None:
        state = st.session_state
        low = state[f"band_{index}_low"]
        high = state[f"band_{index}_high"]
        if low >= high:  # keep the band orderable without fighting the user
            high = low * 1.05
            state[f"band_{index}_high"] = high
        studio.replace_band(
            index,
            f_low=low,
            f_high=high,
            level=Studio.from_db(state[f"band_{index}_level"]),
            t60=state[f"band_{index}_t60"],
        )


class MasterStrip:
    """Everything that applies to the whole drum.

    Tension is here rather than with the modes because it is one global scalar
    on every mode frequency at once — that is the mechanism, not a convenience.
    """

    @staticmethod
    def render(studio: Studio) -> None:
        params = studio.params

        with st.container(border=True):
            st.markdown("**Output**")
            st.number_input(
                "Output gain (dB)",
                min_value=-40.0, max_value=24.0,
                value=Studio.to_db(params.output_gain),
                step=0.5, format="%.1f",
                key="master_gain",
                on_change=MasterStrip._commit_gain, args=(studio,),
            )
            st.caption(
                f"A unit strike peaks near "
                f"{params.output_gain * float(np.sum(params.mode_gains())):.2f}"
            )

        with st.container(border=True):
            st.markdown("**Tension** — the pitch glide")
            glide = 12.0 * np.log2(1.0 + params.tension.k) if params.tension.k > 0 else 0.0
            st.slider(
                "Glide depth at full strike (semitones)",
                min_value=0.0, max_value=6.0, value=float(glide), step=0.01,
                key="master_glide",
                help=(
                    "Energy raises head tension, which raises every mode "
                    "frequency by one ratio. A hard hit glides deep and a soft "
                    "hit barely moves, from the same number."
                ),
                on_change=MasterStrip._commit_tension, args=(studio,),
            )
            st.slider(
                "Settle (tau, s)",
                min_value=0.01, max_value=1.0, value=float(params.tension.tau), step=0.005,
                key="master_tau",
                help="Smoothing of the energy estimate. Sets how long the glide takes.",
                on_change=MasterStrip._commit_tension, args=(studio,),
            )
            st.caption(f"k = {params.tension.k:.6g}")

        with st.container(border=True):
            st.markdown("**Global transforms**")
            st.slider(
                "Tune (semitones)", min_value=-24.0, max_value=24.0, value=0.0, step=0.1,
                key="master_tune",
                help="Scales every mode frequency. Applied on release, then reset.",
                on_change=MasterStrip._apply_tune, args=(studio,),
            )
            st.slider(
                "Damping (t60 ×)", min_value=0.1, max_value=3.0, value=1.0, step=0.01,
                key="master_damp",
                help="Scales every mode t60. This is what hand muting is.",
                on_change=MasterStrip._apply_damping, args=(studio,),
            )

    @staticmethod
    def _commit_gain(studio: Studio) -> None:
        studio.params.output_gain = Studio.from_db(st.session_state.master_gain)

    @staticmethod
    def _commit_tension(studio: Studio) -> None:
        semitones = st.session_state.master_glide
        studio.set_tension(
            k=2.0 ** (semitones / 12.0) - 1.0, tau=st.session_state.master_tau
        )

    @staticmethod
    def _apply_tune(studio: Studio) -> None:
        semitones = st.session_state.master_tune
        if semitones == 0.0:
            return
        ratio = 2.0 ** (semitones / 12.0)
        studio.params.modes = [m.scaled(freq_ratio=ratio) for m in studio.params.modes]
        st.session_state.master_tune = 0.0
        studio.clear_widget_state()

    @staticmethod
    def _apply_damping(studio: Studio) -> None:
        factor = st.session_state.master_damp
        if factor == 1.0:
            return
        studio.params.modes = [m.scaled(t60_ratio=factor) for m in studio.params.modes]
        st.session_state.master_damp = 1.0
        studio.clear_widget_state()


class Meter:
    """Live level and glide readout, driven by engine telemetry.

    Compact on purpose: it lives in a narrow sidebar above the strike controls,
    and `st.metric` at full size pushes them off the screen.
    """

    #: Occasional late blocks are normal on a busy machine and mean nothing.
    #: Warn only once they are frequent enough to be audible.
    UNDERRUN_WARNING: int = 12

    @staticmethod
    def render(studio: Studio) -> None:
        synth = studio.engine()
        if synth is None:
            st.caption(":gray[engine stopped]")
            return

        status = synth.status
        peak_db = Studio.to_db(status.peak)
        semitones = Cents.from_ratio(max(status.frequency_ratio, 1e-9)) / 100.0

        st.progress(
            float(np.clip(status.peak, 0.0, 1.0)),
            text=(
                f"**{peak_db:.1f} dB**"
                + (" · :red[clip]" if status.clipped else "")
            ),
        )
        st.caption(
            f"glide {semitones:+.2f} st · load {status.load * 100:.0f}% · "
            f"{status.strikes} hits"
        )

        if status.underruns >= Meter.UNDERRUN_WARNING:
            st.caption(
                f":orange[{status.underruns} late blocks — raise the block size]"
            )
