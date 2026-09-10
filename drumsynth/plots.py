"""Figures: what a model kept, what it sounds like, and where it is wrong.

Every function here returns a matplotlib figure and takes plain arrays or a
model — nothing takes a fit result, so the same figure serves a saved run, a
live page and a notebook.

The set is deliberately redundant. A relative error of 1e-4 says one number
about a whole hit; it does not say whether what is left is a missing partial,
a smeared transient or a tail that decays too fast, and those want different
fixes. Waveform, spectrogram, spectrum and error-by-band each fail in a
different direction, so between them the residual has somewhere to show.

matplotlib is optional for the rest of the package, so importing this module
is what makes it required.
"""

from __future__ import annotations

import numpy as np

from .spectral.stft import StftSpec, analyze

#: Anything quieter than this below the peak is floor, not signal.
FLOOR_DB = -90.0


def _pyplot():
    try:
        import matplotlib
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "drumsynth.plots needs matplotlib: pip install 'drumsynth[plots]'"
        ) from error

    if matplotlib.get_backend().lower() not in {"agg", "module://matplotlib_inline.backend_inline"}:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def db(values: np.ndarray, reference: float | None = None) -> np.ndarray:
    """Amplitude to dB, floored, relative to `reference` or to the peak."""
    magnitude = np.abs(np.asarray(values, dtype=np.float64))
    peak = float(reference if reference is not None else magnitude.max() or 1.0)
    with np.errstate(divide="ignore"):
        out = 20.0 * np.log10(np.maximum(magnitude, 1e-30) / max(peak, 1e-30))
    return np.maximum(out, FLOOR_DB)


def _times(n_frames: int, spec: StftSpec, sample_rate: int) -> np.ndarray:
    return np.arange(n_frames) * spec.hop / float(sample_rate)


def _finish(figure, title: str | None = None):
    if title:
        figure.suptitle(title, fontsize=11)
    if figure.get_layout_engine() is None:
        figure.tight_layout()
    return figure


# ============================================================================
# What the model kept
# ============================================================================


def bin_map(model, reference: np.ndarray | None = None, components: np.ndarray | None = None):
    """Which bins the model keeps, against the spectrum it chose them from."""
    plt = _pyplot()
    spec = model.candidate.spec
    frequencies = spec.bin_frequencies(model.sample_rate)
    kept = model.bins

    figure, axes = plt.subplots(figsize=(11, 4))

    if reference is not None:
        energy = np.sum(np.abs(analyze(reference, spec)) ** 2, axis=1)
        axes.semilogx(
            frequencies[1:],
            db(np.sqrt(energy[1:])),
            color="0.75",
            linewidth=0.8,
            label="the whole spectrum",
        )

    heights = (
        db(np.abs(components).max(axis=1)) if components is not None else np.zeros(kept.size)
    )
    axes.vlines(
        np.maximum(frequencies[kept], 1.0),
        FLOOR_DB,
        heights,
        color="crimson",
        linewidth=1.0,
        alpha=0.8,
        label=f"{kept.size} kept bins",
    )

    axes.set_xlim(max(frequencies[1], 10.0), frequencies[-1])
    axes.set_ylim(FLOOR_DB, 5)
    axes.set_xlabel("frequency (Hz)")
    axes.set_ylabel("dB below peak")
    axes.grid(alpha=0.25, which="both")
    axes.legend(fontsize=8, loc="upper right")
    return _finish(figure, "Kept bins")


def envelope_heatmap(components: np.ndarray, frequencies: np.ndarray, times: np.ndarray):
    """Every partial's amplitude over time, as a picture of the hit."""
    plt = _pyplot()
    figure, axes = plt.subplots(figsize=(11, 4.5), layout="constrained")

    levels = db(np.abs(components))
    order = np.argsort(frequencies)
    image = axes.pcolormesh(
        times,
        frequencies[order],
        levels[order],
        shading="nearest",
        cmap="magma",
        vmin=-80,
        vmax=0,
    )
    axes.set_yscale("symlog", linthresh=100)
    axes.set_xlabel("time (s)")
    axes.set_ylabel("frequency (Hz)")
    figure.colorbar(image, ax=axes, label="dB below peak")
    return _finish(figure, "Envelopes")


def envelope_traces(
    components: np.ndarray, frequencies: np.ndarray, times: np.ndarray, count: int = 8
):
    """The loudest partials, one line each — decay shape, ripple and all."""
    plt = _pyplot()
    figure, axes = plt.subplots(figsize=(11, 4.5))

    magnitude = np.abs(components)
    loudest = np.argsort(magnitude.max(axis=1))[::-1][:count]
    peak = magnitude.max()

    for index in loudest:
        axes.plot(
            times,
            db(magnitude[index], peak),
            linewidth=1.1,
            label=f"{frequencies[index]:.0f} Hz",
        )

    axes.set_ylim(FLOOR_DB, 3)
    axes.set_xlabel("time (s)")
    axes.set_ylabel("dB below peak")
    axes.grid(alpha=0.25)
    axes.legend(fontsize=8, ncol=2, loc="upper right")
    return _finish(figure, f"The {len(loudest)} loudest partials")


def decay_times(components: np.ndarray, frequencies: np.ndarray, times: np.ndarray):
    """How long each partial rings, against its frequency.

    A straight line fitted to each envelope in dB, from its peak down, and
    extended to -60 dB. Drum partials are not exactly exponential and the fit
    says so — the scatter around the trend is the part a single decay constant
    would throw away.
    """
    plt = _pyplot()
    magnitude = np.abs(components)
    levels = db(magnitude, magnitude.max())

    t60 = np.full(magnitude.shape[0], np.nan)
    for index in range(magnitude.shape[0]):
        start = int(np.argmax(levels[index]))
        usable = np.flatnonzero(levels[index][start:] > FLOOR_DB + 10) + start
        if usable.size < 4:
            continue
        slope, _ = np.polyfit(times[usable], levels[index][usable], 1)
        if slope < -1e-6:
            t60[index] = -60.0 / slope

    figure, axes = plt.subplots(figsize=(11, 4))
    sizes = 8 + 90 * (magnitude.max(axis=1) / max(magnitude.max(), 1e-30))
    # DC is kept by the model when it carries energy, and has no decay to speak
    # of; it also cannot go on a log axis.
    tonal = np.asarray(frequencies) > 0
    axes.scatter(
        np.asarray(frequencies)[tonal],
        t60[tonal],
        s=sizes[tonal],
        alpha=0.6,
        color="teal",
        edgecolors="none",
    )
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlabel("frequency (Hz)")
    axes.set_ylabel("time to -60 dB (s)")
    axes.grid(alpha=0.25, which="both")
    axes.set_title("Decay per partial (marker size = how loud it is)", fontsize=10)
    return _finish(figure)


def envelope_spectrum(components: np.ndarray, frame_rate: float):
    """How fast the envelopes themselves move.

    The spectrum of each envelope, averaged: 0 Hz is a partial that holds
    steady, and everything above it is decay and tremolo. It is the reason
    envelopes compress at all — nearly all of their energy is near DC — and
    the reason a truncated basis rings when it is cut too hard.
    """
    plt = _pyplot()
    magnitude = np.abs(components)
    centred = magnitude - magnitude.mean(axis=1, keepdims=True)
    spectrum = np.abs(np.fft.rfft(centred, axis=1)).mean(axis=0)
    rates = np.fft.rfftfreq(magnitude.shape[1], 1.0 / frame_rate)

    figure, axes = plt.subplots(figsize=(11, 3.6))
    axes.semilogx(rates[1:], db(spectrum[1:]), color="darkorange", linewidth=1.2)
    axes.set_xlabel("envelope frequency (Hz)")
    axes.set_ylabel("dB below peak")
    axes.set_ylim(FLOOR_DB / 2, 3)
    axes.grid(alpha=0.25, which="both")
    return _finish(figure, "Spectrum of the envelopes")


# ============================================================================
# Reference against model
# ============================================================================


def waveform(reference: np.ndarray, estimate: np.ndarray, sample_rate: int, attack: float = 0.03):
    """The two signals over each other, their difference, and the attack up close."""
    plt = _pyplot()
    times = np.arange(reference.size) / float(sample_rate)

    figure, (top, middle, bottom) = plt.subplots(
        3, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [2, 1, 2]}
    )

    top.plot(times, reference, linewidth=0.6, color="0.4", label="recording")
    top.plot(times, estimate, linewidth=0.6, color="crimson", alpha=0.8, label="model")
    top.set_ylabel("amplitude")
    top.legend(fontsize=8, loc="upper right")
    top.grid(alpha=0.2)

    middle.plot(times, reference - estimate, linewidth=0.6, color="teal")
    middle.set_ylabel("residual")
    middle.set_ylim(top.get_ylim())
    middle.grid(alpha=0.2)

    window = slice(0, min(int(attack * sample_rate), reference.size))
    bottom.plot(times[window], reference[window], linewidth=1.0, color="0.4")
    bottom.plot(times[window], estimate[window], linewidth=1.0, color="crimson", alpha=0.8)
    bottom.set_xlabel("time (s)")
    bottom.set_ylabel(f"first {1000 * attack:.0f} ms")
    bottom.grid(alpha=0.2)
    return _finish(figure, "Waveform")


def spectrogram_panel(
    reference: np.ndarray, estimate: np.ndarray, sample_rate: int, spec: StftSpec | None = None
):
    """Recording, model and difference, on one colour scale."""
    plt = _pyplot()
    spec = spec or StftSpec(1024, 256)

    left = np.abs(analyze(reference, spec))
    right = np.abs(analyze(estimate, spec))
    difference = np.abs(analyze(reference - estimate, spec))
    peak = max(left.max(), 1e-30)

    frequencies = spec.bin_frequencies(sample_rate)
    times = _times(left.shape[1], spec, sample_rate)

    figure, axes = plt.subplots(
        1, 3, figsize=(13, 4), sharex=True, sharey=True, layout="constrained"
    )
    for panel, data, title in zip(
        axes, (left, right, difference), ("recording", "model", "difference")
    ):
        image = panel.pcolormesh(
            times,
            frequencies,
            db(data, peak),
            shading="nearest",
            cmap="magma",
            vmin=-90,
            vmax=0,
        )
        panel.set_yscale("symlog", linthresh=100)
        panel.set_title(title, fontsize=10)
        panel.set_xlabel("time (s)")
    axes[0].set_ylabel("frequency (Hz)")
    figure.colorbar(image, ax=axes, label="dB below the recording's peak")
    return _finish(figure, "Spectrogram")


def spectrum(reference: np.ndarray, estimate: np.ndarray, sample_rate: int):
    """Average spectrum of both, and of what is left over."""
    plt = _pyplot()
    spec = StftSpec(4096, 2048)

    def average(signal):
        return np.sqrt(np.mean(np.abs(analyze(signal, spec)) ** 2, axis=1))

    frequencies = spec.bin_frequencies(sample_rate)
    left, right = average(reference), average(estimate)
    residual = average(reference - estimate)
    peak = max(left.max(), 1e-30)

    figure, axes = plt.subplots(figsize=(11, 4.5))
    axes.semilogx(frequencies[1:], db(left[1:], peak), color="0.4", linewidth=1.0, label="recording")
    axes.semilogx(
        frequencies[1:], db(right[1:], peak), color="crimson", linewidth=1.0, alpha=0.85, label="model"
    )
    axes.semilogx(
        frequencies[1:], db(residual[1:], peak), color="teal", linewidth=0.9, alpha=0.8, label="residual"
    )
    axes.set_xlim(max(frequencies[1], 20.0), frequencies[-1])
    axes.set_ylim(FLOOR_DB, 5)
    axes.set_xlabel("frequency (Hz)")
    axes.set_ylabel("dB below the recording's peak")
    axes.grid(alpha=0.25, which="both")
    axes.legend(fontsize=8)
    return _finish(figure, "Average spectrum")


def error_over_time(reference: np.ndarray, estimate: np.ndarray, sample_rate: int, window: float = 0.02):
    """Relative error in windows through the hit — where in the sound it is wrong."""
    plt = _pyplot()
    size = max(1, int(window * sample_rate))
    n = (reference.size // size) * size

    left = reference[:n].reshape(-1, size)
    difference = (reference[:n] - estimate[:n]).reshape(-1, size)
    signal = np.sum(left**2, axis=1)
    error = np.sum(difference**2, axis=1)

    ratio = np.divide(error, signal, out=np.full(signal.shape, np.nan), where=signal > 0)
    times = np.arange(ratio.size) * window

    figure, axes = plt.subplots(figsize=(11, 3.6))
    axes.semilogy(times, np.maximum(ratio, 1e-12), color="teal", linewidth=1.1)
    axes.plot(times, np.maximum(signal / max(signal.max(), 1e-30), 1e-12), color="0.75",
              linewidth=0.9, label="where the energy is")
    axes.set_xlabel("time (s)")
    axes.set_ylabel("relative error")
    axes.grid(alpha=0.25, which="both")
    axes.legend(fontsize=8)
    return _finish(figure, "Error through the hit")


def error_by_frequency(reference: np.ndarray, estimate: np.ndarray, sample_rate: int, bands: int = 24):
    """Relative error per third-octave band — which part of the sound is wrong."""
    plt = _pyplot()
    spec = StftSpec(4096, 2048)

    signal = np.sum(np.abs(analyze(reference, spec)) ** 2, axis=1)
    residual = np.sum(np.abs(analyze(reference - estimate, spec)) ** 2, axis=1)
    frequencies = spec.bin_frequencies(sample_rate)

    edges = np.geomspace(max(frequencies[1], 20.0), frequencies[-1], bands + 1)
    centres = np.sqrt(edges[:-1] * edges[1:])
    ratios = np.full(bands, np.nan)
    shares = np.zeros(bands)
    for band, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        inside = (frequencies >= low) & (frequencies < high)
        if not inside.any() or signal[inside].sum() <= 0:
            continue
        ratios[band] = residual[inside].sum() / signal[inside].sum()
        shares[band] = signal[inside].sum() / signal.sum()

    figure, axes = plt.subplots(figsize=(11, 4))
    axes.stairs(np.nan_to_num(ratios, nan=1e-12), edges, fill=True, color="teal", alpha=0.75)
    axes.set_xscale("log")
    # A filled stairs autoscales towards zero, which a log axis cannot hold.
    axes.set_xlim(edges[0], edges[-1])
    axes.set_yscale("log")
    axes.set_xlabel("frequency (Hz)")
    axes.set_ylabel("relative error in band")

    twin = axes.twinx()
    twin.plot(centres, shares, color="0.5", linewidth=1.0, marker=".", label="share of energy")
    twin.set_ylabel("share of the recording's energy")
    twin.legend(fontsize=8, loc="upper right")
    axes.grid(alpha=0.25, which="both")
    return _finish(figure, "Error by band")


# ============================================================================
# A whole instrument
# ============================================================================


def velocity_curves(model):
    """The field's weights against velocity — how the drum's shape is dialled in."""
    plt = _pyplot()
    figure, axes = plt.subplots(figsize=(11, 4))

    weights = model.field.weights
    for column in range(min(weights.shape[1], 8)):
        axes.plot(
            model.velocities,
            weights[:, column],
            marker=".",
            linewidth=1.1,
            label=f"pattern {column}",
        )

    axes.set_xlabel("velocity")
    axes.set_ylabel("weight")
    axes.grid(alpha=0.25)
    axes.legend(fontsize=8, ncol=2)
    return _finish(figure, "Velocity curves")


def velocity_response(model, count: int = 24):
    """What the drum actually does as it is hit harder: level, brightness, ring."""
    plt = _pyplot()
    low, high = model.velocity_range
    velocities = np.linspace(low, high, count) if high > low else np.array([low])

    frequencies = model.frequencies()
    level, centroid, ring = [], [], []
    for velocity in velocities:
        magnitude = np.abs(model.field.at(velocity))
        energy = magnitude**2
        total = energy.sum()
        level.append(np.sqrt(total))
        centroid.append(float((energy.sum(axis=1) @ frequencies) / max(total, 1e-30)))

        summed = magnitude.sum(axis=0)
        peak = summed.max()
        above = np.flatnonzero(summed > peak * 1e-3)
        ring.append(
            (above[-1] - above[0]) * model.candidate.hop / model.sample_rate if above.size else 0.0
        )

    figure, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for panel, values, label in zip(
        axes,
        (db(np.asarray(level)), centroid, ring),
        ("level (dB below loudest)", "spectral centroid (Hz)", "time above -60 dB (s)"),
    ):
        panel.plot(velocities, values, marker=".", linewidth=1.2, color="crimson")
        for velocity in model.velocities:
            panel.axvline(velocity, color="0.85", linewidth=0.6, zorder=0)
        panel.set_xlabel("velocity")
        panel.set_title(label, fontsize=9)
        panel.grid(alpha=0.25)
    return _finish(figure, "Response to velocity (grey lines: recorded layers)")


def velocity_spectrum(model, count: int = 40):
    """Spectrum against velocity: the whole instrument in one picture."""
    plt = _pyplot()
    low, high = model.velocity_range
    velocities = np.linspace(low, high, count) if high > low else np.array([low])

    frequencies = model.frequencies()
    order = np.argsort(frequencies)
    energy = np.stack(
        [np.sqrt(np.sum(np.abs(model.field.at(v)) ** 2, axis=1))[order] for v in velocities]
    )

    figure, axes = plt.subplots(figsize=(11, 4.5), layout="constrained")
    image = axes.pcolormesh(
        velocities, frequencies[order], db(energy).T, shading="nearest", cmap="magma", vmin=-80, vmax=0
    )
    axes.set_yscale("symlog", linthresh=100)
    axes.set_xlabel("velocity")
    axes.set_ylabel("frequency (Hz)")
    figure.colorbar(image, ax=axes, label="dB below the loudest partial")
    for velocity in model.velocities:
        axes.axvline(velocity, color="white", alpha=0.15, linewidth=0.6)
    return _finish(figure, "Spectrum against velocity")


# ============================================================================
# Whole sets, written to a directory
# ============================================================================

#: Written at this resolution: readable on a screen, small enough to keep.
DPI = 120


def to_png(figure, dpi: int | None = None) -> bytes:
    """Render a figure to PNG bytes and close it.

    Closing matters: a page that draws a dozen figures per rerun leaks every
    one of them into pyplot's registry otherwise.
    """
    import io

    plt = _pyplot()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi or DPI)
    plt.close(figure)
    return buffer.getvalue()


def _save(figures: dict, out_dir) -> dict:
    from pathlib import Path

    plt = _pyplot()
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)

    written = {}
    for name, figure in figures.items():
        path = directory / f"{name}.png"
        figure.savefig(path, dpi=DPI)
        plt.close(figure)
        written[name] = path
    return written


def save_hit_figures(model, reference: np.ndarray, out_dir) -> dict:
    """Every figure a single-hit model supports, into `out_dir`."""
    estimate = model.render()
    components = model.components()
    frequencies = model.frequencies()
    times = _times(model.n_frames, model.candidate.spec, model.sample_rate)
    frame_rate = model.sample_rate / model.candidate.hop

    return _save(
        {
            "waveform": waveform(reference, estimate, model.sample_rate),
            "spectrogram": spectrogram_panel(reference, estimate, model.sample_rate),
            "spectrum": spectrum(reference, estimate, model.sample_rate),
            "error_over_time": error_over_time(reference, estimate, model.sample_rate),
            "error_by_frequency": error_by_frequency(reference, estimate, model.sample_rate),
            "bins": bin_map(model, reference, components),
            "envelopes": envelope_heatmap(components, frequencies, times),
            "envelope_traces": envelope_traces(components, frequencies, times),
            "decay_times": decay_times(components, frequencies, times),
            "envelope_spectrum": envelope_spectrum(components, frame_rate),
        },
        out_dir,
    )


def save_instrument_figures(model, layers=None, take: int = 0, out_dir=".") -> dict:
    """Every figure a velocity model supports, into `out_dir`.

    The waveform-level comparisons are drawn at the loudest recorded velocity,
    where the drum has the most to get wrong.
    """
    velocity = float(model.velocities[-1])
    components = model.components(velocity)
    frequencies = model.frequencies()
    times = _times(model.n_frames, model.candidate.spec, model.sample_rate)

    figures = {
        "velocity_curves": velocity_curves(model),
        "velocity_response": velocity_response(model),
        "velocity_spectrum": velocity_spectrum(model),
        "envelopes": envelope_heatmap(components, frequencies, times),
        "envelope_traces": envelope_traces(components, frequencies, times),
        "decay_times": decay_times(components, frequencies, times),
        "envelope_spectrum": envelope_spectrum(
            components, model.sample_rate / model.candidate.hop
        ),
    }

    if layers is not None:
        reference = layers.audio(layers.n_layers - 1, take)
        estimate = model.render(velocity)
        figures.update(
            {
                "bins": bin_map(model, reference, components),
                "waveform": waveform(reference, estimate, model.sample_rate),
                "spectrogram": spectrogram_panel(reference, estimate, model.sample_rate),
                "spectrum": spectrum(reference, estimate, model.sample_rate),
                "error_over_time": error_over_time(reference, estimate, model.sample_rate),
                "error_by_frequency": error_by_frequency(
                    reference, estimate, model.sample_rate
                ),
            }
        )

    return _save(figures, out_dir)
