"""Rendering. Separate from scoring so the metric never depends on plotting.

Nothing here computes anything. If a number appears in a report it came from a
ScoreCard or a SoundDescriptors, so a report can never disagree with the score
it is rendering.
"""

from __future__ import annotations

import numpy as np

from .comparator import ScoreCard
from .descriptors import SoundDescriptors


class ScoreReport:
    """Text tables and, if matplotlib is installed, diagnostic plots."""

    WIDTH: int = 78

    def __init__(self, card: ScoreCard) -> None:
        self.card = card

    # -- text -----------------------------------------------------------------

    def text(self, include_modes: bool = True) -> str:
        lines = [
            "=" * ScoreReport.WIDTH,
            f"  TOTAL {self.card.total:.3f}".ljust(ScoreReport.WIDTH - 24)
            + f"STFT loss {self.card.stft_loss:8.3f} dB",
            "=" * ScoreReport.WIDTH,
            f"  {'component':<18} {'score':>7}   {'error':>9}  {'unit':<6} weight",
            "  " + "-" * (ScoreReport.WIDTH - 4),
        ]

        from .comparator import Comparator

        for component in self.card.components:
            weight = Comparator.DEFAULT_WEIGHTS.get(component.name, 0.0)
            if component.available:
                lines.append(
                    f"  {component.name:<18} {component.value:7.3f}   "
                    f"{component.raw_error:9.3f}  {component.unit:<6} {weight:.2f}"
                )
            else:
                lines.append(
                    f"  {component.name:<18} {'--':>7}   {'no data':>9}"
                    f"  {'':<6} {weight:.2f}"
                )

        worst = self.card.worst(3)
        if worst:
            lines.append("")
            lines.append("  worst components, start here:")
            for component in worst:
                targets = ", ".join(component.parameters) or "-"
                lines.append(f"    {component.name:<18} -> {targets}")
                for key, value in list(component.detail.items())[:4]:
                    lines.append(f"        {key}: {self._format(value)}")

        if include_modes and self.card.mode_matches:
            lines.append("")
            lines.append(self.mode_table())

        if self.card.warnings:
            lines.append("")
            lines.append("  warnings:")
            lines.extend(f"    ! {warning}" for warning in self.card.warnings)

        lines.append("=" * ScoreReport.WIDTH)
        return "\n".join(lines)

    @staticmethod
    def _format(value) -> str:
        if isinstance(value, float):
            return f"{value:.4g}"
        if isinstance(value, list):
            return ", ".join(str(item) for item in value) if value else "none"
        return str(value)

    def mode_table(self, limit: int = 24) -> str:
        """Per-mode: ref freq | gen freq | cents | ref t60 | gen t60 | ratio | dB."""
        header = (
            f"  {'ref Hz':>9} {'gen Hz':>9} {'cents':>8} "
            f"{'ref t60':>8} {'gen t60':>8} {'ratio':>7} {'dB':>7}"
        )
        lines = ["  mode comparison:", header, "  " + "-" * (len(header) - 2)]

        shown = sorted(self.card.mode_matches, key=lambda match: -match.weight)[:limit]
        for match in sorted(shown, key=lambda match: match.freq):
            if match.is_missing:
                mode = match.reference
                lines.append(
                    f"  {mode.freq:9.2f} {'--':>9} {'MISSING':>8} "
                    f"{mode.t60:8.3f} {'--':>8} {'--':>7} {mode.amplitude_db():7.1f}"
                )
            elif match.is_spurious:
                mode = match.generated
                lines.append(
                    f"  {'--':>9} {mode.freq:9.2f} {'SPURIOUS':>8} "
                    f"{'--':>8} {mode.t60:8.3f} {'--':>7} {mode.amplitude_db():7.1f}"
                )
            else:
                lines.append(
                    f"  {match.reference.freq:9.2f} {match.generated.freq:9.2f} "
                    f"{match.freq_error_cents:8.1f} {match.reference.t60:8.3f} "
                    f"{match.generated.t60:8.3f} {match.t60_error_ratio:7.3f} "
                    f"{match.amplitude_error_db:7.1f}"
                )

        hidden = len(self.card.mode_matches) - len(shown)
        if hidden > 0:
            lines.append(f"  ... {hidden} quieter modes not shown")
        return "\n".join(lines)

    @staticmethod
    def suggestions(edits, limit: int = 12) -> str:
        """Render a list of `ParameterSuggestion`s."""
        if not edits:
            return "  no parameter edits suggested — every component is inside tolerance"
        lines = ["  suggested edits, most important first:"]
        lines.extend(f"    {edit}" for edit in edits[:limit])
        if len(edits) > limit:
            lines.append(f"    ... {len(edits) - limit} more")
        return "\n".join(lines)

    # -- plots ----------------------------------------------------------------

    @staticmethod
    def _pyplot():
        try:
            import matplotlib.pyplot as plt
        except ImportError as error:  # pragma: no cover
            raise ImportError(
                "plotting needs matplotlib: pip install matplotlib. The scores "
                "themselves never require it."
            ) from error
        return plt

    def plot_band_decays(
        self, reference: SoundDescriptors, generated: SoundDescriptors
    ):
        plt = self._pyplot()
        figure, axis = plt.subplots(figsize=(9, 5))
        for descriptors, style, label in (
            (reference, "-", "reference"),
            (generated, "--", "generated"),
        ):
            for band in descriptors.bands:
                if not band.is_valid:
                    continue
                times = np.array(band.valid_range)
                axis.plot(times, band.level_db + band.slope_db_s * times, style,
                          label=f"{label} {band.f_low:.0f}-{band.f_high:.0f}")
        axis.set_xlabel("time (s)")
        axis.set_ylabel("level (dB)")
        axis.set_title("Band decays: reference (solid) vs generated (dashed)")
        axis.grid(alpha=0.3)
        return figure

    def plot_glide(self, reference: SoundDescriptors, generated: SoundDescriptors):
        plt = self._pyplot()
        figure, axis = plt.subplots(figsize=(9, 4))
        for descriptors, style, label in (
            (reference, "-", "reference"),
            (generated, "--", "generated"),
        ):
            if descriptors.glide is None:
                continue
            times, freqs = descriptors.glide.reliable_track()
            axis.plot(times, freqs, style, label=label)
        axis.set_xlabel("time (s)")
        axis.set_ylabel("f0 (Hz)")
        axis.set_title("Fundamental trajectory")
        axis.legend()
        axis.grid(alpha=0.3)
        return figure

    def plot_spectrogram_diff(
        self, reference: np.ndarray, generated: np.ndarray, sr: int = 44100,
        n_fft: int = 2048,
    ):
        """Log-magnitude difference. Diagnostic only — never the score itself."""
        plt = self._pyplot()
        from .stft import MultiResolutionSTFTLoss

        loss = MultiResolutionSTFTLoss(sr, fft_sizes=(n_fft,))
        reference, generated = loss._align(reference, generated)
        reference_db = loss._spectrogram_db(reference, n_fft)
        generated_db = loss._spectrogram_db(generated, n_fft)
        rows = min(len(reference_db), len(generated_db))
        difference = (generated_db[:rows] - reference_db[:rows]).T

        figure, axis = plt.subplots(figsize=(10, 5))
        image = axis.imshow(
            difference, origin="lower", aspect="auto", cmap="RdBu_r",
            vmin=-24, vmax=24,
            extent=[0, rows * n_fft / 4 / sr, 0, sr / 2],
        )
        axis.set_ylim(0, 6000)
        axis.set_xlabel("time (s)")
        axis.set_ylabel("frequency (Hz)")
        axis.set_title("generated - reference (dB). Red = too loud, blue = too quiet")
        figure.colorbar(image, ax=axis, label="dB")
        return figure
