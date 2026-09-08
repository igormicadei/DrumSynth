"""The analysis chain, run end to end.

Use ONE `Analyzer` instance for both signals. Not because it holds state — it
does not — but because every sub-analyzer's window length, band edges and hop
size is a choice that biases the result, and the only way those biases cancel
is if both sides get exactly the same ones.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.constants import Audio
from .bands import BandDecayAnalyzer
from .descriptors import SoundDescriptors
from .envelope import EnvelopeAnalyzer
from .glide import GlideAnalyzer
from .modal import ModalAnalyzer
from .noise import NoiseAnalyzer
from .prep import SignalPrep


class Analyzer:
    """Runs every sub-analyzer and assembles a `SoundDescriptors`."""

    def __init__(
        self,
        sr: int = Audio.DEFAULT_SR,
        prep: SignalPrep | None = None,
        modal: ModalAnalyzer | None = None,
        bands: BandDecayAnalyzer | None = None,
        glide: GlideAnalyzer | None = None,
        envelope: EnvelopeAnalyzer | None = None,
        noise: NoiseAnalyzer | None = None,
        f_range: tuple[float, float] | None = None,
    ) -> None:
        self.sr = int(sr)
        self.prep = prep or SignalPrep(self.sr)
        self.modal = modal or ModalAnalyzer(self.sr)
        self.bands = bands or BandDecayAnalyzer(self.sr)
        self.envelope = envelope or EnvelopeAnalyzer(self.sr)
        self.noise = noise or NoiseAnalyzer(self.sr)
        self.glide = glide or GlideAnalyzer(self.sr, f_range or (60.0, 140.0))

    # -- construction helpers -------------------------------------------------

    @classmethod
    def for_fundamental(cls, f0: float, sr: int = Audio.DEFAULT_SR) -> "Analyzer":
        """An analyzer whose glide range brackets a known fundamental.

        The glide tracker searches a narrow band. Pointed at the wrong octave it
        locks onto a partial and reports a confident, wrong trajectory, so the
        range is worth setting explicitly whenever f0 is known.
        """
        return cls(sr, f_range=(f0 * 0.6, f0 * 1.5))

    # -- analysis -------------------------------------------------------------

    def analyze(
        self,
        x: np.ndarray,
        prepared: bool = True,
        noise_floor_db: float | None = None,
    ) -> SoundDescriptors:
        """Full descriptor set for one signal.

        `prepared=False` runs `SignalPrep.prepare` first. Pass already-prepared
        audio for both sides, or unprepared audio for both sides — mixing them
        compares one signal's alignment against the other's.

        `noise_floor_db` overrides the floor measured from this signal, and
        `DrumScorer` always passes the REFERENCE's floor for both sides. That
        is the guard rail made real: a synthesized hit decays into digital
        silence and can be measured 100 dB down, while the reference flattens
        at its codec floor 60 dB down. Letting each side use its own floor
        measures each decay over a different range and reports a disagreement
        that is entirely an artifact of the recording.
        """
        x = np.asarray(x, dtype=np.float64)
        if not prepared:
            x = self.prep.prepare(x)

        warnings: list[str] = []
        if x.size == 0:
            return SoundDescriptors(sr=self.sr, duration=0.0,
                                    warnings=["empty signal"])

        measured_floor = self.prep.estimate_noise_floor(x)
        if noise_floor_db is None:
            noise_floor_db = measured_floor
        elif measured_floor < noise_floor_db - 3.0:
            warnings.append(
                f"analysis floored at the reference's {noise_floor_db:.1f} dB "
                f"although this signal is clean to {measured_floor:.1f} dB; the "
                "reference cannot show what happens below its own floor"
            )
        usable = self.prep.usable_duration(x, noise_floor_db)
        duration = len(x) / self.sr

        if usable < 0.25 * duration and duration > 0.5:
            warnings.append(
                f"only {usable:.2f}s of {duration:.2f}s sits above the noise floor "
                f"({noise_floor_db:.1f} dB); decay fits use the usable region only"
            )

        window = x[: max(1, int(round(usable * self.sr)))] if usable > 0 else x
        modes = self.modal.analyze(window)

        descriptors = SoundDescriptors(
            sr=self.sr,
            duration=duration,
            modes=modes,
            bands=self.bands.analyze(x, noise_floor_db),
            glide=self.glide.analyze(window),
            envelope=self.envelope.analyze(x),
            noise=self.noise.analyze(window, modes),
            noise_floor_db=noise_floor_db,
            warnings=warnings,
        )

        if not modes:
            descriptors.warnings.append(
                "modal extraction found nothing; band decays are the only "
                "decay evidence available"
            )
        if descriptors.glide is not None and not np.isfinite(descriptors.glide.f_initial):
            descriptors.warnings.append(
                "glide tracking found no reliable frames in its frequency range"
            )
        return descriptors

    def analyze_file(self, path: str | Path) -> SoundDescriptors:
        """Load, prepare and analyze in one step."""
        return self.analyze(self.prep.prepare(self.prep.load(path)))
