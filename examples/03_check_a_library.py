"""Run the quality gates over a sample library BEFORE any fitting.

Every problem this reports is one that produces a plausible-looking fit and a
wrong drum. Read the output before you fit anything.

    python examples/03_check_a_library.py /path/to/samples
    python examples/03_check_a_library.py            # synthetic demo session
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

from drumsynth import AudioIO, DrumPresets, DrumVoice, SampleLibrary

SR = 44100


class LibraryCheck:
    """Scan, validate, calibrate, report."""

    def __init__(self, root: Path, sr: int = SR) -> None:
        self.root = root
        self.sr = sr

    def run(self) -> None:
        library = SampleLibrary.scan(self.root, sr=self.sr)
        if not len(library):
            print(f"no parseable samples under {self.root}")
            print("expected filenames like 'floor_tom_16_v100_take2.wav'")
            return

        print(library.summary())
        for sample_set in library:
            print()
            print("-" * 78)
            print(sample_set.calibrate().report())
            print()
            reference = sample_set.reference_sample()
            print(f"  reference sample for fitting f_static and t60: {reference}")
            print(
                "    chosen for fit quality — best SNR and longest usable tail — "
                "not for loudness"
            )
            train, holdout = sample_set.split(0.2)
            print(
                f"  split: train v{[s.velocity for s in train]} "
                f"holdout v{[s.velocity for s in holdout]}"
            )
            print("    interior velocities held out: the question is whether the")
            print("    fitted curve INTERPOLATES to strengths never recorded.")

    @staticmethod
    def synthetic_session(destination: Path, sr: int = SR) -> Path:
        """A demo session rendered from the tom preset, with one bad take."""
        destination.mkdir(parents=True, exist_ok=True)
        params = DrumPresets.tom()
        rng = np.random.default_rng(0)

        for velocity in (35, 50, 65, 80, 95, 110, 125):
            voice = DrumVoice(params, sr, 64, seed=velocity)
            hit = voice.render_hit(3.5, amplitude=(velocity / 127.0) ** 1.6)
            padded = np.concatenate([np.zeros(600), hit])
            padded += 4e-5 * rng.standard_normal(len(padded))
            AudioIO.write(
                destination / f"floor_tom_16_v{velocity}.wav", padded, sr,
                normalize=False,
            )

        # One take cut short, to show what truncation detection catches.
        voice = DrumVoice(params, sr, 64, seed=7)
        short = voice.render_hit(0.6, amplitude=0.6)
        AudioIO.write(
            destination / "floor_tom_16_v70.wav", short, sr, normalize=False
        )
        return destination


if __name__ == "__main__":
    if len(sys.argv) > 1:
        LibraryCheck(Path(sys.argv[1])).run()
    else:
        print("no path given — generating a synthetic demo session\n")
        temporary = Path(tempfile.mkdtemp()) / "session"
        LibraryCheck(LibraryCheck.synthetic_session(temporary)).run()
