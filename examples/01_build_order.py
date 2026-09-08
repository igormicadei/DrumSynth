"""Build order from docs/ARCHITECTURE.md §11, one audible step at a time.

Each step writes a WAV. Listen at every step and change one variable at a time
— that is the whole method, and it is worth more than any metric here.

    python examples/01_build_order.py [output_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from drumsynth import (
    AudioIO,
    BandDecayAnalyzer,
    DrumPresets,
    DrumVoice,
    EnvelopeAnalyzer,
    GlideAnalyzer,
)

SR = 44100


class BuildOrder:
    """Renders each step of the build order and prints what to listen for."""

    def __init__(self, output: Path, sr: int = SR) -> None:
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.sr = sr

    def _write(self, name: str, audio: np.ndarray) -> Path:
        path = AudioIO.write(self.output / f"{name}.wav", audio, self.sr)
        print(f"    wrote {path}")
        return path

    def step_2_linear_bank(self) -> None:
        """A pure linear modal bank should already sound like a plausible drum.
        If it does not, nothing added later will fix it."""
        print("\n[2] modal bank only — no noise, no tension")
        params = DrumPresets.tom(with_noise=False, with_tension=False)
        audio = DrumVoice(params, self.sr, 64, seed=0).render_hit(4.0)
        self._write("step2_modal_only", audio)

        for band in BandDecayAnalyzer(self.sr).analyze(audio, -110.0):
            if band.is_valid:
                print(f"    {band}")
        print("    listen for: pitch, warble from the close pair, a tail that")
        print("    darkens as it decays. It will sound too clicky at the front.")

    def step_4_add_noise(self) -> None:
        print("\n[4] + noise bank")
        params = DrumPresets.tom(with_noise=True, with_tension=False)
        self._write("step4_with_noise", DrumVoice(params, self.sr, 64, seed=0).render_hit(4.0))
        print("    listen for: stick contact at the front. The noise bank has no")
        print("    attack — its level starts at maximum and decays immediately.")

    def step_5_turn_on_tension(self) -> None:
        print("\n[5] + tension feedback (the pitch glide)")
        params = DrumPresets.tom()
        voice = DrumVoice(params, self.sr, 64, seed=0)
        audio = voice.render_hit(4.0)
        self._write("step5_with_tension", audio)

        track = GlideAnalyzer(self.sr, (60.0, 140.0)).analyze(audio)
        print(f"    measured: {track}")
        print("    reference: 104.3 Hz at 10 ms -> 92.5 Hz asymptote, 2.07 semitones")

    def step_5b_velocity_invariance(self) -> None:
        """The falsifiable claim: k is velocity-invariant.

        The glide deepens on hard hits automatically, because more energy enters
        the bank. A fit wanting a different k per velocity means the energy
        feedback is wrong — not that k needs a velocity term.
        """
        print("\n[5b] the same k at three strengths")
        voice = DrumVoice(DrumPresets.tom(), self.sr, 64, seed=0)
        for label, amplitude in (("soft", 0.15), ("medium", 0.5), ("hard", 1.0)):
            _, freqs = voice.glide_trace(2.0, amplitude)
            depth = 12 * np.log2(freqs.max() / freqs[-1])
            print(f"    {label:<7} amplitude {amplitude:.2f} -> {depth:.2f} semitones")
            self._write(f"step5b_{label}", voice.render_hit(4.0, amplitude))
        print("    no velocity term appears anywhere in the glide path.")

    def step_the_attack(self) -> None:
        """Where the architecture and the measurement disagree. See FINDINGS.md."""
        print("\n[!] the attack, measured against the reference")
        audio = DrumVoice(DrumPresets.tom(), self.sr, 64, seed=0).render_hit(4.0)
        envelope = EnvelopeAnalyzer(self.sr, 0.001).analyze(audio)
        print(f"    generated: {envelope}")
        print("    reference: envelope peak at 9.8 ms, rise 4.40 ms, crest 19.7 dB")
        print("    Every mode is impulse-excited at phase zero, so the sum is at")
        print("    its maximum on the first sample and can only fall. The ~10 ms")
        print("    peak does not emerge — it needs a finite force pulse.")

    def step_superposition(self) -> None:
        """Flams, ghost notes and rolls need no special-case code."""
        print("\n[+] superposition: one bank, persistent state")
        voice = DrumVoice(DrumPresets.tom(), self.sr, 64, seed=0)
        self._write("flam", voice.render_sequence([(0.0, 0.4), (0.025, 1.0)], 3.0))
        self._write(
            "roll",
            voice.render_sequence(
                [(index * 0.08, 1.0 if index % 4 == 0 else 0.45) for index in range(12)],
                4.0,
            ),
        )
        print("    no crossfade, no voice stealing, no special case for either.")

    def run(self) -> None:
        self.step_2_linear_bank()
        self.step_4_add_noise()
        self.step_5_turn_on_tension()
        self.step_5b_velocity_invariance()
        self.step_the_attack()
        self.step_superposition()
        print(f"\ndone. {len(list(self.output.glob('*.wav')))} files in {self.output}")


if __name__ == "__main__":
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("out/build_order")
    BuildOrder(destination).run()
