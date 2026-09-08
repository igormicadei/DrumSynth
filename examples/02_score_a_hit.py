"""Score a generated hit against a reference, and read the result.

With no arguments it scores the tom preset against a deliberately broken copy
of itself, so the whole path runs without any audio on disk.

    python examples/02_score_a_hit.py
    python examples/02_score_a_hit.py reference.wav generated.wav
"""

from __future__ import annotations

import sys

from drumsynth import (
    DrumParams,
    DrumPresets,
    DrumScorer,
    DrumVoice,
    ScoreReport,
    Tension,
)

SR = 44100


class ScoringDemo:
    """The three ways to use the scorer: score, attribute, aggregate."""

    def __init__(self, sr: int = SR) -> None:
        self.sr = sr
        self.reference_params = DrumPresets.tom()
        self.scorer = DrumScorer.for_fundamental(
            self.reference_params.fundamental(), sr
        )

    def broken(self) -> DrumParams:
        """A copy with three specific, known errors in it."""
        params = self.reference_params.copy()
        params.modes[2] = params.modes[2].scaled(freq_ratio=2 ** (45 / 1200))
        params.modes[5] = params.modes[5].scaled(t60_ratio=2.2)
        params.tension = Tension(k=params.tension.k * 0.4, tau=0.12)
        params.name = "broken_tom"
        return params

    def score_one_hit(self) -> None:
        print("=" * 78)
        print("  ONE HIT: preset against a copy with three known errors")
        print("=" * 78)
        broken = self.broken()
        reference = DrumVoice(self.reference_params, self.sr, 64, seed=1).render_hit(3.5)
        generated = DrumVoice(broken, self.sr, 64, seed=1).render_hit(3.5)

        card, edits = self.scorer.score_and_attribute(reference, generated, broken)
        print(card.report())
        print()
        print(ScoreReport.suggestions(edits, 8))
        print()
        print("  the three errors that were actually introduced:")
        print("    modes[2].f_static  +45 cents")
        print("    modes[5].t60       x2.2")
        print("    tension.k          x0.4")

    def score_a_set(self) -> None:
        """Never fit to a single hit.

        A parameter set tuned against one sample will match it and generalize to
        nothing, and you will not find out until velocity mapping goes in.
        `aggregate` reports the WORST component across the set, not the mean —
        a parameter right for three hits and wrong for the fourth is broken, and
        averaging hides exactly the signal you need.
        """
        print()
        print("=" * 78)
        print("  A SET: four strengths, aggregated on the WORST hit")
        print("=" * 78)
        broken = self.broken()
        reference_voice = DrumVoice(self.reference_params, self.sr, 64, seed=1)
        generated_voice = DrumVoice(broken, self.sr, 64, seed=1)

        pairs = [
            (
                reference_voice.render_hit(3.0, amplitude),
                generated_voice.render_hit(3.0, amplitude),
            )
            for amplitude in (0.15, 0.4, 0.7, 1.0)
        ]
        cards = self.scorer.score_set(pairs)
        for amplitude, card in zip((0.15, 0.4, 0.7, 1.0), cards):
            print(f"  amplitude {amplitude:.2f} -> total {card.total:.3f}")

        aggregate = self.scorer.aggregate(cards)
        mean = sum(card.total for card in cards) / len(cards)
        print(f"\n  mean of the totals : {mean:.3f}  <- hides the worst hit")
        print(f"  aggregate total    : {aggregate.total:.3f}  <- what to trust")
        print()
        for component in aggregate.worst(3):
            worst_hit = component.detail.get("worst_hit_index", "?")
            print(
                f"    {component.name:<18} {component.value:.3f}  "
                f"(worst at hit {worst_hit}, per-hit {component.detail.get('hit_values')})"
            )

    def score_files(self, reference_path: str, generated_path: str) -> None:
        card = self.scorer.score_files(reference_path, generated_path)
        print(card.report())


if __name__ == "__main__":
    demo = ScoringDemo()
    if len(sys.argv) == 3:
        demo.score_files(sys.argv[1], sys.argv[2])
    else:
        demo.score_one_hit()
        demo.score_a_set()
