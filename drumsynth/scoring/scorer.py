"""The one class you actually call.

    scorer = DrumScorer()
    card = scorer.score_files("reference.wav", "generated.wav")
    print(card.report())

    card, edits = scorer.score_and_attribute(reference, generated, params)
    for edit in edits[:5]:
        print(edit)
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from ..core.constants import Audio
from .analyzer import Analyzer
from .attribution import Attributor, ParameterSuggestion
from .comparator import Comparator, ComponentScore, ScoreCard
from .descriptors import SoundDescriptors
from .stft import MultiResolutionSTFTLoss


class DrumScorer:
    """Analysis, comparison, attribution and aggregation, wired together."""

    def __init__(
        self,
        sr: int = Audio.DEFAULT_SR,
        analyzer: Analyzer | None = None,
        comparator: Comparator | None = None,
        attributor: Attributor | None = None,
        stft_loss: MultiResolutionSTFTLoss | None = None,
    ) -> None:
        self.sr = int(sr)
        # One analyzer, deliberately: both signals must go through identical
        # windows, band edges and hop sizes or the comparison measures the
        # analysis rather than the audio.
        self.analyzer = analyzer or Analyzer(self.sr)
        self.comparator = comparator or Comparator()
        self.attributor = attributor or Attributor()
        self.stft_loss = stft_loss or MultiResolutionSTFTLoss(self.sr)

    @classmethod
    def for_fundamental(cls, f0: float, sr: int = Audio.DEFAULT_SR) -> "DrumScorer":
        """A scorer whose glide tracker is pointed at a known fundamental."""
        return cls(sr, analyzer=Analyzer.for_fundamental(f0, sr))

    # -- scoring --------------------------------------------------------------

    def describe(self, x: np.ndarray) -> SoundDescriptors:
        """Prepare and analyze one signal. Exposed because looking at the
        descriptors directly is usually more informative than the score."""
        return self.analyzer.analyze(self.analyzer.prep.prepare(x))

    def score(self, reference: np.ndarray, generated: np.ndarray) -> ScoreCard:
        reference_prepared = self.analyzer.prep.prepare(reference)
        generated_prepared = self.analyzer.prep.prepare(generated)

        # Never score below the reference's noise floor — on EITHER side. The
        # generated hit decays into digital silence; the reference does not,
        # and measuring them over different ranges reports the recording's
        # limits as if they were the synthesizer's errors.
        floor = self.analyzer.prep.estimate_noise_floor(reference_prepared)

        card = self.comparator.compare(
            self.analyzer.analyze(reference_prepared, noise_floor_db=floor),
            self.analyzer.analyze(generated_prepared, noise_floor_db=floor),
        )
        # Reported alongside the card, never folded into `total`.
        card.stft_loss = self.stft_loss(reference_prepared, generated_prepared)
        return card

    def score_files(
        self, reference_path: str | Path, generated_path: str | Path
    ) -> ScoreCard:
        prep = self.analyzer.prep
        return self.score(prep.load(reference_path), prep.load(generated_path))

    def score_and_attribute(
        self, reference: np.ndarray, generated: np.ndarray, params
    ) -> tuple[ScoreCard, list[ParameterSuggestion]]:
        card = self.score(reference, generated)
        return card, self.attributor.attribute(card, params)

    # -- sets -----------------------------------------------------------------

    def score_set(
        self, pairs: Sequence[tuple[np.ndarray, np.ndarray]]
    ) -> list[ScoreCard]:
        """Score several hits at once — soft, medium, hard, different drums.

        A parameter set tuned against a single sample will match it and
        generalize to nothing, and you will not find out until velocity mapping
        goes in. Always evaluate on a set.
        """
        return [self.score(reference, generated) for reference, generated in pairs]

    def aggregate(self, cards: Sequence[ScoreCard]) -> ScoreCard:
        """Combine per-hit cards, reporting the WORST per component.

        Not the mean. A parameter that is right for three hits and badly wrong
        for the fourth is a broken parameter, and averaging hides exactly the
        signal you need.
        """
        cards = list(cards)
        if not cards:
            return ScoreCard(total=0.0, components=[], warnings=["no cards to aggregate"])
        if len(cards) == 1:
            return cards[0]

        names = [component.name for component in cards[0].components]
        aggregated: list[ComponentScore] = []

        for name in names:
            found = []
            for index, card in enumerate(cards):
                try:
                    component = card.by_name(name)
                except KeyError:
                    continue
                if component.available:
                    found.append((index, component))

            if not found:
                template = cards[0].by_name(name)
                aggregated.append(
                    ComponentScore(name, 0.0, np.nan, template.unit,
                                   available=False, parameters=template.parameters)
                )
                continue

            worst_index, worst = min(found, key=lambda item: item[1].value)
            detail = dict(worst.detail)
            detail["worst_hit_index"] = worst_index
            detail["hit_values"] = [round(component.value, 4) for _, component in found]
            detail["mean_value"] = round(
                float(np.mean([component.value for _, component in found])), 4
            )
            aggregated.append(
                ComponentScore(
                    name=name,
                    value=worst.value,
                    raw_error=max(
                        (
                            component.raw_error
                            for _, component in found
                            if np.isfinite(component.raw_error)
                        ),
                        default=np.nan,
                    ),
                    unit=worst.unit,
                    detail=detail,
                    parameters=worst.parameters,
                    available=True,
                )
            )

        warnings = [f"aggregate of {len(cards)} hits; each component is the WORST hit"]
        for index, card in enumerate(cards):
            warnings.extend(f"hit {index}: {warning}" for warning in card.warnings)

        return ScoreCard(
            total=self.comparator.total(aggregated),
            components=aggregated,
            mode_matches=cards[int(np.argmin([card.total for card in cards]))].mode_matches,
            stft_loss=float(
                np.max([card.stft_loss for card in cards if np.isfinite(card.stft_loss)])
                if any(np.isfinite(card.stft_loss) for card in cards)
                else np.nan
            ),
            warnings=warnings,
        )
