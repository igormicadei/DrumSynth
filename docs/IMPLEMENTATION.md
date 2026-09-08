# Implementation map

Where each part of [ARCHITECTURE.md](ARCHITECTURE.md) lives, and the decisions
that are specific to the code rather than to the design.

The three skeleton modules became three packages under a shared `core`:

| skeleton | package |
|---|---|
| `drum_synth.py` | `drumsynth.synth` |
| `drum_score.py` | `drumsynth.scoring` |
| `drum_samples.py` | `drumsynth.samples` |
| — | `drumsynth.core` (units, audio I/O, shared DSP) |

Every public name is importable from `drumsynth` directly.

---

## Layout

```
drumsynth/
├── core/                  units, I/O and DSP shared by all three subsystems
│   ├── constants.py       Audio, Decibels, Cents, Decay
│   ├── audio_io.py        AudioIO
│   └── dsp.py             FilterDesign, EnvelopeFollower, LinearRegression, SpectralPeak
├── synth/                 §3-§5 of the architecture
│   ├── params.py          Mode, NoiseBand, Tension, DrumParams
│   ├── resonators.py      ModeResonator, ModalBank, TensionTracker
│   ├── noise.py           NoiseVoice, NoiseBank
│   ├── voice.py           DrumVoice, DrumKit
│   ├── kit.py             DrumSequencer, Hit
│   └── presets.py         DampingCurve, ModalLayout, ExcitationTilt, DrumPresets
├── scoring/               §6
│   ├── descriptors.py     ModeEstimate, BandDecay, GlideTrack, EnvelopeCurve,
│   │                      NoiseStats, SoundDescriptors
│   ├── prep.py            SignalPrep
│   ├── modal.py           ModalAnalyzer      (ESPRIT / matrix pencil)
│   ├── bands.py           BandDecayAnalyzer
│   ├── glide.py           GlideAnalyzer
│   ├── envelope.py        EnvelopeAnalyzer
│   ├── noise.py           NoiseAnalyzer
│   ├── analyzer.py        Analyzer           (runs all of the above)
│   ├── matching.py        ModeMatch, ModeMatcher
│   ├── comparator.py      ComponentScore, ScoreCard, Comparator
│   ├── stft.py            MultiResolutionSTFTLoss
│   ├── attribution.py     ParameterSuggestion, Attributor
│   ├── scorer.py          DrumScorer
│   └── report.py          ScoreReport
└── samples/               §7
    ├── quality.py         SampleQuality, QualityChecker
    ├── sample.py          Sample
    ├── calibration.py     VelocityCalibration
    ├── sample_set.py      SampleSet
    └── library.py         SampleLibrary
```

Dependency direction is one-way: `synth` and `scoring` both depend on `core`
and on nothing else; `samples` depends on `scoring` for analysis. The
synthesizer never imports the scorer, which keeps the input contract honest —
scoring takes audio on both sides, always.

---

## Implementation decisions

### `ModalBank` is a fast path, and `ModeResonator` is the definition

`ModeResonator` is the architecture written out literally: the coupled ("magic
circle") form, one sample at a time, four multiplies and two adds. A 4-second
render of 30 modes that way is 5.3 million Python loop iterations.

`ModalBank` produces identical output by exploiting the fact that over any span
where the frequency is held constant, the mode is linear and time-invariant.
Its state recurrence

```
x[n] = 2·r·cos(w)·x[n-1] − r²·x[n-2]
```

has poles at `r·exp(±jw)`, so the whole span has a closed form

```
x[k] = r^k · (A·cos(w·k) + B·sin(w·k))
```

with `A` and `B` fixed by two explicit steps. That turns a per-sample loop into
two array expressions per control block. A 4-second, 31-mode render at
`control_period=64` takes ~0.29 s.

The equivalence is not assumed. `tests/test_synth.py::TestModalBankMatchesReference`
checks output and final state against `ModeResonator` to 1e-10, across a single
block, across 64-sample blocks, and across a mid-render frequency change. If
the reference implementation changes, that test is what catches the fast path
drifting away from it.

Blocks shorter than 4 samples take a direct path — setting up the closed form
costs more than a handful of iterations saves.

### Noise levels are RMS amplitudes, not raw filter multipliers

Each `NoiseVoice` divides out its own bandpass's noise gain (computed from the
filter's frequency response, so it is deterministic and free). Without it,
`level` means something different in a 600 Hz-wide band and a 9 kHz-wide one,
and a fitter comparing two bands is really comparing their bandwidths.

### Mode gains are normalized to unit energy

`ExcitationTilt.normalized` scales the gains so `sum(gain²) == 1`, which puts
the bank's energy at exactly 1.0 immediately after a unit strike. `tension.k`
then reads directly as *peak frequency ratio minus one* — the tom's
`k = 0.127` is 2.07 semitones, by construction. Without it, `k` silently
changes meaning every time a mode is added. `output_gain` absorbs the level
change and is set analytically: the first sample is exactly
`output_gain · sum(gain)`, so the peak of a unit strike is known without
rendering anything.

### The analysis chain is shared, not merely equivalent

`DrumScorer` holds **one** `Analyzer` and runs both signals through it. Not
because it holds state — it does not — but because every window length, band
edge and hop size biases the result, and the only way those biases cancel is if
both sides get exactly the same ones. `Analyzer.for_fundamental(f0)` is the
one thing worth setting per drum: the glide tracker searches a narrow band, and
pointed at the wrong octave it locks onto a partial and reports a confident,
wrong trajectory.

### Normalization shape is selectable

`Comparator(normalization="linear")` is the architecture's contract: 1.0 at zero
error, 0.0 at the tolerance, nothing below. That is right for reading a card —
the tolerances are acceptance thresholds, and past them the component is simply
wrong. `"soft"` uses `exp(-error/tolerance)`: 0.37 at the tolerance, asymptotic
to zero, never flat. Use it when something automated is following the gradient,
because a component pinned at 0.0 gives an optimizer nothing to descend.

Either way, `raw_error` carries the error in native units and is what you
should actually read.

### `MultiResolutionSTFTLoss` is reported, never totalled

`ScoreCard.stft_loss` sits next to `total` and is excluded from it, per §6.5.
A scalar says "worse", not "which of the 109 numbers".

---

## Performance

Measured on a 4-second render at 44.1 kHz, 31 modes and 4 noise bands:

| operation | time |
|---|---|
| `DrumVoice.render_hit(4.0)`, `control_period=64` | ~0.29 s |
| `DrumVoice.render_hit(4.0)`, tension off | ~0.21 s |
| `DrumVoice.render_hit(4.0)`, `control_period=1` | ~5.9 s |
| `Analyzer.analyze` (full chain, including ESPRIT) | ~0.39 s |
| `DrumScorer.score` (both sides + STFT loss) | ~0.86 s |

`control_period=1` with tension active is ~20× slower than 64 and is there for
validation, as the architecture intends. With `tension.k = 0` the
bank is linear, the ratio never moves, and the render proceeds in 8192-sample
chunks regardless of `control_period`.

---

## Testing

122 tests, ~50 s. Four files:

* `tests/test_synth.py` — superposition, decay accuracy, tension behaviour, and
  the `ModalBank` ↔ `ModeResonator` equivalence.
* `tests/test_scoring.py` — analysis accuracy against signals with exactly known
  content, mode matching under a missing mode, and score discrimination.
* `tests/test_samples.py` — quality gates, velocity calibration, splits.
* `tests/test_integration.py` — the full path, plus the architecture's §2
  measurements reproduced from a render, plus the guard rails.

The tests that matter most are the ones pinning claims that are easy to break
by accident:

| test | claim |
|---|---|
| `test_excite_superposes_rather_than_replacing` | `+=` not `=` in `excite()` |
| `TestModalBankMatchesReference` | the fast path is not a different synthesizer |
| `test_a_missing_mode_does_not_shift_the_rest` | assignment, not sort-and-zip |
| `test_a_different_noise_seed_barely_moves_the_score` | noise compared statistically |
| `test_scoring_stops_above_the_reference_noise_floor` | the reference's floor bounds both sides |
| `test_the_attack_does_not_match_and_that_is_known` | finding #1 is not a regression |
