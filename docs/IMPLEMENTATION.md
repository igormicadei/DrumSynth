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
├── samples/               §7
│   ├── quality.py         SampleQuality, QualityChecker
│   ├── sample.py          Sample
│   ├── calibration.py     VelocityCalibration
│   ├── sample_set.py      SampleSet
│   └── library.py         SampleLibrary
├── live/                  the always-running audio process
│   ├── protocol.py        Command, Event
│   ├── engine.py          LiveEngine
│   ├── sinks.py           NullSink, WavSink, DeviceSink
│   ├── worker.py          the subprocess entry point
│   └── client.py          LiveSynth
└── fitting/               §8
    ├── targets.py         Layer, FitTarget, TargetBuilder, DrumCatalogue
    ├── objective.py       TensionTrajectory, LinearVoiceBasis, SpectralTarget,
    │                      LevelMatch
    ├── stages.py          ModalStage, ExcitationStage, TensionStage,
    │                      InspectionStage, VelocityCurveStage, JointStage
    ├── trainer.py         DrumTrainer, FitResult, FitEvaluator
    ├── worker.py          the subprocess entry point
    └── client.py          TrainingRun
```

### The sample library is generated, not curated

Training assets live outside the Python package in `data/`, and every file in
it is produced by `tools/import_samples.py` from the vendor library — including
`data/file_tree.txt`. Nothing there is hand-edited, so a manifest can never
drift from the files it describes, and re-running reproduces the same output
byte for byte.

Two properties are load-bearing and have regression tests:

* **One manifest is one physical drum.** The instrument key is the whole folder
  chain below the family root, not its first element. Keyed on the first
  element, `Toms_Stereo/Tom1..Tom4` collapses into one instrument — four
  different drums sharing one `DrumParams`, which is the thing ARCHITECTURE.md
  §7.4 exists to prevent.
* **A re-run leaves nothing behind.** Manifests from a previous naming scheme
  are deleted, and sample directories from one are reported (`--prune` removes
  them). A stale manifest is indistinguishable from a real one and points at
  sample directories that no longer exist.

`tests/test_data_integrity.py` checks the committed manifests against
`data/file_tree.txt`: every row resolves to a file that exists, paths are
relative to the manifest, source paths carry no drive letters, velocity bands
are internally consistent, and no manifest spans two source instruments. It
needs no audio, so it runs anywhere.

See [DATA.md](DATA.md) for what the library actually contains.

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

### Live edits do not restart the ring

`DrumVoice.update_params` adopts a new parameter set without clearing state,
and rebuilds only what actually changed. That is what makes the studio worth
having: a knob moved while the drum is decaying changes the rest of *that*
decay. It works because of the same property that lets the tension feedback
modulate frequency every control period — the coupled form keeps amplitude in
(x, y) and frequency in a separate coefficient, so both can be retuned mid-ring.

Rebuilding selectively matters: redesigning a noise band's bandpass resets its
filter state, which clicks, and it must not happen because a level slider moved.

`ModalBank.monitor_mask` is a per-mode output scale applied at the mix. It is a
listening aid, not a drum parameter — not in `DrumParams`, never serialized,
never seen by a fit. Muting through `gain` instead would only take effect on the
next strike.

### The audio process owns its own state

All synthesizer mutation happens on the audio thread, which drains a deque of
commands at the top of every block. No lock goes near the render call, because a
lock held by the UI side is a dropout on the audio side. See
[STUDIO.md](STUDIO.md).

### `MultiResolutionSTFTLoss` is reported, never totalled

`ScoreCard.stft_loss` sits next to `total` and is excluded from it, per §6.5.
A scalar says "worse", not "which of the 109 numbers".

---

### Fitting is measurement first, search last

Four of the five stages in `drumsynth/fitting/` never call an optimizer.
Frequencies come from ESPRIT on a soft layer, damping from `DampingCurve`
through measured band decays, gains from non-negative least squares against a
linearized basis, and the velocity curve from a log-log line fit. Only stage 5
searches.

The lever that makes this affordable is that the render is **linear in the gains
and the noise levels**, so `LinearVoiceBasis` evaluates a candidate as a matrix
product against a cached unit-gain basis — 170× faster than rendering. The basis
is built on a tension trajectory captured from a real render, and relinearized
once the gains are roughly right; without that relinearization the *true*
excitation scores 11.68 dB instead of 0.07 dB, so the fit is being asked to find
something that is not the answer.

`ModalBank.process_modes` returning per-mode rows costs the same as returning
the mix (0.0044 s against 0.11 s before the shared `_advance` refactor), which
is what makes building the basis cheap enough to redo.

See [TRAINING.md](TRAINING.md) for the stages, the measured baselines, and the
two places a plausible-looking fit goes wrong.

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

378 tests, ~4 min. Nine files (`test_data_integrity.py` is heavily parametrized — one case per manifest per check):

* `tests/test_synth.py` — superposition, decay accuracy, tension behaviour, and
  the `ModalBank` ↔ `ModeResonator` equivalence.
* `tests/test_scoring.py` — analysis accuracy against signals with exactly known
  content, mode matching under a missing mode, and score discrimination.
* `tests/test_samples.py` — quality gates, velocity calibration, splits.
* `tests/test_integration.py` — the full path, plus the architecture's §2
  measurements reproduced from a render, plus the guard rails.
* `tests/test_import_samples.py` — the SFZ importer, on miniature source trees
  with the same folder shapes as the real library.
* `tests/test_data_integrity.py` — the committed manifests against
  `data/file_tree.txt`. Skipped when a checkout has no imported library.
* `tests/test_live.py` — the audio process end to end against the `null` sink:
  strikes superposing, parameters changing mid-ring, monitoring, recording.
* `tests/test_studio.py` — the Streamlit app through `AppTest`, in-process,
  including the Training page's result views driven by a real fit.
* `tests/test_fitting.py` — the fitter against synthetic drums with known
  parameters. A fit that matches a recording it was fitted to proves nothing;
  these ask whether it puts the modes back where they were.

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
| `test_parameters_change_without_restarting_the_ring` | live edits keep the decay |
| `test_solo_does_not_touch_the_gains` | monitoring is not a parameter |
