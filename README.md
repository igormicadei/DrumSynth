# DrumSynth

Modal synthesis for membrane drums — kick, toms, snare shell — plus the scoring
and sample-handling infrastructure to fit it against real recordings.

A drum is ~109 numbers rather than a folder of WAVs. Velocity is continuous
instead of quantized to recorded layers, overlapping hits superpose with correct
physics, and tuning and damping are live parameters.

```python
from drumsynth import AudioIO, DrumPresets, DrumScorer, DrumVoice

params = DrumPresets.tom()                       # 92.5 Hz floor tom, 31 modes
voice = DrumVoice(params, control_period=64, seed=0)
AudioIO.write("tom.wav", voice.render_hit(4.0))

card = DrumScorer.for_fundamental(92.5).score(reference_audio, voice.render_hit(4.0))
print(card.report())
```

## Install

```bash
pip install -e ".[dev]"
pytest
```

## The studio

```bash
pip install -e ".[studio]"
streamlit run streamlit_app.py
```

A mixer for the parameters — one strip per mode, one per noise band, one master
— with the audio running in a **separate process that never stops**. Strike the
drum, and while it is still decaying, move a `t60` knob: the rest of *that*
decay changes. Strike again and the new hit superposes onto what is left of the
old one. S and M solo and mute individual partials live, so you can hear which
ring belongs to which mode.

Solo and mute are a monitoring mask applied at the mix, never a change to
`gain`: muting through gain would only take effect on the next strike, and a
sound designed while soloing would be silently wrong when saved.

The engine talks JSON over a pipe (`drumsynth.live`), so a crash in the UI
cannot glitch the audio and a stalled device cannot hang the UI. Its `null`
sink runs the whole thing without a sound card, which is how the live path is
tested in CI. See [docs/STUDIO.md](docs/STUDIO.md).

Needs Python 3.10+, numpy and scipy. `soundfile` is optional but recommended —
without it the stdlib `wave` fallback handles plain PCM only. `matplotlib` is
needed only for the diagnostic plots in `ScoreReport`.

## The three subsystems

```
drumsynth.synth      DrumParams in, audio out
drumsynth.scoring    two signals in, a ScoreCard out
drumsynth.samples    a directory in, validated sample sets out
drumsynth.live       an always-on audio process, driven over a pipe
drumsynth.studio     the Streamlit mixer on top of it
```

They are deliberately separable. The synthesizer does not import the scorer and
the scorer does not import the synthesizer, because **scoring takes audio on
both sides, always** — the reference is a WAV with no parameters attached, so
any descriptor that cannot be extracted from raw audio cannot be compared.
`DrumParams` is an optional extra input, used only to turn "mode near 197 Hz has
t60 40% too long" into `modes[7].t60 = 0.52`.

## Synthesis

```
strike ──> excitation ──> [ modal bank ] ──> mix ──> out
                      └─> [ noise bank ] ──┘
                                 ▲
                          tension feedback
                       (bank energy → mode freq)
```

25-35 impulse-excited modes, 3-4 bands of contact noise, and one global
nonlinearity. Three things fall out of that structure rather than being
parameters:

* **Overlapping hits.** The bank has persistent state and `strike()` adds to it.
  Flams, ghost notes, rolls and the resonance of the previous hit need no
  special-case code.

  ```python
  voice.render_sequence([(0.0, 0.4), (0.025, 1.0)], seconds=3.0)   # a flam
  ```

* **The pitch glide.** Total modal energy raises effective head tension, which
  raises every mode frequency by the same ratio. A hard hit glides down ~2
  semitones as it decays and a soft hit barely moves, with no velocity term
  anywhere in the glide path — measured on the preset: 0.05 / 0.54 / 2.07
  semitones at amplitudes 0.15 / 0.5 / 1.0, same `k`.

* **Warble.** Two modes 92 cents apart beat at ~4.8 Hz. Cost: one extra
  resonator. There is no `beat_rate` parameter.

`DrumPresets.tom()` is not a fitted model — it is the reference measurements in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) §2 turned into parameters, so a
correctly wired engine reproduces them. It does, to within ~10% on every decay
band, with one exception documented below.

## Scoring

```
reference.wav ──┐
                ├── Analyzer ──> SoundDescriptors ──┐
generated.wav ──┘                                   ├── Comparator ──> ScoreCard
                                                    │
DrumParams (optional) ──────────────────────────────┘   (attribution only)
```

The descriptors *are* the parameters: a parameter set is re-derived from the
reference audio, then diffed parameter against parameter. Eight weighted
components, each reported in its own units alongside a normalized value, so the
card says *where* to look rather than just *how bad*.

```
  component            score       error  unit   weight
  --------------------------------------------------------
  mode_frequency       0.992       0.165  cents  0.25
  mode_t60             0.991       0.001  ratio  0.20
  mode_amplitude       0.991       0.028  dB     0.10
  mode_completeness    0.985       9.000  modes  0.10
  band_decay           0.999       0.011  %      0.15
  glide                1.000       0.000  cents  0.10
  envelope             0.969       0.031  dB     0.05
  noise                0.983       0.101  dB     0.05
```

Modes are recovered with ESPRIT on decimated complex baseband, which resolves
partials far closer than the FFT bin spacing — on a test signal it separates
88.0 from 92.5 Hz and returns every frequency to under a cent, every t60 to
within 3%, with no spurious modes.

Five things the scorer refuses to do, each for a reason that costs you if
ignored:

* **Never score below the reference's noise floor** — on *either* side. A clean
  render measured 100 dB down against a reference that flattens at 60 dB
  reports the recording's limit as a synthesis error. Measured: a 3.75×
  fabricated decay error in one band, gone once both sides use the reference's
  floor.
* **Never null-test.** A 0.1 Hz frequency error decorrelates phase within
  seconds; subtraction is far stricter than the ear.
* **Never compare noise sample-wise.** Two realizations of the same process are
  uncorrelated. Statistics only — two renders differing only in noise seed score
  0.991.
* **Never sort-and-zip modes.** One missing mode shifts every subsequent pair.
  Hungarian assignment on a cents-distance cost matrix, with a gate.
* **Never trust a single scalar while hand-tuning.** `MultiResolutionSTFTLoss`
  is reported next to the total and deliberately excluded from it.

## Sample handling

The normalized DrumModalSynth library lives under `data/` — 3614 WAVs across 14
membrane drums and 15 cymbals, with slugified names and JSON manifests. The
audio itself is gitignored (3.5 GB); `data/file_tree.txt` is the generated
record of what the import produced, and the integrity tests check the manifests
against it without needing a single WAV. Re-populate `data/samples/` with:

```bash
python tools/import_samples.py <path-to-DrumModalSynth/data>
```

Velocity in this library is an SFZ *range*, not a recorded label, so every row
carries `velocity_low`/`velocity_high` and `velocity_is_exact: false`. The
`velocity` field is the band midpoint and exists only for code written against
a plain label — it is not a measurement, and the calibration must not treat it
as one. Round-robin layers are alternate recordings of the same band, which is
what `take` means, so both are kept.

[docs/DATA.md](docs/DATA.md) has the per-drum table, and the two things that are
genuinely missing: 20 SFZ regions pointing at files that were never recorded,
and 276 WAVs no region references. Neither was papered over.

Velocity is not a property of the drum. `f_static` and `t60` are the drum;
`gain`, noise `level` and `contact_time` are the hit. One `DrumParams` per drum,
always — a `SampleSet` is deliberately single-drum so nothing is tempted to
share one across two.

```python
library = SampleLibrary.scan("samples/")
for issue in library["floor_tom_16"].validate():
    print(issue)
```

`validate()` reports everything that produces a plausible-looking fit and a
wrong drum: clipping, second hits, truncated tails, velocity coverage gaps, a
set too short to fit the slowest decay, a drum retuned mid-session, and a
non-monotone velocity calibration.

That last one matters more than it looks. MIDI velocity is a controller value,
not a unit of energy — across a real span the steps compress badly:

```
     label     measured     step  normalized
      40.0      1.59 dB               0.0000
      55.0      6.03 dB    +4.44      0.3035
      70.0      9.38 dB    +3.35      0.5330
      85.0     12.05 dB    +2.67      0.7156
     100.0     14.32 dB    +2.27      0.8707
     115.0     16.21 dB    +1.89      1.0000
```

Fit against `velocity_normalized`, not `velocity`. If the calibration comes out
non-monotone, the session has a problem — a mislabeled take, a moved mic, a drum
retuned partway — and it should be investigated, not fitted around.

## Training

`drumsynth.fitting` implements ARCHITECTURE.md §8: five stages, each freezing
what the last one settled. Stages 1, 2 and 4 are **measurement and closed-form
solves**, not search — frequencies from ESPRIT on a soft layer (the drum at
rest), damping from measured band decays, gains from non-negative least squares
against a linearized basis. Only stage 5 is an optimizer.

Stage 3 is the part that matters. Per-velocity fitting always succeeds, so the
evidence that the velocity model is right is that the fitted table moves
smoothly and in the physical direction — brighter and louder with velocity, and
one `tension.k` for the whole drum.

```bash
streamlit run streamlit_app.py     # → the Training page
```

Pick a drum, watch the stages, read the ScoreCard against the samples, and load
the result straight into the live synth to play it.

While it runs you get the current and best generation, the loss curve, and live
GPU telemetry. When it finishes you get the ScoreCard, where the time went, and
the generated drum beside the sample as waveform, spectrum, band decay and
spectrogram — then it loads into the live engine so you can hit it. Every run is
kept under `out/runs/` and reopenable.

Stage 5 evaluates a whole generation in one batched call, which is what makes
CUDA worth using for it:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install nvidia-ml-py     # live GPU readings while it runs
```

Then pick the device on the page, or pass `--device cuda` to the worker. Stages
1-4 are LAPACK on small matrices and stay on the CPU. See
[docs/TRAINING.md](docs/TRAINING.md) for the stages, the measured baselines,
and what has and has not been verified.

## Where the design was wrong

Five things in the architecture did not survive being built. They are documented
with measurements in [docs/FINDINGS.md](docs/FINDINGS.md), not edited out of the
spec. The largest:

**The ~10 ms envelope peak does not emerge from summing impulse-excited modes,
and cannot.** Every mode starts at cosine phase — at its own maximum — so the
sum is at its maximum on the first sample and can only fall. The generated
attack peaks at 1.0 ms against the reference's 9.8 ms, with a 34 dB crest factor
against 19.7 dB. No mode count or frequency spread changes this.

`strike()` is implemented exactly as specified anyway, the disagreement is
pinned by a test, and error attribution reports it as
`excitation.contact_time` rather than proposing gain changes — because scaling
gains to chase an envelope shape is fitting around a missing mechanism. Fixing
it properly means building the excitation model the architecture already names
as a seam, and that belongs with velocity fitting.

## Examples

```bash
python examples/01_build_order.py out/     # each build step, one WAV at a time
python examples/02_score_a_hit.py          # score, attribute, aggregate
python examples/03_check_a_library.py      # quality gates on a demo session
```

## Documentation

* [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — the design and the physical
  measurements behind every decision. Read this before changing anything
  structural.
* [docs/FINDINGS.md](docs/FINDINGS.md) — where measurement disagreed with the
  design, and what was done about it.
* [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md) — module map, the fast-path
  derivation, performance, and what the tests actually pin.
* [docs/DATA.md](docs/DATA.md) — the imported sample library: what is in it,
  how velocity is represented, and what is missing.
* [docs/STUDIO.md](docs/STUDIO.md) — the parameter mixer and the live audio
  process behind it.
* [docs/TRAINING.md](docs/TRAINING.md) — fitting a drum to its samples: the
  five stages, what is measured rather than searched, and the ground-truth
  recovery numbers.

## Not built yet

* **Excitation / `contact_time`.** The seam exists; the model does not. See
  finding #1.
* **Snare wires.** The tom module plus one threshold-nonlinearity subsystem,
  ~5-6 parameters. `DrumPresets.snare_shell()` is the ~90% that already works.
* **Cymbals.** A separate project. A struck cymbal cascades energy from low
  modes into high ones, which a linear modal bank cannot do at any setting.
  ARCHITECTURE.md §9.
