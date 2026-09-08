# Modal Drum Synthesis — Architecture Specification

> **Status.** This is the original handoff specification, kept as written so it
> stays the record of intent. It is implemented in `drumsynth/` — see
> [IMPLEMENTATION.md](IMPLEMENTATION.md) for the module map.
>
> Five things in here did not survive contact with the implementation. They are
> not edited out; they are documented, with measurements, in
> [FINDINGS.md](FINDINGS.md). The largest is §2.2: the ~10 ms envelope peak does
> **not** emerge from summing impulse-excited modes, and cannot.

Handoff document. Implements a physically-motivated modal synthesizer for
membrane drums (kick, toms, snare), plus the scoring and data-loading
infrastructure needed to fit it against recorded samples.

The three skeleton modules that accompanied this spec — `drum_synth.py`,
`drum_score.py`, `drum_samples.py` — are implemented as the packages
`drumsynth.synth`, `drumsynth.scoring` and `drumsynth.samples`.

---

## 1. Goal

Synthesize realistic membrane-drum hits from a small parametric model
(~109 numbers per drum) rather than from samples, so that:

- strike velocity is continuous, not quantized to recorded layers
- overlapping hits (flams, ghost notes, rolls) superpose with correct physics
- tuning and damping are live parameters
- strike position becomes a gain vector later, with no restructuring

Scope for now: **membrane drums only.** Cymbals and hi-hat are explicitly a
separate project — see §9.

---

## 2. Physical basis

These are measurements from a reference floor-tom hit. They are the
justification for every design decision below; an implementer who changes the
architecture should check the change against them.

### 2.1 Frequency-dependent damping

Fitted T60 per band from one hit:

| band | T60 |
|---|---|
| 40–130 Hz | 2.33 s |
| 230–400 Hz | 0.75 s |
| 400–900 Hz | 0.55 s |
| 900–4000 Hz | 0.31 s |

A 7.5× spread. The −60 dB bandwidth edge collapses from 12.1 kHz in the first
20 ms → 474 Hz by 150 ms → 228 Hz in the tail.

**Implication:** `t60` falls roughly as 1/f. This trend is most of the realism.

### 2.2 Attack

- 10–90% rise: 4.4 ms
- envelope peak at 9.8 ms after onset (**not** at t=0)
- crest factor 19.7 dB
- 12.5% of total energy in the first 30 ms; 50% in the first 150 ms

**Implication:** the ~10 ms peak is *emergent* from summing modes at different
frequencies. It is not an attack envelope. Modes are impulse-excited.

### 2.3 Pitch glide

Instantaneous f0: 104.3 Hz at 10 ms → 98.4 at 160 ms → 93.9 at 610 ms →
asymptote 92.5 Hz. Depth 2.07 semitones, settling by ~1 s.

Verified as a genuine frequency change, not two modes trading dominance.
Short-window spectral peak location:

```
t=0.00s  peak @ 100.3 Hz
t=0.10s  peak @  97.7 Hz
t=0.25s  peak @  95.6 Hz
t=0.50s  peak @  93.9 Hz
t=1.00s  peak @  92.8 Hz
```

Two static modes cannot place a spectral peak at 95.6 Hz — a sum of two
resolved sinusoids peaks at one frequency or the other, never in between.
Single sliding lobe confirms real frequency modulation.

**Mechanism:** large-amplitude membrane displacement raises effective head
tension, which raises mode frequencies. The drum is physically tighter during
the loud part of its own decay.

**Implication:** the glide must be driven by resonator *energy*, not by time.
A time-based pitch envelope reproduces one velocity and is wrong at all others.

### 2.4 Decay linearity

Individual modes decay as a single exponential — a straight line in dB.

Every non-straight decay observed in the reference was explained by *more than
one mode*, never by one mode decaying non-exponentially:

- band curvature = sum of exponentials with different t60 (convex in dB)
- shallow-steep-shallow warble in an isolated band = two close modes beating
  (peaks found at 88.0 and 92.8 Hz)
- an apparent floor at −105 dB = the codec noise floor of a lossy reference,
  not the drum

**Implication:** no `decay_shape` parameter. If a fitted band looks curved, the
answer is more modes.

### 2.5 Modal content

f0 ≈ 92.5 Hz. Partials at ratios 1.63 and 2.13 — near the ideal circular
membrane values 1.594 and 2.136 (Bessel zeros). Membrane modes are
**inharmonic**; do not assume integer ratios.

---

## 3. Architecture

```
strike ──> excitation ──> [ modal bank ] ──> mix ──> out
                      └─> [ noise bank ] ──┘
                                 ▲
                          tension feedback
                       (bank energy → mode freq)
```

The split is **excitation + resonator**, NOT attack-resonator +
tail-resonator. The attack is the input that makes the system ring; the
resonant part is the bank's impulse response. There is no separate tail
generator and nothing to mix back together at the end.

Consequences that fall out for free:

- **Sequential hits.** The bank has persistent state. A new strike superposes
  onto the decaying state. Flams, ghost notes, rolls and "resonance of the
  previous hit" need no special-case code.
- **Tuning** = scaling mode frequencies. One parameter.
- **Muting** = raising mode damping, live. One parameter.
- **Strike position** = per-mode excitation gains (a mode with a node at the
  strike point receives no energy).

### 3.1 Mode count

Do not model thousands of modes. Mode density grows roughly as f², so almost
all of them sit above 2 kHz where the ear stops resolving individuals and
integrates them into a noise band.

- resolve individually up to ~1.5 kHz → **25–35 modes**
- above that → band-limited noise with fast decay, folded into the excitation

Cost is not the constraint: 35 modes ≈ 35 two-pole resonators ≈ a few hundred
MACs per sample. Trivial even on an MCU with an FPU.

---

## 4. Parameter model

### Mode (×25–35)

| param | unit | notes |
|---|---|---|
| `f_static` | Hz | frequency at rest |
| `gain` | linear | initial amplitude from a unit strike |
| `t60` | s | time to −60 dB |

Place **close pairs deliberately** (e.g. 88.0 and 92.8 Hz). The beating between
them is a large part of why real drums sound alive and synthesized ones sound
dead. Cost: one extra resonator.

No phase parameter — impulse excitation starts every mode in phase, which is
physically correct and a large part of why a hit sounds like a hit.

### NoiseBand (×3–4)

| param | unit | notes |
|---|---|---|
| `f_low`, `f_high` | Hz | e.g. 200–800, 800–2k, 2k–6k, 6k–15k |
| `level` | linear | initial amplitude |
| `t60` | s | 0.005–0.15, shorter for higher bands |

No attack. Level starts at max and decays immediately.

### Tension (×1)

| param | unit | notes |
|---|---|---|
| `k` | — | energy → frequency-ratio scaling |
| `tau` | s | smoothing of the energy estimate, ~0.05–0.2 |

```
ratio = 1 + k · smoothed_energy
f_i(t) = f_static_i · ratio
```

Global scalar applied to every mode. Set `k = 0` for a linear bank.

### Excitation (deferred, but the seam must exist)

`contact_time` (0.3–3 ms) is the width of the force pulse, which acts as a
lowpass on the excitation. Short contact = bright, long = dull. This is what
separates a stick from a mallet, and a hard hit from a soft one.

At a single fixed velocity it is absorbed into the fitted `gain` values. When
velocity is added, `contact_time` is reintroduced and the gains are **refit
against it** — not merely scaled.

### Budget

```
30 modes × 3      =  90
4 noise bands × 4 =  16
tension           =   2
output_gain       =   1
                     ───
                     109
```

---

## 5. Runtime — `drum_synth.py`

### 5.1 Resonator realization

**Use the coupled ("magic circle") form, not a direct-form biquad.** Frequency
is modulated every control period by the tension feedback, and a direct-form
2-pole jumps in amplitude when its coefficients change mid-ring.

```
eps = 2·sin(π·f/sr)

x += eps·y
y -= eps·x
x *= r
y *= r
```

with `r = exp(-ln(1000) / (t60·sr))`, `ln(1000) = 6.907755278982137`.

Output = `x` (or `y`). Energy = `x² + y²`. Cost: 4 multiplies, 2 adds.

**Strike:** `x += amplitude · gain`, `y` unchanged, for every mode
simultaneously. The `+=` (not `=`) is what makes overlapping hits superpose
correctly — this is load-bearing and deserves a unit test, because it is easy
to "simplify" into `=` later.

### 5.2 Per-sample update order

```
1. energy = Σ (x_i² + y_i²)          over all modes
2. energy_smoothed += (energy - energy_smoothed) · (1/(tau·sr))
3. ratio = 1 + k · energy_smoothed
4. for each mode: set frequency to f_static·ratio; step; accumulate
5. for each noise band: white → bandpass → × env; env *= decay_coef
6. out = output_gain · (Σ modes + Σ noise)
```

**Energy must be read before the modes advance**, or the feedback loop is
delayless and can go unstable.

### 5.3 `control_period`

The tension ratio does not need per-sample updates — the glide moves on a ~1 s
timescale. Recomputing `eps` for 30 modes every sample is the only real cost in
the engine. Update every 32–64 samples in production; leave at 1 while
validating.

### 5.4 Known issue in the skeleton

`DrumVoice.is_silent` is declared as a property taking a `threshold_db`
argument. A property cannot take arguments. Convert to a plain method or
hardcode the threshold.

---

## 6. Scoring — `drum_score.py`

### 6.1 Input contract

**Audio in, always.** Both signals go through the identical analysis chain and
produce the same `SoundDescriptors`. The reference is a WAV with no parameters
attached, so any descriptor that cannot be extracted from raw audio cannot be
compared.

`DrumParams` is an **optional** second input, used only by `Attributor` to
convert "mode near 197 Hz has t60 40% too long" into `modes[7].t60 = 0.52`.

The descriptors *are* the parameters. You re-derive the parameter set from the
reference audio, then diff parameter against parameter — which gives
per-parameter error without the reference ever having parameters.

### 6.2 Why not spectrogram L2

Linear-magnitude L2 is dominated by the loudest bins — in this sound, the
fundamental in the first 200 ms, roughly 40 dB above everything else. Such a
metric reports "very similar" while the entire transient region and the whole
3-second tail are wrong.

Fixes, in order of importance:

1. **Log magnitude.** Puts a −60 dB error and a −6 dB error on comparable
   footing, roughly matching perception. This single change matters more than
   the rest combined.
2. **Multi-resolution.** The 88.0/92.8 Hz pair needs ~500 ms to resolve; the
   4.4 ms attack needs ~5 ms. Use 256 / 1024 / 4096 / 16384 and sum.
3. **Separate modal from noise scoring.** The modal part is deterministic and
   matchable parameter-wise. The noise part is stochastic — two realizations of
   the same process have ~zero correlation. Only its statistics are matchable.

### 6.3 Component scores

Compare in native units and normalize:

| quantity | unit | tolerance |
|---|---|---|
| mode frequency | cents | ±10–20 c below 500 Hz, looser above |
| mode t60 | ratio | within ~15% |
| mode gain | dB | ±3 dB |
| glide trajectory f0(t) | cents | ±10 c |
| band decay slope | % | ~10% |
| broadband RMS envelope | dB | ±1 dB |

Weight each mode's frequency error by its amplitude — a 50-cent error on a
−55 dB partial is inaudible; the same error on the fundamental is not.

Score the glide as a **full trajectory**, not a depth number. Depth alone is
satisfiable by the wrong mechanism, and comparing the trajectory separates
`tension.k` (depth) from `tension.tau` (settle time).

### 6.4 Mode matching is an assignment problem

Sorting both lists by frequency and pairing by index breaks catastrophically
the first time a mode is missing — every subsequent pair shifts and all errors
look enormous. Use Hungarian assignment on a cents-distance cost matrix, with a
gate (~150 cents) above which pairs are refused and both sides report as
unmatched. Missing vs spurious modes are then their own score component.

### 6.5 Guard rails

- **Never score below the reference's noise floor.** A lossy reference flattens
  at the codec floor; fitting slopes into that region measures the codec.
- **Never null-test.** A 0.1 Hz frequency error decorrelates phase within
  seconds. Subtraction is far stricter than the ear and fails perceptually
  perfect matches.
- **Never compare noise sample-wise.** Statistics only.
- **Never trust a single scalar while hand-tuning.** It says "worse", not
  "which of the 109 numbers". `MultiResolutionSTFTLoss` is reported alongside
  the ScoreCard but deliberately excluded from `total`.

Ears remain the acceptance test. Metrics say where to look, not when to stop.

---

## 7. Training data — `drum_samples.py`

### 7.1 Velocity is not a property of the drum

`f_static` and `t60` are the drum. `gain`, noise `level` and `contact_time` are
the hit. A drum does not retune itself between a soft and a hard stroke.

**One `DrumParams` per drum, always.** Velocity enters at `strike()`. Fitting
per-velocity parameter sets and interpolating between them is sample-based
synthesis with extra steps — and if interpolation is needed, the parameters are
not capturing the mechanism, which is the entire reason to do modal synthesis.

### 7.2 `velocity` vs `velocity_normalized`

MIDI velocity is a controller value, not a unit of energy. MIDI 100 and 110 are
not reliably 10 units apart in hit energy, and the curve differs per drum and
per session. Fitting against the raw label bakes the controller's response
curve into the drum model.

`VelocityCalibration` measures actual energy per sample, fits a monotone curve
against the labels, and produces a physical 0–1 scale. **Fit against
`velocity_normalized`.**

If the calibration comes out non-monotone, the session has a problem
(mislabeled takes, moved mic, drum retuned partway). Investigate — do not fit
around it.

### 7.3 Tail truncation

The reference's slowest mode had t60 ≈ 2.3 s. A WAV cut before the tail reaches
the noise floor fits a **shorter t60 than the truth, silently** — nothing about
the result looks wrong. `SampleQuality.usable_duration` and `tail_truncated`
are checked on load; `SampleSet.validate()` reports whether the set can support
the slowest decay expected.

### 7.4 Other structure

- `SampleSet` is deliberately **single-drum**. Two drums in one set invites a
  shared `DrumParams` across both, which is physically wrong.
- `reference_sample()` picks for **fit quality**, not loudness: best SNR and
  longest usable tail, normally mid-to-upper velocity. The loudest hit is a bad
  choice — most nonlinear behavior, most likely clipped.
- `split()` holds out **interior** velocities. The question is whether the
  fitted curve interpolates to unrecorded strengths; holding out endpoints
  tests extrapolation, a different and easier-to-fail question.

---

## 8. Fitting procedure

Staged. Each stage freezes what the previous one established.

1. **Fit `f_static` and `t60`** from one mid-velocity reference sample, then
   **freeze them permanently.** They never move again.
2. **Fit excitation only, per velocity, independently:** `gain[]`, noise
   `level[]`, `contact_time`. ~35 numbers instead of 109, with the hard part
   already solved.
3. **Inspect the resulting table.** Does `contact_time` decrease smoothly and
   monotonically with velocity? Does the gain tilt across modes match what that
   `contact_time` predicts as a lowpass?
4. **Fit a smooth 2–3 parameter curve** through the per-velocity excitation
   parameters. Velocity is now continuous.
5. **Joint refinement** across all velocities with the mapping in place.

**Stage 3 is the real experiment.** Smooth, physically-directed variation means
the velocity model is right. Values that jump around, or a `contact_time` going
the wrong direction, mean the model is missing a mechanism — no interpolation
will paper over it. Per-velocity fitting alone *always* succeeds, which is why
it tells you nothing.

### 8.1 Falsifiable checks

- **`tension_k` must be velocity-invariant.** The glide deepens on hard hits
  automatically because more energy enters the bank. A fit wanting different
  `k` per velocity means the energy feedback is wrong.
- **If frozen `t60` genuinely cannot match hard hits**, that is real —
  amplitude-dependent damping exists. Model it as damping driven by the *same*
  energy signal the tension block already computes, not as per-velocity `t60`.
  One extra parameter, mechanism preserved.

### 8.2 Never fit to a single hit

A parameter set tuned against one sample will match it and generalize to
nothing, and this will not surface until velocity mapping goes in. Evaluate on
a set. `DrumScorer.aggregate` reports the **worst** per component, not the mean
— a parameter right for three hits and wrong for the fourth is broken, and
averaging hides exactly the signal needed.

---

## 9. Scope boundary: cymbals

Three families, split by **membrane vs cymbal** — not by wires.

| family | engine | reuse |
|---|---|---|
| kick, toms | modal bank | 100% |
| snare | modal bank + wire subsystem | ~90% |
| cymbals, hi-hat | nonlinear cascade model | excitation only |

**Cymbals break the modal bank at the physics level.** A struck cymbal is
strongly nonlinear: energy injected into low modes *cascades upward* into high
modes over the first few hundred ms, which is why a crash brightens after the
strike before fading. In a linear system energy only leaves a mode, never
transfers between modes — no parameter tuning produces this.

**Snare is the tom module plus one subsystem.** The wires are a threshold
nonlinearity: they buzz only when bottom-head displacement exceeds a gap. A
separate block reading bottom-head motion and injecting noise above threshold,
~5–6 parameters (gap, wire count, tension, noise band, coupling gain).
Sympathetic buzz comes free.

**Do not let cymbal requirements influence the membrane design.** Cymbal
physical modelling is genuinely hard; banded waveguides and nonlinear modal
coupling both exist and neither is a clean win. Sampling cymbals is a
defensible choice until the membrane side works.

---

## 10. Deliberately absent

Do not add these without re-reading the reason.

| omitted | why |
|---|---|
| `attack`, `attack_shape` | modes are impulse-excited; the ~10 ms peak emerges from mode summation |
| `decay_shape` | modal decay is a single exponential; all observed curvature came from multiple modes. A shape parameter lets a fitter hide missing modes behind a fake curve |
| `pitch_envelope` | the glide is energy-driven, not time-driven; a fixed envelope is right at one velocity and wrong at all others |
| `beat_rate` / `warble` | emerges from close mode pairs |
| per-velocity `DrumParams` | velocity varies excitation, not the drum |
| sustain / release | ADSR is a keyboard model assuming a key-down state. A drum has none. Every envelope here is two numbers: initial level, decay rate |

---

## 11. Build order

Each step is independently audible or checkable.

1. `ModeResonator` + `t60_to_coef`. Unit-test that `excite()` superposes.
2. `make_test_tom()` by hand: f0 ≈ 92.5 Hz, t60 2.3 s at the fundamental
   falling to ~0.3 s by 2 kHz, one close pair for warble, `tension.k = 0`,
   no noise bands.
3. `render_hit()` → `write_wav()`. **Listen.** A pure linear modal bank should
   already sound like a plausible drum.
4. Add the noise bank. Listen.
5. Turn on tension; tune `k` until the glide reads ~104 → 92.5 Hz. Listen.
6. `SignalPrep` + `BandDecayAnalyzer` + `GlideAnalyzer`. These three alone give
   a usable comparison and are all straightforward.
7. `Sample` / `SampleSet` / `VelocityCalibration`. Run `validate()` on the real
   library **before** any fitting.
8. `ModalAnalyzer` (ESPRIT / matrix pencil) last — it is the hard one. Use band
   decays as the proxy until the synth is roughly in the right place.
9. Staged fit per §8.

Change one variable at a time. Listen at every step.
