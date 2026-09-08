# Findings

Five places where the architecture as specified did not survive being built.
Each one is a measurement, not an opinion, and each is reproducible from the
repository. Nothing here was quietly worked around — the deviations are in the
code with the reasoning attached, and the ones that were left alone are pinned
by tests so they cannot be mistaken for regressions.

---

## 1. The ~10 ms envelope peak does not emerge. It cannot.

**Architecture §2.2 claims:**

> the ~10 ms peak is *emergent* from summing modes at different frequencies. It
> is not an attack envelope. Modes are impulse-excited.

**It is not, and no arrangement of modes makes it so.** Every mode is excited by
`x += amplitude * gain` with `y` unchanged, which starts it at cosine phase —
at its own maximum. The output is the sum of those maxima on the first sample,
and a sum of positive numbers can only decrease. The waveform peak is at t=0 by
construction, for any mode count, any frequency spread, any gain distribution.

Measured, tom preset against the reference hit:

| | reference | generated |
|---|---|---|
| envelope peak | 9.8 ms | 1.0 ms |
| 10-90% rise | 4.40 ms | 0.00 ms |
| crest factor | 19.7 dB | 34.0 dB |
| energy in first 30 ms | 12.5% | 42.8% |

Everything else about the preset matches the reference closely (see §5 below),
which is what makes this one worth stating plainly rather than tuning around.

**What was done.** Nothing, deliberately. `strike()` is implemented exactly as
specified, the disagreement is pinned by
`tests/test_integration.py::test_the_attack_does_not_match_and_that_is_known`,
and `Attributor.suggest_envelope_edits` reports it as
`excitation.contact_time` rather than proposing gain changes — because scaling
gains to chase an envelope shape is fitting around a missing mechanism, which
is the failure mode §8.3 of the architecture warns about.

**What would fix it.** The seam the architecture already names: `contact_time`,
the width of the force pulse (§4, "Excitation (deferred, but the seam must
exist)"). Two candidates, both measured on the tom preset:

| excitation | peak | rise 10-90 | crest |
|---|---|---|---|
| impulse into `x` (as specified) | 2.0 ms | 0.00 ms | 33.9 dB |
| impulse into `y` (velocity, not displacement) | 2.0 ms | 0.00 ms | 31.8 dB |
| 3 ms half-sine force into `x` | 18.0 ms | 14.97 ms | 23.3 dB |
| 8 ms half-sine force into `x` | 20.5 ms | 5.99 ms | 19.8 dB |
| *reference* | *9.8 ms* | *4.40 ms* | *19.7 dB* |

A finite force pulse reaches the right region. A pulse somewhere between 3 and
8 ms would land close on all three numbers at once. That is a real design
decision about the excitation model, not a parameter tweak, and it belongs with
velocity fitting where `contact_time` gets fitted rather than guessed.

---

## 2. A symmetric tension smoother makes the glide start backwards.

**Architecture §5.2 specifies:**

```
energy_smoothed += (energy - energy_smoothed) · (1/(tau·sr))
```

Starting from zero, that smoother has to *climb* to meet the energy that
arrived instantaneously at the strike. The frequency ratio therefore rises for
the first few `tau`, peaks, and only then falls — a pitch **rise** over the
first ~0.3 s. The reference (§2.3) falls monotonically from 10 ms onward:
104.3 → 98.4 → 93.9 → 92.5 Hz.

No value of `k` or `tau` fixes the direction. The shape is wrong at the start
regardless.

**What was done.** `Tension.instant_attack` (default `True`): rising energy is
followed immediately, falling energy is smoothed with `tau`. Physically the
head tightens the instant it is displaced; the smoothing exists to stop the
ratio tracking the beat between close mode pairs, not to model a delay. Setting
it `False` restores the literal symmetric one-pole, which is worth hearing once.

This also makes `tau` mean what `Attributor` assumes it means. With a symmetric
smoother, `tau` sets both the rise and the settle, so "glide depth wrong → k,
settle time wrong → tau" (§6.3) is not separable. With instant attack it is.

**Result.** The preset's glide, measured by `GlideAnalyzer`:
102.2 → 90.0 Hz, 2.21 semitones, monotone throughout. Reference: 2.07 semitones.

---

## 3. `control_period` had no effect on the smoothing coefficient.

Following §5.3 literally — update the ratio every 32-64 samples using a
per-sample coefficient — stretches the glide's settle time by that same factor.
A voice at `control_period=64` would glide 64× more slowly than one at 1, which
is the exact artifact `control_period` is supposed to be free of.

**What was done.** `Tension.smoothing_coef(sr, period)` takes the update period,
and `TensionTracker` is constructed with it. At `control_period=64` the tracked
fundamental stays within 3.3 cents of the `control_period=1` trajectory
everywhere.

Worth noting what that 3.3 cents does to a waveform comparison: the two renders
correlate at only 0.988, because a 3-cent difference fully decorrelates phase
within a couple of seconds. That is the architecture's own "never null-test"
guard rail (§6.5), demonstrated on the engine's own output, and it is why
`tests/test_synth.py::test_control_period_is_inaudible` compares trajectories
rather than samples.

---

## 4. Guard rails have to be enforced across BOTH signals, not each separately.

**Architecture §6.5:** *"Never score below the reference's noise floor."*

Implemented per-signal, this does the opposite of what it says. A synthesized
hit decays into digital silence and can be measured 100 dB down; a recorded
reference flattens at its codec floor 60 dB down. Each side then gets fitted
over a *different* range, and the difference reads as a synthesis error.

Measured, scoring a clean render against a copy of itself with noise added at
-60 dB (so the two are acoustically identical):

| band | fitted t60 ratio, per-signal floors | with the reference's floor on both |
|---|---|---|
| 40-130 Hz | 0.957 | 1.000 |
| 130-230 Hz | 1.045 | 0.997 |
| 230-400 Hz | 1.028 | 0.999 |
| 400-900 Hz | 0.975 | 1.001 |
| **900-2000 Hz** | **3.751** | **0.996** |

The 900-2000 Hz band sits far below the fundamental, so it is the first to
disappear into the reference's floor — and the first to report a confident,
fabricated 3.75× decay error.

**What was done.** Two changes.

* `Analyzer.analyze(..., noise_floor_db=...)` takes an override, and
  `DrumScorer.score` measures the reference's floor and passes it for **both**
  sides, warning when the generated signal is materially cleaner.
* `BandDecayAnalyzer` estimates a **per-band** floor as well. The broadband
  floor is not the floor in every band: a band 50 dB below the fundamental is
  almost entirely noise, and a fit anchored to the broadband floor never clips.

---

## 5. Everything else in §2 reproduces.

Stated for balance — the four findings above are the exceptions.

The tom preset in `DrumPresets.tom()` is not fitted. It is the architecture's
own claims turned into parameters: the §2.1 damping table interpolated in
log-log space, circular-membrane Bessel ratios for the frequencies, the §2.5
head stretch that puts the first two partials at 1.63 and 2.13, one deliberate
close pair, and `k` set from the §2.3 glide depth. Rendered and measured back:

| quantity | reference | generated |
|---|---|---|
| 40-130 Hz t60 | 2.33 s | 2.55 s |
| 230-400 Hz t60 | 0.75 s | 0.84 s |
| 400-900 Hz t60 | 0.55 s | 0.61 s |
| glide depth | 2.07 semitones | 2.21 semitones |
| f0 asymptote | 92.5 Hz | 90.0 Hz |

Within ~10% on every decay band and ~7% on the glide, with nothing fitted. The
frequency-dependent damping curve really is most of the realism, and the
energy-driven glide really does reproduce the measured trajectory.

Two limits of the preset worth knowing:

* **30 modes on a 92.5 Hz drum reach ~700 Hz, not 1.5 kHz.** Mode density grows
  as f², so §3.1's "25-35 modes, resolved to ~1.5 kHz" needs about 140 modes at
  this fundamental. Everything above ~700 Hz is the noise bank's job, which is
  what §3.1 actually prescribes — but it means the 900-2000 Hz band decays
  faster than the reference's 0.31 s. The fix is more modes, never a
  `decay_shape` parameter. `ModalLayout.ratios()` extrapolates past the
  tabulated Bessel zeros so raising the count is a one-argument change.
* **The tension glide makes modes non-stationary**, so `ModalAnalyzer`'s
  exponential model is slightly misspecified on a glided render and reports
  frequencies shifted up and t60s stretched. Both signals get the same
  treatment, so comparison is unaffected — but a mode table read off a glided
  hit is not a parameter table.

---

## Smaller things

* **`DrumVoice.is_silent` cannot be a property with an argument.** Flagged in
  §5.4 of the spec itself. It is a method; `silent` is the zero-argument
  property.
* **ESPRIT recovers the analysis filter's own transient as a mode.** The
  bandpass rings for 124 ms in the 30-150 Hz band, which the subspace method
  cannot distinguish from a real partial — it came back as poles 20 dB *louder*
  than the fundamental with t60 around 30 ms. Fixed by skipping the filter's
  measured ringdown before fitting and rejecting poles that decay faster than
  it. Spurious modes on a six-mode test signal: 25 before, 0 after.
* **Onsets are fast rises, not level crossings.** A drum with a deliberate close
  pair beats by 6-8 dB well into its decay, so any threshold low enough to catch
  a ghost note gets crossed several times per second. Level-crossing detection
  reported every tom in a test library as a double hit.
* **Tail truncation cannot be detected from the end level.** The noise floor is
  estimated as a low quantile of the frame levels, and in a truncated file the
  quietest frames *are* the cut-off tail — so the floor estimate follows the
  truncation down and the file looks like it reached it. Detected from the
  slope at the end of the file instead: a recording that decayed into its floor
  has a flat tail, a cut one is still descending.
* **`GlideAnalyzer`'s window is 150 ms, not 250 ms.** A window reports nothing
  before its own center, and 250 ms is blind until t=125 ms — where most of the
  glide's depth happens. At 150 ms the tracked frequency lands within 2 cents of
  the engine's ground truth from the second frame on, against ~10 cents at
  250 ms. Still far short of the ~500 ms needed to resolve the 88.0/92.8 Hz
  pair, which is deliberate.
