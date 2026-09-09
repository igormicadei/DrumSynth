# Training

Fitting one drum's `DrumParams` to its recorded samples, following the staged
procedure in [ARCHITECTURE.md](ARCHITECTURE.md) §8.

```bash
streamlit run streamlit_app.py     # → the Training page
```

One drum at a time, chosen from a dropdown. `f_static` and `t60` are properties
of a specific physical drum, so there is nothing a second drum could contribute
to a fit except a way to get them confused.

---

## The five stages

Each stage freezes what the last one settled. Nothing later is allowed to move
it.

| stage | what it produces | how |
|---|---|---|
| **1 — the drum** | `f_static[]`, `t60[]` | measured, not searched |
| **2 — the excitation** | `gain[]`, `level[]`, `contact_time`, per velocity | closed-form solve, then coordinate descent |
| **2b — the glide** | `tension.k`, `tension.tau` | grid + refinement on the band loss |
| **3 — the inspection** | a verdict, not parameters | **this is the experiment** |
| **4 — the curve** | velocity → excitation, two numbers per quantity | log-log power law |
| **5 — the refinement** | everything jointly | differential evolution |

### Stage 1 is measurement

Frequencies come from ESPRIT on a **soft** layer. That is not an arbitrary
choice: `f_static` means *the drum at rest*, and a loud hit is not at rest —
the tension feedback glides it. Measured on a synthetic drum with known
partials, the soft layer gives a **2.3 cent** median error against **47.7
cents** from the loudest hit. The same drum, the same estimator; only the
excitation differs.

Damping does **not** come from ESPRIT. Per-mode subspace damping estimates came
out 56% off on the same signal, because a mode's amplitude envelope is not a
clean exponential once neighbouring partials leak into its bin. Fitting
`DampingCurve` through *measured band decays* instead lands at 13%.

Neither of these is a search. There is no loss to descend, and no starting
point to get wrong.

### Stage 2 is a linear problem

Once the modes are frozen, the render is **linear in the gains and the noise
levels**:

```
audio = output_gain * (gains @ modal_basis + levels @ noise_basis)
```

`LinearVoiceBasis` builds the unit-gain response of every mode and every band
once, and a candidate is then a matrix product rather than a render — measured
at **170× faster** per evaluation. Gains are initialized by non-negative least
squares against the reference power spectrum and polished by coordinate
descent.

The basis is built on a **frozen tension trajectory** captured from a real
render, so it rings at the frequencies the true engine would, glide included.
That detail is not optional: with a trajectory taken from unit gains instead,
the *true* excitation scored 11.68 dB — worse than most wrong answers.
Relinearizing (solve → rebuild the basis → re-solve) brings the true excitation
to **0.07 dB**.

Only the reference layer fits every gain. The others fit **two numbers**: the
strike strength, measured from the level ratio, and the contact time. §4 says
`contact_time` is absorbed into the gains at a single fixed velocity and
reintroduced when velocity is added, and this split is exactly that.

### Stage 2b: the glide is fitted on the spectrum

The obvious way to fit a glide is to track the fundamental in both signals and
match the curves. It was built that way first, and it does not work well
enough. The tracked fundamental of a drum is short, noisy and partly masked;
the comparison has to be made in cents relative to each track's *own* asymptote
(an absolute offset would swamp the shape being fitted); and what survives all
that is a proxy an optimizer can win against while getting `k` wrong by a
factor of six — which then ruins every gain refitted afterwards, because the
partials end up in the wrong place.

Measured on a drum with a known glide, the **band loss** has a sharp minimum at
the true `k` — **1.83 dB against 4.01 dB for no glide at all** — at a point
where the pitch track returned zero. The glide moves energy between bands, so
the loss the fit is judged on can see it directly.

Two guards come with that:

* **The search is bounded in musical units, not in `k`.** `Tension.max_ratio`
  allows a doubling, and a search left to itself will take it: a huge `k` with a
  very short `tau` pins every mode at the clamp for a few milliseconds, smearing
  the attack across the coarse STFT frames. It scores well. It is not a drum.
  `MAX_RATIO_SPAN` caps the grid at three semitones.
* **A layer that cannot see a glide reports `NaN`, not zero.** "This layer says
  nothing" and "this drum does not glide" are different claims. `ratio = 1 + k *
  energy`, so a soft hit barely moves — that is the mechanism working, and
  counting it as `k = 0` would drag the estimate down and fail stage 3 on every
  drum ever fitted.

`k` is taken from the **loudest** layer carrying evidence, not from a median.
`_canonicalize` fixes the gain scale by convention on that layer, so its
amplitude is exact by definition while every other layer's is measured.

### Stage 3 is the experiment

Per-velocity fitting **always** succeeds. Six velocities fitted independently
produce six parameter sets that each match their own recording, whether or not
the velocity model is right. So the table is the evidence, not the losses:

* **Brightness must rise with velocity.** Harder hits are shorter contacts.
  Falling brightness is the mechanism backwards.
* **Amplitude must be monotone.** If it is not, the velocity calibration is
  wrong and nothing below it can be read.
* **`tension.k` must be velocity-invariant** (§8.1). The glide deepens on hard
  hits by itself, because more energy enters the bank. A fit that wants a
  different `k` per velocity is saying the energy feedback is wrong — *not* that
  `k` needs a velocity term.

That last check has one trap, and stage 3 now checks for it before passing
judgement. `k` is fitted against each layer's bank energy, which goes as
amplitude squared, so an **amplitude** off by a factor comes back as a `k` off
by its **square**. When `k` varies but `k · amplitude²` does not, the energy
feedback is fine and the velocity calibration is what to look at. Stage 3 says
so in those words rather than blaming the mechanism.

### Stages 4 and 5

Stage 4 fits a two-parameter power law per quantity in log-log. Velocity is
remapped onto `[MIN_STRENGTH, 1]` first: calibration puts the softest recorded
hit at exactly 0, which is the bottom of the *observed* range, not of the
physical one — feeding 0 to a power law makes the quietest layer silent and
drags `log(0)` into the fit, corrupting the exponent for every other layer.

Stage 5 refines the **velocity mapping** with differential evolution — six
numbers, the two-parameter laws for amplitude, contact time and noise level —
scoring the **worst** layer rather than the mean (§8.2). Each generation is
reported as it completes, which is what the progress chart draws.

It is worth being precise about what stage 5 does *not* touch: `f_static`,
`t60` and the per-mode gain shape are frozen. On a drum where stage 1 already
recovered the modes, six parameters over a dozen generations converges in
seconds and the chart is nearly flat because stage 4 already landed on the
optimum. On a hard drum, the chart is *also* nearly flat — but for the opposite
reason. The remaining error lives in the modes and the shape, and no setting of
the velocity mapping can reach it. A flat generation chart is therefore not
evidence that the fit is good; the stage 3 verdict and the per-layer losses are.

---

## What the run costs, and where it runs

Stages 1, 2 and 4 are measurement and closed-form solves; only stage 5 is a
search. A 6-layer fit at 2.5 s per hit and 30 modes takes **1-3 minutes**.

### Stage 5 evaluates a whole generation at once

Not one candidate at a time, and **never with `workers > 1`**. Handing
`differential_evolution` a process count puts each candidate in its own process,
which:

* **crashes on Windows.** The objective is a closure over the prepared bases,
  `spawn` pickles the function to send it, and a local object cannot be pickled:
  `AttributeError: Can't get local object 'JointStage.run.<locals>.objective'`.
* **is slow where it does work.** Each task ships tens of megabytes of basis
  matrices down a pipe to save a few milliseconds of arithmetic.

The fix is not a picklable objective. It is a **vectorized** one: the population
arrives as `(S, 6)`, becomes one `(S, modes) @ (modes, samples)` product and one
batched STFT per velocity layer, and returns `S` losses. Nothing is sent to a
process. On a 4-core container that alone took a 60-candidate generation from
1.22 s to 0.91 s.

### Why stage 5 used to do nothing

A run reported this, and it is worth reading as a symptom:

```
gen  1  loss 11.106  best 11.106
gen 10  loss 11.106  best 11.106
gen 20  loss 11.106  best 11.106
```

Twenty generations, four decimal places of movement. That is not a converged
search — it is a search on a basis that cannot represent the drum.

`LinearVoiceBasis` is unit-gain per mode, so the gains it is built with matter
for exactly one thing: the **tension trajectory**, which it captures from a real
render and then freezes. `ratio = 1 + k * bank_energy`. Stage 5 was building its
basis from `modal.modes`, whose gains come out of stage 1 as an ESPRIT
by-product on no particular scale. Measured on a synthetic drum: those summed to
**211** against a true **4.8**, which pinned the glide at `Tension.max_ratio` —
a full octave — for the entire hit. Every mode in the basis rang up to an octave
sharp, and no combination of the six parameters being searched could move it
back.

Two changes:

* Stage 5 linearizes around the excitation the fit actually arrived at, per
  layer, and **relinearizes once** around the answer the first pass found. On
  the same drum, stage 5 went from flat at 8.28 dB to descending from 5.07 —
  the range stage 2 was already reaching.
* Stage 1 normalizes its amplitudes to unit bank energy, so the trap cannot fire
  again from somewhere else.

Two more guards came with it, because a search that can go wrong quietly is
worse than one that fails:

* **The reported loss is measured on a real render**, not on the basis. On one
  run the basis said 4.99 dB where a real render of the same parameters said
  5.18. Letting the approximation grade its own answer is how a fit comes to
  look better than it is.
* **Stage 5 never returns something worse than it was given.** It measures the
  incoming stage-4 curve the same way, and keeps whichever is better. A
  six-parameter search scored through an approximate basis can land somewhere
  worse than it started, and handing that back as "refined" makes it a coin flip
  the user pays a minute for.

A flat generation chart is still possible and still means something specific:
stage 5 searches **six numbers**, the velocity mapping, with the modes, the
damping and the per-mode gain shape frozen by the stages before it. Flat means
the remaining error is somewhere those six numbers cannot reach. The per-layer
losses in the velocity table are where to look next.

### CUDA

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121   # your CUDA version
```

Then pick the device on the Training page, or `--device cuda` on the worker.
`auto` uses CUDA when torch reports a device and the CPU otherwise; `cuda`
**fails loudly** rather than falling back, because someone who picked the GPU
wants to know if it did not happen.

What moves to the GPU is stage 5 and only stage 5. The bases, band matrices and
reference spectrograms are uploaded once and stay in VRAM for the whole run —
for six layers of a 2.5 s hit at 30 modes that is well under a gigabyte — and
only the `(S, modes)` candidate gains cross the bus per generation, a few
kilobytes. CUDA computes in float32; measured against the float64 path on the
same candidates the largest disagreement was **1e-6 dB**, against a loss floor
of 1.1 dB.

Stages 2, 2b and 5 all take the device now — each has a batched inner loop and
a torch path. Stages 1, 3 and 4 stay on the CPU: one ESPRIT, some band-decay
regressions and a line fit, run once each on small matrices, and not something
a GPU improves.

CUDA computes the glide render in float32. The glide is a feedback loop over
more than a thousand blocks, so that is worth checking rather than assuming:
measured against the float64 path over a full 2.5-second hit, the worst
relative error is 1e-3 — about 0.009 dB, against a loss floor of 1.1 dB and
decision margins of 0.02.

#### Every stage that can be batched, is

Stage 5 was the first stage to have a batched inner loop, and for a while it
was the only one that could use a GPU. Profiling the other two showed why, and
what to do about it.

**Stage 2b — the glide.** 7.7 seconds per layer, and **89% of it inside
`ModalBank._advance`, called 77,549 times**. That is 1,723 calls per render —
one per 64-sample control block — because `ratio = 1 + k * bank_energy` makes
the frequency at block N depend on the state at block N-1. A glide render
cannot be a matrix product.

But the forty-eight grid candidates are independent *of each other*, even
though none is independent of its own history. `BatchedGlideRender` steps them
in lockstep: one state array shaped `(candidates, modes)`, one block at a time.
**7.72 s → 3.70 s.**

That is only 2.1×, and the reason is worth stating plainly: batching removes
call overhead, not arithmetic. The work is `candidates × modes × samples` =
48 × 23 × 110250 ≈ **244 million cosines and as many sines**, and it does not
shrink. Measured: the render takes the same time at a control period of 64
(1,723 blocks) and 512 (216 blocks), which says the loop is not the cost.

Two things follow. The CPU win is the overhead. And 244 million independent
transcendentals per layer is exactly what a GPU is for — so `BatchedGlideRender`
has a torch path, and before the batching there was nothing here CUDA could
have taken at all.

**Stage 2 — the excitation.** 4.9 seconds per layer, **88% inside
`SpectralTarget.distance`**, called 379 times. The coordinate descent's five
probes per parameter are independent — the same candidate with one number moved
— so they go to the loss as one batch. **4.87 s → 3.41 s**, identical result.

Here too the arithmetic is unchanged: the same number of FFTs over the same
signal. What the batching buys on the CPU is set-up; what it buys overall is
that the loss now runs through `LossBackend`, which has a CUDA path.

**Stages 1, 3 and 4 stay on the CPU and should.** Stage 1 is one ESPRIT and a
handful of band-decay regressions, run once, on small matrices — 0.4 s of a
42 s run. Stages 3 and 4 are an inspection and a line fit and do not register.

The shape of a run changes accordingly. On a real tom at 40 modes and four
layers:

| stage | before | after |
|---|---|---|
| stage 2b — the glide | 40% | 13% |
| stage 2 — excitation | 32% | 26% |
| stage 5 — refinement | 28% | 41% |

One thing that did **not** work, recorded because it looked obvious: fitting the
glide on a shorter window. The glide settles within a few hundred milliseconds,
so scoring 2.5 seconds of tail seems wasteful — but measured, a 0.6 s window
picks a *different* k than the full one. The tail is where a wrong glide shows
up as detuned partials. The window stays.

#### "I installed the CUDA wheel and the GPU sits at 0%"

Three things can be true, and the report separates them.

**Stage 5 is most of the run, so it is not that the GPU part is small.**
Measured on a 6-layer fit, 30 modes, 2.5 s hits, 20 generations:

| stage | seconds | share |
|---|---|---|
| stage 1 — modes and damping | 8.8 | 1% |
| stage 2 — excitation, pass 1 | 54.6 | 7% |
| stage 2b — the glide | 61.0 | 8% |
| stage 2 — excitation, pass 2 | 64.4 | 9% |
| stages 3 and 4 | 0.0 | 0% |
| **stage 5 — joint refinement** | **563.7** | **75%** |

Every run reports its own version of this table under **Where the time went**.

**The fit reports whether CUDA was actually used.** `torch.cuda.is_available()`
says a device exists; it does not say a tensor ever reached it. Stage 5 records
`torch.cuda.max_memory_allocated()` and the report says which of these happened:

* *Stage 5 ran on CUDA and allocated N MiB* — it worked. A device that still
  reads 0% in `nvidia-smi` is being sampled between kernels; NVML polls at about
  a second and the utilization counter is a duty cycle, not a load average.
* *The device was reported as CUDA but no tensor was ever allocated* — it ran on
  the CPU. Check that the training subprocess uses the same interpreter as the
  app (it inherits `sys.executable`) and that `torch` there is a cu-tagged wheel.
* *Stage 5 ran on the CPU* — the picker resolved to CPU, which `auto` does
  silently when torch reports no device. Pick **CUDA** explicitly and it will
  raise instead, naming the reason.

**Live telemetry while it runs.** The dashboard reads NVML directly through
`pynvml` — device name, utilization, VRAM, temperature and power, sampled once a
second:

```bash
pip install nvidia-ml-py
```

Without it the panel says so rather than showing zeros, which would be
indistinguishable from an idle GPU.

One more thing was fixed while chasing this: the population's gains were being
built with a Python loop over candidates, sixty of them times six layers, which
is 360 small numpy calls per generation that a GPU spends waiting for.
`VelocityCurve.batch_gains` does it in one broadcast.

> **Correcting an earlier claim.** An earlier version of this document said the
> GPU "would not help", on the grounds that a single evaluation is a small,
> latency-bound job where a PCIe round trip costs more than the arithmetic
> saves. That is true of a single evaluation and it was the wrong thing to
> measure. A generation is sixty candidates across six layers — 360 renders and
> 1080 STFTs, all independent, all against data that never has to move. Batched,
> it is exactly the shape CUDA is for. The unbatched design was the problem, not
> the device.

---

## When the audio is not there

The manifests are committed; the WAVs are not. A partial checkout, an
interrupted copy, or one file that failed to transfer all look the same to the
fitter, and it used to be fatal — `load_all` raised on the first missing file
and the run died with a stack trace.

It now skips them and says so:

```
103 of 104 samples named by the manifest are not on disk and were left out
(rr2-01-tom3-stereo-rr2.wav, rr3-..., rr4-..., ...).
```

Fitting 101 of 104 samples with a warning is a better answer than fitting none
with a traceback. Two things still stop a run: no usable sample at all, and —
as a loud warning rather than an error — fewer than three velocity layers, at
which point stage 3 cannot run its experiment and stage 4 is fitting
two-parameter curves through fewer points than they have parameters. §8.2 says
never fit to a single hit, and a one-layer fit produces a confident, meaningless
answer.

---

## What is verified, and what is not

The sample WAVs are gitignored — 3.5 GB — so **CI has never fitted a real
drum.** Everything below is ground-truth recovery on synthetic drums whose
parameters are known exactly, which is the only way to ask whether a fit is
*right* rather than merely low-loss.

On a small drum (8 modes, 4 velocity layers), the fit is essentially exact:

| quantity | recovered |
|---|---|
| mode frequency | 2.3 cents |
| per-layer band loss | 1.14 - 1.23 dB (floor ≈ 1.10 dB) |
| STFT loss | 0.59 dB |
| ScoreCard total | 0.740 |

On a hard one (31 modes with near-degenerate pairs, a real glide, 6 layers), it
is partial and honestly so:

| quantity | recovered | note |
|---|---|---|
| modes found | 29 of 31 | close pairs merge |
| mode frequency | 13.7 cents median, 193 cents p90 | the tail is the merged pairs |
| `t60` | 22% median | |
| per-layer band loss | 1.2 - 3.6 dB | |
| `tension.k` | within 10% on the layers that carry evidence | |

The loss floor itself is **1.10 dB**: two renders of *identical* parameters with
different noise seeds sit that far apart under the band loss. Any number near it
means the fit has run out of signal, not effort.

Three measured floors are worth keeping in mind when reading a fit:

| baseline | band loss |
|---|---|
| identical parameters, different noise seed | 1.10 dB |
| correct modes, gains fitted from scratch | 0.55 dB |
| the true excitation, on a relinearized basis | 0.07 dB |
| a bin-wise (not band-aggregated) loss on identical parameters | 11.7 dB |

That last row is why the loss aggregates power into 64 log-spaced bands. §6.5
is explicit that two realizations of the same noise process are uncorrelated;
comparing them bin by bin measures the seed, and an optimizer handed that
spends its budget chasing one noise realization.

---

## Three things the fit was being paid to get wrong

Both were found by taking a real fit apart rather than by looking at the code.

### The loss was rewarding a good impression of the noise floor

Decomposing a real tom fit's 7.6 dB band loss by how far below the reference's
peak each cell sat:

| reference level | share of the cells | share of the LOSS |
|---|---|---|
| 0 to -20 dB | 22% | 10% |
| -20 to -40 dB | 22% | 13% |
| -40 to -60 dB | 40% | **47%** |
| -60 to -80 dB | 16% | **30%** |

**77% of the loss came from cells 40 to 80 dB below the peak** — a region that
on a real recording is the room and the preamp, not the drum. And the fit knew
it: adding plain broadband hiss at -30 dB to the model *improved* the loss from
7.63 dB to 6.15 dB. Anything scored that way spends its gains buying a noise
floor, which is what that fit's 18.9 dB mode-amplitude error was.

`SpectralTarget.FLOOR_DB` was already meant to stop this, but one global
threshold cannot: a drum has more than 70 dB between its fundamental and its
own floor, so a cut at -70 dB relative to the loudest band anywhere lands *on*
the floor rather than above it. There is now a second condition — a cell also
has to be above **its own band's** quiet level, the same per-band convention
`BandDecayAnalyzer` uses. Hiss now makes the loss worse, as it must.

It costs nothing where there is no floor to reject. On a synthetic drum, two
renders of identical parameters score 0.030 dB (against 0.025 before) and a
drum detuned by a major third still sits 7.3 dB away.

### The transient bank was a click, and the fit switched it off

§4 fixes the noise bank's shape — four bands, fixed edges — and the fit was
only ever allowed to move their **levels**. Their decays came from the
architecture's defaults: 55, 28, 14 and 7 ms.

Subtract the modal bank's own render from a real tom sample and measure what is
left:

| band | default `t60` | the residual's actual `t60` | r² | level |
|---|---|---|---|---|
| 200-800 Hz | 55 ms | **1645 ms** | 0.98 | -28 dB |
| 800-2000 Hz | 28 ms | 2900 ms | 0.58 | -66 dB |
| 2000-6000 Hz | 14 ms | 4674 ms | 0.52 | -72 dB |
| 6000-15000 Hz | 7 ms | 5252 ms | 0.60 | -76 dB |

The first row is a large, clean, well-determined signal — twenty-odd resolved
partials do not cover a membrane's dense high-order content — and the band that
should have carried it was pinned at 55 ms. With a decay thirty times too short
it cannot help at any level, so the gain solve switched it off: that band's
fitted `level` came back at 5.9e-09, which is silence.

The other three rows are what a flat noise floor looks like when a straight
line is fitted to it: 66 to 76 dB down, r² around 0.55, "decays" of three to
five seconds.

So `NoiseDecayStage` measures each band's decay **on the residual** — the
reference minus the modal bank, because the noise bank's job is exactly what
the modal bank could not do — and then checks it. A band passes only with r² ≥
0.80, a level within 55 dB of the peak, and a `t60` inside a drum's range.
A band that fails keeps the architecture's default, and the run says which ones
did and why.

A measurement is a proposal, not a result. `TensionStage` makes the glide prove
itself against the band loss before it is kept, and a decay deserves the same
test: the residual is what the modes could not explain, which is not the same
thing as what this band should be doing about it. So each measured decay is
scored against the objective alongside the default and a few multiples of
itself, with the band's level re-solved and scanned for every candidate —
without that scan the comparison is between loudnesses rather than decays, and
it "preferred" a 55 ms default over a 1035 ms measurement on a drum whose band
genuinely rang for 900 ms.

**What the transients get, in full:** their `level` is fitted twice — closed
form plus coordinate descent in stage 2, then two of stage 5's six differential
evolution parameters (`level_scale`, `level_exponent`) so it tracks velocity.
Their `t60` is measured on the residual and verified against the objective.
Their band edges are fixed by §4 and are not fitted.

### The noise levels were being solved by correlation, and noise does not correlate

`_initial_levels` gave every noise band its starting level by least-squares
projection of the residual onto that band's basis row. That is exactly right
for a mode — a sine at a known frequency and phase — and structurally wrong for
a band of noise: the basis row is one realization of a random process and the
reference holds a different one, and two uncorrelated signals project onto each
other at approximately zero **whatever their levels**.

Measured on a synthetic drum whose noise band was at 0.05: the solve returned
**1.3e-08**, which is silence, and the layer sat at 4.86 dB. Matching by
**energy inside the band** instead — the quantity that survives the
realizations being different, which is the same reason §6.5 says the loss has
to aggregate into bands before comparing — returns 0.085 and **1.97 dB**.

On the real tom this is the difference between two of four transient bands
reading `5.9e-09` and all four carrying real levels.

### Stage 1 is now checked against the residual

Stage 1 is the ceiling on everything after it — no later stage moves
`f_static` — and nothing used to check whether it had found the partials that
are there.

The check is the residual: subtract the fitted model from the reference and
what remains is, by construction, everything the model cannot produce. A peak
standing clear of it at a frequency no mode covers is a partial with nowhere to
go. `ResidualModeStage` proposes those, the trainer re-fits with them, and
keeps the result only if it earns its place.

Two guards, and the first one was learned the hard way. **Proposing on the
average spectrum alone is not enough**: the noise bank's own residual is
broadband ripple, and at a 16k transform a ripple is a perfectly good local
maximum. On a synthetic drum with exactly six partials it proposed fourteen
more — at 953, 3537, 4110, 5292, 6656 and 7136 Hz, where the truth has nothing
— and every one of them lowered the loss slightly, because another resonator is
another free parameter. A real partial is an exponential: prominent at the
start of the hit and **still prominent later**. The residual is now cut into
spans and a peak has to survive most of them, which is the difference between
"there is energy here" and "something is ringing here". With that in place the
same drum proposes nothing.

**And the acceptance is measured on held-out layers.** The reference layer's
gains are all free, so adding resonators can only improve it; every other layer
inherits the same frozen shape and fits two numbers, which makes it a genuine
test of whether the new modes describe the drum or one recording. The gain also
has to clear 0.10 dB — two renders of identical parameters differ by about
0.03, so anything under that is not a measurement.

On the real tom, with the noise levels fixed, the seventeen proposed partials
are now **reverted**: the held-out layers gained 0.083 dB, which is not worth
seventeen resonators. That is the system working — the transient bank now
explains what they were chasing.

### The loss was nearly blind above the fundamental

The third one is the largest, and it explains why the two above helped less
than they should have: **the objective was not really looking at most of the
spectrum.**

The loss sums FFT power into 64 log-spaced bands, at three resolutions. A
log-spaced band from 30 Hz is about 10% wide, so below `bin_spacing / 0.1`
there is no bin to put in one — and the old bank handled that by snapping each
empty band onto its **nearest bin**. That does not add resolution. It makes N
copies of one bin and then counts that bin N times.

Measured on a real tom fit:

| resolution | bin spacing | 64 bands became | worst duplication |
|---|---|---|---|
| 256 | 172 Hz | 33 distinct bins | **11 bands all reading bin 0** |
| 1024 | 43 Hz | 46 distinct bins | 8 bands all reading bin 1 |
| 4096 | 11 Hz | 58 distinct bins | 3 bands |

Bin 0 at a 256-point transform is **DC**, which is not inside any of the eleven
bands that were reading it. A third of that resolution's loss was one number,
repeated.

Pooled over all three resolutions, the weight the loss placed on each region:

| region | before | after |
|---|---|---|
| below 210 Hz | **60.1%** | 27.1% |
| 210 - 600 Hz | 27.2% | 30.6% |
| 600 - 2000 Hz | 11.0% | **32.4%** |
| above 2 kHz | **1.8%** | 9.8% |

So the fit was near-blind everywhere except the fundamental — which is exactly
where it was leaving holes, and exactly why nothing the model could do about
them ever paid. On the real tom: the 350-500 Hz region, visibly empty in the
generated spectrum against continuous content in the sample, carried 3% of the
loss weight. Closing it with noise bought **0.07 dB of a 3.68 dB loss**. And
deleting 24 of the fit's 40 modes — everything above 1 kHz, the region that
makes a tom sound like a struck bell — cost **0.26 dB**. Sixty percent of the
bank was buying seven percent of the loss.

The fix is to **drop bands with no bin in them rather than snap them**. A
256-point FFT genuinely cannot say anything about 90 Hz; that resolution is in
the set for its time resolution, and the 4096-point one covers the low end. The
surviving bands still tile without overlap, so no part of the spectrum is
weighted twice.

The same measurements, after:

| | before | after |
|---|---|---|
| narrow noise bands closing the 350-500 Hz hole | -0.07 dB | **-0.37 dB** |
| cost of deleting every mode above 1 kHz | +0.26 dB | **+1.18 dB** |

Five times the incentive to fill the hole, four times the value on the modes
that were being wasted. The loss floors move with it: two renders of identical
parameters with different seeds sit 0.19 dB apart (was 0.03), a drum detuned by
a major third sits **9.05 dB** away (was 7.3), and broadband hiss at -30 dB
still makes the loss worse, as it must.

The report's spectrogram had the identical defect — 96 bands at 1024 points,
of which the bottom 32 were duplicates — so the bottom third of the picture
people were reading a fit from was a smear of repeated rows.

### The run was storing a pair it had not scored

`AudioIO.write` peak-normalizes by default, and a run wrote its generated hit
and its reference **separately**. The two have different crest factors, so
equal peaks mean unequal loudness: measured on a real fit, the stored pair came
out **4.4 dB apart in RMS**.

The `ScoreCard` was unharmed — it RMS-normalizes both sides — and so were the
report's charts, which go through `Comparison` and do the same. What was harmed
was everything reading the files as written, including **the report page's two
audio players**, which is how anyone actually decides whether a drum sounds
right. Those normalized each side again, separately, on playback. A run is now
written with one scale for the pair, taken from the louder peak so the ratio is
exact and nothing clips, and the players share one gain.

### Two smaller ones from the same fit

**Stage 1 was spending slots on partials that were not there.** ESPRIT reported
four estimates at 92.97, 93.94, 94.80 and 96.04 Hz — one partial, four times,
four of the thirty slots the bank has. Three of the four then came back from
the gain solve at the 1e-9 floor. Estimates closer than 40 cents are now merged
into one at the amplitude-weighted centroid (the closest genuine pair in the
test fixtures is 86 cents apart). On the real tom this took `mode_completeness`
from 0.69 to 0.98.

**Stage 3 was reading the brightness backwards.** A mode at the 1e-9 floor
reads as -180 dB, and the brightness slope is a straight line through gain
against log frequency. Three dead modes at the bottom of the range turned a
genuine **-12 dB/decade** tilt into **+70**, and the whole velocity table
reported +70 to +89 dB/decade — which stage 3 read as the excitation getting
brighter, the check it thinks it is performing. The line is now fitted through
the modes that are actually sounding.

---

## Every run is kept

A fit takes minutes and produces a drum you cannot judge in one listen. The
useful comparison is between runs — this drum at 20 modes against the same drum
at 34, the fit before the tension stage was corrected against the one after —
and that is impossible if each run overwrites the last.

So a run is a directory:

```
out/runs/toms-stereo-tom3/2026-09-09T14-22-05/
    run.json          settings, timings, warnings, the whole summary
    params.json       the DrumParams, loadable by the live synth
    score.json        the ScoreCard, per component and per layer
    audio/            one generated WAV per velocity layer
    reference/        the sample each was fitted against
```

The reference audio is stored, not just referenced, because re-rendering needs
only the parameters but the *sample* needs the 3.5 GB library, which may not be
where it was. A run directory is self-contained.

`DRUMSYNTH_RUNS` overrides the root for both the app and the worker subprocess.

**Past runs** on the Training page lists every one and opens any of them into
the same report the fit produced. **Trained drums** in the sidebar loads any
stored fit straight into the live engine, from any page and in a session that
never ran a fit — press Strike and you are hearing it.

---

## Reading a result

The page shows, in order:

1. **Live progress** — the current stage, and a per-generation loss chart once
   stage 5 starts. Stages 1-4 have no generations to show because they are not
   searches.
2. **The stage 3 verdict** with its findings, pass or fail.
3. **The velocity table** — the per-velocity excitation, which is what stage 3
   inspected.
4. **The report** — the project's own `DrumScorer`, not a second opinion
   invented for the fitter, aggregated worst-component-first across every
   velocity.
5. **Stage 5 — the search**: the generation chart, what it started and ended
   at, and the loss measured on a real render.
6. **Where the time went**: seconds per stage, and whether CUDA was used.
7. **Generated against the sample**, per velocity — both signals as audio, then
   four views of the same pair:

   | view | what it shows |
   |---|---|
   | **Waveform** | the min/max envelope of each, stacked, plus the level envelope in dB |
   | **Spectrum** | magnitude averaged over the whole hit, log frequency — peaks that line up are modes the fit found, peaks in the sample with nothing under them are modes it missed |
   | **Decay** | t60 per band, the quantity stage 1 fits its damping curve through |
   | **Spectrogram** | both signals and their difference, on ONE shared dB scale |

   Everything is level-matched first and drawn on one pair of axes. Two charts
   side by side, each auto-scaled to its own maximum, is the most common way to
   make a bad fit look fine.

8. **Load into the live synth** — the fitted drum in the running engine, so it
   can be struck and edited on the Mixer page while it rings.

A fit that scores well and **fails stage 3** is the case to be suspicious of.
It means the parameters match these particular recordings without the velocity
model being right, and it will not interpolate to velocities that were not
sampled.
