# Findings

Where measurement contradicted the design. Each entry is something that was
built, measured and then changed — the changes are already in the code, and
this file is the record of why.

Unless stated otherwise the measurement is the tom that ships with the
repository (`data/samples/drums/toms-stereo-tom3/rr1-01-tom3-stereo-rr1.wav`,
3.5 s at 44.1 kHz), analyzed at `n_fft=2048`, `hop=1024` (151 frames), keeping
`k=128` bins, and the error is relative waveform MSE of the rendered
reconstruction.

The velocity findings (§8 onwards) cannot use that file: a velocity model needs
a whole grid of recordings and the repository ships one of them. They were
measured on a **synthetic drum** instead — 26 velocities, four strikes each,
3.5 s, brightening and lengthening with velocity, every strike with its own
phases — built by `examples/04_velocity_interpolation.py`. Numbers from a
synthetic instrument say what the machinery does, not what a real tom does; the
two are distinguished throughout.

---

## 1. Log envelopes lose to linear ones by two orders of magnitude

The first version compressed `log(|Z|/peak + floor)`, on the usual reasoning: a
partial decays exponentially, which is a straight line in log amplitude, so the
log domain is where a decay is cheap to describe.

At rank 16, phase kept per frame:

| what is factorized | relative MSE |
|---|---|
| linear `\|Z\|` | **9.2e-6** |
| `log(\|Z\|/peak + 1e-2)` | 3.4e-3 |
| `log(\|Z\|/peak + 1e-4)` | 1.7e-2 |

370x worse at the same size, and worse the lower the floor. The reason is that
the objective is squared error on the waveform, not on a log envelope. An error
of ε in the log domain is a *relative* error of ε everywhere, so it is spent
mostly where the signal is loud — which is exactly where squared error is
measured. A truncated SVD of the linear modulus, by Eckart–Young, is the best
rank-r approximation under precisely the norm being scored.

The log transform was removed. It would come back the moment the objective
becomes perceptual, and that is the point: it was not wrong, it was matched to
a different measure than the one in use.

## 2. A per-component DCT never reached a frontier

The DCT of each envelope, taken alone, was one of the original codecs. It never
appeared on a Pareto frontier in any measurement, on any of three sounds, in
either the log or the linear domain:

| codec | scalars | relative MSE |
|---|---|---|
| linear DCT, 32 coefficients | 23,575 | 1.4e-2 |
| linear SVD, rank 16 | 23,943 | **9.2e-6** |

A drum envelope starts with a step. A truncated cosine basis spends its
coefficients on that discontinuity and rings around it, and it pays that cost
once per component because nothing is shared. The SVD gets the same shapes for
a fraction of the numbers, because all the partials of one hit want nearly the
same curves. The codec was deleted rather than kept as an option — an axis
that has never won costs search time and reader attention on every run.

## 3. Any approximation of phase lands near 1e-2

Storing phase every `s` frames and interpolating is the obvious way to halve a
model, since phase is half of what `raw` holds. It does not work, and it does
not work in a specific way: quality does not degrade gradually with `s`, it
falls off a cliff between `s=1` and `s=2` and then barely moves.

All with the `raw` codec, so the only thing changing is the phase:

| phase | model scalars | relative MSE |
|---|---|---|
| every frame | 38,935 | **8.9e-6** |
| every 2nd frame | 29,260 | 1.2e-2 |
| every 4th frame | 24,487 | 3.1e-2 |
| every 8th frame | 22,036 | 5.3e-2 |
| degree-5 polynomial per component | 20,224 | 2.0e-2 |
| least-squares line per component | 19,712 | 3.7e-2 |

Between exact phase and any approximation of it there is nothing. Spending
9,500 more numbers on phase every second frame rather than a line per component
buys a factor of three in error, while keeping every frame — 19,000 more —
buys three orders of magnitude.
Once phase is wrong by a radian somewhere loud, the waveform error is
dominated by that and the details of how it got wrong stop mattering.

Two consequences. First, a fit with a tight target will always keep phase per
frame, so the compression has to come from somewhere else — which is what
motivated §4. Second, the polynomial phase codec that was written to exploit
the "phase is nearly linear" observation was deleted: at every size, something
else was better. Striding stayed, because on noisy sounds in the lossy regime
it does reach the frontier, and it costs one integer to offer.

Worth being precise about what this measures. A waveform error is brutally
phase-sensitive and human hearing is not; a reconstruction with the right
partials at the wrong phase can be indistinguishable and still measure 20 dB
down. This finding is a fact about the objective as much as about the signal.

## 4. Dividing out the bin rotation makes the block low rank, and that is the win

Each kept bin advances by a known constant phase per frame. Dividing that out
first — the component block `Z` — leaves rows that move slowly, and a struck
tonal drum turns out to be a handful of decaying complex exponentials in that
form. A complex SVD of `Z` then stores amplitude and phase together:

| model | scalars | relative MSE |
|---|---|---|
| `raw` | 38,935 | 8.9e-6 |
| `shared` rank 16 + per-frame phase | 24,071 | 9.2e-6 |
| `lowrank` rank 16 | **9,056** | 1.1e-5 |

Same error, a quarter of the numbers. Run over the whole default grid with a
1e-5 target, every point on the frontier for this tom is `lowrank`, from 248
scalars to 105,000, and the chosen model is 40.9 kB against 141.2 kB for the
best `raw` model the same search found at the same target.

Without the drift compensation this codec is useless: each row rotates at its
own rate, the block is full rank by construction, and rank 16 explains nothing.
The whole gain comes from one line in `components.to_components`.

## 5. Which codec wins is a property of the sound

`lowrank` does not win everywhere, and this is why the search exists rather
than a rule. Numbers needed to hold the same block within 1% error
(`python examples/02_compare_codecs.py`):

| | tonal hit | noise burst |
|---|---|---|
| `raw` | 16,705 | 16,705 |
| `shared` | 8,771 | **14,561** |
| `lowrank` | **3,088** | 18,528 |

On noise, `lowrank` is not merely beaten, it is worse than storing everything:
a rank-48 complex factorization costs more than the block it approximates.
Noise has no low-rank structure to find, and asking for one buys nothing at any
rank. On a synthetic cymbal wash — 120 detuned partials with independent decays over
a noise floor — `shared` with strided phase takes several frontier points, the
option §3 says is nearly always a bad deal.

Any of these three codecs, chosen in advance for all inputs, would be the wrong
one for some hit in a kit by a factor of three to five in size.

## 6. float32 storage is free at these targets

Every model quantizes to float32 (complex64 for `lowrank`) at encode time, and
the fit measures what that produces, so nothing is hidden. The cost is nil:

| | relative MSE |
|---|---|
| `raw` at float64 | 8.903628926e-6 |
| `raw` at float32 | 8.903628984e-6 |
| quantization alone, one against the other | 9.4e-14 |

Nine orders of magnitude below the tightest target anyone would set. float64
storage would double every model for nothing, and was never worth an option.

## 7. A grid axis can be silently empty

The prototype this project came from searched hop sizes as ratios of `n_fft`
(0.25, 0.375, 0.5) and skipped any that did not divide evenly. `0.375·n_fft`
never divides `n_fft`, so a third of the grid — every candidate on that axis —
was generated and dropped without a word, run after run.

Hops are now written as integer overlaps (`n_fft // overlap`), which makes the
invalid combinations unrepresentable rather than silently discarded, and
`SearchSpace.candidates()` is tested for what it produces rather than trusted
for what it looks like it produces.

## 8. Velocity interpolates, and it interpolates linearly

The point of a velocity model is the velocities nobody recorded. Holding out
every other layer of the synthetic drum and predicting it from its neighbours
(`python examples/04_velocity_interpolation.py`):

| predicting a held-out velocity | magnitude error |
|---|---|
| interpolated linearly | **0.0224** |
| interpolated in dB | 0.0258 |
| nearest recorded layer, no interpolation | 0.1328 |
| *for scale:* one recorded strike against another at the same velocity | 0.0282 |

Two things in that table. Linear beats dB, for the same reason the per-hit
codecs work on linear modulus (§1): the error being scored is squared error on
magnitude, and interpolating the quantity that is scored beats interpolating
its logarithm. And interpolation beats taking the nearest layer by six times,
which is what makes the velocity axis worth modelling rather than quantizing.

The last row is the one that decides whether any of this is good enough. A
velocity the model invents lands *closer* to the truth than a second strike of
the same drum at a velocity that was recorded. Below that, more accuracy is not
meaningful: the drum does not repeat itself that precisely.

## 9. Phase is stored best by factorizing something else

A donor stores one layer's phase. Storing that phase directly — as values, or
as a factorization of the unwrapped phase — loses badly to factorizing the
complex block and keeping only its argument. Same drum, same field, same size,
only the donor differs:

| donor | scalars | relative MSE |
|---|---|---|
| `exact` (every phase value) | 1,005,236 | 1.50e-4 |
| `lowrank` rank 16 (argument of a complex factorization) | 734,836 | **1.64e-4** |
| `phase` rank 32 (factorization of the phase itself) | 734,836 | 4.96e-2 |
| `lowrank` rank 8 | 618,772 | 5.85e-3 |
| `phase` rank 16 | 618,772 | 9.97e-2 |

At identical size, `lowrank` is three hundred times more accurate than `phase`.
The reason is what each factorization is fitted to. A least-squares fit of the
complex block is weighted by amplitude for free: a bin with no energy in it
contributes nothing, so the rank is spent where the sound is. A least-squares
fit of the *phase* weights every bin the same, so it spends its rank tracking
the unwrapped ramps of near-silent bins — which are large numbers, and which
nobody can hear.

So `lowrank` stores a complex factorization, uses only its argument, throws the
modulus away, and is still the cheapest way to hold a phase field.

## 10. Phase does not travel across velocity either

Since phase is stored per layer, an obvious economy is to store fewer of them
and let each velocity borrow from the nearest. It does not degrade — it breaks:

| donors | model scalars | error at a donor's velocity | error between donors |
|---|---|---|---|
| all 26 | 268,228 | 2.8e-3 | — |
| 13 | 152,151 | 3.2e-3 | **1.80** |
| 7 | 98,577 | 4.0e-3 | 1.97 |
| 4 | 71,790 | 5.5e-3 | 1.70 |
| 1 | 45,003 | 1.7e-3 | 1.86 |

A relative error of 1.0 is what you get by rendering silence. Everything above
that is a signal *anti*-correlated with the target — which is exactly what two
strikes with unrelated phase are. Halving the model this way does not make it
slightly worse at velocities between donors; it makes them a different signal.

This is the same wall as §3, seen along a different axis, and it is why every
layer donates by default. But note what the number does *not* say. A hit with
the right spectrum and an unrelated phase measures 1.8 and may well sound like
a perfectly good drum: the waveform objective cannot tell "wrong phase" from
"wrong sound". `between_sweep.wav` exists so a person can settle what the
metric cannot.

## 11. The mean of four strikes predicts the fifth; it just is not any of them

Each velocity has four round robins. Building the model from one of them, or
from their average, is a real choice, and both sides of it are measurable —
error against the takes each option was *not* built from:

| the layer's magnitudes come from | reproduces take 0 | predicts takes 1-3 |
|---|---|---|
| take 0 | 0.0000 | 0.0280 |
| the mean of all four | 0.0105 | 0.0105 |
| the mean of takes 1-3 | 0.0186 | — |
| take 1 | 0.0279 | — |

An average predicts an unseen strike about 1.5 times better than any single
strike does (0.0186 against 0.0279, both measured on takes they never saw). It
also stops reproducing anything that was recorded, which takes the one number
the fit can hold itself to — reconstruction — and puts a floor under it that no
codec can remove.

Neither is right in general, so the fit does not choose: `--average-takes` is
the switch, one take per layer is the default, and the report says which was
used. The default is the one whose numbers can go to zero.

## 12. A search can arrange to fool itself

The velocity search renders each candidate at a spread of velocities rather
than all of them. Donors, when sparse, are chosen evenly across the same range.
Both used `np.linspace`, so for a while every probe landed exactly on a donor —
and a model with four donors out of nine measured as well as one with nine,
while being half the size. The search dutifully chose it, and by §10 that model
is a total loss at five of its nine velocities.

Nothing was wrong with either rule; they were wrong together. Two fixes went in:
the probes are now offset to fall *between* the evenly spaced donors, and the
donor count stopped being a search axis at all — it is a setting, because the
choice it makes is about what the model is for, not about size.

The general lesson is not "be careful with linspace". It is that a measurement
which shares a construction rule with the thing it measures is not a
measurement, and that this is invisible in the result: the search reported a
smaller model with equal error, which is exactly what success looks like.

## 13. The framing that reproduces a recording is not the one that predicts a hit

Interpolation error at held-out velocities on the synthetic drum, varying only
the analysis framing:

| framing | frame spacing | interpolation error |
|---|---|---|
| n_fft 1024, hop 512 | 11.6 ms | 0.093 |
| n_fft 2048, hop 1024 | 23 ms | 0.021 |
| n_fft 4096, hop 2048 | 46 ms | **0.011** |

Finer time resolution reconstructs a recording better and predicts an unrecorded
velocity worse, because what it resolves — the exact timing texture of one
strike — is the part that does not carry from one hit to the next.

Against that, the field's own compression barely matters: at any of those
framings, `full`, `velocity` rank 6 and `separable` rank 8 predict held-out
velocities within 0.0004 of each other. The velocity axis can be compressed
hard for free. What cannot be compressed is phase, and what cannot be traded
away is time resolution's effect on what generalizes.

A fit optimizes reconstruction, so it will pick the fine framing. If what you
want is an instrument to play rather than a codec for a grid of WAVs, the
coarser framing is the better model and the report's interpolation number is
where that shows.

## 14. What a model costs to play has nothing to do with how good it is

Playing a model is an inverse FFT per frame, and an inverse FFT costs the same
whether the model filled 32 of its bins or 512:

| partials kept | per audio block | fraction of one core |
|---|---|---|
| 32 | 0.025 ms | 0.43% |
| 128 | 0.024 ms | 0.41% |
| 512 | 0.024 ms | 0.41% |

Accuracy is nearly free to play. What the partial count actually buys is a
longer trigger — 0.33 ms at 32 partials, 2.89 ms at 512 — and more memory per
sounding voice, both linear in the count.

That is the opposite of the trade a bank of oscillators makes, where every
partial is a per-sample cost for as long as it rings, and it is worth knowing
which side of it a design is on before optimizing anything: here the number to
watch is not the model's size but *when* its work happens. The whole
measurement, and what does cost something, is in [LIVE.md](LIVE.md).
