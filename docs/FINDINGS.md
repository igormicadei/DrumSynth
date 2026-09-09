# Findings

Where measurement contradicted the design. Each entry is something that was
built, measured and then changed — the changes are already in the code, and
this file is the record of why.

Unless stated otherwise the measurement is the tom that ships with the
repository (`data/samples/drums/toms-stereo-tom3/rr1-01-tom3-stereo-rr1.wav`,
3.5 s at 44.1 kHz), analyzed at `n_fft=2048`, `hop=1024` (151 frames), keeping
`k=128` bins, and the error is relative waveform MSE of the rendered
reconstruction.

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
