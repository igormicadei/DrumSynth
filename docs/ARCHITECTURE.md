# Architecture

What the model is, how a fit chooses one, and which decisions are load-bearing.

---

## 1. The representation

A hit is stored as a small set of FFT bins and the complex amplitude of each
one over time.

```
signal ──> STFT ──> select k bins ──> divide out each bin's own rotation
                                                     │
                                                component block Z
                                                (k rows x n_frames, complex)
```

**The transform.** Periodic Hann window, hop dividing `n_fft`, zero-padded by
`n_fft/2` at the front so frame 0 is centred on sample 0. The inverse is a
weighted overlap-add: analysis multiplies each frame by the window, synthesis
multiplies it again, and dividing by the summed window power is exact wherever
that sum is non-zero. An unmodified spectrogram round trips to 1e-15, which is
what makes a measured reconstruction error the *model's* error and not the
transform's. `tests/test_stft.py` pins this at five framings and four lengths.

**Bin selection.** Bins are ranked by total energy and the top `k` are kept.
Energy, not summed magnitude: the objective is a squared waveform error, and by
Parseval the bins that dominate it are the energetic ones. Nothing looks for
peaks — a windowed partial spreads over several bins and the skirts carry real
energy, so ranking by energy keeps a loud partial's skirt ahead of a quiet
partial's centre, which is the right order for this objective.

**The component block.** Bin `b` advances by `2π·b·hop/n_fft` radians per
frame, whatever the sound does. That rotation is divided out, leaving

    Z = D[bins] · exp(-i · drift)

which moves slowly: the amplitude envelope in `|Z|`, the detuning from bin
centre and any phase wander in `arg Z`. Every codec works on `Z`, and the two
low-rank codecs only work at all because of this step — undivided, each row
rotates at its own rate and the block is full rank by construction.

## 2. The three codecs

| codec | stores | scalars |
|---|---|---|
| `raw` | `\|Z\|` and `arg Z`, every frame | `kT + k·⌈T/s⌉` |
| `shared` | rank-`r` real factorization of `\|Z\|`, plus phase | `r(k+T) + k·⌈T/s⌉` |
| `lowrank` | rank-`r` complex factorization of `Z` | `2r(k+T)` |

`raw` is the baseline that exists to be beaten; a model that loses to it is not
compact enough to justify what it lost. `shared` assumes the partials of a hit
share a few temporal shapes and differ mostly in level. `lowrank` assumes the
whole hit is a few decaying complex exponentials — amplitude and phase at once,
which is why it has no phase stride: the phase is inside the factorization.

Both factorizations are truncated SVDs of the **linear** modulus or of `Z`
itself, not of a log envelope. A truncated SVD is exactly the best rank-`r`
approximation under squared error, which is the error being measured; moving to
a log domain gives that up, and measured about 100x worse at equal size
([FINDINGS §1](FINDINGS.md#1-log-envelopes-lose-to-linear-ones-by-two-orders-of-magnitude)).

Phase, where a codec stores it separately, is unwrapped and may be kept every
`s` frames and interpolated between. That option nearly always loses
([FINDINGS §3](FINDINGS.md#3-any-approximation-of-phase-lands-near-1e-2)) and
is kept because for noisy sounds in the lossy regime it does win.

## 3. The search

```
for each (n_fft, hop):            one STFT
    for each k:                   one bin selection, one component block
        for each codec:           one factorization
            for each setting:     encode -> decode -> render -> measure
```

Every candidate is quantized to float32 at encode time and rendered all the way
back to audio before it is scored, so what the search reports is the error of
the model **as the file will contain it**. There is no analytic shortcut and no
proxy metric; `tests/test_fitting.py` checks that re-encoding the winner
reproduces the winning number exactly.

What makes that affordable is that candidates share their expensive parts. A
framing is analyzed once for the thousands of candidates that use it; a bin set
gives one component block; each codec's factorization is computed once per
block and then sliced to every rank. What remains per candidate is one inverse
transform and one comparison — the part that genuinely differs. `--jobs` splits
whole groups across processes.

Candidates whose model would hold more numbers than the waveform it replaces
are never generated. Whatever else such a thing is, it is not a compression of
anything.

## 4. Choosing

Two numbers describe an evaluated candidate: how big it is and how wrong it is.
The **frontier** is everything nothing else beats on both counts, found by one
sweep over candidates sorted by size. The **choice** is the smallest model
meeting the target, ties broken by accuracy and then by frame count; when
nothing meets the target the most accurate candidate is returned and the result
says the target was missed, loudly, in `summary()`, in `report.json` and on
stderr.

That policy lives in one function (`fitting.pareto.preferred`) and is applied
both incrementally during the search and in batch afterwards, so "best so far"
and "best" cannot drift apart.

## 5. One drum, every velocity

A drum in the library is not one recording, it is a grid: 26 velocity bands for
tom 3, four round robins in each. Fitting them one at a time gives 26 unrelated
models and nothing in between them. A velocity model is one object that plays
at any velocity in the recorded range.

```
recordings ──> align onsets ──> layers[velocity][take]
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
            magnitude field                       phase donors
    M[v] ≈ Σ weights[v,r]·patterns[r]      one strike's phase per layer
    modelled, compressed, interpolated      stored whole, borrowed whole
                    │                                   │
                    └────────────► render(v) ◄──────────┘
```

**Why the split.** Magnitude varies smoothly with how hard a drum is hit —
louder, brighter, ringing longer — which is exactly the kind of thing a
factorization across the velocity axis can hold and interpolate. Phase does
not vary smoothly with anything: it is set by one strike, and crossfading two
strikes cancels rather than blends. So phase is never interpolated. It is
borrowed, whole, from the layer nearest in velocity.

**Alignment first.** Recordings arrive with different lead-ins, and two hits
whose transients are 2 ms apart cannot be compared, averaged or interpolated —
every layer is shifted so its onset sits at the same place before anything else
happens.

**One take per layer.** A model is built from one round robin of each velocity,
not from an average of them. Averaging four strike positions produces a hit
nobody played, and — because the phase can still only come from one of them —
puts an error into every number the fit reports that nothing can remove. The
other takes are not wasted: they are what generalization is measured against.

**Three codecs for the field**, on the same ladder as the per-hit ones:
`full` keeps every layer, `velocity` factorizes across velocity, `separable`
factorizes the resulting patterns again into spectral and temporal parts.
**Three for the donors**: `exact` phase, the argument of a complex low-rank
block, or a factorization of the phase itself.

### What a velocity fit measures

Three questions, and only one of them can be asked with a waveform:

| | asks | measured on |
|---|---|---|
| reconstruction | does it reproduce the recording it was built from? | waveform |
| generalization | does it predict a different strike at that velocity? | magnitude |
| interpolation | does it predict velocities held out of the fit? | magnitude |

The last two are magnitude-only because a different strike has unrelated phase;
a sample-by-sample comparison there would be measuring noise and reporting it
as failure. Reconstruction is the search target, since it is the one the model
can actually be held to.

Two departures from the per-hit search, both deliberate:

* candidates are rendered at a **spread of velocities**, not all of them, and
  the winner is then measured at every one. A full-size drum is 26 layers and
  the search is over a thousand candidates.
* the **number of donors is a setting, not a search axis**. Every layer
  donating is what reproduces every recorded velocity; fewer is a decision to
  accept that velocities far from a donor play one strike's phase under another
  strike's spectrum. That is a judgement about what the model is for, and the
  fit should not make it quietly on size grounds.

## 6. Layout

```
drumsynth/
├── core/           audio_io, constants        mono float64, units, WAV in and out
├── spectral/
│   ├── stft.py     framing, analysis, weighted overlap-add
│   ├── bins.py     which bins a model keeps
│   ├── components.py  the component block and the three codecs
│   ├── phase.py    unwrapping, striding, interpolating
│   ├── model.py    Candidate and SpectralModel
│   └── encode.py   audio + Candidate -> SpectralModel
├── fitting/
│   ├── metrics.py  relative MSE, SNR, correlation, level
│   ├── search.py   the grid, the grouping, the measured search
│   ├── pareto.py   the frontier and the one selection policy
│   └── report.py   writing a run to disk
├── instrument/
│   ├── layers.py   recordings in, aligned velocity layers out
│   ├── field.py    how magnitude changes with velocity
│   ├── donors.py   where a rendered hit borrows its phase
│   ├── model.py    InstrumentCandidate, InstrumentAnalysis, InstrumentModel
│   ├── fit.py      the search, and the three things it measures
│   └── report.py   writing a velocity fit to disk
├── streaming.py    a triggered voice, read out a block at a time
├── bench.py        what that costs: trigger, per block, polyphony, memory
├── plots.py        every figure, from arrays or from a model
├── runs.py         the run store: fits kept rather than overwritten
├── corpus.py       the shipped sample library, as data
└── cli.py          fit, fit-drum, runs, decode, play, bench, inspect
```

`spectral` never chooses and `fitting` never invents a representation. The
codecs are measured by something that has no stake in them, which is the only
reason their claims mean anything.

Three more boundaries worth stating:

* **Streaming and rendering produce the same samples.** `Voice` exists because
  a callback cannot afford a whole hit at once, not because playback is a
  different computation; the tests hold the two to bit-for-bit equality, so a
  measured error is also a heard error ([LIVE.md](LIVE.md)).

* **A model file is self-contained.** One `.npz` holds the arrays and the
  metadata; playing it back needs the package and nothing from the session that
  produced it. See [FORMAT.md](FORMAT.md).
* **The UI holds no logic.** `app_pages/` calls the same functions the CLI
  calls. Deleting it removes a way to look at a fit, not a way to run one.

## 7. Limits

* **The objective is a waveform error, and waveforms are phase-sensitive in a
  way that hearing is not.** A reconstruction with the right partials at the
  wrong phase can sound identical and measure 20 dB down. Everything the fit
  concludes about phase is a consequence of that choice of objective; a
  perceptual target would need a different metric, not a different codec.
* **Nothing here is real time.** A model renders through an inverse STFT of the
  whole signal.
* **A velocity model is one drum at one articulation.** It interpolates
  velocity and nothing else: no tuning, no damping, no strike position, and no
  round-robin variation at render time — a given velocity always plays the same
  hit. Nothing in it has a physical meaning; it is a coder for a grid of
  recordings, not a physical model of the drum that made them.
* **Phase is borrowed, so a rendered velocity carries its donor's detuning.**
  A drum bends pitch when struck harder; the magnitude field follows that, and
  the borrowed phase does not. With every layer donating, the error is confined
  to velocities between recordings. With sparse donors it is not.
* **The search is a grid.** It reports the best point it evaluated, which is
  not the best point that exists; `--space full` widens the grid when the
  answer sits against a limit.

## What changed in 0.2

0.1 was a modal synthesizer: a bank of resonators with `t60` and gain per mode,
noise bands, tension feedback, a scorer that compared descriptors extracted
from two signals, a five-stage fitter, and a live audio process driven over a
pipe. It fitted *parameters of a physical model*, and its ceiling was the model
— a linear modal bank cannot make a cymbal, so cymbals were out of scope.

0.2 fits *the signal*. The representation carries no physics, which costs the
things physics gave — no velocity axis, no tuning knob, no meaning to any
single number — and buys generality: a cymbal is no harder than a tom, it just
chooses a different codec. Nothing from 0.1 is loadable, and none of its
modules survive.
