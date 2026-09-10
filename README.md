# DrumSynth

Fit a drum hit to a compact spectral model, and play it back.

A recorded hit becomes a handful of FFT bins and the complex amplitude of each
one over time. Nothing about the representation is fixed in advance: the fit
encodes, quantizes and **renders** every candidate, compares the audio it gets
back against the audio it was given, and keeps the smallest model that stayed
inside the quality you asked for.

```bash
pip install -e ".[dev]"
drumsynth fit data/samples/drums/toms-stereo-tom3/rr1-01-tom3-stereo-rr1.wav
```

```
representation   lowrank rank=16 n_fft=2048 hop=1024 k=256
frames           151
model            13280 numbers, 49.9 kB, 6.0x smaller than 16-bit PCM
relative MSE     3.905e-06 (54.1 dB SNR) — target 1.0e-05 reached
correlation      0.999998048
level            rms 0.999996x, peak 1.000294x
searched         111 candidates in 12.9 s
```

3.5 seconds of tom, 54 dB down, in 13,280 numbers.

## The model

One analysis, three choices, one measurement.

```
audio ──> STFT ──> keep the k most energetic bins ──> component block Z
                                                           │
                                    ┌──────────────────────┴──────────────────┐
                                    │   raw       every value, as measured    │
                                    │   shared    rank-r modulus + phase      │
                                    │   lowrank   rank-r complex, phase in it │
                                    └──────────────────────┬──────────────────┘
                                                           │
                          model ──> render ──> compare against the input
```

Bin `b` is a sinusoid at that bin's own centre frequency, so its phase advances
by a known constant every frame no matter what the drum does. Divide that
rotation out and what is left — the **component block** — moves slowly: the
envelope in the modulus, the detuning and any wander in the argument. The
codecs differ in what they assume about that block, and which one wins is a
property of the sound:

| | tonal hit | noise burst |
|---|---|---|
| `raw` | 16,705 | 16,705 |
| `shared` | 8,771 | **14,561** |
| `lowrank` | **3,088** | 18,528 |

*Numbers stored to hold the same block within 1% error —
`python examples/02_compare_codecs.py`.*

A struck tom really is a few decaying complex exponentials, so `lowrank` holds
it in a quarter of what per-frame phase costs. A cymbal wash has no such
structure, `lowrank` collapses, and the codec that keeps every phase wins. No
formula picks between those two cases from the outside, which is why this
project searches instead of deciding.

## A whole drum, every velocity

A drum in the library is a grid: 26 velocity bands for tom 3, four round robins
in each, 104 recordings of the same object hit harder and harder. Fitting them
one at a time gives 26 models with nothing between them. Fitting them together
gives one model that plays at any velocity, including the ones nobody recorded.

```bash
drumsynth drums                             # what the library holds
drumsynth fit-drum toms-stereo-tom3         # every velocity, one model
drumsynth play toms-stereo-tom3_fit/instrument.npz out.wav --velocity 96
```

The model is two things that behave completely differently:

```
magnitude field      how the spectrum changes with velocity — modelled,
                     compressed across the velocity axis, interpolated

phase donors         one strike's phase per layer — stored, and borrowed
                     whole by whatever velocity is nearest
```

Phase is not interpolated because phase does not interpolate: two strikes of
the same drum are two different sets of initial phases, and crossfading them
cancels instead of blending. Magnitude, on the other hand, moves smoothly with
how hard a drum is hit, which is exactly what makes a velocity axis modellable
at all.

```
instrument       toms-stereo-tom3
velocities       26 layers, 2.5 to 110, from 104 recordings
representation   separable rank=12 pattern=8 +lowrank rank=16 donors=all n_fft=4096 hop=2048 k=64
model            130348 numbers, 498.7 kB, 15.7x smaller than the 26 recordings it holds
reconstruction   8.962e-04 mean, 3.526e-03 at velocity 2.5 — target 1.0e-03 reached
generalization   1.313e-02 against strikes the model never saw
interpolation    1.770e-02 at held-out velocities
searched         1323 candidates in 47.5 s
```

*(A drum-shaped synthetic: the library's audio is not in the repository, so the
numbers a real tom gives will differ — `python examples/04_velocity_interpolation.py`
builds the same drum this came from.)*

A velocity fit reports three numbers, because it is answering three questions
and only one of them can be asked with a waveform comparison:

| | what it asks | measured on |
|---|---|---|
| reconstruction | does it reproduce the recording it was built from? | the waveform |
| generalization | does it predict a *different* strike at that velocity? | magnitude |
| interpolation | does it predict velocities held out of the fit? | magnitude |

The last two are magnitude-only on purpose: a different strike has unrelated
phase, so a sample-by-sample comparison would be measuring noise. Interpolation
landing near strike-to-strike variation is the bar: a velocity the model invents
should be no further from the truth than one real hit is from the next
([FINDINGS §8](docs/FINDINGS.md#8-velocity-interpolates-and-it-interpolates-linearly)).

`--average-takes` builds each velocity from all of its round robins instead of
one, which predicts the next strike better and reproduces no particular
recording exactly. `--donors N` stores fewer phase fields — smaller, and by the
waveform metric a total loss at every velocity in between, which is worth
understanding before using it
([FINDINGS §10](docs/FINDINGS.md#10-phase-does-not-travel-across-velocity-either)).

## Using it

```bash
drumsynth fit hit.wav                       # search, write the run to hit_fit/
drumsynth fit hit.wav --target-mse 1e-6     # ask for 60 dB instead of 50
drumsynth fit *.wav -o fits/ --space full   # a directory of hits, wider grid
drumsynth decode hit_fit/model.npz out.wav  # model file in, audio out
drumsynth inspect hit_fit/model.npz
```

`--space` picks how much of the grid to search: `quick` is around a hundred
candidates (seconds), `default` a few thousand (a minute or two on four cores),
`full` tens of thousands — worth it when the chosen model sits against an edge
of the default ranges. `--jobs` defaults to one process per core.

A run writes the model, what it renders to, what it missed, and how it was
chosen:

```
hit_fit/
├── model.npz           arrays + metadata; nothing else is needed to play it
├── model.json          the same metadata, without numpy
├── reconstruction.wav  what the model renders to
├── residual.wav        reference minus reconstruction — listen to this
├── report.json         the choice, the target, the Pareto frontier
├── evaluations.csv     every candidate measured, one row each
└── frontier.png        size against error
```

A drum fit writes the model, three sweeps to listen to — the recordings, the
model at those same velocities, and the model *between* them — plus the report
and a per-velocity table.

From Python:

```python
from drumsynth import AudioIO, SpectralModel, fit, save_fit

signal, sr = AudioIO.read("hit.wav")
result = fit(signal, sr, target_mse=1e-5, jobs=0)
print(result.summary())
save_fit(result, "hit_fit", reference=signal)

model = SpectralModel.load("hit_fit/model.npz")
AudioIO.write("again.wav", model.render(), model.sample_rate)
```

and for a whole drum:

```python
from drumsynth import AudioIO
from drumsynth.instrument import VelocityLayers, fit_instrument, save_instrument_fit

layers = VelocityLayers.from_corpus("toms-stereo-tom3")
result = fit_instrument(layers, target_mse=1e-4)
print(result.summary())
save_instrument_fit(result, "tom3_fit", layers=layers)

hit = result.model.render(velocity=96)   # anywhere in the recorded range
AudioIO.write("tom3-96.wav", hit, result.model.sample_rate)
```

## Runs are kept

A fit is expensive and worth going back to, so every one is written to a run
store — `runs/`, or `$DRUMSYNTH_RUNS` — instead of overwriting the last:

```
runs/instrument/toms-stereo-tom3/20260910-004530/
├── instrument.npz        the model
├── report.json           every measurement, and the frontier
├── layers.csv            per-velocity errors
├── *_sweep.wav           the recordings, the model, and between the two
└── figures/              fourteen plots of what it kept and what it missed
```

```bash
drumsynth runs                     # every drum that has been fitted, and when
drumsynth runs toms-stereo-tom3    # every training of one of them
drumsynth bench runs/.../instrument.npz
```

`-o DIR` on either fit command writes there instead, for one-off work.

## The studio

```bash
pip install -e ".[studio]"
streamlit run streamlit_app.py
```

**Instruments** lists what has been fitted and every training each one has had,
newest first, with the error and size of each. Pick one and it opens in

**Report**, which is everything measurable about a model on one page: what it
kept (bins, envelopes, decay per partial, the spectrum of the envelopes),
what it sounds like against the recording (waveform, spectrogram, spectrum,
error through the hit, error by band), how it moves with velocity, and what it
costs to play — memory, trigger time, per-block load and polyphony, measured
live. A velocity slider and a resonator count let you hear the model rebuild
itself while the figures follow.

**Hit** and **Drum** are the two fits: one WAV, or a whole drum. Everything the
pages do lives in the library; the CLI does the same job.

## What is where

```
drumsynth/spectral     the representation — audio + Candidate -> SpectralModel
drumsynth/fitting      the search        — audio -> the smallest model that fits
drumsynth/instrument   one drum across every velocity, as one model
drumsynth/streaming    playing a model a block at a time, the way a sampler does
drumsynth/bench        what that costs: trigger, per block, polyphony, memory
drumsynth/plots        every figure the report is made of
drumsynth/runs         where fits are kept afterwards
drumsynth/corpus       the sample library that ships with the repo
drumsynth/core         audio I/O and units
drumsynth/cli          fit, fit-drum, runs, decode, play, bench, inspect
```

`spectral` never chooses a candidate and `fitting` never invents a
representation: choosing is a measurement problem, and keeping it out of the
codecs is what stops a codec from being trusted rather than measured.

Python 3.10+, numpy. `soundfile` is strongly recommended (without it the stdlib
`wave` fallback reads plain PCM only); `scipy` gives polyphase resampling and
`matplotlib` draws the frontier — the fit runs, reports and saves without
either.

## Reading further

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — why the model is shaped this way
- [docs/FINDINGS.md](docs/FINDINGS.md) — where measurement contradicted the design
- [docs/FORMAT.md](docs/FORMAT.md) — what a `.npz` model holds, hit and drum
- [docs/LIVE.md](docs/LIVE.md) — what playing a model costs, measured
- [docs/DATA.md](docs/DATA.md) — the sample library and how it was imported

## Version 0.2

0.2 is a rewrite. 0.1 was a modal synthesizer — a bank of resonators fitted to
a recording in five stages — and none of it survives here: no `DrumParams`, no
resonators, no scorer, no live engine. The model is now spectral and generic,
the fit is a measured search rather than a staged optimizer, and nothing from
0.1 loads. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#what-changed-in-02).
