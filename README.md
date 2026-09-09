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

## The studio

```bash
pip install -e ".[studio]"
streamlit run streamlit_app.py
```

Pick a sample or drop in a WAV, run a search, and listen to three things: the
input, the reconstruction, and the residual. The frontier is on the same page,
so the cost of the last 10 dB is visible while you decide whether you wanted
it. Everything the pages do lives in the library; the CLI does the same job.

## What is where

```
drumsynth/spectral   the representation — audio + Candidate -> SpectralModel
drumsynth/fitting    the search        — audio -> the smallest model that fits
drumsynth/corpus     the sample library that ships with the repo
drumsynth/core       audio I/O and units
drumsynth/cli        fit, decode, inspect
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
- [docs/FORMAT.md](docs/FORMAT.md) — what a `.npz` model holds
- [docs/DATA.md](docs/DATA.md) — the sample library and how it was imported

## Version 0.2

0.2 is a rewrite. 0.1 was a modal synthesizer — a bank of resonators fitted to
a recording in five stages — and none of it survives here: no `DrumParams`, no
resonators, no scorer, no live engine. The model is now spectral and generic,
the fit is a measured search rather than a staged optimizer, and nothing from
0.1 loads. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#what-changed-in-02).
