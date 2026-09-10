# Playing a model

What it costs to hit a drum, in the units an audio callback cares about. Every
number here comes from `python examples/05_live_cost.py` on the tom that ships
with the repository (3.5 s at 44.1 kHz), on one core of an ordinary machine,
in Python with numpy. Absolute milliseconds will differ on yours; the shapes
are what the design follows from.

The budget is what a callback has before it is late. At 44.1 kHz a 256-sample
block must be filled in **5.8 ms**, and everything below is quoted against
that.

---

## Two ways to play a hit

**Render on trigger.** `model.render()` produces the whole hit at once — 3.5 s
of audio in 8 ms, 420 times faster than real time — and the sampler reads out
of that buffer afterwards. Simple, and it puts every millisecond in the
callback that struck the drum: 8 ms is one and a half blocks late.

**Stream it** (`drumsynth.streaming.Voice`). Triggering decodes the component
block; each block after that inverse transforms only the frames it overlaps.
The total arithmetic is the same and none of it lands in one callback. The
samples are bit-for-bit those of `render()`, which is checked rather than
assumed — two paths that disagreed would mean the thing measured is not the
thing heard.

Everything below streams.

## Partials are free; the transform is not

Playing more partials costs nothing per block. The inverse FFT is over the
whole spectrum whether the model filled 32 bins of it or 512:

| partials | trigger | per block | of a core | model | per voice |
|---|---|---|---|---|---|
| 32 | 0.33 ms | 0.025 ms | 0.43% | 23 kB | 76 kB |
| 64 | 0.48 ms | 0.023 ms | 0.40% | 27 kB | 151 kB |
| 128 | 0.99 ms | 0.024 ms | 0.41% | 35 kB | 302 kB |
| 256 | 1.65 ms | 0.025 ms | 0.42% | 52 kB | 604 kB |
| 512 | 2.89 ms | 0.024 ms | 0.41% | 85 kB | 1208 kB |

What the partial count does buy is a longer trigger and more memory per
sounding voice, both linear in `k`. So quality is nearly free to *play* and
costs at the moment of striking — the opposite of the trade a bank of
oscillators makes, where every partial is a per-sample cost forever.

Transform size runs the other way:

| n_fft | trigger | per block | worst block | of a core | per voice |
|---|---|---|---|---|---|
| 512 | 2.81 ms | 0.042 ms | 0.20 ms | 0.72% | 1206 kB |
| 1024 | 1.67 ms | 0.030 ms | 0.20 ms | 0.51% | 604 kB |
| 2048 | 0.90 ms | 0.022 ms | 0.13 ms | 0.39% | 302 kB |
| 4096 | 0.50 ms | 0.021 ms | 0.22 ms | 0.36% | 152 kB |
| 8192 | 0.30 ms | 0.020 ms | 0.44 ms | 0.35% | 76 kB |

A bigger transform means fewer frames per second, and per-frame overhead —
building a spectrum, one irfft call, one windowed add — dominates the
arithmetic inside it at these sizes. The mean cost halves from 512 to 8192.
The *worst* block does not: a big transform does its work in fewer, chunkier
pieces, so the spikes get taller even as the average falls.

## The block size decides the polyphony, not the CPU

| block | budget | per block | of a core | worst block | voices |
|---|---|---|---|---|---|
| 64 | 1.45 ms | 0.014 ms | 0.95% | 0.44 ms | 3 |
| 128 | 2.90 ms | 0.018 ms | 0.62% | 0.22 ms | 13 |
| 256 | 5.80 ms | 0.022 ms | 0.38% | 0.12 ms | 48 |
| 512 | 11.6 ms | 0.047 ms | 0.40% | 0.40 ms | 29 |
| 1024 | 23.2 ms | 0.068 ms | 0.29% | 0.22 ms | 105 |

The average voice takes well under 1% of a core at every block size. The limit
is not the average — it is that one block in every `hop/block` has a frame to
transform and the rest have none. At a 64-sample block with a 1024-sample hop,
fifteen blocks do nothing and the sixteenth does everything, and that
sixteenth is what the polyphony has to fit in.

This is the one real scheduling hazard in the design, and it has ordinary
fixes: a larger block, or spreading a frame's transform across the blocks
before it is needed. The measurement says which to reach for.

## Polyphony

Voices struck at different times, mixed, at a 256-sample block:

| voices | mean | p95 | max | worst block |
|---|---|---|---|---|
| 1 | 0.032 ms | 0.094 ms | 0.207 ms | 3.6% |
| 4 | 0.120 ms | 0.284 ms | 0.469 ms | 8.1% |
| 8 | 0.201 ms | 0.300 ms | 0.434 ms | 7.5% |
| 16 | 0.393 ms | 0.519 ms | 0.822 ms | 14.2% |
| 32 | 0.840 ms | 1.412 ms | 2.345 ms | 40.4% |
| 64 | 1.842 ms | 3.042 ms | 3.645 ms | 62.8% |

Sixty-four simultaneous 3.5-second tails, in Python, use under two thirds of
one core at its worst block. Voices staggered in time do not line up their
expensive blocks, so the cost grows a little better than linearly up to 16 and
then tracks it.

For scale: a fast drummer plays perhaps 12 hits a second, and a 3.5-second
tail means about 40 voices sounding. That fits, with the caveat above about
small blocks.

## Memory

| | |
|---|---|
| model on disk | 35 kB |
| model in memory | 35 kB |
| **per sounding voice** | **302 kB** |

The decoded voice is an order of magnitude larger than the model it came from,
because triggering materializes the component block — `k × frames` complex
numbers — while the model stores a rank-16 factorization of it. Forty voices is
12 MB, which is nothing; but if it mattered, the lever is decoding each frame
as it is played rather than the whole block at trigger, which would also flatten
the trigger spike to nearly nothing. Nothing here needs that yet, so it is not
built.

## What this means

Running one model live is cheap: **under half a percent of a core**, and the
transform is not what to optimize. Two things are worth engineering around:

* **the trigger**, 0.3–3 ms depending on the model, which is a block or two of
  work landing in the callback that struck the drum. Trigger from a control
  thread, or accept one block of latency, or decode lazily.
* **frame granularity**, which makes small blocks lumpy. Use 256 samples or
  more, or spread each frame's transform across the blocks leading up to it.

Neither is a property of the model; both are properties of when its work is
scheduled. The measurements are in `drumsynth.bench`, and
`drumsynth bench <model.npz>` runs them on anything you have fitted.
