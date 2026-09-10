# The model files

Two formats: one hit, and one drum across every velocity. Both are a single
`.npz` holding the arrays and the metadata that describes them.

---

# A single hit

One `.npz` per model. It holds the arrays and the metadata that describes them,
so playing a model back needs this package and nothing else — no sidecar JSON,
no generated decoder script, no memory of the session that produced it.

```python
from drumsynth import AudioIO, SpectralModel

model = SpectralModel.load("hit_fit/model.npz")
AudioIO.write("hit.wav", model.render(), model.sample_rate)
```

or `drumsynth decode hit_fit/model.npz hit.wav`.

---

## What is inside

`np.savez_compressed` with these entries:

| key | dtype | shape | meaning |
|---|---|---|---|
| `meta` | str | scalar | the JSON below |
| `bins` | int32 | (k,) | which FFT bins the model keeps, ascending |
| `codec__*` | codec's | codec's | everything the codec stored |

The `codec__` prefix is how a loader knows which arrays belong to the codec
without knowing what the codec is. What sits under it depends on `codec`:

| codec | arrays |
|---|---|
| `raw` | `magnitude` (float32, k×T), `phase_indices` (int32), `phase_values` (float32, k×kept) |
| `shared` | `weights` (float32, k×r), `basis` (float32, r×T), `phase_indices`, `phase_values` |
| `lowrank` | `weights` (complex64, k×r), `basis` (complex64, r×T) |

Everything is float32 or complex64. That quantization happens at encode time,
before the fit measures anything, so a reported error is the error of the file
([FINDINGS §6](FINDINGS.md#6-float32-storage-is-free-at-these-targets)).

## The metadata

```json
{
  "format": "drumsynth.spectral/1",
  "candidate": {
    "n_fft": 2048, "hop": 1024, "n_components": 256,
    "codec": "lowrank", "codec_param": 16, "phase_stride": 1
  },
  "sample_rate": 44100,
  "n_samples": 154350,
  "n_frames": 151,
  "duration": 3.5,
  "n_components": 256,
  "n_scalars": 13280,
  "arrays": ["bins", "codec__basis", "codec__weights"]
}
```

`candidate` is everything needed to reconstruct: the framing, how many bins,
which codec at which setting. `n_samples` is the length to render back to —
the transform's own length is longer, and trimming to the original is part of
playback, not a convenience.

A run also writes `model.json`, which is this same block on its own for reading
without numpy. It is a copy, not a dependency: deleting it changes nothing.

## Versioning

`format` is checked on load and a mismatch is refused with both versions named.
There is no migration path and there will not be one: models are cheap to
refit, and silently reinterpreting an old layout is a worse failure than
refusing it.

`drumsynth.spectral/1` is the layout above, introduced in 0.2. Files from 0.1
are not models in this sense at all — that version stored synthesizer
parameters — and nothing reads them.

## Size

`model.n_bytes()` is the compressed size of exactly this file, measured by
writing it to memory, and `compression_ratio()` compares that against 16-bit
PCM of the same audio. `n_scalars` counts stored numbers instead, with a
complex value counting as two; it is what the search ranks candidates by,
because it does not depend on how well a particular block happens to zip.

---

# A whole drum

`drumsynth.instrument/1`, written by `drumsynth fit-drum`. One file holds every
velocity of one drum and the velocities between them.

```python
from drumsynth import AudioIO
from drumsynth.instrument import InstrumentModel

model = InstrumentModel.load("tom3_fit/instrument.npz")
AudioIO.write("tom3-96.wav", model.render(velocity=96), model.sample_rate)
```

or `drumsynth play tom3_fit/instrument.npz out.wav --velocity 96`.

## What is inside

| key | dtype | shape | meaning |
|---|---|---|---|
| `meta` | str | scalar | the JSON below |
| `bins` | int32 | (k,) | which FFT bins the model keeps |
| `velocities` | float32 | (V,) | the velocities that were recorded |
| `donor_velocities` | float32 | (D,) | which of them donate phase |
| `field__*` | codec's | codec's | the magnitude field |
| `donor<i>__*` | codec's | codec's | one phase field per donor |

The magnitude field always decodes to `weights (V, r)` and `patterns
(r, bins, frames)`, whatever stored it:

| field codec | arrays |
|---|---|
| `full` | `patterns` (float32, V×k×T); the weights are the identity |
| `velocity` | `weights` (float32, V×r), `patterns` (float32, r×k×T) |
| `separable` | `weights` (float32, V×r), `pattern_weights` (r×k×p), `pattern_basis` (r×p×T) |

and each donor stores a phase field:

| donor codec | arrays |
|---|---|
| `exact` | `phase` (float32, k×T) |
| `lowrank` | `weights` (complex64, k×r), `basis` (complex64, r×T) — the argument of the product is the phase |
| `phase` | `weights` (float32, k×r), `basis` (float32, r×T) |

## The metadata

```json
{
  "format": "drumsynth.instrument/1",
  "name": "toms-stereo-tom3",
  "candidate": {
    "n_fft": 2048, "hop": 1024, "n_components": 256,
    "field_codec": "separable", "field_rank": 6, "pattern_rank": 16,
    "donor_codec": "lowrank", "donor_rank": 16, "n_donors": 0
  },
  "sample_rate": 44100, "n_samples": 154350, "n_frames": 151,
  "n_layers": 26, "velocity_range": [2.5, 109.5], "n_donors": 26
}
```

`n_donors` in `candidate` is the *setting* (0 meaning every layer); the
top-level `n_donors` is how many there turned out to be.

Rendering outside `velocity_range` clamps to the nearest end rather than
extrapolating. The model knows what the drum did between 2.5 and 109.5 because
it was shown that; what it would do at 127 is a guess, and it does not make it.
