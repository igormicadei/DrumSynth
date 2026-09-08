# The sample library

The DrumModalSynth library, normalized into this project's layout by
[`tools/import_samples.py`](../tools/import_samples.py).

```
data/
├── file_tree.txt          generated listing of every imported file
├── metadata/
│   ├── library.json       index: drums, manifests, inventory, what is missing
│   ├── drums-<slug>.json  one manifest per physical drum
│   └── cymbals-<slug>.json
└── samples/
    ├── drums/<slug>/*.wav
    └── cymbals/<slug>/*.wav
```

**The WAVs are not in the repository.** 3.5 GB of audio, gitignored.
`data/file_tree.txt` is the generated record of exactly which files the import
produced, and it is what the integrity tests check the manifests against — so
the whole of `tests/test_data_integrity.py` runs in CI without a single WAV.

To populate `data/samples/` on a machine that has the source library:

```bash
python tools/import_samples.py C:\Projetos\DrumModalSynth\data
```

It is idempotent: re-running reproduces the same manifests byte for byte,
deletes manifests left behind by an earlier run, and reports sample directories
from an earlier naming scheme (`--prune` deletes those too).

---

## What is in it

3614 WAVs across 29 instruments. 3338 of them carry SFZ velocity metadata; the
remaining 276 are copied and inventoried but not manifested.

### Membrane drums — 1612 WAVs, 14 instruments

| drum | samples | velocity bands | round robins | controller coverage |
|---|---|---|---|---|
| `kick` | 256 | 32 | 8 | v1-v127 |
| `snare65-reg-stereo` | 272 | 34 | 8 | v1-v127 |
| `snare65-nr-stereo` | 64 | 32 | 2 | v1-v127 |
| `snare67-nr-stereo` | 64 | 32 | 2 | v1-v127 |
| `smd-snr-str-hyb1` | 64 | 32 | 2 | v1-v127 |
| `smd-snr-str-hyb2` | 64 | 32 | 2 | v1-v127 |
| `smd-snr-str-hyb3` | 64 | 32 | 2 | v1-v127 |
| `smd-rimshot-str-hyb1` | 64 | 32 | 2 | v1-v127 |
| `rimshot-stereo` | 40 | 10 | 4 | v1-v127 |
| `sidestick-stereo` | 232 | 29 | 8 | v1-v127 |
| `toms-stereo-tom1` | 100 | 25 | 4 | v1-v122 |
| `toms-stereo-tom2` | 104 | 26 | 4 | v1-v127 |
| `toms-stereo-tom3` | 104 | 26 | 4 | **v1-v111** |
| `toms-stereo-tom4` | 100 | 25 | 4 | v1-v127 |

The four toms are four separate drums, and the manifests keep them that way.
They arrive from the vendor under one `Toms_Stereo/` parent with `Tom1`..`Tom4`
beneath it, which is easy to import as a single instrument called
`toms-stereo` — and that is wrong in a way that matters here rather than
cosmetically. `f_static` and `t60` are properties of one physical drum, so a
manifest spanning four of them is a manifest the fitter must not be handed.
`tests/test_data_integrity.py::test_one_manifest_is_one_physical_drum` checks
every manifest for it.

### Cymbals — 2002 WAVs, 15 instruments

Crashes (13/15/16/17 inch, china), rides (17/20 inch plus their bells), and six
hi-hat articulations. Stored, inventoried, and **not** fitting data: a struck
cymbal cascades energy from low modes into high ones over the first few hundred
milliseconds, which a linear modal bank cannot do at any setting. See
[ARCHITECTURE.md §9](ARCHITECTURE.md#9-scope-boundary-cymbals).

Every manifest carries `family`, and the filename prefix matches it, so
selecting membrane drums is `data/metadata/drums-*.json` and nothing else.

---

## Velocity is a range, not a label

The source library maps velocity as SFZ regions, so a sample covers a *band* of
controller values rather than sitting at one. The manifests keep the band:

```json
{
  "path": "../samples/drums/toms-stereo-tom1/rr1-13-tom1-stereo-rr1.wav",
  "drum": "toms-stereo-tom1",
  "velocity": 61.0,
  "velocity_low": 59.0,
  "velocity_high": 63.0,
  "velocity_is_exact": false,
  "velocity_ranges": [[59.0, 63.0]],
  "round_robin": 1,
  "take": 1
}
```

`velocity` is the band midpoint and exists only so code written against a plain
label still works. **`velocity_is_exact` is false on every row in this library**,
which is the part that matters: a band midpoint is not a measurement, and
`VelocityCalibration` must not treat it as one. Fit against
`velocity_normalized`, which is derived from measured energy — the labels here
are doubly indirect, being controller values *and* midpoints.

Round-robin layers are alternate recordings of the same velocity band, which is
exactly what `take` means, so `round_robin` is carried through and mirrored into
`take`.

---

## What is missing, and why

Two different failure modes, both left visible rather than papered over.

### 20 SFZ regions reference files that do not exist

| mapping | references | files |
|---|---|---|
| `smdrums_sfz_tom1.sfz` | `26_Tom1_Stereo_RR1..RR4` | 4 |
| `smdrums_sfz_tom3.sfz` | `27..30_Tom3_Stereo_RR1..RR4` | 16 |

Listed in `library.json` under `missing_mappings`. This is why Tom1 stops at
v122 and Tom3 at v111 — those are their top velocity layers, and nothing was
recorded for them.

### 276 WAVs that no SFZ region references

| instrument | files |
|---|---|
| `cymbals/hi-hat-clsd-2` | 256 |
| `drums/toms-stereo-tom2` | 16 (`27..30_Tom2_Stereo_RR1..RR4`) |
| `drums/toms-stereo-tom4` | 4 (`26_Tom4_Stereo_RR1..RR4`) |

Copied and inventoried with `"mapped": false`, and deliberately absent from
every manifest.

The two lists are suggestive: Tom1 wants a layer 26 it does not have while Tom4
has an unreferenced layer 26; Tom3 wants layers 27-30 while Tom2 has exactly
those unreferenced. The source library looks cross-wired at the top of the tom
range. **No remapping was applied.** Guessing which tom a file belongs to would
put a Tom4 recording into Tom1's velocity curve, and the resulting fit would be
wrong in a way that looks fine — exactly the class of error this project's
validation exists to catch. If the vendor's intent can be confirmed, the fix
belongs in the SFZ, not in the importer.

---

## Before fitting anything

```python
from drumsynth import SampleSet

toms = SampleSet.from_manifest("data/metadata/drums-toms-stereo-tom2.json")
for issue in toms.validate():
    print(issue)
```

`validate()` loads the audio, so it needs `data/samples/` populated. It reports
clipping, second hits, truncated tails, velocity coverage gaps, a set too short
to fit the slowest decay, a drum retuned mid-session, a non-monotone velocity
calibration, and — added for this library — a set that stops short of the
controller range at either end, where the fitted curve extrapolates instead of
interpolating.

`toms-stereo-tom2` and `toms-stereo-tom4` are the complete tom sets, 104 and 100
samples across the full v1-v127 range with four round robins each. They are the
obvious place to start: the reference material behind the whole architecture is
a floor tom.
