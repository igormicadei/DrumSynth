# The sample library

The DrumModalSynth library, normalized into this project's layout by
[`tools/import_samples.py`](../tools/import_samples.py).

```
data/
├── file_tree.txt          generated listing of every imported file
├── metadata/
│   ├── library.json       index: every drum, every sample, the whole inventory
│   ├── drums-<slug>.json  one manifest per physical drum
│   └── cymbals-<slug>.json
└── samples/
    ├── drums/<slug>/*.wav
    └── cymbals/<slug>/*.wav
```

**The WAVs are not in the repository.** 3.5 GB of audio, gitignored, with one
exception: `data/samples/drums/toms-stereo-tom3/rr1-01-tom3-stereo-rr1.wav` is
checked in so the tests have a real recording to fit. `data/file_tree.txt` is
the generated record of exactly which files the import produced.

To populate `data/samples/` on a machine that has the source library:

```bash
python tools/import_samples.py C:\Projetos\DrumModalSynth\data
```

It is idempotent: re-running reproduces the same manifests byte for byte,
deletes manifests left behind by an earlier run, and reports sample directories
from an earlier naming scheme (`--prune` deletes those too).

---

## Reading it

```python
from drumsynth import Corpus

corpus = Corpus.load()
corpus.names                          # every drum
corpus.samples("kick")                # every kick sample, described
corpus.samples(present_only=True)     # the ones whose audio is on this machine
signal, sr = corpus.samples("kick")[0].load()
```

A `Sample` is a description that may or may not have a file behind it —
`present()` is how you ask, and `present_only=True` is how you iterate over
what is actually here. That distinction is the whole reason the class exists:
the index describes 3,338 samples and a fresh clone has one of them.

## What is in it

3,614 WAVs across 29 instruments at 44.1 kHz. 3,338 carry SFZ velocity
metadata and appear in `library.json`; the remaining 276 are copied and
inventoried but not indexed.

### Membrane drums — 1,592 indexed samples, 14 instruments

`kick`, four toms, six snares and hybrids, two rimshots, a sidestick — up to 34
velocity bands with 2 to 8 round robins each, covering v1–v127.

The four toms are four separate drums and the manifests keep them that way.
They arrive from the vendor under one `Toms_Stereo/` parent with `Tom1`..`Tom4`
beneath it, which is easy to import as a single instrument called
`toms-stereo`. It would be wrong: these are four physical drums with different
sizes and tunings, and a fit is per hit.

### Cymbals — 1,746 indexed samples, 15 instruments

Crashes (13/15/16/17 inch, china), rides (17/20 inch plus their bells), and six
hi-hat articulations.

In 0.1 these were stored but out of scope — a modal bank cannot make a cymbal
at any setting. The spectral model has no such boundary: a cymbal is fitted the
same way a tom is, and the only thing that changes is which codec wins
([FINDINGS §5](FINDINGS.md#5-which-codec-wins-is-a-property-of-the-sound)).
They cost more, because a wash of noise has less structure to exploit than a
struck drum, and that shows up honestly in the model size rather than as a
scope note.

## Velocity

Every entry carries `velocity`, its band (`velocity_low`, `velocity_high`) and
whether the band is exact. That grid is what `drumsynth fit-drum` fits: one
model per drum, covering every velocity that was recorded and the ones between
them.

```bash
drumsynth drums                       # what is indexed, and what is on disk
drumsynth fit-drum toms-stereo-tom3   # 104 recordings -> one model
```

The four round robins of each velocity are not four velocities. One of them
anchors each layer and the rest are what the fit measures generalization
against, unless `--average-takes` says otherwise
([FINDINGS §11](FINDINGS.md#11-the-mean-of-four-strikes-predicts-the-fifth-it-just-is-not-any-of-them)).
