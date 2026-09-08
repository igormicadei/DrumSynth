# The studio

A mixer for the synthesizer's parameters, with the audio running in a separate
process that never stops.

```bash
pip install -e ".[studio]"
streamlit run streamlit_app.py
```

The point is that the drum keeps ringing while you work. Strike it, and while
it is still decaying, move a `t60` knob: the rest of *that* decay changes.
Strike it again and the new hit superposes onto what is left of the old one.
Neither needs special-case code — a resonator bank with persistent state does
both for free — but both need the synthesizer to be *running*, which is why
this is a process and not a render-on-demand call.

---

## Layout

Three sets of controls, matching the three groups of parameters the model
actually has.

| tab | what it edits | parameters |
|---|---|---|
| **Master** | the whole drum | `output_gain`, `tension.k`, `tension.tau`, plus one-shot tune and damping transforms |
| **Modes** | one strip per resonant partial | `f_static`, `gain`, `t60` — 25-35 of them |
| **Transients** | one strip per contact-noise band | `f_low`, `f_high`, `level`, `t60` |

Gains and levels are edited in **dB**. A linear 0-1 fader spends nearly all its
travel in the top 6 dB, which makes the quiet partials impossible to set — and
those are the ones that decide whether a drum sounds alive.

Each mode strip shows its ratio to the fundamental (`×1.594`, `×2.136`)
alongside its frequency. Membrane modes are inharmonic and the ratio is the
physically meaningful number; 147.4 Hz on its own is not.

### S and M are not parameters

Solo and mute work **live**, on a bank that is already ringing, so you can find
out which partial a ring belongs to by silencing the rest of them mid-decay.

They are a monitoring mask applied at the mix (`ModalBank.monitor_mask`), never
a change to `gain`. Two reasons, and the second is the important one:

* Muting through `gain` would only take effect on the *next* strike, because
  gain is excitation — it is spent the instant the mode is struck.
* A sound designed while soloing would be silently wrong when saved.

The mask is not part of `DrumParams`, is never serialized, and never reaches a
fit.

---

## The audio process

```
Streamlit                                      drumsynth.live.engine
─────────                                      ─────────────────────
LiveSynth ──── {"cmd":"strike","amplitude":1} ──> stdin thread → deque
          <─── {"ev":"status","peak":0.11,...} ── stdout ← audio thread
```

Newline-delimited JSON over a pipe. Slow, and deliberately so: a pipe decouples
the two processes, so a crash in the UI cannot glitch the audio and a stalled
audio device cannot hang the UI. The traffic is tiny — a strike is 40 bytes and
a whole parameter set a few kB, neither at audio rate.

Inside the engine:

| thread | job |
|---|---|
| stdin | parse commands, append to a deque, nothing else |
| audio | drain the deque at the top of every block, apply everything, render |
| main | publish the latest telemetry frame on a timer |

**All mutation happens on the audio thread.** There is no lock anywhere near
the render call, because a lock held by the UI side is a dropout on the audio
side. `deque.append` and `popleft` are atomic under the GIL, and that is the
only synchronization this needs.

The Streamlit side keeps the handle in `st.cache_resource` — the one place
Streamlit holds a long-lived unserializable object — so a slider move reruns the
script without respawning the process.

### Running it directly

```bash
python -m drumsynth.live.engine --list-devices
python -m drumsynth.live.engine --sink device --block-size 256
echo '{"cmd":"strike","amplitude":0.8}' | python -m drumsynth.live.engine --sink null
```

```python
from drumsynth.live import LiveSynth

with LiveSynth() as synth:
    synth.strike(0.8)
    synth.load(params)     # applied mid-ring, nothing restarts
    synth.strike(0.5)      # superposes onto what is still decaying
    print(synth.status.peak, synth.status.frequency_ratio)
```

### Sinks

| sink | what it does |
|---|---|
| `device` | PortAudio output. Needs `pip install -e ".[live]"`. |
| `null` | Renders on a timer and discards. Every meter, command and live edit works; there is just no sound. |
| `wav` | Renders on the same timer and accumulates to a file. |

`null` is not a stub. It runs the entire engine, which is what makes the live
path testable in CI and usable over SSH or in a container with no sound card —
`tests/test_live.py` is 27 tests against it.

### Latency and load

Block size sets the strike latency: 256 frames is 5.8 ms at 44.1 kHz. Rendering
31 modes and 4 noise bands costs about 10% of the block period, so there is
plenty of headroom; drop to 128 or 64 for a tighter feel and watch the load
readout.

`control_period` is how many samples pass between tension updates. 64 is
inaudible (the glide stays within ~3 cents of `control_period=1`) and much
cheaper. Leave it at 64 unless you are validating.

---

## Analysis

The second page renders a hit **offline** from the current parameters — the live
engine keeps playing, untouched — and runs the same analysis chain the scorer
uses: band decays with their t60 and straightness, the f0 trajectory, and the
broadband envelope.

It is there because you cannot design a decay curve by ear alone; the reference
numbers (2.33 s at 40-130 Hz falling to 0.31 s by 2 kHz, a 2.07-semitone glide)
are what the tables are for. Metrics say where to look. Ears remain the
acceptance test.

---

## Things worth knowing

* **Mode order is identity.** Mode *i* keeps mode *i*'s ring across a live
  parameter change. Adding modes is safe — they start silent — and removing
  them takes their ring with them. Reordering the list while playing reassigns
  ring state between partials, so the table view applies on a button rather
  than on every edit.
* **Nothing autosaves.** Use *Save DrumParams JSON* in the sidebar. Solo and
  mute are deliberately not in that file.
* **The engine does not start on page load.** Spawning an audio process when a
  browser connects would respawn it on every reconnect. Press *Start audio*.
* **Changing device or block size restarts the process.** Both are fixed for the
  life of a PortAudio stream. Parameters are not — those change live.
