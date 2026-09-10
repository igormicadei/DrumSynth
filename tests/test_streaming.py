"""Playing a model block by block has to give the same samples as rendering it."""

from __future__ import annotations

import numpy as np
import pytest

from drumsynth.spectral import Candidate, encode
from drumsynth.streaming import Voice, stream, voice_for

CANDIDATES = [
    Candidate(512, 256, 32, "raw"),
    Candidate(512, 128, 32, "raw", phase_stride=4),
    Candidate(1024, 512, 48, "shared", 6),
    Candidate(1024, 512, 48, "lowrank", 8),
]


@pytest.mark.parametrize("candidate", CANDIDATES, ids=lambda c: c.label())
@pytest.mark.parametrize("block", [1, 64, 257, 4096])
def test_streaming_matches_rendering_exactly(candidate, block, tonal_hit, sr):
    model = encode(tonal_hit, sr, candidate)

    assert np.allclose(stream(model, block=block), model.render(), atol=1e-12)


def test_a_voice_reports_where_it_is(tonal_hit, sr):
    model = encode(tonal_hit, sr, CANDIDATES[0])
    voice = voice_for(model)

    assert voice.position == 0
    assert not voice.finished

    voice.read(1000)
    assert voice.position == 1000

    voice.read(model.n_samples)
    assert voice.finished


def test_reading_past_the_end_gives_silence(tonal_hit, sr):
    model = encode(tonal_hit, sr, CANDIDATES[0])
    voice = voice_for(model)
    voice.read(model.n_samples)

    assert not np.any(voice.read(512))


def test_the_rest_of_a_hit_can_be_taken_in_one_call(tonal_hit, sr):
    model = encode(tonal_hit, sr, CANDIDATES[3])
    voice = voice_for(model)

    first = voice.read(700)
    rest = voice.render_rest()

    assert np.allclose(np.concatenate([first, rest]), model.render(), atol=1e-12)


def test_an_instrument_streams_at_any_velocity(velocity_layers):
    from drumsynth.instrument import InstrumentCandidate, InstrumentModel
    from drumsynth.spectral.stft import StftSpec

    spec = StftSpec(512, 256)
    bins = velocity_layers.select_bins(spec, 32)
    magnitudes, blocks = velocity_layers.analyse(spec, bins)
    model = InstrumentModel.build(
        "drum",
        InstrumentCandidate(512, 256, 32, "velocity", 3, 0, "lowrank", 8, 0),
        sample_rate=velocity_layers.sample_rate,
        n_samples=velocity_layers.n_samples,
        velocities=velocity_layers.velocities,
        bins=bins,
        magnitudes=magnitudes,
        blocks=blocks,
    )

    for velocity in (velocity_layers.velocities[0], 42.0, velocity_layers.velocities[-1]):
        assert np.allclose(stream(model, velocity, block=256), model.render(velocity), atol=1e-12)


def test_an_instrument_will_not_be_triggered_without_a_velocity(velocity_layers, tmp_path):
    from drumsynth.instrument import InstrumentCandidate, InstrumentModel
    from drumsynth.spectral.stft import StftSpec

    spec = StftSpec(512, 256)
    bins = velocity_layers.select_bins(spec, 16)
    magnitudes, blocks = velocity_layers.analyse(spec, bins)
    model = InstrumentModel.build(
        "drum",
        InstrumentCandidate(512, 256, 16, "full", 0, 0, "exact", 0, 0),
        sample_rate=velocity_layers.sample_rate,
        n_samples=velocity_layers.n_samples,
        velocities=velocity_layers.velocities,
        bins=bins,
        magnitudes=magnitudes,
        blocks=blocks,
    )

    with pytest.raises(ValueError, match="needs a velocity"):
        voice_for(model)


def test_a_voice_holds_only_what_it_still_needs(tonal_hit, sr):
    """The accumulator is dropped behind the read cursor, not kept for the hit."""
    model = encode(tonal_hit, sr, Candidate(1024, 512, 48, "lowrank", 8))
    voice = voice_for(model)

    voice.read(model.n_samples // 2)

    assert isinstance(voice, Voice)
    assert voice._buffer.size <= 4 * model.candidate.n_fft
