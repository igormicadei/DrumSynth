"""The codecs: what each one keeps, what it throws away, and what it costs."""

from __future__ import annotations

import numpy as np
import pytest

from drumsynth.spectral import phase as phase_codec
from drumsynth.spectral.bins import select_bins
from drumsynth.spectral.components import (
    CODECS,
    LowRankComponents,
    RawComponents,
    SharedComponents,
    codec_for,
    drift,
    from_components,
    to_components,
)
from drumsynth.spectral.stft import StftSpec, analyze

SPEC = StftSpec(512, 128)


@pytest.fixture
def analysis(tonal_hit):
    spectrogram = analyze(tonal_hit, SPEC)
    bins = select_bins(spectrogram, 32)
    return spectrogram, bins, to_components(spectrogram, bins, SPEC)


@pytest.fixture
def block(analysis):
    return analysis[2]


@pytest.fixture
def noisy_block(noisy_hit):
    spectrogram = analyze(noisy_hit, SPEC)
    bins = select_bins(spectrogram, 32)
    return to_components(spectrogram, bins, SPEC)


def relative_error(reference, estimate):
    return float(
        np.sum(np.abs(reference - estimate) ** 2) / np.sum(np.abs(reference) ** 2)
    )


def test_removing_and_restoring_the_drift_is_lossless(analysis):
    spectrogram, bins, components = analysis

    restored = from_components(components, bins, SPEC)

    assert np.allclose(restored, spectrogram[bins], atol=1e-12)


def test_drift_is_the_rotation_of_the_bin_itself():
    bins = np.array([0, 1, 4], dtype=np.int32)
    spec = StftSpec(8, 4)

    advance = drift(bins, spec, 3)[:, 1]

    assert advance[0] == pytest.approx(0.0)
    assert advance[1] == pytest.approx(np.pi)
    assert advance[2] == pytest.approx(4 * np.pi)


def test_raw_keeps_everything_but_the_float32(block):
    components = block

    codec = RawComponents
    decoded = codec.decode(codec.encode(codec.prepare(components), 0, 1), components.shape[1])

    assert relative_error(components, decoded) < 1e-12


@pytest.mark.parametrize("codec", [SharedComponents, LowRankComponents])
def test_full_rank_reproduces_the_block(codec, block):
    components = block
    rank = min(components.shape)

    decoded = codec.decode(codec.encode(codec.prepare(components), rank, 1), components.shape[1])

    assert relative_error(components, decoded) < 1e-10


@pytest.mark.parametrize("codec", [SharedComponents, LowRankComponents])
def test_more_rank_never_measures_worse(codec, block):
    components = block
    prepared = codec.prepare(components)

    errors = [
        relative_error(
            components, codec.decode(codec.encode(prepared, rank, 1), components.shape[1])
        )
        for rank in (1, 2, 4, 8)
    ]

    assert errors == sorted(errors, reverse=True)


def cheapest_within(codec, components, target: float) -> int:
    """Fewest scalars this codec needs to hold the block within `target`."""
    k, frames = components.shape
    prepared = codec.prepare(components)
    costs = [
        codec.n_scalars(k, frames, rank, 1)
        for rank in range(1, min(k, frames) + 1)
        if relative_error(components, codec.decode(codec.encode(prepared, rank, 1), frames))
        <= target
    ]
    return min(costs, default=10**9)


def test_which_codec_wins_depends_on_the_sound(block, noisy_block):
    """The premise of the whole search, as a measurement.

    Low rank is the cheap way to hold a struck tonal hit — its partials really
    are a few decaying complex exponentials — and it is the expensive way to
    hold a noise burst, which has no low-rank structure to find.
    """
    target = 1e-3

    assert cheapest_within(LowRankComponents, block, target) < cheapest_within(
        SharedComponents, block, target
    )
    assert cheapest_within(SharedComponents, noisy_block, target) < cheapest_within(
        LowRankComponents, noisy_block, target
    )


@pytest.mark.parametrize("name", sorted(CODECS))
def test_scalar_count_matches_the_arrays_actually_stored(name, block):
    components = block
    codec = codec_for(name)
    k, frames = components.shape
    stride = 2 if codec.uses_phase else 1

    arrays = codec.encode(codec.prepare(components), 8, stride)
    stored = sum(a.size * (2 if np.iscomplexobj(a) else 1) for a in arrays.values())

    assert codec.n_scalars(k, frames, 8, stride) == stored


def test_unknown_codec_names_itself():
    with pytest.raises(ValueError, match="unknown component codec"):
        codec_for("nope")


def test_phase_stride_keeps_first_and_last_frame():
    indices = phase_codec.stored_frames(10, 4)

    assert indices[0] == 0
    assert indices[-1] == 9
    assert list(indices) == [0, 4, 8, 9]


def test_phase_stride_of_one_is_exact(block):
    components = block
    unwrapped = phase_codec.unwrap(np.angle(components))

    indices, values = phase_codec.sample(unwrapped, 1)
    decoded = phase_codec.decode(indices, values, components.shape[1])

    assert np.allclose(decoded, unwrapped, atol=1e-5)


def test_phase_stride_rejects_zero():
    with pytest.raises(ValueError, match="at least 1"):
        phase_codec.stored_frames(10, 0)
