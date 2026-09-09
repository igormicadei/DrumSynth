"""The sample library ships as metadata; most of the audio does not."""

from __future__ import annotations

import json

import pytest

from drumsynth.corpus import Corpus, default_library


def test_the_repository_ships_a_library():
    assert default_library() is not None


@pytest.fixture(scope="module")
def corpus():
    return Corpus.load()


def test_every_drum_has_samples(corpus):
    assert corpus.names
    for drum in corpus.names:
        assert corpus.samples(drum)


def test_entries_describe_files_that_may_not_be_checked_out(corpus):
    samples = corpus.samples()
    present = corpus.samples(present_only=True)

    assert len(present) < len(samples)
    assert all(sample.present() for sample in present)


def test_a_present_sample_loads_as_mono_audio_at_the_library_rate(corpus):
    present = corpus.samples(present_only=True)
    if not present:
        pytest.skip("no sample audio checked out")

    signal, rate = present[0].load()

    assert signal.ndim == 1
    assert rate == corpus.sample_rate
    assert signal.size > rate // 10


def test_samples_know_which_drum_they_are(corpus):
    for drum in corpus.names:
        assert all(sample.drum == drum for sample in corpus.samples(drum))
        assert all(drum in sample.name for sample in corpus.samples(drum))


def test_an_unknown_drum_says_what_there_is(corpus):
    with pytest.raises(KeyError, match="no drum named"):
        corpus.samples("kazoo")


def test_a_missing_library_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        Corpus.load(tmp_path / "nothing.json")


def test_a_library_can_be_read_from_anywhere(tmp_path):
    (tmp_path / "library.json").write_text(
        json.dumps(
            {
                "sr": 48000,
                "drums": {"tom": [{"path": "../samples/tom.wav", "drum": "tom", "velocity": 4}]},
            }
        )
    )

    corpus = Corpus.load(tmp_path / "library.json")

    assert corpus.sample_rate == 48000
    assert corpus.samples("tom")[0].path == (tmp_path.parent / "samples" / "tom.wav")
    assert not corpus.samples(present_only=True)
