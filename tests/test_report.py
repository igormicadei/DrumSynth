"""What a run leaves on disk, and whether it can be picked up again."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from drumsynth import AudioIO, SpectralModel
from drumsynth.fitting import SearchSpace, fit, save_fit
from drumsynth.fitting.report import report_dict

SMALL = SearchSpace(
    n_ffts=(512,), overlaps=(2,), components=(16, 32), ranks=(4, 16), phase_strides=(1,)
)


@pytest.fixture(scope="module")
def written(tonal_hit, sr, tmp_path_factory):
    result = fit(tonal_hit, sr, target_mse=1e-3, space=SMALL)
    out = tmp_path_factory.mktemp("run")
    return result, save_fit(result, out, reference=tonal_hit, input_path="hit.wav"), out


def test_every_artefact_is_written(written):
    _, paths, out = written

    assert set(paths) >= {"model", "metadata", "reconstruction", "residual", "report", "evaluations"}
    for path in paths.values():
        assert path.exists() and path.stat().st_size > 0
        assert path.parent == out


def test_the_reconstruction_on_disk_is_the_model_s_own(written, sr):
    result, paths, _ = written

    audio, rate = AudioIO.read(paths["reconstruction"])

    assert rate == sr
    assert np.allclose(audio, result.model.render(), atol=1e-6)


def test_the_residual_is_what_the_model_missed(written, tonal_hit):
    result, paths, _ = written

    residual, _ = AudioIO.read(paths["residual"])

    assert np.allclose(residual, tonal_hit - result.model.render(), atol=1e-6)


def test_the_model_file_stands_on_its_own(written):
    result, paths, _ = written

    reloaded = SpectralModel.load(paths["model"])

    assert reloaded.candidate == result.candidate
    assert np.array_equal(reloaded.render(), result.model.render())


def test_the_report_records_the_target_and_the_frontier(written):
    result, paths, _ = written

    report = json.loads(paths["report"].read_text())

    assert report["input"] == "hit.wav"
    assert report["target_relative_mse"] == result.target_mse
    assert report["target_reached"] is result.target_reached
    assert len(report["frontier"]) == len(result.frontier)
    assert report["model"]["candidate"] == result.candidate.to_dict()


def test_every_candidate_appears_in_the_csv(written):
    result, paths, _ = written

    with open(paths["evaluations"], newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == len(result.evaluations)
    assert [int(r["n_scalars"]) for r in rows] == sorted(int(r["n_scalars"]) for r in rows)
    assert {r["codec"] for r in rows} <= {"raw", "shared", "lowrank"}


def test_the_report_is_json_safe(written):
    result, _, _ = written

    json.dumps(report_dict(result))


def test_a_run_can_skip_the_plot(tonal_hit, sr, tmp_path):
    result = fit(tonal_hit, sr, target_mse=1e-2, space=SMALL)

    paths = save_fit(result, tmp_path, plot=False)

    assert "plot" not in paths
    assert not (tmp_path / "frontier.png").exists()
