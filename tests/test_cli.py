"""The command line, end to end: fit a file, then play the model back from disk."""

from __future__ import annotations

import numpy as np
import pytest

from drumsynth import AudioIO
from drumsynth.cli import main
from drumsynth.fitting.metrics import relative_mse


@pytest.fixture
def hit(tonal_hit, sr, tmp_path):
    return AudioIO.write(tmp_path / "hit.wav", tonal_hit, sr), tonal_hit, sr


def test_fit_writes_a_run_and_decode_plays_it_back(hit, tmp_path, capsys):
    path, signal, sr = hit
    out = tmp_path / "run"

    assert main(["fit", str(path), "-o", str(out), "--space", "quick", "--target-mse", "1e-3", "-q"]) == 0

    run = out / "hit"
    assert (run / "model.npz").exists()
    assert "relative MSE" in capsys.readouterr().out

    assert main(["decode", str(run / "model.npz"), str(tmp_path / "again.wav")]) == 0

    decoded, rate = AudioIO.read(tmp_path / "again.wav")
    assert rate == sr
    assert relative_mse(signal, decoded) <= 1e-3


def test_fit_keeps_the_run_in_the_store_by_default(hit, tmp_path, monkeypatch):
    path, _, _ = hit
    monkeypatch.setenv("DRUMSYNTH_RUNS", str(tmp_path / "runs"))

    main(["fit", str(path), "--space", "quick", "--target-mse", "1e-2", "-q", "--no-plot"])

    from drumsynth.runs import RunStore

    kept = RunStore.default().runs("hit", kind="hit")
    assert len(kept) == 1
    assert kept[0].model_path.exists()


def test_fit_reports_a_target_it_could_not_reach(hit, capsys):
    path, _, _ = hit

    main(["fit", str(path), "--space", "quick", "--target-mse", "1e-30", "-q", "--no-plot"])

    assert "not reached" in capsys.readouterr().err.lower()


def test_progress_is_printed_unless_asked_not_to(hit, capsys):
    path, _, _ = hit

    main(["fit", str(path), "--space", "quick", "--target-mse", "1e-2", "--no-plot"])

    assert "best:" in capsys.readouterr().out


def test_inspect_describes_the_model(hit, tmp_path, capsys):
    path, _, _ = hit
    main(["fit", str(path), "-o", str(tmp_path / "run"), "--space", "quick", "-q", "--no-plot"])

    assert main(["inspect", str(tmp_path / "run" / "hit" / "model.npz")]) == 0

    printed = capsys.readouterr().out
    assert "representation" in printed
    assert "Hz" in printed


def test_decode_defaults_its_output_beside_the_model(hit, tmp_path):
    path, _, _ = hit
    main(["fit", str(path), "-o", str(tmp_path / "run"), "--space", "quick", "-q", "--no-plot"])

    main(["decode", str(tmp_path / "run" / "hit" / "model.npz")])

    assert (tmp_path / "run" / "hit" / "model.wav").exists()


def test_fitting_resamples_when_asked(hit, tmp_path):
    path, _, sr = hit

    main(
        [
            "fit", str(path), "-o", str(tmp_path / "run"), "--space", "quick",
            "--sample-rate", str(sr // 2), "-q", "--no-plot",
        ]
    )

    _, rate = AudioIO.read(tmp_path / "run" / "hit" / "reconstruction.wav")
    assert rate == sr // 2


def test_several_inputs_get_one_directory_each(hit, tmp_path):
    path, signal, sr = hit
    second = AudioIO.write(tmp_path / "other.wav", np.flip(signal), sr)

    main(["fit", str(path), str(second), "-o", str(tmp_path / "run"), "--space", "quick", "-q", "--no-plot"])

    assert (tmp_path / "run" / "hit" / "model.npz").exists()
    assert (tmp_path / "run" / "other" / "model.npz").exists()


# -- a whole drum -------------------------------------------------------------


def test_drums_lists_the_library(capsys):
    assert main(["drums"]) == 0

    printed = capsys.readouterr().out
    assert "toms-stereo-tom3" in printed
    assert "on disk" in printed


def test_fitting_a_drum_writes_an_instrument_that_plays_back(tmp_path, capsys):
    out = tmp_path / "run"

    assert (
        main(
            [
                "fit-drum", "toms-stereo-tom3", "-o", str(out),
                "--space", "quick", "--target-mse", "1e-2", "-q", "--no-plot",
            ]
        )
        == 0
    )

    model = out / "instrument.npz"
    assert model.exists()
    assert "reconstruction" in capsys.readouterr().out

    assert main(["play", str(model), str(tmp_path / "hit.wav"), "--velocity", "2.5"]) == 0
    played, rate = AudioIO.read(tmp_path / "hit.wav")
    assert rate == 44100
    assert played.size > rate


def test_inspecting_an_instrument_describes_its_velocities(tmp_path, capsys):
    out = tmp_path / "run"
    main(
        [
            "fit-drum", "toms-stereo-tom3", "-o", str(out),
            "--space", "quick", "--target-mse", "1e-2", "-q", "--no-plot",
        ]
    )
    capsys.readouterr()

    assert main(["inspect", str(out / "instrument.npz")]) == 0

    printed = capsys.readouterr().out
    assert "instrument" in printed
    assert "donating phase" in printed


def test_playing_outside_the_recorded_range_says_so(tmp_path, capsys):
    out = tmp_path / "run"
    main(
        [
            "fit-drum", "toms-stereo-tom3", "-o", str(out),
            "--space", "quick", "--target-mse", "1e-2", "-q", "--no-plot",
        ]
    )
    capsys.readouterr()

    main(["play", str(out / "instrument.npz"), str(tmp_path / "x.wav"), "--velocity", "127"])

    assert "outside the recorded range" in capsys.readouterr().err


# -- runs and benchmarks ------------------------------------------------------


def test_runs_lists_what_was_kept(hit, tmp_path, monkeypatch, capsys):
    path, _, _ = hit
    monkeypatch.setenv("DRUMSYNTH_RUNS", str(tmp_path / "runs"))
    main(["fit", str(path), "--space", "quick", "--target-mse", "1e-2", "-q", "--no-plot"])
    capsys.readouterr()

    assert main(["runs", "--kind", "hit"]) == 0

    printed = capsys.readouterr().out
    assert "hit" in printed
    assert "representation" in printed


def test_runs_says_so_when_there_are_none(tmp_path, capsys):
    assert main(["runs", "--runs", str(tmp_path / "empty")]) == 0

    assert "no instrument runs" in capsys.readouterr().out


def test_bench_measures_a_saved_model(hit, tmp_path, monkeypatch, capsys):
    path, _, _ = hit
    monkeypatch.setenv("DRUMSYNTH_RUNS", str(tmp_path / "runs"))
    main(["fit", str(path), "--space", "quick", "--target-mse", "1e-2", "-q", "--no-plot"])
    from drumsynth.runs import RunStore

    model = RunStore.default().runs("hit", kind="hit")[0].model_path
    capsys.readouterr()

    assert main(["bench", str(model), "--block", "128", "--voices", "1", "4"]) == 0

    printed = capsys.readouterr().out
    assert "per block" in printed
    assert "voices" in printed


def test_bench_reads_a_velocity_model_too(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DRUMSYNTH_RUNS", str(tmp_path / "runs"))
    main(["fit-drum", "toms-stereo-tom3", "--space", "quick", "-q", "--no-plot"])
    from drumsynth.runs import RunStore

    model = RunStore.default().runs("toms-stereo-tom3")[0].model_path
    capsys.readouterr()

    assert main(["bench", str(model), "--voices"]) == 0
    assert "trigger" in capsys.readouterr().out
