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


def test_fit_defaults_the_output_directory_next_to_the_input(hit, capsys):
    path, _, _ = hit

    main(["fit", str(path), "--space", "quick", "--target-mse", "1e-2", "-q", "--no-plot"])

    assert (path.parent / "hit_fit" / "model.npz").exists()


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
