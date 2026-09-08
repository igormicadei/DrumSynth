from __future__ import annotations

import json
from pathlib import Path

from tools.import_samples import import_library, slugify


def test_slugify_is_stable():
    assert slugify("Crash 13 (Samples)") == "crash-13-samples"
    assert slugify("Kick & Snare") == "kick-and-snare"


def test_importer_copies_only_existing_wavs_and_preserves_sfz_ranges(tmp_path):
    source = tmp_path / "source"
    sample = (
        source
        / "Samples"
        / "SMDrum Stereo (Samples)"
        / "Kik_Stereo"
        / "RR1"
        / "01_Kik Stereo_RR1.wav"
    )
    sample.parent.mkdir(parents=True)
    sample.write_bytes(b"not audio, but enough for the copier")
    mappings = source / "mappings"
    mappings.mkdir()
    (mappings / "kick.sfz").write_text(
        "<group>\n<region>\n"
        "sample=Samples\\SMDrum Stereo (Samples)\\Kik_Stereo\\RR1\\01_Kik Stereo_RR1.wav\n"
        "lovel=12\n hivel=15\n"
        "<region>\n"
        "sample=Samples\\SMDrum Stereo (Samples)\\Kik_Stereo\\RR1\\missing.wav\n"
        "lovel=16\n hivel=19\n",
        encoding="utf-8",
    )

    result = import_library(source, tmp_path / "project")
    assert result["source_counts"] == {"wav": 1, "mapped_wav": 1}
    assert len(result["missing_mappings"]) == 1

    manifest = json.loads(
        (tmp_path / "project" / "data" / "metadata" / "drums-kick.json").read_text()
    )
    entry = manifest["samples"][0]
    assert entry["drum"] == "kick"
    assert entry["velocity_low"] == 12
    assert entry["velocity_high"] == 15
    assert entry["velocity_is_exact"] is False
    assert (
        (tmp_path / "project" / "data" / "metadata" / entry["path"]).resolve().is_file()
    )
    assert entry["source_path"].startswith("Samples/")
