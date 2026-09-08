"""Checks on the committed library metadata.

The 3.5 GB of audio is not in the repository — `data/file_tree.txt` is, and it
is the generated record of exactly which files the import produced. That makes
every one of these checks runnable in CI without a single WAV, and each one
corresponds to a way the first import went wrong:

  * a manifest whose paths resolved against the wrong root (`drums-kik-stereo`)
  * a manifest left behind by an earlier run, naming a drum that no longer
    exists and pointing at a directory that was never created
  * a manifest holding four physically different drums (`toms-stereo`)
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pytest

from drumsynth import SampleLibrary, SampleSet

DATA = Path(__file__).resolve().parents[1] / "data"
METADATA = DATA / "metadata"

pytestmark = pytest.mark.skipif(
    not (METADATA / "library.json").exists(),
    reason="no imported library in this checkout",
)


def _tree_paths() -> set[str]:
    """Every path in `data/file_tree.txt`, reconstructed from the box drawing."""
    lines = (DATA / "file_tree.txt").read_text(encoding="utf-8").splitlines()
    root, stack, out = lines[0].strip(), {}, set()
    for raw in lines[1:]:
        if not raw.strip():
            continue
        match = re.match(r"^(.*?)(?:┣|┗) (.*)$", raw)
        assert match, f"unparseable tree line: {raw!r}"
        prefix, name = match.groups()
        depth = (len(prefix) - 1) // 2 + 1
        stack[depth] = name
        out.add("/".join([root] + [stack[level] for level in range(1, depth + 1)]))
    return out


def _manifests() -> list[Path]:
    return sorted(p for p in METADATA.glob("*.json") if p.name != "library.json")


def _rows(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["samples"]


@pytest.fixture(scope="module")
def tree() -> set[str]:
    return _tree_paths()


@pytest.fixture(scope="module")
def library() -> dict:
    return json.loads((METADATA / "library.json").read_text(encoding="utf-8"))


class TestInventory:
    def test_inventory_matches_the_file_tree_exactly(self, library, tree):
        inventory = {"data/" + entry["path"] for entry in library["inventory"]}
        wavs = {path for path in tree if path.endswith(".wav")}
        assert inventory == wavs

    def test_counts_agree(self, library, tree):
        wavs = [path for path in tree if path.endswith(".wav")]
        assert library["source_counts"]["wav"] == len(wavs)
        assert library["source_counts"]["mapped_wav"] == sum(
            entry["mapped"] for entry in library["inventory"]
        )

    def test_the_import_left_nothing_stale_behind(self, library):
        assert library["stale_manifests_removed"] == []
        assert library["stale_sample_dirs"] == []


class TestManifests:
    def test_every_manifest_is_listed_in_the_library(self, library):
        listed = {Path(path).name for path in library["manifests"].values()}
        assert listed == {path.name for path in _manifests()}

    @pytest.mark.parametrize("manifest", _manifests(), ids=lambda p: p.stem)
    def test_every_row_points_at_a_file_that_exists(self, manifest, tree):
        missing = []
        for row in _rows(manifest):
            resolved = (METADATA / row["path"]).resolve()
            relative = "data/" + str(resolved).split("/data/", 1)[1]
            if relative not in tree:
                missing.append(row["path"])
        assert not missing, f"{len(missing)} rows point at files that do not exist"

    @pytest.mark.parametrize("manifest", _manifests(), ids=lambda p: p.stem)
    def test_paths_are_relative_to_the_manifest(self, manifest):
        assert all(row["path"].startswith("../samples/") for row in _rows(manifest))

    @pytest.mark.parametrize("manifest", _manifests(), ids=lambda p: p.stem)
    def test_source_paths_are_portable(self, manifest):
        """No drive letters and no absolute paths — the source lives on one
        machine and the manifest has to mean something everywhere else."""
        for row in _rows(manifest):
            source = row["source_path"]
            assert not re.match(r"^[a-zA-Z]:", source), source
            assert not source.startswith("/"), source

    @pytest.mark.parametrize("manifest", _manifests(), ids=lambda p: p.stem)
    def test_one_manifest_is_one_physical_drum(self, manifest):
        """f_static and t60 belong to a specific physical drum, so a manifest
        spanning two source instruments is one the fitter must not be handed."""
        instruments = Counter()
        for row in _rows(manifest):
            parts = [
                part
                for part in re.split(r"[\\/]", row["source_path"])
                if part and not re.fullmatch(r"RR\d+", part, re.IGNORECASE)
            ]
            instruments[parts[-2] if len(parts) >= 2 else "?"] += 1
        assert len(instruments) == 1, f"spans {dict(instruments)}"

    @pytest.mark.parametrize("manifest", _manifests(), ids=lambda p: p.stem)
    def test_velocity_fields_are_consistent(self, manifest):
        for row in _rows(manifest):
            low, high = row["velocity_low"], row["velocity_high"]
            assert 1 <= low <= high <= 127
            assert row["velocity"] == pytest.approx((low + high) / 2.0)
            # These are SFZ ranges, not recorded MIDI labels. Marking them exact
            # would let the calibration treat a band midpoint as a measurement.
            assert row["velocity_is_exact"] is False
            assert [low, high] in [list(item) for item in row["velocity_ranges"]]


class TestLoading:
    def test_the_library_loads(self, library):
        loaded = SampleLibrary.load_manifest(METADATA / "library.json")
        assert len(loaded) == len(library["drums"])
        assert loaded.drums() == sorted(library["drums"])

    def test_each_manifest_loads_as_a_single_drum_set(self):
        for manifest in _manifests():
            sample_set = SampleSet.from_manifest(manifest)
            assert len(sample_set) > 0
            assert {sample.drum for sample in sample_set} == {sample_set.drum}

    def test_membrane_drums_are_separable_from_cymbals(self):
        """Cymbals are stored for later. They break the modal bank at the
        physics level and must never be pulled into membrane fitting by
        accident."""
        families = {
            manifest.stem: {row["family"] for row in _rows(manifest)}
            for manifest in _manifests()
        }
        assert all(len(value) == 1 for value in families.values())
        for name, (family,) in ((k, tuple(v)) for k, v in families.items()):
            assert name.startswith(family + "-"), (name, family)

    def test_the_toms_are_four_separate_drums(self):
        """The reference material for this whole project is a floor tom, so
        this is the set that matters most."""
        toms = sorted(METADATA.glob("drums-toms-stereo-tom*.json"))
        assert len(toms) == 4
        for manifest in toms:
            sample_set = SampleSet.from_manifest(manifest)
            assert 90 <= len(sample_set) <= 120
            low, high = sample_set.velocity_range()
            assert low < 10 and high > 100
