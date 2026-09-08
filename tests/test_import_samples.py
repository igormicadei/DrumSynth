"""The SFZ importer, and the two mistakes it is now shaped to prevent.

The real library is 3.5 GB and lives outside the repository, so these tests
build miniature source trees with the same folder shapes. Every assertion here
corresponds to something that was actually wrong in the first import run.
"""

from __future__ import annotations

import json
from pathlib import Path

from tools.import_samples import import_library, render_file_tree, slugify


def _write_sample(root: Path, *parts: str) -> Path:
    path = root.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not audio, but enough for the copier")
    return path


def _write_sfz(root: Path, name: str, regions: list[tuple[str, int, int]]) -> Path:
    lines = []
    for sample, low, high in regions:
        lines += ["<region>", f"sample={sample}", f"lovel={low}", f"hivel={high}"]
    mappings = root / "mappings"
    mappings.mkdir(parents=True, exist_ok=True)
    path = mappings / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class TestSlugify:
    def test_is_stable(self):
        assert slugify("Crash 13 (Samples)") == "crash-13-samples"
        assert slugify("Kick & Snare") == "kick-and-snare"

    def test_never_returns_empty(self):
        assert slugify("---") == "unknown"
        assert slugify("") == "unknown"


class TestBasicImport:
    def test_copies_only_existing_wavs_and_preserves_sfz_ranges(self, tmp_path):
        source = tmp_path / "source"
        _write_sample(
            source, "Samples", "SMDrum Stereo (Samples)", "Kik_Stereo", "RR1",
            "01_Kik Stereo_RR1.wav",
        )
        _write_sfz(
            source, "kick.sfz",
            [
                ("Samples\\SMDrum Stereo (Samples)\\Kik_Stereo\\RR1\\01_Kik Stereo_RR1.wav", 12, 15),
                ("Samples\\SMDrum Stereo (Samples)\\Kik_Stereo\\RR1\\missing.wav", 16, 19),
            ],
        )

        result = import_library(source, tmp_path / "project")
        assert result["source_counts"] == {"wav": 1, "mapped_wav": 1}
        assert len(result["missing_mappings"]) == 1

        metadata = tmp_path / "project" / "data" / "metadata"
        entry = json.loads((metadata / "drums-kick.json").read_text())["samples"][0]
        assert entry["drum"] == "kick"
        assert (entry["velocity_low"], entry["velocity_high"]) == (12, 15)
        assert entry["velocity_is_exact"] is False
        assert entry["source_path"].startswith("Samples/")
        assert (metadata / entry["path"]).resolve().is_file()

    def test_manifest_paths_are_relative_to_the_manifest(self, tmp_path):
        """`SampleSet.from_manifest` resolves against the manifest's directory,
        so a path written relative to anything else silently points nowhere."""
        source = tmp_path / "source"
        _write_sample(source, "Samples", "Kit", "Snare", "RR1", "01_Snare_RR1.wav")
        _write_sfz(source, "s.sfz", [("Samples\\Kit\\Snare\\RR1\\01_Snare_RR1.wav", 1, 127)])

        import_library(source, tmp_path / "project")
        metadata = tmp_path / "project" / "data" / "metadata"
        entry = json.loads((metadata / "drums-snare.json").read_text())["samples"][0]
        assert entry["path"].startswith("../samples/")
        assert (metadata / entry["path"]).resolve().is_file()

    def test_unmapped_wavs_are_copied_but_not_manifested(self, tmp_path):
        """Cymbals with no SFZ are still worth keeping — but they are not
        fitting data, and nothing should hand them to a fitter."""
        source = tmp_path / "source"
        _write_sample(source, "Samples", "Cymbals", "Hat", "RR1", "01_Hat_RR1.wav")
        _write_sfz(source, "empty.sfz", [])

        result = import_library(source, tmp_path / "project")
        assert result["source_counts"] == {"wav": 1, "mapped_wav": 0}
        assert result["manifests"] == {}
        assert result["inventory"][0]["mapped"] is False
        assert (tmp_path / "project" / "data" / "samples" / "cymbals" / "hat"
                / "rr1-01-hat-rr1.wav").is_file()


class TestOneInstrumentPerManifest:
    """Regression: the first import collapsed Tom1-Tom4 into one `toms-stereo`.

    A SampleSet is deliberately single-drum because f_static and t60 belong to a
    specific physical drum. Keying the instrument on the first folder below the
    family root — rather than on the whole chain — merges every drum that shares
    a parent folder.
    """

    @staticmethod
    def _tom_source(root: Path) -> Path:
        source = root / "source"
        regions = []
        for tom in (1, 2):
            for rr in (1, 2):
                _write_sample(
                    source, "Samples", "SMDrum Stereo (Samples)", "Toms_Stereo",
                    f"Tom{tom}", f"RR{rr}", f"01_Tom{tom}_Stereo_RR{rr}.wav",
                )
                regions.append((
                    f"Samples\\SMDrum Stereo (Samples)\\Toms_Stereo\\Tom{tom}"
                    f"\\RR{rr}\\01_Tom{tom}_Stereo_RR{rr}.wav",
                    1, 10,
                ))
        _write_sfz(source, "toms.sfz", regions)
        return source

    def test_nested_folders_produce_one_manifest_each(self, tmp_path):
        result = import_library(self._tom_source(tmp_path), tmp_path / "project")
        assert set(result["manifests"]) == {
            "drums/toms-stereo-tom1",
            "drums/toms-stereo-tom2",
        }
        assert set(result["drums"]) == {"toms-stereo-tom1", "toms-stereo-tom2"}

    def test_each_manifest_holds_exactly_one_physical_drum(self, tmp_path):
        import_library(self._tom_source(tmp_path), tmp_path / "project")
        metadata = tmp_path / "project" / "data" / "metadata"
        for tom in (1, 2):
            rows = json.loads(
                (metadata / f"drums-toms-stereo-tom{tom}.json").read_text()
            )["samples"]
            assert {row["drum"] for row in rows} == {f"toms-stereo-tom{tom}"}
            assert all(f"Tom{tom}" in row["source_path"] for row in rows)

    def test_samples_land_in_separate_directories(self, tmp_path):
        import_library(self._tom_source(tmp_path), tmp_path / "project")
        drums = tmp_path / "project" / "data" / "samples" / "drums"
        assert sorted(item.name for item in drums.iterdir()) == [
            "toms-stereo-tom1",
            "toms-stereo-tom2",
        ]

    def test_a_single_folder_drum_is_unaffected(self, tmp_path):
        source = tmp_path / "source"
        _write_sample(source, "Samples", "Kit", "Kik_Stereo", "RR1", "01_K_RR1.wav")
        _write_sfz(source, "k.sfz", [("Samples\\Kit\\Kik_Stereo\\RR1\\01_K_RR1.wav", 1, 9)])
        result = import_library(source, tmp_path / "project")
        assert set(result["drums"]) == {"kick"}  # the one aliased name


class TestRerunIsRepeatable:
    """Regression: the first run left `drums-kik-stereo.json` behind.

    It named a drum that no longer existed, pointed at a sample directory that
    was never created, and was indistinguishable from a real manifest.
    """

    @staticmethod
    def _source(tmp_path: Path) -> Path:
        source = tmp_path / "source"
        _write_sample(source, "Samples", "Kit", "Snare", "RR1", "01_Snare_RR1.wav")
        _write_sfz(source, "s.sfz", [("Samples\\Kit\\Snare\\RR1\\01_Snare_RR1.wav", 1, 127)])
        return source

    def test_stale_manifests_are_removed(self, tmp_path):
        project = tmp_path / "project"
        metadata = project / "data" / "metadata"
        metadata.mkdir(parents=True)
        (metadata / "drums-from-an-old-run.json").write_text("{}", encoding="utf-8")

        result = import_library(self._source(tmp_path), project)
        assert result["stale_manifests_removed"] == ["drums-from-an-old-run.json"]
        assert not (metadata / "drums-from-an-old-run.json").exists()

    def test_stale_sample_directories_are_reported_not_deleted(self, tmp_path):
        project = tmp_path / "project"
        stale = project / "data" / "samples" / "drums" / "old-name"
        stale.mkdir(parents=True)
        (stale / "keep.wav").write_bytes(b"x")

        result = import_library(self._source(tmp_path), project)
        assert result["stale_sample_dirs"] == ["drums/old-name"]
        assert result["stale_sample_dirs_pruned"] is False
        assert (stale / "keep.wav").exists(), "must not delete audio without --prune"

    def test_prune_deletes_them(self, tmp_path):
        project = tmp_path / "project"
        stale = project / "data" / "samples" / "drums" / "old-name"
        stale.mkdir(parents=True)
        (stale / "keep.wav").write_bytes(b"x")

        result = import_library(self._source(tmp_path), project, prune=True)
        assert result["stale_sample_dirs_pruned"] is True
        assert not stale.exists()

    def test_a_second_run_produces_identical_metadata(self, tmp_path):
        source = self._source(tmp_path)
        project = tmp_path / "project"
        first = import_library(source, project)
        snapshot = {
            item.name: item.read_text()
            for item in (project / "data" / "metadata").glob("*.json")
        }
        second = import_library(source, project)
        assert first["source_counts"] == second["source_counts"]
        assert snapshot == {
            item.name: item.read_text()
            for item in (project / "data" / "metadata").glob("*.json")
        }


class TestFileTree:
    def test_lists_every_imported_file(self, tmp_path):
        source = tmp_path / "source"
        _write_sample(source, "Samples", "Kit", "Snare", "RR1", "01_Snare_RR1.wav")
        _write_sfz(source, "s.sfz", [("Samples\\Kit\\Snare\\RR1\\01_Snare_RR1.wav", 1, 127)])

        import_library(source, tmp_path / "project")
        tree = (tmp_path / "project" / "data" / "file_tree.txt").read_text(
            encoding="utf-8"
        )
        assert tree.startswith("data\n")
        assert "rr1-01-snare-rr1.wav" in tree
        assert "drums-snare.json" in tree

    def test_is_generated_from_the_directory(self, tmp_path):
        root = tmp_path / "data"
        (root / "a").mkdir(parents=True)
        (root / "a" / "one.txt").write_text("x")
        (root / "b.txt").write_text("x")
        tree = render_file_tree(root)
        assert tree.splitlines() == ["data", " ┣ a", " ┃ ┗ one.txt", " ┗ b.txt"]
