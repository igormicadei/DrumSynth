"""Copy the DrumModalSynth sample library into DrumSynth's JSON-backed layout.

The source library stores its metadata in SFZ mappings. This importer keeps
only files that exist, copies every WAV (including currently unused cymbals),
and writes JSON manifests compatible with :mod:`drumsynth.samples`.

Usage::

    python tools/import_samples.py C:\\Projetos\\DrumModalSynth\\data
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

SLUG = re.compile(r"[^a-z0-9]+")


@dataclass
class Region:
    source: Path
    mapping: str
    family: str
    instrument: str
    round_robin: int | None
    velocity_low: float
    velocity_high: float


@dataclass
class SampleRecord:
    source: Path
    destination: Path
    family: str
    instrument: str
    velocity_ranges: list[tuple[float, float]] = field(default_factory=list)
    mappings: list[str] = field(default_factory=list)
    round_robin: int | None = None

    @property
    def velocity(self) -> float:
        low, high = self.velocity_ranges[0]
        return (low + high) / 2.0

    def to_dict(self, root: Path, source_root: Path) -> dict:
        low, high = self.velocity_ranges[0]
        return {
            "path": Path(os.path.relpath(self.destination, root)).as_posix(),
            "drum": self.instrument,
            "velocity": self.velocity,
            "velocity_low": low,
            "velocity_high": high,
            "velocity_is_exact": False,
            "velocity_ranges": [list(item) for item in self.velocity_ranges],
            "articulation": "center",
            "take": self.round_robin or 0,
            "family": self.family,
            "round_robin": self.round_robin,
            "source_path": self.source.relative_to(source_root).as_posix(),
            "source_mappings": sorted(self.mappings),
        }


def render_file_tree(root: Path) -> str:
    """Box-drawing listing of the imported library.

    The WAVs are far too large to commit, so this listing is what stands in for
    them: it is the record of which files the import actually produced, and it
    is generated here rather than by hand so it cannot drift from the manifests.
    """
    lines = [root.name]

    def walk(directory: Path, prefix: str) -> None:
        entries = sorted(
            directory.iterdir(), key=lambda item: (item.is_file(), item.name)
        )
        for index, entry in enumerate(entries):
            last = index == len(entries) - 1
            lines.append(f"{prefix}{'┗' if last else '┣'} {entry.name}")
            if entry.is_dir():
                walk(entry, prefix + "┃ ")

    walk(root, " ")
    return "\n".join(lines) + "\n"


def slugify(value: str) -> str:
    """Return a deterministic lowercase filename/directory slug."""
    value = value.replace("&", " and ").lower()
    value = SLUG.sub("-", value).strip("-")
    return value or "unknown"


#: Source folder names that should be renamed on the way in.
INSTRUMENT_ALIASES: dict[str, str] = {"kik stereo": "kick"}


def _family_and_instrument(
    source: Path, samples_root: Path
) -> tuple[str, str, int | None]:
    """Split a source path into (family, instrument slug, round-robin index).

    The instrument is the WHOLE folder chain below the family root, not just
    its first element. That distinction is load-bearing: the toms live at
    ``Toms_Stereo/Tom1/RR1/...``, and keying on the first folder alone collapses
    four physically different drums into one instrument called ``toms-stereo``.

    A `SampleSet` is deliberately single-drum, because `f_static` and `t60`
    belong to a specific physical drum — so a manifest holding Tom1 through
    Tom4 is not a naming inconvenience, it is a manifest the fitter must not be
    handed. See docs/ARCHITECTURE.md §7.4.
    """
    parts = list(source.relative_to(samples_root).parts)
    family_root = parts.pop(0)
    family = "cymbals" if "cymbal" in family_root.lower() else "drums"

    rr = None
    if len(parts) >= 2 and re.fullmatch(r"rr\d+", parts[-2], re.IGNORECASE):
        rr = int(re.search(r"\d+", parts[-2]).group())
        parts.pop(-2)

    folders = [
        re.sub(r"\s*\(samples\)", "", item, flags=re.IGNORECASE) for item in parts[:-1]
    ]
    instrument = "-".join(folders) if folders else source.stem
    alias = instrument.lower().replace("_", " ").strip()
    return family, slugify(INSTRUMENT_ALIASES.get(alias, instrument)), rr


def parse_sfz(path: Path, data_root: Path) -> list[Region]:
    """Parse the sample and velocity fields used by this library's SFZ files."""
    group: dict[str, str] = {}
    region: dict[str, str] = {}
    in_region = False
    regions: list[Region] = []

    def flush() -> None:
        if "sample" not in region:
            return
        source = data_root / Path(region["sample"].replace("\\", "/"))
        if not source.is_file():
            return
        family, instrument, rr = _family_and_instrument(source, data_root / "Samples")
        low = float(region.get("lovel", "1"))
        high = float(region.get("hivel", "127"))
        regions.append(Region(source, path.name, family, instrument, rr, low, high))

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.split("//", 1)[0].strip()
        if not line:
            continue
        if line.lower() == "<group>":
            flush()
            group = {}
            region = {}
            in_region = False
            continue
        if line.lower() == "<region>":
            flush()
            region = dict(group)
            in_region = True
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        fields = {key.strip(): value.strip()}
        if in_region:
            region.update(fields)
        else:
            group.update(fields)
    flush()
    return regions


def collect_regions(data_root: Path) -> tuple[list[Region], list[dict]]:
    mappings_root = data_root / "mappings"
    regions: list[Region] = []
    missing: list[dict] = []
    for mapping in sorted(mappings_root.glob("*.sfz")):
        for raw_line in mapping.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            line = raw_line.split("//", 1)[0].strip()
            if not line.startswith("sample="):
                continue
            source_name = line.split("=", 1)[1].strip()
            source = data_root / Path(source_name.replace("\\", "/"))
            if not source.is_file():
                missing.append({"mapping": mapping.name, "source_path": source_name})
        regions.extend(parse_sfz(mapping, data_root))
    return regions, missing


def import_library(
    source: Path,
    destination: Path,
    sample_root: Path | None = None,
    prune: bool = False,
) -> dict:
    """Copy the library and write its manifests.

    `prune` also deletes sample directories left behind by an earlier run under
    a different naming scheme. It is off by default because it removes files
    from the destination mirror, and reporting them is enough to act on.
    """
    source = source.resolve()
    destination = destination.resolve()
    sample_root = sample_root or destination / "data" / "samples"
    regions, missing = collect_regions(source)
    all_wavs = sorted((source / "Samples").rglob("*.wav"))
    records: dict[Path, SampleRecord] = {}
    for wav in all_wavs:
        family, instrument, rr = _family_and_instrument(wav, source / "Samples")
        stem = slugify(wav.stem)
        rr_prefix = f"rr{rr}-" if rr is not None else ""
        relative_destination = Path(family) / instrument / f"{rr_prefix}{stem}.wav"
        record = SampleRecord(
            wav, sample_root / relative_destination, family, instrument, round_robin=rr
        )
        records[wav] = record
    for region in regions:
        record = records[region.source]
        interval = (region.velocity_low, region.velocity_high)
        if interval not in record.velocity_ranges:
            record.velocity_ranges.append(interval)
        if region.mapping not in record.mappings:
            record.mappings.append(region.mapping)
        if record.round_robin is None:
            record.round_robin = region.round_robin
    for record in records.values():
        record.destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record.source, record.destination)

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    metadata_root = destination / "data" / "metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)
    for record in records.values():
        if record.velocity_ranges:
            grouped[(record.family, record.instrument)].append(
                record.to_dict(metadata_root, source)
            )
    manifests = {}
    written: set[str] = {"library.json"}
    for (family, instrument), entries in sorted(grouped.items()):
        manifest = {
            "schema_version": 1,
            "family": family,
            "drum": instrument,
            "sr": 44100,
            "samples": sorted(
                entries, key=lambda item: (item["velocity"], item["path"])
            ),
        }
        manifest_path = metadata_root / f"{family}-{instrument}.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        manifests[f"{family}/{instrument}"] = manifest_path.relative_to(
            destination
        ).as_posix()
        written.add(manifest_path.name)
    inventory = [
        {
            "path": record.destination.relative_to(destination / "data").as_posix(),
            "source_path": record.source.relative_to(source).as_posix(),
            "family": record.family,
            "drum": record.instrument,
            "mapped": bool(record.velocity_ranges),
        }
        for record in sorted(
            records.values(), key=lambda item: item.destination.as_posix()
        )
    ]
    library_entries = {
        record.instrument: [] for record in records.values() if record.velocity_ranges
    }
    for record in records.values():
        if record.velocity_ranges:
            library_entries[record.instrument].append(
                record.to_dict(metadata_root, source)
            )
    # Anything left over is from a previous run under a different naming scheme.
    # Left in place it is indistinguishable from a real manifest, and it points
    # at sample directories that no longer exist.
    stale = sorted(
        item.name for item in metadata_root.glob("*.json") if item.name not in written
    )
    for name in stale:
        (metadata_root / name).unlink()

    # Sample directories from an earlier scheme are the same hazard: they hold
    # real audio under a name nothing maps to any more.
    current = {record.destination.parent.resolve() for record in records.values()}
    stale_dirs = sorted(
        item.relative_to(sample_root).as_posix()
        for item in sample_root.glob("*/*")
        if item.is_dir() and item.resolve() not in current
    )
    if prune:
        for name in stale_dirs:
            shutil.rmtree(sample_root / name)

    library = {
        "schema_version": 1,
        "sr": 44100,
        "drums": {
            key: sorted(value, key=lambda item: (item["velocity"], item["path"]))
            for key, value in sorted(library_entries.items())
        },
        "manifests": manifests,
        "inventory": inventory,
        "missing_mappings": missing,
        "source_counts": {
            "wav": len(all_wavs),
            "mapped_wav": sum(item["mapped"] for item in inventory),
        },
        "stale_manifests_removed": stale,
        "stale_sample_dirs": stale_dirs,
        "stale_sample_dirs_pruned": bool(prune),
    }
    (metadata_root / "library.json").write_text(
        json.dumps(library, indent=2) + "\n", encoding="utf-8"
    )
    (destination / "data" / "file_tree.txt").write_text(
        render_file_tree(destination / "data"), encoding="utf-8"
    )
    return library


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="DrumModalSynth data directory")
    parser.add_argument(
        "--destination", type=Path, default=Path.cwd(), help="DrumSynth repository"
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="delete sample directories left over from an earlier run",
    )
    args = parser.parse_args()
    result = import_library(args.source, args.destination, prune=args.prune)
    counts = result["source_counts"]
    unmapped = counts["wav"] - counts["mapped_wav"]
    print(f"copied {counts['wav']} WAV files; {counts['mapped_wav']} have SFZ metadata")
    print(f"instruments: {len(result['manifests'])}")
    if unmapped:
        print(
            f"copied but unmapped (no SFZ region references them): {unmapped} — "
            "stored for later, not fitting data"
        )
    if result["missing_mappings"]:
        print(
            f"unresolved SFZ references (mapping points at a file that is not "
            f"there): {len(result['missing_mappings'])}"
        )
    if result["stale_manifests_removed"]:
        print(
            "removed stale manifests from an earlier run: "
            + ", ".join(result["stale_manifests_removed"])
        )
    if result["stale_sample_dirs"]:
        verb = "removed" if result["stale_sample_dirs_pruned"] else "left in place"
        print(
            f"sample directories from an earlier run ({verb}): "
            + ", ".join(result["stale_sample_dirs"])
        )
        if not result["stale_sample_dirs_pruned"]:
            print("  re-run with --prune to delete them")


if __name__ == "__main__":
    main()
