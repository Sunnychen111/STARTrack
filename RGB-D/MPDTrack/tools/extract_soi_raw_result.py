#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract raw tracking result files for mined SOI sequences/clips.

Recommended location:
  /home/cps/czl/SOI_mining/extract_soi_raw_results.py

Purpose:
  You already have raw_result folders from full-dataset evaluations, e.g.
    results for LaSOT
    results for LaSOT-ext
    results for TNL2K

  This script reads selected_clips.csv / selected_sequences.json from SOI mining,
  then copies only the selected SOI sequences' raw result txt/csv files into a
  clean SOI-Test result directory:

    output-root/
      OSTrack/
        LaSOT/
          cup-1.txt
        LaSOT-ext/
          ...
        TNL2K/
          ...
      ODTrack/
        ...
      SUTrack/
        ...

Inputs:
  --selected-clips:
    /home/cps/czl/SOI_mining/SOI-Test-150_mid/selected_clips.csv

  --raw-roots:
    Dataset-specific raw result roots, for example:
      LaSOT=/home/cps/czl/results/LaSOT,LaSOT-ext=/home/cps/czl/results/LaSOT-ext,TNL2K=/home/cps/czl/results/TNL2K

    If you have one common root containing three dataset folders, use:
      --raw-root /home/cps/czl/results
    and the script will search:
      raw-root/LaSOT
      raw-root/LaSOT-ext
      raw-root/TNL2K

Typical usage:
  cd /home/cps/czl/SOI_mining

  python extract_soi_raw_results.py \
    --selected-clips /home/cps/czl/SOI_mining/SOI-Test-150_mid/selected_clips.csv \
    --raw-root /home/cps/czl/SOI_mining/raw_results_full \
    --trackers OSTrack,ODTrack,SUTrack \
    --output-root /home/cps/czl/SOI_mining/SOI-Test-150_mid/raw_results

If your three full-result dataset folders are separate:
  python extract_soi_raw_results.py \
    --selected-clips /home/cps/czl/SOI_mining/SOI-Test-150_mid/selected_clips.csv \
    --raw-roots LaSOT=/path/to/lasot_results,LaSOT-ext=/path/to/lasotext_results,TNL2K=/path/to/tnl2k_results \
    --trackers OSTTrack,ODTrack,SUTrack \
    --output-root /home/cps/czl/SOI_mining/SOI-Test-150_mid/raw_results

If your raw files use a fixed structure, use --raw-pattern:
  python extract_soi_raw_results.py \
    --selected-clips ... \
    --raw-root /home/cps/czl/full_results \
    --raw-pattern "{raw_root}/{dataset}/{tracker}/{sequence}.txt" \
    --trackers OSTrack,ODTrack,SUTrack \
    --output-root ...

Supported placeholders in --raw-pattern:
  {raw_root}, {dataset}, {tracker}, {sequence}, {param}

Outputs:
  output-root/
    <Tracker>/<Dataset>/<sequence>.txt
    soi_result_manifest.csv
    missing_raw_results.csv
    sequence_list.txt
    report.json

Notes:
  - The script copies the whole sequence raw_result file, not only clip frames.
    This is intentional: the evaluation script can then evaluate clip intervals by
    indexing frame_id.
  - If a sequence appears in multiple clips, it is copied only once per tracker/dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


# ============================================================
# Records
# ============================================================

@dataclass
class SOISequence:
    dataset: str
    sequence: str
    num_clips: int
    min_start_frame: int
    max_end_frame: int


@dataclass
class CopyRecord:
    tracker: str
    dataset: str
    sequence: str
    source_file: str
    output_file: str
    status: str
    num_clips: int
    min_start_frame: int
    max_end_frame: int


@dataclass
class MissingRecord:
    tracker: str
    dataset: str
    sequence: str
    searched_root: str
    status: str
    num_clips: int
    min_start_frame: int
    max_end_frame: int


# ============================================================
# Parsing helpers
# ============================================================

def parse_csv_arg(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_raw_roots(s: str) -> Dict[str, Path]:
    """
    Parse:
      LaSOT=/a/b,LaSOT-ext=/c/d,TNL2K=/e/f
    """
    out: Dict[str, Path] = {}
    if not s:
        return out

    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid --raw-roots item: {item}. Expected Dataset=/path")
        k, v = item.split("=", 1)
        out[k.strip()] = Path(v.strip()).expanduser().resolve()
    return out


def read_selected_clips(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"selected file not found: {path}")

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise RuntimeError("selected json must be a list")
        return [dict(x) for x in data]

    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_soi_sequence_list(selected_path: Path) -> List[SOISequence]:
    rows = read_selected_clips(selected_path)

    merged: Dict[Tuple[str, str], Dict[str, int]] = {}

    for r in rows:
        dataset = str(r["dataset"])
        sequence = str(r["sequence"])
        start = int(float(r.get("start_frame", r.get("clip_start", 0))))
        end = int(float(r.get("end_frame", r.get("clip_end", start))))

        key = (dataset, sequence)
        if key not in merged:
            merged[key] = {
                "num_clips": 0,
                "min_start_frame": start,
                "max_end_frame": end,
            }

        merged[key]["num_clips"] += 1
        merged[key]["min_start_frame"] = min(merged[key]["min_start_frame"], start)
        merged[key]["max_end_frame"] = max(merged[key]["max_end_frame"], end)

    out: List[SOISequence] = []
    for (dataset, sequence), info in sorted(merged.items()):
        out.append(
            SOISequence(
                dataset=dataset,
                sequence=sequence,
                num_clips=info["num_clips"],
                min_start_frame=info["min_start_frame"],
                max_end_frame=info["max_end_frame"],
            )
        )
    return out


# ============================================================
# Raw result searching
# ============================================================

def format_pattern(pattern: str, raw_root: Path, tracker: str, dataset: str, sequence: str, param: str) -> Path:
    return Path(
        pattern.format(
            raw_root=str(raw_root),
            tracker=tracker,
            dataset=dataset,
            sequence=sequence,
            param=param,
        )
    ).expanduser()


def candidate_paths(raw_root: Path, tracker: str, dataset: str, sequence: str, param: str) -> List[Path]:
    """
    Try common tracking result layouts.

    raw_root can be either:
      - a common root containing dataset folders
      - a specific dataset root
    """
    names = [
        f"{sequence}.txt",
        f"{sequence}.csv",
        f"{sequence}_001.txt",
        f"{sequence}_001.csv",
    ]

    roots = [
        raw_root,
        raw_root / dataset,
    ]

    dirs: List[Path] = []
    for root in roots:
        dirs.extend([
            root / tracker,
            root / tracker / param,
            root / param / tracker,
            root / dataset / tracker,
            root / tracker / dataset,
            root / tracker / param / dataset,
            root / tracker / dataset / param,
            root / dataset / tracker / param,
            root / param / dataset / tracker,
            root / dataset / param / tracker,
            root,
        ])

    out: List[Path] = []
    for d in dirs:
        for name in names:
            out.append(d / name)

    return out


def find_raw_file(
    dataset_root: Path,
    tracker: str,
    dataset: str,
    sequence: str,
    param: str,
    raw_pattern: str,
    recursive: bool,
) -> Optional[Path]:
    if raw_pattern:
        p = format_pattern(raw_pattern, dataset_root, tracker, dataset, sequence, param)
        return p if p.exists() else None

    for p in candidate_paths(dataset_root, tracker, dataset, sequence, param):
        if p.exists():
            return p

    if not recursive:
        return None

    # Prefer recursive search inside tracker folder if possible.
    search_roots = [
        dataset_root / tracker,
        dataset_root / dataset / tracker,
        dataset_root / tracker / dataset,
        dataset_root,
    ]

    matches: List[Path] = []
    for sr in search_roots:
        if not sr.exists():
            continue
        for suffix in [".txt", ".csv"]:
            matches.extend(sr.rglob(f"{sequence}{suffix}"))
            matches.extend(sr.rglob(f"{sequence}_001{suffix}"))

    if not matches:
        return None

    def score(p: Path) -> Tuple[int, int]:
        s = str(p)
        # More hits are better, shorter path is better.
        hits = int(tracker in s) + int(dataset in s) + int(sequence in p.stem)
        return (-hits, len(s))

    return sorted(set(matches), key=score)[0]


# ============================================================
# I/O
# ============================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: List[Dict], fieldnames: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def copy_or_link(src: Path, dst: Path, mode: str) -> None:
    ensure_dir(dst.parent)

    if dst.exists():
        dst.unlink()

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "hardlink":
        try:
            dst.hardlink_to(src.resolve())
        except Exception:
            shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ============================================================
# Main
# ============================================================

def run(args: argparse.Namespace) -> None:
    selected_path = Path(args.selected_clips).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    ensure_dir(output_root)

    trackers = parse_csv_arg(args.trackers)
    params = {t: "" for t in trackers}
    if args.params:
        for item in args.params.split(","):
            if not item.strip():
                continue
            if "=" not in item:
                raise ValueError("--params should be Tracker=Param,Tracker=Param")
            k, v = item.split("=", 1)
            params[k.strip()] = v.strip()

    raw_roots = parse_raw_roots(args.raw_roots)
    common_raw_root = Path(args.raw_root).expanduser().resolve() if args.raw_root else None

    soi_sequences = build_soi_sequence_list(selected_path)

    if args.datasets:
        dataset_filter = set(parse_csv_arg(args.datasets))
        soi_sequences = [s for s in soi_sequences if s.dataset in dataset_filter]

    print("[INFO] Extract SOI raw results")
    print(f"[INFO] selected_clips = {selected_path}")
    print(f"[INFO] trackers       = {trackers}")
    print(f"[INFO] num sequences  = {len(soi_sequences)}")
    print(f"[INFO] output_root    = {output_root}")
    print(f"[INFO] mode           = {args.mode}")
    print(f"[INFO] recursive      = {args.recursive}")
    if common_raw_root:
        print(f"[INFO] raw_root       = {common_raw_root}")
    if raw_roots:
        print("[INFO] raw_roots:")
        for k, v in raw_roots.items():
            print(f"  {k}: {v}")

    copied: List[CopyRecord] = []
    missing: List[MissingRecord] = []

    for s in soi_sequences:
        if s.dataset in raw_roots:
            dataset_root = raw_roots[s.dataset]
        elif common_raw_root is not None:
            dataset_root = common_raw_root / s.dataset
            if not dataset_root.exists():
                dataset_root = common_raw_root
        else:
            raise RuntimeError("Please provide --raw-root or --raw-roots")

        for tracker in trackers:
            param = params.get(tracker, "")

            src = find_raw_file(
                dataset_root=dataset_root,
                tracker=tracker,
                dataset=s.dataset,
                sequence=s.sequence,
                param=param,
                raw_pattern=args.raw_pattern,
                recursive=args.recursive,
            )

            if src is None:
                missing.append(
                    MissingRecord(
                        tracker=tracker,
                        dataset=s.dataset,
                        sequence=s.sequence,
                        searched_root=str(dataset_root),
                        status="missing",
                        num_clips=s.num_clips,
                        min_start_frame=s.min_start_frame,
                        max_end_frame=s.max_end_frame,
                    )
                )
                continue

            suffix = src.suffix if src.suffix else ".txt"
            dst = output_root / tracker / s.dataset / f"{s.sequence}{suffix}"

            try:
                copy_or_link(src, dst, args.mode)
                copied.append(
                    CopyRecord(
                        tracker=tracker,
                        dataset=s.dataset,
                        sequence=s.sequence,
                        source_file=str(src),
                        output_file=str(dst),
                        status=args.mode,
                        num_clips=s.num_clips,
                        min_start_frame=s.min_start_frame,
                        max_end_frame=s.max_end_frame,
                    )
                )
            except Exception as e:
                missing.append(
                    MissingRecord(
                        tracker=tracker,
                        dataset=s.dataset,
                        sequence=s.sequence,
                        searched_root=str(dataset_root),
                        status=f"copy_error:{repr(e)}",
                        num_clips=s.num_clips,
                        min_start_frame=s.min_start_frame,
                        max_end_frame=s.max_end_frame,
                    )
                )

    # Save manifest.
    write_csv(
        output_root / "soi_sequence_list.csv",
        [asdict(x) for x in soi_sequences],
        list(SOISequence.__dataclass_fields__.keys()),
    )
    write_csv(
        output_root / "soi_result_manifest.csv",
        [asdict(x) for x in copied],
        list(CopyRecord.__dataclass_fields__.keys()),
    )
    write_csv(
        output_root / "missing_raw_results.csv",
        [asdict(x) for x in missing],
        list(MissingRecord.__dataclass_fields__.keys()),
    )

    with (output_root / "sequence_list.txt").open("w", encoding="utf-8") as f:
        for s in soi_sequences:
            f.write(f"{s.dataset}/{s.sequence} clips={s.num_clips} frames={s.min_start_frame}-{s.max_end_frame}\n")

    dataset_counts: Dict[str, int] = {}
    for s in soi_sequences:
        dataset_counts[s.dataset] = dataset_counts.get(s.dataset, 0) + 1

    copied_counts: Dict[str, int] = {}
    for r in copied:
        key = f"{r.tracker}/{r.dataset}"
        copied_counts[key] = copied_counts.get(key, 0) + 1

    report = {
        "selected_clips": str(selected_path),
        "output_root": str(output_root),
        "trackers": trackers,
        "num_unique_soi_sequences": len(soi_sequences),
        "dataset_sequence_counts": dataset_counts,
        "num_copied_records": len(copied),
        "num_missing_records": len(missing),
        "copied_counts": copied_counts,
        "mode": args.mode,
    }
    with (output_root / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n[DONE]")
    print(f"  unique SOI sequences: {len(soi_sequences)}")
    print(f"  copied/link records:  {len(copied)}")
    print(f"  missing records:      {len(missing)}")
    print(f"  output_root:          {output_root}")

    if missing:
        print("\n[WARN] Some raw result files were not found.")
        print(f"       See: {output_root / 'missing_raw_results.csv'}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Extract raw_result files for selected SOI sequences")

    parser.add_argument("--selected-clips", type=str, required=True,
                        help="selected_clips.csv or selected_sequences.json")
    parser.add_argument("--trackers", type=str, required=True,
                        help="Comma-separated tracker names, e.g. OSTrack,ODTrack,SUTrack")
    parser.add_argument("--datasets", type=str, default="",
                        help="Optional dataset filter, e.g. LaSOT,LaSOT-ext,TNL2K")

    parser.add_argument("--raw-root", type=str, default="",
                        help="Common root containing raw results or dataset folders.")
    parser.add_argument("--raw-roots", type=str, default="",
                        help="Dataset-specific roots: LaSOT=/path,LaSOT-ext=/path,TNL2K=/path")
    parser.add_argument("--raw-pattern", type=str, default="",
                        help="Optional exact pattern with placeholders: {raw_root},{tracker},{dataset},{sequence},{param}")

    parser.add_argument("--params", type=str, default="",
                        help="Optional Tracker=Param mapping used in raw search/pattern, e.g. OSTrack=vitb,SUTrack=sutrack_b224")

    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--mode", type=str, default="copy", choices=["copy", "symlink", "hardlink"],
                        help="copy files, create symlinks, or hardlinks. symlink is fastest.")
    parser.add_argument("--recursive", action="store_true",
                        help="Enable recursive search when common layouts fail.")

    return parser


def main() -> None:
    args = build_argparser().parse_args()
    run(args)


if __name__ == "__main__":
    main()