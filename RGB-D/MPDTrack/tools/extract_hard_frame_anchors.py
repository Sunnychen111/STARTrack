#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract hard-frame anchors from offline STARTrack/SUTrack .pt feature files with tqdm progress bars.

Hard-frame definition:
    baseline_iou = topk_ious[0]
    best_iou     = max(topk_ious)
    best_idx     = argmax(topk_ious)
    iou_gain     = best_iou - baseline_iou

Recommended clean hard-frame condition:
    best_idx != 0
    baseline_iou >= 0.30
    best_iou >= 0.50
    iou_gain >= 0.10

Output format for Stage4:
    <seq_key> <frame_idx>
"""

import argparse
import csv
import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def unique_keep_order(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def to_numpy(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


def parse_got10k_zero_based_id(stem):
    """GOT-10k_Train_000080 -> 000079"""
    m = re.search(r"(\d{6})$", stem)
    if m is None:
        return None
    idx = int(m.group(1))
    if idx <= 0:
        return f"{idx:06d}"
    return f"{idx - 1:06d}"


def infer_dataset_and_seq_key(pt_path):
    p = Path(pt_path)
    stem = p.stem
    lower_line = str(p).lower()
    lower_parts = [x.lower() for x in p.parts]

    if "lasot" in lower_parts or "/lasot/" in lower_line:
        return "lasot", stem

    if (
        "got10k" in lower_parts
        or "got-10k" in lower_parts
        or "/got10k/" in lower_line
        or "/got-10k/" in lower_line
        or stem.lower().startswith("got-10")
    ):
        got_id = parse_got10k_zero_based_id(stem)
        if got_id is not None:
            return "got10k", got_id
        return "got10k", stem

    return "unknown", stem


def get_first_existing(d, keys, default=None):
    if not isinstance(d, dict):
        return default
    for k in keys:
        if k in d:
            return d[k]
    return default


def extract_frame_items(data):
    """Return list of {frame_idx:int, topk_ious:np.ndarray}."""
    items = []

    if isinstance(data, list):
        frames = data
    elif isinstance(data, dict):
        frames = None
        for key in ["frames", "frame_records", "records", "data", "samples"]:
            if key in data and isinstance(data[key], list):
                frames = data[key]
                break
        if frames is None and any(k in data for k in ["topk_ious", "topk_iou", "candidate_ious", "ious"]):
            frames = [data]
    else:
        frames = None

    if frames is not None:
        for i, fr in enumerate(frames):
            if not isinstance(fr, dict):
                continue
            topk_ious = get_first_existing(fr, ["topk_ious", "topk_iou", "candidate_ious", "ious"])
            topk_ious = to_numpy(topk_ious)
            if topk_ious is None:
                continue
            topk_ious = topk_ious.reshape(-1)
            frame_idx = get_first_existing(
                fr,
                ["frame_idx", "frame_id", "search_frame_idx", "search_frame_id", "idx", "image_id"],
                default=i,
            )
            try:
                frame_idx = int(frame_idx)
            except Exception:
                frame_idx = int(i)
            items.append({"frame_idx": frame_idx, "topk_ious": topk_ious})
        return items

    if isinstance(data, dict):
        topk_ious = get_first_existing(data, ["topk_ious", "topk_iou", "candidate_ious", "ious"])
        arr = to_numpy(topk_ious)
        if arr is not None:
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            for i in range(arr.shape[0]):
                items.append({"frame_idx": int(i), "topk_ious": arr[i].reshape(-1)})
    return items


def is_hard_frame(
    topk_ious,
    min_baseline_iou=None,
    max_baseline_iou=None,
    min_best_iou=0.50,
    min_iou_gain=0.10,
    require_non_top1=True,
    max_best_idx=None,
):
    if topk_ious is None:
        return None
    topk_ious = np.asarray(topk_ious, dtype=np.float32).reshape(-1)
    if topk_ious.size == 0 or not np.isfinite(topk_ious).all():
        return None

    baseline_iou = float(topk_ious[0])
    best_idx = int(np.argmax(topk_ious))
    best_iou = float(topk_ious[best_idx])
    iou_gain = float(best_iou - baseline_iou)

    if require_non_top1 and best_idx == 0:
        return None
    if min_baseline_iou is not None and baseline_iou < float(min_baseline_iou):
        return None
    if max_baseline_iou is not None and baseline_iou > float(max_baseline_iou):
        return None
    if best_iou < float(min_best_iou):
        return None
    if iou_gain < float(min_iou_gain):
        return None
    if max_best_idx is not None and best_idx > int(max_best_idx):
        return None

    return {
        "baseline_iou": baseline_iou,
        "best_iou": best_iou,
        "iou_gain": iou_gain,
        "best_idx": best_idx,
    }


def filter_with_gap_and_limit(rows, min_frame_gap=5, max_frames_per_seq=200):
    if len(rows) == 0:
        return []
    rows_sorted = sorted(rows, key=lambda x: x["iou_gain"], reverse=True)
    selected = []
    for r in rows_sorted:
        f = int(r["frame_idx"])
        if min_frame_gap > 0:
            if any(abs(f - int(s["frame_idx"])) < min_frame_gap for s in selected):
                continue
        selected.append(r)
        if max_frames_per_seq is not None and max_frames_per_seq > 0 and len(selected) >= max_frames_per_seq:
            break
    return sorted(selected, key=lambda x: int(x["frame_idx"]))


def read_pt_paths(args):
    paths = []
    if args.pt_list is not None:
        with open(args.pt_list, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                pt = line.split()[0]
                if pt.startswith("home/"):
                    pt = "/" + pt
                paths.append(str(Path(pt)))
    if args.feature_root:
        roots = args.feature_root
        root_iter = roots if args.no_progress or tqdm is None else tqdm(roots, desc="Collecting roots", dynamic_ncols=True)
        for root in root_iter:
            root = Path(root)
            paths.extend([str(p) for p in root.rglob("*.pt")])
    return unique_keep_order(paths)


def save_anchor_lists(out_dir, selected_rows):
    combined_txt = out_dir / "hard_frame_anchor_list.txt"
    lasot_txt = out_dir / "lasot_hard_frame_anchor_list.txt"
    got10k_txt = out_dir / "got10k_hard_frame_anchor_list.txt"
    unknown_txt = out_dir / "unknown_hard_frame_anchor_list.txt"

    with open(combined_txt, "w", encoding="utf-8") as f_all, \
         open(lasot_txt, "w", encoding="utf-8") as f_lasot, \
         open(got10k_txt, "w", encoding="utf-8") as f_got, \
         open(unknown_txt, "w", encoding="utf-8") as f_unknown:
        for r in selected_rows:
            line = f"{r['seq_key']} {r['frame_idx']}\n"
            f_all.write(line)
            if r["dataset"] == "lasot":
                f_lasot.write(line)
            elif r["dataset"] == "got10k":
                f_got.write(line)
            else:
                f_unknown.write(line)
    return combined_txt, lasot_txt, got10k_txt, unknown_txt


def save_details_csv(path, selected_rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset", "seq_key", "frame_idx", "best_idx",
                "baseline_iou", "best_iou", "iou_gain", "pt_path",
            ],
        )
        writer.writeheader()
        for r in selected_rows:
            writer.writerow(r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt-list", type=str, default=None)
    parser.add_argument("--feature-root", type=str, nargs="*", default=None)
    parser.add_argument("--out-dir", type=str, default="stage4_hard_frames_gain010_clean")
    parser.add_argument("--dataset-filter", type=str, default=None, choices=[None, "lasot", "got10k", "unknown"])
    parser.add_argument("--min-baseline-iou", type=float, default=None)
    parser.add_argument("--max-baseline-iou", type=float, default=None)
    parser.add_argument("--min-best-iou", type=float, default=0.50)
    parser.add_argument("--min-iou-gain", type=float, default=0.10)
    parser.add_argument("--max-best-idx", type=int, default=None)
    parser.add_argument("--allow-top1", action="store_true")
    parser.add_argument("--min-frame-gap", type=int, default=5)
    parser.add_argument("--max-frames-per-seq", type=int, default=200)
    parser.add_argument("--output-frame-offset", type=int, default=0)
    parser.add_argument("--load-weights-only", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = read_pt_paths(args)
    if len(paths) == 0:
        raise ValueError("No .pt files found. Provide --pt-list or --feature-root.")

    use_progress = (not args.no_progress) and tqdm is not None

    print("=" * 100)
    print("[INFO] Extract hard frame anchors with progress bar")
    print(f"[INFO] num pt files          : {len(paths)}")
    print(f"[INFO] out_dir               : {out_dir}")
    print(f"[INFO] dataset_filter        : {args.dataset_filter}")
    print(f"[INFO] min_baseline_iou      : {args.min_baseline_iou}")
    print(f"[INFO] max_baseline_iou      : {args.max_baseline_iou}")
    print(f"[INFO] min_best_iou          : {args.min_best_iou}")
    print(f"[INFO] min_iou_gain          : {args.min_iou_gain}")
    print(f"[INFO] require_non_top1      : {not args.allow_top1}")
    print(f"[INFO] min_frame_gap         : {args.min_frame_gap}")
    print(f"[INFO] max_frames_per_seq    : {args.max_frames_per_seq}")
    print(f"[INFO] progress_bar          : {use_progress}")
    print("=" * 100)

    raw_by_seq = defaultdict(list)
    failed = []
    no_ious = []
    loaded_count = 0
    scanned_frames = 0
    raw_hard_count = 0

    iterator = tqdm(paths, total=len(paths), desc="Scanning .pt", dynamic_ncols=True, mininterval=0.5) if use_progress else paths

    for pt_path in iterator:
        pt_path = str(pt_path)
        dataset, seq_key = infer_dataset_and_seq_key(pt_path)
        if args.dataset_filter is not None and dataset != args.dataset_filter:
            continue

        try:
            data = torch.load(pt_path, map_location="cpu", weights_only=bool(args.load_weights_only))
            loaded_count += 1
        except Exception as e:
            failed.append({"path": pt_path, "error": repr(e)})
            if not use_progress:
                print(f"[WARN] failed to load {pt_path}: {repr(e)}")
            continue

        items = extract_frame_items(data)
        if len(items) == 0:
            no_ious.append(pt_path)
            continue
        scanned_frames += len(items)

        for item in items:
            frame_idx = int(item["frame_idx"]) + int(args.output_frame_offset)
            stat = is_hard_frame(
                item["topk_ious"],
                min_baseline_iou=args.min_baseline_iou,
                max_baseline_iou=args.max_baseline_iou,
                min_best_iou=args.min_best_iou,
                min_iou_gain=args.min_iou_gain,
                require_non_top1=not args.allow_top1,
                max_best_idx=args.max_best_idx,
            )
            if stat is None:
                continue
            raw_hard_count += 1
            raw_by_seq[(dataset, seq_key)].append({
                "dataset": dataset,
                "seq_key": seq_key,
                "frame_idx": frame_idx,
                "best_idx": stat["best_idx"],
                "baseline_iou": stat["baseline_iou"],
                "best_iou": stat["best_iou"],
                "iou_gain": stat["iou_gain"],
                "pt_path": pt_path,
            })

        if use_progress:
            iterator.set_postfix({
                "loaded": loaded_count,
                "frames": scanned_frames,
                "raw_hard": raw_hard_count,
                "seqs": len(raw_by_seq),
            })

    selected_rows = []
    seq_items = list(raw_by_seq.items())
    seq_iterator = tqdm(seq_items, total=len(seq_items), desc="Filtering per sequence", dynamic_ncols=True, mininterval=0.5) if use_progress else seq_items
    for (_, _), rows in seq_iterator:
        selected = filter_with_gap_and_limit(rows, args.min_frame_gap, args.max_frames_per_seq)
        selected_rows.extend(selected)
        if use_progress:
            seq_iterator.set_postfix({"selected": len(selected_rows)})

    selected_rows = sorted(selected_rows, key=lambda x: (x["dataset"], x["seq_key"], int(x["frame_idx"])))
    combined_txt, lasot_txt, got10k_txt, unknown_txt = save_anchor_lists(out_dir, selected_rows)
    details_csv = out_dir / "hard_frame_anchor_details.csv"
    save_details_csv(details_csv, selected_rows)

    if failed:
        with open(out_dir / "failed_pt_files.json", "w", encoding="utf-8") as f:
            json.dump(failed, f, indent=2)
    if no_ious:
        with open(out_dir / "pt_files_without_topk_ious.txt", "w", encoding="utf-8") as f:
            for p in no_ious:
                f.write(p + "\n")

    count_by_dataset = defaultdict(int)
    count_by_seq = defaultdict(int)
    for r in selected_rows:
        count_by_dataset[r["dataset"]] += 1
        count_by_seq[f"{r['dataset']}::{r['seq_key']}"] += 1

    summary = {
        "num_pt_files_input": len(paths),
        "num_pt_files_loaded": loaded_count,
        "num_pt_files_failed": len(failed),
        "num_pt_files_without_topk_ious": len(no_ious),
        "num_scanned_frames": scanned_frames,
        "num_raw_hard_frames_before_gap_filter": raw_hard_count,
        "num_sequences_with_raw_hard_frames": len(raw_by_seq),
        "num_selected_hard_frames": len(selected_rows),
        "count_by_dataset": dict(count_by_dataset),
        "num_sequences_selected": len(count_by_seq),
        "thresholds": {
            "dataset_filter": args.dataset_filter,
            "min_baseline_iou": args.min_baseline_iou,
            "max_baseline_iou": args.max_baseline_iou,
            "min_best_iou": args.min_best_iou,
            "min_iou_gain": args.min_iou_gain,
            "max_best_idx": args.max_best_idx,
            "require_non_top1": not args.allow_top1,
            "min_frame_gap": args.min_frame_gap,
            "max_frames_per_seq": args.max_frames_per_seq,
            "output_frame_offset": args.output_frame_offset,
        },
        "outputs": {
            "combined_anchor_list": str(combined_txt),
            "lasot_anchor_list": str(lasot_txt),
            "got10k_anchor_list": str(got10k_txt),
            "unknown_anchor_list": str(unknown_txt),
            "details_csv": str(details_csv),
        },
    }
    summary_json = out_dir / "hard_frame_anchor_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 100)
    print("[DONE] Hard frame extraction finished")
    print(f"[OUT] combined anchors : {combined_txt}")
    print(f"[OUT] LaSOT anchors    : {lasot_txt}")
    print(f"[OUT] GOT10K anchors   : {got10k_txt}")
    print(f"[OUT] Unknown anchors  : {unknown_txt}")
    print(f"[OUT] details CSV      : {details_csv}")
    print(f"[OUT] summary JSON     : {summary_json}")
    print("-" * 100)
    print(f"Loaded pt files        : {loaded_count}/{len(paths)}")
    print(f"Failed pt files        : {len(failed)}")
    print(f"No topk_ious files     : {len(no_ious)}")
    print(f"Scanned frames         : {scanned_frames}")
    print(f"Raw hard frames        : {raw_hard_count}")
    print(f"Selected hard frames   : {len(selected_rows)}")
    print(f"Count by dataset       : {dict(count_by_dataset)}")
    print("=" * 100)

    print("\n[Preview]")
    for r in selected_rows[:20]:
        print(
            f"{r['dataset']:7s} {r['seq_key']:30s} "
            f"frame={int(r['frame_idx']):6d} "
            f"best_idx={int(r['best_idx'])} "
            f"base={float(r['baseline_iou']):.4f} "
            f"best={float(r['best_iou']):.4f} "
            f"gain={float(r['iou_gain']):.4f}"
        )


if __name__ == "__main__":
    main()
