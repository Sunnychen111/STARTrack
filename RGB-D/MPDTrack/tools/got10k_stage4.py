#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run STARTrack Stage4 checkpoint on GOT-10k Test without GT metrics.

It does inference and saves:

1. GOT-10k official-style outputs:
    out_dir/tracking_results/<tracker_name>/<result_param>/<sequence>.txt
    out_dir/tracking_results/<tracker_name>/<result_param>/<sequence>_time.txt

2. Legacy debug outputs:
    out_dir/predictions/<sequence>.txt
    out_dir/predictions/<sequence>_time.txt
    out_dir/peak_info/<sequence>_peak_info.csv

Official bbox format:
    x y w h
    no comma, no header

Official time format:
    one value per frame, in seconds

It supports:
    --seq GOT-10k_Test_000001
    --seq 000000   # treated as zero-based, maps to GOT-10k_Test_000001
    --seq-list got10k_test_seq.txt

Resume / skip:
    By default, if official outputs already exist and contain the expected
    number of rows, the sequence will be skipped in the next run.
    Use --force-rerun to ignore existing outputs and recompute.

Initialization:
    1. If groundtruth.txt exists, use its first line as init bbox.
    2. Otherwise, for single sequence you can pass --init-bbox x,y,w,h.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch


DEFAULT_PROJECT_ROOT = "/home/cps/czl/STARTrack_v6"


# ============================================================
# Project setup
# ============================================================

def setup_project_root(project_root):
    project_root = os.path.abspath(project_root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return project_root


# ============================================================
# Basic IO
# ============================================================

def read_rgb(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def list_images(seq_dir):
    seq_dir = Path(seq_dir)
    files = sorted([
        p for p in seq_dir.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
    ])
    if len(files) == 0:
        raise FileNotFoundError(f"No image files found in: {seq_dir}")
    return files


def load_txt_array(path, delimiter=None):
    path = str(path)
    try:
        arr = np.loadtxt(path, delimiter=delimiter, dtype=np.float32)
    except Exception:
        try:
            arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
        except Exception:
            arr = np.loadtxt(path, dtype=np.float32)

    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


# ============================================================
# GOT-10k sequence loading
# ============================================================

def got10k_candidates(root, seq_token, split_hint="test"):
    root = Path(root)

    if seq_token.startswith("GOT-10k") or seq_token.startswith("GOT-10K"):
        return [root / seq_token]

    if seq_token.isdigit() and len(seq_token) == 6:
        zero_based = int(seq_token)
        one_based = zero_based + 1

        split_hint_lower = split_hint.lower()
        if "train" in split_hint_lower:
            prefixes = ["GOT-10k_Train", "GOT-10K_Train"]
        elif "val" in split_hint_lower:
            prefixes = ["GOT-10k_Val", "GOT-10K_Val"]
        else:
            prefixes = ["GOT-10k_Test", "GOT-10K_Test"]

        cands = []
        for prefix in prefixes:
            # Your convention:
            #   000000 -> GOT-10k_Test_000001
            #   000079 -> GOT-10k_Test_000080
            cands.append(root / f"{prefix}_{one_based:06d}")

            # Also try direct indexing in case user passes official 1-based ID.
            cands.append(root / f"{prefix}_{zero_based:06d}")

        return cands

    return [root / seq_token]


def parse_init_bbox(init_bbox_str):
    if init_bbox_str is None:
        return None

    parts = init_bbox_str.replace(",", " ").split()
    if len(parts) != 4:
        raise ValueError(f"--init-bbox must have 4 numbers, got: {init_bbox_str}")

    return [float(x) for x in parts]


def load_got10k_test_sequence(root, seq_token, split_hint="test", init_bbox=None):
    seq_dir = None
    tried = []

    for cand in got10k_candidates(root, seq_token, split_hint=split_hint):
        tried.append(str(cand))
        if cand.is_dir():
            seq_dir = cand
            break

    if seq_dir is None:
        raise FileNotFoundError(
            "Cannot find GOT-10k sequence dir. Tried:\n" + "\n".join(tried)
        )

    img_files = list_images(seq_dir)

    gt_path = seq_dir / "groundtruth.txt"

    if gt_path.is_file():
        gt = load_txt_array(gt_path, delimiter=",")[:, :4].astype(np.float32)
        if gt.shape[0] < 1:
            raise ValueError(f"Empty groundtruth.txt: {gt_path}")
        init = gt[0].tolist()
    else:
        if init_bbox is None:
            raise FileNotFoundError(
                f"No groundtruth.txt found for init bbox: {gt_path}\n"
                f"For GOT-10k Test, provide --init-bbox x,y,w,h if your copy does not include first-frame init."
            )
        init = init_bbox

    return seq_dir.name, img_files, init


# ============================================================
# Saving
# ============================================================

def save_bboxes_legacy_comma(path, bboxes):
    """
    Debug/legacy copy:
        x,y,w,h
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(path), bboxes, fmt="%.4f", delimiter=",")


def save_got10k_bboxes(path, bboxes, fmt="%.4f"):
    """
    GOT-10k official style:
        x y w h
    no comma, no header.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(path), bboxes, fmt=fmt, delimiter="\t")


def save_got10k_times(path, times):
    """
    GOT-10k official style:
        one tracking time per frame, in seconds.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(path), times.reshape(-1, 1), fmt="%.6f", delimiter="\t")


def save_tracker_result_got_style(root_dir, tracker_name, param_name, seq_name, bboxes, times, bbox_fmt="%.4f"):
    """
    Output:
        root_dir/tracker_name/param_name/seq_name.txt
        root_dir/tracker_name/param_name/seq_name_time.txt
    """
    out_dir = Path(root_dir) / tracker_name / param_name
    out_dir.mkdir(parents=True, exist_ok=True)

    bbox_path = out_dir / f"{seq_name}.txt"
    time_path = out_dir / f"{seq_name}_time.txt"

    save_got10k_bboxes(bbox_path, bboxes, fmt=bbox_fmt)
    save_got10k_times(time_path, times)

    return bbox_path, time_path


# ============================================================
# Resume / skip existing results
# ============================================================

def _load_txt_for_resume_check(path, kind):
    """
    Load an existing result file for resume checking.

    Returns:
        arr, reason
    If loading fails, arr is None and reason explains why.
    """
    path = Path(path)

    if not path.is_file():
        return None, f"missing {kind}: {path}"

    if path.stat().st_size <= 0:
        return None, f"empty {kind}: {path}"

    try:
        arr = np.loadtxt(str(path), dtype=np.float32)
    except Exception as e:
        return None, f"cannot read {kind}: {path}, error={repr(e)}"

    if arr.size == 0:
        return None, f"empty parsed {kind}: {path}"

    if not np.all(np.isfinite(arr)):
        return None, f"non-finite values in {kind}: {path}"

    return arr, "ok"


def _num_rows_from_loaded_txt(arr):
    """
    np.loadtxt returns:
        scalar for one value,
        1D for one-column files or one-row multi-column files,
        2D for normal files.

    For bbox files, one-row 4-column text becomes shape=(4,), so the caller
    should separately validate columns when needed.
    """
    if arr.ndim == 0:
        return 1
    if arr.ndim == 1:
        return int(arr.shape[0])
    return int(arr.shape[0])


def check_existing_sequence_outputs(
    official_root,
    tracker_name,
    param_name,
    seq_name,
    expected_frames,
):
    """
    Check whether official GOT-style outputs are complete enough to skip.

    A sequence is considered complete only if:
      1. <seq>.txt exists and has expected_frames rows with 4 bbox columns.
      2. <seq>_time.txt exists and has expected_frames timing rows.
      3. All parsed values are finite.

    Returns:
        complete: bool
        reason: str
        paths: dict
    """
    result_dir = Path(official_root) / tracker_name / param_name
    bbox_path = result_dir / f"{seq_name}.txt"
    time_path = result_dir / f"{seq_name}_time.txt"

    paths = {
        "official_bbox": str(bbox_path),
        "official_time": str(time_path),
    }

    bbox_arr, reason = _load_txt_for_resume_check(bbox_path, "bbox")
    if bbox_arr is None:
        return False, reason, paths

    # For a normal bbox file:
    #   many frames -> shape=(N, 4)
    #   one frame   -> shape=(4,), convert to shape=(1, 4)
    if bbox_arr.ndim == 1:
        if bbox_arr.shape[0] == 4:
            bbox_arr = bbox_arr.reshape(1, 4)
        else:
            return False, f"invalid bbox shape {bbox_arr.shape}: {bbox_path}", paths
    elif bbox_arr.ndim == 2:
        if bbox_arr.shape[1] < 4:
            return False, f"bbox columns < 4, shape={bbox_arr.shape}: {bbox_path}", paths
    else:
        return False, f"invalid bbox ndim={bbox_arr.ndim}: {bbox_path}", paths

    if bbox_arr.shape[0] != expected_frames:
        return (
            False,
            f"bbox rows mismatch: got {bbox_arr.shape[0]}, expected {expected_frames}: {bbox_path}",
            paths,
        )

    time_arr, reason = _load_txt_for_resume_check(time_path, "time")
    if time_arr is None:
        return False, reason, paths

    if time_arr.ndim == 0:
        time_rows = 1
    elif time_arr.ndim == 1:
        time_rows = int(time_arr.shape[0])
    else:
        time_rows = int(time_arr.shape[0])

    if time_rows != expected_frames:
        return (
            False,
            f"time rows mismatch: got {time_rows}, expected {expected_frames}: {time_path}",
            paths,
        )

    return True, "complete", paths


# ============================================================
# Checkpoint compatibility
# ============================================================

def make_startrack_test_ckpt(src_path, out_dir, tag):
    """
    Allows direct passing Stage4 best.pth:
        ckpt["disambiguator"] -> temporary ckpt["model"]

    If checkpoint already has ckpt["model"], returns original path.
    """
    src_path = Path(src_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(str(src_path), map_location="cpu")

    if isinstance(ckpt, dict) and "model" in ckpt:
        return str(src_path)

    if isinstance(ckpt, dict) and "disambiguator" in ckpt:
        dst = out_dir / f"{tag}_for_test.pth"
        out = {
            "epoch": ckpt.get("epoch", -1),
            "model": ckpt["disambiguator"],
            "args": ckpt.get("args", {}),
            "metrics": ckpt.get("metrics", {}),
            "stage": ckpt.get("stage", "stage4_online_joint"),
            "source": str(src_path),
        }
        torch.save(out, str(dst))
        print(f"[INFO] Converted Stage4 checkpoint for test: {dst}")
        return str(dst)

    raise KeyError(
        f"Cannot find 'model' or 'disambiguator' in {src_path}. keys={list(ckpt.keys())}"
    )


# ============================================================
# Tracker params
# ============================================================

def patch_params(params):
    if not hasattr(params, "debug"):
        params.debug = 0
    if not hasattr(params, "save_all_boxes"):
        params.save_all_boxes = False
    if not hasattr(params, "visualization"):
        params.visualization = False
    if not hasattr(params, "use_visdom"):
        params.use_visdom = False
    if not hasattr(params, "visdom_info"):
        params.visdom_info = None
    return params


def override_startrack_params(params, star_ckpt=None, mode=None):
    if star_ckpt is not None:
        params.startrack_ckpt = star_ckpt
        params.reranker_ckpt = star_ckpt
        params.STARTRACK_CKPT = star_ckpt

        if hasattr(params, "cfg") and hasattr(params.cfg, "MODEL"):
            try:
                params.cfg.MODEL.STARTRACK_CKPT = star_ckpt
                params.cfg.MODEL.USE_STARTRACK = True
            except Exception:
                pass

    if mode is not None:
        params.startrack_state_mode = mode
        params.STARTRACK_STATE_MODE = mode

        if hasattr(params, "cfg") and hasattr(params.cfg, "MODEL"):
            try:
                params.cfg.MODEL.STARTRACK_STATE_MODE = mode
                params.cfg.MODEL.USE_STARTRACK = True
            except Exception:
                pass

    return params


# ============================================================
# Running tracker
# ============================================================

def run_tracker_on_sequence(
    param_name,
    dataset_name,
    img_files,
    init_bbox,
    base_checkpoint,
    stage4_ckpt,
    mode=None,
):
    from lib.test.parameter.sutrack import parameters
    from lib.test.tracker.sutrack import get_tracker_class

    TrackerClass = get_tracker_class()

    params = parameters(param_name)
    params = patch_params(params)

    if base_checkpoint is not None:
        params.checkpoint = base_checkpoint
        if hasattr(params, "cfg") and hasattr(params.cfg, "TEST"):
            try:
                params.cfg.TEST.CHECKPOINT = base_checkpoint
            except Exception:
                pass

    params = override_startrack_params(params, star_ckpt=stage4_ckpt, mode=mode)

    tracker = TrackerClass(params, dataset_name)

    first_img = read_rgb(img_files[0])

    t0 = time.time()
    tracker.initialize(first_img, {"init_bbox": init_bbox})
    init_time = time.time() - t0

    pred_bboxes = np.zeros((len(img_files), 4), dtype=np.float32)
    pred_bboxes[0] = np.array(init_bbox, dtype=np.float32)

    # Keep the same number of rows as bbox file.
    # times[0] is initialization time.
    times = np.zeros((len(img_files),), dtype=np.float32)
    times[0] = float(init_time)

    peak_infos = []

    for i in range(1, len(img_files)):
        img = read_rgb(img_files[i])

        t0 = time.time()
        with torch.no_grad():
            out = tracker.track(img, info={})
        times[i] = float(time.time() - t0)

        pred_bboxes[i] = np.array(out["target_bbox"], dtype=np.float32)

        peak_info = out.get("peak_info", {})
        peak_infos.append({
            "frame": int(i),
            "rerank_used": bool(peak_info.get("rerank_used", False)),
            "should_update": bool(peak_info.get("should_update", False)),
            "selected_idx": int(peak_info.get("selected_idx", -1)),
            "target_prob": float(peak_info.get("target_prob", 0.0)),
            "identity_margin": float(peak_info.get("identity_margin", 0.0)),
            "ambiguity_ratio": float(peak_info.get("ambiguity_ratio", 0.0)),
            "history_len": int(peak_info.get("history_len", -1)),
            "update_history_count": int(peak_info.get("update_history_count", 0)),
            "time": float(times[i]),
        })

        if i % 200 == 0:
            print(f"[{dataset_name}] processed {i}/{len(img_files)} frames")

    return pred_bboxes, times, peak_infos


def save_peak_info(path, peak_infos):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "frame,rerank_used,should_update,selected_idx,"
            "target_prob,identity_margin,ambiguity_ratio,"
            "history_len,update_history_count,time\n"
        )
        for item in peak_infos:
            f.write(
                f"{item.get('frame', -1)},"
                f"{int(item.get('rerank_used', False))},"
                f"{int(item.get('should_update', False))},"
                f"{item.get('selected_idx', -1)},"
                f"{item.get('target_prob', 0.0)},"
                f"{item.get('identity_margin', 0.0)},"
                f"{item.get('ambiguity_ratio', 0.0)},"
                f"{item.get('history_len', -1)},"
                f"{item.get('update_history_count', 0)},"
                f"{item.get('time', 0.0)}\n"
            )


def read_seq_list(path):
    if path is None:
        return []

    seqs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) == 1:
                seqs.append(parts[0])
            else:
                seqs.append(parts[-1])

    return seqs


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--project-root", type=str, default=DEFAULT_PROJECT_ROOT)

    parser.add_argument("--root", type=str, default="/home/cps/SOT_dataset/got10k/test_data/test", help="GOT-10k test root")
    parser.add_argument("--got-split", type=str, default="test")

    parser.add_argument("--seq", type=str, default=None)
    parser.add_argument("--seq-list", type=str, default="/home/cps/SOT_dataset/got10k/test_data/test/list.txt")
    parser.add_argument("--max-seqs", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)

    parser.add_argument("--star-param", type=str, default="sutrack_b384_startrack")
    parser.add_argument("--tracker-dataset-name", type=str, default="got10k_test")

    parser.add_argument("--base-checkpoint", type=str, default="/home/cps/czl/STARTrack_v6/checkpoints/sutrack_b384/SUTRACK_ep0180.pth.tar")
    parser.add_argument("--stage4-ckpt", type=str, default="/home/cps/czl/STARTrack_v6/checkpoints/stage4_got10k_three_stage_b384_v2/last.pth")
    parser.add_argument("--star-mode", type=str, default="coach")

    parser.add_argument("--init-bbox", type=str, default=None, help="Only used when sequence has no groundtruth.txt")
    parser.add_argument("--out-dir", type=str, default="got10k_test_newstage4")

    # Official GOT-style output controls
    parser.add_argument("--tracker-name", type=str, default="MPDTrack")
    parser.add_argument("--result-param", type=str, default="MPDtrack_5.18_v2")
    parser.add_argument(
        "--bbox-fmt",
        type=str,
        default="%.0f",
        help="Use %.0f if you want integer-like bbox output.",
    )

    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip a sequence when official bbox/time outputs already exist and have expected rows. Default: True.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Ignore existing outputs and rerun all selected sequences.",
    )

    args = parser.parse_args()

    setup_project_root(args.project_root)

    seqs = []
    if args.seq is not None:
        seqs.append(args.seq)
    seqs.extend(read_seq_list(args.seq_list))

    # Unique keep order
    seen = set()
    unique = []
    for s in seqs:
        if s not in seen:
            unique.append(s)
            seen.add(s)
    seqs = unique

    if args.max_seqs is not None:
        seqs = seqs[:args.max_seqs]

    if len(seqs) == 0:
        raise ValueError("Please provide --seq or --seq-list.")

    converted_dir = Path(args.out_dir) / "_converted_ckpts"
    # Lazy conversion: if all selected sequences are skipped by resume checking,
    # we do not need to load/convert the Stage4 checkpoint at all.
    stage4_ckpt = None

    init_bbox = parse_init_bbox(args.init_bbox)

    print("=" * 100)
    print("[INFO] GOT-10k Test Stage4 inference only")
    print(f"[INFO] root              : {args.root}")
    print(f"[INFO] num sequences     : {len(seqs)}")
    print(f"[INFO] star param        : {args.star_param}")
    print(f"[INFO] tracker dataset   : {args.tracker_dataset_name}")
    print(f"[INFO] base checkpoint   : {args.base_checkpoint}")
    print(f"[INFO] stage4 checkpoint : {args.stage4_ckpt}")
    print(f"[INFO] star mode         : {args.star_mode}")
    print(f"[INFO] resume/skip       : {bool(args.resume and not args.force_rerun)}")
    print(f"[INFO] force rerun       : {bool(args.force_rerun)}")
    print(f"[INFO] out dir           : {args.out_dir}")
    print(f"[INFO] official root     : {Path(args.out_dir) / 'tracking_results'}")
    print(f"[INFO] tracker name      : {args.tracker_name}")
    print(f"[INFO] result param      : {args.result_param}")
    print(f"[INFO] bbox fmt          : {args.bbox_fmt}")
    print("=" * 100)

    out_root = Path(args.out_dir)

    # Legacy debug folders
    pred_dir = out_root / "predictions"
    peak_dir = out_root / "peak_info"
    pred_dir.mkdir(parents=True, exist_ok=True)
    peak_dir.mkdir(parents=True, exist_ok=True)

    # Official GOT-style root
    official_root = out_root / "tracking_results"

    success = []
    skipped = []
    failed = []
    sequence_outputs = {}

    for seq in seqs:
        try:
            seq_name, img_files, seq_init_bbox = load_got10k_test_sequence(
                root=args.root,
                seq_token=seq,
                split_hint=args.got_split,
                init_bbox=init_bbox,
            )

            if args.max_frames is not None:
                img_files = img_files[:args.max_frames]

            expected_frames = len(img_files)

            if args.resume and not args.force_rerun:
                complete, reason, existing_paths = check_existing_sequence_outputs(
                    official_root=official_root,
                    tracker_name=args.tracker_name,
                    param_name=args.result_param,
                    seq_name=seq_name,
                    expected_frames=expected_frames,
                )

                if complete:
                    print(f"[SKIP] {seq_name} already complete | frames={expected_frames}")
                    print(f"  official bbox : {existing_paths['official_bbox']}")
                    print(f"  official time : {existing_paths['official_time']}")

                    skipped.append(seq_name)
                    sequence_outputs[seq_name] = {
                        "status": "skipped_existing",
                        "official_bbox": existing_paths["official_bbox"],
                        "official_time": existing_paths["official_time"],
                        "frames": expected_frames,
                        "skip_reason": reason,
                    }
                    continue

                print(f"[RESUME] {seq_name} will run because existing output is not complete: {reason}")

            if stage4_ckpt is None:
                stage4_ckpt = make_startrack_test_ckpt(args.stage4_ckpt, converted_dir, "stage4")

            print("\n" + "=" * 100)
            print(f"[RUN] {seq_name} | frames={len(img_files)} | init_bbox={seq_init_bbox}")
            print("=" * 100)

            pred, times, peak_infos = run_tracker_on_sequence(
                param_name=args.star_param,
                dataset_name=args.tracker_dataset_name,
                img_files=img_files,
                init_bbox=seq_init_bbox,
                base_checkpoint=args.base_checkpoint,
                stage4_ckpt=stage4_ckpt,
                mode=args.star_mode,
            )

            # Legacy debug copies
            pred_path = pred_dir / f"{seq_name}.txt"
            time_debug_path = pred_dir / f"{seq_name}_time.txt"
            peak_path = peak_dir / f"{seq_name}_peak_info.csv"

            save_bboxes_legacy_comma(pred_path, pred)
            save_got10k_times(time_debug_path, times)
            save_peak_info(peak_path, peak_infos)

            # GOT-10k official style
            official_bbox_path, official_time_path = save_tracker_result_got_style(
                root_dir=official_root,
                tracker_name=args.tracker_name,
                param_name=args.result_param,
                seq_name=seq_name,
                bboxes=pred,
                times=times,
                bbox_fmt=args.bbox_fmt,
            )

            rerank_count = sum(int(x.get("rerank_used", False)) for x in peak_infos)
            update_count = sum(int(x.get("should_update", False)) for x in peak_infos)

            print(f"[DONE] {seq_name}")
            print(f"  legacy pred       : {pred_path}")
            print(f"  legacy time       : {time_debug_path}")
            print(f"  official bbox     : {official_bbox_path}")
            print(f"  official time     : {official_time_path}")
            print(f"  peak_info         : {peak_path}")
            print(f"  rerank_used_frames   : {rerank_count}")
            print(f"  should_update_frames : {update_count}")
            print(f"  avg_time_per_frame   : {float(np.mean(times)):.6f}s")
            print(f"  fps                  : {float(1.0 / max(np.mean(times), 1e-12)):.2f}")

            success.append(seq_name)
            sequence_outputs[seq_name] = {
                "status": "ran",
                "legacy_pred": str(pred_path),
                "legacy_time": str(time_debug_path),
                "official_bbox": str(official_bbox_path),
                "official_time": str(official_time_path),
                "peak_info": str(peak_path),
                "frames": len(img_files),
                "rerank_used_frames": rerank_count,
                "should_update_frames": update_count,
                "avg_time_per_frame": float(np.mean(times)),
                "fps": float(1.0 / max(np.mean(times), 1e-12)),
            }

        except Exception as e:
            print(f"[ERROR] sequence failed: {seq}, error={repr(e)}")
            failed.append((seq, repr(e)))

    summary = {
        "success": success,
        "skipped": skipped,
        "failed": failed,
        "num_success": len(success),
        "num_skipped": len(skipped),
        "num_failed": len(failed),
        "out_dir": str(out_root),
        "official_tracking_results": str(official_root),
        "tracker_name": args.tracker_name,
        "result_param": args.result_param,
        "sequence_outputs": sequence_outputs,
    }

    with open(out_root / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 100)
    print("[SUMMARY]")
    print(f"success: {len(success)}")
    print(f"skipped: {len(skipped)}")
    print(f"failed : {len(failed)}")
    print(f"legacy predictions saved to : {pred_dir}")
    print(f"official results saved to   : {official_root / args.tracker_name / args.result_param}")
    print(f"summary saved to            : {out_root / 'run_summary.json'}")
    print("=" * 100)


if __name__ == "__main__":
    main()