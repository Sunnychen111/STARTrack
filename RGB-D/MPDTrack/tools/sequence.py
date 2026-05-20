#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Actual inference comparison on one LaSOT sequence.

It runs:
    1. Baseline SUTrack
    2. STARTrack-SUTrack

Then saves:
    baseline_pred.txt
    startrack_pred.txt
    gt.txt
    frame_compare.csv
    summary.json

Metrics:
    - Success AUC
    - Precision @20px
    - Normalized Precision AUC
    - Mean IoU
    - Mean center error

BBox format:
    x, y, w, h
"""

import os
import sys
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch


DEFAULT_PROJECT_ROOT = "/home/cps/czl/STARTrack_v5"


# ============================================================
# Project setup
# ============================================================

def setup_project_root(project_root):
    project_root = os.path.abspath(project_root)

    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    return project_root


# ============================================================
# Robust IO
# ============================================================

def read_rgb(path):
    img = cv2.imread(path)

    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_txt_array(path, delimiter=None):
    """
    Robust txt loader.

    Supports:
        1,2,3,4
        1 2 3 4
        1\t2\t3\t4
        LaSOT one-line flags:
            0,0,0,0,0,0,...
    """
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


def load_lasot_sequence(root, seq_name):
    """
    Load one LaSOT sequence.

    Expected structure:
        root/class_name/seq_name/img/*.jpg
        root/class_name/seq_name/groundtruth.txt
        root/class_name/seq_name/full_occlusion.txt   optional
        root/class_name/seq_name/out_of_view.txt      optional
    """
    cls_name = seq_name.split("-")[0]

    seq_dir = os.path.join(root, cls_name, seq_name)
    img_dir = os.path.join(seq_dir, "img")
    gt_path = os.path.join(seq_dir, "groundtruth.txt")

    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Image dir not found: {img_dir}")

    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"GT file not found: {gt_path}")

    img_files = sorted([
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ])

    gt = load_txt_array(gt_path, delimiter=",")[:, :4].astype(np.float32)

    n = min(len(img_files), len(gt))
    img_files = img_files[:n]
    gt = gt[:n]

    valid = np.ones(n, dtype=bool)

    # LaSOT flags are usually one-line comma-separated txt files.
    full_occ_path = os.path.join(seq_dir, "full_occlusion.txt")
    out_view_path = os.path.join(seq_dir, "out_of_view.txt")

    if os.path.isfile(full_occ_path):
        full_occ = load_txt_array(full_occ_path, delimiter=",").reshape(-1)[:n]
        valid &= full_occ == 0

    if os.path.isfile(out_view_path):
        out_view = load_txt_array(out_view_path, delimiter=",").reshape(-1)[:n]
        valid &= out_view == 0

    valid &= gt[:, 2] > 0
    valid &= gt[:, 3] > 0

    return img_files, gt, valid


def save_bboxes(path, bboxes):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savetxt(
        str(path),
        bboxes,
        fmt="%.4f",
        delimiter=",",
    )


# ============================================================
# Tracker running
# ============================================================

def patch_params(params):
    """
    Make TrackerParams compatible with direct script running.
    """
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


def run_tracker(
    param_name,
    dataset_name,
    img_files,
    gt,
    checkpoint_override=None,
):
    """
    Run one tracker on one sequence.

    param_name:
        sutrack_b224
        sutrack_b224_startrack

    checkpoint_override:
        Force both baseline and STARTrack to use the same original SUTrack
        checkpoint, e.g. SUTRACK_ep0180.pth.tar.
    """
    from lib.test.parameter.sutrack import parameters
    from lib.test.tracker.sutrack import get_tracker_class

    TrackerClass = get_tracker_class()

    params = parameters(param_name)
    params = patch_params(params)

    if checkpoint_override is not None:
        params.checkpoint = checkpoint_override
        print(f"[INFO] Override checkpoint for {param_name}: {params.checkpoint}")

    tracker = TrackerClass(params, dataset_name)

    first_img = read_rgb(img_files[0])
    tracker.initialize(first_img, {"init_bbox": gt[0].tolist()})

    pred_bboxes = np.zeros_like(gt, dtype=np.float32)
    pred_bboxes[0] = gt[0]

    peak_infos = []

    for i in range(1, len(img_files)):
        img = read_rgb(img_files[i])

        with torch.no_grad():
            out = tracker.track(img, info={})

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
        })

        if i % 500 == 0:
            print(f"[{param_name}] processed {i}/{len(img_files)} frames")

    return pred_bboxes, peak_infos


# ============================================================
# Metrics
# ============================================================

def bbox_iou_xywh(pred, gt):
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)

    px1 = pred[:, 0]
    py1 = pred[:, 1]
    px2 = pred[:, 0] + pred[:, 2]
    py2 = pred[:, 1] + pred[:, 3]

    gx1 = gt[:, 0]
    gy1 = gt[:, 1]
    gx2 = gt[:, 0] + gt[:, 2]
    gy2 = gt[:, 1] + gt[:, 3]

    ix1 = np.maximum(px1, gx1)
    iy1 = np.maximum(py1, gy1)
    ix2 = np.minimum(px2, gx2)
    iy2 = np.minimum(py2, gy2)

    iw = np.maximum(ix2 - ix1, 0.0)
    ih = np.maximum(iy2 - iy1, 0.0)

    inter = iw * ih

    area_p = np.maximum(pred[:, 2], 0.0) * np.maximum(pred[:, 3], 0.0)
    area_g = np.maximum(gt[:, 2], 0.0) * np.maximum(gt[:, 3], 0.0)

    union = area_p + area_g - inter

    return inter / np.maximum(union, 1e-12)


def center_error_xywh(pred, gt):
    pcx = pred[:, 0] + 0.5 * pred[:, 2]
    pcy = pred[:, 1] + 0.5 * pred[:, 3]

    gcx = gt[:, 0] + 0.5 * gt[:, 2]
    gcy = gt[:, 1] + 0.5 * gt[:, 3]

    return np.sqrt((pcx - gcx) ** 2 + (pcy - gcy) ** 2)


def normalized_center_error_xywh(pred, gt):
    center_err = center_error_xywh(pred, gt)

    # Common approximation for normalized precision:
    # normalize center error by sqrt(gt_w * gt_h)
    norm = np.sqrt(np.maximum(gt[:, 2] * gt[:, 3], 1e-12))

    return center_err / norm


def success_auc(iou):
    thresholds = np.linspace(0.0, 1.0, 21)
    success = np.array(
        [(iou >= t).mean() for t in thresholds],
        dtype=np.float32,
    )

    return float(success.mean()), thresholds, success


def precision_at_20(center_err):
    return float((center_err <= 20.0).mean())


def norm_precision_auc(norm_err):
    thresholds = np.linspace(0.0, 0.5, 51)

    precision = np.array(
        [(norm_err <= t).mean() for t in thresholds],
        dtype=np.float32,
    )

    return float(precision.mean()), thresholds, precision


def compute_metrics(pred, gt, valid):
    pred_v = pred[valid]
    gt_v = gt[valid]

    iou = bbox_iou_xywh(pred_v, gt_v)
    center_err = center_error_xywh(pred_v, gt_v)
    norm_err = normalized_center_error_xywh(pred_v, gt_v)

    auc, _, _ = success_auc(iou)
    p20 = precision_at_20(center_err)
    pnorm, _, _ = norm_precision_auc(norm_err)

    return {
        "valid_frames": int(valid.sum()),
        "success_auc": auc,
        "precision_20": p20,
        "norm_precision_auc": pnorm,
        "mean_iou": float(iou.mean()),
        "mean_center_error": float(center_err.mean()),
        "median_center_error": float(np.median(center_err)),
    }


def compare_frame_level(base_pred, star_pred, gt, valid, out_csv):
    base_iou = bbox_iou_xywh(base_pred, gt)
    star_iou = bbox_iou_xywh(star_pred, gt)
    iou_gain = star_iou - base_iou

    base_ce = center_error_xywh(base_pred, gt)
    star_ce = center_error_xywh(star_pred, gt)

    # Positive center_error_gain means STARTrack has smaller center error.
    center_error_gain = base_ce - star_ce

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(
            "frame,valid,"
            "base_x,base_y,base_w,base_h,"
            "star_x,star_y,star_w,star_h,"
            "gt_x,gt_y,gt_w,gt_h,"
            "base_iou,star_iou,iou_gain,"
            "base_center_error,star_center_error,center_error_gain\n"
        )

        for i in range(len(gt)):
            row = [
                i,
                int(valid[i]),
                *base_pred[i].tolist(),
                *star_pred[i].tolist(),
                *gt[i].tolist(),
                float(base_iou[i]),
                float(star_iou[i]),
                float(iou_gain[i]),
                float(base_ce[i]),
                float(star_ce[i]),
                float(center_error_gain[i]),
            ]

            f.write(",".join(map(str, row)) + "\n")

    corrected = int(((iou_gain > 0.01) & valid).sum())
    hurt = int(((iou_gain < -0.01) & valid).sum())
    same = int(((np.abs(iou_gain) <= 0.01) & valid).sum())

    return {
        "corrected_frames_iou": corrected,
        "hurt_frames_iou": hurt,
        "same_frames_iou": same,
        "csv": str(out_csv),
    }


def print_metric_table(base_metrics, star_metrics):
    keys = [
        "success_auc",
        "precision_20",
        "norm_precision_auc",
        "mean_iou",
        "mean_center_error",
        "median_center_error",
    ]

    print("\n" + "=" * 100)
    print("[LaSOT-style metrics]")
    print("-" * 100)
    print(
        f"{'metric':28s} "
        f"{'baseline':>12s} "
        f"{'startrack':>12s} "
        f"{'delta':>12s}"
    )

    for k in keys:
        b = base_metrics[k]
        s = star_metrics[k]
        delta = s - b

        print(f"{k:28s} {b:12.6f} {s:12.6f} {delta:12.6f}")

    print("-" * 100)
    print(f"valid_frames: {base_metrics['valid_frames']}")
    print("=" * 100)


def save_peak_info(path, peak_infos):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "frame,rerank_used,should_update,selected_idx,"
            "target_prob,identity_margin,ambiguity_ratio,"
            "history_len,update_history_count\n"
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
                f"{item.get('update_history_count', 0)}\n"
            )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--project-root", type=str, default=DEFAULT_PROJECT_ROOT)

    parser.add_argument("--dataset", type=str, default="lasot")
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--seq", type=str, required=True)

    parser.add_argument("--baseline-param", type=str, default="sutrack_b224")
    parser.add_argument("--star-param", type=str, default="sutrack_b224_startrack")

    parser.add_argument(
        "--base-checkpoint",
        type=str,
        required=True,
        help="Original SUTrack checkpoint, e.g. SUTRACK_ep0180.pth.tar",
    )

    parser.add_argument("--max-frames", type=int, default=None)

    parser.add_argument(
        "--out-dir",
        type=str,
        default="sequence_eval_outputs",
    )

    args = parser.parse_args()

    setup_project_root(args.project_root)

    if args.dataset.lower() != "lasot":
        raise ValueError("This script currently supports LaSOT only.")

    img_files, gt, valid = load_lasot_sequence(args.root, args.seq)

    if args.max_frames is not None:
        img_files = img_files[:args.max_frames]
        gt = gt[:args.max_frames]
        valid = valid[:args.max_frames]

    out_dir = Path(args.out_dir) / args.seq
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("[INFO] LaSOT single-sequence actual inference evaluation")
    print(f"[INFO] project_root     : {args.project_root}")
    print(f"[INFO] sequence         : {args.seq}")
    print(f"[INFO] frames           : {len(img_files)}")
    print(f"[INFO] valid frames     : {int(valid.sum())}")
    print(f"[INFO] baseline param   : {args.baseline_param}")
    print(f"[INFO] star param       : {args.star_param}")
    print(f"[INFO] base checkpoint  : {args.base_checkpoint}")
    print(f"[INFO] output dir       : {out_dir}")
    print("=" * 100)

    print("\n[RUN] Baseline SUTrack actual inference...")
    base_pred, base_peak_info = run_tracker(
        param_name=args.baseline_param,
        dataset_name=args.dataset,
        img_files=img_files,
        gt=gt,
        checkpoint_override=args.base_checkpoint,
    )

    print("\n[RUN] STARTrack-SUTrack actual inference...")
    star_pred, star_peak_info = run_tracker(
        param_name=args.star_param,
        dataset_name=args.dataset,
        img_files=img_files,
        gt=gt,
        checkpoint_override=args.base_checkpoint,
    )

    base_txt = out_dir / "baseline_pred.txt"
    star_txt = out_dir / "startrack_pred.txt"
    gt_txt = out_dir / "gt.txt"
    valid_txt = out_dir / "valid.txt"

    save_bboxes(base_txt, base_pred)
    save_bboxes(star_txt, star_pred)
    save_bboxes(gt_txt, gt)
    np.savetxt(str(valid_txt), valid.astype(np.int32), fmt="%d", delimiter=",")

    save_peak_info(out_dir / "startrack_peak_info.csv", star_peak_info)

    base_metrics = compute_metrics(base_pred, gt, valid)
    star_metrics = compute_metrics(star_pred, gt, valid)

    print_metric_table(base_metrics, star_metrics)

    frame_compare = compare_frame_level(
        base_pred=base_pred,
        star_pred=star_pred,
        gt=gt,
        valid=valid,
        out_csv=out_dir / "frame_compare.csv",
    )

    rerank_count = sum(int(x.get("rerank_used", False)) for x in star_peak_info)
    update_count = sum(int(x.get("should_update", False)) for x in star_peak_info)

    summary = {
        "sequence": args.seq,
        "frames": len(img_files),
        "valid_frames": int(valid.sum()),
        "baseline_metrics": base_metrics,
        "startrack_metrics": star_metrics,
        "delta": {
            "success_auc": star_metrics["success_auc"] - base_metrics["success_auc"],
            "precision_20": star_metrics["precision_20"] - base_metrics["precision_20"],
            "norm_precision_auc": star_metrics["norm_precision_auc"] - base_metrics["norm_precision_auc"],
            "mean_iou": star_metrics["mean_iou"] - base_metrics["mean_iou"],
            "mean_center_error": star_metrics["mean_center_error"] - base_metrics["mean_center_error"],
            "median_center_error": star_metrics["median_center_error"] - base_metrics["median_center_error"],
        },
        "rerank_used_frames": int(rerank_count),
        "should_update_frames": int(update_count),
        "frame_compare": frame_compare,
        "files": {
            "baseline_pred": str(base_txt),
            "startrack_pred": str(star_txt),
            "gt": str(gt_txt),
            "valid": str(valid_txt),
            "frame_compare_csv": str(out_dir / "frame_compare.csv"),
            "startrack_peak_info": str(out_dir / "startrack_peak_info.csv"),
        },
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 100)
    print("[OUTPUT FILES]")
    print(f"baseline pred       : {base_txt}")
    print(f"startrack pred      : {star_txt}")
    print(f"gt                  : {gt_txt}")
    print(f"valid mask          : {valid_txt}")
    print(f"frame compare       : {out_dir / 'frame_compare.csv'}")
    print(f"startrack peak info : {out_dir / 'startrack_peak_info.csv'}")
    print(f"summary             : {out_dir / 'summary.json'}")
    print("-" * 100)
    print(f"rerank_used_frames   : {rerank_count}")
    print(f"should_update_frames : {update_count}")
    print(f"corrected_iou_frames : {frame_compare['corrected_frames_iou']}")
    print(f"hurt_iou_frames      : {frame_compare['hurt_frames_iou']}")
    print(f"same_iou_frames      : {frame_compare['same_frames_iou']}")
    print("=" * 100)


if __name__ == "__main__":
    main()