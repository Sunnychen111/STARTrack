#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Three-way actual inference comparison.

It runs:
    1. Baseline SUTrack
    2. STARTrack with offline reranker checkpoint
    3. STARTrack with Stage4 online-joint checkpoint

Supports:
    - LaSOT
    - GOT-10k train/val style folders with groundtruth.txt

Outputs per sequence:
    baseline_pred.txt
    offline_pred.txt
    stage4_pred.txt
    gt.txt
    valid.txt
    frame_compare.csv
    offline_peak_info.csv
    stage4_peak_info.csv
    summary.json

Aggregate outputs:
    aggregate_summary.csv
    aggregate_summary.json

BBox format:
    x, y, w, h
"""

import os
import sys
import json
import argparse
import traceback
import builtins
import pathlib
import importlib
import gc
from pathlib import Path

import cv2
import numpy as np
import multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass
import torch
import yaml
import gc
import ctypes


DEFAULT_PROJECT_ROOT = "/home/cps/czl/STARTrack_v5"


# ============================================================
# Project setup
# ============================================================

def setup_project_root(project_root):
    project_root = os.path.abspath(project_root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return project_root


def sanity_check_runtime(*paths):
    assert builtins.list is list, "builtins.list has been polluted"
    assert callable(builtins.isinstance), "builtins.isinstance is not callable"
    assert pathlib.Path is Path, "pathlib.Path binding has been polluted"
    p = pathlib.Path("/tmp") / "abc"
    assert isinstance(p, pathlib.Path), "pathlib.Path did not produce a Path"
    assert p.parent is not None, "pathlib.Path.parent is broken"
    data = yaml.safe_load("A: 1\n")
    assert data["A"] == 1, "yaml.safe_load sanity check failed"
    for path in paths:
        if path is not None:
            str(path)



def check_torch_functional(tag=""):
    """Detect runtime pollution of torch.nn.functional.linear.

    This is intentionally passive: it does not monkey-patch torch modules.
    It catches the class of errors where F.linear becomes a Parameter/tuple/etc.
    """
    import torch.nn as nn
    import torch.nn.functional as F

    if not callable(F.linear):
        raise RuntimeError(
            f"[SANITY FAILED][{tag}] torch.nn.functional.linear corrupted: "
            f"type={type(F.linear)}, value={F.linear}"
        )

    # A tiny nn.Linear forward is a stronger sanity check than callable() alone.
    x = torch.randn(1, 4)
    m = nn.Linear(4, 2)
    _ = m(x)


def cleanup_after_tracker_run():
    """Best-effort cleanup between baseline/offline/stage4 and sequences."""
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            torch.cuda.empty_cache()


# ============================================================
# Robust IO
# ============================================================

def read_rgb(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


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


def save_bboxes(path, bboxes):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(path), bboxes, fmt="%.4f", delimiter=",")


def list_images(img_dir):
    img_dir = Path(img_dir)
    files = sorted([
        p for p in img_dir.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
    ])
    if len(files) == 0:
        raise FileNotFoundError(f"No image files found in: {img_dir}")
    return files


# ============================================================
# Dataset loaders
# ============================================================

def load_lasot_sequence(root, seq_name):
    """
    Expected:
        root/class_name/seq_name/img/*.jpg
        root/class_name/seq_name/groundtruth.txt

    Return:
        img_files, gt, valid, seq_dir

    Note:
        valid here is only a convenience mask for frame-level comparison.
        The official/aligned LaSOT metric is computed later by
        evaluate_sequence_aligned_metrics(), which reads target_visible
        from out_of_view.txt and full_occlusion.txt.
    """
    root = Path(root)
    cls_name = seq_name.split("-")[0]

    seq_dir = root / cls_name / seq_name
    img_dir = seq_dir / "img"
    gt_path = seq_dir / "groundtruth.txt"

    if not img_dir.is_dir():
        raise FileNotFoundError(f"LaSOT image dir not found: {img_dir}")
    if not gt_path.is_file():
        raise FileNotFoundError(f"LaSOT GT file not found: {gt_path}")

    img_files = list_images(img_dir)
    gt = load_txt_array(gt_path, delimiter=",")[:, :4].astype(np.float32)

    n = min(len(img_files), len(gt))
    img_files = img_files[:n]
    gt = gt[:n]

    target_visible = derive_target_visible(seq_dir, n=n)
    if target_visible is None:
        valid = np.ones(n, dtype=bool)
    else:
        valid = target_visible.astype(bool)

    valid &= gt[:, 2] > 0
    valid &= gt[:, 3] > 0

    return img_files, gt, valid, str(seq_dir)


def got10k_candidates(root, seq_token, split_hint="train"):
    """
    Supports:
        seq_token = GOT-10k_Train_000080
        seq_token = 000079  -> GOT-10k_Train_000080 by your convention
        seq_token = 000080  -> also try direct 000080 variants

    Your convention:
        GOT-10k_Train_000080 is recorded as 000079.
    """
    root = Path(root)

    if seq_token.startswith("GOT-10k") or seq_token.startswith("GOT-10K"):
        return [root / seq_token]

    if seq_token.isdigit() and len(seq_token) == 6:
        zero_based = int(seq_token)
        one_based = zero_based + 1

        split_hint_lower = split_hint.lower()

        prefixes = []
        if "train" in split_hint_lower:
            prefixes.extend(["GOT-10k_Train", "GOT-10K_Train"])
        elif "val" in split_hint_lower:
            prefixes.extend(["GOT-10k_Val", "GOT-10K_Val"])
        elif "test" in split_hint_lower:
            prefixes.extend(["GOT-10k_Test", "GOT-10K_Test"])
        else:
            prefixes.extend([
                "GOT-10k_Train", "GOT-10K_Train",
                "GOT-10k_Val", "GOT-10K_Val",
                "GOT-10k_Test", "GOT-10K_Test",
            ])

        cands = []
        for prefix in prefixes:
            cands.append(root / f"{prefix}_{one_based:06d}")
            cands.append(root / f"{prefix}_{zero_based:06d}")
        return cands

    return [root / seq_token]


def load_got10k_sequence(root, seq_token, split_hint="train"):
    """
    Expected:
        root/GOT-10k_Train_000080/*.jpg
        root/GOT-10k_Train_000080/groundtruth.txt

    If seq_token is 000079, it first tries GOT-10k_Train_000080.
    """
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

    gt_path = seq_dir / "groundtruth.txt"
    if not gt_path.is_file():
        raise FileNotFoundError(f"GOT-10k GT file not found: {gt_path}")

    img_files = list_images(seq_dir)
    gt = load_txt_array(gt_path, delimiter=",")[:, :4].astype(np.float32)

    n = min(len(img_files), len(gt))
    img_files = img_files[:n]
    gt = gt[:n]

    valid = np.ones(n, dtype=bool)
    valid &= gt[:, 2] > 0
    valid &= gt[:, 3] > 0

    return img_files, gt, valid


def load_sequence(dataset, root, seq, got_split="train"):
    dataset_l = dataset.lower()

    if dataset_l == "lasot":
        return load_lasot_sequence(root, seq)

    if dataset_l in ["got10k", "got-10k", "got10k_train", "got10k_val"]:
        img_files, gt, valid = load_got10k_sequence(root, seq, split_hint=got_split)
        return img_files, gt, valid, None

    raise ValueError(f"Unsupported dataset: {dataset}")


# ============================================================
# Checkpoint compatibility
# ============================================================

def make_startrack_test_ckpt(src_path, out_dir, tag):
    """
    Tracker usually expects offline format:
        ckpt["model"]

    Stage4 checkpoint may contain:
        ckpt["disambiguator"]

    This function converts Stage4 format into test-compatible format.
    """
    if src_path is None:
        return None

    src_path = Path(src_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(str(src_path), map_location="cpu")

    if "model" in ckpt:
        return str(src_path)

    if "disambiguator" not in ckpt:
        raise KeyError(
            f"Cannot find 'model' or 'disambiguator' in {src_path}. "
            f"keys={list(ckpt.keys())}"
        )

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
    return str(dst)


# ============================================================
# Tracker running
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


def set_nested_cfg_value(obj, keys, value):
    cur = obj
    for k in keys[:-1]:
        if not hasattr(cur, k):
            return False
        cur = getattr(cur, k)
    if hasattr(cur, keys[-1]):
        setattr(cur, keys[-1], value)
        return True
    return False



def fresh_sutrack_parameters(param_name):
    """Load SUTrack parameters from a fresh config module each tracker run.

    SUTrack uses a global cfg in lib.config.sutrack.config. In compare_three.py we
    construct baseline/offline/stage4 trackers repeatedly in the same Python
    process. If cfg is mutated by one tracker, the next tracker/sequence can inherit
    polluted state. Reloading config + parameter before each tracker construction
    avoids this cross-run leakage.
    """
    import lib.config.sutrack.config as sutrack_config
    import lib.test.parameter.sutrack as sutrack_param

    importlib.reload(sutrack_config)
    sutrack_param = importlib.reload(sutrack_param)

    return sutrack_param.parameters(param_name)


def override_startrack_params(params, star_ckpt=None, mode=None):
    """
    Try multiple possible locations, because different SUTrack/STARTrack
    versions read config in different ways.
    """
    if star_ckpt is not None:
        star_ckpt = str(star_ckpt)
        # Common direct attributes
        params.startrack_ckpt = star_ckpt
        params.reranker_ckpt = star_ckpt
        params.STARTRACK_CKPT = star_ckpt

        # Config object
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


def run_tracker(
    param_name,
    dataset_name,
    img_files,
    gt,
    base_checkpoint,
    star_ckpt=None,
    mode=None,
):
    """Run one tracker instance on one sequence.

    Important implementation detail:
    every call creates params from a freshly reloaded SUTrack config module.
    This prevents baseline/offline/stage4 or different sequences from sharing a
    mutated global cfg in one Python process.
    """
    check_torch_functional("run_tracker/start")

    # Import tracker after project root has been injected. Reloading the tracker
    # module here is intentionally avoided by default because reloading heavy model
    # modules can invalidate class identities. The config/parameter modules are the
    # important mutable globals to reset.
    from lib.test.tracker.sutrack import get_tracker_class

    check_torch_functional("after import get_tracker_class")
    TrackerClass = get_tracker_class()
    check_torch_functional("after get_tracker_class")

    params = fresh_sutrack_parameters(param_name)
    check_torch_functional("after fresh_sutrack_parameters")

    params = patch_params(params)
    check_torch_functional("after patch_params")

    # Force all variants to use the same original SUTrack base checkpoint.
    if base_checkpoint is not None:
        base_checkpoint = str(base_checkpoint)
        params.checkpoint = base_checkpoint
        if hasattr(params, "cfg") and hasattr(params.cfg, "TEST"):
            try:
                params.cfg.TEST.CHECKPOINT = base_checkpoint
            except Exception:
                pass

    params = override_startrack_params(params, star_ckpt=star_ckpt, mode=mode)
    check_torch_functional("after override_startrack_params")

    tracker = None
    try:
        tracker = TrackerClass(params, dataset_name)
        check_torch_functional("after TrackerClass init")

        first_img = read_rgb(img_files[0])
        tracker.initialize(first_img, {"init_bbox": gt[0].tolist()})
        check_torch_functional("after tracker.initialize")

        pred_bboxes = np.zeros_like(gt, dtype=np.float32)
        pred_bboxes[0] = gt[0]

        peak_infos = []

        sanity_every_frame = os.environ.get("STARTRACK_SANITY_EVERY_FRAME", "0") == "1"

        for i in range(1, len(img_files)):
            img = read_rgb(img_files[i])

            if sanity_every_frame or i <= 3:
                check_torch_functional(f"before tracker.track frame={i}")

            with torch.no_grad():
                out = tracker.track(img, info={})

            if sanity_every_frame or i <= 3:
                check_torch_functional(f"after tracker.track frame={i}")

            pred_bboxes[i] = np.array(out["target_bbox"], dtype=np.float32)

            peak_info = out.get("peak_info", {})
            peak_infos.append({
                "frame": int(i),
                "rerank_used": bool(peak_info.get("rerank_used", False)),
                "should_update": bool(peak_info.get("should_update", False)),
                "should_update_history": bool(peak_info.get("should_update_history", peak_info.get("should_update", False))),
                "selected_idx": int(peak_info.get("selected_idx", -1)),
                "selected_idx_raw": int(peak_info.get("selected_idx_raw", peak_info.get("selected_idx", -1))),
                "selected_idx_final": int(peak_info.get("selected_idx_final", peak_info.get("selected_idx", -1))),
                "target_prob": float(peak_info.get("target_prob", 0.0)),
                "identity_margin": float(peak_info.get("identity_margin", 0.0)),
                "ambiguity_ratio": float(peak_info.get("ambiguity_ratio", 0.0)),
                "drift_gate_enable": bool(peak_info.get("drift_gate_enable", False)),
                "drift_gate_allow": bool(peak_info.get("drift_gate_allow", True)),
                "drift_gate_reason": str(peak_info.get("drift_gate_reason", "")),
                "drift_gate_cooldown_before": int(peak_info.get("drift_gate_cooldown_before", 0)),
                "drift_gate_cooldown_after": int(peak_info.get("drift_gate_cooldown_after", 0)),
                "drift_gate_baseline_center_jump": float(peak_info.get("drift_gate_baseline_center_jump", 0.0)),
                "drift_gate_state_center_jump": float(peak_info.get("drift_gate_state_center_jump", 0.0)),
                "drift_gate_baseline_iou": float(peak_info.get("drift_gate_baseline_iou", 1.0)),
                "drift_gate_width_ratio": float(peak_info.get("drift_gate_width_ratio", 1.0)),
                "drift_gate_height_ratio": float(peak_info.get("drift_gate_height_ratio", 1.0)),
                "drift_gate_area_ratio": float(peak_info.get("drift_gate_area_ratio", 1.0)),
                "drift_gate_bypass": bool(peak_info.get("drift_gate_bypass", False)),
                "history_len": int(peak_info.get("history_len", -1)),
                "update_history_count": int(peak_info.get("update_history_count", 0)),
            })

        return pred_bboxes, peak_infos

    finally:
        # Release references before constructing the next tracker/sequence.
        try:
            del tracker
        except Exception:
            pass
        try:
            del params
        except Exception:
            pass
        cleanup_after_tracker_run()


# ============================================================
# Metrics
# ============================================================
# The LaSOT metrics below are aligned with toolkit_label_final.py:
#   - center: x + 0.5 * (w - 1)
#   - IoU: x+w-1/y+h-1 with +1 intersection size
#   - first frame is forced to GT
#   - LaSOT target_visible = ~(out_of_view | full_occlusion)
#   - AUC/P/PNorm are percentage values in [0, 100]
#   - overall aggregation averages curves first, then computes AUC/P/PNorm


def load_status_array(path):
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    try:
        arr = np.loadtxt(str(path), delimiter=",", dtype=np.float64)
    except Exception:
        try:
            arr = np.loadtxt(str(path), delimiter="\t", dtype=np.float64)
        except Exception:
            arr = np.loadtxt(str(path), dtype=np.float64)
    return np.asarray(arr).reshape(-1)


def find_status_files(seq_dir):
    seq_dir = Path(seq_dir)
    oov_p = seq_dir / "out_of_view.txt"
    occ_p = seq_dir / "full_occlusion.txt"
    return (oov_p if oov_p.is_file() else None,
            occ_p if occ_p.is_file() else None)


def derive_target_visible(seq_dir, n=None):
    """
    LaSOT target_visible = ~(out_of_view | full_occlusion), aligned with toolkit_label_final.py.
    """
    if seq_dir is None:
        return None

    oov_file, occ_file = find_status_files(seq_dir)
    oov = load_status_array(oov_file)
    occ = load_status_array(occ_file)

    if oov is None and occ is None:
        return None
    if oov is None:
        oov = np.zeros_like(occ)
    if occ is None:
        occ = np.zeros_like(oov)

    m = min(len(oov), len(occ))
    visible = ~((oov[:m] > 0.5) | (occ[:m] > 0.5))

    if n is not None:
        if len(visible) >= n:
            visible = visible[:n]
        else:
            visible = np.concatenate([visible, np.ones(n - len(visible), dtype=bool)], axis=0)

    return visible.astype(np.uint8)


def calc_err_center_aligned(pred_bb, anno_bb, normalized=False):
    pred_center = pred_bb[:, :2] + 0.5 * (pred_bb[:, 2:] - 1.0)
    anno_center = anno_bb[:, :2] + 0.5 * (anno_bb[:, 2:] - 1.0)

    if normalized:
        pred_center = pred_center / anno_bb[:, 2:]
        anno_center = anno_center / anno_bb[:, 2:]

    return np.sqrt(((pred_center - anno_center) ** 2).sum(axis=1))


def calc_iou_overlap_aligned(pred_bb, anno_bb):
    pred_bb = pred_bb.astype(np.float64)
    anno_bb = anno_bb.astype(np.float64)

    tl = np.maximum(pred_bb[:, :2], anno_bb[:, :2])
    br = np.minimum(
        pred_bb[:, :2] + pred_bb[:, 2:] - 1.0,
        anno_bb[:, :2] + anno_bb[:, 2:] - 1.0,
    )
    sz = np.clip(br - tl + 1.0, a_min=0.0, a_max=None)

    intersection = sz[:, 0] * sz[:, 1]
    union = pred_bb[:, 2] * pred_bb[:, 3] + anno_bb[:, 2] * anno_bb[:, 3] - intersection

    overlap = np.zeros_like(intersection, dtype=np.float64)
    valid = union > 0
    overlap[valid] = intersection[valid] / union[valid]
    return overlap


def calc_seq_err_robust_aligned(pred_bb, anno_bb, dataset="lasot", target_visible=None):
    """
    Numpy-aligned implementation of toolkit_label_final.py / extract_results.py logic.
    """
    pred_bb = pred_bb.copy().astype(np.float64)
    anno_bb = anno_bb.copy().astype(np.float64)

    if np.isnan(pred_bb).any() or (pred_bb[:, 2:] < 0.0).any():
        raise Exception("Error: Invalid results")

    if np.isnan(anno_bb).any():
        if dataset == "uav":
            pass
        else:
            raise Exception("Warning: NaNs in annotation")

    # If predicted width/height is zero, copy previous prediction.
    if (pred_bb[:, 2:] == 0.0).any():
        for i in range(1, pred_bb.shape[0]):
            if (pred_bb[i, 2:] == 0.0).any() and not np.isnan(anno_bb[i, :]).any():
                pred_bb[i, :] = pred_bb[i - 1, :]

    # Length alignment.
    if pred_bb.shape[0] != anno_bb.shape[0]:
        if dataset == "lasot":
            if pred_bb.shape[0] > anno_bb.shape[0]:
                pred_bb = pred_bb[:anno_bb.shape[0], :]
            else:
                raise Exception("Mis-match in tracker prediction and GT lengths")
        else:
            if pred_bb.shape[0] > anno_bb.shape[0]:
                pred_bb = pred_bb[:anno_bb.shape[0], :]
            else:
                pad = np.zeros((anno_bb.shape[0] - pred_bb.shape[0], 4), dtype=pred_bb.dtype)
                pred_bb = np.concatenate((pred_bb, pad), axis=0)

    # First frame is GT.
    pred_bb[0, :] = anno_bb[0, :]

    if target_visible is not None:
        target_visible = target_visible.astype(bool).reshape(-1)
        if len(target_visible) > anno_bb.shape[0]:
            target_visible = target_visible[:anno_bb.shape[0]]
        elif len(target_visible) < anno_bb.shape[0]:
            target_visible = np.concatenate(
                [target_visible, np.ones(anno_bb.shape[0] - len(target_visible), dtype=bool)], axis=0
            )
        valid = ((anno_bb[:, 2:] > 0.0).sum(axis=1) == 2) & target_visible
    else:
        valid = ((anno_bb[:, 2:] > 0.0).sum(axis=1) == 2)

    err_center = calc_err_center_aligned(pred_bb, anno_bb, normalized=False)
    err_center_normalized = calc_err_center_aligned(pred_bb, anno_bb, normalized=True)
    err_overlap = calc_iou_overlap_aligned(pred_bb, anno_bb)

    if dataset in ["uav"]:
        err_center[~valid] = -1.0
    else:
        err_center[~valid] = float("Inf")

    err_center_normalized[~valid] = -1.0
    err_overlap[~valid] = -1.0

    if dataset == "lasot" and target_visible is not None:
        err_center_normalized[~target_visible] = float("Inf")
        err_center[~target_visible] = float("Inf")

    if np.isnan(err_overlap).any():
        raise Exception("Nans in calculated overlap")

    return err_overlap, err_center, err_center_normalized, valid


def evaluate_sequence_aligned_metrics(
    pred_boxes,
    gt_boxes,
    seq_dir,
    dataset="lasot",
    exclude_invalid_frames=False,
):
    """
    toolkit_label_final.py style metrics.

    Returns percentage-style scalar metrics:
        success_auc / precision_20 / norm_precision_auc in [0, 100].
    Also returns success/precision/norm-precision curves for overall aggregation.
    """
    target_visible = derive_target_visible(seq_dir, n=len(gt_boxes)) if dataset == "lasot" else None

    err_overlap, err_center, err_center_norm, valid = calc_seq_err_robust_aligned(
        pred_boxes,
        gt_boxes,
        dataset=dataset,
        target_visible=target_visible,
    )

    threshold_set_overlap = np.arange(0.0, 1.0 + 0.05, 0.05, dtype=np.float64)
    threshold_set_center = np.arange(0, 51, 1, dtype=np.float64)
    threshold_set_center_norm = np.arange(0, 51, 1, dtype=np.float64) / 100.0

    if exclude_invalid_frames:
        seq_length = int(valid.astype(np.int64).sum())
    else:
        seq_length = gt_boxes.shape[0]

    if seq_length <= 0:
        raise Exception("Seq length zero")

    succ_curve = np.array([(err_overlap > t).sum() / seq_length for t in threshold_set_overlap], dtype=np.float64)
    prec_curve = np.array([(err_center <= t).sum() / seq_length for t in threshold_set_center], dtype=np.float64)
    nprec_curve = np.array([(err_center_norm <= t).sum() / seq_length for t in threshold_set_center_norm], dtype=np.float64)

    finite_center = err_center[np.isfinite(err_center) & valid]
    avg_overlap = err_overlap[valid].mean() if valid.any() else 0.0
    mean_center_error = finite_center.mean() if finite_center.size > 0 else float("inf")
    median_center_error = np.median(finite_center) if finite_center.size > 0 else float("inf")

    return {
        "num_frames": int(gt_boxes.shape[0]),
        "valid_frames": int(valid.astype(np.int64).sum()),
        "success_auc": float(succ_curve.mean() * 100.0),
        "precision_20": float(prec_curve[20] * 100.0),
        "norm_precision_auc": float(nprec_curve[20] * 100.0),
        "mean_iou": float(avg_overlap * 100.0),
        "mean_center_error": float(mean_center_error),
        "median_center_error": float(median_center_error),
        "succ_curve": succ_curve,
        "prec_curve": prec_curve,
        "nprec_curve": nprec_curve,
        "valid_mask": valid,
        "target_visible_used": target_visible is not None,
    }


def summarize_aligned_metrics(seq_metrics_list):
    """
    Overall aggregation aligned with toolkit_label_final.py:
        average curves first, then compute AUC/P/PNorm.
    """
    if len(seq_metrics_list) == 0:
        return {
            "num_sequences": 0,
            "success_auc": 0.0,
            "precision_20": 0.0,
            "norm_precision_auc": 0.0,
        }

    avg_succ_curve = np.mean(np.stack([m["succ_curve"] for m in seq_metrics_list], axis=0), axis=0)
    avg_prec_curve = np.mean(np.stack([m["prec_curve"] for m in seq_metrics_list], axis=0), axis=0)
    avg_nprec_curve = np.mean(np.stack([m["nprec_curve"] for m in seq_metrics_list], axis=0), axis=0)

    return {
        "num_sequences": int(len(seq_metrics_list)),
        "success_auc": float(avg_succ_curve.mean() * 100.0),
        "precision_20": float(avg_prec_curve[20] * 100.0),
        "norm_precision_auc": float(avg_nprec_curve[20] * 100.0),
    }


def strip_metric_for_json(metrics):
    out = dict(metrics)
    out.pop("succ_curve", None)
    out.pop("prec_curve", None)
    out.pop("nprec_curve", None)
    out.pop("valid_mask", None)
    return out


# Generic fallback metrics for non-LaSOT datasets.
def bbox_iou_xywh(pred, gt):
    return calc_iou_overlap_aligned(pred.astype(np.float64), gt.astype(np.float64))


def center_error_xywh(pred, gt):
    return calc_err_center_aligned(pred.astype(np.float64), gt.astype(np.float64), normalized=False)


def normalized_center_error_xywh(pred, gt):
    return calc_err_center_aligned(pred.astype(np.float64), gt.astype(np.float64), normalized=True)


def success_auc(iou):
    thresholds = np.arange(0.0, 1.0 + 0.05, 0.05, dtype=np.float64)
    success = np.array([(iou > t).mean() for t in thresholds], dtype=np.float64)
    return float(success.mean() * 100.0)


def precision_at_20(center_err):
    return float((center_err <= 20.0).mean() * 100.0)


def norm_precision_at_20(norm_err):
    return float((norm_err <= 0.20).mean() * 100.0)


def compute_metrics(pred, gt, valid):
    pred_v = pred[valid]
    gt_v = gt[valid]

    if pred_v.shape[0] == 0:
        return {
            "valid_frames": 0,
            "success_auc": 0.0,
            "precision_20": 0.0,
            "norm_precision_auc": 0.0,
            "mean_iou": 0.0,
            "mean_center_error": float("inf"),
            "median_center_error": float("inf"),
        }

    iou = bbox_iou_xywh(pred_v, gt_v)
    center_err = center_error_xywh(pred_v, gt_v)
    norm_err = normalized_center_error_xywh(pred_v, gt_v)

    return {
        "valid_frames": int(valid.sum()),
        "success_auc": success_auc(iou),
        "precision_20": precision_at_20(center_err),
        "norm_precision_auc": norm_precision_at_20(norm_err),
        "mean_iou": float(iou.mean() * 100.0),
        "mean_center_error": float(center_err.mean()),
        "median_center_error": float(np.median(center_err)),
    }


def compare_frame_level(preds, gt, valid, out_csv):
    """
    Save frame-level comparison using aligned IoU/center-error calculation.
    """
    names = list(preds.keys())
    base = preds["baseline"]

    base_iou = bbox_iou_xywh(base, gt)
    base_ce = center_error_xywh(base, gt)

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", encoding="utf-8") as f:
        header = ["frame", "valid"]
        for name in names:
            header += [
                f"{name}_x", f"{name}_y", f"{name}_w", f"{name}_h",
                f"{name}_iou", f"{name}_center_error",
            ]
        for name in names:
            if name == "baseline":
                continue
            header += [f"{name}_iou_gain", f"{name}_center_error_gain"]
        f.write(",".join(header) + "\n")

        all_ious = {name: bbox_iou_xywh(preds[name], gt) for name in names}
        all_ces = {name: center_error_xywh(preds[name], gt) for name in names}

        for i in range(len(gt)):
            row = [i, int(valid[i])]
            for name in names:
                row += [
                    *preds[name][i].tolist(),
                    float(all_ious[name][i]),
                    float(all_ces[name][i]),
                ]
            for name in names:
                if name == "baseline":
                    continue
                row += [
                    float(all_ious[name][i] - base_iou[i]),
                    float(base_ce[i] - all_ces[name][i]),
                ]
            f.write(",".join(map(str, row)) + "\n")

    summary = {}
    for name in names:
        if name == "baseline":
            continue
        iou_gain = bbox_iou_xywh(preds[name], gt) - base_iou
        summary[name] = {
            "corrected_frames_iou": int(((iou_gain > 0.01) & valid).sum()),
            "hurt_frames_iou": int(((iou_gain < -0.01) & valid).sum()),
            "same_frames_iou": int(((np.abs(iou_gain) <= 0.01) & valid).sum()),
        }

    return summary


def save_peak_info(path, peak_infos):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        fieldnames = [
            "frame",
            "rerank_used",
            "should_update",
            "should_update_history",
            "selected_idx",
            "selected_idx_raw",
            "selected_idx_final",
            "target_prob",
            "identity_margin",
            "ambiguity_ratio",
            "drift_gate_enable",
            "drift_gate_allow",
            "drift_gate_reason",
            "drift_gate_cooldown_before",
            "drift_gate_cooldown_after",
            "drift_gate_baseline_center_jump",
            "drift_gate_state_center_jump",
            "drift_gate_baseline_iou",
            "drift_gate_width_ratio",
            "drift_gate_height_ratio",
            "drift_gate_area_ratio",
            "drift_gate_bypass",
            "history_len",
            "update_history_count",
        ]
        f.write(",".join(fieldnames) + "\n")
        for item in peak_infos:
            row = [str(item.get(name, "")) for name in fieldnames]
            f.write(",".join(row) + "\n")


def print_metric_table(seq, metrics):
    keys = [
        "success_auc",
        "precision_20",
        "norm_precision_auc",
        "mean_iou",
        "mean_center_error",
        "median_center_error",
    ]

    print("\n" + "=" * 100)
    print(f"[Metrics] {seq}")
    print("-" * 100)
    print(f"{'metric':28s} {'baseline':>12s} {'offline':>12s} {'stage4':>12s} {'off-base':>12s} {'s4-base':>12s}")

    base = metrics["baseline"]
    offline = metrics["offline"]
    stage4 = metrics["stage4"]

    for k in keys:
        b = base[k]
        o = offline[k]
        s4 = stage4[k]
        print(f"{k:28s} {b:12.6f} {o:12.6f} {s4:12.6f} {o-b:12.6f} {s4-b:12.6f}")

    print("-" * 100)
    print(f"valid_frames: {base['valid_frames']}")
    print("=" * 100)


# ============================================================
# Sequence list
# ============================================================

def read_seq_list(path, dataset):
    if path is None:
        return None

    seqs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()

            # Supports:
            #   helmet-17
            #   LASOT helmet-17
            #   GOT10K_vottrain 000079
            if len(parts) == 1:
                seqs.append(parts[0])
            else:
                ds = parts[0].lower()
                seq = parts[1]
                if dataset.lower() == "lasot" and "lasot" in ds:
                    seqs.append(seq)
                elif dataset.lower().startswith("got") and "got" in ds:
                    seqs.append(seq)
                elif dataset.lower() not in ["lasot"] and "got" not in dataset.lower():
                    seqs.append(seq)

    return seqs


# ============================================================
# Main one sequence
# ============================================================

def run_one_sequence(args, seq, offline_ckpt, stage4_ckpt):
    img_files, gt, valid, seq_dir = load_sequence(
        dataset=args.dataset,
        root=args.root,
        seq=seq,
        got_split=args.got_split,
    )

    if args.max_frames is not None:
        img_files = img_files[:args.max_frames]
        gt = gt[:args.max_frames]
        valid = valid[:args.max_frames]

    out_dir = Path(args.out_dir) / args.dataset / seq
    out_dir.mkdir(parents=True, exist_ok=True)

    tracker_dataset_name = args.tracker_dataset_name or args.dataset

    print("\n" + "=" * 100)
    print(f"[RUN] seq={seq} dataset={args.dataset} frames={len(img_files)} valid={int(valid.sum())}")
    print(f"[OUT] {out_dir}")
    print("=" * 100)

    print("[RUN] baseline...")
    base_pred, base_peak = run_tracker(
        param_name=args.baseline_param,
        dataset_name=tracker_dataset_name,
        img_files=img_files,
        gt=gt,
        base_checkpoint=args.base_checkpoint,
        star_ckpt=None,
        mode=None,
    )

    print("[RUN] offline STARTrack...")
    offline_pred, offline_peak = run_tracker(
        param_name=args.star_param,
        dataset_name=tracker_dataset_name,
        img_files=img_files,
        gt=gt,
        base_checkpoint=args.base_checkpoint,
        star_ckpt=offline_ckpt,
        mode=args.star_mode,
    )

    print("[RUN] stage4 STARTrack...")
    stage4_pred, stage4_peak = run_tracker(
        param_name=args.star_param,
        dataset_name=tracker_dataset_name,
        img_files=img_files,
        gt=gt,
        base_checkpoint=args.base_checkpoint,
        star_ckpt=stage4_ckpt,
        mode=args.star_mode,
    )

    preds = {
        "baseline": base_pred,
        "offline": offline_pred,
        "stage4": stage4_pred,
    }

    save_bboxes(out_dir / "baseline_pred.txt", base_pred)
    save_bboxes(out_dir / "offline_pred.txt", offline_pred)
    save_bboxes(out_dir / "stage4_pred.txt", stage4_pred)
    save_bboxes(out_dir / "gt.txt", gt)

    save_peak_info(out_dir / "offline_peak_info.csv", offline_peak)
    save_peak_info(out_dir / "stage4_peak_info.csv", stage4_peak)

    if args.dataset.lower() == "lasot":
        metrics_full = {
            name: evaluate_sequence_aligned_metrics(
                pred_boxes=pred,
                gt_boxes=gt,
                seq_dir=seq_dir,
                dataset="lasot",
                exclude_invalid_frames=args.exclude_invalid_frames,
            )
            for name, pred in preds.items()
        }
        # Use the aligned valid mask for frame-level comparison.
        valid = metrics_full["baseline"]["valid_mask"].astype(bool)
        metrics = {name: strip_metric_for_json(m) for name, m in metrics_full.items()}
    else:
        metrics_full = None
        metrics = {
            name: compute_metrics(pred, gt, valid)
            for name, pred in preds.items()
        }

    # Save aligned/updated valid mask.
    np.savetxt(str(out_dir / "valid.txt"), valid.astype(np.int32), fmt="%d", delimiter=",")

    frame_summary = compare_frame_level(
        preds=preds,
        gt=gt,
        valid=valid,
        out_csv=out_dir / "frame_compare.csv",
    )

    peak_summary = {
        "offline": {
            "rerank_used_frames": int(sum(int(x.get("rerank_used", False)) for x in offline_peak)),
            "should_update_frames": int(sum(int(x.get("should_update", False)) for x in offline_peak)),
        },
        "stage4": {
            "rerank_used_frames": int(sum(int(x.get("rerank_used", False)) for x in stage4_peak)),
            "should_update_frames": int(sum(int(x.get("should_update", False)) for x in stage4_peak)),
        },
    }

    summary = {
        "sequence": seq,
        "dataset": args.dataset,
        "frames": len(img_files),
        "valid_frames": int(valid.sum()),
        "metrics": metrics,
        "_metrics_full": metrics_full,
        "delta": {
            "offline_minus_baseline": {
                k: metrics["offline"][k] - metrics["baseline"][k]
                for k in metrics["baseline"]
                if isinstance(metrics["baseline"][k], (int, float))
            },
            "stage4_minus_baseline": {
                k: metrics["stage4"][k] - metrics["baseline"][k]
                for k in metrics["baseline"]
                if isinstance(metrics["baseline"][k], (int, float))
            },
            "stage4_minus_offline": {
                k: metrics["stage4"][k] - metrics["offline"][k]
                for k in metrics["offline"]
                if isinstance(metrics["offline"][k], (int, float))
            },
        },
        "frame_summary": frame_summary,
        "peak_summary": peak_summary,
        "files": {
            "baseline_pred": str(out_dir / "baseline_pred.txt"),
            "offline_pred": str(out_dir / "offline_pred.txt"),
            "stage4_pred": str(out_dir / "stage4_pred.txt"),
            "gt": str(out_dir / "gt.txt"),
            "valid": str(out_dir / "valid.txt"),
            "frame_compare_csv": str(out_dir / "frame_compare.csv"),
            "offline_peak_info": str(out_dir / "offline_peak_info.csv"),
            "stage4_peak_info": str(out_dir / "stage4_peak_info.csv"),
        },
    }

    summary_for_json = dict(summary)
    summary_for_json.pop("_metrics_full", None)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_for_json, f, indent=2)

    print_metric_table(seq, metrics)

    print("[Frame summary]")
    print(json.dumps(frame_summary, indent=2))
    print("[Peak summary]")
    print(json.dumps(peak_summary, indent=2))

    return summary


def save_aggregate(summaries, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "aggregate_summary.csv"
    json_path = out_dir / "aggregate_summary.json"

    # 你要的两个核心 TXT
    all_seq_txt_path = out_dir / "all_sequence_compare.txt"
    overall_txt_path = out_dir / "overall_average_compare.txt"

    # 可选 CSV，方便后续 Excel / Python 画图
    all_seq_csv_path = out_dir / "all_sequence_compare.csv"
    overall_csv_path = out_dir / "overall_average_compare.csv"

    metric_keys = [
        "success_auc",
        "norm_precision_auc",
        "precision_20",
        "mean_iou",
        "mean_center_error",
        "median_center_error",
    ]

    # 展示名
    metric_name_map = {
        "success_auc": "AUC",
        "norm_precision_auc": "PNorm",
        "precision_20": "P",
        "mean_iou": "MeanIoU",
        "mean_center_error": "MeanCE",
        "median_center_error": "MedianCE",
    }

    main_metrics = [
        ("AUC", "success_auc"),
        ("PNorm", "norm_precision_auc"),
        ("P", "precision_20"),
    ]

    if len(summaries) == 0:
        print("[WARN] No valid sequence summary. Nothing to aggregate.")
        return

    # ============================================================
    # 1. 保留原始 aggregate_summary.csv
    # ============================================================
    with open(csv_path, "w", encoding="utf-8") as f:
        header = [
            "dataset", "sequence", "frames", "valid_frames",
        ]

        for variant in ["baseline", "offline", "stage4"]:
            for k in metric_keys:
                header.append(f"{variant}_{k}")

        for prefix in [
            "offline_minus_baseline",
            "stage4_minus_baseline",
            "stage4_minus_offline",
        ]:
            for k in metric_keys:
                header.append(f"{prefix}_{k}")

        for variant in ["offline", "stage4"]:
            header += [
                f"{variant}_corrected_frames_iou",
                f"{variant}_hurt_frames_iou",
                f"{variant}_same_frames_iou",
                f"{variant}_rerank_used_frames",
                f"{variant}_should_update_frames",
            ]

        f.write(",".join(header) + "\n")

        for s in summaries:
            row = [
                s["dataset"],
                s["sequence"],
                s["frames"],
                s["valid_frames"],
            ]

            for variant in ["baseline", "offline", "stage4"]:
                for k in metric_keys:
                    row.append(s["metrics"][variant][k])

            for prefix in [
                "offline_minus_baseline",
                "stage4_minus_baseline",
                "stage4_minus_offline",
            ]:
                for k in metric_keys:
                    row.append(s["delta"][prefix][k])

            for variant in ["offline", "stage4"]:
                fs = s.get("frame_summary", {}).get(variant, {})
                ps = s.get("peak_summary", {}).get(variant, {})

                row += [
                    fs.get("corrected_frames_iou", 0),
                    fs.get("hurt_frames_iou", 0),
                    fs.get("same_frames_iou", 0),
                    ps.get("rerank_used_frames", 0),
                    ps.get("should_update_frames", 0),
                ]

            f.write(",".join(map(str, row)) + "\n")

    # ============================================================
    # 2. 保留原始 aggregate_summary.json
    # ============================================================
    summaries_for_json = []
    for item in summaries:
        item_json = dict(item)
        item_json.pop("_metrics_full", None)
        summaries_for_json.append(item_json)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summaries_for_json, f, indent=2)

    # ============================================================
    # 3. 所有序列对比：一个 TXT 文件
    # ============================================================
    with open(all_seq_txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 160 + "\n")
        f.write("All Sequence Comparison: Baseline vs Offline STARTrack vs Stage4 STARTrack\n")
        f.write("AUC   = success_auc\n")
        f.write("PNorm = normalized precision AUC\n")
        f.write("P     = precision @ 20px\n")
        f.write("=" * 160 + "\n\n")

        header = (
            f"{'Seq':30s} "
            f"{'Base_AUC':>9s} {'Off_AUC':>9s} {'S4_AUC':>9s} "
            f"{'Off-B':>9s} {'S4-B':>9s} {'S4-Off':>9s} | "
            f"{'Base_PN':>9s} {'Off_PN':>9s} {'S4_PN':>9s} "
            f"{'Off-B':>9s} {'S4-B':>9s} {'S4-Off':>9s} | "
            f"{'Base_P':>9s} {'Off_P':>9s} {'S4_P':>9s} "
            f"{'Off-B':>9s} {'S4-B':>9s} {'S4-Off':>9s} | "
            f"{'Off_C/H':>12s} {'S4_C/H':>12s} "
            f"{'Off_Rerank':>10s} {'S4_Rerank':>10s}\n"
        )

        f.write(header)
        f.write("-" * 160 + "\n")

        for s in summaries:
            seq = s["sequence"]

            base = s["metrics"]["baseline"]
            off = s["metrics"]["offline"]
            s4 = s["metrics"]["stage4"]

            off_frame = s.get("frame_summary", {}).get("offline", {})
            s4_frame = s.get("frame_summary", {}).get("stage4", {})
            off_peak = s.get("peak_summary", {}).get("offline", {})
            s4_peak = s.get("peak_summary", {}).get("stage4", {})

            off_c = int(off_frame.get("corrected_frames_iou", 0))
            off_h = int(off_frame.get("hurt_frames_iou", 0))
            s4_c = int(s4_frame.get("corrected_frames_iou", 0))
            s4_h = int(s4_frame.get("hurt_frames_iou", 0))

            off_r = int(off_peak.get("rerank_used_frames", 0))
            s4_r = int(s4_peak.get("rerank_used_frames", 0))

            line = (
                f"{seq:30s} "
                f"{base['success_auc']:9.4f} {off['success_auc']:9.4f} {s4['success_auc']:9.4f} "
                f"{off['success_auc'] - base['success_auc']:9.4f} "
                f"{s4['success_auc'] - base['success_auc']:9.4f} "
                f"{s4['success_auc'] - off['success_auc']:9.4f} | "
                f"{base['norm_precision_auc']:9.4f} {off['norm_precision_auc']:9.4f} {s4['norm_precision_auc']:9.4f} "
                f"{off['norm_precision_auc'] - base['norm_precision_auc']:9.4f} "
                f"{s4['norm_precision_auc'] - base['norm_precision_auc']:9.4f} "
                f"{s4['norm_precision_auc'] - off['norm_precision_auc']:9.4f} | "
                f"{base['precision_20']:9.4f} {off['precision_20']:9.4f} {s4['precision_20']:9.4f} "
                f"{off['precision_20'] - base['precision_20']:9.4f} "
                f"{s4['precision_20'] - base['precision_20']:9.4f} "
                f"{s4['precision_20'] - off['precision_20']:9.4f} | "
                f"{off_c}/{off_h:<7d} {s4_c}/{s4_h:<7d} "
                f"{off_r:10d} {s4_r:10d}\n"
            )

            f.write(line)

        f.write("-" * 160 + "\n")

    # ============================================================
    # 4. 所有序列对比：一个 CSV 文件
    # ============================================================
    with open(all_seq_csv_path, "w", encoding="utf-8") as f:
        header = [
            "dataset", "sequence", "frames", "valid_frames",

            "baseline_AUC", "offline_AUC", "stage4_AUC",
            "offline_minus_baseline_AUC",
            "stage4_minus_baseline_AUC",
            "stage4_minus_offline_AUC",

            "baseline_PNorm", "offline_PNorm", "stage4_PNorm",
            "offline_minus_baseline_PNorm",
            "stage4_minus_baseline_PNorm",
            "stage4_minus_offline_PNorm",

            "baseline_P", "offline_P", "stage4_P",
            "offline_minus_baseline_P",
            "stage4_minus_baseline_P",
            "stage4_minus_offline_P",

            "offline_corrected_frames_iou",
            "offline_hurt_frames_iou",
            "offline_same_frames_iou",
            "offline_rerank_used_frames",
            "offline_should_update_frames",

            "stage4_corrected_frames_iou",
            "stage4_hurt_frames_iou",
            "stage4_same_frames_iou",
            "stage4_rerank_used_frames",
            "stage4_should_update_frames",
        ]
        f.write(",".join(header) + "\n")

        for s in summaries:
            base = s["metrics"]["baseline"]
            off = s["metrics"]["offline"]
            s4 = s["metrics"]["stage4"]

            off_frame = s.get("frame_summary", {}).get("offline", {})
            s4_frame = s.get("frame_summary", {}).get("stage4", {})
            off_peak = s.get("peak_summary", {}).get("offline", {})
            s4_peak = s.get("peak_summary", {}).get("stage4", {})

            row = [
                s["dataset"],
                s["sequence"],
                s["frames"],
                s["valid_frames"],

                base["success_auc"],
                off["success_auc"],
                s4["success_auc"],
                off["success_auc"] - base["success_auc"],
                s4["success_auc"] - base["success_auc"],
                s4["success_auc"] - off["success_auc"],

                base["norm_precision_auc"],
                off["norm_precision_auc"],
                s4["norm_precision_auc"],
                off["norm_precision_auc"] - base["norm_precision_auc"],
                s4["norm_precision_auc"] - base["norm_precision_auc"],
                s4["norm_precision_auc"] - off["norm_precision_auc"],

                base["precision_20"],
                off["precision_20"],
                s4["precision_20"],
                off["precision_20"] - base["precision_20"],
                s4["precision_20"] - base["precision_20"],
                s4["precision_20"] - off["precision_20"],

                off_frame.get("corrected_frames_iou", 0),
                off_frame.get("hurt_frames_iou", 0),
                off_frame.get("same_frames_iou", 0),
                off_peak.get("rerank_used_frames", 0),
                off_peak.get("should_update_frames", 0),

                s4_frame.get("corrected_frames_iou", 0),
                s4_frame.get("hurt_frames_iou", 0),
                s4_frame.get("same_frames_iou", 0),
                s4_peak.get("rerank_used_frames", 0),
                s4_peak.get("should_update_frames", 0),
            ]

            f.write(",".join(map(str, row)) + "\n")

    # ============================================================
    # 5. 计算总体平均：一个 TXT 文件
    #    这里默认是 sequence-level macro average
    # ============================================================
    overall = {
        "num_sequences": len(summaries),
        "baseline": {},
        "offline": {},
        "stage4": {},
        "offline_minus_baseline": {},
        "stage4_minus_baseline": {},
        "stage4_minus_offline": {},
    }

    # Overall aggregation:
    # For LaSOT aligned metrics, follow toolkit_label_final.py: average curves first,
    # then compute AUC/P/PNorm. If curve data are unavailable, fall back to scalar mean.
    if len(summaries) > 0 and summaries[0].get("_metrics_full") is not None:
        baseline_overall = summarize_aligned_metrics([s["_metrics_full"]["baseline"] for s in summaries])
        offline_overall = summarize_aligned_metrics([s["_metrics_full"]["offline"] for s in summaries])
        stage4_overall = summarize_aligned_metrics([s["_metrics_full"]["stage4"] for s in summaries])

        overall["baseline"]["AUC"] = baseline_overall["success_auc"]
        overall["baseline"]["PNorm"] = baseline_overall["norm_precision_auc"]
        overall["baseline"]["P"] = baseline_overall["precision_20"]

        overall["offline"]["AUC"] = offline_overall["success_auc"]
        overall["offline"]["PNorm"] = offline_overall["norm_precision_auc"]
        overall["offline"]["P"] = offline_overall["precision_20"]

        overall["stage4"]["AUC"] = stage4_overall["success_auc"]
        overall["stage4"]["PNorm"] = stage4_overall["norm_precision_auc"]
        overall["stage4"]["P"] = stage4_overall["precision_20"]
    else:
        for display_name, metric_key in main_metrics:
            base_values = [s["metrics"]["baseline"][metric_key] for s in summaries]
            off_values = [s["metrics"]["offline"][metric_key] for s in summaries]
            s4_values = [s["metrics"]["stage4"][metric_key] for s in summaries]

            overall["baseline"][display_name] = float(np.mean(base_values))
            overall["offline"][display_name] = float(np.mean(off_values))
            overall["stage4"][display_name] = float(np.mean(s4_values))

    for m in ["AUC", "PNorm", "P"]:
        overall["offline_minus_baseline"][m] = overall["offline"][m] - overall["baseline"][m]
        overall["stage4_minus_baseline"][m] = overall["stage4"][m] - overall["baseline"][m]
        overall["stage4_minus_offline"][m] = overall["stage4"][m] - overall["offline"][m]

    # 额外总体 corrected / hurt / rerank 统计
    for variant in ["offline", "stage4"]:
        corrected = 0
        hurt = 0
        same = 0
        rerank = 0
        update = 0

        for s in summaries:
            fs = s.get("frame_summary", {}).get(variant, {})
            ps = s.get("peak_summary", {}).get(variant, {})

            corrected += int(fs.get("corrected_frames_iou", 0))
            hurt += int(fs.get("hurt_frames_iou", 0))
            same += int(fs.get("same_frames_iou", 0))
            rerank += int(ps.get("rerank_used_frames", 0))
            update += int(ps.get("should_update_frames", 0))

        overall[variant]["corrected_frames_iou"] = corrected
        overall[variant]["hurt_frames_iou"] = hurt
        overall[variant]["same_frames_iou"] = same
        overall[variant]["rerank_used_frames"] = rerank
        overall[variant]["should_update_frames"] = update
        overall[variant]["corrected_hurt_ratio"] = float(corrected / max(hurt, 1))

    with open(overall_txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("Overall Average Comparison\n")
        f.write("AUC   = success_auc\n")
        f.write("PNorm = normalized precision AUC\n")
        f.write("P     = precision @ 20px\n")
        f.write("Average type: LaSOT aligned curve average first, then AUC/P/PNorm\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"Num sequences: {overall['num_sequences']}\n\n")

        f.write(
            f"{'Metric':10s} "
            f"{'Baseline':>12s} "
            f"{'Offline':>12s} "
            f"{'Stage4':>12s} "
            f"{'Off-Base':>12s} "
            f"{'S4-Base':>12s} "
            f"{'S4-Off':>12s}\n"
        )
        f.write("-" * 100 + "\n")

        for m in ["AUC", "PNorm", "P"]:
            f.write(
                f"{m:10s} "
                f"{overall['baseline'][m]:12.6f} "
                f"{overall['offline'][m]:12.6f} "
                f"{overall['stage4'][m]:12.6f} "
                f"{overall['offline_minus_baseline'][m]:12.6f} "
                f"{overall['stage4_minus_baseline'][m]:12.6f} "
                f"{overall['stage4_minus_offline'][m]:12.6f}\n"
            )

        f.write("-" * 100 + "\n\n")

        f.write("[Frame-level correction summary]\n")

        for variant in ["offline", "stage4"]:
            f.write(f"\n{variant}:\n")
            f.write(f"  corrected_frames_iou : {overall[variant]['corrected_frames_iou']}\n")
            f.write(f"  hurt_frames_iou      : {overall[variant]['hurt_frames_iou']}\n")
            f.write(f"  same_frames_iou      : {overall[variant]['same_frames_iou']}\n")
            f.write(f"  corrected/hurt ratio : {overall[variant]['corrected_hurt_ratio']:.4f}\n")
            f.write(f"  rerank_used_frames   : {overall[variant]['rerank_used_frames']}\n")
            f.write(f"  should_update_frames : {overall[variant]['should_update_frames']}\n")

        f.write("\n" + "=" * 100 + "\n")

    # ============================================================
    # 6. 总体平均：一个 CSV 文件
    # ============================================================
    with open(overall_csv_path, "w", encoding="utf-8") as f:
        header = [
            "num_sequences",

            "baseline_AUC",
            "offline_AUC",
            "stage4_AUC",
            "offline_minus_baseline_AUC",
            "stage4_minus_baseline_AUC",
            "stage4_minus_offline_AUC",

            "baseline_PNorm",
            "offline_PNorm",
            "stage4_PNorm",
            "offline_minus_baseline_PNorm",
            "stage4_minus_baseline_PNorm",
            "stage4_minus_offline_PNorm",

            "baseline_P",
            "offline_P",
            "stage4_P",
            "offline_minus_baseline_P",
            "stage4_minus_baseline_P",
            "stage4_minus_offline_P",

            "offline_corrected_frames_iou",
            "offline_hurt_frames_iou",
            "offline_corrected_hurt_ratio",
            "offline_rerank_used_frames",
            "offline_should_update_frames",

            "stage4_corrected_frames_iou",
            "stage4_hurt_frames_iou",
            "stage4_corrected_hurt_ratio",
            "stage4_rerank_used_frames",
            "stage4_should_update_frames",
        ]

        f.write(",".join(header) + "\n")

        row = [
            overall["num_sequences"],

            overall["baseline"]["AUC"],
            overall["offline"]["AUC"],
            overall["stage4"]["AUC"],
            overall["offline_minus_baseline"]["AUC"],
            overall["stage4_minus_baseline"]["AUC"],
            overall["stage4_minus_offline"]["AUC"],

            overall["baseline"]["PNorm"],
            overall["offline"]["PNorm"],
            overall["stage4"]["PNorm"],
            overall["offline_minus_baseline"]["PNorm"],
            overall["stage4_minus_baseline"]["PNorm"],
            overall["stage4_minus_offline"]["PNorm"],

            overall["baseline"]["P"],
            overall["offline"]["P"],
            overall["stage4"]["P"],
            overall["offline_minus_baseline"]["P"],
            overall["stage4_minus_baseline"]["P"],
            overall["stage4_minus_offline"]["P"],

            overall["offline"]["corrected_frames_iou"],
            overall["offline"]["hurt_frames_iou"],
            overall["offline"]["corrected_hurt_ratio"],
            overall["offline"]["rerank_used_frames"],
            overall["offline"]["should_update_frames"],

            overall["stage4"]["corrected_frames_iou"],
            overall["stage4"]["hurt_frames_iou"],
            overall["stage4"]["corrected_hurt_ratio"],
            overall["stage4"]["rerank_used_frames"],
            overall["stage4"]["should_update_frames"],
        ]

        f.write(",".join(map(str, row)) + "\n")

    print("\n" + "=" * 100)
    print("[AGGREGATE SAVED]")
    print("raw aggregate csv       :", csv_path)
    print("raw aggregate json      :", json_path)
    print("all sequence txt        :", all_seq_txt_path)
    print("all sequence csv        :", all_seq_csv_path)
    print("overall average txt     :", overall_txt_path)
    print("overall average csv     :", overall_csv_path)
    print("=" * 100)

    print("\n" + "=" * 100)
    print("[OVERALL AUC / PNorm / P]")
    print(
        f"{'Metric':10s} "
        f"{'Baseline':>12s} "
        f"{'Offline':>12s} "
        f"{'Stage4':>12s} "
        f"{'Off-Base':>12s} "
        f"{'S4-Base':>12s} "
        f"{'S4-Off':>12s}"
    )
    print("-" * 100)

    for m in ["AUC", "PNorm", "P"]:
        print(
            f"{m:10s} "
            f"{overall['baseline'][m]:12.6f} "
            f"{overall['offline'][m]:12.6f} "
            f"{overall['stage4'][m]:12.6f} "
            f"{overall['offline_minus_baseline'][m]:12.6f} "
            f"{overall['stage4_minus_baseline'][m]:12.6f} "
            f"{overall['stage4_minus_offline'][m]:12.6f}"
        )

    print("=" * 100)



# ============================================================
# Two-way override: baseline vs stage4 only
# ============================================================

def reset_yaml_runtime_state():
    """Reload PyYAML internals before loading a SUTrack YAML config."""
    import importlib
    import yaml
    import yaml.resolver
    import yaml.composer
    import yaml.constructor
    import yaml.loader

    importlib.reload(yaml.resolver)
    importlib.reload(yaml.composer)
    importlib.reload(yaml.constructor)
    importlib.reload(yaml.loader)
    importlib.reload(yaml)

    from yaml import SafeLoader
    if not isinstance(SafeLoader.yaml_implicit_resolvers, dict):
        raise RuntimeError(
            f"PyYAML SafeLoader.yaml_implicit_resolvers corrupted: "
            f"type={type(SafeLoader.yaml_implicit_resolvers)}, "
            f"value={repr(SafeLoader.yaml_implicit_resolvers)}"
        )
    if not callable(SafeLoader.yaml_implicit_resolvers.get):
        raise RuntimeError(
            f"PyYAML SafeLoader.yaml_implicit_resolvers.get corrupted: "
            f"type={type(SafeLoader.yaml_implicit_resolvers.get)}, "
            f"value={repr(SafeLoader.yaml_implicit_resolvers.get)}"
        )


def fresh_sutrack_parameters(param_name):
    """Freshly reload YAML/config/parameter modules before each tracker construction."""
    import importlib

    reset_yaml_runtime_state()

    import lib.config.sutrack.config as sutrack_config
    import lib.test.parameter.sutrack as sutrack_param

    importlib.reload(sutrack_config)
    sutrack_param = importlib.reload(sutrack_param)

    return sutrack_param.parameters(param_name)


def print_metric_table(seq, metrics):
    keys = [
        "success_auc",
        "precision_20",
        "norm_precision_auc",
        "mean_iou",
        "mean_center_error",
        "median_center_error",
    ]

    print("\n" + "=" * 100)
    print(f"[Metrics] {seq}")
    print("-" * 100)
    print(f"{'metric':28s} {'baseline':>12s} {'stage4':>12s} {'s4-base':>12s}")

    base = metrics["baseline"]
    stage4 = metrics["stage4"]

    for k in keys:
        b = base[k]
        s4 = stage4[k]
        print(f"{k:28s} {b:12.6f} {s4:12.6f} {s4-b:12.6f}")

    print("-" * 100)
    print(f"valid_frames: {base['valid_frames']}")
    print("=" * 100)


def run_one_sequence(args, seq, offline_ckpt, stage4_ckpt):
    """Run only baseline SUTrack and Stage4 STARTrack on one sequence."""
    img_files, gt, valid, seq_dir = load_sequence(
        dataset=args.dataset,
        root=args.root,
        seq=seq,
        got_split=args.got_split,
    )

    if args.max_frames is not None:
        img_files = img_files[:args.max_frames]
        gt = gt[:args.max_frames]
        valid = valid[:args.max_frames]

    out_dir = Path(args.out_dir) / args.dataset / seq
    out_dir.mkdir(parents=True, exist_ok=True)

    tracker_dataset_name = args.tracker_dataset_name or args.dataset

    print("\n" + "=" * 100)
    print(f"[RUN] seq={seq} dataset={args.dataset} frames={len(img_files)} valid={int(valid.sum())}")
    print(f"[OUT] {out_dir}")
    print("[MODE] baseline vs stage4 only; offline STARTrack is skipped")
    print("=" * 100)

    print("[RUN] baseline...")
    base_pred, base_peak = run_tracker(
        param_name=args.baseline_param,
        dataset_name=tracker_dataset_name,
        img_files=img_files,
        gt=gt,
        base_checkpoint=args.base_checkpoint,
        star_ckpt=None,
        mode=None,
    )

    print("[RUN] stage4 STARTrack...")
    stage4_pred, stage4_peak = run_tracker(
        param_name=args.star_param,
        dataset_name=tracker_dataset_name,
        img_files=img_files,
        gt=gt,
        base_checkpoint=args.base_checkpoint,
        star_ckpt=stage4_ckpt,
        mode=args.star_mode,
    )

    preds = {
        "baseline": base_pred,
        "stage4": stage4_pred,
    }

    save_bboxes(out_dir / "baseline_pred.txt", base_pred)
    save_bboxes(out_dir / "stage4_pred.txt", stage4_pred)
    save_bboxes(out_dir / "gt.txt", gt)
    save_peak_info(out_dir / "stage4_peak_info.csv", stage4_peak)

    if args.dataset.lower() == "lasot":
        metrics_full = {
            name: evaluate_sequence_aligned_metrics(
                pred_boxes=pred,
                gt_boxes=gt,
                seq_dir=seq_dir,
                dataset="lasot",
                exclude_invalid_frames=args.exclude_invalid_frames,
            )
            for name, pred in preds.items()
        }
        valid = metrics_full["baseline"]["valid_mask"].astype(bool)
        metrics = {name: strip_metric_for_json(m) for name, m in metrics_full.items()}
    else:
        metrics_full = None
        metrics = {name: compute_metrics(pred, gt, valid) for name, pred in preds.items()}

    np.savetxt(str(out_dir / "valid.txt"), valid.astype(np.int32), fmt="%d", delimiter=",")

    frame_summary = compare_frame_level(
        preds=preds,
        gt=gt,
        valid=valid,
        out_csv=out_dir / "frame_compare.csv",
    )

    peak_summary = {
        "stage4": {
            "rerank_used_frames": int(sum(int(x.get("rerank_used", False)) for x in stage4_peak)),
            "should_update_frames": int(sum(int(x.get("should_update", False)) for x in stage4_peak)),
        },
    }

    summary = {
        "sequence": seq,
        "dataset": args.dataset,
        "frames": len(img_files),
        "valid_frames": int(valid.sum()),
        "metrics": metrics,
        "_metrics_full": metrics_full,
        "delta": {
            "stage4_minus_baseline": {
                k: metrics["stage4"][k] - metrics["baseline"][k]
                for k in metrics["baseline"]
                if isinstance(metrics["baseline"][k], (int, float))
            },
        },
        "frame_summary": frame_summary,
        "peak_summary": peak_summary,
        "files": {
            "baseline_pred": str(out_dir / "baseline_pred.txt"),
            "stage4_pred": str(out_dir / "stage4_pred.txt"),
            "gt": str(out_dir / "gt.txt"),
            "valid": str(out_dir / "valid.txt"),
            "frame_compare_csv": str(out_dir / "frame_compare.csv"),
            "stage4_peak_info": str(out_dir / "stage4_peak_info.csv"),
        },
    }

    summary_for_json = dict(summary)
    summary_for_json.pop("_metrics_full", None)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_for_json, f, indent=2)

    print_metric_table(seq, metrics)

    print("[Frame summary]")
    print(json.dumps(frame_summary, indent=2))
    print("[Peak summary]")
    print(json.dumps(peak_summary, indent=2))

    return summary


def save_aggregate(summaries, out_dir):
    """Aggregate baseline vs stage4 only."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "aggregate_summary.csv"
    json_path = out_dir / "aggregate_summary.json"
    all_seq_txt_path = out_dir / "all_sequence_compare.txt"
    overall_txt_path = out_dir / "overall_average_compare.txt"
    all_seq_csv_path = out_dir / "all_sequence_compare.csv"
    overall_csv_path = out_dir / "overall_average_compare.csv"

    metric_keys = [
        "success_auc",
        "norm_precision_auc",
        "precision_20",
        "mean_iou",
        "mean_center_error",
        "median_center_error",
    ]

    if len(summaries) == 0:
        print("[WARN] No valid sequence summary. Nothing to aggregate.")
        return

    with open(csv_path, "w", encoding="utf-8") as f:
        header = ["dataset", "sequence", "frames", "valid_frames"]
        for variant in ["baseline", "stage4"]:
            for k in metric_keys:
                header.append(f"{variant}_{k}")
        for k in metric_keys:
            header.append(f"stage4_minus_baseline_{k}")
        header += [
            "stage4_corrected_frames_iou",
            "stage4_hurt_frames_iou",
            "stage4_same_frames_iou",
            "stage4_rerank_used_frames",
            "stage4_should_update_frames",
        ]
        f.write(",".join(header) + "\n")

        for s in summaries:
            row = [s["dataset"], s["sequence"], s["frames"], s["valid_frames"]]
            for variant in ["baseline", "stage4"]:
                for k in metric_keys:
                    row.append(s["metrics"][variant][k])
            for k in metric_keys:
                row.append(s["delta"]["stage4_minus_baseline"][k])
            fs = s.get("frame_summary", {}).get("stage4", {})
            ps = s.get("peak_summary", {}).get("stage4", {})
            row += [
                fs.get("corrected_frames_iou", 0),
                fs.get("hurt_frames_iou", 0),
                fs.get("same_frames_iou", 0),
                ps.get("rerank_used_frames", 0),
                ps.get("should_update_frames", 0),
            ]
            f.write(",".join(map(str, row)) + "\n")

    summaries_for_json = []
    for item in summaries:
        item_json = dict(item)
        item_json.pop("_metrics_full", None)
        summaries_for_json.append(item_json)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summaries_for_json, f, indent=2)

    with open(all_seq_txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 130 + "\n")
        f.write("All Sequence Comparison: Baseline SUTrack vs Stage4 STARTrack\n")
        f.write("AUC   = success_auc\n")
        f.write("PNorm = normalized precision AUC\n")
        f.write("P     = precision @ 20px\n")
        f.write("=" * 130 + "\n\n")
        header = (
            f"{'Seq':30s} "
            f"{'Base_AUC':>9s} {'S4_AUC':>9s} {'S4-B':>9s} | "
            f"{'Base_PN':>9s} {'S4_PN':>9s} {'S4-B':>9s} | "
            f"{'Base_P':>9s} {'S4_P':>9s} {'S4-B':>9s} | "
            f"{'S4_C/H':>12s} {'S4_Rerank':>10s}\n"
        )
        f.write(header)
        f.write("-" * 130 + "\n")
        for s in summaries:
            seq = s["sequence"]
            base = s["metrics"]["baseline"]
            s4 = s["metrics"]["stage4"]
            s4_frame = s.get("frame_summary", {}).get("stage4", {})
            s4_peak = s.get("peak_summary", {}).get("stage4", {})
            s4_c = int(s4_frame.get("corrected_frames_iou", 0))
            s4_h = int(s4_frame.get("hurt_frames_iou", 0))
            s4_r = int(s4_peak.get("rerank_used_frames", 0))
            line = (
                f"{seq:30s} "
                f"{base['success_auc']:9.4f} {s4['success_auc']:9.4f} {s4['success_auc'] - base['success_auc']:9.4f} | "
                f"{base['norm_precision_auc']:9.4f} {s4['norm_precision_auc']:9.4f} {s4['norm_precision_auc'] - base['norm_precision_auc']:9.4f} | "
                f"{base['precision_20']:9.4f} {s4['precision_20']:9.4f} {s4['precision_20'] - base['precision_20']:9.4f} | "
                f"{s4_c}/{s4_h:<7d} {s4_r:10d}\n"
            )
            f.write(line)
        f.write("-" * 130 + "\n")

    with open(all_seq_csv_path, "w", encoding="utf-8") as f:
        header = [
            "dataset", "sequence", "frames", "valid_frames",
            "baseline_AUC", "stage4_AUC", "stage4_minus_baseline_AUC",
            "baseline_PNorm", "stage4_PNorm", "stage4_minus_baseline_PNorm",
            "baseline_P", "stage4_P", "stage4_minus_baseline_P",
            "stage4_corrected_frames_iou", "stage4_hurt_frames_iou", "stage4_same_frames_iou",
            "stage4_rerank_used_frames", "stage4_should_update_frames",
        ]
        f.write(",".join(header) + "\n")
        for s in summaries:
            base = s["metrics"]["baseline"]
            s4 = s["metrics"]["stage4"]
            s4_frame = s.get("frame_summary", {}).get("stage4", {})
            s4_peak = s.get("peak_summary", {}).get("stage4", {})
            row = [
                s["dataset"], s["sequence"], s["frames"], s["valid_frames"],
                base["success_auc"], s4["success_auc"], s4["success_auc"] - base["success_auc"],
                base["norm_precision_auc"], s4["norm_precision_auc"], s4["norm_precision_auc"] - base["norm_precision_auc"],
                base["precision_20"], s4["precision_20"], s4["precision_20"] - base["precision_20"],
                s4_frame.get("corrected_frames_iou", 0),
                s4_frame.get("hurt_frames_iou", 0),
                s4_frame.get("same_frames_iou", 0),
                s4_peak.get("rerank_used_frames", 0),
                s4_peak.get("should_update_frames", 0),
            ]
            f.write(",".join(map(str, row)) + "\n")

    overall = {
        "num_sequences": len(summaries),
        "baseline": {},
        "stage4": {},
        "stage4_minus_baseline": {},
    }

    if len(summaries) > 0 and summaries[0].get("_metrics_full") is not None:
        baseline_overall = summarize_aligned_metrics([s["_metrics_full"]["baseline"] for s in summaries])
        stage4_overall = summarize_aligned_metrics([s["_metrics_full"]["stage4"] for s in summaries])
        overall["baseline"]["AUC"] = baseline_overall["success_auc"]
        overall["baseline"]["PNorm"] = baseline_overall["norm_precision_auc"]
        overall["baseline"]["P"] = baseline_overall["precision_20"]
        overall["stage4"]["AUC"] = stage4_overall["success_auc"]
        overall["stage4"]["PNorm"] = stage4_overall["norm_precision_auc"]
        overall["stage4"]["P"] = stage4_overall["precision_20"]
    else:
        for display_name, metric_key in [("AUC", "success_auc"), ("PNorm", "norm_precision_auc"), ("P", "precision_20")]:
            base_values = [s["metrics"]["baseline"][metric_key] for s in summaries]
            s4_values = [s["metrics"]["stage4"][metric_key] for s in summaries]
            overall["baseline"][display_name] = float(np.mean(base_values))
            overall["stage4"][display_name] = float(np.mean(s4_values))

    for m in ["AUC", "PNorm", "P"]:
        overall["stage4_minus_baseline"][m] = overall["stage4"][m] - overall["baseline"][m]

    corrected = hurt = same = rerank = update = 0
    for s in summaries:
        fs = s.get("frame_summary", {}).get("stage4", {})
        ps = s.get("peak_summary", {}).get("stage4", {})
        corrected += int(fs.get("corrected_frames_iou", 0))
        hurt += int(fs.get("hurt_frames_iou", 0))
        same += int(fs.get("same_frames_iou", 0))
        rerank += int(ps.get("rerank_used_frames", 0))
        update += int(ps.get("should_update_frames", 0))

    overall["stage4"]["corrected_frames_iou"] = corrected
    overall["stage4"]["hurt_frames_iou"] = hurt
    overall["stage4"]["same_frames_iou"] = same
    overall["stage4"]["rerank_used_frames"] = rerank
    overall["stage4"]["should_update_frames"] = update
    overall["stage4"]["corrected_hurt_ratio"] = float(corrected / max(hurt, 1))

    with open(overall_txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("Overall Average Comparison: Baseline vs Stage4\n")
        f.write("AUC   = success_auc\n")
        f.write("PNorm = normalized precision AUC\n")
        f.write("P     = precision @ 20px\n")
        f.write("Average type: LaSOT aligned curve average first, then AUC/P/PNorm\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Num sequences: {overall['num_sequences']}\n\n")
        f.write(
            f"{'Metric':10s} {'Baseline':>12s} {'Stage4':>12s} {'S4-Base':>12s}\n"
        )
        f.write("-" * 100 + "\n")
        for m in ["AUC", "PNorm", "P"]:
            f.write(
                f"{m:10s} "
                f"{overall['baseline'][m]:12.6f} "
                f"{overall['stage4'][m]:12.6f} "
                f"{overall['stage4_minus_baseline'][m]:12.6f}\n"
            )
        f.write("-" * 100 + "\n\n")
        f.write("[Frame-level correction summary]\n")
        f.write("\nstage4:\n")
        f.write(f"  corrected_frames_iou : {overall['stage4']['corrected_frames_iou']}\n")
        f.write(f"  hurt_frames_iou      : {overall['stage4']['hurt_frames_iou']}\n")
        f.write(f"  same_frames_iou      : {overall['stage4']['same_frames_iou']}\n")
        f.write(f"  corrected/hurt ratio : {overall['stage4']['corrected_hurt_ratio']:.4f}\n")
        f.write(f"  rerank_used_frames   : {overall['stage4']['rerank_used_frames']}\n")
        f.write(f"  should_update_frames : {overall['stage4']['should_update_frames']}\n")
        f.write("\n" + "=" * 100 + "\n")

    with open(overall_csv_path, "w", encoding="utf-8") as f:
        header = [
            "num_sequences",
            "baseline_AUC", "stage4_AUC", "stage4_minus_baseline_AUC",
            "baseline_PNorm", "stage4_PNorm", "stage4_minus_baseline_PNorm",
            "baseline_P", "stage4_P", "stage4_minus_baseline_P",
            "stage4_corrected_frames_iou", "stage4_hurt_frames_iou", "stage4_corrected_hurt_ratio",
            "stage4_rerank_used_frames", "stage4_should_update_frames",
        ]
        f.write(",".join(header) + "\n")
        row = [
            overall["num_sequences"],
            overall["baseline"]["AUC"], overall["stage4"]["AUC"], overall["stage4_minus_baseline"]["AUC"],
            overall["baseline"]["PNorm"], overall["stage4"]["PNorm"], overall["stage4_minus_baseline"]["PNorm"],
            overall["baseline"]["P"], overall["stage4"]["P"], overall["stage4_minus_baseline"]["P"],
            overall["stage4"]["corrected_frames_iou"],
            overall["stage4"]["hurt_frames_iou"],
            overall["stage4"]["corrected_hurt_ratio"],
            overall["stage4"]["rerank_used_frames"],
            overall["stage4"]["should_update_frames"],
        ]
        f.write(",".join(map(str, row)) + "\n")

    print("\n" + "=" * 100)
    print("[AGGREGATE SAVED]")
    print("raw aggregate csv       :", csv_path)
    print("raw aggregate json      :", json_path)
    print("all sequence txt        :", all_seq_txt_path)
    print("all sequence csv        :", all_seq_csv_path)
    print("overall average txt     :", overall_txt_path)
    print("overall average csv     :", overall_csv_path)
    print("=" * 100)

    print("\n" + "=" * 100)
    print("[OVERALL AUC / PNorm / P]")
    print(f"{'Metric':10s} {'Baseline':>12s} {'Stage4':>12s} {'S4-Base':>12s}")
    print("-" * 100)
    for m in ["AUC", "PNorm", "P"]:
        print(
            f"{m:10s} "
            f"{overall['baseline'][m]:12.6f} "
            f"{overall['stage4'][m]:12.6f} "
            f"{overall['stage4_minus_baseline'][m]:12.6f}"
        )
    print("=" * 100)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--project-root", type=str, default=DEFAULT_PROJECT_ROOT)

    parser.add_argument("--dataset", type=str, required=True, choices=["lasot", "got10k"])
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--got-split", type=str, default="train")

    parser.add_argument("--seq", type=str, default=None)
    parser.add_argument("--seq-list", type=str, default=None)
    parser.add_argument("--max-seqs", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)

    parser.add_argument("--baseline-param", type=str, default="sutrack_b224")
    parser.add_argument("--star-param", type=str, default="sutrack_b224_startrack")
    parser.add_argument("--tracker-dataset-name", type=str, default=None)

    parser.add_argument("--base-checkpoint", type=str, required=True)
    parser.add_argument("--offline-ckpt", type=str, default=None, help="Unused in two-way mode; kept for CLI compatibility.")
    parser.add_argument("--stage4-ckpt", type=str, required=True)
    parser.add_argument("--star-mode", type=str, default=None, help="Optional override: coach/gated/shadow/etc.")

    parser.add_argument("--out-dir", type=str, default="threeway_eval_outputs")
    parser.add_argument(
        "--exclude-invalid-frames",
        action="store_true",
        help="Use valid-frame count as denominator, aligned with toolkit_label_final.py optional switch. Default matches toolkit default: False.",
    )
    parser.add_argument("--continue-on-error", action="store_true")

    args = parser.parse_args()

    setup_project_root(args.project_root)
    sanity_check_runtime(args.base_checkpoint, args.offline_ckpt, args.stage4_ckpt, args.out_dir)
    check_torch_functional("main/start")

    seqs = []
    if args.seq is not None:
        seqs.append(args.seq)

    list_seqs = read_seq_list(args.seq_list, args.dataset)
    if list_seqs:
        seqs.extend(list_seqs)

    # unique keep order
    seen = set()
    seqs_unique = []
    for s in seqs:
        if s not in seen:
            seqs_unique.append(s)
            seen.add(s)
    seqs = seqs_unique

    if args.max_seqs is not None:
        seqs = seqs[:args.max_seqs]

    if len(seqs) == 0:
        raise ValueError("Please provide --seq or --seq-list.")

    ckpt_out_dir = Path(args.out_dir) / "_converted_ckpts"
    offline_ckpt = make_startrack_test_ckpt(args.offline_ckpt, ckpt_out_dir, "offline")
    stage4_ckpt = make_startrack_test_ckpt(args.stage4_ckpt, ckpt_out_dir, "stage4")

    print("=" * 100)
    print("[INFO] Three-way inference comparison")
    print(f"[INFO] dataset        : {args.dataset}")
    print(f"[INFO] root           : {args.root}")
    print(f"[INFO] num sequences  : {len(seqs)}")
    print(f"[INFO] base ckpt      : {args.base_checkpoint}")
    print(f"[INFO] offline ckpt   : {offline_ckpt}")
    print(f"[INFO] stage4 ckpt    : {stage4_ckpt}")
    print(f"[INFO] baseline param : {args.baseline_param}")
    print(f"[INFO] star param     : {args.star_param}")
    print(f"[INFO] star mode      : {args.star_mode}")
    print(f"[INFO] out dir        : {args.out_dir}")
    print("=" * 100)

    summaries = []
    success_count = 0  # 新增：用于统计成功跑完或加载的视频数量

    try:
        libc = ctypes.CDLL("libc.so.6")
    except Exception:
        libc = None

    for idx, seq in enumerate(seqs):
        # ==========================================
        # 🟢 新增：断点重推逻辑 (Resume capability)
        # ==========================================
        seq_out_dir = Path(args.out_dir) / args.dataset / seq
        summary_path = seq_out_dir / "summary.json"
        
        if summary_path.exists():
            try:
                # 尝试读取已存在的 json，如果成功则说明该序列之前已经顺利跑完
                with open(summary_path, "r", encoding="utf-8") as f:
                    cached_summary = json.load(f)
                summaries.append(cached_summary)
                success_count += 1
                print(f"\n[{idx + 1}/{len(seqs)}] ⏭️ [SKIP] 序列 {seq} 已测试完毕，直接加载缓存: {summary_path}")
                continue  # 直接跳过，进入下一个视频
            except Exception as e:
                print(f"\n[{idx + 1}/{len(seqs)}] ⚠️ [WARN] 序列 {seq} 的缓存文件损坏，将重新推理。错误: {e}")
        # ==========================================

        try:
            print(f"\n[{idx + 1}/{len(seqs)}] ▶️ [RUN] 开始评测序列: {seq}")
            summary = run_one_sequence(args, seq, offline_ckpt, stage4_ckpt)
            summaries.append(summary)
            success_count += 1
            
            # ==========================================
            # ☢️ 核弹级内存强制回收机制
            # ==========================================
            # 1. Python 层级垃圾回收
            gc.collect()
            
            # 2. PyTorch 显存与进程通信缓存回收
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                
            # 3. 强制让 OS 回收 C++ (Mamba/CLIP) 泄露的深层内存碎片
            if libc is not None:
                try:
                    libc.malloc_trim(0)
                except Exception:
                    pass
            # ==========================================

        except Exception as e:
            print(f"❌ [ERROR] sequence failed: {seq}, error={e}")
            traceback.print_exc()

    # 聚合所有结果并保存
    save_aggregate(summaries, args.out_dir)

    # ==========================================
    # 🟢 新增：最终成功数量统计输出
    # ==========================================
    print("\n" + "★" * 100)
    print(f"🎉 [FINISHED] 所有评测任务执行结束！")
    print(f"📊 [SUMMARY] 目标序列总数 : {len(seqs)}")
    print(f"✅ [SUMMARY] 成功处理数量 : {success_count} (占比 {(success_count/max(len(seqs), 1))*100:.2f}%)")
    print(f"❌ [SUMMARY] 失败/跳过数量: {len(seqs) - success_count}")
    print("★" * 100 + "\n")


if __name__ == "__main__":
    main()
