#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run MPDTrack / SUTrack-style tracker on TNL2K and save results in the same
folder layout used by SUTrack/OSTrack-style toolkits:

    <output_dir>/<tracker_name>/<tracker_param>/<dataset_name>/<sequence>.txt
    <output_dir>/<tracker_name>/<tracker_param>/<dataset_name>/<sequence>_time.txt
    <output_dir>/<tracker_name>/<tracker_param>/<dataset_name>/<sequence>_score.txt

Example:

python test_mpdtrack_tnl2k_sutrack_format.py \
  --project-root /home/cps/czl/STARTrack_v6 \
  --tnl2k-root /home/cps/datasets/TNL2K/test \
  --config experiments/sutrack/sutrack_b384_startrack.yaml \
  --sutrack-ckpt /home/cps/czl/STARTrack_v6/checkpoints/sutrack_b384/SUTRACK_ep0180.pth.tar \
  --mpd-ckpt /home/cps/czl/STARTrack_v6/checkpoints/startrack_mamba_iou_b384/best.pth \
  --output-dir /home/cps/czl/STARTrack_v6/test/tracking_results \
  --tracker-name sutrack \
  --tracker-param sutrack_b384_startrack \
  --dataset-name tnl2k \
  --use-mpd 1 \
  --use-nlp 1

By default this script loads the tracker only once and reuses it across sequences.
This keeps the output format unchanged but avoids reloading checkpoints for every sequence.

Then evaluate with the original SUTrack toolkit by using trackerlist with:
    name='sutrack', parameter_name='sutrack_b384_startrack', dataset_name='tnl2k'
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch


# -----------------------------------------------------------------------------
# Path / import helpers
# -----------------------------------------------------------------------------

def add_project_to_path(project_root: str) -> Path:
    root = Path(project_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"project_root does not exist: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    init_paths = root / "_init_paths.py"
    if init_paths.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("_init_paths", str(init_paths))
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    return root


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


# -----------------------------------------------------------------------------
# TNL2K sequence reader
# -----------------------------------------------------------------------------

@dataclass
class SequenceInfo:
    name: str
    seq_dir: Path
    frames: List[Path]
    gt: np.ndarray  # [N, 4], xywh
    language: Optional[str] = None


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        return path.read_text(encoding="gbk", errors="ignore").strip()


def find_image_files(seq_dir: Path) -> List[Path]:
    candidates = [
        seq_dir / "imgs",
        seq_dir / "img",
        seq_dir / "images",
        seq_dir / "image",
        seq_dir / "color",
        seq_dir / "rgb",
        seq_dir,
    ]
    for d in candidates:
        if d.exists() and d.is_dir():
            imgs = [p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS]
            if imgs:
                return sorted(imgs, key=lambda p: natural_key(p.name))

    imgs = [p for p in seq_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    imgs = [p for p in imgs if not any(x in p.parts for x in ["vis", "visual", "results", "tracking_results"])]
    return sorted(imgs, key=lambda p: natural_key(str(p.relative_to(seq_dir))))


def parse_box_line(line: str) -> Optional[List[float]]:
    line = line.strip().replace(",", " ").replace("\t", " ")
    if not line:
        return None
    vals: List[float] = []
    for x in line.split():
        try:
            vals.append(float(x))
        except ValueError:
            pass
    if len(vals) < 4:
        return None
    if len(vals) >= 8:
        xs = vals[0::2][:4]
        ys = vals[1::2][:4]
        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)
        return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]
    return vals[:4]


def find_gt_file(seq_dir: Path) -> Optional[Path]:
    names = [
        "groundtruth.txt",
        "groundtruth_rect.txt",
        "groundtruth_rect.1.txt",
        "gt.txt",
        "ground_truth.txt",
    ]
    for name in names:
        p = seq_dir / name
        if p.exists() and p.is_file():
            return p
    for p in seq_dir.glob("*.txt"):
        low = p.name.lower()
        if "ground" in low or low == "gt.txt":
            return p
    return None


def load_gt(seq_dir: Path, nframes: int) -> np.ndarray:
    gt_file = find_gt_file(seq_dir)
    if gt_file is None:
        raise FileNotFoundError(f"No groundtruth file found in {seq_dir}")

    boxes: List[List[float]] = []
    for line in read_text_file(gt_file).splitlines():
        box = parse_box_line(line)
        if box is not None:
            boxes.append(box)
    if not boxes:
        raise ValueError(f"Empty or invalid groundtruth file: {gt_file}")

    gt = np.asarray(boxes, dtype=np.float32)
    if len(gt) < nframes:
        pad = np.repeat(gt[-1:], nframes - len(gt), axis=0)
        gt = np.concatenate([gt, pad], axis=0)
    elif len(gt) > nframes:
        gt = gt[:nframes]
    return gt


def find_language(seq_dir: Path) -> Optional[str]:
    names = [
        "language.txt",
        "nlp.txt",
        "nl.txt",
        "init_nlp.txt",
        "description.txt",
        "text.txt",
    ]
    for name in names:
        p = seq_dir / name
        if p.exists() and p.is_file():
            txt = read_text_file(p)
            if txt:
                return " ".join(txt.splitlines()).strip()
    return None


def discover_tnl2k_sequences(root: Path, seq_list: Optional[Path] = None) -> List[SequenceInfo]:
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"TNL2K root does not exist: {root}")

    if (root / "test").exists() and any((root / "test").iterdir()):
        scan_root = root / "test"
    else:
        scan_root = root

    wanted: Optional[set] = None
    if seq_list is not None:
        seq_list = seq_list.expanduser().resolve()
        lines = [x.strip() for x in read_text_file(seq_list).splitlines() if x.strip()]
        wanted = set(lines)

    seq_dirs = sorted([p for p in scan_root.iterdir() if p.is_dir()], key=lambda p: natural_key(p.name))

    sequences: List[SequenceInfo] = []
    for seq_dir in seq_dirs:
        if wanted is not None and seq_dir.name not in wanted:
            continue
        frames = find_image_files(seq_dir)
        if len(frames) == 0:
            print(f"[WARN] skip {seq_dir.name}: no images found")
            continue
        try:
            gt = load_gt(seq_dir, len(frames))
        except Exception as e:
            print(f"[WARN] skip {seq_dir.name}: failed to load GT: {e}")
            continue
        sequences.append(SequenceInfo(seq_dir.name, seq_dir, frames, gt, find_language(seq_dir)))

    if not sequences:
        raise RuntimeError(f"No valid TNL2K sequences found under {scan_root}")
    return sequences


# -----------------------------------------------------------------------------
# Config helpers
# -----------------------------------------------------------------------------

def get_cfg_node(root: Any, path: str, create: bool = False):
    cur = root
    parts = path.split(".")
    for name in parts[:-1]:
        if hasattr(cur, name):
            cur = getattr(cur, name)
        elif isinstance(cur, dict) and name in cur:
            cur = cur[name]
        elif create:
            try:
                from easydict import EasyDict
                node = EasyDict()
            except Exception:
                node = {}
            if isinstance(cur, dict):
                cur[name] = node
            else:
                setattr(cur, name, node)
            cur = node
        else:
            return None, parts[-1]
    return cur, parts[-1]


def set_cfg_value(cfg: Any, path: str, value: Any):
    node, key = get_cfg_node(cfg, path, create=True)
    if isinstance(node, dict):
        node[key] = value
    else:
        setattr(node, key, value)


def build_params(config: str, sutrack_ckpt: str, mpd_ckpt: Optional[str], use_mpd: bool, use_nlp: Optional[bool], debug: int):
    from lib.test.parameter.sutrack import parameters

    try:
        params = parameters(config)
    except TypeError:
        params = parameters()

    params.checkpoint = str(Path(sutrack_ckpt).expanduser().resolve())
    params.debug = int(debug)

    cfg = params.cfg
    set_cfg_value(cfg, "MODEL.USE_STARTRACK", bool(use_mpd))
    if mpd_ckpt:
        mpd_ckpt = str(Path(mpd_ckpt).expanduser().resolve())
        # Set both cfg and direct attributes, because different STARTrack/SUTrack
        # versions may read the checkpoint path from different places.
        params.startrack_ckpt = mpd_ckpt
        params.reranker_ckpt = mpd_ckpt
        params.STARTRACK_CKPT = mpd_ckpt
        set_cfg_value(cfg, "MODEL.STARTRACK_CKPT", mpd_ckpt)

    if use_nlp is not None:
        use_nlp_bool = bool(use_nlp)
        set_cfg_value(cfg, "TEST.USE_NLP.TNL2K", use_nlp_bool)
        set_cfg_value(cfg, "DATA.USE_NLP.TNL2K", use_nlp_bool)
        # Some SUTrack versions use these tables to decide whether to pass the text branch.
        set_cfg_value(cfg, "TEST.MULTI_MODAL_LANGUAGE.TNL2K", use_nlp_bool)
        set_cfg_value(cfg, "DATA.MULTI_MODAL_LANGUAGE", use_nlp_bool)
    return params




# -----------------------------------------------------------------------------
# Checkpoint compatibility
# -----------------------------------------------------------------------------

def make_startrack_test_ckpt(src_path: str, out_dir: Path, tag: str) -> str:
    """
    Convert different STARTrack/MPDTrack checkpoint formats into the test-time
    format expected by lib.test.tracker.sutrack.SUTRACK._setup_startrack().

    Expected test-time format:
        ckpt["model"] = state_dict of PostDecoderDisambiguator

    Common Stage4 format:
        ckpt["disambiguator"] = state_dict of PostDecoderDisambiguator

    If ckpt already contains "model", return the original path.
    """
    src = Path(src_path).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(str(src), map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model" in ckpt:
        print(f"[INFO] MPD checkpoint already has key 'model': {src}")
        return str(src)

    if isinstance(ckpt, dict) and "disambiguator" in ckpt:
        dst = out_dir / f"{tag}_for_test.pth"
        out = {
            "epoch": ckpt.get("epoch", -1),
            "model": ckpt["disambiguator"],
            "args": ckpt.get("args", {}),
            "metrics": ckpt.get("metrics", {}),
            "stage": ckpt.get("stage", "stage4_online_joint"),
            "source": str(src),
        }
        torch.save(out, str(dst))
        print(f"[INFO] Converted Stage4 MPD checkpoint for test: {dst}")
        return str(dst)

    keys = list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)
    raise KeyError(
        f"Cannot find 'model' or 'disambiguator' in MPD checkpoint: {src}. keys={keys}"
    )


# -----------------------------------------------------------------------------
# Output helpers: SUTrack toolkit-compatible format
# -----------------------------------------------------------------------------

def infer_tracker_param(config: str) -> str:
    c = str(config)
    if c.endswith(".yaml") or c.endswith(".yml"):
        return Path(c).stem
    return c.replace("/", "_").replace("\\", "_")


def get_result_dir(args) -> Path:
    # SUTrack/OSTrack-style directory:
    # output_dir / tracker_name / tracker_param / dataset_name
    return (
        Path(args.output_dir).expanduser().resolve()
        / str(args.tracker_name)
        / str(args.tracker_param)
        / str(args.dataset_name).lower()
    )


def write_prediction_txt(path: Path, pred: np.ndarray, delimiter: str = "\t"):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(path), pred.astype(np.float32), delimiter=delimiter, fmt="%.4f")


def write_vector_txt(path: Path, values: List[float]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for x in values:
            f.write(f"{float(x):.6f}\n")


# -----------------------------------------------------------------------------
# Optional quick metrics. Official paper numbers should be computed by toolkit.
# -----------------------------------------------------------------------------

def bbox_iou_xywh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    ax1, ay1 = a[:, 0], a[:, 1]
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1 = b[:, 0], b[:, 1]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    ix1, iy1 = np.maximum(ax1, bx1), np.maximum(ay1, by1)
    ix2, iy2 = np.minimum(ax2, bx2), np.minimum(ay2, by2)
    iw, ih = np.maximum(0.0, ix2 - ix1), np.maximum(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = np.maximum(0.0, a[:, 2]) * np.maximum(0.0, a[:, 3])
    area_b = np.maximum(0.0, b[:, 2]) * np.maximum(0.0, b[:, 3])
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-6), 0.0)


def center_error_xywh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ac = np.stack([a[:, 0] + 0.5 * a[:, 2], a[:, 1] + 0.5 * a[:, 3]], axis=1)
    bc = np.stack([b[:, 0] + 0.5 * b[:, 2], b[:, 1] + 0.5 * b[:, 3]], axis=1)
    return np.linalg.norm(ac - bc, axis=1)


def success_auc(ious: np.ndarray) -> float:
    thresholds = np.linspace(0, 1, 101)
    return float(np.mean([(ious >= th).mean() for th in thresholds]) * 100.0)


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    n = min(len(pred), len(gt))
    pred = pred[:n]
    gt = gt[:n]
    valid = (gt[:, 2] > 0) & (gt[:, 3] > 0)
    if not np.any(valid):
        return {"success_auc": 0.0, "precision_20": 0.0, "norm_precision_auc": 0.0, "mean_iou": 0.0, "valid_frames": 0}
    iou = bbox_iou_xywh(pred[valid], gt[valid])
    ce = center_error_xywh(pred[valid], gt[valid])
    diag = np.sqrt(np.maximum(gt[valid, 2] * gt[valid, 3], 1.0))
    nce = ce / np.maximum(diag, 1e-6)
    nth = np.linspace(0, 0.5, 101)
    return {
        "success_auc": success_auc(iou),
        "precision_20": float((ce <= 20).mean() * 100.0),
        "norm_precision_auc": float(np.mean([(nce <= th).mean() for th in nth]) * 100.0),
        "mean_iou": float(iou.mean() * 100.0),
        "valid_frames": int(valid.sum()),
    }


def read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"failed to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# -----------------------------------------------------------------------------
# Main tracking loop
# -----------------------------------------------------------------------------

def run_sequence(seq: SequenceInfo, args, tracker) -> Dict[str, Any]:
    result_dir = get_result_dir(args)
    pred_file = result_dir / f"{seq.name}.txt"
    time_file = result_dir / f"{seq.name}_time.txt"
    score_file = result_dir / f"{seq.name}_score.txt"

    debug_dir = result_dir / "_debug" / seq.name
    summary_file = debug_dir / "summary.json"

    if args.skip_existing and pred_file.exists():
        if summary_file.exists():
            try:
                return json.loads(summary_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "sequence": seq.name,
            "num_frames": len(seq.frames),
            "fps": 0.0,
            "metrics": {},
            "prediction_file": str(pred_file),
            "skipped": True,
        }

    first_img = read_rgb(seq.frames[0])
    init_info: Dict[str, Any] = {"init_bbox": seq.gt[0].tolist()}
    if args.use_nlp and seq.language:
        init_info["init_nlp"] = seq.language

    t0 = time.perf_counter()
    tracker.initialize(first_img, init_info)
    init_time = time.perf_counter() - t0

    pred: List[List[float]] = [seq.gt[0].astype(float).tolist()]
    scores: List[float] = [1.0]
    times: List[float] = [init_time]

    for frame_path in seq.frames[1:]:
        img = read_rgb(frame_path)
        t1 = time.perf_counter()
        out = tracker.track(img, {})
        times.append(time.perf_counter() - t1)
        pred.append([float(x) for x in out["target_bbox"]])
        scores.append(float(out.get("best_score", 0.0)))

    pred_np = np.asarray(pred, dtype=np.float32)
    metrics = compute_metrics(pred_np, seq.gt)
    track_time_sum = sum(times[1:])
    fps = max(len(seq.frames) - 1, 1) / max(track_time_sum, 1e-9)

    # SUTrack toolkit-compatible output files.
    write_prediction_txt(pred_file, pred_np, delimiter=str(args.delimiter).encode("utf-8").decode("unicode_escape"))
    write_vector_txt(time_file, times)
    write_vector_txt(score_file, scores)

    # Extra debug files are placed under _debug, so they will not disturb toolkit loading.
    if args.save_debug:
        debug_dir.mkdir(parents=True, exist_ok=True)
        if seq.language:
            (debug_dir / "language.txt").write_text(seq.language + "\n", encoding="utf-8")
        summary = {
            "sequence": seq.name,
            "num_frames": len(seq.frames),
            "fps": fps,
            "language": seq.language,
            "metrics": metrics,
            "prediction_file": str(pred_file),
            "time_file": str(time_file),
            "score_file": str(score_file),
        }
        summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        summary = {
            "sequence": seq.name,
            "num_frames": len(seq.frames),
            "fps": fps,
            "metrics": metrics,
            "prediction_file": str(pred_file),
        }

    return summary


def parse_args():
    p = argparse.ArgumentParser(description="Test MPDTrack on TNL2K and save SUTrack-toolkit-compatible results.")
    p.add_argument("--project-root", default="/home/cps/czl/STARTrack_v6", help="STARTrack/MPDTrack project root, e.g. /home/cps/czl/STARTrack_v6")
    p.add_argument("--tnl2k-root", default="/home/cps/SOT_dataset/tnl2k", help="TNL2K root or TNL2K/test directory")
    p.add_argument("--config", default="sutrack_b384_startrack", help="SUTrack YAML path or config name, e.g. experiments/sutrack/sutrack_b384_startrack.yaml")
    p.add_argument("--sutrack-ckpt", default="/home/cps/czl/STARTrack_v6/checkpoints/sutrack_b384/SUTRACK_ep0180.pth.tar", help="Base SUTrack checkpoint")
    p.add_argument("--mpd-ckpt", default="/home/cps/czl/STARTrack_v6/checkpoints/stage4_got10k_three_stage_b384_v2/last.pth", help="MPDTrack/STARTrack post-disambiguator checkpoint")

    p.add_argument("--output-dir", default="/home/cps/czl/STARTrack_v6/test/tracking_results", help="SUTrack results root, e.g. /home/cps/czl/STARTrack_v6/test/tracking_results")
    p.add_argument("--tracker-name", default="sutrack", help="Toolkit tracker name folder. Use 'sutrack' to match SUTrack trackerlist.")
    p.add_argument("--tracker-param", default="sutrack_b384_startrack", help="Toolkit parameter folder. Default: stem of --config")
    p.add_argument("--dataset-name", default="tnl2k", help="Toolkit dataset folder name. Default: tnl2k")
    p.add_argument("--delimiter", default="\\t", help="Delimiter for bbox txt. Default: tab, matching common SUTrack/OSTrack output.")

    p.add_argument("--seq-list", default="", help="Optional text file containing sequence names to evaluate")
    p.add_argument("--use-mpd", type=int, default=1, help="1: enable MPD/STARTrack, 0: pure SUTrack baseline")
    p.add_argument("--use-nlp", type=int, default=1, help="1: feed TNL2K language if available")
    p.add_argument("--debug", type=int, default=0)
    p.add_argument("--skip-existing", action="store_true", help="Skip sequence if prediction txt already exists")
    p.add_argument("--max-seq", type=int, default=0, help="For quick debug. 0 means all sequences.")
    p.add_argument("--save-debug", action="store_true", help="Save summary/language under result_dir/_debug")
    p.add_argument("--rebuild-each-seq", action="store_true", help="Rebuild and reload the tracker for every sequence. Default is to load once and reuse it.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    if not args.tracker_param:
        args.tracker_param = infer_tracker_param(args.config)

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
        torch.backends.cudnn.benchmark = True

    project_root = add_project_to_path(args.project_root)
    from lib.test.tracker.sutrack import get_tracker_class
    TrackerClass = get_tracker_class()

    seq_list = Path(args.seq_list) if args.seq_list else None
    sequences = discover_tnl2k_sequences(Path(args.tnl2k_root), seq_list=seq_list)
    if args.max_seq and args.max_seq > 0:
        sequences = sequences[: int(args.max_seq)]

    # GOT-10K script converts Stage4 checkpoints before passing them to the tracker.
    # Do the same here; otherwise a Stage4 ckpt with ckpt["disambiguator"] may be
    # loaded as if it were ckpt["model"], causing missing/unexpected keys.
    mpd_ckpt_for_test = None
    if bool(args.use_mpd) and args.mpd_ckpt:
        mpd_ckpt_for_test = make_startrack_test_ckpt(
            args.mpd_ckpt,
            Path(args.output_dir).expanduser().resolve() / "_converted_ckpts",
            "mpd_stage4",
        )

    params = build_params(
        config=args.config,
        sutrack_ckpt=args.sutrack_ckpt,
        mpd_ckpt=mpd_ckpt_for_test,
        use_mpd=bool(args.use_mpd),
        use_nlp=bool(args.use_nlp),
        debug=args.debug,
    )

    result_dir = get_result_dir(args)

    print("=" * 100)
    print("[INFO] MPDTrack TNL2K test with SUTrack-compatible output format")
    print(f"[INFO] project_root : {project_root}")
    print(f"[INFO] tnl2k_root   : {Path(args.tnl2k_root).expanduser().resolve()}")
    print(f"[INFO] config       : {args.config}")
    print(f"[INFO] sutrack_ckpt : {args.sutrack_ckpt}")
    print(f"[INFO] mpd_ckpt     : {args.mpd_ckpt}")
    print(f"[INFO] mpd_ckpt_test: {mpd_ckpt_for_test}")
    print(f"[INFO] use_mpd      : {bool(args.use_mpd)}")
    print(f"[INFO] use_nlp      : {bool(args.use_nlp)}")
    print(f"[INFO] result_dir   : {result_dir}")
    print(f"[INFO] layout       : output_dir/tracker_name/tracker_param/dataset_name/sequence.txt")
    print(f"[INFO] sequences    : {len(sequences)}")
    print("=" * 100)

    all_rows: List[Dict[str, Any]] = []

    # Load the tracker only once by default. The SUTRACK.initialize() function
    # resets online state, template state, and STARTrack/MPD memory for each
    # sequence, so reusing the object is much faster than reloading the model
    # and checkpoints 700 times. Use --rebuild-each-seq only for debugging.
    tracker = None
    if not args.rebuild_each_seq:
        print("[INFO] Building tracker once and reusing it for all sequences...")
        tracker = TrackerClass(params, "TNL2K")

    for i, seq in enumerate(sequences, start=1):
        print(f"[{i}/{len(sequences)}] ▶ {seq.name} | frames={len(seq.frames)} | nlp={'yes' if seq.language else 'no'}")
        try:
            if args.rebuild_each_seq:
                tracker_i = TrackerClass(params, "TNL2K")
            else:
                tracker_i = tracker
            assert tracker_i is not None

            summary = run_sequence(seq, args, tracker_i)
            m = summary.get("metrics", {})
            if m:
                print(
                    f"    AUC={m.get('success_auc', 0):.2f} P20={m.get('precision_20', 0):.2f} "
                    f"NPrec={m.get('norm_precision_auc', 0):.2f} mIoU={m.get('mean_iou', 0):.2f} "
                    f"FPS={summary.get('fps', 0):.2f}"
                )
                all_rows.append({"sequence": seq.name, "fps": summary.get("fps", 0.0), **m})
            else:
                print(f"    skipped existing: {summary.get('prediction_file')}")
        except Exception as e:
            print(f"[ERROR] sequence failed: {seq.name}, error={repr(e)}")
            if args.debug:
                raise

    if all_rows:
        debug_root = result_dir / "_debug"
        debug_root.mkdir(parents=True, exist_ok=True)
        csv_file = debug_root / "tnl2k_summary.csv"
        fieldnames = list(all_rows[0].keys())
        with csv_file.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

        valid_sum = sum(int(r.get("valid_frames", 0)) for r in all_rows)
        aggregate: Dict[str, float] = {"num_sequences": float(len(all_rows)), "valid_frames": float(valid_sum)}
        for key in ["success_auc", "precision_20", "norm_precision_auc", "mean_iou"]:
            aggregate[key] = sum(float(r[key]) * int(r.get("valid_frames", 0)) for r in all_rows) / max(valid_sum, 1)
        aggregate["fps"] = sum(float(r["fps"]) for r in all_rows) / max(len(all_rows), 1)
        (debug_root / "aggregate.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")

        print("=" * 100)
        print("[Quick Aggregate - official numbers should be computed by SUTrack toolkit]")
        for k, v in aggregate.items():
            print(f"{k:20s}: {v:.4f}")
        print(f"[Saved debug summary] {csv_file}")
        print(f"[Toolkit result dir] {result_dir}")
    else:
        print("[WARN] No valid sequence was evaluated.")


if __name__ == "__main__":
    main()