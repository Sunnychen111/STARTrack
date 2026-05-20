#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run only Stage4 STARTrack/SUTrack on LaSOT and save results in SUTrack-style format.

Output format:
    <save_root>/<tracker_name>/<run_name>/<seq>.txt

Each result file contains one bbox per frame:
    x\ty\tw\th

The parent process launches one fresh Python worker per sequence, so failures or
runtime pollution in one sequence will not affect the next one.

Resume behavior:
    By default, if <seq>.txt and <seq>_time.txt already exist and have the
    expected number of frames, the sequence is skipped and counted as success.
    Only missing/incomplete/failed sequences are re-inferred.
"""

import os
import sys
import json
import argparse
import subprocess
import traceback
import time
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np


DEFAULT_PROJECT_ROOT = "/home/cps/czl/STARTrack_v6"


def setup_project_root(project_root: str) -> str:
    project_root = os.path.abspath(project_root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return project_root


def read_rgb(path: str):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_txt_array(path: str, delimiter=None) -> np.ndarray:
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


def list_images(img_dir: str) -> List[Path]:
    img_dir = Path(img_dir)
    files = sorted([
        p for p in img_dir.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
    ])
    if len(files) == 0:
        raise FileNotFoundError(f"No image files found in: {img_dir}")
    return files


def load_lasot_sequence(root: str, seq_name: str) -> Tuple[List[Path], np.ndarray, str]:
    root_p = Path(root)
    cls_name = seq_name.split("-")[0]
    seq_dir = root_p / cls_name / seq_name
    img_dir = seq_dir / "img"
    gt_path = seq_dir / "groundtruth.txt"

    if not img_dir.is_dir():
        raise FileNotFoundError(f"LaSOT image dir not found: {img_dir}")
    if not gt_path.is_file():
        raise FileNotFoundError(f"LaSOT GT file not found: {gt_path}")

    img_files = list_images(str(img_dir))
    gt = load_txt_array(str(gt_path), delimiter=",")[:, :4].astype(np.float32)

    n = min(len(img_files), len(gt))
    return img_files[:n], gt[:n], str(seq_dir)


def discover_lasot_sequences(root: str) -> List[str]:
    root_p = Path(root)
    seqs = []
    for cls_dir in sorted([p for p in root_p.iterdir() if p.is_dir()]):
        for seq_dir in sorted([p for p in cls_dir.iterdir() if p.is_dir()]):
            if (seq_dir / "img").is_dir() and (seq_dir / "groundtruth.txt").is_file():
                seqs.append(seq_dir.name)
    return seqs


def read_seq_list(path: Optional[str]) -> Optional[List[str]]:
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
            #   umbrella-11
            #   LASOT umbrella-11
            seqs.append(parts[-1])
    return seqs


def save_sutrack_result_txt(path: str, bboxes: np.ndarray):
    """Save bbox results in SUTrack/LaSOT style: <seq>.txt.

    User requested no decimal places, so bbox values are rounded to integer and
    saved as tab-delimited x\ty\tw\th.
    """
    path_p = Path(path)
    path_p.parent.mkdir(parents=True, exist_ok=True)
    bboxes_int = np.rint(bboxes).astype(np.int64)
    np.savetxt(str(path_p), bboxes_int, fmt="%d", delimiter="\t")


def save_sutrack_time_txt(path: str, times: np.ndarray):
    """Save per-frame runtime in SUTrack/PyTracking style: <seq>_time.txt.

    Time is kept as seconds with 6 decimals; otherwise most frame times would be
    written as 0.
    """
    path_p = Path(path)
    path_p.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(path_p), np.asarray(times, dtype=np.float64), fmt="%.6f", delimiter="\t")


def _count_nonempty_lines(path: Path) -> int:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return -1


def _load_txt_shape(path: Path):
    try:
        arr = np.loadtxt(str(path), dtype=np.float64)
    except Exception as e:
        return None, repr(e)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        # A one-line bbox becomes shape (4,), while a one-line time becomes (1,).
        arr = arr.reshape(1, -1)
    return arr.shape, None


def expected_frame_count(root: str, seq: str, max_frames: Optional[int] = None) -> int:
    img_files, gt, _ = load_lasot_sequence(root, seq)
    n = min(len(img_files), len(gt))
    if max_frames is not None:
        n = min(n, int(max_frames))
    return int(n)


def check_existing_output(
    root: str,
    seq: str,
    result_path: Path,
    time_path: Path,
    max_frames: Optional[int] = None,
    require_time: bool = True,
):
    """Return (ok, info) for existing SUTrack-style output files.

    A sequence is treated as completed only when:
      1. <seq>.txt exists;
      2. it has the expected number of rows;
      3. each bbox row has 4 columns;
      4. <seq>_time.txt also exists and has the expected number of rows, unless require_time=False.
    """
    info = {
        "sequence": seq,
        "result_path": str(result_path),
        "time_path": str(time_path),
        "exists_result": result_path.is_file(),
        "exists_time": time_path.is_file(),
    }

    try:
        expected = expected_frame_count(root, seq, max_frames=max_frames)
    except Exception as e:
        info["reason"] = f"cannot_load_sequence: {repr(e)}"
        return False, info

    info["expected_frames"] = expected

    if not result_path.is_file():
        info["reason"] = "missing_result_txt"
        return False, info

    bbox_shape, bbox_err = _load_txt_shape(result_path)
    info["bbox_shape"] = bbox_shape
    if bbox_err is not None:
        info["reason"] = f"cannot_read_result_txt: {bbox_err}"
        return False, info
    if bbox_shape is None or len(bbox_shape) != 2 or bbox_shape[1] != 4:
        info["reason"] = f"bad_result_shape: {bbox_shape}"
        return False, info
    if bbox_shape[0] != expected:
        info["reason"] = f"result_frame_count_mismatch: got={bbox_shape[0]}, expected={expected}"
        return False, info

    if require_time:
        if not time_path.is_file():
            info["reason"] = "missing_time_txt"
            return False, info
        # Count lines instead of np.loadtxt shape, because time txt is one column.
        time_lines = _count_nonempty_lines(time_path)
        info["time_lines"] = time_lines
        if time_lines != expected:
            info["reason"] = f"time_frame_count_mismatch: got={time_lines}, expected={expected}"
            return False, info

    info["reason"] = "complete"
    return True, info


def make_startrack_test_ckpt(src_path: str, out_dir: str, tag: str) -> str:
    """Convert Stage4 checkpoint with key 'disambiguator' to test format key 'model'."""
    import torch

    src = Path(src_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(str(src), map_location="cpu")
    if "model" in ckpt:
        return str(src)
    if "disambiguator" not in ckpt:
        raise KeyError(f"Cannot find 'model' or 'disambiguator' in {src}. keys={list(ckpt.keys())}")

    dst = out / f"{tag}_for_test.pth"
    torch.save({
        "epoch": ckpt.get("epoch", -1),
        "model": ckpt["disambiguator"],
        "args": ckpt.get("args", {}),
        "metrics": ckpt.get("metrics", {}),
        "stage": ckpt.get("stage", "stage4_online_joint"),
        "source": str(src),
    }, str(dst))
    return str(dst)


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


def override_stage4_params(params, stage4_ckpt: str, mode: Optional[str] = None):
    stage4_ckpt = str(stage4_ckpt)

    # Common direct attributes used by different STARTrack versions.
    params.startrack_ckpt = stage4_ckpt
    params.reranker_ckpt = stage4_ckpt
    params.STARTRACK_CKPT = stage4_ckpt

    if mode is not None:
        params.startrack_state_mode = mode
        params.STARTRACK_STATE_MODE = mode

    # Config object fields used by some versions.
    if hasattr(params, "cfg") and hasattr(params.cfg, "MODEL"):
        try:
            params.cfg.MODEL.STARTRACK_CKPT = stage4_ckpt
            params.cfg.MODEL.USE_STARTRACK = True
        except Exception:
            pass
        if mode is not None:
            try:
                params.cfg.MODEL.STARTRACK_STATE_MODE = mode
            except Exception:
                pass

    return params


def force_project_dir_if_possible(project_root: str):
    """Best-effort fix for env_settings().prj_dir pointing to an old repo."""
    try:
        from lib.test.evaluation.environment import env_settings
        settings = env_settings()
        if hasattr(settings, "prj_dir"):
            settings.prj_dir = str(project_root)
    except Exception:
        pass


def run_stage4_one_sequence(args) -> dict:
    setup_project_root(args.project_root)
    force_project_dir_if_possible(args.project_root)

    import torch
    from lib.test.parameter.sutrack import parameters
    from lib.test.tracker.sutrack import get_tracker_class

    img_files, gt, seq_dir = load_lasot_sequence(args.root, args.seq)
    if args.max_frames is not None:
        img_files = img_files[:args.max_frames]
        gt = gt[:args.max_frames]

    result_dir = Path(args.save_root) / args.tracker_name / args.run_name
    result_path = result_dir / f"{args.seq}.txt"
    time_path = result_dir / f"{args.seq}_time.txt"

    print("=" * 100)
    print(f"[WORKER] seq={args.seq} frames={len(img_files)}")
    print(f"[WORKER] result={result_path}")
    print(f"[WORKER] time={time_path}")
    print(f"[WORKER] param={args.star_param}")
    print(f"[WORKER] base_checkpoint={args.base_checkpoint}")
    print(f"[WORKER] stage4_ckpt={args.stage4_ckpt}")
    print("=" * 100)

    params = parameters(args.star_param)
    params = patch_params(params)

    # Use the same base SUTrack checkpoint for the backbone/decoder.
    if args.base_checkpoint is not None:
        params.checkpoint = str(args.base_checkpoint)
        if hasattr(params, "cfg") and hasattr(params.cfg, "TEST"):
            try:
                params.cfg.TEST.CHECKPOINT = str(args.base_checkpoint)
            except Exception:
                pass

    params = override_stage4_params(params, args.stage4_ckpt, mode=args.star_mode)

    TrackerClass = get_tracker_class()
    tracker = TrackerClass(params, args.tracker_dataset_name or "lasot")

    # Stage4 evaluation is inference-only.
    try:
        tracker.network.eval()
    except Exception:
        pass

    pred = np.zeros_like(gt, dtype=np.float32)
    times = np.zeros((len(img_files),), dtype=np.float64)
    pred[0] = gt[0]

    first_img = read_rgb(str(img_files[0]))
    t0 = time.perf_counter()
    with torch.no_grad():
        tracker.initialize(first_img, {"init_bbox": gt[0].tolist()})
    times[0] = time.perf_counter() - t0

    for i in range(1, len(img_files)):
        img = read_rgb(str(img_files[i]))
        t0 = time.perf_counter()
        with torch.no_grad():
            out = tracker.track(img, info={})
        times[i] = time.perf_counter() - t0
        pred[i] = np.asarray(out["target_bbox"], dtype=np.float32)
        if args.print_interval > 0 and (i % args.print_interval == 0):
            print(f"[WORKER] {args.seq}: frame {i}/{len(img_files)-1}")

    save_sutrack_result_txt(str(result_path), pred)
    save_sutrack_time_txt(str(time_path), times)

    return {
        "sequence": args.seq,
        "frames": int(len(img_files)),
        "result_path": str(result_path),
        "time_path": str(time_path),
        "success": True,
    }


def build_worker_cmd(parent_args, seq: str, stage4_ckpt_for_test: str) -> List[str]:
    script = Path(__file__).resolve()
    cmd = [
        sys.executable, str(script),
        "--worker",
        "--project-root", str(parent_args.project_root),
        "--root", str(parent_args.root),
        "--seq", str(seq),
        "--star-param", str(parent_args.star_param),
        "--base-checkpoint", str(parent_args.base_checkpoint),
        "--stage4-ckpt", str(stage4_ckpt_for_test),
        "--save-root", str(parent_args.save_root),
        "--tracker-name", str(parent_args.tracker_name),
        "--run-name", str(parent_args.run_name),
        "--print-interval", str(parent_args.print_interval),
    ]
    if parent_args.tracker_dataset_name is not None:
        cmd += ["--tracker-dataset-name", str(parent_args.tracker_dataset_name)]
    if parent_args.star_mode is not None:
        cmd += ["--star-mode", str(parent_args.star_mode)]
    if parent_args.max_frames is not None:
        cmd += ["--max-frames", str(parent_args.max_frames)]
    return cmd


def run_parent(args):
    setup_project_root(args.project_root)

    if args.run_name is None:
        args.run_name = f"{args.star_param}_stage4"

    result_dir = Path(args.save_root) / args.tracker_name / args.run_name
    log_dir = result_dir / "_logs"
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    seqs = []
    if args.seq:
        seqs.append(args.seq)
    seq_list = read_seq_list(args.seq_list)
    if seq_list:
        seqs.extend(seq_list)
    if not seqs and args.discover:
        seqs = discover_lasot_sequences(args.root)

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
        raise ValueError("Please provide --seq, --seq-list, or use --discover.")

    converted_dir = result_dir / "_converted_ckpts"
    stage4_ckpt_for_test = make_startrack_test_ckpt(args.stage4_ckpt, str(converted_dir), "stage4")

    print("=" * 100)
    print("[INFO] Run Stage4 only on LaSOT")
    print(f"[INFO] project_root : {args.project_root}")
    print(f"[INFO] root         : {args.root}")
    print(f"[INFO] num seqs     : {len(seqs)}")
    print(f"[INFO] result dir   : {result_dir}")
    print(f"[INFO] stage4 ckpt  : {stage4_ckpt_for_test}")
    print("=" * 100)

    successes = []
    failures = []

    for idx, seq in enumerate(seqs, start=1):
        print("\n" + "=" * 100)
        print(f"[RUN] {idx}/{len(seqs)} seq={seq}")
        print("=" * 100)

        result_path = result_dir / f"{seq}.txt"
        time_path = result_dir / f"{seq}_time.txt"
        log_path = log_dir / f"{seq}.log"

        if args.skip_existing:
            ok_existing, existing_info = check_existing_output(
                root=args.root,
                seq=seq,
                result_path=result_path,
                time_path=time_path,
                max_frames=args.max_frames,
                require_time=not args.accept_existing_without_time,
            )
            if ok_existing:
                successes.append({
                    "sequence": seq,
                    "result_path": str(result_path),
                    "time_path": str(time_path),
                    "log_path": str(log_path) if log_path.is_file() else None,
                    "skipped_existing": True,
                    "expected_frames": existing_info.get("expected_frames"),
                })
                print(f"[SKIP] {seq}: existing result is complete -> {result_path}, {time_path}")
                continue
            else:
                print(f"[RERUN] {seq}: existing output incomplete/missing, reason={existing_info.get('reason')}")

        cmd = build_worker_cmd(args, seq, stage4_ckpt_for_test)

        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(args.project_root),
            )
        except Exception as e:
            failures.append({"sequence": seq, "error": repr(e), "returncode": None})
            log_path.write_text(traceback.format_exc(), encoding="utf-8")
            print(f"[FAIL] {seq}: {repr(e)}")
            continue

        log_path.write_text(proc.stdout or "", encoding="utf-8")

        result_path = result_dir / f"{seq}.txt"
        time_path = result_dir / f"{seq}_time.txt"
        if proc.returncode == 0 and result_path.is_file() and time_path.is_file():
            successes.append({
                "sequence": seq,
                "result_path": str(result_path),
                "time_path": str(time_path),
                "log_path": str(log_path),
                "skipped_existing": False,
            })
            print(f"[OK] {seq} -> {result_path}, {time_path}")
        else:
            failures.append({
                "sequence": seq,
                "returncode": proc.returncode,
                "result_path": str(result_path),
                "time_path": str(time_path),
                "log_path": str(log_path),
                "tail": "\n".join((proc.stdout or "").splitlines()[-40:]),
            })
            print(f"[FAIL] {seq}, returncode={proc.returncode}. log={log_path}")
            if not args.continue_on_error:
                break

    skipped_existing = sum(1 for x in successes if x.get("skipped_existing"))
    newly_run_success = sum(1 for x in successes if not x.get("skipped_existing"))

    summary = {
        "total": len(seqs),
        "success": len(successes),
        "skipped_existing": skipped_existing,
        "newly_run_success": newly_run_success,
        "failed": len(failures),
        "successes": successes,
        "failures": failures,
        "result_dir": str(result_dir),
    }

    summary_json = result_dir / "stage4_run_summary.json"
    summary_txt = result_dir / "stage4_run_summary.txt"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"total: {len(seqs)}\n")
        f.write(f"success: {len(successes)}\n")
        f.write(f"skipped_existing: {skipped_existing}\n")
        f.write(f"newly_run_success: {newly_run_success}\n")
        f.write(f"failed: {len(failures)}\n\n")
        f.write("[SUCCESS]\n")
        for item in successes:
            f.write(f"{item['sequence']}\t{item['result_path']}\t{item['time_path']}\n")
        f.write("\n[FAILED]\n")
        for item in failures:
            f.write(f"{item['sequence']}\treturncode={item.get('returncode')}\tlog={item.get('log_path')}\n")

    print("\n" + "=" * 100)
    print("[DONE] Stage4-only LaSOT run finished")
    print(f"[SUMMARY] total={len(seqs)}, success={len(successes)}, skipped_existing={skipped_existing}, newly_run_success={newly_run_success}, failed={len(failures)}")
    print(f"[SUMMARY] result_dir={result_dir}")
    print(f"[SUMMARY] summary_json={summary_json}")
    print(f"[SUMMARY] summary_txt={summary_txt}")
    print("=" * 100)

    # Return non-zero only when user wants strict failure behavior.
    if failures and not args.allow_partial_exit_zero:
        return 1
    return 0


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=str, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--root", type=str, default="/home/cps/SOT_dataset/LaSOT-Extension")
    parser.add_argument("--seq", type=str, default=None)
    parser.add_argument("--seq-list", type=str, default="/home/cps/czl/STARTrack_v6/extention.txt")
    parser.add_argument("--discover", action="store_true", help="Discover all LaSOT sequences from root/class/seq folders")
    parser.add_argument("--max-seqs", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)

    parser.add_argument("--star-param", type=str, default="sutrack_b384_startrack")
    parser.add_argument("--tracker-dataset-name", type=str, default="lasot")
    parser.add_argument("--base-checkpoint", type=str, default="/home/cps/czl/STARTrack_v6/checkpoints/sutrack_b384/SUTRACK_ep0180.pth.tar")
    parser.add_argument("--stage4-ckpt", type=str, default="/home/cps/czl/STARTrack_v6/checkpoints/stage4_lasot_three_stage_b384_v2/last.pth")
    parser.add_argument("--star-mode", type=str, default='safe-perdicted')

    parser.add_argument("--save-root", type=str, default="/home/cps/czl/STARTrack_v6/test/tracking_results/MPDTrack/MPDTrack_5.19/lasot")
    parser.add_argument("--tracker-name", type=str, default="MPDTrack")
    parser.add_argument("--run-name", type=str, default=None)

    parser.add_argument("--continue-on-error", action="store_true", default=True)
    parser.add_argument("--stop-on-error", dest="continue_on_error", action="store_false")
    parser.add_argument("--allow-partial-exit-zero", action="store_true", default=True)
    parser.add_argument("--strict-exit-code", dest="allow_partial_exit_zero", action="store_false")
    parser.add_argument("--print-interval", type=int, default=200)

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip sequences whose <seq>.txt and <seq>_time.txt already exist and match expected frame count. Default: True.",
    ) 
    parser.add_argument(
        "--rerun-existing",
        dest="skip_existing",
        action="store_false",
        help="Ignore existing txt files and re-run all selected sequences.",
    )
    parser.add_argument(
        "--accept-existing-without-time",
        action="store_true",
        help="Treat <seq>.txt as complete even when <seq>_time.txt is missing. Not recommended if you need SUTrack-style time files.",
    )

    # Internal worker mode.
    parser.add_argument("--worker", action="store_true")
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    if args.save_root is None:
        args.save_root = str(Path(args.project_root) / "test" / "tracking_results")

    if args.worker:
        try:
            info = run_stage4_one_sequence(args)
            print(json.dumps(info, indent=2, ensure_ascii=False))
            return 0
        except Exception:
            traceback.print_exc()
            return 2

    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
