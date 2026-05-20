#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate raw tracking results on mined SOI clips.

Input:
  1) selected_clips.csv or selected_sequences.json from SOI mining.
  2) evidence/<Tracker>/<Dataset>/<Sequence>.json for GT by frame_id.
  3) raw_result txt/csv files for each tracker.

Metrics:
  - Success AUC: mean success rate over IoU thresholds [0, 1].
  - Precision@20: center error <= 20 px.
  - PNorm AUC: normalized center precision AUC over thresholds [0, 0.5].

Default frame alignment:
  raw prediction line index = frame_id - 1
because evidence frame_id is usually 1-based and raw results usually include frame 1.
Use --pred-index-offset 0 if your raw txt starts from frame 2.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ClipRecord:
    clip_id: int
    dataset: str
    sequence: str
    start_frame: int
    end_frame: int
    soi_frame_count: int = 0
    rank_score: float = 0.0


@dataclass
class FrameEvalRecord:
    tracker: str
    dataset: str
    sequence: str
    clip_id: int
    frame_id: int
    pred_index: int
    iou: float
    center_error: float
    norm_center_error: float
    gt_bbox: str
    pred_bbox: str
    raw_file: str


@dataclass
class MetricRecord:
    tracker: str
    scope: str
    dataset: str
    sequence: str
    clip_id: str
    num_frames: int
    success_auc: float
    precision_20: float
    pnorm_auc: float
    mean_iou: float
    mean_center_error: float
    mean_norm_center_error: float


def parse_csv_arg(s: str) -> List[str]:
    return [x.strip() for x in s.split(',') if x.strip()]


def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def valid_bbox_xywh(box: Any, eps: float = 1e-6) -> bool:
    if not isinstance(box, (list, tuple, np.ndarray)) or len(box) < 4:
        return False
    vals = [safe_float(v) for v in list(box)[:4]]
    return not any(v is None for v in vals) and vals[2] > eps and vals[3] > eps


def bbox_to_str(box: Sequence[Any]) -> str:
    vals = []
    for v in list(box)[:4]:
        fv = safe_float(v)
        vals.append(0.0 if fv is None else fv)
    return '[' + ','.join(f'{v:.3f}' for v in vals) + ']'


def xywh_to_xyxy(box: Sequence[float]) -> np.ndarray:
    x, y, w, h = [float(v) for v in box[:4]]
    return np.asarray([x, y, x + max(0.0, w), y + max(0.0, h)], dtype=np.float64)


def iou_xywh(a: Sequence[float], b: Sequence[float]) -> float:
    if not valid_bbox_xywh(a) or not valid_bbox_xywh(b):
        return 0.0
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 1e-12 else float(inter / union)


def center_xywh(box: Sequence[float]) -> np.ndarray:
    x, y, w, h = [float(v) for v in box[:4]]
    return np.asarray([x + 0.5 * w, y + 0.5 * h], dtype=np.float64)


def center_error_xywh(pred: Sequence[float], gt: Sequence[float]) -> float:
    if not valid_bbox_xywh(pred) or not valid_bbox_xywh(gt):
        return float('inf')
    return float(np.linalg.norm(center_xywh(pred) - center_xywh(gt)))


def norm_denominator(gt: Sequence[float], mode: str) -> float:
    _, _, w, h = [float(v) for v in gt[:4]]
    if mode == 'sqrt_area':
        return max(math.sqrt(max(w, 1e-6) * max(h, 1e-6)), 1e-6)
    if mode == 'diag':
        return max(math.sqrt(max(w, 1e-6) ** 2 + max(h, 1e-6) ** 2), 1e-6)
    if mode == 'max_side':
        return max(max(w, h), 1e-6)
    if mode == 'min_side':
        return max(min(w, h), 1e-6)
    raise ValueError(f'Unknown norm mode: {mode}')


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, '') for k in fieldnames})


def load_selected_clips(path: Path) -> List[ClipRecord]:
    if not path.exists():
        raise FileNotFoundError(f'selected clips not found: {path}')
    clips: List[ClipRecord] = []
    if path.suffix.lower() == '.json':
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, list):
            raise RuntimeError('selected json must be a list')
        for i, r in enumerate(data, start=1):
            clips.append(ClipRecord(
                clip_id=i,
                dataset=str(r['dataset']),
                sequence=str(r['sequence']),
                start_frame=int(r['start_frame']),
                end_frame=int(r['end_frame']),
                soi_frame_count=int(r.get('soi_frame_count', 0) or 0),
                rank_score=float(r.get('rank_score', 0.0) or 0.0),
            ))
        return clips
    with path.open('r', encoding='utf-8', newline='') as f:
        for i, r in enumerate(csv.DictReader(f), start=1):
            clips.append(ClipRecord(
                clip_id=i,
                dataset=str(r['dataset']),
                sequence=str(r['sequence']),
                start_frame=int(float(r['start_frame'])),
                end_frame=int(float(r['end_frame'])),
                soi_frame_count=int(float(r.get('soi_frame_count', 0) or 0)),
                rank_score=float(r.get('rank_score', 0.0) or 0.0),
            ))
    return clips


def infer_selected_frames_path(selected_clips: Path) -> Optional[Path]:
    p = selected_clips.parent / 'selected_frames.csv'
    return p if p.exists() else None


def load_selected_frame_ids(path: Path) -> Dict[Tuple[str, str, int], List[int]]:
    out: Dict[Tuple[str, str, int], List[int]] = {}
    with path.open('r', encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            ds, seq = str(r['dataset']), str(r['sequence'])
            fid = int(float(r['frame_id']))
            if 'clip_id' in r and r['clip_id'] not in ['', None]:
                cid = int(float(r['clip_id']))
            elif 'clip_start' in r and 'clip_end' in r:
                cid = int(float(r['clip_start'])) * 1000000 + int(float(r['clip_end']))
            else:
                cid = -1
            out.setdefault((ds, seq, cid), []).append(fid)
    return {k: sorted(set(v)) for k, v in out.items()}


def load_json(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            return None, 'json_root_not_dict'
        return data, None
    except Exception as e:
        return None, repr(e)


def build_gt_from_evidence(data: Dict[str, Any]) -> Dict[int, List[float]]:
    out: Dict[int, List[float]] = {}
    frames = data.get('frames', [])
    if not isinstance(frames, list):
        return out
    for fr in frames:
        if not isinstance(fr, dict):
            continue
        try:
            fid = int(fr.get('frame_id'))
            gt = fr.get('gt_bbox')
            if valid_bbox_xywh(gt):
                out[fid] = [float(v) for v in gt[:4]]
        except Exception:
            continue
    return out


def find_evidence_json(evidence_root: Path, evidence_trackers: List[str], dataset: str, sequence: str) -> Optional[Path]:
    for trk in evidence_trackers:
        p = evidence_root / trk / dataset / f'{sequence}.json'
        if p.exists():
            return p
    return None


def load_gt_cache(clips: List[ClipRecord], evidence_root: Path, evidence_trackers: List[str]) -> Tuple[Dict[Tuple[str, str], Dict[int, List[float]]], List[Dict[str, Any]]]:
    cache: Dict[Tuple[str, str], Dict[int, List[float]]] = {}
    bad: List[Dict[str, Any]] = []
    for c in clips:
        key = (c.dataset, c.sequence)
        if key in cache:
            continue
        p = find_evidence_json(evidence_root, evidence_trackers, c.dataset, c.sequence)
        if p is None:
            bad.append({'dataset': c.dataset, 'sequence': c.sequence, 'issue': 'missing_evidence_json', 'file': ''})
            cache[key] = {}
            continue
        data, err = load_json(p)
        if err or data is None:
            bad.append({'dataset': c.dataset, 'sequence': c.sequence, 'issue': err or 'bad_json', 'file': str(p)})
            cache[key] = {}
            continue
        cache[key] = build_gt_from_evidence(data)
    return cache, bad


def parse_raw_result_file(path: Path) -> List[List[float]]:
    preds: List[List[float]] = []
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals: List[float] = []
            for p in re.split(r'[,\s]+', line):
                if not p:
                    continue
                try:
                    vals.append(float(p))
                except ValueError:
                    pass
            if len(vals) >= 4:
                preds.append(vals[:4])
    return preds


def format_raw_pattern(pattern: str, raw_root: Path, tracker: str, dataset: str, sequence: str, param: str) -> Path:
    return Path(pattern.format(raw_root=str(raw_root), tracker=tracker, dataset=dataset, sequence=sequence, param=param)).expanduser()


def candidate_raw_paths(raw_root: Path, tracker: str, dataset: str, sequence: str, param: str) -> List[Path]:
    names = [f'{sequence}.txt', f'{sequence}.csv', f'{sequence}_001.txt', f'{sequence}_001.csv']
    dirs = [
        raw_root / tracker / dataset,
        raw_root / tracker / param / dataset,
        raw_root / tracker / dataset / param,
        raw_root / tracker,
        raw_root / dataset / tracker,
        raw_root / dataset / tracker / param,
        raw_root / param / tracker / dataset,
        raw_root / param / dataset / tracker,
    ]
    return [d / name for d in dirs for name in names]


def find_raw_result_file(raw_root: Path, tracker: str, dataset: str, sequence: str, param: str, raw_pattern: str = '') -> Optional[Path]:
    if raw_pattern:
        p = format_raw_pattern(raw_pattern, raw_root, tracker, dataset, sequence, param)
        return p if p.exists() else None
    for p in candidate_raw_paths(raw_root, tracker, dataset, sequence, param):
        if p.exists():
            return p
    tracker_root = raw_root / tracker
    if tracker_root.exists():
        matches: List[Path] = []
        for suffix in ['.txt', '.csv']:
            matches.extend(tracker_root.rglob(f'{sequence}{suffix}'))
        if matches:
            return sorted(matches, key=lambda x: len(str(x)))[0]
    matches = []
    for suffix in ['.txt', '.csv']:
        matches.extend(raw_root.rglob(f'{sequence}{suffix}'))
    if matches:
        def score(p: Path) -> Tuple[int, int]:
            s = str(p)
            hit = int(tracker in s) + int(dataset in s)
            return (-hit, len(s))
        return sorted(matches, key=score)[0]
    return None


def compute_metric_record(tracker: str, scope: str, dataset: str, sequence: str, clip_id: str, records: List[FrameEvalRecord]) -> MetricRecord:
    if not records:
        return MetricRecord(tracker, scope, dataset, sequence, str(clip_id), 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    ious = np.asarray([r.iou for r in records], dtype=np.float64)
    ce = np.asarray([r.center_error for r in records], dtype=np.float64)
    nce = np.asarray([r.norm_center_error for r in records], dtype=np.float64)
    success_thresholds = np.linspace(0.0, 1.0, 101)
    success_auc = float(np.asarray([(ious >= t).mean() for t in success_thresholds]).mean())
    precision_20 = float((ce <= 20.0).mean())
    norm_thresholds = np.linspace(0.0, 0.5, 51)
    pnorm_auc = float(np.asarray([(nce <= t).mean() for t in norm_thresholds]).mean())
    return MetricRecord(
        tracker=tracker,
        scope=scope,
        dataset=dataset,
        sequence=sequence,
        clip_id=str(clip_id),
        num_frames=len(records),
        success_auc=success_auc,
        precision_20=precision_20,
        pnorm_auc=pnorm_auc,
        mean_iou=float(np.mean(ious)),
        mean_center_error=float(np.mean(ce)),
        mean_norm_center_error=float(np.mean(nce)),
    )


def frames_for_clip(clip: ClipRecord, gt_map: Dict[int, List[float]], frame_mode: str, selected_frame_map: Optional[Dict[Tuple[str, str, int], List[int]]] = None) -> List[int]:
    if frame_mode == 'clip_all':
        return [fid for fid in range(clip.start_frame, clip.end_frame + 1) if fid in gt_map]
    if frame_mode == 'sequence_all':
        return sorted(gt_map.keys())
    if frame_mode == 'selected_frames':
        if selected_frame_map is None:
            raise RuntimeError('--frame-mode selected_frames requires selected_frames.csv')
        direct_key = (clip.dataset, clip.sequence, clip.clip_id)
        if direct_key in selected_frame_map:
            return [fid for fid in selected_frame_map[direct_key] if fid in gt_map]
        range_key = (clip.dataset, clip.sequence, clip.start_frame * 1000000 + clip.end_frame)
        if range_key in selected_frame_map:
            return [fid for fid in selected_frame_map[range_key] if fid in gt_map]
        out: List[int] = []
        for (ds, seq, _), fids in selected_frame_map.items():
            if ds == clip.dataset and seq == clip.sequence:
                out.extend([fid for fid in fids if clip.start_frame <= fid <= clip.end_frame and fid in gt_map])
        return sorted(set(out))
    raise ValueError(f'Unknown frame_mode: {frame_mode}')


def evaluate(args: argparse.Namespace) -> None:
    selected_path = Path(args.selected_clips).expanduser().resolve()
    evidence_root = Path(args.evidence_root).expanduser().resolve()
    raw_root = Path(args.raw_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    ensure_dir(output_root)

    trackers = parse_csv_arg(args.trackers)
    datasets_filter = set(parse_csv_arg(args.datasets)) if args.datasets else set()
    evidence_trackers = parse_csv_arg(args.evidence_trackers) if args.evidence_trackers else trackers

    clips = load_selected_clips(selected_path)
    if datasets_filter:
        clips = [c for c in clips if c.dataset in datasets_filter]

    print('[INFO] SOI raw result evaluation')
    print(f'[INFO] selected_clips    = {selected_path}')
    print(f'[INFO] evidence_root     = {evidence_root}')
    print(f'[INFO] raw_root          = {raw_root}')
    print(f'[INFO] output_root       = {output_root}')
    print(f'[INFO] trackers          = {trackers}')
    print(f'[INFO] evidence_trackers = {evidence_trackers}')
    print(f'[INFO] num_clips         = {len(clips)}')
    print(f'[INFO] frame_mode        = {args.frame_mode}')
    print(f'[INFO] pred_index_offset = {args.pred_index_offset}')

    selected_frame_map = None
    if args.frame_mode == 'selected_frames':
        sfp = Path(args.selected_frames).expanduser().resolve() if args.selected_frames else infer_selected_frames_path(selected_path)
        if sfp is None:
            raise RuntimeError('selected_frames.csv not found. Provide --selected-frames.')
        print(f'[INFO] selected_frames   = {sfp}')
        selected_frame_map = load_selected_frame_ids(sfp)

    gt_cache, bad_gt = load_gt_cache(clips, evidence_root, evidence_trackers)
    raw_cache: Dict[Tuple[str, str, str], Tuple[Optional[Path], List[List[float]]]] = {}
    frame_records: List[FrameEvalRecord] = []
    bad_rows: List[Dict[str, Any]] = []

    for tracker in trackers:
        print(f'[EVAL] tracker={tracker}')
        evaluated_sequence_keys = set()
        for clip in clips:
            if args.frame_mode == 'sequence_all':
                seq_key = (tracker, clip.dataset, clip.sequence)
                if seq_key in evaluated_sequence_keys:
                    continue
                evaluated_sequence_keys.add(seq_key)

            gt_map = gt_cache.get((clip.dataset, clip.sequence), {})
            if not gt_map:
                bad_rows.append({'tracker': tracker, 'dataset': clip.dataset, 'sequence': clip.sequence, 'clip_id': clip.clip_id, 'issue': 'missing_gt_map', 'file': ''})
                continue
            raw_key = (tracker, clip.dataset, clip.sequence)
            if raw_key not in raw_cache:
                raw_file = find_raw_result_file(raw_root, tracker, clip.dataset, clip.sequence, args.param, args.raw_pattern)
                raw_cache[raw_key] = (raw_file, parse_raw_result_file(raw_file) if raw_file else [])
            raw_file, preds = raw_cache[raw_key]
            if raw_file is None or not preds:
                bad_rows.append({'tracker': tracker, 'dataset': clip.dataset, 'sequence': clip.sequence, 'clip_id': clip.clip_id, 'issue': 'missing_or_empty_raw_result', 'file': '' if raw_file is None else str(raw_file)})
                continue
            fids = frames_for_clip(clip, gt_map, args.frame_mode, selected_frame_map)
            if not fids:
                bad_rows.append({'tracker': tracker, 'dataset': clip.dataset, 'sequence': clip.sequence, 'clip_id': clip.clip_id, 'issue': 'no_valid_eval_frames_in_clip', 'file': str(raw_file)})
                continue
            for fid in fids:
                pred_idx = int(fid + args.pred_index_offset)
                if pred_idx < 0 or pred_idx >= len(preds):
                    bad_rows.append({'tracker': tracker, 'dataset': clip.dataset, 'sequence': clip.sequence, 'clip_id': clip.clip_id, 'issue': f'pred_index_out_of_range:frame={fid},idx={pred_idx},len={len(preds)}', 'file': str(raw_file)})
                    continue
                gt_box = gt_map[fid]
                pred_box = preds[pred_idx]
                if not valid_bbox_xywh(pred_box) or not valid_bbox_xywh(gt_box):
                    bad_rows.append({'tracker': tracker, 'dataset': clip.dataset, 'sequence': clip.sequence, 'clip_id': clip.clip_id, 'issue': f'invalid_bbox:frame={fid}', 'file': str(raw_file)})
                    continue
                ce = center_error_xywh(pred_box, gt_box)
                frame_records.append(FrameEvalRecord(
                    tracker=tracker,
                    dataset=clip.dataset,
                    sequence=clip.sequence,
                    clip_id=clip.clip_id,
                    frame_id=fid,
                    pred_index=pred_idx,
                    iou=iou_xywh(pred_box, gt_box),
                    center_error=ce,
                    norm_center_error=ce / norm_denominator(gt_box, args.norm_mode),
                    gt_bbox=bbox_to_str(gt_box),
                    pred_bbox=bbox_to_str(pred_box),
                    raw_file=str(raw_file),
                ))
        print(f'  frames evaluated so far: {len(frame_records)}')

    metric_rows: List[MetricRecord] = []
    for tracker in trackers:
        recs = [r for r in frame_records if r.tracker == tracker]
        metric_rows.append(compute_metric_record(tracker, 'overall', 'ALL', 'ALL', 'ALL', recs))
    for tracker in trackers:
        for ds in sorted(set(c.dataset for c in clips)):
            recs = [r for r in frame_records if r.tracker == tracker and r.dataset == ds]
            metric_rows.append(compute_metric_record(tracker, 'dataset', ds, 'ALL', 'ALL', recs))
    for tracker, ds, seq in sorted(set((r.tracker, r.dataset, r.sequence) for r in frame_records)):
        recs = [r for r in frame_records if r.tracker == tracker and r.dataset == ds and r.sequence == seq]
        metric_rows.append(compute_metric_record(tracker, 'sequence', ds, seq, 'ALL', recs))
    for tracker, ds, seq, cid in sorted(set((r.tracker, r.dataset, r.sequence, r.clip_id) for r in frame_records)):
        recs = [r for r in frame_records if r.tracker == tracker and r.dataset == ds and r.sequence == seq and r.clip_id == cid]
        metric_rows.append(compute_metric_record(tracker, 'clip', ds, seq, str(cid), recs))

    write_csv(output_root / 'soi_manifest.csv', [asdict(c) for c in clips], list(ClipRecord.__dataclass_fields__.keys()))
    write_csv(output_root / 'evaluated_frames.csv', [asdict(r) for r in frame_records], list(FrameEvalRecord.__dataclass_fields__.keys()))
    write_csv(output_root / 'metrics.csv', [asdict(r) for r in metric_rows], list(MetricRecord.__dataclass_fields__.keys()))
    write_csv(output_root / 'metrics_overall.csv', [asdict(r) for r in metric_rows if r.scope == 'overall'], list(MetricRecord.__dataclass_fields__.keys()))
    write_csv(output_root / 'metrics_per_dataset.csv', [asdict(r) for r in metric_rows if r.scope == 'dataset'], list(MetricRecord.__dataclass_fields__.keys()))
    write_csv(output_root / 'metrics_per_sequence.csv', [asdict(r) for r in metric_rows if r.scope == 'sequence'], list(MetricRecord.__dataclass_fields__.keys()))
    write_csv(output_root / 'metrics_per_clip.csv', [asdict(r) for r in metric_rows if r.scope == 'clip'], list(MetricRecord.__dataclass_fields__.keys()))
    write_csv(output_root / 'bad_rows.csv', bad_rows, ['tracker', 'dataset', 'sequence', 'clip_id', 'issue', 'file'])

    with (output_root / 'summary.txt').open('w', encoding='utf-8') as f:
        f.write('SOI raw result evaluation summary\n')
        f.write('=================================\n\n')
        f.write(f'selected_clips: {selected_path}\n')
        f.write(f'raw_root: {raw_root}\n')
        f.write(f'frame_mode: {args.frame_mode}\n')
        f.write(f'norm_mode: {args.norm_mode}\n')
        f.write(f'pred_index_offset: {args.pred_index_offset}\n\n')
        f.write('Overall metrics:\n')
        for r in [x for x in metric_rows if x.scope == 'overall']:
            f.write(f'  {r.tracker:12s} frames={r.num_frames:6d} AUC={r.success_auc:.4f} Prec20={r.precision_20:.4f} PNorm={r.pnorm_auc:.4f} MeanIoU={r.mean_iou:.4f}\n')
        f.write('\nDataset metrics:\n')
        for r in [x for x in metric_rows if x.scope == 'dataset']:
            f.write(f'  {r.tracker:12s} {r.dataset:10s} frames={r.num_frames:6d} AUC={r.success_auc:.4f} Prec20={r.precision_20:.4f} PNorm={r.pnorm_auc:.4f}\n')
        f.write(f'\nBad rows: {len(bad_rows)}\n')
        f.write(f'GT loading issues: {len(bad_gt)}\n')

    report = {
        'selected_clips': str(selected_path),
        'evidence_root': str(evidence_root),
        'raw_root': str(raw_root),
        'output_root': str(output_root),
        'trackers': trackers,
        'num_clips': len(clips),
        'num_evaluated_frames': len(frame_records),
        'num_bad_rows': len(bad_rows),
        'gt_loading_issues': bad_gt,
        'frame_mode': args.frame_mode,
        'norm_mode': args.norm_mode,
        'pred_index_offset': args.pred_index_offset,
    }
    with (output_root / 'report.json').open('w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print('\n[DONE]')
    print(f'  evaluated frames: {len(frame_records)}')
    print(f'  metric rows:      {len(metric_rows)}')
    print(f'  bad rows:         {len(bad_rows)}')
    print(f'  output root:      {output_root}')
    print('\n[OVERALL]')
    for r in [x for x in metric_rows if x.scope == 'overall']:
        print(f'  {r.tracker:12s} frames={r.num_frames:6d} AUC={r.success_auc:.4f} Prec20={r.precision_20:.4f} PNorm={r.pnorm_auc:.4f} MeanIoU={r.mean_iou:.4f}')


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser('Evaluate raw tracking results on SOI clips')
    p.add_argument('--selected-clips', type=str, required=True, help='selected_clips.csv or selected_sequences.json')
    p.add_argument('--selected-frames', type=str, default='', help='optional selected_frames.csv when --frame-mode selected_frames')
    p.add_argument('--evidence-root', type=str, required=True)
    p.add_argument('--evidence-trackers', type=str, default='', help='trackers used to load GT from evidence; default = --trackers')
    p.add_argument('--raw-root', type=str, required=True)
    p.add_argument('--raw-pattern', type=str, default='', help='pattern with {raw_root},{tracker},{dataset},{sequence},{param}')
    p.add_argument('--param', type=str, default='', help='optional param placeholder for raw layout')
    p.add_argument('--trackers', type=str, required=True)
    p.add_argument('--datasets', type=str, default='', help='optional dataset filter')
    p.add_argument('--output-root', type=str, required=True)
    p.add_argument('--frame-mode', type=str, default='clip_all', choices=['clip_all', 'selected_frames', 'sequence_all'])
    p.add_argument('--pred-index-offset', type=int, default=-1, help='prediction line index = frame_id + offset')
    p.add_argument('--norm-mode', type=str, default='sqrt_area', choices=['sqrt_area', 'diag', 'max_side', 'min_side'])
    return p


def main() -> None:
    args = build_argparser().parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()