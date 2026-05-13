#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm
import os

from focal_rank_loss import RerankLoss
from lib.config.sutrack.config import cfg, update_config_from_file
from lib.models.sutrack import build_sutrack
from lib.models.sutrack.post_decoder_disambiguator import (
    PostDecoderDisambiguator,
    sample_feature_at_peaks,
    topk_peaks_nms,
)
from lib.test.tracker.utils import sample_target, transform_image_to_crop
from lib.test.utils.hann import hann2d
from lib.train.admin.settings import Settings
from lib.train.base_functions import names2datasets
from lib.train.data import LTRLoader, opencv_loader, sampler
from lib.utils import TensorDict
from lib.utils.box_ops import box_iou, box_xywh_to_xyxy


class RawSequenceProcessing:
    """Keep raw frames from TrackingSampler and mark the sample valid."""

    def __call__(self, data):
        data["valid"] = True
        return data


def _load_train_yaml(path):
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"train yaml not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"train yaml must be a mapping/dict, got {type(data)} from {path}")
    return data


def _yaml_defaults_from_dict(data):
    """Flatten the YAML structure used by stage4_got10k_from_lasot_hard.yaml.

    YAML sections are mapped to argparse attribute names. Unknown YAML keys are
    ignored deliberately, so the YAML can keep notes such as `cli` or
    `second_stage_decoder_finetune` without breaking this script.
    """
    if not data:
        return {}

    section_key_map = {
        "paths": {
            "sutrack_ckpt": "sutrack_ckpt",
            "reranker_ckpt": "reranker_ckpt",
            "output_dir": "output_dir",
            "config": "config",
        },
        "data": {
            "dataset": "dataset",
            "use_lmdb": "use_lmdb",
            "stage4_sampling_mode": "stage4_sampling_mode",
            "hard_frame_list": "hard_frame_list",
            "hard_seq_list": "hard_seq_list",
            "hard_frame_ratio": "hard_frame_ratio",
            "hard_seq_ratio": "hard_seq_ratio",
            "full_ratio": "full_ratio",
            "anchor_jitter": "anchor_jitter",
            "min_anchor_valid_frames": "min_anchor_valid_frames",
            "max_anchor_retry": "max_anchor_retry",
        },
        "training": {
            "epochs": "epochs",
            "samples_per_epoch": "samples_per_epoch",
            "rollout_len": "rollout_len",
            "topk": "topk",
            "batch_size": "batch_size",
            "num_workers": "num_workers",
            "device": "device",
            "seed": "seed",
            "freeze_encoder": "freeze_encoder",
            "unfreeze_decoder": "unfreeze_decoder",
            "detach_history": "detach_history",
            "history_len": "history_len",
            "update_mode": "update_mode",
            "coach_epochs": "coach_epochs",
            "scheduled_takeover_epochs": "scheduled_takeover_epochs",
            "takeover_prob_thresh": "takeover_prob_thresh",
            "safe_pred_prob_thresh": "safe_pred_prob_thresh",
            "max_motion_norm": "max_motion_norm",
            "max_scale_ratio": "max_scale_ratio",
            "use_heuristic_reliability_gate": "use_heuristic_reliability_gate",
            "reliability_update_thresh": "reliability_update_thresh",
            "rel_target_weight": "rel_target_weight",
            "rel_score_weight": "rel_score_weight",
            "rel_margin_weight": "rel_margin_weight",
            "rel_ambiguity_weight": "rel_ambiguity_weight",
            "rel_motion_weight": "rel_motion_weight",
            "rel_history_weight": "rel_history_weight",
            "teacher_prob_start": "teacher_prob_start",
            "teacher_prob_end": "teacher_prob_end",
            "teacher_decay_epochs": "teacher_decay_epochs",
        },
        "loss": {
            "iou_thresh": "iou_thresh",
            "apply_gain_thr": "apply_gain_thr",
            "best_iou_thr": "best_iou_thr",
            "apply_threshold": "apply_threshold",
            "rank_lambda": "rank_lambda",
            "margin": "margin",
            "use_heuristic_reliability_gate": "use_heuristic_reliability_gate",
            "reliability_update_thresh": "reliability_update_thresh",
            "rel_target_weight": "rel_target_weight",
            "rel_score_weight": "rel_score_weight",
            "rel_margin_weight": "rel_margin_weight",
            "rel_ambiguity_weight": "rel_ambiguity_weight",
            "rel_motion_weight": "rel_motion_weight",
            "rel_history_weight": "rel_history_weight",
        },
        "optimizer": {
            "lr_reranker": "lr_reranker",
            "lr_decoder": "lr_decoder",
            "weight_decay": "weight_decay",
            "grad_clip_norm": "grad_clip_norm",
        },
        "logging": {
            "save_every": "save_every",
            "print_every": "print_every",
        },
    }

    defaults = {}
    for section_name, key_map in section_key_map.items():
        section = data.get(section_name, {})
        if section is None:
            continue
        if not isinstance(section, dict):
            raise ValueError(f"YAML section '{section_name}' must be a dict, got {type(section)}")
        for yaml_key, arg_key in key_map.items():
            if yaml_key in section:
                defaults[arg_key] = section[yaml_key]

    return defaults


def _default(defaults, key, fallback=None):
    return defaults[key] if key in defaults else fallback


def parse_args():
    # First pass: only read --train-yaml so YAML can provide defaults for all
    # other arguments. Command-line values in the second pass still override YAML.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--train-yaml", default=None)
    pre_args, _ = pre_parser.parse_known_args()

    yaml_data = _load_train_yaml(pre_args.train_yaml)
    yaml_defaults = _yaml_defaults_from_dict(yaml_data)

    parser = argparse.ArgumentParser("STARTrack Stage4 online joint training")
    parser.add_argument("--train-yaml", default=pre_args.train_yaml,
                        help="Optional YAML file that provides training defaults. CLI args override YAML values.")

    # These three used to be required=True. They are now validated after parsing
    # so they can be supplied by --train-yaml.
    parser.add_argument("--sutrack-ckpt", default=_default(yaml_defaults, "sutrack_ckpt"))
    parser.add_argument("--reranker-ckpt", default=_default(yaml_defaults, "reranker_ckpt"))
    parser.add_argument("--output-dir", default=_default(yaml_defaults, "output_dir"))
    parser.add_argument("--dataset", default=_default(yaml_defaults, "dataset", "lasot"))
    parser.add_argument("--config", default=_default(yaml_defaults, "config", "experiments/sutrack/sutrack_b224.yaml"))
    parser.add_argument("--use-lmdb", action=argparse.BooleanOptionalAction,
                        default=bool(_default(yaml_defaults, "use_lmdb", False)))

    parser.add_argument("--epochs", type=int, default=int(_default(yaml_defaults, "epochs", 1)))
    parser.add_argument("--samples-per-epoch", type=int, default=int(_default(yaml_defaults, "samples_per_epoch", 1000)))
    parser.add_argument("--rollout-len", type=int, default=int(_default(yaml_defaults, "rollout_len", 8)))
    parser.add_argument("--topk", type=int, default=int(_default(yaml_defaults, "topk", 8)))
    parser.add_argument("--batch-size", type=int, default=int(_default(yaml_defaults, "batch_size", 1)))
    parser.add_argument("--num-workers", type=int, default=int(_default(yaml_defaults, "num_workers", 0)))

    parser.add_argument("--hard-frame-list", default=_default(yaml_defaults, "hard_frame_list"))
    parser.add_argument("--hard-frame-ratio", type=float, default=float(_default(yaml_defaults, "hard_frame_ratio", 0.6)))
    parser.add_argument("--hard-seq-list", default=_default(yaml_defaults, "hard_seq_list"))
    parser.add_argument("--hard-seq-ratio", type=float, default=float(_default(yaml_defaults, "hard_seq_ratio", 0.2)))
    parser.add_argument("--full-ratio", type=float, default=float(_default(yaml_defaults, "full_ratio", 0.2)))
    parser.add_argument("--anchor-jitter", type=int, default=int(_default(yaml_defaults, "anchor_jitter", -1)))
    parser.add_argument("--min-anchor-valid-frames", type=int, default=int(_default(yaml_defaults, "min_anchor_valid_frames", 1)))
    parser.add_argument("--max-anchor-retry", type=int, default=int(_default(yaml_defaults, "max_anchor_retry", 20)))
    parser.add_argument(
        "--stage4-sampling-mode",
        choices=["random", "hard_frame_mixed"],
        default=_default(yaml_defaults, "stage4_sampling_mode", "random"),
    )

    parser.add_argument("--iou-thresh", type=float, default=float(_default(yaml_defaults, "iou_thresh", 0.3)))
    parser.add_argument("--apply-gain-thr", type=float, default=float(_default(yaml_defaults, "apply_gain_thr", 0.10)))
    parser.add_argument("--best-iou-thr", type=float, default=float(_default(yaml_defaults, "best_iou_thr", 0.50)))
    parser.add_argument("--apply-threshold", type=float, default=float(_default(yaml_defaults, "apply_threshold", 0.50)))

    parser.add_argument("--lr-reranker", type=float, default=float(_default(yaml_defaults, "lr_reranker", 1e-5)))
    parser.add_argument("--lr-decoder", type=float, default=float(_default(yaml_defaults, "lr_decoder", 1e-6)))
    parser.add_argument("--weight-decay", type=float, default=float(_default(yaml_defaults, "weight_decay", 1e-4)))
    parser.add_argument("--rank-lambda", type=float, default=float(_default(yaml_defaults, "rank_lambda", 0.05)))
    parser.add_argument("--margin", type=float, default=float(_default(yaml_defaults, "margin", 0.2)))
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training (e.g. last.pth)")
    parser.add_argument("--grad-clip-norm", type=float, default=float(_default(yaml_defaults, "grad_clip_norm", 1.0)))

    parser.add_argument("--freeze-encoder", action=argparse.BooleanOptionalAction,
                        default=bool(_default(yaml_defaults, "freeze_encoder", True)))
    parser.add_argument("--unfreeze-decoder", action=argparse.BooleanOptionalAction,
                        default=bool(_default(yaml_defaults, "unfreeze_decoder", False)))
    parser.add_argument("--history-len", type=int, default=int(_default(yaml_defaults, "history_len", 32)))
    parser.add_argument(
        "--update-mode",
        choices=["auto", "coach", "scheduled_takeover", "safe_predicted"],
        default=_default(yaml_defaults, "update_mode", "auto"),
        help=(
            "Stage4 rollout curriculum. 'auto' switches by epoch: "
            "coach -> scheduled_takeover -> safe_predicted. "
            "The old teacher/pred/gated/shadow modes are intentionally removed."
        ),
    )
    parser.add_argument(
        "--coach-epochs",
        type=int,
        default=int(_default(yaml_defaults, "coach_epochs", -1)),
        help="Number of warmup epochs using coach mode when --update-mode=auto. -1 uses 30% of total epochs.",
    )
    parser.add_argument(
        "--scheduled-takeover-epochs",
        type=int,
        default=int(_default(yaml_defaults, "scheduled_takeover_epochs", -1)),
        help="Number of middle epochs using scheduled_takeover when --update-mode=auto. -1 uses 30% of total epochs.",
    )
    parser.add_argument(
        "--takeover-prob-thresh",
        type=float,
        default=float(_default(yaml_defaults, "takeover_prob_thresh", 0.35)),
        help="Minimum reranker probability for predicted-state takeover in scheduled_takeover.",
    )
    parser.add_argument(
        "--safe-pred-prob-thresh",
        type=float,
        default=float(_default(yaml_defaults, "safe_pred_prob_thresh", 0.45)),
        help="Minimum reranker probability for predicted-state takeover in safe_predicted.",
    )
    parser.add_argument(
        "--max-motion-norm",
        type=float,
        default=float(_default(yaml_defaults, "max_motion_norm", 4.0)),
        help="Reject predicted-state takeover when center jump exceeds this value normalized by previous target size.",
    )
    parser.add_argument(
        "--max-scale-ratio",
        type=float,
        default=float(_default(yaml_defaults, "max_scale_ratio", 4.0)),
        help="Reject predicted-state takeover when area ratio between candidate and previous state is too large.",
    )
    parser.add_argument(
        "--use-heuristic-reliability-gate",
        action=argparse.BooleanOptionalAction,
        default=bool(_default(yaml_defaults, "use_heuristic_reliability_gate", True)),
        help="Gate predicted state/history updates with an inference-only heuristic reliability score.",
    )
    parser.add_argument(
        "--reliability-update-thresh",
        type=float,
        default=float(_default(yaml_defaults, "reliability_update_thresh", 0.50)),
    )
    parser.add_argument("--rel-target-weight", type=float, default=float(_default(yaml_defaults, "rel_target_weight", 0.25)))
    parser.add_argument("--rel-score-weight", type=float, default=float(_default(yaml_defaults, "rel_score_weight", 0.20)))
    parser.add_argument("--rel-margin-weight", type=float, default=float(_default(yaml_defaults, "rel_margin_weight", 0.15)))
    parser.add_argument("--rel-ambiguity-weight", type=float, default=float(_default(yaml_defaults, "rel_ambiguity_weight", 0.15)))
    parser.add_argument("--rel-motion-weight", type=float, default=float(_default(yaml_defaults, "rel_motion_weight", 0.10)))
    parser.add_argument("--rel-history-weight", type=float, default=float(_default(yaml_defaults, "rel_history_weight", 0.15)))
    parser.add_argument("--teacher-prob-start", type=float, default=float(_default(yaml_defaults, "teacher_prob_start", 1.0)))
    parser.add_argument("--teacher-prob-end", type=float, default=float(_default(yaml_defaults, "teacher_prob_end", 0.2)))
    parser.add_argument("--teacher-decay-epochs", type=int, default=int(_default(yaml_defaults, "teacher_decay_epochs", 10)))
    parser.add_argument("--detach-history", action=argparse.BooleanOptionalAction,
                        default=bool(_default(yaml_defaults, "detach_history", True)))

    parser.add_argument("--save-every", type=int, default=int(_default(yaml_defaults, "save_every", 1)))
    parser.add_argument("--print-every", type=int, default=int(_default(yaml_defaults, "print_every", 20)))
    parser.add_argument("--device", default=_default(yaml_defaults, "device", "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=int(_default(yaml_defaults, "seed", 42)))

    args = parser.parse_args()

    missing = []
    for key in ["sutrack_ckpt", "reranker_ckpt", "output_dir"]:
        if getattr(args, key) in (None, ""):
            missing.append("--" + key.replace("_", "-"))
    if missing:
        raise ValueError(
            "Missing required arguments: " + ", ".join(missing) +
            ". Provide them on CLI or in --train-yaml under paths."
        )

    args.train_yaml_data = yaml_data
    return args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dataset_name_from_arg(name):
    mapping = {
        "lasot": "LASOT",
        "lasot_hard": "LASOT_HARD",
        "got10k": "GOT10K_vottrain",
        "got10k_vottrain": "GOT10K_vottrain",
        "got10k_hard": "GOT10K_vottrain_HARD",
        "tnl2k": "TNL2K_train",
        "tnl2k_train": "TNL2K_train",
        "tnl2k_hard": "TNL2K_train_HARD",
    }
    key = str(name).lower()
    return mapping.get(key, name)


def make_stage4_settings(args, cfg_obj):
    settings = Settings()
    settings.use_lmdb = bool(args.use_lmdb)
    settings.multi_modal_vision = bool(getattr(cfg_obj.DATA, "MULTI_MODAL_VISION", False))
    settings.multi_modal_language = False
    settings.use_nlp = getattr(cfg_obj.DATA, "USE_NLP", {})
    return settings


def normalize_dataset_tag(name):
    value = str(name).lower()
    if "got" in value:
        return "got10k"
    if "lasot" in value:
        return "lasot"
    return value


def parse_seq_key(seq_key):
    seq_key = str(seq_key).strip()
    if len(seq_key) == 6 and seq_key.isdigit():
        return "got10k", f"GOT-10k_Train_{int(seq_key) + 1:06d}"
    return "lasot", seq_key


def read_hard_frame_list(path):
    if path is None:
        return []
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"hard-frame list not found: {path}")
    anchors = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if len(line) == 0 or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Invalid hard-frame line {path}:{line_no}: {line}")
            tag, seq_name = parse_seq_key(parts[0])
            anchors.append({
                "seq_key": parts[0],
                "dataset_tag": tag,
                "seq_name": seq_name,
                "frame_idx": int(parts[1]),
            })
    return anchors


def read_hard_seq_list(path):
    if path is None:
        return []
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"hard-seq list not found: {path}")
    seqs = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if len(line) == 0 or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 1:
                raise ValueError(f"Invalid hard-seq line {path}:{line_no}: {line}")
            tag, seq_name = parse_seq_key(parts[0])
            seqs.append({"seq_key": parts[0], "dataset_tag": tag, "seq_name": seq_name})
    return seqs


def needed_dataset_names(full_dataset_name, anchors, hard_seqs):
    names = [full_dataset_name]
    required_tags = {item["dataset_tag"] for item in anchors}
    required_tags.update(item["dataset_tag"] for item in hard_seqs)
    if "lasot" in required_tags and "LASOT" not in names:
        names.append("LASOT")
    if "got10k" in required_tags and "GOT10K_vottrain" not in names:
        names.append("GOT10K_vottrain")
    return names


class Stage4HardFrameMixedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        full_datasets,
        lookup_datasets,
        hard_frame_anchors,
        hard_seqs,
        args,
        cfg_obj,
    ):
        self.full_datasets = list(full_datasets)
        self.lookup_datasets = list(lookup_datasets)
        self.samples_per_epoch = int(args.samples_per_epoch)
        self.rollout_len = int(args.rollout_len)
        self.num_template_frames = int(cfg_obj.DATA.TEMPLATE.NUMBER)
        self.max_gap = self._get_max_gap_value(cfg_obj.DATA.MAX_SAMPLE_INTERVAL)
        self.anchor_jitter = self.rollout_len - 1 if int(args.anchor_jitter) < 0 else int(args.anchor_jitter)
        self.min_anchor_valid_frames = int(args.min_anchor_valid_frames)
        self.max_anchor_retry = int(args.max_anchor_retry)
        self.hard_frame_ratio = float(args.hard_frame_ratio)
        self.hard_seq_ratio = float(args.hard_seq_ratio)
        self.full_ratio = float(args.full_ratio)
        self.stage4_sampling_mode = "hard_frame_mixed"
        self._seq_info_cache = {}

        self.sequence_index = self._build_sequence_index(self.lookup_datasets)
        self.full_sequence_refs = self._build_full_sequence_refs(self.full_datasets)
        self.hard_frames = self._resolve_hard_frames(hard_frame_anchors)
        self.hard_frame_by_sequence = self._build_hard_frame_index(self.hard_frames)
        if len(hard_seqs) == 0:
            hard_seqs = self._derive_hard_seqs_from_frames(self.hard_frames)
        self.hard_seqs = self._resolve_hard_seqs(hard_seqs)

        self.hard_frame_count = len(self.hard_frames)
        self.hard_seq_count = len(self.hard_seqs)

    def __len__(self):
        return self.samples_per_epoch

    @staticmethod
    def _get_max_gap_value(value):
        if isinstance(value, (list, tuple)):
            return max(int(v) for v in value)
        return int(value)

    def _build_sequence_index(self, datasets):
        index = {}
        for dataset in datasets:
            tag = normalize_dataset_tag(dataset.get_name())
            if not hasattr(dataset, "sequence_list"):
                continue
            for seq_id, seq_name in enumerate(dataset.sequence_list):
                index.setdefault((tag, str(seq_name)), (dataset, seq_id))
        return index

    def _build_full_sequence_refs(self, datasets):
        refs = []
        for dataset in datasets:
            for seq_id in range(dataset.get_num_sequences()):
                refs.append((dataset, seq_id))
        if len(refs) == 0:
            raise RuntimeError("No full dataset sequences available for Stage4 hard-frame mixed sampling.")
        return refs

    def _resolve_hard_frames(self, anchors):
        resolved = []
        missing = []
        for item in anchors:
            ref = self.sequence_index.get((item["dataset_tag"], item["seq_name"]))
            if ref is None:
                missing.append(f"{item['seq_key']}->{item['seq_name']}")
                continue
            dataset, seq_id = ref
            resolved.append({**item, "dataset": dataset, "seq_id": seq_id})
        if len(missing) > 0:
            preview = ", ".join(missing[:10])
            print(f"[WARN] {len(missing)} hard-frame anchors were not found in loaded datasets: {preview}")
        return resolved

    def _build_hard_frame_index(self, hard_frames):
        frame_index = {}
        for item in hard_frames:
            key = (id(item["dataset"]), int(item["seq_id"]))
            frame_index.setdefault(key, set()).add(int(item["frame_idx"]))
        return frame_index

    def _derive_hard_seqs_from_frames(self, hard_frames):
        seqs = {}
        for item in hard_frames:
            key = (item["dataset_tag"], item["seq_name"])
            seqs.setdefault(key, {
                "seq_key": item["seq_key"],
                "dataset_tag": item["dataset_tag"],
                "seq_name": item["seq_name"],
            })
        return list(seqs.values())

    def _resolve_hard_seqs(self, seqs):
        resolved = []
        missing = []
        seen = set()
        for item in seqs:
            key = (item["dataset_tag"], item["seq_name"])
            if key in seen:
                continue
            seen.add(key)
            ref = self.sequence_index.get(key)
            if ref is None:
                missing.append(f"{item['seq_key']}->{item['seq_name']}")
                continue
            dataset, seq_id = ref
            resolved.append({**item, "dataset": dataset, "seq_id": seq_id})
        if len(missing) > 0:
            preview = ", ".join(missing[:10])
            print(f"[WARN] {len(missing)} hard sequences were not found in loaded datasets: {preview}")
        return resolved

    def _effective_ratios(self):
        hard_frame_ratio = self.hard_frame_ratio if len(self.hard_frames) > 0 else 0.0
        hard_seq_ratio = self.hard_seq_ratio if len(self.hard_seqs) > 0 else 0.0
        full_ratio = max(self.full_ratio, 0.0)
        if len(self.hard_frames) == 0:
            full_ratio += max(self.hard_frame_ratio, 0.0)
        if len(self.hard_seqs) == 0:
            full_ratio += max(self.hard_seq_ratio, 0.0)
        total = hard_frame_ratio + hard_seq_ratio + full_ratio
        if total <= 0:
            return 0.0, 0.0, 1.0
        return hard_frame_ratio / total, hard_seq_ratio / total, full_ratio / total

    def _choose_source(self):
        hard_frame_ratio, hard_seq_ratio, _ = self._effective_ratios()
        r = random.random()
        if r < hard_frame_ratio:
            return "hard_frame"
        if r < hard_frame_ratio + hard_seq_ratio:
            return "hard_seq"
        return "full"

    def __getitem__(self, index):
        last_error = None
        for _ in range(max(self.max_anchor_retry * 10, 50)):
            source = self._choose_source()
            try:
                if source == "hard_frame":
                    return self._sample_hard_frame_clip()
                if source == "hard_seq":
                    return self._sample_sequence_clip(self.hard_seqs, sample_source="hard_seq")
                return self._sample_sequence_clip(self.full_sequence_refs, sample_source="full")
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Stage4 hard-frame mixed sampler failed after retries: {last_error}")

    def _get_sequence_info(self, dataset, seq_id):
        key = (id(dataset), int(seq_id))
        if key not in self._seq_info_cache:
            self._seq_info_cache[key] = dataset.get_sequence_info(seq_id)
        return self._seq_info_cache[key]

    def _visible_tensor(self, seq_info):
        if "visible" in seq_info:
            return seq_info["visible"]
        return seq_info["valid"]

    def _is_visible(self, visible, frame_id):
        return bool(visible[int(frame_id)].item() if torch.is_tensor(visible) else visible[int(frame_id)])

    def _select_template_ids(self, seq_info, search_start, anchor_frame=None):
        visible = self._visible_tensor(seq_info)
        seq_len = len(visible)
        history_end = max(int(search_start), 0)
        history_ids = [idx for idx in range(history_end) if self._is_visible(visible, idx)]
        if len(history_ids) >= self.num_template_frames:
            if self.num_template_frames == 1:
                return [history_ids[0]]
            select_indices = np.linspace(0, len(history_ids) - 1, num=self.num_template_frames, dtype=int).tolist()
            return [history_ids[idx] for idx in select_indices]

        limit = seq_len if anchor_frame is None else min(int(anchor_frame) + 1, seq_len)
        fallback_ids = [idx for idx in range(limit) if self._is_visible(visible, idx)]
        template_id = fallback_ids[0] if len(fallback_ids) > 0 else 0
        return [template_id for _ in range(self.num_template_frames)]

    def _sample_hard_frame_clip(self):
        if len(self.hard_frames) == 0:
            raise RuntimeError("No resolved hard-frame anchors available.")
        jitter = max(int(self.anchor_jitter), 0)
        for _ in range(max(self.max_anchor_retry, 1)):
            item = random.choice(self.hard_frames)
            dataset = item["dataset"]
            seq_id = int(item["seq_id"])
            anchor = int(item["frame_idx"])
            seq_info = self._get_sequence_info(dataset, seq_id)
            visible = self._visible_tensor(seq_info)
            seq_len = len(visible)
            if seq_len < self.rollout_len or anchor < 0 or anchor >= seq_len:
                continue
            max_start = max(seq_len - self.rollout_len, 0)
            start = anchor - random.randint(0, jitter)
            start = min(max(start, 0), max_start)
            search_frame_ids = list(range(start, start + self.rollout_len))
            hard_frames = self.hard_frame_by_sequence.get((id(dataset), seq_id), set())
            anchor_valid_count = sum(
                1
                for frame_id in search_frame_ids
                if frame_id in hard_frames and self._is_visible(visible, frame_id)
            )
            if anchor_valid_count < self.min_anchor_valid_frames:
                continue
            template_frame_ids = self._select_template_ids(seq_info, start, anchor_frame=anchor)
            return self._make_data(
                dataset,
                seq_id,
                seq_info,
                template_frame_ids,
                search_frame_ids,
                sample_source="hard_frame",
                anchor_frame=anchor,
                anchor_valid_frames=anchor_valid_count,
            )
        raise RuntimeError("Failed to sample a valid hard-frame anchored clip.")

    def _sample_sequence_clip(self, refs, sample_source):
        if len(refs) == 0:
            raise RuntimeError(f"No sequences available for source={sample_source}.")
        for _ in range(max(self.max_anchor_retry, 1)):
            ref = random.choice(refs)
            if isinstance(ref, dict):
                dataset = ref["dataset"]
                seq_id = int(ref["seq_id"])
            else:
                dataset, seq_id = ref
                seq_id = int(seq_id)
            seq_info = self._get_sequence_info(dataset, seq_id)
            visible = self._visible_tensor(seq_info)
            seq_len = len(visible)
            if seq_len < self.rollout_len:
                continue

            candidates = []
            for end_id in range(self.rollout_len - 1, seq_len):
                start_id = end_id - self.rollout_len + 1
                if not all(self._is_visible(visible, idx) for idx in range(start_id, end_id + 1)):
                    continue
                history_min_id = max(0, start_id - self.max_gap)
                history_ids = [idx for idx in range(history_min_id, start_id) if self._is_visible(visible, idx)]
                if len(history_ids) >= self.num_template_frames:
                    candidates.append(start_id)
            if len(candidates) == 0:
                continue

            start = random.choice(candidates)
            search_frame_ids = list(range(start, start + self.rollout_len))
            template_frame_ids = self._select_template_ids(seq_info, start)
            return self._make_data(
                dataset,
                seq_id,
                seq_info,
                template_frame_ids,
                search_frame_ids,
                sample_source=sample_source,
                anchor_frame=-1,
                anchor_valid_frames=-1,
            )
        raise RuntimeError(f"Failed to sample a valid {sample_source} clip.")

    def _make_data(
        self,
        dataset,
        seq_id,
        seq_info,
        template_frame_ids,
        search_frame_ids,
        sample_source,
        anchor_frame,
        anchor_valid_frames,
    ):
        template_frames, template_anno, _ = dataset.get_frames(seq_id, template_frame_ids, seq_info)
        search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids, seq_info)
        height, width, _ = template_frames[0].shape
        template_masks = template_anno["mask"] if "mask" in template_anno else [torch.zeros((height, width))] * len(template_frame_ids)
        search_masks = search_anno["mask"] if "mask" in search_anno else [torch.zeros((height, width))] * len(search_frame_ids)
        seq_name = dataset.sequence_list[seq_id] if hasattr(dataset, "sequence_list") else str(seq_id)
        return TensorDict({
            "template_images": template_frames,
            "template_anno": template_anno["bbox"],
            "template_masks": template_masks,
            "search_images": search_frames,
            "search_anno": search_anno["bbox"],
            "search_masks": search_masks,
            "dataset": dataset.get_name(),
            "test_class": meta_obj_test.get("object_class_name") if isinstance(meta_obj_test, dict) else None,
            "seq_name": seq_name,
            "seq_id": int(seq_id),
            "template_frame_ids": template_frame_ids,
            "search_frame_ids": search_frame_ids,
            "sample_source": sample_source,
            "anchor_frame": int(anchor_frame),
            "anchor_valid_frames": int(anchor_valid_frames),
            "valid": True,
        })


def load_config(config_path):
    config_path = str(config_path)
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    update_config_from_file(config_path)
    return cfg


def build_online_loader(args, cfg_obj):
    settings = make_stage4_settings(args, cfg_obj)
    dataset_name = dataset_name_from_arg(args.dataset)

    if args.stage4_sampling_mode == "hard_frame_mixed" and args.hard_frame_list is not None:
        hard_frame_anchors = read_hard_frame_list(args.hard_frame_list)
        hard_seqs = read_hard_seq_list(args.hard_seq_list)
        full_datasets = names2datasets([dataset_name], settings, opencv_loader)
        lookup_names = needed_dataset_names(dataset_name, hard_frame_anchors, hard_seqs)
        lookup_datasets = names2datasets(lookup_names, settings, opencv_loader)
        dataset_train = Stage4HardFrameMixedDataset(
            full_datasets=full_datasets,
            lookup_datasets=lookup_datasets,
            hard_frame_anchors=hard_frame_anchors,
            hard_seqs=hard_seqs,
            args=args,
            cfg_obj=cfg_obj,
        )
        return LTRLoader(
            "stage4_hard_frame_mixed",
            dataset_train,
            training=True,
            batch_size=int(args.batch_size),
            shuffle=True,
            num_workers=int(args.num_workers),
            drop_last=True,
            stack_dim=0,
            pin_memory=torch.cuda.is_available(),
        )

    datasets = names2datasets([dataset_name], settings, opencv_loader)
    dataset_train = sampler.TrackingSampler(
        datasets=datasets,
        p_datasets=[1],
        samples_per_epoch=int(args.samples_per_epoch),
        max_gap=cfg_obj.DATA.MAX_SAMPLE_INTERVAL,
        num_search_frames=int(args.rollout_len),
        num_template_frames=int(cfg_obj.DATA.TEMPLATE.NUMBER),
        processing=RawSequenceProcessing(),
        frame_sample_mode="online_temporal",
        multi_modal_language=False,
    )
    return LTRLoader(
        "stage4_online",
        dataset_train,
        training=True,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        drop_last=True,
        stack_dim=0,
        pin_memory=torch.cuda.is_available(),
    )


def unwrap_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("net", "model", "sutrack", "state_dict"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]
    return checkpoint


def strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items() if k.startswith("module.")}
    return state_dict


def load_sutrack_checkpoint(model, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = strip_module_prefix(unwrap_state_dict(ckpt))
    try:
        model.load_state_dict(state, strict=True)
        print(f"[INFO] SUTrack checkpoint loaded strict=True: {path}")
        return
    except RuntimeError as exc:
        print(f"[WARN] Strict SUTrack load failed, retry strict=False: {exc}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[INFO] SUTrack checkpoint loaded strict=False: {path}")
    print(f"[INFO] SUTrack missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")


def build_disambiguator(model, args):
    feat_dim = int(getattr(model.encoder, "num_channels", 512))
    disambiguator = PostDecoderDisambiguator(
        feat_dim=feat_dim,
        template_feat_dim=feat_dim,
        topk_peaks=int(args.topk),
        history_len=int(args.history_len),
        use_mamba_history=True,
        use_mamba_history_bank=True,
        use_template_anchor=False,
        use_first_frame_anchor=False,
        use_history_aware_rerank_score=False,
    )
    ckpt = torch.load(args.reranker_ckpt, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "disambiguator" in ckpt:
        state = ckpt["disambiguator"]
    else:
        state = unwrap_state_dict(ckpt)
    state = strip_module_prefix(state)
    missing, unexpected = disambiguator.load_state_dict(state, strict=False)
    print(f"[INFO] Reranker checkpoint loaded: {args.reranker_ckpt}")
    print(f"[INFO] Reranker missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")
    return disambiguator


def set_trainable(model, disambiguator, args):
    for param in model.parameters():
        param.requires_grad_(False)
    model.encoder.eval()
    if getattr(model, "text_encoder", None) is not None:
        model.text_encoder.eval()
    if getattr(model, "encoder_postprocess", None) is not None:
        model.encoder_postprocess.eval()

    if args.unfreeze_decoder:
        model.decoder.train()
        for param in model.decoder.parameters():
            param.requires_grad_(True)
    else:
        model.decoder.eval()
        for param in model.decoder.parameters():
            param.requires_grad_(False)

    if getattr(model, "task_decoder", None) is not None:
        model.task_decoder.eval()
        for param in model.task_decoder.parameters():
            param.requires_grad_(False)

    disambiguator.train()
    for param in disambiguator.parameters():
        param.requires_grad_(True)


def count_params(module):
    return sum(int(p.numel()) for p in module.parameters() if p.requires_grad)


def build_optimizer(model, disambiguator, args):
    param_groups = [{"params": [p for p in disambiguator.parameters() if p.requires_grad], "lr": args.lr_reranker}]
    if args.unfreeze_decoder:
        decoder_params = [p for p in model.decoder.parameters() if p.requires_grad]
        if len(decoder_params) > 0:
            param_groups.append({"params": decoder_params, "lr": args.lr_decoder})
    return torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)


def make_history_tensor(history_tokens, history_len):
    if len(history_tokens) == 0:
        return None
    tokens = history_tokens[-int(history_len):]
    return torch.stack(tokens, dim=0).unsqueeze(0)


def teacher_prob(epoch, args):
    total = max(int(args.teacher_decay_epochs), 1)
    alpha = min(max((int(epoch) - 1) / float(total), 0.0), 1.0)
    return float(args.teacher_prob_start + alpha * (args.teacher_prob_end - args.teacher_prob_start))


def resolve_curriculum_epochs(args):
    """Return (coach_epochs, scheduled_takeover_epochs) for the 3-stage Stage4 curriculum."""
    total_epochs = max(int(args.epochs), 1)
    if int(args.coach_epochs) >= 0:
        coach_epochs = int(args.coach_epochs)
    else:
        coach_epochs = max(1, int(math.ceil(0.30 * total_epochs)))

    if int(args.scheduled_takeover_epochs) >= 0:
        scheduled_epochs = int(args.scheduled_takeover_epochs)
    else:
        scheduled_epochs = max(1, int(math.ceil(0.30 * total_epochs)))

    # Keep at least the final epoch for safe_predicted when possible.
    if coach_epochs + scheduled_epochs >= total_epochs and total_epochs >= 3:
        scheduled_epochs = max(1, total_epochs - coach_epochs - 1)
    if coach_epochs >= total_epochs and total_epochs >= 2:
        coach_epochs = total_epochs - 1
    return max(coach_epochs, 0), max(scheduled_epochs, 0)


def effective_update_mode(epoch, args):
    """Map an epoch to the actual rollout stage.

    auto:
      Stage4-A: coach
      Stage4-B: scheduled_takeover
      Stage4-C: safe_predicted
    """
    mode = str(args.update_mode)
    if mode != "auto":
        return mode

    coach_epochs, scheduled_epochs = resolve_curriculum_epochs(args)
    if int(epoch) <= coach_epochs:
        return "coach"
    if int(epoch) <= coach_epochs + scheduled_epochs:
        return "scheduled_takeover"
    return "safe_predicted"


def candidate_motion_ok(candidate_bbox, prev_state, args):
    """A light inference-style safety gate for predicted-state takeover.

    This avoids letting a high-probability but spatially implausible candidate
    immediately move the online crop center and poison the rollout.
    """
    cand = torch.as_tensor(candidate_bbox).detach().float()
    prev = torch.as_tensor(prev_state).detach().float().to(device=cand.device)
    cand_cx = cand[0] + 0.5 * cand[2]
    cand_cy = cand[1] + 0.5 * cand[3]
    prev_cx = prev[0] + 0.5 * prev[2]
    prev_cy = prev[1] + 0.5 * prev[3]

    prev_size = torch.sqrt((prev[2] * prev[3]).clamp_min(1.0))
    jump = torch.sqrt((cand_cx - prev_cx) ** 2 + (cand_cy - prev_cy) ** 2) / prev_size

    cand_area = (cand[2] * cand[3]).clamp_min(1.0)
    prev_area = (prev[2] * prev[3]).clamp_min(1.0)
    scale_ratio = torch.maximum(cand_area / prev_area, prev_area / cand_area)

    return bool(
        float(jump.item()) <= float(args.max_motion_norm)
        and float(scale_ratio.item()) <= float(args.max_scale_ratio)
    )


def candidate_motion_conf(candidate_bbox, prev_state, args):
    cand = torch.as_tensor(candidate_bbox).detach().float()
    prev = torch.as_tensor(prev_state).detach().float().to(device=cand.device)
    cand_cx = cand[0] + 0.5 * cand[2]
    cand_cy = cand[1] + 0.5 * cand[3]
    prev_cx = prev[0] + 0.5 * prev[2]
    prev_cy = prev[1] + 0.5 * prev[3]

    prev_size = torch.sqrt((prev[2] * prev[3]).clamp_min(1.0))
    jump = torch.sqrt((cand_cx - prev_cx) ** 2 + (cand_cy - prev_cy) ** 2) / prev_size

    cand_area = (cand[2] * cand[3]).clamp_min(1.0)
    prev_area = (prev[2] * prev[3]).clamp_min(1.0)
    scale_ratio = torch.maximum(cand_area / prev_area, prev_area / cand_area)

    max_motion = max(float(args.max_motion_norm), 1e-6)
    jump_conf = 1.0 - min(float(jump.item()) / max_motion, 1.0)
    max_scale = max(float(args.max_scale_ratio), 1.0 + 1e-6)
    scale_conf = 1.0 - min(
        math.log(max(float(scale_ratio.item()), 1.0)) / math.log(max_scale),
        1.0,
    )
    motion_ok = float(jump.item()) <= max_motion and float(scale_ratio.item()) <= max_scale
    motion_conf = 0.5 * (jump_conf + scale_conf)
    if not motion_ok:
        motion_conf *= 0.5
    return float(max(0.0, min(1.0, motion_conf))), {
        "motion_ok": bool(motion_ok),
        "motion_jump": float(jump.item()),
        "motion_scale_ratio": float(scale_ratio.item()),
        "motion_conf": float(max(0.0, min(1.0, motion_conf))),
    }


def _history_similarity_conf(topk_feats, pred_idx, history_tokens):
    if history_tokens is None or len(history_tokens) == 0 or topk_feats is None:
        return 0.5, 0.0

    feats = topk_feats.detach()
    if feats.dim() == 3:
        feats = feats[0]
    selected_feat = feats[int(pred_idx)].float().view(1, -1)

    if torch.is_tensor(history_tokens):
        hist = history_tokens.detach()
        if hist.dim() == 3:
            hist = hist[0]
    else:
        hist = torch.stack([token.detach() for token in history_tokens], dim=0)
    if hist.numel() == 0:
        return 0.5, 0.0

    hist = hist.to(device=selected_feat.device, dtype=selected_feat.dtype).view(-1, selected_feat.size(-1))
    hist_mean = hist.mean(dim=0, keepdim=True)
    sim = F.cosine_similarity(selected_feat, hist_mean, dim=-1, eps=1e-6)
    sim_value = float(torch.nan_to_num(sim, nan=0.0).clamp(-1.0, 1.0).item())
    return float(max(0.0, min(1.0, 0.5 * (sim_value + 1.0)))), sim_value


def compute_heuristic_reliability_for_update(
    logits_t,
    topk_scores,
    pred_idx,
    topk_bboxes_xywh,
    prev_state,
    args,
    history_tokens=None,
    topk_feats=None,
):
    """Inference-only reliability score for predicted Stage4 updates."""
    with torch.no_grad():
        probs = torch.softmax(logits_t.detach(), dim=-1)
        pred_prob = float(probs[0, int(pred_idx)].clamp(0.0, 1.0).item())
        sorted_probs, _ = torch.sort(probs[0], descending=True)
        top1_prob = float(sorted_probs[0].item()) if sorted_probs.numel() > 0 else pred_prob
        top2_prob = float(sorted_probs[1].item()) if sorted_probs.numel() > 1 else 0.0
        margin_conf = max(0.0, min(1.0, top1_prob - top2_prob))

        raw_scores = torch.nan_to_num(
            topk_scores.detach().float(),
            nan=0.0,
            posinf=20.0,
            neginf=-20.0,
        )
        if raw_scores.dim() == 2:
            raw_scores = raw_scores[0]
        if raw_scores.numel() == 0:
            score_conf_values = torch.zeros_like(probs[0])
        elif float(raw_scores.min().item()) >= 0.0 and float(raw_scores.max().item()) <= 1.0:
            score_conf_values = raw_scores.clamp(0.0, 1.0)
        else:
            score_conf_values = torch.softmax(raw_scores.clamp(-20.0, 20.0), dim=0)

        selected_score_conf = float(score_conf_values[int(pred_idx)].clamp(0.0, 1.0).item())
        sorted_scores, _ = torch.sort(score_conf_values, descending=True)
        top1_score = float(sorted_scores[0].clamp_min(1e-6).item()) if sorted_scores.numel() > 0 else 1e-6
        top2_score = float(sorted_scores[1].item()) if sorted_scores.numel() > 1 else 0.0
        ambiguity_conf = max(0.0, min(1.0, 1.0 - top2_score / max(top1_score, 1e-6)))

        motion_conf, motion_debug = candidate_motion_conf(topk_bboxes_xywh[int(pred_idx)], prev_state, args)
        history_conf, history_cosine = _history_similarity_conf(topk_feats, pred_idx, history_tokens)

        weights = {
            "target": float(args.rel_target_weight),
            "score": float(args.rel_score_weight),
            "margin": float(args.rel_margin_weight),
            "ambiguity": float(args.rel_ambiguity_weight),
            "motion": float(args.rel_motion_weight),
            "history": float(args.rel_history_weight),
        }
        weight_sum = max(sum(weights.values()), 1e-6)
        reliability_score = (
            weights["target"] * pred_prob
            + weights["score"] * selected_score_conf
            + weights["margin"] * margin_conf
            + weights["ambiguity"] * ambiguity_conf
            + weights["motion"] * motion_conf
            + weights["history"] * history_conf
        ) / weight_sum
        reliability_score = float(max(0.0, min(1.0, reliability_score)))

    reliability_debug = {
        "reliability_score": reliability_score,
        "reliability_pass": bool(reliability_score >= float(args.reliability_update_thresh)),
        "pred_prob": pred_prob,
        "selected_score_conf": selected_score_conf,
        "margin_conf": margin_conf,
        "ambiguity_conf": ambiguity_conf,
        "motion_conf": motion_conf,
        "history_conf": history_conf,
        "history_cosine": history_cosine,
        "top1_prob": top1_prob,
        "top2_prob": top2_prob,
        "top1_score": top1_score,
        "top2_score": top2_score,
    }
    reliability_debug.update(motion_debug)
    return reliability_score, reliability_debug


def preprocess_image(img_arr, device, multi_modal_vision=False):
    tensor = torch.as_tensor(img_arr, device=device).float().permute(2, 0, 1).unsqueeze(0)
    tensor = tensor / 255.0
    if multi_modal_vision and tensor.size(1) == 3:
        tensor = torch.cat([tensor, tensor], dim=1)
    if tensor.size(1) == 6:
        mean = torch.tensor([0.485, 0.456, 0.406, 0.485, 0.456, 0.406], device=device).view(1, 6, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225, 0.229, 0.224, 0.225], device=device).view(1, 6, 1, 1)
    else:
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (tensor - mean) / std


def as_single_image_list(value):
    if isinstance(value, (list, tuple)):
        return [to_numpy_image(item) for item in value]
    if torch.is_tensor(value):
        if value.dim() == 5:
            if value.size(0) == 1:
                return [to_numpy_image(value[0, i]) for i in range(value.size(1))]
            return [to_numpy_image(value[i, 0]) for i in range(value.size(0))]
        if value.dim() == 4:
            return [to_numpy_image(value)]
    raise ValueError(f"Unsupported image sequence container: {type(value)}")


def to_numpy_image(value):
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        if tensor.dim() == 4:
            tensor = tensor[0]
        arr = tensor.numpy()
    else:
        arr = np.asarray(value)
    if arr.ndim != 3:
        raise ValueError(f"Expected HWC image, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def as_single_bbox_list(value):
    if isinstance(value, (list, tuple)):
        return [to_bbox_tensor(item) for item in value]
    if torch.is_tensor(value):
        tensor = value.detach().cpu().float()
        if tensor.dim() == 3:
            if tensor.size(0) == 1:
                return [tensor[0, i] for i in range(tensor.size(1))]
            return [tensor[i, 0] for i in range(tensor.size(0))]
        if tensor.dim() == 2:
            return [tensor[i] for i in range(tensor.size(0))]
    raise ValueError(f"Unsupported bbox sequence container: {type(value)}")


def to_bbox_tensor(value):
    if torch.is_tensor(value):
        tensor = value.detach().cpu().float()
        if tensor.dim() == 2:
            tensor = tensor[0]
        return tensor
    return torch.as_tensor(value, dtype=torch.float32)


def get_batch_meta(data, key, default=""):
    value = data.get(key, default)
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return default
        first = value[0]
        if torch.is_tensor(first) and first.numel() == 1:
            return int(first.item())
        return first
    if torch.is_tensor(value) and value.numel() == 1:
        return int(value.item())
    return value


def build_template_inputs(template_images, template_bboxes, cfg_obj, args):
    template_list = []
    template_anno_list = []
    device = torch.device(args.device)
    multi_modal_vision = bool(getattr(cfg_obj.DATA, "MULTI_MODAL_VISION", False))
    for img, bbox in zip(template_images, template_bboxes):
        patch, resize_factor = sample_target(
            img,
            bbox.tolist(),
            cfg_obj.TEST.TEMPLATE_FACTOR,
            output_sz=cfg_obj.TEST.TEMPLATE_SIZE,
        )
        template = preprocess_image(patch, device=device, multi_modal_vision=multi_modal_vision)
        anno = transform_image_to_crop(
            bbox,
            bbox,
            resize_factor,
            torch.tensor([cfg_obj.TEST.TEMPLATE_SIZE, cfg_obj.TEST.TEMPLATE_SIZE]),
            normalize=True,
        ).to(device=device).unsqueeze(0)
        template_list.append(template)
        template_anno_list.append(anno)
    return template_list, template_anno_list


def extract_topk_candidates(out_dict, model, args):
    score_map = out_dict["score_map"]
    size_map = out_dict.get("size_map", None)
    offset_map = out_dict.get("offset_map", None)
    feat_map = out_dict["f_map"]
    if size_map is None or offset_map is None or not hasattr(model.decoder, "cal_bbox_at_peaks"):
        raise RuntimeError("Stage4 currently requires CENTER decoder outputs: score_map, size_map, offset_map.")

    peaks_xy, topk_scores = topk_peaks_nms(score_map, topk=args.topk, kernel_size=5)
    topk_bboxes_norm = model.decoder.cal_bbox_at_peaks(peaks_xy, size_map, offset_map)
    topk_feats = sample_feature_at_peaks(feat_map, peaks_xy=peaks_xy)
    return topk_feats, topk_scores, peaks_xy, topk_bboxes_norm


def maybe_apply_tracking_window(out_dict, cfg_obj):
    if not bool(getattr(cfg_obj.TEST, "WINDOW", False)):
        return out_dict
    score_map = out_dict["score_map"]
    height, width = score_map.shape[-2:]
    window = hann2d(torch.tensor([height, width], device=score_map.device).long(), centered=True)
    out = dict(out_dict)
    out["score_map"] = score_map * window.to(device=score_map.device, dtype=score_map.dtype)
    return out


def map_boxes_back_batch(pred_boxes_norm, state_xywh, image_hw, search_size, resize_factor):
    device = pred_boxes_norm.device
    dtype = pred_boxes_norm.dtype
    state = torch.as_tensor(state_xywh, device=device, dtype=dtype)
    cx_prev = state[0] + 0.5 * state[2]
    cy_prev = state[1] + 0.5 * state[3]
    half_side = 0.5 * float(search_size) / float(resize_factor)
    pred = pred_boxes_norm * float(search_size) / float(resize_factor)
    cx, cy, w, h = pred.unbind(-1)
    cx_real = cx + (cx_prev - half_side)
    cy_real = cy + (cy_prev - half_side)
    x1 = cx_real - 0.5 * w
    y1 = cy_real - 0.5 * h
    boxes = torch.stack([x1, y1, w, h], dim=-1)

    img_h, img_w = image_hw
    x2 = (boxes[..., 0] + boxes[..., 2]).clamp(0.0, float(img_w))
    y2 = (boxes[..., 1] + boxes[..., 3]).clamp(0.0, float(img_h))
    x1 = boxes[..., 0].clamp(0.0, float(img_w))
    y1 = boxes[..., 1].clamp(0.0, float(img_h))
    w = (x2 - x1).clamp_min(1.0)
    h = (y2 - y1).clamp_min(1.0)
    return torch.stack([x1, y1, w, h], dim=-1)


def gt_inside_search_crop(state_xywh, gt_xywh, search_size, resize_factor):
    state = torch.as_tensor(state_xywh, dtype=torch.float32)
    gt = torch.as_tensor(gt_xywh, dtype=torch.float32)
    cx_prev = state[0] + 0.5 * state[2]
    cy_prev = state[1] + 0.5 * state[3]
    half_side = 0.5 * float(search_size) / float(resize_factor)
    cx_gt = gt[0] + 0.5 * gt[2]
    cy_gt = gt[1] + 0.5 * gt[3]
    return bool(
        (cx_gt >= cx_prev - half_side)
        and (cx_gt <= cx_prev + half_side)
        and (cy_gt >= cy_prev - half_side)
        and (cy_gt <= cy_prev + half_side)
    )


def compute_topk_ious(topk_bboxes_xywh, gt_xywh):
    if topk_bboxes_xywh.dim() != 2 or topk_bboxes_xywh.size(-1) != 4:
        raise ValueError(f"topk_bboxes_xywh must be [K, 4], got {tuple(topk_bboxes_xywh.shape)}")

    gt = torch.as_tensor(
        gt_xywh,
        device=topk_bboxes_xywh.device,
        dtype=topk_bboxes_xywh.dtype,
    ).view(1, 4)

    pred_xyxy = box_xywh_to_xyxy(topk_bboxes_xywh)
    gt_xyxy = box_xywh_to_xyxy(gt)
    ious, _ = box_iou(pred_xyxy, gt_xyxy)
    if ious.dim() == 2:
        ious = ious[:, 0]
    out = ious.nan_to_num(0.0).clamp(0.0, 1.0)
    if out.dim() != 1:
        raise RuntimeError(f"topk IoU output must be [K], got {tuple(out.shape)}")
    return out


def assert_topk_shapes(topk_feats, topk_scores, topk_bboxes_xywh, topk_ious, args):
    if topk_feats.dim() != 3:
        raise ValueError(f"topk_feats must be [B, K, C], got {tuple(topk_feats.shape)}")
    if topk_scores.dim() != 2:
        raise ValueError(f"topk_scores must be [B, K], got {tuple(topk_scores.shape)}")
    if topk_bboxes_xywh.dim() != 2:
        raise ValueError(f"topk_bboxes_xywh must be [K, 4], got {tuple(topk_bboxes_xywh.shape)}")
    if topk_ious.dim() != 1:
        raise ValueError(f"topk_ious must be [K], got {tuple(topk_ious.shape)}")
    if topk_feats.size(1) != int(args.topk):
        raise ValueError(f"Expected topk_feats K={args.topk}, got {topk_feats.size(1)}")
    if topk_scores.size(1) != int(args.topk):
        raise ValueError(f"Expected topk_scores K={args.topk}, got {topk_scores.size(1)}")
    if topk_bboxes_xywh.size(0) != int(args.topk):
        raise ValueError(f"Expected topk_bboxes K={args.topk}, got {topk_bboxes_xywh.size(0)}")
    if topk_ious.size(0) != int(args.topk):
        raise ValueError(f"Expected topk_ious K={args.topk}, got {topk_ious.size(0)}")


def assert_iou_target_shapes(best_iou, best_idx, baseline_iou, valid):
    if best_iou.dim() != 0:
        raise RuntimeError(f"best_iou must be scalar, got {tuple(best_iou.shape)}")
    if best_idx.dim() != 0:
        raise RuntimeError(f"best_idx must be scalar, got {tuple(best_idx.shape)}")
    if baseline_iou.dim() != 0:
        raise RuntimeError(f"baseline_iou must be scalar, got {tuple(baseline_iou.shape)}")
    if valid.dim() != 0 or valid.dtype != torch.bool:
        raise RuntimeError(f"valid must be scalar bool tensor, got shape={tuple(valid.shape)} dtype={valid.dtype}")


def choose_update(
    logits_t,
    topk_bboxes_xywh,
    topk_feats,
    target_idx,
    valid,
    epoch,
    args,
    prev_state=None,
    topk_scores=None,
    history_tokens=None,
    reliability_score=None,
    reliability_debug=None,
):
    """Choose the state/history update for the 3-stage Stage4 curriculum.

    Stage4-A / coach:
        state_idx = 0, so the crop center stays on SUTrack top-1.
        history_idx is teacher-forced early and gradually becomes predicted.

    Stage4-B / scheduled_takeover:
        high-confidence, spatially plausible predictions can take over state.
        history can still be teacher-forced according to teacher_prob.

    Stage4-C / safe_predicted:
        inference-like predicted-state rollout.
        prediction takes over only when confidence and motion gates are satisfied;
        otherwise it falls back to baseline top-1 and avoids teacher leakage.
    """
    probs = torch.softmax(logits_t.detach(), dim=-1)
    pred_idx = int(torch.argmax(probs, dim=-1).item())
    pred_prob = float(probs[0, pred_idx].item())
    teacher_idx = int(target_idx.item()) if bool(valid.item()) else 0
    use_teacher = bool(valid.item()) and (random.random() < teacher_prob(epoch, args))
    mode = effective_update_mode(epoch, args)

    if reliability_score is None:
        if topk_scores is not None and prev_state is not None:
            reliability_score, reliability_debug = compute_heuristic_reliability_for_update(
                logits_t=logits_t,
                topk_scores=topk_scores,
                pred_idx=pred_idx,
                topk_bboxes_xywh=topk_bboxes_xywh,
                prev_state=prev_state,
                args=args,
                history_tokens=history_tokens,
                topk_feats=topk_feats,
            )
        else:
            reliability_score = pred_prob
            reliability_debug = {
                "reliability_score": reliability_score,
                "reliability_pass": True,
                "pred_prob": pred_prob,
            }
    if reliability_debug is None:
        reliability_debug = {}

    use_reliability_gate = bool(getattr(args, "use_heuristic_reliability_gate", True))
    raw_reliability_pass = float(reliability_score) >= float(args.reliability_update_thresh)
    reliability_pass = (not use_reliability_gate) or raw_reliability_pass

    # Predicted-state takeover is only meaningful for a non-baseline candidate.
    motion_ok = (
        True
        if prev_state is None
        else candidate_motion_ok(topk_bboxes_xywh[pred_idx], prev_state, args)
    )
    scheduled_safe = (
        pred_idx != 0
        and pred_prob >= float(args.takeover_prob_thresh)
        and motion_ok
        and reliability_pass
    )
    safe_pred = (
        pred_idx != 0
        and pred_prob >= float(args.safe_pred_prob_thresh)
        and motion_ok
        and reliability_pass
    )

    if mode == "coach":
        state_idx = 0
        history_idx = teacher_idx if use_teacher else pred_idx
        state_takeover = False
        history_from_teacher = bool(use_teacher)

    elif mode == "scheduled_takeover":
        if scheduled_safe:
            state_idx = pred_idx
            state_takeover = True
        else:
            state_idx = 0
            state_takeover = False

        # Keep the memory curriculum softer than the state curriculum.
        # Teacher updates stabilize history early; otherwise use the actual rollout state.
        history_idx = teacher_idx if use_teacher else state_idx
        history_from_teacher = bool(use_teacher)

    elif mode == "safe_predicted":
        if safe_pred:
            state_idx = pred_idx
            state_takeover = True
        else:
            state_idx = 0
            state_takeover = False

        # No teacher leakage in the final inference-like stage.
        history_idx = state_idx
        history_from_teacher = False

    else:
        raise ValueError(f"Unknown update_mode/effective stage: {mode}")

    if (
        use_reliability_gate
        and not history_from_teacher
        and history_idx == pred_idx
        and pred_idx != 0
        and not raw_reliability_pass
    ):
        if state_idx != pred_idx:
            history_idx = state_idx
        else:
            history_idx = 0

    state_bbox = topk_bboxes_xywh[state_idx]
    history_feat = topk_feats[history_idx]
    if args.detach_history:
        history_feat = history_feat.detach()

    return {
        "state_bbox": state_bbox,
        "history_feat": history_feat,
        "pred_idx": pred_idx,
        "pred_prob": pred_prob,
        "teacher_idx": teacher_idx,
        "use_teacher": use_teacher,
        "state_idx": int(state_idx),
        "history_idx": int(history_idx),
        "state_takeover": bool(state_takeover),
        "history_from_teacher": bool(history_from_teacher),
        "effective_update_mode": mode,
        "reliability_score": float(reliability_score),
        "reliability_pass": bool(reliability_pass),
        "raw_reliability_pass": bool(raw_reliability_pass),
        "reliability_blocked": bool(use_reliability_gate and pred_idx != 0 and not raw_reliability_pass),
        "reliability_debug": dict(reliability_debug),
    }

def grad_norm(module):
    total = 0.0
    for param in module.parameters():
        if param.grad is not None:
            total += float(param.grad.detach().norm().item())
    return total


def cuda_mem(device):
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        return 0.0, 0.0
    dev = torch.device(device)
    return (
        torch.cuda.memory_allocated(dev) / (1024.0 ** 2),
        torch.cuda.memory_reserved(dev) / (1024.0 ** 2),
    )


def run_clip(model, disambiguator, criterion, data, epoch, args, cfg_obj):
    device = torch.device(args.device)
    template_images = as_single_image_list(data["template_images"])
    search_images = as_single_image_list(data["search_images"])
    template_bboxes = as_single_bbox_list(data["template_anno"])
    search_bboxes = as_single_bbox_list(data["search_anno"])
    if len(search_images) == 0:
        raise RuntimeError("Empty search clip.")

    template_list, template_anno_list = build_template_inputs(template_images, template_bboxes, cfg_obj, args)
    state = template_bboxes[-1].float().tolist()
    history_tokens = []
    disambiguator.reset_history(batch_size=1)

    logits_list = []
    target_list = []
    valid_list = []
    iou_list = []
    baseline_iou_list = []
    best_iou_list = []
    apply_label_list = []
    pred_idx_list = []
    state_idx_list = []
    history_idx_list = []
    state_takeover_list = []
    history_teacher_list = []
    reliability_score_list = []
    reliability_pass_list = []
    reliability_block_list = []
    pred_prob_list = []
    pred_correct_list = []
    target_nonzero_list = []
    fallback_frames = 0

    text_src = None
    task_index = torch.tensor([0], device=device, dtype=torch.long)

    for t, (image, gt_bbox_cpu) in enumerate(zip(search_images, search_bboxes)):
        image_h, image_w = image.shape[:2]
        x_patch, resize_factor = sample_target(
            image,
            state,
            cfg_obj.TEST.SEARCH_FACTOR,
            output_sz=cfg_obj.TEST.SEARCH_SIZE,
        )
        search = preprocess_image(
            x_patch,
            device=device,
            multi_modal_vision=bool(getattr(cfg_obj.DATA, "MULTI_MODAL_VISION", False)),
        )

        with torch.no_grad():
            enc_opt = model.forward_encoder(template_list, [search], template_anno_list, text_src, task_index)

        if args.unfreeze_decoder:
            out_dict = model.forward_decoder(enc_opt)
        else:
            with torch.no_grad():
                out_dict = model.forward_decoder(enc_opt)
        out_dict = maybe_apply_tracking_window(out_dict, cfg_obj)

        try:
            topk_feats, topk_scores, peaks_xy, topk_bboxes_norm = extract_topk_candidates(out_dict, model, args)
            topk_bboxes_xywh = map_boxes_back_batch(
                topk_bboxes_norm.squeeze(0),
                state,
                image_hw=(image_h, image_w),
                search_size=cfg_obj.TEST.SEARCH_SIZE,
                resize_factor=resize_factor,
            )
            topk_ious = compute_topk_ious(topk_bboxes_xywh, gt_bbox_cpu.to(device=device))
            assert_topk_shapes(topk_feats, topk_scores, topk_bboxes_xywh, topk_ious, args)
            best_iou, best_idx = torch.max(topk_ious, dim=-1)
            baseline_iou = topk_ious[0]
            inside_crop = gt_inside_search_crop(state, gt_bbox_cpu, cfg_obj.TEST.SEARCH_SIZE, resize_factor)
            valid = torch.tensor(
                bool(inside_crop) and float(best_iou.item()) >= float(args.iou_thresh),
                device=device,
                dtype=torch.bool,
            )
            assert_iou_target_shapes(best_iou, best_idx, baseline_iou, valid)
        except Exception as exc:
            fallback_frames += 1
            print(f"[WARN] top-k extraction failed at t={t}: {exc}")
            score_map = out_dict["score_map"]
            size_map = out_dict.get("size_map", None)
            offset_map = out_dict.get("offset_map", None)
            if size_map is None or offset_map is None:
                raise
            bbox_norm, top1_score = model.decoder.cal_bbox(score_map, size_map, offset_map, return_score=True)
            topk_bboxes_norm = bbox_norm.view(1, 1, 4).expand(1, args.topk, 4).contiguous()
            topk_bboxes_xywh = map_boxes_back_batch(
                topk_bboxes_norm.squeeze(0),
                state,
                image_hw=(image_h, image_w),
                search_size=cfg_obj.TEST.SEARCH_SIZE,
                resize_factor=resize_factor,
            )
            topk_feats = out_dict["f_map"].flatten(2).mean(dim=-1).unsqueeze(1).expand(1, args.topk, -1).contiguous()
            topk_scores = top1_score.view(1, 1).expand(1, args.topk).contiguous()
            topk_ious = compute_topk_ious(topk_bboxes_xywh, gt_bbox_cpu.to(device=device))
            best_iou, best_idx = torch.max(topk_ious, dim=-1)
            baseline_iou = topk_ious[0]
            valid = torch.tensor(False, device=device, dtype=torch.bool)
            assert_iou_target_shapes(best_iou, best_idx, baseline_iou, valid)

        hist_tensor = make_history_tensor(history_tokens, args.history_len)
        logits_t = disambiguator.forward_topk(
            topk_feats,
            topk_scores,
            history_tokens=hist_tensor,
        )

        iou_gain = best_iou - baseline_iou
        apply_label = (
            int(best_idx.item()) != 0
            and float(iou_gain.item()) >= float(args.apply_gain_thr)
            and float(best_iou.item()) >= float(args.best_iou_thr)
        )

        update_info = choose_update(
            logits_t=logits_t,
            topk_bboxes_xywh=topk_bboxes_xywh,
            topk_feats=topk_feats.squeeze(0),
            target_idx=best_idx.long(),
            valid=valid,
            epoch=epoch,
            args=args,
            prev_state=state,
            topk_scores=topk_scores,
            history_tokens=history_tokens,
        )

        state = update_info["state_bbox"].detach().cpu().tolist()
        history_tokens.append(update_info["history_feat"])
        if len(history_tokens) > int(args.history_len):
            history_tokens = history_tokens[-int(args.history_len):]

        logits_list.append(logits_t.squeeze(0))
        target_list.append(best_idx.long())
        valid_list.append(valid)
        iou_list.append(topk_ious.detach())
        baseline_iou_list.append(baseline_iou.detach())
        best_iou_list.append(best_iou.detach())
        apply_label_list.append(float(apply_label))
        pred_idx_list.append(update_info["pred_idx"])
        state_idx_list.append(update_info["state_idx"])
        history_idx_list.append(update_info["history_idx"])
        state_takeover_list.append(float(update_info["state_takeover"]))
        history_teacher_list.append(float(update_info["history_from_teacher"]))
        reliability_score_list.append(float(update_info["reliability_score"]))
        reliability_pass_list.append(float(update_info["reliability_pass"]))
        reliability_block_list.append(float(update_info["reliability_blocked"]))
        pred_prob_list.append(float(update_info["pred_prob"]))
        pred_correct_list.append(float(update_info["pred_idx"] == int(best_idx.item()) and bool(valid.item())))
        target_nonzero_list.append(float(int(best_idx.item()) != 0 and bool(valid.item())))

        del out_dict, enc_opt, search, topk_feats, topk_scores, topk_bboxes_norm

    logits = torch.stack(logits_list, dim=0)
    target_idx = torch.stack(target_list, dim=0).long()
    valid_mask = torch.stack(valid_list, dim=0).bool()
    topk_ious = torch.stack(iou_list, dim=0)
    loss_dict = criterion(logits, target_idx, valid_mask, topk_ious=topk_ious)

    valid_count = int(valid_mask.sum().detach().item())
    valid_float = valid_mask.float()
    denom = max(valid_count, 1)
    clip_len = max(len(pred_idx_list), 1)
    pred_nonzero = sum(float(idx != 0) for idx in pred_idx_list) / clip_len
    state_takeover_ratio = sum(state_takeover_list) / clip_len
    history_teacher_ratio = sum(history_teacher_list) / clip_len
    avg_reliability_score = sum(reliability_score_list) / clip_len
    min_reliability_score = min(reliability_score_list) if len(reliability_score_list) > 0 else 0.0
    reliability_pass_ratio = sum(reliability_pass_list) / clip_len
    reliability_block_ratio = sum(reliability_block_list) / clip_len
    pred_correct = sum(pred_correct_list) / float(denom)
    target_nonzero = sum(target_nonzero_list) / float(denom)
    apply_positive = sum(apply_label_list) / float(max(len(apply_label_list), 1))

    metrics = {
        "loss": float(loss_dict["loss"].detach().item()),
        "ce_loss": float(loss_dict["ce_loss"].detach().item()),
        "rank_loss": float(loss_dict["rank_loss"].detach().item()),
        "valid_frames": valid_count,
        "avg_baseline_iou": float((torch.stack(baseline_iou_list) * valid_float).sum().item() / denom),
        "avg_best_iou": float((torch.stack(best_iou_list) * valid_float).sum().item() / denom),
        "avg_iou_gain": float(((torch.stack(best_iou_list) - torch.stack(baseline_iou_list)) * valid_float).sum().item() / denom),
        "target_nonzero_ratio": target_nonzero,
        "apply_positive_ratio": apply_positive,
        "pred_nonzero_ratio": pred_nonzero,
        "state_takeover_ratio": state_takeover_ratio,
        "history_teacher_ratio": history_teacher_ratio,
        "avg_reliability_score": avg_reliability_score,
        "min_reliability_score": min_reliability_score,
        "reliability_pass_ratio": reliability_pass_ratio,
        "reliability_block_ratio": reliability_block_ratio,
        "pred_correct_ratio": pred_correct,
        "fallback_frames": fallback_frames,
        "effective_update_mode": effective_update_mode(epoch, args),
        "last_target_idx": int(target_idx[-1].item()) if target_idx.numel() > 0 else -1,
        "last_pred_idx": int(pred_idx_list[-1]) if len(pred_idx_list) > 0 else -1,
        "last_state_idx": int(state_idx_list[-1]) if len(state_idx_list) > 0 else -1,
        "last_history_idx": int(history_idx_list[-1]) if len(history_idx_list) > 0 else -1,
        "last_reliability_score": float(reliability_score_list[-1]) if len(reliability_score_list) > 0 else 0.0,
        "last_reliability_pass": bool(reliability_pass_list[-1]) if len(reliability_pass_list) > 0 else False,
        "last_pred_prob": float(pred_prob_list[-1]) if len(pred_prob_list) > 0 else 0.0,
        "last_baseline_iou": float(baseline_iou_list[-1].item()) if len(baseline_iou_list) > 0 else 0.0,
        "last_best_iou": float(best_iou_list[-1].item()) if len(best_iou_list) > 0 else 0.0,
        "last_valid": bool(valid_mask[-1].item()) if valid_mask.numel() > 0 else False,
    }
    return loss_dict, metrics


def save_checkpoint(path, model, disambiguator, optimizer, epoch, args, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "sutrack": model.state_dict(),
            "disambiguator": disambiguator.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "metrics": metrics,
            "stage": "stage4_online_joint",
        },
        path,
    )


def save_delta_checkpoint(path, model, disambiguator, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "disambiguator": disambiguator.state_dict(),
        "stage": "stage4_online_joint_delta",
    }
    if args.unfreeze_decoder:
        payload["decoder"] = model.decoder.state_dict()
    torch.save(payload, path)


def main():
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("Stage4 online rollout currently requires --batch-size 1.")
    if not args.freeze_encoder:
        raise ValueError("Stage4 first implementation keeps SUTrack encoder frozen; use --freeze-encoder.")
    if args.rollout_len > 16:
        print("[WARN] rollout_len > 16 can OOM quickly; smoke tests should start with rollout_len=8.")

    set_seed(args.seed)
    cfg_obj = load_config(args.config)
    device = torch.device(args.device)

    loader = build_online_loader(args, cfg_obj)
    model = build_sutrack(cfg_obj).to(device)
    load_sutrack_checkpoint(model, args.sutrack_ckpt)
    disambiguator = build_disambiguator(model, args).to(device)

    set_trainable(model, disambiguator, args)
    optimizer = build_optimizer(model, disambiguator, args)
    criterion = RerankLoss(rank_lambda=args.rank_lambda, margin=args.margin, loss_mode="hard").to(device)
    output_dir = Path(args.output_dir)

    print("=" * 100)
    print("[INFO] STARTrack Stage4 online joint training")
    print(f"[INFO] train_yaml             : {args.train_yaml}")
    print(f"[INFO] dataset                : {args.dataset}")
    print(f"[INFO] config                 : {args.config}")
    print(f"[INFO] stage4_sampling_mode   : {getattr(loader.dataset, 'stage4_sampling_mode', 'random')}")
    print(f"[INFO] hard_frame_list        : {args.hard_frame_list}")
    print(f"[INFO] hard_frame_count       : {getattr(loader.dataset, 'hard_frame_count', 0)}")
    print(f"[INFO] hard_seq_count         : {getattr(loader.dataset, 'hard_seq_count', 0)}")
    print(f"[INFO] hard_frame_ratio       : {args.hard_frame_ratio}")
    print(f"[INFO] hard_seq_ratio         : {args.hard_seq_ratio}")
    print(f"[INFO] full_ratio             : {args.full_ratio}")
    print(f"[INFO] anchor_jitter          : {getattr(loader.dataset, 'anchor_jitter', int(args.rollout_len) - 1 if int(args.anchor_jitter) < 0 else int(args.anchor_jitter))}")
    coach_epochs, scheduled_epochs = resolve_curriculum_epochs(args)
    print(f"[INFO] update_mode            : {args.update_mode}")
    print(f"[INFO] curriculum             : coach({coach_epochs}) -> scheduled_takeover({scheduled_epochs}) -> safe_predicted(rest)")
    print(f"[INFO] takeover_prob_thresh   : {args.takeover_prob_thresh}")
    print(f"[INFO] safe_pred_prob_thresh  : {args.safe_pred_prob_thresh}")
    print(f"[INFO] max_motion_norm        : {args.max_motion_norm}")
    print(f"[INFO] max_scale_ratio        : {args.max_scale_ratio}")
    print(f"[INFO] reliability_gate       : {args.use_heuristic_reliability_gate}")
    print(f"[INFO] reliability_thresh     : {args.reliability_update_thresh}")
    print(f"[INFO] rollout_len            : {args.rollout_len}")
    print(f"[INFO] topk                   : {args.topk}")
    print(f"[INFO] iou_thresh             : {args.iou_thresh}")
    print(f"[INFO] freeze_encoder         : {args.freeze_encoder}")
    print(f"[INFO] unfreeze_decoder       : {args.unfreeze_decoder}")
    print(f"[INFO] detach_history         : {args.detach_history}")
    print(f"[INFO] trainable reranker     : {count_params(disambiguator):,}")
    print(f"[INFO] trainable decoder      : {count_params(model.decoder):,}")
    print("=" * 100)

    best_loss = math.inf
    start_epoch = 1
    if args.resume and os.path.isfile(args.resume):
        print(f"[*] Loading resume checkpoint from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device)
        
        # 恢复主干网络 SUTrack 状态
        if "sutrack" in ckpt:
            model.load_state_dict(ckpt["sutrack"])
        # 恢复重排序器 Mamba 状态
        if "disambiguator" in ckpt:
            disambiguator.load_state_dict(ckpt["disambiguator"])
        # 恢复优化器状态 (保留动量和学习率衰减进度)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            
        start_epoch = ckpt.get("epoch", 0) + 1
        best_loss = ckpt.get("metrics", {}).get("loss", math.inf)
        print(f"[*] Resumed training from epoch {start_epoch}, previous best loss: {best_loss:.6f}")
    for epoch in range(start_epoch, args.epochs + 1):
        model.eval()
        model.encoder.eval()
        if args.unfreeze_decoder:
            model.decoder.train()
        else:
            model.decoder.eval()
        disambiguator.train()

        totals = {
            "loss": 0.0,
            "ce_loss": 0.0,
            "rank_loss": 0.0,
            "valid_frames": 0,
            "avg_baseline_iou": 0.0,
            "avg_best_iou": 0.0,
            "avg_iou_gain": 0.0,
            "target_nonzero_ratio": 0.0,
            "apply_positive_ratio": 0.0,
            "pred_nonzero_ratio": 0.0,
            "state_takeover_ratio": 0.0,
            "history_teacher_ratio": 0.0,
            "avg_reliability_score": 0.0,
            "min_reliability_score": 0.0,
            "reliability_pass_ratio": 0.0,
            "reliability_block_ratio": 0.0,
            "pred_correct_ratio": 0.0,
            "clips": 0,
            "skipped": 0,
            "time": 0.0,
            "grad_reranker": 0.0,
            "grad_decoder": 0.0,
            "sampled_hard_frame_clips": 0,
            "sampled_hard_seq_clips": 0,
            "sampled_full_clips": 0,
        }
        prob = teacher_prob(epoch, args)
        pbar = tqdm(loader, desc=f"Stage4 Epoch {epoch:03d}/{args.epochs}", dynamic_ncols=True, leave=False)
        for step, data in enumerate(pbar, start=1):
            start_time = time.time()
            optimizer.zero_grad(set_to_none=True)
            sample_source = get_batch_meta(data, "sample_source", "full")
            anchor_frame = get_batch_meta(data, "anchor_frame", -1)
            if sample_source == "hard_frame":
                totals["sampled_hard_frame_clips"] += 1
            elif sample_source == "hard_seq":
                totals["sampled_hard_seq_clips"] += 1
            else:
                totals["sampled_full_clips"] += 1

            try:
                loss_dict, clip_metrics = run_clip(model, disambiguator, criterion, data, epoch, args, cfg_obj)
            except Exception as exc:
                totals["skipped"] += 1
                print(f"[WARN] skip clip due to error: {exc}")
                optimizer.zero_grad(set_to_none=True)
                continue

            if (
                sample_source == "hard_frame"
                and clip_metrics["avg_iou_gain"] <= 0.0
                and clip_metrics["target_nonzero_ratio"] <= 0.0
            ):
                totals["skipped"] += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            loss = loss_dict["loss"]
            loss_value = float(loss.detach().item())
            if (not loss.requires_grad) or (not torch.isfinite(loss).all()):
                totals["skipped"] += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()
            gr = grad_norm(disambiguator)
            gd = grad_norm(model.decoder) if args.unfreeze_decoder else 0.0
            torch.nn.utils.clip_grad_norm_(
                [p for group in optimizer.param_groups for p in group["params"]],
                max_norm=float(args.grad_clip_norm),
            )
            optimizer.step()

            elapsed = time.time() - start_time
            totals["clips"] += 1
            totals["time"] += elapsed
            totals["loss"] += loss_value
            totals["ce_loss"] += clip_metrics["ce_loss"]
            totals["rank_loss"] += clip_metrics["rank_loss"]
            totals["valid_frames"] += clip_metrics["valid_frames"]
            for key in (
                "avg_baseline_iou",
                "avg_best_iou",
                "avg_iou_gain",
                "target_nonzero_ratio",
                "apply_positive_ratio",
                "pred_nonzero_ratio",
                "state_takeover_ratio",
                "history_teacher_ratio",
                "avg_reliability_score",
                "min_reliability_score",
                "reliability_pass_ratio",
                "reliability_block_ratio",
                "pred_correct_ratio",
            ):
                totals[key] += clip_metrics[key]
            totals["grad_reranker"] += gr
            totals["grad_decoder"] += gd

            if step % max(int(args.print_every), 1) == 0:
                mem_alloc, mem_reserved = cuda_mem(args.device)
                seq_name = get_batch_meta(data, "seq_name", "unknown")
                frame_range = get_batch_meta(data, "search_frame_ids", "")
                pbar.set_postfix({
                    "loss": f"{loss_value:.3f}",
                    "valid": clip_metrics["valid_frames"],
                    "best": f"{clip_metrics['last_best_iou']:.3f}",
                    "gain": f"{clip_metrics['last_best_iou'] - clip_metrics['last_baseline_iou']:.3f}",
                    "rel": f"{clip_metrics['last_reliability_score']:.3f}",
                    "mem": f"{mem_alloc:.0f}/{mem_reserved:.0f}MB",
                })
                print(
                    f"[Clip] seq_name={seq_name} frame_range={frame_range} "
                    f"sample_source={sample_source} anchor_frame={anchor_frame} "
                    f"stage={clip_metrics['effective_update_mode']} "
                    f"target_idx={clip_metrics['last_target_idx']} pred_idx={clip_metrics['last_pred_idx']} "
                    f"state_idx={clip_metrics['last_state_idx']} history_idx={clip_metrics['last_history_idx']} "
                    f"pred_prob={clip_metrics['last_pred_prob']:.4f} "
                    f"reliability_score={clip_metrics['last_reliability_score']:.4f} "
                    f"reliability_pass={clip_metrics['last_reliability_pass']} "
                    f"baseline_iou={clip_metrics['last_baseline_iou']:.4f} "
                    f"best_iou={clip_metrics['last_best_iou']:.4f} "
                    f"iou_gain={clip_metrics['last_best_iou'] - clip_metrics['last_baseline_iou']:.4f} "
                    f"valid={clip_metrics['last_valid']} "
                    f"cuda_mem={mem_alloc:.1f}/{mem_reserved:.1f}MB"
                )

        denom = max(totals["clips"], 1)
        mem_alloc, mem_reserved = cuda_mem(args.device)
        metrics = {
            "loss": totals["loss"] / denom,
            "ce_loss": totals["ce_loss"] / denom,
            "rank_loss": totals["rank_loss"] / denom,
            "valid_frames": totals["valid_frames"],
            "avg_baseline_iou": totals["avg_baseline_iou"] / denom,
            "avg_best_iou": totals["avg_best_iou"] / denom,
            "avg_iou_gain": totals["avg_iou_gain"] / denom,
            "target_nonzero_ratio": totals["target_nonzero_ratio"] / denom,
            "apply_positive_ratio": totals["apply_positive_ratio"] / denom,
            "pred_nonzero_ratio": totals["pred_nonzero_ratio"] / denom,
            "state_takeover_ratio": totals["state_takeover_ratio"] / denom,
            "history_teacher_ratio": totals["history_teacher_ratio"] / denom,
            "avg_reliability_score": totals["avg_reliability_score"] / denom,
            "min_reliability_score": totals["min_reliability_score"] / denom,
            "reliability_pass_ratio": totals["reliability_pass_ratio"] / denom,
            "reliability_block_ratio": totals["reliability_block_ratio"] / denom,
            "pred_correct_ratio": totals["pred_correct_ratio"] / denom,
            "update_mode": args.update_mode,
            "effective_update_mode": effective_update_mode(epoch, args),
            "teacher_prob": prob,
            "unfreeze_decoder": bool(args.unfreeze_decoder),
            "freeze_encoder": bool(args.freeze_encoder),
            "rollout_len": int(args.rollout_len),
            "topk": int(args.topk),
            "grad_reranker": totals["grad_reranker"] / denom,
            "grad_decoder": totals["grad_decoder"] / denom,
            "time_per_clip": totals["time"] / denom,
            "fps": float(args.rollout_len) / max(totals["time"] / denom, 1e-6),
            "cuda_allocated_mb": mem_alloc,
            "cuda_reserved_mb": mem_reserved,
            "skipped": totals["skipped"],
            "sampled_hard_frame_clips": totals["sampled_hard_frame_clips"],
            "sampled_hard_seq_clips": totals["sampled_hard_seq_clips"],
            "sampled_full_clips": totals["sampled_full_clips"],
        }

        print(
            f"[Epoch {epoch:03d}] "
            f"Loss/total={metrics['loss']:.4f} Loss/ce={metrics['ce_loss']:.4f} "
            f"Loss/rank={metrics['rank_loss']:.4f} valid_frames={metrics['valid_frames']} "
            f"avg_baseline_iou={metrics['avg_baseline_iou']:.4f} "
            f"avg_best_iou={metrics['avg_best_iou']:.4f} avg_iou_gain={metrics['avg_iou_gain']:.4f} "
            f"target_nonzero_ratio={metrics['target_nonzero_ratio']:.4f} "
            f"apply_positive_ratio={metrics['apply_positive_ratio']:.4f} "
            f"pred_nonzero_ratio={metrics['pred_nonzero_ratio']:.4f} "
            f"state_takeover_ratio={metrics['state_takeover_ratio']:.4f} "
            f"history_teacher_ratio={metrics['history_teacher_ratio']:.4f} "
            f"avg_reliability_score={metrics['avg_reliability_score']:.4f} "
            f"min_reliability_score={metrics['min_reliability_score']:.4f} "
            f"reliability_pass_ratio={metrics['reliability_pass_ratio']:.4f} "
            f"reliability_block_ratio={metrics['reliability_block_ratio']:.4f} "
            f"pred_correct_ratio={metrics['pred_correct_ratio']:.4f} "
            f"update_mode={metrics['update_mode']} effective_stage={metrics['effective_update_mode']} "
            f"teacher_prob={metrics['teacher_prob']:.3f} "
            f"unfreeze_decoder={metrics['unfreeze_decoder']} freeze_encoder={metrics['freeze_encoder']} "
            f"rollout_len={metrics['rollout_len']} topk={metrics['topk']} "
            f"Grad/reranker={metrics['grad_reranker']:.6f} Grad/decoder={metrics['grad_decoder']:.6f} "
            f"FPS={metrics['fps']:.2f} time/clip={metrics['time_per_clip']:.3f}s "
            f"CUDA={metrics['cuda_allocated_mb']:.1f}/{metrics['cuda_reserved_mb']:.1f}MB "
            f"sampled_hard_frame_clips={metrics['sampled_hard_frame_clips']} "
            f"sampled_hard_seq_clips={metrics['sampled_hard_seq_clips']} "
            f"sampled_full_clips={metrics['sampled_full_clips']}"
        )

        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            save_checkpoint(output_dir / "best.pth", model, disambiguator, optimizer, epoch, args, metrics)
            save_delta_checkpoint(output_dir / "best_delta.pth", model, disambiguator, args)

        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(output_dir / f"epoch_{epoch:03d}.pth", model, disambiguator, optimizer, epoch, args, metrics)
            save_delta_checkpoint(output_dir / f"epoch_{epoch:03d}_delta.pth", model, disambiguator, args)

    save_checkpoint(output_dir / "last.pth", model, disambiguator, optimizer, args.epochs, args, {"best_loss": best_loss})
    save_delta_checkpoint(output_dir / "last_delta.pth", model, disambiguator, args)
    print(f"[DONE] Stage4 online joint training finished. best_loss={best_loss:.6f}")


if __name__ == "__main__":
    main()
