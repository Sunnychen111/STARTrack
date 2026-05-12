#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Offline validation for STARTrack Top-K reranker / Mamba Memory Bank.

This script validates checkpoints on pre-extracted offline .pt features.

It does NOT run image-level SUTrack inference.
It evaluates whether the trained reranker can select oracle_best_idx among Top-K candidates.

Supported eval modes:
    framewise:
        no history, no Mamba memory.

    gt_tf:
        teacher forcing with gt_feat history.

    oracle_tf:
        teacher forcing with oracle candidate feature history.

    autoreg:
        predicted candidate feature history, closest to inference-time autoregression.

Recommended checkpoint comparison:
    best.pth
    epoch_025.pth
    epoch_026.pth
    last.pth
"""

import os
import csv
import math
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F
from tqdm import tqdm


try:
    from post_decoder_disambiguator_4 import PostDecoderDisambiguator
except ImportError:
    from lib.models.sutrack.post_decoder_disambiguator import PostDecoderDisambiguator


# ============================================================
# 1. IO
# ============================================================

def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def find_pt_files(feature_dir, max_files=None):
    feature_dir = Path(feature_dir)

    pt_files = sorted([
        p for p in feature_dir.rglob("*.pt")
        if p.is_file()
    ])

    if max_files is not None:
        pt_files = pt_files[:int(max_files)]

    return pt_files


def find_checkpoints(paths):
    ckpts = []

    for x in paths:
        p = Path(x)

        if p.is_file():
            ckpts.append(p)

        elif p.is_dir():
            for name in [
                "best.pth",
                "epoch_025.pth",
                "epoch_026.pth",
                "epoch_030.pth",
                "last.pth",
            ]:
                q = p / name
                if q.is_file():
                    ckpts.append(q)

        else:
            print(f"[WARNING] checkpoint path not found: {x}")

    # unique while preserving order
    seen = set()
    unique = []

    for p in ckpts:
        s = str(p)
        if s not in seen:
            seen.add(s)
            unique.append(p)

    return unique


# ============================================================
# 2. Data conversion
# ============================================================

def sequence_to_tensors(data, device, max_frames_per_seq=None):
    frames = data.get("frames", [])

    if max_frames_per_seq is not None:
        frames = frames[:int(max_frames_per_seq)]

    topk_feats = []
    topk_scores = []
    oracle_best_idx = []
    gt_inside_crop = []
    gt_feat = []

    for f in frames:
        required = [
            "topk_feats",
            "topk_scores",
            "oracle_best_idx",
            "gt_inside_crop",
            "gt_feat",
        ]

        if any(k not in f for k in required):
            continue

        topk_feats.append(f["topk_feats"].float())
        topk_scores.append(f["topk_scores"].float())
        oracle_best_idx.append(int(f["oracle_best_idx"]))
        gt_inside_crop.append(bool(f["gt_inside_crop"]))
        gt_feat.append(f["gt_feat"].float())

    if len(topk_feats) == 0:
        return None

    out = {
        "topk_feats": torch.stack(topk_feats, dim=0).to(device),          # [T,K,C]
        "topk_scores": torch.stack(topk_scores, dim=0).to(device),        # [T,K]
        "oracle_best_idx": torch.tensor(
            oracle_best_idx,
            dtype=torch.long,
            device=device,
        ),                                                               # [T]
        "gt_inside_crop": torch.tensor(
            gt_inside_crop,
            dtype=torch.bool,
            device=device,
        ),                                                               # [T]
        "gt_feat": torch.stack(gt_feat, dim=0).to(device),                # [T,C]
    }

    return out


# ============================================================
# 3. Model
# ============================================================

def infer_feat_dim_from_feature_dir(feature_dir):
    pt_files = find_pt_files(feature_dir, max_files=50)

    for p in pt_files:
        data = safe_torch_load(p, map_location="cpu")
        frames = data.get("frames", [])

        for f in frames:
            if "topk_feats" in f:
                return int(f["topk_feats"].shape[-1])

    raise RuntimeError(f"Could not infer feat_dim from: {feature_dir}")


def build_model(feat_dim, topk, ckpt_args, cli_args):
    history_len = int(ckpt_args.get("history_len", cli_args.history_len))
    mamba_d_state = int(ckpt_args.get("mamba_d_state", cli_args.mamba_d_state))
    mamba_expand = int(ckpt_args.get("mamba_expand", cli_args.mamba_expand))

    use_mamba_history = bool(
        ckpt_args.get("use_mamba_history", cli_args.use_mamba_history)
    )

    model = PostDecoderDisambiguator(
        feat_dim=feat_dim,
        history_len=history_len,
        topk_peaks=topk,

        use_mamba_history=use_mamba_history,
        use_mamba_history_bank=use_mamba_history,
        mamba_d_state=mamba_d_state,
        mamba_expand=mamba_expand,

        use_template_anchor=False,
        use_first_frame_anchor=False,
        use_history_aware_rerank_score=False,
    )

    return model


def load_model_from_checkpoint(ckpt_path, feat_dim, topk, device, cli_args):
    ckpt = safe_torch_load(ckpt_path, map_location="cpu")

    ckpt_args = ckpt.get("args", {})
    state = ckpt.get("model", ckpt)

    model = build_model(
        feat_dim=feat_dim,
        topk=topk,
        ckpt_args=ckpt_args,
        cli_args=cli_args,
    )

    missing, unexpected = model.load_state_dict(state, strict=False)

    model.to(device)
    model.eval()

    print(f"[INFO] Loaded checkpoint: {ckpt_path}")
    print(f"[INFO] missing keys   : {len(missing)}")
    print(f"[INFO] unexpected keys: {len(unexpected)}")

    if len(missing) > 0:
        print("[WARNING] First missing keys:", missing[:5])

    if len(unexpected) > 0:
        print("[WARNING] First unexpected keys:", unexpected[:5])

    return model, ckpt_args


# ============================================================
# 4. History helpers
# ============================================================

def make_history_tensor(history_tokens, history_len):
    if len(history_tokens) == 0:
        return None

    tokens = history_tokens[-int(history_len):]
    hist = torch.stack(tokens, dim=0).unsqueeze(0)  # [1,L,C]

    return hist


def select_update_feature(
    eval_mode,
    t,
    pred_idx,
    sample,
):
    topk_feats = sample["topk_feats"]              # [T,K,C]
    oracle_best_idx = sample["oracle_best_idx"]    # [T]
    gt_feat = sample["gt_feat"]                    # [T,C]

    if eval_mode == "gt_tf":
        return gt_feat[t]

    if eval_mode == "oracle_tf":
        oracle_idx = int(oracle_best_idx[t].item())
        return topk_feats[t, oracle_idx]

    if eval_mode == "autoreg":
        return topk_feats[t, int(pred_idx)]

    return None


# ============================================================
# 5. Metrics
# ============================================================

def init_metrics():
    return {
        "num_sequences": 0,
        "valid_frames": 0,

        "baseline_correct": 0,
        "pred_correct": 0,

        "easy_frames": 0,
        "hard_frames": 0,

        "easy_keep_correct": 0,
        "wrong_switch_easy": 0,

        "hard_correct": 0,
        "hard_nonzero_pred": 0,
        "hard_to_top1_wrong": 0,

        "changed_frames": 0,

        "loss_sum": 0.0,
        "loss_count": 0,

        "oracle_hist": Counter(),
        "pred_hist": Counter(),
        "seq_hard_acc_sum": 0.0,
        "seq_hard_acc_count": 0,
    }


def update_metrics(metrics, logits, labels, valid_mask):
    """
    logits: [T,K]
    labels: [T]
    valid_mask: [T]
    """
    if valid_mask.sum().item() == 0:
        return

    logits_valid = logits[valid_mask]
    labels_valid = labels[valid_mask]

    pred = torch.argmax(logits_valid, dim=-1)

    ce = F.cross_entropy(logits_valid, labels_valid, reduction="mean")
    metrics["loss_sum"] += float(ce.detach().item())
    metrics["loss_count"] += 1

    labels_cpu = labels_valid.detach().cpu()
    pred_cpu = pred.detach().cpu()

    for y, p in zip(labels_cpu.tolist(), pred_cpu.tolist()):
        y = int(y)
        p = int(p)

        metrics["valid_frames"] += 1

        # baseline always selects top1, i.e., index 0
        if y == 0:
            metrics["baseline_correct"] += 1

        if p == y:
            metrics["pred_correct"] += 1

        if p != 0:
            metrics["changed_frames"] += 1

        metrics["oracle_hist"][y] += 1
        metrics["pred_hist"][p] += 1

        if y == 0:
            metrics["easy_frames"] += 1

            if p == 0:
                metrics["easy_keep_correct"] += 1
            else:
                metrics["wrong_switch_easy"] += 1

        else:
            metrics["hard_frames"] += 1

            if p == y:
                metrics["hard_correct"] += 1

            if p != 0:
                metrics["hard_nonzero_pred"] += 1
            else:
                metrics["hard_to_top1_wrong"] += 1


def finalize_metrics(metrics):
    eps = 1e-12

    valid = metrics["valid_frames"]
    easy = metrics["easy_frames"]
    hard = metrics["hard_frames"]

    out = dict(metrics)

    out["avg_ce"] = metrics["loss_sum"] / max(metrics["loss_count"], 1)

    out["baseline_acc"] = metrics["baseline_correct"] / max(valid, 1)
    out["rerank_acc"] = metrics["pred_correct"] / max(valid, 1)
    out["gain_vs_baseline"] = out["rerank_acc"] - out["baseline_acc"]

    out["easy_keep_acc"] = metrics["easy_keep_correct"] / max(easy, 1)
    out["wrong_switch_rate"] = metrics["wrong_switch_easy"] / max(easy, 1)

    out["hard_acc"] = metrics["hard_correct"] / max(hard, 1)
    out["hard_nonzero_recall"] = metrics["hard_nonzero_pred"] / max(hard, 1)
    out["hard_to_top1_wrong_rate"] = metrics["hard_to_top1_wrong"] / max(hard, 1)

    out["change_rate"] = metrics["changed_frames"] / max(valid, 1)
    out["hard_ratio"] = hard / max(valid, 1)

    out["oracle_hist_str"] = str(dict(sorted(metrics["oracle_hist"].items())))
    out["pred_hist_str"] = str(dict(sorted(metrics["pred_hist"].items())))

    return out


# ============================================================
# 6. Evaluation
# ============================================================

@torch.no_grad()
def evaluate_one_sequence(model, sample, eval_mode, history_len):
    topk_feats = sample["topk_feats"]              # [T,K,C]
    topk_scores = sample["topk_scores"]            # [T,K]
    gt_inside_crop = sample["gt_inside_crop"]      # [T]

    T = topk_feats.shape[0]

    model.reset_history(batch_size=1)

    logits_list = []
    history_tokens = []

    for t in range(T):
        if eval_mode == "framewise":
            hist_tensor = None
        else:
            hist_tensor = make_history_tensor(
                history_tokens,
                history_len=history_len,
            )

        logits_t = model.forward_topk(
            topk_feats[t:t + 1],
            topk_scores[t:t + 1],
            history_tokens=hist_tensor,
        )  # [1,K]

        logits_list.append(logits_t.squeeze(0))

        pred_idx = int(torch.argmax(logits_t, dim=-1).item())

        if eval_mode == "framewise":
            continue

        if not bool(gt_inside_crop[t].item()):
            continue

        update_feat = select_update_feature(
            eval_mode=eval_mode,
            t=t,
            pred_idx=pred_idx,
            sample=sample,
        )

        if update_feat is None:
            continue

        # Offline validation: input features are constants.
        history_tokens.append(update_feat.detach())

        if len(history_tokens) > int(history_len):
            history_tokens = history_tokens[-int(history_len):]

    logits = torch.stack(logits_list, dim=0)  # [T,K]

    return logits


def evaluate_checkpoint(
    ckpt_path,
    feature_files,
    feat_dim,
    topk,
    device,
    args,
):
    model, ckpt_args = load_model_from_checkpoint(
        ckpt_path,
        feat_dim=feat_dim,
        topk=topk,
        device=device,
        cli_args=args,
    )

    history_len = int(ckpt_args.get("history_len", args.history_len))

    all_results = []

    for eval_mode in args.eval_modes:
        metrics = init_metrics()

        pbar = tqdm(
            feature_files,
            desc=f"Eval {Path(ckpt_path).name} [{eval_mode}]",
            dynamic_ncols=True,
            leave=False,
        )

        for pt_path in pbar:
            try:
                data = safe_torch_load(pt_path, map_location="cpu")
                sample = sequence_to_tensors(
                    data,
                    device=device,
                    max_frames_per_seq=args.max_frames_per_seq,
                )

                if sample is None:
                    continue

                logits = evaluate_one_sequence(
                    model=model,
                    sample=sample,
                    eval_mode=eval_mode,
                    history_len=history_len,
                )

                labels = sample["oracle_best_idx"]
                valid_mask = sample["gt_inside_crop"]

                update_metrics(
                    metrics,
                    logits=logits,
                    labels=labels,
                    valid_mask=valid_mask,
                )

                metrics["num_sequences"] += 1

            except Exception as e:
                print(f"\n[ERROR] Failed evaluating {pt_path}: {repr(e)}")

        result = finalize_metrics(metrics)
        result["checkpoint"] = str(ckpt_path)
        result["checkpoint_name"] = Path(ckpt_path).name
        result["eval_mode"] = eval_mode

        all_results.append(result)

    return all_results


# ============================================================
# 7. Report
# ============================================================

def print_result_table(results):
    headers = [
        "checkpoint_name",
        "eval_mode",
        "valid_frames",
        "hard_frames",
        "baseline_acc",
        "rerank_acc",
        "gain_vs_baseline",
        "hard_acc",
        "hard_nonzero_recall",
        "easy_keep_acc",
        "wrong_switch_rate",
        "change_rate",
        "avg_ce",
    ]

    print("\n" + "=" * 140)
    print("[VALIDATION SUMMARY]")
    print("-" * 140)

    print(
        f"{'ckpt':24s} {'mode':12s} "
        f"{'valid':>8s} {'hard':>8s} "
        f"{'base':>8s} {'rerank':>8s} {'gain':>8s} "
        f"{'hard_acc':>9s} {'hard_rec':>9s} "
        f"{'easy_keep':>10s} {'wrong_sw':>10s} "
        f"{'chg':>8s} {'ce':>8s}"
    )

    for r in results:
        print(
            f"{r['checkpoint_name'][:24]:24s} "
            f"{r['eval_mode'][:12]:12s} "
            f"{int(r['valid_frames']):8d} "
            f"{int(r['hard_frames']):8d} "
            f"{r['baseline_acc']:8.4f} "
            f"{r['rerank_acc']:8.4f} "
            f"{r['gain_vs_baseline']:8.4f} "
            f"{r['hard_acc']:9.4f} "
            f"{r['hard_nonzero_recall']:9.4f} "
            f"{r['easy_keep_acc']:10.4f} "
            f"{r['wrong_switch_rate']:10.4f} "
            f"{r['change_rate']:8.4f} "
            f"{r['avg_ce']:8.4f}"
        )

    print("=" * 140)


def save_csv(results, out_csv):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "checkpoint",
        "checkpoint_name",
        "eval_mode",
        "num_sequences",
        "valid_frames",
        "easy_frames",
        "hard_frames",
        "hard_ratio",
        "baseline_acc",
        "rerank_acc",
        "gain_vs_baseline",
        "easy_keep_acc",
        "wrong_switch_rate",
        "hard_acc",
        "hard_nonzero_recall",
        "hard_to_top1_wrong_rate",
        "change_rate",
        "avg_ce",
        "oracle_hist_str",
        "pred_hist_str",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            row = {k: r.get(k, "") for k in fieldnames}
            writer.writerow(row)

    print(f"[INFO] Saved CSV to: {out_csv}")


# ============================================================
# 8. Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--feature-dir",
        type=str,
        required=True,
        help="Directory containing offline .pt features.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        nargs="+",
        required=True,
        help="Checkpoint file(s) or checkpoint directory.",
    )

    parser.add_argument(
        "--eval-modes",
        type=str,
        nargs="+",
        default=["framewise", "oracle_tf", "autoreg"],
        choices=["framewise", "gt_tf", "oracle_tf", "autoreg"],
    )

    parser.add_argument(
        "--output-csv",
        type=str,
        default="offline_validation_results.csv",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-frames-per-seq", type=int, default=None)

    parser.add_argument("--history-len", type=int, default=32)
    parser.add_argument("--mamba-d-state", type=int, default=16)
    parser.add_argument("--mamba-expand", type=int, default=2)
    parser.add_argument(
        "--use-mamba-history",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    feature_files = find_pt_files(
        args.feature_dir,
        max_files=args.max_files,
    )

    if len(feature_files) == 0:
        raise RuntimeError(f"No .pt files found under: {args.feature_dir}")

    ckpts = find_checkpoints(args.checkpoint)

    if len(ckpts) == 0:
        raise RuntimeError("No checkpoint found.")

    feat_dim = infer_feat_dim_from_feature_dir(args.feature_dir)

    first_data = safe_torch_load(feature_files[0], map_location="cpu")
    first_frame = first_data["frames"][0]
    topk = int(first_frame["topk_feats"].shape[0])

    print("=" * 100)
    print("[INFO] STARTrack offline validation")
    print(f"[INFO] feature_dir       : {args.feature_dir}")
    print(f"[INFO] num feature files : {len(feature_files)}")
    print(f"[INFO] checkpoints       : {len(ckpts)}")
    print(f"[INFO] eval_modes        : {args.eval_modes}")
    print(f"[INFO] feat_dim          : {feat_dim}")
    print(f"[INFO] topk              : {topk}")
    print(f"[INFO] device            : {args.device}")
    print("=" * 100)

    all_results = []

    for ckpt in ckpts:
        results = evaluate_checkpoint(
            ckpt_path=ckpt,
            feature_files=feature_files,
            feat_dim=feat_dim,
            topk=topk,
            device=args.device,
            args=args,
        )

        all_results.extend(results)

    print_result_table(all_results)
    save_csv(all_results, args.output_csv)


if __name__ == "__main__":
    main()