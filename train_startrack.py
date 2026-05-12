#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import random
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from focal_rank_loss import RerankLoss
from trajectory_dataset import TrajectoryDataset
from lib.models.sutrack.post_decoder_disambiguator import PostDecoderDisambiguator


# ============================================================
# 1. Args (参数配置)
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train STARTrack Top-K reranker using IoU-aware offline features.")

    parser.add_argument("--feature-dir", required=True, help="Directory containing offline .pt files with topk_ious.")
    parser.add_argument("--output-dir", default="checkpoints/startrack_mamba_iou", help="Checkpoint directory.")
    
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file to resume training (e.g., last.pth)")
    
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--snippet-len", type=int, default=32)
    parser.add_argument("--samples-per-epoch", type=int, default=5000)
    parser.add_argument("--hard-prob", type=float, default=0.9)

    parser.add_argument("--iou-thresh", type=float, default=0.3, help="Top1 IoU must be >= this to compute loss.")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rank-lambda", type=float, default=0.05)
    parser.add_argument("--margin", type=float, default=0.2)

    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--history-len", type=int, default=32)
    parser.add_argument("--mamba-d-state", type=int, default=16)
    parser.add_argument("--mamba-expand", type=int, default=2)
    parser.add_argument("--use-mamba-history", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--history-mode", type=str, choices=["detached", "differentiable"], default="differentiable")
    parser.add_argument("--stage3-oracle-start", type=float, default=1.0)
    parser.add_argument("--stage3-oracle-end", type=float, default=0.2)

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--print-every", type=int, default=50)

    return parser.parse_args()


# ============================================================
# 2. Basic utilities (基础工具)
# ============================================================

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def stage_for_epoch(epoch):
    """课程学习 4 阶段划分"""
    if epoch <= 5: return 0
    if epoch <= 15: return 1
    if epoch <= 25: return 2
    return 3

def stage3_oracle_prob(epoch, start_prob, end_prob):
    """Stage 3 教师强制概率的线性衰减"""
    if epoch <= 26: return float(start_prob)
    if epoch >= 40: return float(end_prob)
    alpha = (epoch - 26) / float(40 - 26)
    return float(start_prob + alpha * (end_prob - start_prob))

def squeeze_batch1(batch, device):
    """去除 DataLoader 带来的 batch_size=1 的维度"""
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.squeeze(0).to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def build_model(feat_dim, args):
    """构建带 Mamba 记忆库的重排序模型"""
    return PostDecoderDisambiguator(
        feat_dim=feat_dim,
        history_len=args.history_len,
        topk_peaks=8,
        use_mamba_history=args.use_mamba_history,
        use_mamba_history_bank=args.use_mamba_history,
        mamba_d_state=args.mamba_d_state,
        mamba_expand=args.mamba_expand,
        use_template_anchor=False,
        use_first_frame_anchor=False,
        use_history_aware_rerank_score=False,
    )


# ============================================================
# 3. History Helpers (历史记忆库相关工具 - 基于 IoU)
# ============================================================

def make_history_tensor(history_tokens, history_len):
    """将 Python List 转为 Mamba 可用的 Tensor [1, L, C]"""
    if len(history_tokens) == 0: return None
    return torch.stack(history_tokens[-int(history_len):], dim=0).unsqueeze(0)

def select_update_feature(stage, epoch, t, topk_feats, best_idx_by_iou, gt_feat, logits_t, args):
    """根据不同的 Stage 决定拿什么特征去更新记忆库"""
    if stage == 1: return gt_feat[t]
    if stage == 2: return topk_feats[t, int(best_idx_by_iou[t].item())]

    # Stage 3: 概率决定是用真实最好框还是模型预测框
    oracle_prob = stage3_oracle_prob(epoch, args.stage3_oracle_start, args.stage3_oracle_end)
    if random.random() < oracle_prob:
        return topk_feats[t, int(best_idx_by_iou[t].item())]

    pred_idx = int(torch.argmax(logits_t.detach(), dim=-1).item())
    return topk_feats[t, pred_idx]

def get_mamba_grad_norm(model):
    """监控 Mamba 权重是否接收到了梯度"""
    total, count = 0.0, 0
    for name, p in model.named_parameters():
        if "history_bank.mamba" in name and p.grad is not None:
            total += float(p.grad.detach().norm().item())
            count += 1
    return total, count


# ============================================================
# 4. Forward functions (前向传播核心逻辑)
# ============================================================

def _compute_iou_metrics(logits, topk_ious, best_iou, valid_mask, loss_dict):
    """计算原始 IoU 和 Rerank 后的 IoU (在不计算梯度的环境下执行)"""
    with torch.no_grad():
        pred_idx = torch.argmax(logits, dim=-1) # 模型预测的 idx [T]
        baseline_iou = topk_ious[:, 0]          # 原始的 Top1 IoU [T]
        pred_iou = topk_ious.gather(1, pred_idx.unsqueeze(1)).squeeze(1) # Rerank后的 IoU [T]

        # 仅对 valid_mask=True 的帧求均值
        loss_dict["avg_best_iou"] = float(best_iou[valid_mask].mean().item()) if valid_mask.any() else 0.0
        loss_dict["avg_base_iou"] = float(baseline_iou[valid_mask].mean().item()) if valid_mask.any() else 0.0
        loss_dict["avg_pred_iou"] = float(pred_iou[valid_mask].mean().item()) if valid_mask.any() else 0.0
    return loss_dict


def forward_stage0(model, sample, criterion, args):
    """Stage 0: 纯空间重排序，不使用历史特征"""
    topk_feats = sample["topk_feats"]
    topk_scores = sample["topk_scores"]
    topk_ious = sample["topk_ious"]
    
    # 动态选取每一帧 IoU 最高的作为目标
    best_iou, best_idx_by_iou = torch.max(topk_ious, dim=-1)
    
    # 过滤掉出视野或候选框全军覆没的垃圾帧
    valid_mask = sample["gt_inside_crop"].bool() & (best_iou >= args.iou_thresh)

    model.reset_history(batch_size=1)
    logits = []
    for t in range(topk_feats.size(0)):
        logits_t = model.forward_topk(topk_feats[t:t+1], topk_scores[t:t+1], history_tokens=None)
        logits.append(logits_t.squeeze(0))
        
    logits = torch.stack(logits, dim=0)
    loss_dict = criterion(logits, best_idx_by_iou, valid_mask)
    
    # 🟢 增加 IoU 统计
    loss_dict = _compute_iou_metrics(logits, topk_ious, best_iou, valid_mask, loss_dict)
    return loss_dict, logits

def forward_autoregressive(model, sample, criterion, stage, epoch, args):
    """Stage 1-3: 自回归更新，训练 Mamba 记忆力"""
    topk_feats = sample["topk_feats"]
    topk_scores = sample["topk_scores"]
    topk_ious = sample["topk_ious"]
    gt_feat = sample["gt_feat"]

    best_iou, best_idx_by_iou = torch.max(topk_ious, dim=-1)
    valid_mask = sample["gt_inside_crop"].bool() & (best_iou >= args.iou_thresh)

    model.reset_history(batch_size=1)
    logits = []
    history_tokens = []

    for t in range(topk_feats.size(0)):
        hist_tensor = make_history_tensor(history_tokens, args.history_len)
        logits_t = model.forward_topk(topk_feats[t:t+1], topk_scores[t:t+1], history_tokens=hist_tensor)
        logits.append(logits_t.squeeze(0))

        if not valid_mask[t].item():
            continue

        update_feat = select_update_feature(stage, epoch, t, topk_feats, best_idx_by_iou, gt_feat, logits_t, args)
        history_tokens.append(update_feat.detach())
        if len(history_tokens) > int(args.history_len):
            history_tokens = history_tokens[-int(args.history_len):]

    logits = torch.stack(logits, dim=0)
    loss_dict = criterion(logits, best_idx_by_iou, valid_mask)
    
    # 🟢 增加 IoU 统计
    loss_dict = _compute_iou_metrics(logits, topk_ious, best_iou, valid_mask, loss_dict)
    return loss_dict, logits


# ============================================================
# 5. Checkpoint & Main
# ============================================================

def save_checkpoint(path, model, optimizer, epoch, args, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "metrics": metrics,
    }, path)

def main():
    args = parse_args()
    set_seed(args.seed)

    # 初始化 Dataset
    dataset = TrajectoryDataset(
        args.feature_dir,
        snippet_len=args.snippet_len,
        hard_prob=args.hard_prob,
        samples_per_epoch=args.samples_per_epoch,
        seed=args.seed,
    )
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    first = dataset[0]
    feat_dim = int(first["topk_feats"].shape[-1])

    # 构建模型、Loss 和优化器
    model = build_model(feat_dim, args).to(args.device)
    criterion = RerankLoss(rank_lambda=args.rank_lambda, margin=args.margin)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    output_dir = Path(args.output_dir)
    best_loss = math.inf
    start_epoch = 1

    if args.resume and os.path.isfile(args.resume):
        print(f"[*] Loading checkpoint from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("metrics", {}).get("loss", math.inf)
        print(f"[*] Resumed training from epoch {start_epoch}, previous best loss: {best_loss:.4f}")

    print("=" * 100)
    print(f"[INFO] STARTrack offline IoU-Aware training")
    print(f"[INFO] feature_dir: {args.feature_dir}")
    print(f"[INFO] iou_thresh : {args.iou_thresh} | Starting Epoch: {start_epoch}/{args.epochs}")
    print("=" * 100)

    for epoch in range(start_epoch, args.epochs + 1):
        stage = stage_for_epoch(epoch)
        model.train()

        # 强制每个 Epoch 为 Dataset 的独立采样器注入新的随机种子
        dataset.rng = random.Random(args.seed + epoch)

        totals = { "loss": 0.0, "ce_loss": 0.0, "rank_loss": 0.0, "valid_frames": 0, 
                   "avg_best_iou": 0.0, "avg_base_iou": 0.0, "avg_pred_iou": 0.0, 
                   "mamba_grad_before": 0.0, "num_steps": 0 }

        pbar = tqdm(loader, desc=f"Epoch {epoch:03d}/{args.epochs} [Stage {stage}]", leave=False, dynamic_ncols=True)

        for step, batch in enumerate(pbar, start=1):
            sample = squeeze_batch1(batch, args.device)
            optimizer.zero_grad(set_to_none=True)

            # 前向传播
            if stage == 0:
                loss_dict, _ = forward_stage0(model, sample, criterion, args)
            else:
                loss_dict, _ = forward_autoregressive(model, sample, criterion, stage, epoch, args)

            loss = loss_dict["loss"]
            
            # 安全提取各项 Loss 以便打印
            ce_loss_val = float(loss_dict.get("ce_loss", torch.tensor(0.0)).detach().item())
            rank_loss_val = float(loss_dict.get("rank_loss", torch.tensor(0.0)).detach().item())
            
            mamba_before, mamba_count = 0.0, 0

            # 反向传播与权重更新
            if loss.requires_grad and torch.isfinite(loss).all() and float(loss.detach().item()) > 0.0:
                loss.backward()
                mamba_before, mamba_count = get_mamba_grad_norm(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
                optimizer.step()

            # 统计
            totals["loss"] += float(loss.detach().item())
            totals["ce_loss"] += ce_loss_val
            totals["rank_loss"] += rank_loss_val
            totals["valid_frames"] += int(loss_dict["valid_frames"])
            totals["avg_best_iou"] += loss_dict.get("avg_best_iou", 0.0)
            totals["avg_base_iou"] += loss_dict.get("avg_base_iou", 0.0)
            totals["avg_pred_iou"] += loss_dict.get("avg_pred_iou", 0.0)
            totals["mamba_grad_before"] += float(mamba_before)
            totals["num_steps"] += 1

            if step % max(int(args.print_every), 1) == 0:
                # 🟢 进度条也加上 P-IoU (模型预测的 IoU) 的实时监控
                pbar_dict = { 
                    "loss": f"{loss.item():.3f}", 
                    "ce": f"{ce_loss_val:.3f}", 
                    "P-IoU": f"{loss_dict.get('avg_pred_iou', 0.0):.3f}" 
                }
                if stage >= 1: pbar_dict["mamba_g"] = f"{mamba_before:.3f}"
                pbar.set_postfix(pbar_dict)

        denom = max(totals["num_steps"], 1)
        metrics = {
            "loss": totals["loss"] / denom,
            "ce_loss": totals["ce_loss"] / denom,
            "rank_loss": totals["rank_loss"] / denom,
            "valid_frames": totals["valid_frames"],
            "avg_best_iou": totals["avg_best_iou"] / denom,
            "avg_base_iou": totals["avg_base_iou"] / denom,
            "avg_pred_iou": totals["avg_pred_iou"] / denom,
            "stage": stage,
        }

        # 🟢 在每轮结束时打印三组 IoU
        print(f"✅ Ep {epoch:03d} | Stg {stage} | Loss: {metrics['loss']:.4f} (CE: {metrics['ce_loss']:.4f}, Rank: {metrics['rank_loss']:.4f}) "
              f"| Valid Frm: {metrics['valid_frames']}")
        print(f"   [IoU] Ceiling(最高): {metrics['avg_best_iou']:.4f} | Base(原始): {metrics['avg_base_iou']:.4f} | Pred(预测): {metrics['avg_pred_iou']:.4f}")

        # 保存检查点
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            save_checkpoint(output_dir / "best.pth", model, optimizer, epoch, args, metrics)
            
        if epoch % args.save_every == 0:
            save_checkpoint(output_dir / f"epoch_{epoch:03d}.pth", model, optimizer, epoch, args, metrics)
            
        save_checkpoint(output_dir / "last.pth", model, optimizer, epoch, args, metrics)

    print("[DONE] Training finished.")

if __name__ == "__main__":
    main()