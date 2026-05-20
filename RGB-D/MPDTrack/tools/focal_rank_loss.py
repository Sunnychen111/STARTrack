import torch
import torch.nn as nn
import torch.nn.functional as F


class RerankLoss(nn.Module):
    def __init__(self, rank_lambda=0.05, margin=0.2, loss_mode="hard"):
        super().__init__()
        if loss_mode not in ("hard", "soft"):
            raise ValueError(f"Unknown loss_mode: {loss_mode}")
        self.rank_lambda = float(rank_lambda)
        self.margin = float(margin)
        self.loss_mode = str(loss_mode)
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        logits,
        target_idx,
        valid_mask,
        target_probs=None,
        topk_ious=None,
    ):
        if logits.dim() != 2:
            raise ValueError(f"logits must be [T, K], got {tuple(logits.shape)}")

        valid_mask = valid_mask.to(device=logits.device, dtype=torch.bool)
        target_idx = target_idx.to(device=logits.device, dtype=torch.long)
        invalid_valid_targets = valid_mask & ((target_idx < 0) | (target_idx >= logits.size(1)))
        if bool(invalid_valid_targets.any().item()):
            bad = target_idx[invalid_valid_targets][:5].detach().cpu().tolist()
            raise ValueError(f"target_idx contains out-of-range valid labels: {bad}")
        safe_target_idx = torch.where(valid_mask, target_idx, torch.zeros_like(target_idx)).clamp(
            min=0,
            max=logits.size(1) - 1,
        )
        valid_count = valid_mask.float().sum()

        if valid_count.item() <= 0:
            zero = logits.sum() * 0.0
            return {
                "loss": zero,
                "ce_loss": zero,
                "soft_loss": zero,
                "rank_loss": zero,
                "valid_frames": 0,
            }

        if self.loss_mode == "hard":
            cls_loss_per_frame = self.ce(logits, safe_target_idx)
            cls_loss = (cls_loss_per_frame * valid_mask.float()).sum() / valid_count.clamp_min(1.0)
            soft_loss = logits.sum() * 0.0
        else:
            if target_probs is None:
                if topk_ious is None:
                    raise ValueError("Soft loss requires target_probs or topk_ious.")
                ious = topk_ious.to(device=logits.device, dtype=logits.dtype).clamp_min(0)
                target_probs = ious / (ious.sum(dim=-1, keepdim=True) + 1e-6)
            else:
                target_probs = target_probs.to(device=logits.device, dtype=logits.dtype)

            log_probs = F.log_softmax(logits, dim=-1)
            cls_loss_per_frame = -(target_probs * log_probs).sum(dim=-1)
            cls_loss = (cls_loss_per_frame * valid_mask.float()).sum() / valid_count.clamp_min(1.0)
            soft_loss = cls_loss

        rank_loss = self._rank_loss(logits, safe_target_idx, valid_mask, topk_ious)
        loss = cls_loss + self.rank_lambda * rank_loss

        return {
            "loss": loss,
            "ce_loss": cls_loss,
            "soft_loss": soft_loss,
            "rank_loss": rank_loss,
            "valid_frames": int(valid_count.detach().item()),
        }

    def _rank_loss(self, logits, target_idx, valid_mask, topk_ious=None):
        if self.rank_lambda <= 0.0:
            return logits.sum() * 0.0

        t = torch.arange(logits.size(0), device=logits.device)
        best_logits = logits[t, target_idx]
        baseline_logits = logits[:, 0]

        rank_valid = valid_mask & (target_idx != 0)
        if topk_ious is not None:
            topk_ious = topk_ious.to(device=logits.device, dtype=logits.dtype)
            best_ious = topk_ious[t, target_idx]
            baseline_ious = topk_ious[:, 0]
            rank_valid = rank_valid & (best_ious > baseline_ious)

        rank_count = rank_valid.float().sum()
        if rank_count.item() <= 0:
            return logits.sum() * 0.0

        per_frame = F.relu(self.margin - (best_logits - baseline_logits))
        return (per_frame * rank_valid.float()).sum() / rank_count.clamp_min(1.0)
