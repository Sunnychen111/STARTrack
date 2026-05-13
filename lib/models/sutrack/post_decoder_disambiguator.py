import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm.modules.mamba_simple import Mamba
except ImportError:
    Mamba = None


class MambaHistoryBank(nn.Module):
    def __init__(self, feat_dim, history_len=32, use_mamba=True, d_state=16, expand=2):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.history_len = int(history_len)
        self.use_mamba = bool(use_mamba)

        if self.use_mamba:
            if Mamba is None:
                raise ImportError("mamba_ssm is required when use_mamba=True.")
            self.norm = nn.LayerNorm(self.feat_dim)
            self.mamba = Mamba(
                d_model=self.feat_dim,
                d_state=d_state,
                expand=expand,
            )
            self._zero_init_output_projection()
        else:
            self.norm = None
            self.mamba = None

        self.cached_tokens = None
        self.cached_history = None
        self.state_batch = None

    def _zero_init_output_projection(self):
        out_proj = getattr(self.mamba, "out_proj", None)
        if out_proj is None:
            return
        nn.init.zeros_(out_proj.weight)
        if out_proj.bias is not None:
            nn.init.zeros_(out_proj.bias)

    def clear_state(self):
        self.cached_tokens = None
        self.cached_history = None
        self.state_batch = None

    def reset_state(self, batch_size):
        self.cached_tokens = None
        self.cached_history = None
        self.state_batch = int(batch_size)

    def _ensure_state_batch(self, batch_size):
        if self.state_batch != int(batch_size):
            self.reset_state(batch_size)

    def get_history(self):
        return self.cached_history

    def encode_sequence(self, history_tokens):
        if history_tokens.dim() != 3:
            raise ValueError(
                f"history_tokens must be [B, T, C], got {tuple(history_tokens.shape)}"
            )
        if history_tokens.size(-1) != self.feat_dim:
            raise ValueError(
                f"history_tokens channel mismatch, expected {self.feat_dim}, "
                f"got {history_tokens.size(-1)}"
            )
        if self.use_mamba:
            fused = history_tokens + self.mamba(self.norm(history_tokens))
            return fused[:, -1, :]
        return history_tokens.mean(dim=1)

    @torch.no_grad()
    def update(self, target_feat):
        if target_feat.dim() != 2:
            raise ValueError(
                f"target_feat must be [B, C], got {tuple(target_feat.shape)}"
            )
        if target_feat.size(-1) != self.feat_dim:
            raise ValueError(
                f"target_feat channel mismatch, expected {self.feat_dim}, "
                f"got {target_feat.size(-1)}"
            )

        bsz = target_feat.size(0)
        self._ensure_state_batch(bsz)
        token = target_feat.detach().unsqueeze(1)

        if self.cached_tokens is None:
            seq = token
        else:
            seq = torch.cat([self.cached_tokens, token], dim=1)

        if seq.size(1) > self.history_len:
            seq = seq[:, -self.history_len:, :]

        self.cached_tokens = seq.detach()
        self.cached_history = self.encode_sequence(seq).detach()
        return self.cached_history


def topk_peaks_nms(score_map, topk=8, kernel_size=5):
    """
    Extract Top-K local peaks from score_map.

    Important fix: Non-peak positions are filled with -1e9, not 0.
    Otherwise, if score_map contains negative or very small values,
    non-peaks may be incorrectly selected by topk().
    """
    if score_map.dim() != 4 or score_map.size(1) != 1:
        raise ValueError(
            f"score_map must be [B, 1, H, W], got {tuple(score_map.shape)}"
        )

    bsz, _, h, w = score_map.shape
    k = min(int(topk), h * w)
    pad = int(kernel_size) // 2

    local_max = F.max_pool2d(
        score_map,
        kernel_size=kernel_size,
        stride=1,
        padding=pad,
    )
    peak_mask = score_map.eq(local_max)
    peak_map = score_map.masked_fill(~peak_mask, -1e9)

    peak_scores, peak_indices = torch.topk(
        peak_map.flatten(1),
        k=k,
        dim=1,
        largest=True,
        sorted=True,
    )
    peak_y = torch.div(peak_indices, w, rounding_mode="floor")
    peak_x = peak_indices % w
    peaks_xy = torch.stack([peak_x, peak_y], dim=-1)
    return peaks_xy, peak_scores


def sample_feature_at_peaks(feat_map, peaks_xy=None, peak_x=None, peak_y=None):
    if feat_map.dim() != 4:
        raise ValueError(
            f"feat_map must be [B, C, H, W], got {tuple(feat_map.shape)}"
        )

    if peaks_xy is not None:
        peak_x = peaks_xy[..., 0]
        peak_y = peaks_xy[..., 1]

    bsz, channels, height, width = feat_map.shape
    peak_x = peak_x.to(device=feat_map.device, dtype=feat_map.dtype)
    peak_y = peak_y.to(device=feat_map.device, dtype=feat_map.dtype)

    if width > 1:
        norm_x = (peak_x / (width - 1.0)) * 2.0 - 1.0
    else:
        norm_x = torch.zeros_like(peak_x)

    if height > 1:
        norm_y = (peak_y / (height - 1.0)) * 2.0 - 1.0
    else:
        norm_y = torch.zeros_like(peak_y)

    grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(1)
    sampled = F.grid_sample(
        feat_map,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.squeeze(2).transpose(1, 2).contiguous()


class CandidateReliabilityGate(nn.Module):
    def __init__(self, feat_dim=512, hidden_dim=128):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.hidden_dim = int(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.feat_dim + 2, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, topk_feats, topk_scores, rerank_prob):
        if topk_feats.dim() != 3:
            raise ValueError(f"topk_feats must be [B, K, C], got {tuple(topk_feats.shape)}")
        if topk_scores.dim() != 2:
            raise ValueError(f"topk_scores must be [B, K], got {tuple(topk_scores.shape)}")
        if rerank_prob.dim() != 2:
            raise ValueError(f"rerank_prob must be [B, K], got {tuple(rerank_prob.shape)}")
        if topk_feats.size(-1) != self.feat_dim:
            raise ValueError(
                f"topk_feats channel mismatch, expected {self.feat_dim}, got {topk_feats.size(-1)}"
            )
        if topk_scores.shape != rerank_prob.shape or topk_scores.shape != topk_feats.shape[:2]:
            raise ValueError(
                "topk_scores, rerank_prob, and topk_feats must agree on [B, K], "
                f"got scores={tuple(topk_scores.shape)}, rerank={tuple(rerank_prob.shape)}, "
                f"feats={tuple(topk_feats.shape[:2])}"
            )

        feats = torch.nan_to_num(topk_feats, nan=0.0, posinf=20.0, neginf=-20.0)
        scores = torch.nan_to_num(
            topk_scores.to(device=topk_feats.device, dtype=topk_feats.dtype),
            nan=0.0,
            posinf=20.0,
            neginf=-20.0,
        ).clamp(-20.0, 20.0)
        probs = torch.nan_to_num(
            rerank_prob.to(device=topk_feats.device, dtype=topk_feats.dtype),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)

        rel_input = torch.cat([feats, scores.unsqueeze(-1), probs.unsqueeze(-1)], dim=-1)
        rel_logits = self.mlp(rel_input).squeeze(-1)
        rel_logits = torch.nan_to_num(rel_logits, nan=0.0, posinf=20.0, neginf=-20.0)
        rel_prob = torch.sigmoid(rel_logits).clamp(0.0, 1.0)
        return rel_logits, rel_prob


class PostDecoderDisambiguator(nn.Module):
    def __init__(
        self,
        feat_dim=512,
        template_feat_dim=None,
        ratio_thresh=0.8,
        topk_peaks=8,
        nms_kernel_size=5,
        multi_peak_ratio_thresh=0.45,
        gaussian_sigma=2.0,
        suppression_strength=0.6,
        history_len=32,
        use_mamba_history=True,
        mamba_d_state=16,
        mamba_expand=2,
        use_template_anchor=False,
        use_first_frame_anchor=False,
        use_mamba_history_bank=True,
        template_anchor_weight=0.35,
        first_frame_anchor_weight=0.40,
        mamba_history_weight=0.25,
        use_history_aware_rerank_score=False,
        history_rerank_weight=1.0,
        target_logit_weight=1.0,
        peak_score_weight=0.2,
        update_ratio_thresh=0.90,
        target_prob_thresh=0.50,
        min_id_margin=0.00,
        # Inference-only heuristic reliability gate.
        # It adds no trainable parameters and does not require retraining.
        use_heuristic_reliability=True,
        reliability_update_thresh=0.55,
        rel_target_weight=0.25,
        rel_score_weight=0.20,
        rel_anchor_weight=0.25,
        rel_history_weight=0.15,
        rel_margin_weight=0.10,
        rel_ambiguity_weight=0.05,
        use_trainable_reliability=True,
        reliability_hidden_dim=128,
        eps=1e-6,
    ):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.template_feat_dim = (
            int(template_feat_dim) if template_feat_dim is not None else self.feat_dim
        )
        self.ratio_thresh = float(ratio_thresh)
        self.topk_peaks = int(topk_peaks)
        self.nms_kernel_size = int(nms_kernel_size)
        self.multi_peak_ratio_thresh = float(multi_peak_ratio_thresh)
        self.gaussian_sigma = float(gaussian_sigma)
        self.suppression_strength = float(suppression_strength)

        self.use_template_anchor = bool(use_template_anchor)
        self.use_first_frame_anchor = bool(use_first_frame_anchor)
        self.use_mamba_history_bank = bool(use_mamba_history_bank)
        self.template_anchor_weight = float(template_anchor_weight)
        self.first_frame_anchor_weight = float(first_frame_anchor_weight)
        self.mamba_history_weight = float(mamba_history_weight)
        self.use_history_aware_rerank_score = bool(use_history_aware_rerank_score)

        self.update_ratio_thresh = float(update_ratio_thresh)
        self.target_prob_thresh = float(target_prob_thresh)
        self.min_id_margin = float(min_id_margin)

        self.use_heuristic_reliability = bool(use_heuristic_reliability)
        self.reliability_update_thresh = float(reliability_update_thresh)
        self.rel_target_weight = float(rel_target_weight)
        self.rel_score_weight = float(rel_score_weight)
        self.rel_anchor_weight = float(rel_anchor_weight)
        self.rel_history_weight = float(rel_history_weight)
        self.rel_margin_weight = float(rel_margin_weight)
        self.rel_ambiguity_weight = float(rel_ambiguity_weight)
        self.use_trainable_reliability = bool(use_trainable_reliability)
        self.reliability_hidden_dim = int(reliability_hidden_dim)

        self.eps = float(eps)
        self.template_anchor = None
        self.first_frame_anchor = None

        self.history_bank = MambaHistoryBank(
            feat_dim=self.feat_dim,
            history_len=history_len,
            use_mamba=use_mamba_history,
            d_state=mamba_d_state,
            expand=mamba_expand,
        )

        if self.template_feat_dim != self.feat_dim:
            self.template_proj = nn.Sequential(
                nn.LayerNorm(self.template_feat_dim),
                nn.Linear(self.template_feat_dim, self.feat_dim),
            )
        else:
            self.template_proj = nn.Identity()

        # Keep this unchanged so that old checkpoints can still be loaded.
        self.reranker_mlp = nn.Sequential(
            nn.LayerNorm(7),
            nn.Linear(7, 32),
            nn.GELU(),
            nn.LayerNorm(32),
            nn.Linear(32, 1),
        )
        if self.use_trainable_reliability:
            self.reliability_gate = CandidateReliabilityGate(
                feat_dim=self.feat_dim,
                hidden_dim=self.reliability_hidden_dim,
            )
        else:
            self.reliability_gate = None

    def clear_state(self):
        self.template_anchor = None
        self.first_frame_anchor = None
        self.history_bank.clear_state()

    def reset_state(self, batch_size):
        self.template_anchor = None
        self.first_frame_anchor = None
        self.history_bank.reset_state(batch_size=batch_size)

    def reset_history(self, batch_size=1):
        self.history_bank.reset_state(batch_size=batch_size)

    @torch.no_grad()
    def update_history(self, target_feat):
        if not self.use_mamba_history_bank:
            return None
        if target_feat.dim() == 3:
            target_feat = target_feat[:, 0, :]
        return self.history_bank.update(target_feat.detach())

    @torch.no_grad()
    def initialize_memory(
        self,
        template_anchor=None,
        first_frame_anchor=None,
        batch_size=None,
        reset_dynamic=True,
    ):
        if reset_dynamic:
            self.history_bank.reset_state(batch_size or 1)
        if template_anchor is not None and self.use_template_anchor:
            self.set_template_anchor(template_anchor)
        if first_frame_anchor is not None and self.use_first_frame_anchor:
            self.set_first_frame_anchor(first_frame_anchor)

    @torch.no_grad()
    def set_template_anchor(self, target_feat):
        if target_feat is None:
            return
        target_feat = self._anchor_tensor(target_feat)
        self.template_anchor = target_feat.detach()

    @torch.no_grad()
    def set_first_frame_anchor(self, target_feat):
        if target_feat is None:
            return
        self.first_frame_anchor = self._anchor_tensor(target_feat).detach()

    def _anchor_tensor(self, anchor):
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        anchor = anchor.to(device=device, dtype=dtype)
        if anchor.dim() == 3:
            anchor = anchor.mean(dim=1)
        if anchor.size(-1) != self.feat_dim:
            anchor = self.template_proj(anchor)
        return anchor

    def _validate_anchor_tensor(self, anchor, batch_size, feat_dim, device, dtype):
        if anchor is None:
            return None
        anchor = anchor.to(device=device, dtype=dtype)
        if anchor.dim() == 3:
            anchor = anchor[:, -1, :]
        if anchor.size(-1) != feat_dim:
            anchor = self.template_proj(anchor)
        if anchor.size(0) == 1 and batch_size > 1:
            anchor = anchor.expand(batch_size, feat_dim)
        return anchor

    def _cosine_peaks_to_anchor(self, peak_feats, anchor):
        if anchor is None:
            return torch.zeros(
                peak_feats.size(0),
                peak_feats.size(1),
                device=peak_feats.device,
                dtype=peak_feats.dtype,
            )
        peak_norm = F.normalize(peak_feats, p=2, dim=-1, eps=self.eps)
        anchor_norm = F.normalize(anchor, p=2, dim=-1, eps=self.eps)
        return (peak_norm * anchor_norm[:, None, :]).sum(dim=-1)

    def _history_for_topk(self, topk_feats, history_tokens=None):
        bsz, _, feat_dim = topk_feats.shape
        template_anchor = None
        first_frame_anchor = None
        dynamic_history = None

        if history_tokens is not None:
            if history_tokens.size(-1) != self.feat_dim:
                history_tokens = self.template_proj(history_tokens)
            if self.use_template_anchor:
                template_anchor = history_tokens.mean(dim=1)
            if self.use_mamba_history_bank:
                dynamic_history = self.history_bank.encode_sequence(history_tokens)
        elif self.use_mamba_history_bank:
            dynamic_history = self.history_bank.get_history()

        if self.use_template_anchor:
            template_anchor = self._validate_anchor_tensor(
                self.template_anchor if self.template_anchor is not None else template_anchor,
                bsz,
                feat_dim,
                topk_feats.device,
                topk_feats.dtype,
            )
        if self.use_first_frame_anchor:
            first_frame_anchor = self._validate_anchor_tensor(
                self.first_frame_anchor,
                bsz,
                feat_dim,
                topk_feats.device,
                topk_feats.dtype,
            )
        dynamic_history = self._validate_anchor_tensor(
            dynamic_history,
            bsz,
            feat_dim,
            topk_feats.device,
            topk_feats.dtype,
        )
        return template_anchor, first_frame_anchor, dynamic_history

    def _prepare_robust_scores(self, raw_scores):
        scores = torch.nan_to_num(
            raw_scores,
            nan=0.0,
            posinf=20.0,
            neginf=-20.0,
        ).clamp(-20.0, 20.0)
        score_mean = scores.mean(dim=-1, keepdim=True)
        score_std = scores.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-4)
        score_z = ((scores - score_mean) / score_std).clamp(-5.0, 5.0)
        score_prob = torch.softmax(score_z, dim=-1)

        top1_prob = score_prob[:, 0].clamp_min(1e-4)
        if score_prob.size(1) > 1:
            top2_prob = score_prob[:, 1]
        else:
            top2_prob = torch.zeros_like(top1_prob)

        score_ratio = (score_prob / top1_prob[:, None]).clamp(0.0, 5.0)
        ambiguity_ratio = (top2_prob / top1_prob).clamp(0.0, 5.0)
        score_gap = (score_z[:, :1] - score_z).clamp(-5.0, 5.0)
        return score_z, score_gap, score_prob, score_ratio, ambiguity_ratio

    @staticmethod
    def _gather_selected_1d(values, selected_idx):
        """
        values: [B, K]
        selected_idx: [B]
        return: [B]
        """
        if values is None:
            return None
        return values.gather(1, selected_idx.view(-1, 1)).squeeze(1)

    @staticmethod
    def _normalize_cosine_to_conf(sim):
        """Convert cosine similarity from [-1, 1] to [0, 1]."""
        if sim is None:
            return None
        return ((sim + 1.0) * 0.5).clamp(0.0, 1.0)

    def _compute_heuristic_reliability(
        self,
        selected_idx,
        selected_prob,
        identity_margin,
        ambiguity_ratio,
        score_prob,
        sim_first=None,
        sim_dynamic=None,
        sim_history=None,
    ):
        """
        Inference-only reliability gate for MemoryBank update.

        It estimates whether the selected candidate is safe enough to be written
        into the dynamic history bank. This does not change the selected bbox.
        """
        selected_score_prob = self._gather_selected_1d(score_prob, selected_idx)
        if selected_score_prob is None:
            selected_score_prob = torch.zeros_like(selected_prob)

        selected_sim_first = self._gather_selected_1d(sim_first, selected_idx)
        selected_sim_dynamic = self._gather_selected_1d(sim_dynamic, selected_idx)
        selected_sim_history = self._gather_selected_1d(sim_history, selected_idx)

        anchor_conf = self._normalize_cosine_to_conf(selected_sim_first)
        dynamic_conf = self._normalize_cosine_to_conf(selected_sim_dynamic)
        history_conf = self._normalize_cosine_to_conf(selected_sim_history)

        if anchor_conf is None:
            anchor_conf = history_conf if history_conf is not None else torch.zeros_like(selected_prob)
        if dynamic_conf is None:
            dynamic_conf = history_conf if history_conf is not None else torch.zeros_like(selected_prob)

        margin_conf = identity_margin.clamp(0.0, 1.0)
        ambiguity_conf = (1.0 - ambiguity_ratio).clamp(0.0, 1.0)

        weights = {
            "target": self.rel_target_weight,
            "score": self.rel_score_weight,
            "anchor": self.rel_anchor_weight,
            "history": self.rel_history_weight,
            "margin": self.rel_margin_weight,
            "ambiguity": self.rel_ambiguity_weight,
        }
        weight_sum = max(float(sum(weights.values())), self.eps)

        reliability = (
            weights["target"] * selected_prob.clamp(0.0, 1.0)
            + weights["score"] * selected_score_prob.clamp(0.0, 1.0)
            + weights["anchor"] * anchor_conf
            + weights["history"] * dynamic_conf
            + weights["margin"] * margin_conf
            + weights["ambiguity"] * ambiguity_conf
        ) / weight_sum

        reliability = torch.nan_to_num(
            reliability,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)

        aux = {
            "reliability_score": reliability.detach(),
            "selected_score_prob": selected_score_prob.detach(),
            "selected_sim_first": (
                selected_sim_first.detach()
                if selected_sim_first is not None
                else torch.zeros_like(selected_prob).detach()
            ),
            "selected_sim_dynamic": (
                selected_sim_dynamic.detach()
                if selected_sim_dynamic is not None
                else torch.zeros_like(selected_prob).detach()
            ),
            "selected_sim_history": (
                selected_sim_history.detach()
                if selected_sim_history is not None
                else torch.zeros_like(selected_prob).detach()
            ),
            "ambiguity_conf": ambiguity_conf.detach(),
        }
        return reliability, aux

    def forward_topk(self, topk_feats, topk_scores, history_tokens=None, return_aux=False):
        bsz, num_peaks, _ = topk_feats.shape
        topk_scores = topk_scores.to(device=topk_feats.device, dtype=topk_feats.dtype)

        score_z, score_gap, score_prob, score_ratio, ambiguity_ratio = (
            self._prepare_robust_scores(topk_scores)
        )

        template_anchor, first_frame_anchor, dynamic_history = self._history_for_topk(
            topk_feats,
            history_tokens=history_tokens,
        )

        sim_template = self._cosine_peaks_to_anchor(topk_feats, template_anchor)
        sim_first = self._cosine_peaks_to_anchor(topk_feats, first_frame_anchor)
        sim_dynamic = self._cosine_peaks_to_anchor(topk_feats, dynamic_history)

        # Keep reranker input dimension unchanged: sim_history is still one channel.
        # First-frame anchor is given higher weight than dynamic history.
        active_sims = []
        active_weights = []
        if first_frame_anchor is not None:
            active_sims.append(sim_first)
            active_weights.append(self.first_frame_anchor_weight)
        if dynamic_history is not None:
            active_sims.append(sim_dynamic)
            active_weights.append(self.mamba_history_weight)

        if len(active_sims) == 0:
            sim_history = sim_dynamic
        else:
            weight_sum = max(sum(active_weights), self.eps)
            sim_history = sum(w * s for w, s in zip(active_weights, active_sims)) / weight_sum

        rank = (
            torch.arange(
                num_peaks,
                device=topk_feats.device,
                dtype=topk_feats.dtype,
            )
            / float(max(1, num_peaks - 1))
        ).view(1, num_peaks).expand(bsz, num_peaks)

        rerank_features = torch.stack(
            [
                score_z,
                score_gap,
                score_ratio,
                sim_template,
                sim_history,
                rank,
                ambiguity_ratio[:, None].expand(bsz, num_peaks),
            ],
            dim=-1,
        )
        rerank_features = torch.nan_to_num(
            rerank_features,
            nan=0.0,
            posinf=5.0,
            neginf=-5.0,
        ).clamp(-5.0, 5.0)

        target_logits = self.reranker_mlp(rerank_features).squeeze(-1)
        if not return_aux:
            return target_logits

        rerank_prob = torch.softmax(target_logits.detach(), dim=-1)
        rel_logits = None
        rel_prob = None
        if self.reliability_gate is not None:
            rel_logits, rel_prob = self.reliability_gate(
                topk_feats=topk_feats,
                topk_scores=topk_scores,
                rerank_prob=rerank_prob,
            )

        return {
            "logits": target_logits,
            "rerank_prob": rerank_prob,
            "rel_logits": rel_logits,
            "rel_prob": rel_prob,
            "target_logits": target_logits,
            "rerank_features": rerank_features,
            "score_prob": score_prob,
            "ambiguity_ratio": ambiguity_ratio,
            "sim_template": sim_template.detach(),
            "sim_first": sim_first.detach(),
            "sim_dynamic": sim_dynamic.detach(),
            "sim_history": sim_history.detach(),
        }

    def _build_sparse_rerank_score_map(
        self,
        score_map,
        peaks_xy,
        target_probs,
        height,
        width,
    ):
        """
        Build sparse Top-K score map.

        This map is used only when selected_idx != 0. If selected_idx == 0,
        original score_map is returned unchanged.
        """
        refined_score_map = score_map.new_zeros(score_map.shape)
        candidate_scores = target_probs.clamp(0.0, 1.0)
        flat_map = refined_score_map.flatten(2)
        flat_idx = (peaks_xy[..., 1] * width + peaks_xy[..., 0]).long().unsqueeze(1)
        flat_map.scatter_(2, flat_idx, candidate_scores.unsqueeze(1))
        return refined_score_map

    def forward(self, score_map, feat_map, history_tokens=None, prev_center=None):
        """
        Post-decoder disambiguation.

        Critical inference rule: If selected_idx == 0, return the original
        score_map unchanged. Only when selected_idx != 0 do we replace the
        score_map. This prevents STARTrack from changing thousands of normal
        frames when the reranker still selects the baseline Top-1 candidate.
        """
        if score_map.dim() != 4 or score_map.size(1) != 1:
            raise ValueError(
                f"score_map must be [B, 1, H, W], got {tuple(score_map.shape)}"
            )
        if feat_map.dim() != 4:
            raise ValueError(f"feat_map must be [B, C, H, W], got {tuple(feat_map.shape)}")

        bsz, _, height, width = feat_map.shape
        peaks_xy, peak_scores = topk_peaks_nms(
            score_map,
            topk=self.topk_peaks,
            kernel_size=self.nms_kernel_size,
        )
        peak_feats = sample_feature_at_peaks(feat_map, peaks_xy=peaks_xy)

        topk_aux = self.forward_topk(
            peak_feats,
            peak_scores,
            history_tokens=history_tokens,
            return_aux=True,
        )
        target_logits = topk_aux["logits"]
        target_probs = torch.softmax(target_logits, dim=-1)
        selected_prob, selected_idx = target_probs.max(dim=-1)

        sorted_probs, _ = torch.sort(target_probs, dim=-1, descending=True)
        if sorted_probs.size(1) > 1:
            second_prob = sorted_probs[:, 1]
        else:
            second_prob = torch.zeros_like(selected_prob)
        identity_margin = selected_prob - second_prob

        batch_idx = torch.arange(bsz, device=score_map.device)
        selected_xy = peaks_xy[batch_idx, selected_idx]
        target_feat = peak_feats[batch_idx, selected_idx, :].detach()

        score_z, score_gap, score_prob, score_ratio, ambiguity_ratio = (
            self._prepare_robust_scores(peak_scores)
        )
        effective_peak_count = (score_ratio > self.multi_peak_ratio_thresh).sum(dim=-1)
        rerank_used = selected_idx != 0

        base_should_update = (
            (selected_prob >= self.target_prob_thresh)
            & (identity_margin >= self.min_id_margin)
            & (ambiguity_ratio <= self.update_ratio_thresh)
            & (selected_idx >= 0)
        )

        rel_prob_all = topk_aux.get("rel_prob", None)
        rel_logits_all = topk_aux.get("rel_logits", None)
        selected_rel_prob = torch.zeros_like(selected_prob)
        trainable_reliability_used = bool(
            self.use_trainable_reliability and rel_prob_all is not None
        )

        if trainable_reliability_used:
            selected_rel_prob = rel_prob_all.gather(
                1, selected_idx.view(-1, 1)
            ).squeeze(1)
            reliability_score = selected_rel_prob
            selected_score_prob = self._gather_selected_1d(score_prob, selected_idx)
            if selected_score_prob is None:
                selected_score_prob = torch.zeros_like(selected_prob)
            reliability_aux = {
                "reliability_score": reliability_score.detach(),
                "selected_score_prob": selected_score_prob.detach(),
                "selected_sim_first": torch.zeros_like(selected_prob).detach(),
                "selected_sim_dynamic": torch.zeros_like(selected_prob).detach(),
                "selected_sim_history": torch.zeros_like(selected_prob).detach(),
                "ambiguity_conf": (1.0 - ambiguity_ratio).clamp(0.0, 1.0).detach(),
            }
            should_update = base_should_update & (
                selected_rel_prob >= self.reliability_update_thresh
            )
        elif self.use_heuristic_reliability:
            reliability_score, reliability_aux = self._compute_heuristic_reliability(
                selected_idx=selected_idx,
                selected_prob=selected_prob,
                identity_margin=identity_margin,
                ambiguity_ratio=ambiguity_ratio,
                score_prob=score_prob,
                sim_first=topk_aux.get("sim_first", None),
                sim_dynamic=topk_aux.get("sim_dynamic", None),
                sim_history=topk_aux.get("sim_history", None),
            )
            should_update = base_should_update & (
                reliability_score >= self.reliability_update_thresh
            )
        else:
            reliability_score = selected_prob.detach()
            selected_rel_prob = reliability_score
            selected_score_prob = self._gather_selected_1d(score_prob, selected_idx)
            if selected_score_prob is None:
                selected_score_prob = torch.zeros_like(selected_prob)
            reliability_aux = {
                "reliability_score": reliability_score.detach(),
                "selected_score_prob": selected_score_prob.detach(),
                "selected_sim_first": torch.zeros_like(selected_prob).detach(),
                "selected_sim_dynamic": torch.zeros_like(selected_prob).detach(),
                "selected_sim_history": torch.zeros_like(selected_prob).detach(),
                "ambiguity_conf": (1.0 - ambiguity_ratio).clamp(0.0, 1.0).detach(),
            }
            should_update = base_should_update

        reranked_score_map = self._build_sparse_rerank_score_map(
            score_map=score_map,
            peaks_xy=peaks_xy,
            target_probs=target_probs,
            height=height,
            width=width,
        )

        # Conservative gate:
        # selected_idx == 0 -> return original SUTrack score_map.
        # selected_idx != 0 -> use sparse reranked score_map.
        if bsz == 1:
            if bool(rerank_used.detach().reshape(-1)[0].item()):
                refined_score_map = reranked_score_map
            else:
                refined_score_map = score_map
        else:
            mask = rerank_used.view(bsz, 1, 1, 1).to(
                device=score_map.device,
                dtype=score_map.dtype,
            )
            refined_score_map = mask * reranked_score_map + (1.0 - mask) * score_map

        aux_info = {
            "target_feat": target_feat,
            "selected_idx": selected_idx.detach(),
            "rerank_idx": selected_idx.detach(),
            "target_prob": selected_prob.detach(),
            "identity_margin": identity_margin.detach(),
            "ambiguity_ratio": ambiguity_ratio.detach(),
            "should_update": should_update.detach(),
            "base_should_update": base_should_update.detach(),
            "rerank_used": rerank_used.detach(),
            "peaks_xy": peaks_xy.detach(),
            "selected_xy": selected_xy.detach(),
            "topk_scores": peak_scores.detach(),
            "target_logits": target_logits,
            "target_probs": target_probs.detach(),
            "score_prob": score_prob.detach(),
            "selected_rel_prob": selected_rel_prob.detach(),
            "rel_prob_all": (
                rel_prob_all.detach()
                if rel_prob_all is not None
                else torch.zeros_like(target_probs).detach()
            ),
            "rel_logits_all": (
                rel_logits_all.detach()
                if rel_logits_all is not None
                else torch.zeros_like(target_probs).detach()
            ),
            "trainable_reliability_used": torch.full(
                (bsz,),
                bool(trainable_reliability_used),
                device=score_map.device,
                dtype=torch.bool,
            ),
            "effective_peak_count": effective_peak_count.detach(),
            "is_ambiguous": (
                (ambiguity_ratio >= self.ratio_thresh) | (effective_peak_count >= 3)
            ).detach(),
            "reliability_score": reliability_score.detach(),
        }
        aux_info.update(topk_aux)
        aux_info.update(reliability_aux)
        return refined_score_map, aux_info
