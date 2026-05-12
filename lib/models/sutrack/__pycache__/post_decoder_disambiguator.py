import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm.modules.mamba_simple import Mamba
except ImportError:
    Mamba = None


class MambaHistoryBank(nn.Module):
    """
    Lightweight online history bank for target features.

    Inputs are target tokens of shape [B, C]. The bank keeps a short sequence
    and optionally fuses it with a tiny Mamba block to produce one compact
    history embedding [B, C] used for post-decoder matching.
    """

    def __init__(
        self,
        feat_dim,
        history_len=32,
        use_mamba=True,
        d_state=16,
        expand=2,
    ):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.history_len = int(history_len)
        self.use_mamba = bool(use_mamba)

        if self.use_mamba:
            if Mamba is None:
                raise ImportError("mamba_ssm is required when MambaHistoryBank.use_mamba=True.")
            self.norm = nn.LayerNorm(self.feat_dim)
            self.mamba = Mamba(d_model=self.feat_dim, d_state=d_state, expand=expand)
            self._zero_init_output_projection()
        else:
            self.norm = None
            self.mamba = None

        self.cached_tokens = None  # [B, T, C]
        self.cached_history = None  # [B, C]
        self.state_batch = None

    def _zero_init_output_projection(self):
        # Keep behavior close to baseline at initialization.
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
        if self.state_batch is None:
            self.state_batch = int(batch_size)
            return
        if self.state_batch != int(batch_size):
            self.clear_state()
            self.state_batch = int(batch_size)

    def get_history(self):
        return self.cached_history

    def encode_sequence(self, history_tokens):
        """
        Differentiable history encoder.

        history_tokens: [B, T, C]
        returns: history embedding [B, C]
        """
        if history_tokens.dim() != 3:
            raise ValueError(f"history_tokens must be [B, T, C], got {tuple(history_tokens.shape)}")
        if history_tokens.size(-1) != self.feat_dim:
            raise ValueError(
                f"history_tokens channel mismatch, expected C={self.feat_dim}, got C={history_tokens.size(-1)}"
            )

        if self.use_mamba:
            fused = history_tokens + self.mamba(self.norm(history_tokens))
            history = fused[:, -1, :]
        else:
            history = history_tokens.mean(dim=1)
        return history

    @torch.no_grad()
    def update(self, target_feat):
        """
        Update bank with the confirmed target feature.

        target_feat: [B, C]
        returns: compact history feature [B, C]
        """
        if target_feat.dim() != 2:
            raise ValueError(f"target_feat must be [B, C], got {tuple(target_feat.shape)}")
        if target_feat.size(-1) != self.feat_dim:
            raise ValueError(
                f"target_feat channel mismatch, expected C={self.feat_dim}, got C={target_feat.size(-1)}"
            )

        bsz = target_feat.size(0)
        self._ensure_state_batch(bsz)

        token = target_feat.detach().unsqueeze(1)  # [B, 1, C]
        if self.cached_tokens is None:
            seq = token
        else:
            seq = torch.cat([self.cached_tokens, token], dim=1)
            if seq.size(1) > self.history_len:
                seq = seq[:, -self.history_len :, :]

        if self.use_mamba:
            fused = seq + self.mamba(self.norm(seq))
            history = fused[:, -1, :]
        else:
            history = seq.mean(dim=1)

        self.cached_tokens = seq.detach()
        self.cached_history = history.detach()
        return self.cached_history


def topk_peaks_nms(score_map, topk=5, kernel_size=5):
    """
    Peak extractor by local-max NMS + top-k.

    score_map: [B, 1, H, W]
    returns:
      peaks_xy: [B, K, 2], (x, y) feature-map coordinates
      peak_scores: [B, K]
    """
    if score_map.dim() != 4 or score_map.size(1) != 1:
        raise ValueError(f"score_map must be [B, 1, H, W], got {tuple(score_map.shape)}")

    bsz, _, h, w = score_map.shape
    k = min(int(topk), h * w)
    if k < 1:
        raise ValueError("topk must be >= 1")

    pad = kernel_size // 2
    local_max = F.max_pool2d(score_map, kernel_size=kernel_size, stride=1, padding=pad)
    peak_mask = score_map.eq(local_max)
    nms_map = torch.where(peak_mask, score_map, torch.zeros_like(score_map))

    peak_scores, peak_indices = torch.topk(nms_map.flatten(1), k=k, dim=1, largest=True, sorted=True)
    peak_y = torch.div(peak_indices, w, rounding_mode="floor")
    peak_x = peak_indices % w
    peaks_xy = torch.stack([peak_x, peak_y], dim=-1)
    return peaks_xy, peak_scores


def sample_feature_at_peaks(feat_map, peaks_xy=None, peak_x=None, peak_y=None):
    """
    Extract [B, K, C] vectors from feature map [B, C, H, W] via grid_sample.

    IMPORTANT:
    - We use coordinate mapping:
        x_norm = (x / (W - 1)) * 2 - 1
        y_norm = (y / (H - 1)) * 2 - 1
      which corresponds to align_corners=True.
    - Therefore grid_sample must explicitly set align_corners=True to avoid
      half-pixel misalignment.
    """
    if feat_map.dim() != 4:
        raise ValueError(f"feat_map must be [B, C, H, W], got {tuple(feat_map.shape)}")
    if peaks_xy is not None:
        if peaks_xy.dim() != 3 or peaks_xy.size(-1) != 2:
            raise ValueError(f"peaks_xy must be [B, K, 2], got {tuple(peaks_xy.shape)}")
        peak_x = peaks_xy[..., 0]
        peak_y = peaks_xy[..., 1]
    if peak_x is None or peak_y is None or peak_x.dim() != 2 or peak_y.dim() != 2:
        raise ValueError("peak_x and peak_y must be [B, K]")

    bsz, channels, height, width = feat_map.shape
    if peak_x.size(0) != bsz or peak_y.size(0) != bsz:
        raise ValueError("batch size of peak_x/peak_y and feat_map does not match.")
    if peak_x.shape != peak_y.shape:
        raise ValueError("peak_x and peak_y must have the same shape [B, K].")

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

    # grid: [B, 1, K, 2], where last dim is (x, y)
    grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(1)

    # [B, C, H, W] & [B, 1, K, 2] -> [B, C, 1, K]
    # Explicit align_corners=True is required for the mapping above.
    sampled = F.grid_sample(
        feat_map,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.squeeze(2).transpose(1, 2).contiguous()  # [B, K, C]


class PostDecoderDisambiguator(nn.Module):
    """
    Delayed & target-aware temporal disambiguation on score map stage.

    Pipeline:
      1) Top-K local peak extraction from the original score_map
      2) Read candidate peak features from F_map
      3) Match candidates with template/history features
      4) Rerank candidates without dense score-map reshaping
    """

    def __init__(
        self,
        feat_dim,
        template_feat_dim=None,
        ratio_thresh=0.8,
        topk_peaks=5,
        nms_kernel_size=5,
        multi_peak_ratio_thresh=0.45,
        gaussian_sigma=2.0,
        suppression_strength=0.6,
        history_len=32,
        use_mamba_history=True,
        mamba_d_state=16,
        mamba_expand=2,
        use_template_anchor=True,
        use_first_frame_anchor=True,
        use_mamba_history_bank=True,
        template_anchor_weight=0.35,
        first_frame_anchor_weight=0.40,
        mamba_history_weight=0.25,
        use_history_aware_rerank_score=True,
        history_rerank_weight=1.0,
        target_logit_weight=1.0,
        peak_score_weight=0.2,
        eps=1e-6,
    ):
        super().__init__()
        self.ratio_thresh = float(ratio_thresh)
        self.topk_peaks = int(topk_peaks)
        self.nms_kernel_size = int(nms_kernel_size)
        self.multi_peak_ratio_thresh = float(multi_peak_ratio_thresh)
        self.gaussian_sigma = float(gaussian_sigma)
        self.suppression_strength = float(suppression_strength)
        self.eps = float(eps)
        self.feat_dim = int(feat_dim)
        self.template_feat_dim = int(template_feat_dim) if template_feat_dim is not None else self.feat_dim
        self.use_template_anchor = bool(use_template_anchor)
        self.use_first_frame_anchor = bool(use_first_frame_anchor)
        self.use_mamba_history_bank = bool(use_mamba_history_bank)
        self.template_anchor_weight = float(template_anchor_weight)
        self.first_frame_anchor_weight = float(first_frame_anchor_weight)
        self.mamba_history_weight = float(mamba_history_weight)
        self.use_history_aware_rerank_score = bool(use_history_aware_rerank_score)
        self.history_rerank_weight = float(history_rerank_weight)
        self.target_logit_weight = float(target_logit_weight)
        self.peak_score_weight = float(peak_score_weight)
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
        # V4 Top-K reranker. Per-candidate features:
        # [score, score_ratio, sim_template, sim_history, dist_to_prev, rank, ambiguity_ratio]
        self.reranker_mlp = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

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
        return self.history_bank.update(target_feat)

    @torch.no_grad()
    def set_template_anchor(self, target_feat):
        if target_feat is None:
            return
        target_feat = target_feat.to(
            device=next(self.parameters()).device,
            dtype=next(self.parameters()).dtype,
        )
        if target_feat.size(-1) != self.feat_dim:
            if target_feat.size(-1) != self.template_feat_dim:
                raise ValueError(
                    f"template_anchor channel mismatch: expected C={self.feat_dim} "
                    f"or template C={self.template_feat_dim}, got C={target_feat.size(-1)}; "
                    f"target_feat_shape={tuple(target_feat.shape)}"
                )
            target_feat = self.template_proj(target_feat)
        if target_feat.dim() == 3:
            target_feat = target_feat.mean(dim=1)
        if target_feat.dim() != 2:
            raise ValueError(f"target_feat must be [B, C] or [B, 1, C], got {tuple(target_feat.shape)}")
        if target_feat.size(-1) != self.feat_dim:
            raise ValueError(
                f"target_feat channel mismatch, expected C={self.feat_dim}, got C={target_feat.size(-1)}"
            )
        self.template_anchor = target_feat.detach()

    def get_template_anchor(self):
        return self.template_anchor

    @torch.no_grad()
    def set_first_frame_anchor(self, target_feat):
        if target_feat is None:
            return
        self.first_frame_anchor = self._validate_anchor_tensor(
            target_feat,
            name="first_frame_anchor",
            batch_size=None,
            feat_dim=self.feat_dim,
            device=target_feat.device,
            dtype=target_feat.dtype,
        ).detach()

    def get_first_frame_anchor(self):
        return self.first_frame_anchor

    @torch.no_grad()
    def initialize_memory(self, template_anchor=None, first_frame_anchor=None, batch_size=None, reset_dynamic=True):
        """
        Initialize long-lived identity anchors and reset the online dynamic bank.

        template_anchor can come from template tokens/features and is projected
        to feat_dim when needed. first_frame_anchor should come from search-side
        decoder F_map so it is already in the same domain as peak_feats.
        """
        if reset_dynamic:
            if batch_size is None:
                self.history_bank.clear_state()
            else:
                self.history_bank.reset_state(batch_size=batch_size)
        if template_anchor is not None and self.use_template_anchor:
            self.set_template_anchor(template_anchor)
        if first_frame_anchor is not None and self.use_first_frame_anchor:
            self.set_first_frame_anchor(first_frame_anchor)

    def _validate_anchor_tensor(self, anchor, name, batch_size, feat_dim, device, dtype):
        if anchor is None:
            return None
        anchor = anchor.to(device=device, dtype=dtype)
        if anchor.dim() == 3:
            if anchor.size(1) != 1:
                raise ValueError(f"{name} must be [B, C] or [B, 1, C], got {tuple(anchor.shape)}")
            anchor = anchor[:, 0, :]
        if anchor.dim() != 2:
            raise ValueError(f"{name} must be [B, C] or [B, 1, C], got {tuple(anchor.shape)}")
        if anchor.size(-1) != feat_dim:
            raise ValueError(
                f"{name} channel mismatch: expected C={feat_dim}, got C={anchor.size(-1)}; "
                f"anchor_shape={tuple(anchor.shape)}"
            )
        if batch_size is not None:
            if anchor.size(0) == 1 and batch_size > 1:
                anchor = anchor.expand(batch_size, feat_dim)
            elif anchor.size(0) != batch_size:
                raise ValueError(
                    f"{name} batch mismatch: expected B={batch_size}, got B={anchor.size(0)}; "
                    f"anchor_shape={tuple(anchor.shape)}"
                )
        return anchor

    def _cosine_peaks_to_anchor(self, peak_feats, anchor, name):
        if anchor is None:
            return torch.zeros(
                peak_feats.size(0),
                peak_feats.size(1),
                device=peak_feats.device,
                dtype=peak_feats.dtype,
            )
        if peak_feats.size(-1) != anchor.size(-1):
            raise ValueError(
                f"{name} cosine dim mismatch: peak_feats={tuple(peak_feats.shape)}, "
                f"anchor={tuple(anchor.shape)}"
            )
        peak_norm = F.normalize(peak_feats, p=2, dim=-1, eps=self.eps)
        anchor_norm = F.normalize(anchor, p=2, dim=-1, eps=self.eps)
        return (peak_norm * anchor_norm[:, None, :]).sum(dim=-1)

    @staticmethod
    def _as_batch_center(center, batch_size, device, dtype):
        if center is None:
            return None
        if not isinstance(center, torch.Tensor):
            center = torch.tensor(center, device=device, dtype=dtype)
        else:
            center = center.to(device=device, dtype=dtype)
        center = center.reshape(-1, 2)
        if center.size(0) == 1 and batch_size > 1:
            center = center.expand(batch_size, 2)
        if center.size(0) != batch_size:
            return None
        return center

    def _build_inverted_gaussian_mask(self, height, width, center_x, center_y, device, dtype, suppress_strength=None):
        """
        Build multiplicative mask in [1-strength, 1], minimum at peak center.
        """
        yy, xx = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        xx = xx.unsqueeze(0)  # [1, H, W]
        yy = yy.unsqueeze(0)

        dx2 = (xx - center_x.view(-1, 1, 1).to(dtype)) ** 2
        dy2 = (yy - center_y.view(-1, 1, 1).to(dtype)) ** 2
        gauss = torch.exp(-(dx2 + dy2) / (2.0 * (self.gaussian_sigma ** 2)))
        strength = self.suppression_strength if suppress_strength is None else suppress_strength
        if not torch.is_tensor(strength):
            strength = torch.tensor(float(strength), device=device, dtype=dtype)
        else:
            strength = strength.to(device=device, dtype=dtype)
        if strength.dim() == 0:
            strength = strength.view(1, 1, 1)
        else:
            strength = strength.view(-1, 1, 1)
        mask = 1.0 - strength * gauss
        return mask.clamp(min=0.0, max=1.0).unsqueeze(1)  # [B, 1, H, W]

    def _normalize_scores(self, x):
        """
        Safe per-sample z-score normalization for [B, K].
        When K == 1 or std is tiny, return zeros_like(x).
        """
        if x.dim() != 2:
            raise ValueError(f"x must be [B, K], got {tuple(x.shape)}")
        if x.size(1) <= 1:
            return torch.zeros_like(x)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        safe_std = std.clamp(min=self.eps)
        normalized = (x - mean) / safe_std
        return torch.where(std > self.eps, normalized, torch.zeros_like(x))

    def _history_for_topk(self, topk_feats, history_tokens=None):
        bsz, _, feat_dim = topk_feats.shape
        template_anchor = None
        first_frame_anchor = None
        dynamic_history = None

        if history_tokens is not None:
            if history_tokens.dim() != 3:
                raise ValueError(f"history_tokens must be [B, T, C], got {tuple(history_tokens.shape)}")
            history_tokens = self.template_proj(history_tokens)
            if history_tokens.size(0) != bsz or history_tokens.size(-1) != feat_dim:
                raise ValueError(
                    f"history_tokens must align with topk_feats [B, K, C], got "
                    f"history={tuple(history_tokens.shape)}, topk_feats={tuple(topk_feats.shape)}"
                )
            if self.use_template_anchor:
                template_anchor = history_tokens.mean(dim=1)
            if self.use_mamba_history_bank:
                dynamic_history = self.history_bank.encode_sequence(history_tokens)
        elif self.use_mamba_history_bank:
            dynamic_history = self.history_bank.get_history()

        if self.use_template_anchor and self.template_anchor is not None:
            template_anchor = self._validate_anchor_tensor(
                self.template_anchor,
                name="template_anchor",
                batch_size=bsz,
                feat_dim=feat_dim,
                device=topk_feats.device,
                dtype=topk_feats.dtype,
            )
        if self.use_first_frame_anchor and self.first_frame_anchor is not None:
            first_frame_anchor = self._validate_anchor_tensor(
                self.first_frame_anchor,
                name="first_frame_anchor",
                batch_size=bsz,
                feat_dim=feat_dim,
                device=topk_feats.device,
                dtype=topk_feats.dtype,
            )
        if dynamic_history is not None:
            if dynamic_history.dim() == 3:
                dynamic_history = dynamic_history[:, -1, :]
            dynamic_history = self._validate_anchor_tensor(
                dynamic_history,
                name="mamba_history_embed",
                batch_size=bsz,
                feat_dim=feat_dim,
                device=topk_feats.device,
                dtype=topk_feats.dtype,
            )

        return template_anchor, first_frame_anchor, dynamic_history

    def forward_topk(self, topk_feats, topk_scores, history_tokens=None, return_aux=False):
        """
        Lightweight Top-K reranker API for offline trajectory training.

        topk_feats:  [B, K, C]
        topk_scores: [B, K]
        returns:
          target_logits [B, K], or (target_logits, aux) when return_aux=True.
        """
        if topk_feats.dim() != 3:
            raise ValueError(f"topk_feats must be [B, K, C], got {tuple(topk_feats.shape)}")
        if topk_scores.dim() != 2:
            raise ValueError(f"topk_scores must be [B, K], got {tuple(topk_scores.shape)}")
        if topk_feats.shape[:2] != topk_scores.shape:
            raise ValueError(
                f"topk_feats/topk_scores shape mismatch: "
                f"{tuple(topk_feats.shape)} vs {tuple(topk_scores.shape)}"
            )
        if topk_feats.size(-1) != self.feat_dim:
            raise ValueError(
                f"topk_feats channel mismatch: expected C={self.feat_dim}, got C={topk_feats.size(-1)}"
            )

        bsz, num_peaks, _ = topk_feats.shape
        topk_scores = topk_scores.to(device=topk_feats.device, dtype=topk_feats.dtype)
        top1 = topk_scores[:, 0].clamp_min(self.eps)
        score_ratio = topk_scores / (top1[:, None] + self.eps)
        top2 = topk_scores[:, 1] if num_peaks > 1 else torch.zeros_like(top1)
        ambiguity_ratio = top2 / (top1 + self.eps)

        template_anchor, first_frame_anchor, dynamic_history = self._history_for_topk(
            topk_feats,
            history_tokens=history_tokens,
        )
        sim_template_anchor = self._cosine_peaks_to_anchor(topk_feats, template_anchor, name="template_anchor")
        sim_first_frame_anchor = self._cosine_peaks_to_anchor(topk_feats, first_frame_anchor, name="first_frame_anchor")
        sim_mamba_history = self._cosine_peaks_to_anchor(topk_feats, dynamic_history, name="mamba_history_embed")

        if num_peaks > 1:
            rank = torch.arange(num_peaks, device=topk_feats.device, dtype=topk_feats.dtype) / float(num_peaks - 1)
        else:
            rank = torch.zeros(num_peaks, device=topk_feats.device, dtype=topk_feats.dtype)
        rank = rank.view(1, num_peaks).expand(bsz, num_peaks)
        dist_to_prev = torch.zeros_like(topk_scores)

        rerank_features = torch.stack(
            [
                topk_scores,
                score_ratio,
                sim_template_anchor,
                sim_mamba_history,
                dist_to_prev,
                rank,
                ambiguity_ratio[:, None].expand(bsz, num_peaks),
            ],
            dim=-1,
        )
        target_logits = self.reranker_mlp(rerank_features).squeeze(-1)
        if not return_aux:
            return target_logits
        return target_logits, {
            "target_logits": target_logits,
            "sim_template_anchor": sim_template_anchor,
            "sim_first_frame_anchor": sim_first_frame_anchor,
            "sim_mamba_history": sim_mamba_history,
            "rerank_features": rerank_features,
        }

    def forward(self, score_map, feat_map, history_tokens=None, prev_center=None):
        """
        score_map: [B, 1, H, W]
        feat_map:  [B, C, H, W]
        history_tokens: optional [B, T, C], differentiable training history
        prev_center: optional, reserved for caller-side online update policies

        returns:
          refined_score_map: [B, 1, H, W]
          aux_info: dict for integration/debug
        """
        if score_map.dim() != 4 or score_map.size(1) != 1:
            raise ValueError(f"score_map must be [B, 1, H, W], got {tuple(score_map.shape)}")
        if feat_map.dim() != 4:
            raise ValueError(f"feat_map must be [B, C, H, W], got {tuple(feat_map.shape)}")
        if score_map.size(0) != feat_map.size(0) or score_map.size(-2) != feat_map.size(-2) or score_map.size(-1) != feat_map.size(-1):
            raise ValueError("score_map and feat_map must share [B, H, W].")

        bsz, peak_feat_dim, height, width = feat_map.shape
        refined_score_map = score_map.clone()

        peaks_xy, peak_scores = topk_peaks_nms(
            score_map,
            topk=self.topk_peaks,
            kernel_size=self.nms_kernel_size,
        )
        peak_feats = sample_feature_at_peaks(feat_map, peaks_xy=peaks_xy)  # [B, K, C]
        num_peaks = peak_scores.size(1)

        top1 = peak_scores[:, 0]
        top2 = peak_scores[:, 1] if num_peaks > 1 else torch.zeros_like(top1)
        score_ratio = peak_scores / (top1[:, None] + self.eps)
        ambiguity_ratio = top2 / (top1 + self.eps)
        effective_peak_count = (score_ratio > self.multi_peak_ratio_thresh).sum(dim=-1)
        is_ambiguous = (ambiguity_ratio >= self.ratio_thresh) | (effective_peak_count >= 3)

        raw_history_shape = None
        projected_history_shape = None
        template_embed = None
        dynamic_history = None
        dynamic_history_source = "none"
        if history_tokens is not None:
            if history_tokens.dim() != 3:
                raise ValueError(f"history_tokens must be [B, T, C], got {tuple(history_tokens.shape)}")
            raw_history_shape = tuple(history_tokens.shape)
            history_tokens = self.template_proj(history_tokens)
            projected_history_shape = tuple(history_tokens.shape)
            if history_tokens.size(-1) != peak_feat_dim:
                raise ValueError(
                    f"Projected history C mismatch: raw_history={raw_history_shape}, "
                    f"projected_history={projected_history_shape}, "
                    f"feat_map={tuple(feat_map.shape)}, "
                    f"history_bank_feat_dim={self.history_bank.feat_dim}"
                )
            if history_tokens.size(-1) != self.history_bank.feat_dim:
                raise ValueError(
                    f"Projected history does not match history bank dim: "
                    f"projected_history={projected_history_shape}, "
                    f"history_bank_feat_dim={self.history_bank.feat_dim}"
                )
            template_embed = history_tokens.mean(dim=1)
            if self.use_mamba_history_bank:
                dynamic_history = self.history_bank.encode_sequence(history_tokens)
                dynamic_history_source = "differentiable_tokens"
        else:
            if self.use_mamba_history_bank:
                dynamic_history = self.history_bank.get_history()
                dynamic_history_source = "cached_bank" if dynamic_history is not None else "none"

        template_anchor = None
        if self.use_template_anchor:
            if self.template_anchor is not None:
                template_anchor = self._validate_anchor_tensor(
                    self.template_anchor,
                    name="template_anchor",
                    batch_size=bsz,
                    feat_dim=peak_feat_dim,
                    device=feat_map.device,
                    dtype=feat_map.dtype,
                )
            elif template_embed is not None:
                template_anchor = self._validate_anchor_tensor(
                    template_embed,
                    name="template_embed_anchor",
                    batch_size=bsz,
                    feat_dim=peak_feat_dim,
                    device=feat_map.device,
                    dtype=feat_map.dtype,
                )

        first_frame_anchor = None
        if self.use_first_frame_anchor and self.first_frame_anchor is not None:
            first_frame_anchor = self._validate_anchor_tensor(
                self.first_frame_anchor,
                name="first_frame_anchor",
                batch_size=bsz,
                feat_dim=peak_feat_dim,
                device=feat_map.device,
                dtype=feat_map.dtype,
            )

        has_dynamic_history = dynamic_history is not None
        has_template_anchor = template_anchor is not None
        has_first_frame_anchor = first_frame_anchor is not None
        history_source = dynamic_history_source

        diag = max((height - 1) ** 2 + (width - 1) ** 2, 1)
        diag = float(diag ** 0.5)

        sim_template_anchor = self._cosine_peaks_to_anchor(
            peak_feats,
            template_anchor,
            name="template_anchor",
        )
        sim_first_frame_anchor = self._cosine_peaks_to_anchor(
            peak_feats,
            first_frame_anchor,
            name="first_frame_anchor",
        )
        sim_mamba_history = torch.zeros(bsz, num_peaks, device=score_map.device, dtype=score_map.dtype)

        if has_dynamic_history:
            history = dynamic_history
            if history.dim() == 3:
                history = history[:, -1, :]
            history = self._validate_anchor_tensor(
                history,
                name="mamba_history_embed",
                batch_size=bsz,
                feat_dim=peak_feat_dim,
                device=feat_map.device,
                dtype=feat_map.dtype,
            )
            sim_mamba_history = self._cosine_peaks_to_anchor(
                peak_feats,
                history,
                name="mamba_history_embed",
            )

        if has_template_anchor and has_first_frame_anchor and has_dynamic_history:
            history_source = f"template_anchor+first_frame_anchor+{dynamic_history_source}"
        elif has_template_anchor and has_first_frame_anchor:
            history_source = "template_anchor+first_frame_anchor"
        elif has_template_anchor and has_dynamic_history:
            history_source = f"template_anchor+{dynamic_history_source}"
        elif has_first_frame_anchor and has_dynamic_history:
            history_source = f"first_frame_anchor+{dynamic_history_source}"
        elif has_template_anchor:
            history_source = "template_anchor"
        elif has_first_frame_anchor:
            history_source = "first_frame_anchor"
        elif has_dynamic_history:
            history_source = dynamic_history_source
        else:
            history_source = "none"

        sim_template = sim_template_anchor
        sim_history = sim_mamba_history
        has_history = has_template_anchor or has_first_frame_anchor or has_dynamic_history
        max_history_sim = sim_history.max(dim=-1).values

        prev_center = self._as_batch_center(prev_center, bsz, score_map.device, score_map.dtype)
        if prev_center is None:
            dist_to_prev = torch.zeros(bsz, num_peaks, device=score_map.device, dtype=score_map.dtype)
        else:
            dist_to_prev = torch.norm(peaks_xy.to(score_map.dtype) - prev_center[:, None, :], dim=-1) / diag

        if num_peaks > 1:
            rank = torch.arange(num_peaks, device=score_map.device, dtype=score_map.dtype) / float(num_peaks - 1)
        else:
            rank = torch.zeros(num_peaks, device=score_map.device, dtype=score_map.dtype)
        rank = rank.view(1, num_peaks).expand(bsz, num_peaks)

        rerank_features = torch.stack(
            [
                peak_scores,
                score_ratio,
                sim_template,
                sim_mamba_history,
                dist_to_prev,
                rank,
                ambiguity_ratio[:, None].expand(bsz, num_peaks),
            ],
            dim=-1,
        )
        target_logits = self.reranker_mlp(rerank_features).squeeze(-1)  # [B, K]
        target_probs = torch.softmax(target_logits, dim=-1)
        # `target_logits/target_probs` are the original classification-head outputs.
        # Identity-aware candidate ranking is computed before the caller-side
        # Safe Gate decides whether to switch away from the baseline top-1 peak.
        identity_score = target_logits
        identity_probs = target_probs
        identity_score_source = "target_logits"
        if has_history and self.use_history_aware_rerank_score:
            identity_score = (
                self.template_anchor_weight * sim_template_anchor +
                self.first_frame_anchor_weight * sim_first_frame_anchor +
                self.mamba_history_weight * sim_mamba_history
            )
            identity_probs = torch.softmax(identity_score, dim=-1)
            identity_score_source = "history_aware"
        # Best identity candidate comes from identity_score. The final selected
        # candidate may still fall back to baseline in SUTRACK._rerank_score_decision.
        identity_confidence, best_id_idx = identity_probs.max(dim=-1)
        batch_idx = torch.arange(bsz, device=score_map.device)
        rerank_center = peaks_xy[batch_idx, best_id_idx].to(score_map.dtype)
        baseline_center = peaks_xy[:, 0, :].to(score_map.dtype)

        # Dense suppression is intentionally disabled for V4 Top-K mode. Keep
        # compatibility fields, but bbox decoding should use selected_center.
        suppress_applied = torch.zeros_like(is_ambiguous)
        suppress_peak_index = torch.full((bsz,), -1, dtype=torch.long, device=score_map.device)
        selected_peak_idx = torch.zeros(bsz, dtype=torch.long, device=score_map.device)
        target_feat = peak_feats[:, 0, :].detach()
        suppression_block_reason = "dense_suppression_disabled_v4_topk"
        refined_identity_reason = "refined_score_map_kept_equal_to_input_score_map"

        aux_info = {
            "is_ambiguous": is_ambiguous,
            "top1_score": top1,
            "top2_score": top2,
            "top2_ratio": ambiguity_ratio,
            "ambiguity_ratio": ambiguity_ratio,
            "effective_peak_count": effective_peak_count,
            "peaks_xy": peaks_xy,
            "peaks_score": peak_scores,
            "top1_xy": peaks_xy[:, 0, :],
            "top2_xy": peaks_xy[:, 1, :] if num_peaks > 1 else peaks_xy[:, 0, :],
            "history_available": torch.tensor(has_history, device=score_map.device, dtype=torch.bool).expand(bsz),
            "history_source": history_source,
            "sim_template": sim_template,
            "sim_history": sim_history,
            "sim_template_anchor": sim_template_anchor,
            "sim_first_frame_anchor": sim_first_frame_anchor,
            "sim_mamba_history": sim_mamba_history,
            "sim_peak1_to_history": sim_history[:, 0],
            "sim_peak2_to_history": sim_history[:, 1] if num_peaks > 1 else sim_history[:, 0],
            "max_history_sim": max_history_sim,
            "target_logits": target_logits,
            "target_probs": target_probs,
            "identity_score": identity_score,
            "identity_probs": identity_probs,
            "identity_idx": best_id_idx,
            "identity_confidence": identity_confidence,
            "identity_score_source": identity_score_source,
            # Backward-compatible aliases for existing logging/loss code.
            "rerank_score": identity_score,
            "rerank_score_source": identity_score_source,
            "use_history_aware_rerank_score": self.use_history_aware_rerank_score,
            "rerank_idx": best_id_idx,
            "rerank_confidence": identity_confidence,
            "rerank_center": rerank_center,
            "baseline_center": baseline_center,
            "selected_peak_idx": best_id_idx.clone(),
            "selected_center": baseline_center,
            "gate_logits": target_logits,
            "gate_probs": target_probs,
            "gate_confidence": identity_confidence,
            "gate_loss_inputs": {
                "features": rerank_features.detach(),
                "is_ambiguous": is_ambiguous.detach(),
                "effective_peak_count": effective_peak_count.detach(),
            },
            "suppress_applied": suppress_applied,
            "suppress_peak_index": suppress_peak_index,  # -1:none
            "suppression_strength": torch.full(
                (bsz,), float(self.suppression_strength), device=score_map.device, dtype=score_map.dtype
            ),
            "suppression_block_reason": suppression_block_reason,
            "refined_identity_reason": refined_identity_reason,
            "target_feat": target_feat,  # [B, C], for external safe-frame update
            "peak_feats": peak_feats.detach(),  # [B, K, C], optional debug
        }
        return refined_score_map, aux_info
