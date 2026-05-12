import torch
from torch import nn

try:
    from mamba_ssm.modules.mamba_simple import Mamba
except ImportError:
    Mamba = None


class TemporalMambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, expand=2):
        super().__init__()
        if Mamba is None:
            raise ImportError("mamba_ssm is required when Temporal Mamba is enabled.")

        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, expand=expand)
        self._zero_init_output_projection()

    def _zero_init_output_projection(self):
        out_proj = getattr(self.mamba, 'out_proj', None)
        if out_proj is None:
            return
        nn.init.zeros_(out_proj.weight)
        if out_proj.bias is not None:
            nn.init.zeros_(out_proj.bias)

    def forward(self, x_seq):
        if x_seq.dim() != 4:
            raise ValueError(f"TemporalMambaBlock expects [B, T, L, C], got {tuple(x_seq.shape)}")

        bsz, time_len, num_tokens, channels = x_seq.shape
        x = x_seq.transpose(1, 2).contiguous().view(bsz * num_tokens, time_len, channels)
        x = x + self.mamba(self.norm(x))
        return x.view(bsz, num_tokens, time_len, channels).transpose(1, 2).contiguous()


class OnlineTemporalMamba(nn.Module):
    def __init__(self, dim, d_state=16, expand=2, use_online_memory_inference=False):
        super().__init__()
        self.temporal_block = TemporalMambaBlock(dim=dim, d_state=d_state, expand=expand)
        self.use_online_memory_inference = bool(use_online_memory_inference)
        self.state_cache = None
        self.state_batch = None

    def clear_state(self):
        self.state_cache = None
        self.state_batch = None

    def reset_state(self, batch_size, device=None, dtype=None):
        self.state_cache = None
        self.state_batch = int(batch_size)

    def _ensure_state_batch(self, batch_size):
        if self.state_batch is None:
            self.state_batch = int(batch_size)
            return
        if self.state_batch != int(batch_size):
            self.clear_state()
            self.state_batch = int(batch_size)

    def forward(self, search_tokens, num_search=1, is_inference=False):
        if search_tokens.dim() != 3:
            raise ValueError(f"OnlineTemporalMamba expects [B, L, C], got {tuple(search_tokens.shape)}")

        bsz = search_tokens.size(0)
        self._ensure_state_batch(bsz)

        if num_search > 1:
            total_tokens = search_tokens.size(1)
            if total_tokens % num_search != 0:
                raise ValueError(
                    f"search token length {total_tokens} is not divisible by num_search={num_search}."
                )

            tokens_per_search = total_tokens // num_search
            x_seq = search_tokens.reshape(bsz, num_search, tokens_per_search, search_tokens.size(-1))
            fused_seq = self.temporal_block(x_seq)
            if is_inference and self.use_online_memory_inference:
                self.state_cache = fused_seq[:, -1, :, :].detach()
            else:
                self.state_cache = None
            return fused_seq.reshape(bsz, total_tokens, search_tokens.size(-1))

        use_online_memory = is_inference and self.use_online_memory_inference
        prev_tokens = self.state_cache if use_online_memory and self.state_cache is not None else search_tokens

        stacked = torch.stack([prev_tokens, search_tokens], dim=1)
        fused = self.temporal_block(stacked)[:, -1, :, :]

        if use_online_memory:
            self.state_cache = fused.detach()
        else:
            self.state_cache = None

        return fused


class EncoderPostprocess(nn.Module):
    def __init__(self, in_dim, cfg):
        super().__init__()
        self.temporal = None

        tmp_cfg = cfg.TEMPORAL_MAMBA
        if tmp_cfg.ENABLED:
            self.temporal = OnlineTemporalMamba(
                dim=in_dim,
                d_state=tmp_cfg.D_STATE,
                expand=tmp_cfg.EXPAND,
                use_online_memory_inference=getattr(tmp_cfg, "USE_ONLINE_MEMORY_INFERENCE", False),
            )

    def clear_online_state(self):
        if self.temporal is not None:
            self.temporal.clear_state()

    def reset_online_state(self, batch_size, device=None, dtype=None):
        if self.temporal is not None:
            self.temporal.reset_state(batch_size=batch_size, device=device, dtype=dtype)

    def forward(self, search_tokens, num_search=1, is_inference=False):
        x = search_tokens
        if self.temporal is not None:
            x = self.temporal(x, num_search=num_search, is_inference=is_inference)
        return x


def build_encoder_postprocess(cfg, encoder):
    post_cfg = cfg.MODEL.ENCODER_POSTPROCESS
    if not post_cfg.ENABLED:
        return None
    return EncoderPostprocess(in_dim=encoder.num_channels, cfg=post_cfg)