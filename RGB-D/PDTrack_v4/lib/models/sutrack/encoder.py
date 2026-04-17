"""
Encoder modules: we use ITPN for the encoder.
"""
import torch
from torch import nn
import torch.utils.checkpoint as checkpoint

from lib.models.sutrack import fastitpn as fastitpn_module
from lib.models.sutrack import itpn as oriitpn_module
from lib.models.sutrack.conditioned_mamba import TemporalMambaBlock, WindowConditionedMambaLayer
from lib.utils.misc import is_main_process


class DepthPatchMLP(nn.Module):
    """Per-patch MLP for depth tokens — no global self-attention, interaction
    happens later in the Mamba fusion stage. Much cheaper than Transformer."""
    def __init__(self, dim, depth=2, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        layers = []
        for _ in range(depth):
            layers.append(nn.LayerNorm(dim))
            layers.append(nn.Linear(dim, hidden))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden, dim))
            layers.append(nn.Dropout(dropout))
        self.blocks = nn.ModuleList(layers)
        self._num_layers = depth
        self._block_size = 6  # norm, linear, gelu, drop, linear, drop

    def forward(self, x):
        for i in range(self._num_layers):
            off = i * self._block_size
            residual = x
            x = self.blocks[off](x)      # LayerNorm
            x = self.blocks[off+1](x)    # Linear
            x = self.blocks[off+2](x)    # GELU
            x = self.blocks[off+3](x)    # Dropout
            x = self.blocks[off+4](x)    # Linear
            x = self.blocks[off+5](x)    # Dropout
            x = residual + x
        return x


class DualStreamMambaFusion(nn.Module):
    def __init__(
        self,
        dim,
        mamba_depth=2,
        num_heads=4,
        window_size=14,
        d_state=8,
        d_conv=3,
        expand=1,
        mod_scale_init=0.0,
        template_gate_init=0.1,
        search_gate_enabled=True,
        template_gate_enabled=True,
    ):
        super().__init__()
        del num_heads

        self.search_mamba = nn.ModuleList(
            [
                WindowConditionedMambaLayer(
                    dim=dim,
                    depth=1,
                    window_size=window_size,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    mod_scale_init=mod_scale_init,
                )
                for _ in range(mamba_depth)
            ]
        )
        self.template_mamba = nn.ModuleList(
            [
                WindowConditionedMambaLayer(
                    dim=dim,
                    depth=1,
                    window_size=window_size,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    mod_scale_init=mod_scale_init,
                )
                for _ in range(mamba_depth)
            ]
        )

        self.search_gate_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.search_gate_proj.weight)
        nn.init.constant_(self.search_gate_proj.bias, -2.0)

        # Simplified template gate: single learnable scalar, no double gating
        self.template_gate_scalar = nn.Parameter(torch.tensor([template_gate_init]))

        self.search_gate_enabled = search_gate_enabled
        self.template_gate_enabled = template_gate_enabled

        # Lightweight cross-attention: search queries template after Mamba fusion
        self.cross_norm_s = nn.LayerNorm(dim)
        self.cross_norm_t = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=4, dropout=0.1, batch_first=True)
        self.cross_gate = nn.Parameter(torch.tensor([0.0]))  # residual gate, starts at 0

        self.norm = nn.LayerNorm(dim)

    def forward(self, rgb_tokens, depth_tokens, search_hw, template_hw, num_search, num_template):
        b, l, c = rgb_tokens.shape
        h_s, w_s = search_hw
        h_t, w_t = template_hw

        l_s_total = num_search * h_s * w_s
        l_t_total = num_template * h_t * w_t

        cls_len = l - l_s_total - l_t_total
        cls_tokens = rgb_tokens[:, :cls_len, :] if cls_len > 0 else None

        s_rgb = rgb_tokens[:, cls_len : cls_len + l_s_total, :]
        s_dep = depth_tokens[:, cls_len : cls_len + l_s_total, :]
        t_rgb = rgb_tokens[:, cls_len + l_s_total :, :]
        t_dep = depth_tokens[:, cls_len + l_s_total :, :]

        s_rgb_2d = s_rgb.reshape(b * num_search, h_s, w_s, c).contiguous()
        s_dep_2d = s_dep.reshape(b * num_search, h_s, w_s, c).contiguous()
        t_rgb_2d = t_rgb.reshape(b * num_template, h_t, w_t, c).contiguous()
        t_dep_2d = t_dep.reshape(b * num_template, h_t, w_t, c).contiguous()

        s_state = s_rgb_2d
        for layer in self.search_mamba:
            s_state = layer(s_state, s_dep_2d)

        t_state = t_rgb_2d
        for layer in self.template_mamba:
            t_state = layer(t_state, t_dep_2d)

        s_state = s_state.reshape(b, l_s_total, c)
        t_state = t_state.reshape(b, l_t_total, c)

        s_delta = s_state - s_rgb
        t_delta = t_state - t_rgb

        # Search gate: per-token sigmoid gating
        search_gate = torch.sigmoid(self.search_gate_proj(s_state)) if self.search_gate_enabled else 1.0
        search_out = s_rgb + search_gate * s_delta

        # Template gate: single learnable scalar (simplified from double gate)
        tg = torch.sigmoid(self.template_gate_scalar) if self.template_gate_enabled else 1.0
        template_out = t_rgb + tg * t_delta

        # Lightweight cross-attention: search attends to template
        cross_s = self.cross_norm_s(search_out)
        cross_t = self.cross_norm_t(template_out)
        cross_out, _ = self.cross_attn(cross_s, cross_t, cross_t)
        search_out = search_out + torch.tanh(self.cross_gate) * cross_out

        out_list = [search_out, template_out]
        if cls_tokens is not None:
            out_list.insert(0, cls_tokens)

        return self.norm(torch.cat(out_list, dim=1))


class EncoderBase(nn.Module):
    def __init__(self, encoder: nn.Module, train_encoder: bool, open_layers: list, num_channels: int):
        super().__init__()
        self.body = encoder
        self.num_channels = num_channels
        self.train_encoder = train_encoder
        self.open_layers = open_layers or []

    def forward(self, template_list, search_list, template_anno_list, text_src, task_index):
        return self.body(template_list, search_list, template_anno_list, text_src, task_index)


class Encoder(EncoderBase):
    """ViT encoder."""

    def __init__(self, name: str, train_encoder: bool, search_size: int, template_size: int, open_layers: list, cfg=None):
        if "fastitpn" in name.lower():
            encoder = getattr(fastitpn_module, name)(
                pretrained=is_main_process(),
                search_size=search_size,
                template_size=template_size,
                drop_rate=0.0,
                in_chans=3,
                drop_path_rate=0.1,
                attn_drop_rate=0.0,
                init_values=0.1,
                drop_block_rate=None,
                use_mean_pooling=True,
                grad_ckpt=False,
                cls_token=cfg.MODEL.ENCODER.CLASS_TOKEN,
                pos_type=cfg.MODEL.ENCODER.POS_TYPE,
                token_type_indicate=cfg.MODEL.ENCODER.TOKEN_TYPE_INDICATE,
                pretrain_type=cfg.MODEL.ENCODER.PRETRAIN_TYPE,
                patchembed_init=cfg.MODEL.ENCODER.PATCHEMBED_INIT,
            )
            if "itpnb" in name:
                num_channels = 512
            elif "itpnl" in name:
                num_channels = 768
            elif "itpnt" in name or "itpns" in name:
                num_channels = 384
            else:
                num_channels = 512
        elif "oriitpn" in name.lower():
            encoder = getattr(oriitpn_module, name)(
                pretrained=is_main_process(),
                search_size=search_size,
                template_size=template_size,
                drop_path_rate=0.1,
                init_values=0.1,
                use_mean_pooling=True,
                ape=True,
                rpe=True,
                pos_type=cfg.MODEL.ENCODER.POS_TYPE,
                token_type_indicate=cfg.MODEL.ENCODER.TOKEN_TYPE_INDICATE,
                task_num=cfg.MODEL.TASK_NUM,
                pretrain_type=cfg.MODEL.ENCODER.PRETRAIN_TYPE,
            )
            num_channels = 512
        else:
            raise ValueError()

        super().__init__(encoder, train_encoder, open_layers, num_channels)

        self.use_mamba_fusion = hasattr(cfg.MODEL, "MAMBA") and getattr(cfg.MODEL.MAMBA, "ENABLE", False)
        self.use_temporal_mamba = hasattr(cfg.MODEL, "MAMBA") and getattr(cfg.MODEL.MAMBA, "TEMPORAL_ENABLE", False)
        self.temporal_checkpoint = hasattr(cfg.MODEL, "MAMBA") and getattr(cfg.MODEL.MAMBA, "TEMPORAL_CHECKPOINT", True)
        self.depth_dropout_prob = getattr(cfg.MODEL.MAMBA, "DEPTH_DROPOUT", 0.0)
        self.temporal_dropout_prob = getattr(cfg.MODEL.MAMBA, "TEMPORAL_DROPOUT", 0.0)
        self.last_num_search = 1
        self.last_num_template = 1
        self.has_printed_mamba_status = False

        if self.use_temporal_mamba:
            temporal_dim = getattr(cfg.MODEL.MAMBA, "TEMPORAL_DIM", None) or num_channels
            if temporal_dim != num_channels:
                raise ValueError(
                    f"Temporal Mamba dim ({temporal_dim}) must match encoder output dim ({num_channels})."
                )
            self.temporal_mamba = TemporalMambaBlock(
                dim=temporal_dim,
                d_state=getattr(cfg.MODEL.MAMBA, "TEMPORAL_D_STATE", 4),
                expand=getattr(cfg.MODEL.MAMBA, "TEMPORAL_EXPAND", 1),
            )

        if self.use_mamba_fusion:
            mamba_dim = getattr(cfg.MODEL.MAMBA, "DIM", num_channels)
            mamba_depth = getattr(cfg.MODEL.MAMBA, "DEPTH", 2)
            mamba_window = getattr(cfg.MODEL.MAMBA, "WINDOW_SIZE", 14)
            if mamba_window == 0:
                mamba_window = 14
            print(f"[Encoder] Initializing Mamba Fusion Block (dim={mamba_dim}, depth={mamba_depth})")
            self.mamba_fusion = DualStreamMambaFusion(
                dim=mamba_dim,
                mamba_depth=mamba_depth,
                window_size=mamba_window,
                d_state=getattr(cfg.MODEL.MAMBA, "D_STATE", 8),
                d_conv=getattr(cfg.MODEL.MAMBA, "D_CONV", 3),
                expand=getattr(cfg.MODEL.MAMBA, "EXPAND", 1),
                mod_scale_init=getattr(cfg.MODEL.MAMBA, "MOD_SCALE_INIT", 0.0),
                template_gate_init=getattr(cfg.MODEL.MAMBA, "TEMPLATE_GATE_INIT", 0.1),
                search_gate_enabled=getattr(cfg.MODEL.MAMBA, "SEARCH_GATE", True),
                template_gate_enabled=getattr(cfg.MODEL.MAMBA, "TEMPLATE_GATE", True),
            )

            patch_size = 16
            self.depth_patch_embed = nn.Conv2d(1, mamba_dim, kernel_size=patch_size, stride=patch_size)
            num_patches_search = (search_size // patch_size) ** 2
            num_patches_template = (template_size // patch_size) ** 2
            self.depth_pos_embed = nn.Parameter(torch.zeros(1, num_patches_search + num_patches_template, mamba_dim))
            nn.init.trunc_normal_(self.depth_pos_embed, std=0.02)

            # Temporal frame embeddings for multi-frame depth (max 8 frames)
            max_temporal_frames = 8
            self.depth_temporal_embed = nn.Parameter(torch.zeros(max_temporal_frames, 1, mamba_dim))
            nn.init.trunc_normal_(self.depth_temporal_embed, std=0.02)

            self.depth_encoder = DepthPatchMLP(dim=mamba_dim, depth=2, mlp_ratio=2.0, dropout=0.1)

        self._apply_freezing_policy()

    def _apply_freezing_policy(self):
        if self.train_encoder:
            return

        open_layer_set = set(self.open_layers or [])
        stage3_prefixes = set()
        if "stage3" in open_layer_set and hasattr(self.body, "num_main_blocks") and hasattr(self.body, "blocks"):
            stage3_start = len(self.body.blocks) - self.body.num_main_blocks
            stage3_prefixes = {f"body.blocks.{idx}." for idx in range(stage3_start, len(self.body.blocks))}

        for name, parameter in self.named_parameters():
            freeze = True

            if "stage3" in open_layer_set:
                if any(name.startswith(prefix) for prefix in stage3_prefixes):
                    freeze = False
                elif name.startswith("body.norm") or name.startswith("body.fc_norm"):
                    freeze = False

            for open_item in open_layer_set:
                if open_item == "stage3":
                    continue
                if name == open_item or name.startswith(f"{open_item}.") or f".{open_item}." in name:
                    freeze = False
                    break

            if freeze:
                parameter.requires_grad_(False)

    def _apply_temporal_mamba(self, tokens, num_search, num_template):
        if not self.use_temporal_mamba or num_search <= 1:
            return tokens

        b, total_len, c = tokens.shape
        cls_len = 1 if self.body.cls_token is not None else 0
        search_token_len = self.body.num_patches_search * num_search
        template_token_len = self.body.num_patches_template * num_template

        search_start = cls_len
        search_end = search_start + search_token_len
        template_end = search_end + template_token_len

        if template_end > total_len:
            raise ValueError(
                f"Token layout mismatch in temporal path: total={total_len}, "
                f"search={search_token_len}, template={template_token_len}, cls={cls_len}."
            )

        search_tokens = tokens[:, search_start:search_end, :]
        search_tokens = search_tokens.view(b, num_search, self.body.num_patches_search, c)

        if self.training and self.temporal_checkpoint:
            search_tokens = checkpoint.checkpoint(self.temporal_mamba, search_tokens, use_reentrant=False)
        else:
            search_tokens = self.temporal_mamba(search_tokens)

        parts = []
        if cls_len > 0:
            parts.append(tokens[:, :cls_len, :])
        parts.append(search_tokens.view(b, search_token_len, c))
        parts.append(tokens[:, search_end:template_end, :])
        if template_end < total_len:
            parts.append(tokens[:, template_end:, :])

        return torch.cat(parts, dim=1)

    def _apply_training_dropouts(self, template_list, search_list):
        if not self.training:
            return template_list, search_list

        dropped_search_list = list(search_list)
        dropped_template_list = list(template_list)

        if self.temporal_dropout_prob > 0.0 and len(dropped_search_list) > 1:
            if torch.rand(1).item() < self.temporal_dropout_prob:
                dropped_search_list = [dropped_search_list[-1]]

        if self.depth_dropout_prob <= 0.0:
            return dropped_template_list, dropped_search_list

        if torch.rand(1).item() >= self.depth_dropout_prob:
            return dropped_template_list, dropped_search_list

        def _zero_depth_channels(tensor):
            if tensor.shape[1] <= 3:
                return tensor
            tensor = tensor.clone()
            tensor[:, 3:, :, :] = 0
            return tensor

        dropped_template_list = [_zero_depth_channels(t) for t in dropped_template_list]
        dropped_search_list = [_zero_depth_channels(t) for t in dropped_search_list]
        return dropped_template_list, dropped_search_list

    def forward(self, template_list, search_list, template_anno_list, text_src, task_index):
        template_list, search_list = self._apply_training_dropouts(template_list, search_list)
        self.last_num_template = len(template_list)
        self.last_num_search = len(search_list)

        if template_list[0].shape[1] == 3:
            template_list = [torch.cat([img, torch.zeros_like(img[:, :1, :, :])], dim=1) for img in template_list]
            search_list = [torch.cat([img, torch.zeros_like(img[:, :1, :, :])], dim=1) for img in search_list]

        first_img = template_list[0]
        # Determine if meaningful depth channels are present (not all zeros).
        depth_present = False
        if self.use_mamba_fusion and first_img.shape[1] == 4:
            try:
                t_depth = template_list[0][:, 3:, :, :]
                s_depth = search_list[-1][:, 3:, :, :]
                depth_sum = float(t_depth.abs().sum().item()) + float(s_depth.abs().sum().item())
                depth_present = depth_sum > 1e-6
            except Exception:
                depth_present = False

        if is_main_process():
            if depth_present:
                if not self.has_printed_mamba_status:
                    print("\n[DEBUG] >>> MambaFusion is ACTIVE (Logged once) <<<")
                    self.has_printed_mamba_status = True
            else:
                if not self.has_printed_mamba_status:
                    print("[DEBUG] MambaFusion is INACTIVE")
                    self.has_printed_mamba_status = True

        if depth_present:
            rgb_template_list = [img[:, :3, :, :] for img in template_list]
            rgb_search_list = [img[:, :3, :, :] for img in search_list]

            b = first_img.shape[0]
            num_template = len(template_list)
            num_search = len(search_list)

            d_s_list = [img[:, 3:, :, :] for img in search_list]
            d_s = torch.stack(d_s_list, dim=0).flatten(0, 1)
            d_s_emb = self.depth_patch_embed(d_s).flatten(2).transpose(1, 2)

            d_t_list = [img[:, 3:, :, :] for img in template_list]
            d_t = torch.stack(d_t_list, dim=0).flatten(0, 1)
            d_t_emb = self.depth_patch_embed(d_t).flatten(2).transpose(1, 2)

            num_patches_s = d_s_emb.shape[1]
            pe_s = self.depth_pos_embed[:, :num_patches_s, :]
            pe_t = self.depth_pos_embed[:, num_patches_s:, :]
            d_s_emb = d_s_emb + pe_s
            d_t_emb = d_t_emb + pe_t

            # Add temporal frame embeddings so each search/template frame is distinguishable
            _, l_s, c = d_s_emb.shape
            d_s_emb = d_s_emb.reshape(num_search, b, l_s, c)  # [T_s, B, L, C]
            for t in range(num_search):
                d_s_emb[t] = d_s_emb[t] + self.depth_temporal_embed[t % self.depth_temporal_embed.shape[0]]
            d_s_emb = d_s_emb.permute(1, 0, 2, 3).flatten(1, 2)  # [B, T_s*L, C]

            _, l_t, c = d_t_emb.shape
            d_t_emb = d_t_emb.reshape(num_template, b, l_t, c)  # [T_t, B, L, C]
            for t in range(num_template):
                idx = (num_search + t) % self.depth_temporal_embed.shape[0]
                d_t_emb[t] = d_t_emb[t] + self.depth_temporal_embed[idx]
            d_t_emb = d_t_emb.permute(1, 0, 2, 3).flatten(1, 2)  # [B, T_t*L, C]

            depth_tokens = torch.cat([d_s_emb, d_t_emb], dim=1)
            depth_tokens = self.depth_encoder(depth_tokens)

            rgb_tokens, rpe_index = self.body.forward_stage1_2(
                rgb_template_list, rgb_search_list, template_anno_list, text_src=text_src, task_index=task_index
            )

            if rgb_tokens.shape[1] != depth_tokens.shape[1]:
                diff = rgb_tokens.shape[1] - depth_tokens.shape[1]
                if diff > 0:
                    cls_dummy = torch.zeros(
                        depth_tokens.shape[0], diff, depth_tokens.shape[2], device=depth_tokens.device, dtype=depth_tokens.dtype
                    )
                    depth_tokens = torch.cat([cls_dummy, depth_tokens], dim=1)
                elif diff < 0:
                    raise ValueError(
                        f"Unexpected token length mismatch: RGB has {rgb_tokens.shape[1]} tokens, "
                        f"Depth has {depth_tokens.shape[1]} tokens after adjustment."
                    )

            patch_size = 16
            hs, ws = search_list[0].shape[2] // patch_size, search_list[0].shape[3] // patch_size
            ht, wt = template_list[0].shape[2] // patch_size, template_list[0].shape[3] // patch_size

            fused_tokens = self.mamba_fusion(
                rgb_tokens,
                depth_tokens,
                search_hw=(hs, ws),
                template_hw=(ht, wt),
                num_search=num_search,
                num_template=num_template,
            )

            out = self.body.forward_stage3(fused_tokens, rpe_index)
            out = self._apply_temporal_mamba(out, num_search=num_search, num_template=num_template)

            if self.training:
                import torch.nn.functional as F

                current_search_depth = d_s_list[-1]
                search_depth_scalar = F.adaptive_avg_pool2d(current_search_depth, (hs, ws)).squeeze(1)

                cls_len = max(0, rgb_tokens.shape[1] - (d_s_emb.shape[1] + d_t_emb.shape[1]))
                l_s = hs * ws
                start_s = cls_len + (num_search - 1) * l_s
                end_s = cls_len + num_search * l_s
                search_fused = fused_tokens[:, start_s:end_s, :]
                template_fused = fused_tokens[:, end_s:, :]

                aux_info = {
                    "search_fused": search_fused,
                    "template_fused": template_fused,
                    "search_depth_scalar": search_depth_scalar,
                    "search_hw": (hs, ws),
                    "template_hw": (ht, wt),
                }
                return [out], aux_info

            return [out]

        # No depth path: strip depth channel before passing to body (expects 3ch)
        rgb_template_list = [img[:, :3, :, :] for img in template_list]
        rgb_search_list = [img[:, :3, :, :] for img in search_list]
        out = self.body(rgb_template_list, rgb_search_list, template_anno_list, text_src, task_index)
        out[0] = self._apply_temporal_mamba(out[0], num_search=len(search_list), num_template=len(template_list))

        if self.training and self.use_mamba_fusion:
            # Even without depth, return aux_info so DGCL loss stays active
            import torch.nn.functional as F

            patch_size = 16
            num_search = len(search_list)
            num_template = len(template_list)
            hs = search_list[0].shape[2] // patch_size
            ws = search_list[0].shape[3] // patch_size
            ht = template_list[0].shape[2] // patch_size
            wt = template_list[0].shape[3] // patch_size

            cls_len = 1 if self.body.cls_token is not None else 0
            l_s = hs * ws
            l_t = ht * wt
            start_s = cls_len + (num_search - 1) * l_s
            end_s = cls_len + num_search * l_s
            end_t = end_s + num_template * l_t

            tokens = out[0]
            search_fused = tokens[:, start_s:end_s, :]
            template_fused = tokens[:, end_s:end_t, :]

            # Synthesize a zero depth scalar map (no depth info available)
            b = tokens.shape[0]
            search_depth_scalar = torch.zeros(b, hs, ws, device=tokens.device, dtype=tokens.dtype)

            aux_info = {
                "search_fused": search_fused,
                "template_fused": template_fused,
                "search_depth_scalar": search_depth_scalar,
                "search_hw": (hs, ws),
                "template_hw": (ht, wt),
            }
            return out, aux_info

        return out


def build_encoder(cfg):
    train_encoder = (cfg.TRAIN.ENCODER_MULTIPLIER > 0) and (cfg.TRAIN.FREEZE_ENCODER is False)
    return Encoder(
        cfg.MODEL.ENCODER.TYPE,
        train_encoder,
        cfg.DATA.SEARCH.SIZE,
        cfg.DATA.TEMPLATE.SIZE,
        cfg.TRAIN.ENCODER_OPEN,
        cfg,
    )
