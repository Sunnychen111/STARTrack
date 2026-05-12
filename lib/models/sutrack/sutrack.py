"""
SUTrack model with optional STARTrack post-decoder score-map reranking.
"""
import math

import torch
from torch import nn

from .clip import build_textencoder
from .decoder import build_decoder
from .encoder import build_encoder
from .encoder_postprocess import build_encoder_postprocess
from .post_decoder_disambiguator import PostDecoderDisambiguator
from .task_decoder import build_task_decoder
from lib.utils.box_ops import box_xyxy_to_cxcywh


class SUTRACK(nn.Module):
    def __init__(
        self,
        text_encoder,
        encoder,
        decoder,
        task_decoder,
        encoder_postprocess=None,
        num_frames=1,
        num_template=1,
        decoder_type="CENTER",
        task_feature_type="average",
        post_disambiguator_cfg=None,
        use_startrack=False,
    ):
        super().__init__()
        self.encoder = encoder
        self.text_encoder = text_encoder
        self.decoder = decoder
        self.task_decoder = task_decoder
        self.encoder_postprocess = encoder_postprocess
        self.decoder_type = decoder_type
        self.task_feature_type = task_feature_type
        self.num_frames = num_frames
        self.num_template = num_template
        self.runtime_num_frames = num_frames
        self.use_startrack = bool(use_startrack)

        self.class_token = False if encoder.body.cls_token is None else True
        self.num_patch_x = self.encoder.body.num_patches_search
        self.num_patch_z = self.encoder.body.num_patches_template
        self.fx_sz = int(math.sqrt(self.num_patch_x))
        self.fz_sz = int(math.sqrt(self.num_patch_z))

        self.post_disambiguator_cfg = post_disambiguator_cfg
        self.post_disambiguator = self._build_post_disambiguator()

    def _get_token_layout(self, tokens, num_search, num_template):
        cls_len = 1 if self.class_token else 0
        search_len = self.num_patch_x * num_search
        template_len = self.num_patch_z * num_template
        search_start = cls_len
        search_end = search_start + search_len
        template_end = search_end + template_len
        if tokens.size(1) < template_end:
            raise ValueError(
                f"Token layout mismatch: total={tokens.size(1)}, "
                f"but cls+search+template requires {template_end}."
            )
        return cls_len, search_start, search_end, template_end

    @staticmethod
    def _merge_tokens(cls_tokens, search_tokens, template_tokens, tail_tokens):
        parts = []
        if cls_tokens is not None:
            parts.append(cls_tokens)
        parts.append(search_tokens)
        parts.append(template_tokens)
        if tail_tokens is not None:
            parts.append(tail_tokens)
        return torch.cat(parts, dim=1)

    def clear_online_state(self):
        if self.encoder_postprocess is not None:
            self.encoder_postprocess.clear_online_state()
        if self.post_disambiguator is not None:
            self.post_disambiguator.clear_state()

    def reset_online_state(self, batch_size=1, device=None, dtype=None):
        if self.encoder_postprocess is not None:
            self.encoder_postprocess.reset_online_state(batch_size=batch_size, device=device, dtype=dtype)
        if self.post_disambiguator is not None:
            self.post_disambiguator.reset_history(batch_size=batch_size)

    def forward(
        self,
        text_data=None,
        template_list=None,
        search_list=None,
        template_anno_list=None,
        search_anno_list=None,
        text_src=None,
        task_index=None,
        feature=None,
        mode="encoder",
    ):
        if mode == "text":
            return self.forward_textencoder(text_data)
        if mode == "encoder":
            return self.forward_encoder(template_list, search_list, template_anno_list, text_src, task_index)
        if mode == "decoder":
            return self.forward_decoder(feature), self.forward_task_decoder(feature)
        if mode == "train":
            return self.forward_train(
                template=template_list,
                search=search_list,
                template_anno=template_anno_list,
                search_anno=search_anno_list,
                text_src=text_src,
                task_index=task_index,
            )
        raise ValueError(f"Unsupported mode: {mode}")

    def forward_textencoder(self, text_data):
        return self.text_encoder(text_data)

    def forward_encoder(self, template_list, search_list, template_anno_list, text_src, task_index):
        self.runtime_num_frames = len(search_list)
        xz = self.encoder(template_list, search_list, template_anno_list, text_src, task_index)
        if self.encoder_postprocess is None:
            return xz

        tokens = xz[0]
        cls_len, search_start, search_end, template_end = self._get_token_layout(
            tokens,
            num_search=len(search_list),
            num_template=len(template_list),
        )
        cls_tokens = tokens[:, :cls_len, :] if cls_len > 0 else None
        search_tokens = tokens[:, search_start:search_end, :]
        template_tokens = tokens[:, search_end:template_end, :]
        tail_tokens = tokens[:, template_end:, :] if template_end < tokens.size(1) else None
        search_tokens = self.encoder_postprocess(
            search_tokens,
            num_search=len(search_list),
            is_inference=not self.training,
        )
        xz[0] = self._merge_tokens(cls_tokens, search_tokens, template_tokens, tail_tokens)
        return xz

    def _extract_search_feature_map(self, feature):
        tokens = feature[0] if isinstance(feature, (list, tuple)) else feature
        num_frames = getattr(self, "runtime_num_frames", self.num_frames)
        _, search_start, search_end, _ = self._get_token_layout(
            tokens,
            num_search=num_frames,
            num_template=self.num_template,
        )
        search_tokens = tokens[:, search_start:search_end, :]
        if num_frames > 1:
            search_tokens = search_tokens.reshape(tokens.size(0), num_frames, self.num_patch_x, tokens.size(-1))
            search_tokens = search_tokens[:, -1, :, :]
        bs, hw, channels = search_tokens.size()
        if hw != self.num_patch_x:
            raise ValueError(f"Expected {self.num_patch_x} search tokens, got {hw}.")
        return search_tokens.transpose(1, 2).contiguous().view(bs, channels, self.fx_sz, self.fx_sz)

    def forward_decoder(self, feature, gt_score_map=None, prev_center=None, frame_id=None, record_rerank_stats=True):
        search_fmap = self._extract_search_feature_map(feature)
        bs = search_fmap.size(0)

        if self.decoder_type == "CORNER":
            pred_box, score_map = self.decoder(search_fmap, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box).view(bs, 1, 4)
            return {"pred_boxes": outputs_coord, "score_map": score_map, "f_map": search_fmap}

        if self.decoder_type == "CENTER":
            score_map, bbox, size_map, offset_map, _ = self.decoder(search_fmap, gt_score_map)
            out = {
                "pred_boxes": bbox.view(bs, 1, 4),
                "score_map": score_map,
                "size_map": size_map,
                "offset_map": offset_map,
                "f_map": search_fmap,
            }
            if self.use_startrack and self.post_disambiguator is not None and not self.training:
                refined_score_map, aux_info = self.post_disambiguator(
                    score_map=score_map,
                    feat_map=search_fmap,
                    history_tokens=None,
                    prev_center=prev_center,
                )
                out["score_map_raw"] = score_map
                out["score_map"] = refined_score_map
                out["startrack_aux"] = aux_info
                out["disamb_aux"] = aux_info
            return out

        if self.decoder_type == "MLP":
            score_map, bbox, offset_map = self.decoder(search_fmap, gt_score_map)
            return {
                "pred_boxes": bbox.view(bs, 1, 4),
                "score_map": score_map,
                "offset_map": offset_map,
                "f_map": search_fmap,
            }

        raise NotImplementedError(f"Unsupported decoder type: {self.decoder_type}")

    def forward_task_decoder(self, feature):
        tokens = feature[0] if isinstance(feature, (list, tuple)) else feature
        if self.task_feature_type == "class":
            task_feature = tokens[:, 0:1]
        elif self.task_feature_type == "text":
            task_feature = tokens[:, -1:]
        elif self.task_feature_type == "average":
            task_feature = tokens.mean(1).unsqueeze(1)
        else:
            raise NotImplementedError("task_feature_type must be class, text, or average")
        return self.task_decoder(task_feature)

    def forward_train(self, template, search, template_anno, search_anno, text_src=None, task_index=None):
        if template.dim() != 5 or search.dim() != 5:
            raise ValueError("forward_train expects template/search with shape [B, T, C, H, W].")
        batch_size, num_search = search.shape[:2]
        num_template = template.shape[1]
        search_folded = search.view(batch_size * num_search, *search.shape[2:]).contiguous()
        template_folded = template.repeat_interleave(num_search, dim=0).contiguous()
        template_anno_folded = template_anno.repeat_interleave(num_search, dim=0).contiguous()
        template_list = [template_folded[:, i, ...].contiguous() for i in range(num_template)]
        template_anno_list = [template_anno_folded[:, i, ...].contiguous() for i in range(num_template)]
        text_src_folded = text_src.repeat_interleave(num_search, dim=0).contiguous() if text_src is not None else None
        task_index_folded = task_index.repeat_interleave(num_search, dim=0).contiguous() if task_index is not None else None

        enc_opt = self.forward_encoder(template_list, [search_folded], template_anno_list, text_src_folded, task_index_folded)
        out_folded = self.forward_decoder(enc_opt)
        task_cls_folded = self.forward_task_decoder(enc_opt)
        cur_idx = 1 if num_search > 1 else 0

        out = {}
        for key, value in out_folded.items():
            if not isinstance(value, torch.Tensor) or value.size(0) != batch_size * num_search:
                out[key] = value
                continue
            out[key] = value.view(batch_size, num_search, *value.shape[1:])[:, cur_idx, ...].contiguous()
        out["task_class"] = task_cls_folded.view(batch_size, num_search, -1)[:, cur_idx, :].contiguous()
        out["task_class_label"] = task_index
        return out

    def _build_post_disambiguator(self):
        if not self.use_startrack:
            return None
        cfg = self.post_disambiguator_cfg
        feat_dim = int(getattr(self.encoder, "num_channels", 512))
        return PostDecoderDisambiguator(
            feat_dim=feat_dim,
            template_feat_dim=feat_dim,
            ratio_thresh=float(getattr(cfg, "RATIO_THRESH", 0.8)) if cfg is not None else 0.8,
            topk_peaks=int(getattr(cfg, "TOPK_PEAKS", 8)) if cfg is not None else 8,
            nms_kernel_size=int(getattr(cfg, "NMS_KERNEL_SIZE", 5)) if cfg is not None else 5,
            multi_peak_ratio_thresh=float(getattr(cfg, "MULTI_PEAK_RATIO_THRESH", 0.45)) if cfg is not None else 0.45,
            gaussian_sigma=float(getattr(cfg, "GAUSSIAN_SIGMA", 2.0)) if cfg is not None else 2.0,
            suppression_strength=float(getattr(cfg, "SUPPRESSION_STRENGTH", 0.6)) if cfg is not None else 0.6,
            history_len=int(getattr(cfg, "HISTORY_LEN", 32)) if cfg is not None else 32,
            use_mamba_history=bool(getattr(cfg, "USE_MAMBA_HISTORY", True)) if cfg is not None else True,
            use_mamba_history_bank=bool(getattr(cfg, "USE_MAMBA_HISTORY_BANK", True)) if cfg is not None else True,
            mamba_d_state=int(getattr(cfg, "MAMBA_D_STATE", 16)) if cfg is not None else 16,
            mamba_expand=int(getattr(cfg, "MAMBA_EXPAND", 2)) if cfg is not None else 2,
            use_template_anchor=False,
            use_first_frame_anchor=False,
            use_history_aware_rerank_score=False,
        )

    @staticmethod
    def _argmax_center(score_map):
        bsz, _, height, width = score_map.shape
        _, flat_idx = torch.max(score_map.flatten(1), dim=1)
        y = torch.div(flat_idx, width, rounding_mode="floor")
        x = flat_idx % width
        return torch.stack([x, y], dim=-1).to(dtype=score_map.dtype)

    @staticmethod
    def _sample_fmap_at_xy(f_map, center_xy):
        if f_map.dim() != 4:
            raise ValueError(f"f_map must be [B, C, H, W], got {tuple(f_map.shape)}")
        if center_xy.dim() != 2 or center_xy.size(-1) != 2:
            raise ValueError(f"center_xy must be [B, 2], got {tuple(center_xy.shape)}")
        bsz, channels, height, width = f_map.shape
        center_xy = center_xy.to(device=f_map.device, dtype=f_map.dtype)
        idx_x = center_xy[:, 0].round().clamp(0, max(width - 1, 0)).long()
        idx_y = center_xy[:, 1].round().clamp(0, max(height - 1, 0)).long()
        flat_idx = (idx_y * width + idx_x).view(bsz, 1, 1).expand(bsz, channels, 1)
        return f_map.flatten(2).gather(dim=2, index=flat_idx).squeeze(-1)


def build_sutrack(cfg):
    encoder = build_encoder(cfg)
    encoder_postprocess = build_encoder_postprocess(cfg, encoder)
    text_encoder = build_textencoder(cfg, encoder) if cfg.DATA.MULTI_MODAL_LANGUAGE else None
    decoder = build_decoder(cfg, encoder)
    task_decoder = build_task_decoder(cfg, encoder)
    post_disamb_cfg = getattr(cfg.MODEL, "POST_DECODER_DISAMBIGUATOR", None)
    model = SUTRACK(
        text_encoder,
        encoder,
        decoder,
        task_decoder,
        encoder_postprocess=encoder_postprocess,
        num_frames=cfg.DATA.SEARCH.NUMBER,
        num_template=cfg.DATA.TEMPLATE.NUMBER,
        decoder_type=cfg.MODEL.DECODER.TYPE,
        task_feature_type=cfg.MODEL.TASK_DECODER.FEATURE_TYPE,
        post_disambiguator_cfg=post_disamb_cfg,
        use_startrack=bool(getattr(cfg.MODEL, "USE_STARTRACK", False)),
    )
    return model
