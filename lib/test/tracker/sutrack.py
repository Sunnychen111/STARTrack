import os

import clip
import cv2
import numpy as np
import torch

from lib.models.sutrack import build_sutrack
from lib.models.sutrack.post_decoder_disambiguator import PostDecoderDisambiguator
from lib.test.tracker.basetracker import BaseTracker
from lib.test.tracker.utils import Preprocessor, sample_target, transform_image_to_crop
from lib.test.utils.hann import hann2d
from lib.utils.box_ops import clip_box


class SUTRACK(BaseTracker):
    """SUTrack test tracker with STARTrack / CRG post-decoder disambiguation.

    Drop-in path:
        lib/test/tracker/sutrack.py

    Key additions:
        1. Supports trainable Candidate Reliability Gate (CRG) in test.
        2. Loads checkpoints with reliability_gate.* keys.
        3. Uses aux_info["should_update"] produced by post_decoder_disambiguator.forward().
        4. Exposes selected_rel_prob / rel_prob_top1 / rel_prob_selected in peak_info.
    """

    def __init__(self, params, dataset_name):
        super(SUTRACK, self).__init__(params)
        self.cfg = params.cfg
        self.use_startrack = self._cfg_bool("USE_STARTRACK", default=False)
        self.startrack_verbose = self._cfg_bool("STARTRACK_VERBOSE", default=False)
        self.startrack_update_count = 0

        network = build_sutrack(params.cfg)
        checkpoint = torch.load(self.params.checkpoint, map_location="cpu", weights_only=False)
        state_dict = checkpoint["net"] if isinstance(checkpoint, dict) and "net" in checkpoint else checkpoint
        self._load_base_checkpoint(network, state_dict)

        self.network = network.cuda()
        self.network.eval()
        self._setup_startrack()

        self.preprocessor = Preprocessor()
        self.state = None
        self.fx_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.ENCODER.STRIDE
        if self.cfg.TEST.WINDOW:
            self.output_window = hann2d(torch.tensor([self.fx_sz, self.fx_sz]).long(), centered=True).cuda()

        self.num_template = self.cfg.TEST.NUM_TEMPLATES
        self.debug = params.debug
        self.frame_id = 0

        dataset_name = dataset_name.upper()
        self.update_intervals = self._dataset_value(self.cfg.TEST.UPDATE_INTERVALS, dataset_name)
        self.update_threshold = self._dataset_value(self.cfg.TEST.UPDATE_THRESHOLD, dataset_name)
        self.multi_modal_vision = self._dataset_value(self.cfg.TEST.MULTI_MODAL_VISION, dataset_name)
        self.multi_modal_language = self._dataset_value(self.cfg.TEST.MULTI_MODAL_LANGUAGE, dataset_name)
        if self.multi_modal_language and getattr(self.network, "text_encoder", None) is None:
            print("Warning: TEST.MULTI_MODAL_LANGUAGE=True but model has no text_encoder. Disable language branch.")
            self.multi_modal_language = False
        self.use_nlp = self._dataset_value(self.cfg.TEST.USE_NLP, dataset_name)
        self.task_index_batch = None

    def _load_base_checkpoint(self, network, state_dict):
        if self.use_startrack and isinstance(state_dict, dict):
            state_dict = {
                k: v for k, v in state_dict.items()
                if not (k.startswith("post_disambiguator.") or k.startswith("module.post_disambiguator."))
            }
        try:
            network.load_state_dict(state_dict, strict=True)
            return
        except RuntimeError:
            if not self.use_startrack:
                raise

        missing_keys, unexpected_keys = network.load_state_dict(state_dict, strict=False)
        non_startrack_missing = [k for k in missing_keys if not k.startswith("post_disambiguator.")]
        if non_startrack_missing:
            raise RuntimeError(
                "Base checkpoint is missing non-STARTrack keys: "
                f"{non_startrack_missing[:20]}"
            )
        print(
            "[STARTrack] base checkpoint loaded with strict=False "
            f"(missing={len(missing_keys)}, unexpected={len(unexpected_keys)})"
        )

    def _cfg_value(self, key, default=None):
        model_cfg = getattr(self.cfg, "MODEL", None)
        test_cfg = getattr(self.cfg, "TEST", None)
        if model_cfg is not None and hasattr(model_cfg, key):
            return getattr(model_cfg, key)
        if test_cfg is not None and hasattr(test_cfg, key):
            return getattr(test_cfg, key)
        return default

    def _cfg_bool(self, key, default=False):
        return bool(self._cfg_value(key, default))

    @staticmethod
    def _dataset_value(table, dataset_name):
        if "GOT10K" in dataset_name:
            dataset_name = "GOT10K"
        elif "LASOT" in dataset_name:
            dataset_name = "LASOT"
        elif "OTB" in dataset_name:
            dataset_name = "TNL2K"
        if hasattr(table, dataset_name):
            return getattr(table, dataset_name)
        return table.DEFAULT

    def _build_startrack_module(self):
        feat_dim = int(getattr(self.network.encoder, "num_channels", 512))
        use_trainable_reliability = bool(
            self._cfg_value("STARTRACK_USE_TRAINABLE_RELIABILITY", True)
        )
        reliability_hidden_dim = int(
            self._cfg_value("STARTRACK_RELIABILITY_HIDDEN_DIM", 128)
        )

        return PostDecoderDisambiguator(
            feat_dim=feat_dim,
            template_feat_dim=feat_dim,
            topk_peaks=int(self._cfg_value("STARTRACK_TOPK", 8)),
            history_len=int(self._cfg_value("STARTRACK_HISTORY_LEN", 32)),
            use_mamba_history=True,
            use_mamba_history_bank=True,
            use_template_anchor=False,
            use_first_frame_anchor=False,
            use_history_aware_rerank_score=False,
            use_trainable_reliability=use_trainable_reliability,
            reliability_hidden_dim=reliability_hidden_dim,
        )

    @staticmethod
    def _extract_disambiguator_state(checkpoint):
        """Accept offline checkpoints, Stage4 checkpoints, and raw state dicts."""
        if isinstance(checkpoint, dict) and "model" in checkpoint and isinstance(checkpoint["model"], dict):
            state = checkpoint["model"]
        elif isinstance(checkpoint, dict) and "disambiguator" in checkpoint and isinstance(checkpoint["disambiguator"], dict):
            state = checkpoint["disambiguator"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            state = checkpoint["state_dict"]
        else:
            state = checkpoint

        if not isinstance(state, dict):
            return state

        if any(k.startswith("post_disambiguator.") for k in state.keys()):
            state = {
                k[len("post_disambiguator."):]: v
                for k, v in state.items()
                if k.startswith("post_disambiguator.")
            }
        elif any(k.startswith("module.post_disambiguator.") for k in state.keys()):
            prefix = "module.post_disambiguator."
            state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        elif any(k.startswith("module.") for k in state.keys()):
            state = {k[len("module."):]: v for k, v in state.items() if k.startswith("module.")}
        return state

    def _setup_startrack(self):
        if not self.use_startrack:
            self.network.use_startrack = False
            return

        print("[STARTrack] enabled")
        self.network.use_startrack = True

        if getattr(self.network, "post_disambiguator", None) is None:
            self.network.post_disambiguator = self._build_startrack_module().to(
                device=next(self.network.parameters()).device
            )

        ckpt_path = str(self._cfg_value("STARTRACK_CKPT", "checkpoints/startrack_mamba_diff_full/last.pth"))
        ckpt_path = os.path.expanduser(ckpt_path)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"[STARTrack] STARTRACK_CKPT not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = self._extract_disambiguator_state(checkpoint)

        reliability_keys = []
        if isinstance(state, dict):
            reliability_keys = [k for k in state.keys() if k.startswith("reliability_gate.")]

        missing, unexpected = self.network.post_disambiguator.load_state_dict(state, strict=False)
        print("[STARTrack] checkpoint loaded:", ckpt_path)
        print(f"[STARTrack] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

        post = self.network.post_disambiguator

        post.use_trainable_reliability = bool(
            self._cfg_value("STARTRACK_USE_TRAINABLE_RELIABILITY", True)
        )
        if post.use_trainable_reliability and len(reliability_keys) == 0:
            print(
                "[STARTrack][WARN] STARTRACK_USE_TRAINABLE_RELIABILITY=True but "
                "checkpoint has no reliability_gate.* keys. Auto-disable trainable CRG "
                "to avoid using randomly initialized reliability_gate."
            )
            post.use_trainable_reliability = False

        post.use_heuristic_reliability = bool(
            self._cfg_value("STARTRACK_USE_HEURISTIC_RELIABILITY", not post.use_trainable_reliability)
        )

        rel_update_default = self._cfg_value(
            "STARTRACK_REL_UPDATE_THRESH",
            self._cfg_value("STARTRACK_RELIABILITY_UPDATE_THRESH", 0.50),
        )
        post.reliability_update_thresh = float(rel_update_default)

        self.startrack_update_ratio_thresh = float(self._cfg_value("STARTRACK_UPDATE_RATIO_THRESH", 0.90))
        self.startrack_target_prob_thresh = float(self._cfg_value("STARTRACK_TARGET_PROB_THRESH", 0.50))
        self.startrack_min_id_margin = float(self._cfg_value("STARTRACK_MIN_ID_MARGIN", 0.00))
        post.update_ratio_thresh = self.startrack_update_ratio_thresh
        post.target_prob_thresh = self.startrack_target_prob_thresh
        post.min_id_margin = self.startrack_min_id_margin

        post.rel_target_weight = float(self._cfg_value("STARTRACK_REL_TARGET_WEIGHT", getattr(post, "rel_target_weight", 0.25)))
        post.rel_score_weight = float(self._cfg_value("STARTRACK_REL_SCORE_WEIGHT", getattr(post, "rel_score_weight", 0.20)))
        post.rel_anchor_weight = float(self._cfg_value("STARTRACK_REL_ANCHOR_WEIGHT", getattr(post, "rel_anchor_weight", 0.25)))
        post.rel_history_weight = float(self._cfg_value("STARTRACK_REL_HISTORY_WEIGHT", getattr(post, "rel_history_weight", 0.15)))
        post.rel_margin_weight = float(self._cfg_value("STARTRACK_REL_MARGIN_WEIGHT", getattr(post, "rel_margin_weight", 0.10)))
        post.rel_ambiguity_weight = float(self._cfg_value("STARTRACK_REL_AMBIGUITY_WEIGHT", getattr(post, "rel_ambiguity_weight", 0.05)))

        print(
            "[STARTrack] reliability: "
            f"trainable={bool(post.use_trainable_reliability)}, "
            f"loaded_reliability_keys={len(reliability_keys)}, "
            f"heuristic={bool(post.use_heuristic_reliability)}, "
            f"rel_update_thresh={float(post.reliability_update_thresh):.3f}"
        )

        post.eval()
        for p in post.parameters():
            p.requires_grad_(False)

    def _reset_startrack_memory(self):
        if not self.use_startrack or getattr(self.network, "post_disambiguator", None) is None:
            return
        self.network.post_disambiguator.reset_history(batch_size=1)
        self.startrack_update_count = 0

    @staticmethod
    def _aux_bool(aux_info, key, default=False):
        value = aux_info.get(key, default)
        if isinstance(value, torch.Tensor):
            return bool(value.detach().reshape(-1)[0].item()) if value.numel() > 0 else default
        return bool(value)

    @staticmethod
    def _aux_float(aux_info, key, default=0.0):
        value = aux_info.get(key, default)
        if isinstance(value, torch.Tensor):
            return float(value.detach().reshape(-1)[0].item()) if value.numel() > 0 else default
        return float(value)

    @staticmethod
    def _aux_int(aux_info, key, default=-1):
        value = aux_info.get(key, default)
        if isinstance(value, torch.Tensor):
            return int(value.detach().reshape(-1)[0].item()) if value.numel() > 0 else default
        return int(value)

    def _aux_rel_top1(self, aux_info, default=0.0):
        rel = aux_info.get("rel_prob_all", None)
        if isinstance(rel, torch.Tensor) and rel.numel() > 0:
            return float(rel.detach().reshape(rel.shape[0], -1)[0, 0].item())
        return float(default)

    def _maybe_update_startrack_history(self, aux_info):
        if not self.use_startrack or not isinstance(aux_info, dict):
            return False
        target_feat = aux_info.get("target_feat", None)
        if not isinstance(target_feat, torch.Tensor):
            return False

        if "should_update" in aux_info:
            should_update = self._aux_bool(aux_info, "should_update", default=False)
        else:
            selected_idx = aux_info.get("selected_idx", aux_info.get("rerank_idx", None))
            valid_idx = isinstance(selected_idx, torch.Tensor) and selected_idx.numel() > 0
            should_update = (
                valid_idx
                and self._aux_float(aux_info, "target_prob", 0.0) >= self.startrack_target_prob_thresh
                and self._aux_float(aux_info, "identity_margin", 0.0) >= self.startrack_min_id_margin
                and self._aux_float(aux_info, "ambiguity_ratio", 1.0) <= self.startrack_update_ratio_thresh
            )
        if not should_update:
            return False

        with torch.no_grad():
            self.network.post_disambiguator.update_history(target_feat)
        self.startrack_update_count += 1
        return True

    def _get_startrack_fmap(self, out_dict, enc_opt):
        if isinstance(out_dict, dict) and isinstance(out_dict.get("f_map", None), torch.Tensor):
            return out_dict["f_map"]
        if enc_opt is None:
            return None

        tokens = enc_opt[0] if isinstance(enc_opt, (list, tuple)) else enc_opt
        num_search = getattr(self.network, "runtime_num_frames", 1)
        _, search_start, search_end, _ = self.network._get_token_layout(
            tokens,
            num_search=num_search,
            num_template=len(self.template_list),
        )
        search_tokens = tokens[:, search_start:search_end, :]
        if num_search > 1:
            search_tokens = search_tokens.view(tokens.size(0), num_search, self.network.num_patch_x, tokens.size(-1))
            search_tokens = search_tokens[:, -1, :, :]
        return search_tokens.transpose(1, 2).contiguous().view(
            tokens.size(0), tokens.size(-1), self.network.fx_sz, self.network.fx_sz
        )

    def _apply_startrack_rerank(self, out_dict, enc_opt=None, prev_center=None):
        if not self.use_startrack:
            return out_dict
        if "startrack_aux" in out_dict:
            return out_dict
        f_map = self._get_startrack_fmap(out_dict, enc_opt)
        if f_map is None:
            return out_dict

        with torch.no_grad():
            refined_score_map, aux_info = self.network.post_disambiguator(
                score_map=out_dict["score_map"],
                feat_map=f_map,
                history_tokens=None,
                prev_center=prev_center,
            )

        out_dict["score_map_raw"] = out_dict["score_map"]
        out_dict["score_map"] = refined_score_map
        out_dict["f_map"] = f_map
        out_dict["startrack_aux"] = aux_info
        return out_dict

    def _prev_center_in_search_feature(self, resize_factor, feat_w, feat_h, device, dtype):
        if self.state is None or resize_factor <= 0:
            return torch.tensor([[0.5 * (feat_w - 1), 0.5 * (feat_h - 1)]], device=device, dtype=dtype)
        cx_prev = float(self.state[0]) + 0.5 * float(self.state[2])
        cy_prev = float(self.state[1]) + 0.5 * float(self.state[3])
        crop_side = float(self.params.search_size) / float(resize_factor)
        crop_x0 = cx_prev - 0.5 * crop_side
        crop_y0 = cy_prev - 0.5 * crop_side
        cx_crop = (cx_prev - crop_x0) * float(resize_factor)
        cy_crop = (cy_prev - crop_y0) * float(resize_factor)
        scale = float(max(int(self.params.search_size) - 1, 1))
        cx_feat = np.clip(cx_crop / scale * float(max(int(feat_w) - 1, 0)), 0.0, float(max(int(feat_w) - 1, 0)))
        cy_feat = np.clip(cy_crop / scale * float(max(int(feat_h) - 1, 0)), 0.0, float(max(int(feat_h) - 1, 0)))
        return torch.tensor([[cx_feat, cy_feat]], device=device, dtype=dtype)

    def initialize(self, image, info: dict):
        z_patch_arr, resize_factor = sample_target(
            image, info["init_bbox"], self.params.template_factor, output_sz=self.params.template_size
        )
        template = self.preprocessor.process(z_patch_arr)
        if self.multi_modal_vision and template.size(1) == 3:
            template = torch.cat((template, template), axis=1)
        self.template_list = [template] * self.num_template

        if hasattr(self.network, "clear_online_state"):
            self.network.clear_online_state()
        if hasattr(self.network, "reset_online_state"):
            self.network.reset_online_state(batch_size=template.size(0), device=template.device, dtype=template.dtype)
        self._reset_startrack_memory()

        self.state = info["init_bbox"]
        prev_box_crop = transform_image_to_crop(
            torch.tensor(info["init_bbox"]),
            torch.tensor(info["init_bbox"]),
            resize_factor,
            torch.Tensor([self.params.template_size, self.params.template_size]),
            normalize=True,
        )
        init_template_anno = prev_box_crop.to(template.device).unsqueeze(0)
        self.template_anno_list = [init_template_anno.clone() for _ in range(self.num_template)]
        self.frame_id = 0

        if self.multi_modal_language and getattr(self.network, "text_encoder", None) is not None:
            init_nlp = info.get("init_nlp") if self.use_nlp else None
            text_data, _ = self.extract_token_from_nlp_clip(init_nlp)
            text_data = text_data.unsqueeze(0).to(template.device)
            with torch.no_grad():
                self.text_src = self.network.forward_textencoder(text_data=text_data)
        else:
            self.text_src = None

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor = sample_target(
            image, self.state, self.params.search_factor, output_sz=self.params.search_size
        )
        search = self.preprocessor.process(x_patch_arr)
        if self.multi_modal_vision and search.size(1) == 3:
            search = torch.cat((search, search), axis=1)

        with torch.no_grad():
            enc_opt = self.network.forward_encoder(
                self.template_list,
                [search],
                self.template_anno_list,
                self.text_src,
                self.task_index_batch,
            )
            prev_center = None
            if self.use_startrack:
                prev_center = self._prev_center_in_search_feature(
                    resize_factor=resize_factor,
                    feat_w=self.fx_sz,
                    feat_h=self.fx_sz,
                    device=search.device,
                    dtype=search.dtype,
                )
            out_dict = self.network.forward_decoder(feature=enc_opt, prev_center=prev_center)
            out_dict = self._apply_startrack_rerank(out_dict, enc_opt=enc_opt, prev_center=prev_center)

        pred_score_map = out_dict["score_map"]
        response = self.output_window * pred_score_map if self.cfg.TEST.WINDOW else pred_score_map
        if "size_map" in out_dict:
            pred_boxes, conf_score = self.network.decoder.cal_bbox(
                response, out_dict["size_map"], out_dict["offset_map"], return_score=True
            )
        else:
            pred_boxes, conf_score = self.network.decoder.cal_bbox(
                response, out_dict["offset_map"], return_score=True
            )

        pred_boxes = pred_boxes.view(-1, 4)
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        aux_info = out_dict.get("startrack_aux", None)
        should_update_history = self._maybe_update_startrack_history(aux_info)

        if self.num_template > 1:
            conf_value = conf_score.item() if isinstance(conf_score, torch.Tensor) else float(conf_score)
            if (self.frame_id % self.update_intervals == 0) and (conf_value > self.update_threshold):
                z_patch_arr, resize_factor = sample_target(
                    image, self.state, self.params.template_factor, output_sz=self.params.template_size
                )
                template = self.preprocessor.process(z_patch_arr)
                if self.multi_modal_vision and template.size(1) == 3:
                    template = torch.cat((template, template), axis=1)
                self.template_list.append(template)
                if len(self.template_list) > self.num_template:
                    self.template_list.pop(1)

                prev_box_crop = transform_image_to_crop(
                    torch.tensor(self.state),
                    torch.tensor(self.state),
                    resize_factor,
                    torch.Tensor([self.params.template_size, self.params.template_size]),
                    normalize=True,
                )
                self.template_anno_list.append(prev_box_crop.to(template.device).unsqueeze(0))
                if len(self.template_anno_list) > self.num_template:
                    self.template_anno_list.pop(1)

        if self.debug == 1:
            image_show = image[:, :, :3] if image.shape[-1] == 6 else image
            x1, y1, w, h = self.state
            image_bgr = cv2.cvtColor(image_show, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_bgr, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
            cv2.imshow("vis", image_bgr)
            cv2.waitKey(1)

        peak_info = {}
        if isinstance(aux_info, dict):
            selected_idx = self._aux_int(aux_info, "selected_idx", -1)
            selected_rel_prob = self._aux_float(aux_info, "selected_rel_prob", self._aux_float(aux_info, "reliability_score", 0.0))
            peak_info = {
                "rerank_used": self._aux_bool(aux_info, "rerank_used", False),
                "should_update": self._aux_bool(aux_info, "should_update", False),
                "should_update_history": bool(should_update_history),
                "base_should_update": self._aux_bool(aux_info, "base_should_update", False),
                "trainable_reliability_used": self._aux_bool(aux_info, "trainable_reliability_used", False),
                "selected_idx": selected_idx,
                "rerank_idx": self._aux_int(aux_info, "rerank_idx", selected_idx),
                "target_prob": self._aux_float(aux_info, "target_prob", 0.0),
                "identity_margin": self._aux_float(aux_info, "identity_margin", 0.0),
                "ambiguity_ratio": self._aux_float(aux_info, "ambiguity_ratio", 0.0),
                "reliability_score": self._aux_float(aux_info, "reliability_score", selected_rel_prob),
                "selected_rel_prob": selected_rel_prob,
                "rel_prob_top1": self._aux_rel_top1(aux_info, default=0.0),
                "rel_prob_selected": selected_rel_prob,
                "selected_score_prob": self._aux_float(aux_info, "selected_score_prob", 0.0),
                "ambiguity_conf": self._aux_float(aux_info, "ambiguity_conf", 0.0),
                "effective_peak_count": self._aux_int(aux_info, "effective_peak_count", -1),
                "is_ambiguous": self._aux_bool(aux_info, "is_ambiguous", False),
                "history_len": int(
                    0 if getattr(self.network.post_disambiguator.history_bank, "cached_tokens", None) is None
                    else self.network.post_disambiguator.history_bank.cached_tokens.size(1)
                ) if self.use_startrack and getattr(self.network, "post_disambiguator", None) is not None else -1,
                "update_history_count": self.startrack_update_count,
            }

        conf_score_val = conf_score.item() if isinstance(conf_score, torch.Tensor) else float(conf_score)
        return {
            "target_bbox": self.state,
            "best_score": conf_score_val,
            "is_soi": bool(peak_info.get("rerank_used", False)),
            "peak_info": peak_info,
            "score_map": out_dict.get("score_map", None),
            "score_map_raw": out_dict.get("score_map_raw", None),
        }

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def extract_token_from_nlp_clip(self, nlp):
        if nlp is None:
            nlp_ids = torch.zeros(77, dtype=torch.long)
            nlp_masks = torch.zeros(77, dtype=torch.long)
        else:
            nlp_ids = clip.tokenize(nlp).squeeze(0)
            nlp_masks = (nlp_ids == 0).long()
        return nlp_ids, nlp_masks


def get_tracker_class():
    return SUTRACK
