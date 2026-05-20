from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, box_xyxy_to_cxcywh, box_cxcywh_to_xyxy, box_iou
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.train.admin import multigpu
from lib.utils.heapmap_utils import generate_heatmap


def _set_module_state(module, requires_grad, training):
    if module is None:
        return
    for parameter in module.parameters():
        parameter.requires_grad_(requires_grad)
    if training:
        module.train()
    else:
        module.eval()


def _find_decoder_blocks(decoder_module):
    if decoder_module is None:
        return None
    for attr_name in ("blocks", "layers", "decoder_blocks", "transformer_blocks", "stages"):
        if not hasattr(decoder_module, attr_name):
            continue
        container = getattr(decoder_module, attr_name)
        if isinstance(container, (nn.ModuleList, nn.Sequential, list, tuple)):
            return list(container)
    return None


def summarize_trainable_parameters(model):
    module = model.module if hasattr(model, "module") else model
    trainable_params = 0
    total_params = 0

    print("Trainable parameters:")
    for name, parameter in module.named_parameters():
        numel = int(parameter.numel())
        total_params += numel
        if parameter.requires_grad:
            trainable_params += numel
            print(name)

    print(
        f"Trainable params: {trainable_params / 1e6:.3f}M / "
        f"{total_params / 1e6:.3f}M"
    )
    return trainable_params, total_params


def set_trainable_state(model, stage, decoder_last_n_blocks=2, stage3_freeze_sutrack=True):
    if stage not in (1, 2, 3):
        raise ValueError(f"Unsupported stage={stage}, expected 1/2/3.")

    encoder = getattr(model, "encoder", None)
    decoder = getattr(model, "decoder", None)
    head = getattr(model, "head", None)
    task_decoder = getattr(model, "task_decoder", None)
    text_encoder = getattr(model, "text_encoder", None)
    encoder_postprocess = getattr(model, "encoder_postprocess", None)
    post_disambiguator = getattr(model, "post_disambiguator", None)

    _set_module_state(encoder_postprocess, requires_grad=False, training=False)
    _set_module_state(text_encoder, requires_grad=False, training=False)

    if stage in (1, 2):
        _set_module_state(encoder, requires_grad=False, training=False)
        _set_module_state(decoder, requires_grad=False, training=False)
        _set_module_state(head, requires_grad=False, training=False)
        if task_decoder is not head:
            _set_module_state(task_decoder, requires_grad=False, training=False)
        _set_module_state(post_disambiguator, requires_grad=True, training=True)
        return

    _set_module_state(encoder, requires_grad=False, training=False)
    _set_module_state(decoder, requires_grad=False, training=False)

    if stage3_freeze_sutrack:
        _set_module_state(head, requires_grad=False, training=False)
        if task_decoder is not head:
            _set_module_state(task_decoder, requires_grad=False, training=False)
        _set_module_state(post_disambiguator, requires_grad=True, training=True)
        return

    decoder_blocks = _find_decoder_blocks(decoder)
    if decoder_blocks is None:
        _set_module_state(decoder, requires_grad=True, training=True)
    else:
        n_blocks = min(max(int(decoder_last_n_blocks), 1), len(decoder_blocks))
        for block in decoder_blocks[-n_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
            block.train()
        decoder.train()

    _set_module_state(head, requires_grad=True, training=True)
    if task_decoder is not head:
        _set_module_state(task_decoder, requires_grad=True, training=True)
    _set_module_state(post_disambiguator, requires_grad=True, training=True)


def build_connected_zero_loss(model, fallback_tensor=None):
    module = model.module if hasattr(model, "module") else model
    post = getattr(module, "post_disambiguator", None)

    zero_terms = []
    if post is not None:
        for parameter in post.parameters():
            if parameter.requires_grad:
                zero_terms.append(parameter.sum() * 0.0)

    if len(zero_terms) > 0:
        return torch.stack(zero_terms).sum()

    if fallback_tensor is not None and isinstance(fallback_tensor, torch.Tensor):
        return fallback_tensor.sum() * 0.0

    raise RuntimeError("Cannot build connected zero loss: no trainable post_disambiguator parameters found.")


def build_total_loss_by_stage(stage, losses, lambda_gate=0.1, use_track_loss_stage2=False,
                              stage3_freeze_sutrack=True, model=None, fallback_tensor=None):
    """
    Stage-aware total loss composition.

    Args:
        stage: int, training stage
        losses: dict that may include:
            - loss_track or loss_total_original
            - loss_giou, loss_l1, loss_location
            - loss_gate
        lambda_gate: weight for gate loss in stage >= 2
        use_track_loss_stage2: whether to keep tracking loss in stage 2

    Returns:
        total_loss: Tensor or None
        log_vars: dict
        skip_backward: bool
    """
    log_vars = {}

    def _to_float(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            if x.numel() == 0:
                return None
            return float(x.detach().mean().item())
        try:
            return float(x)
        except Exception:
            return None

    def _is_valid(x):
        if x is None:
            return False
        if isinstance(x, torch.Tensor):
            if x.numel() == 0:
                return False
            return bool(torch.isfinite(x).all().item())
        try:
            value = float(x)
            return value == value
        except Exception:
            return False

    def _to_tensor(x, ref=None):
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(ref, torch.Tensor):
            return torch.tensor(float(x), dtype=ref.dtype, device=ref.device)
        return torch.tensor(float(x), dtype=torch.float32)

    def _zero_loss():
        if model is None:
            if isinstance(fallback_tensor, torch.Tensor):
                return fallback_tensor.sum() * 0.0
            raise RuntimeError("Cannot build connected zero loss without model or fallback_tensor.")
        return build_connected_zero_loss(model, fallback_tensor=fallback_tensor)

    loss_track = losses.get("loss_track", None)
    if loss_track is None:
        loss_track = losses.get("loss_total_original", None)
    loss_gate = losses.get("loss_gate", None)

    gate_valid = _is_valid(loss_gate)
    valid_gate_count = losses.get("num_valid_gate", None)
    if valid_gate_count is not None:
        try:
            gate_valid = gate_valid and (float(valid_gate_count) > 0)
        except Exception:
            pass

    for key in ("loss_track", "loss_total_original", "loss_giou", "loss_l1", "loss_location", "loss_gate"):
        if key in losses:
            value = _to_float(losses[key])
            if value is not None:
                log_vars["Loss/" + key.replace("loss_", "")] = value

    log_vars["Stage/id"] = float(stage)
    log_vars["Stage/lambda_gate"] = float(lambda_gate)
    log_vars["Stage/use_track_loss_stage2"] = 1.0 if use_track_loss_stage2 else 0.0
    log_vars["Stage/stage3_freeze_sutrack"] = 1.0 if stage3_freeze_sutrack else 0.0
    log_vars["Stage/has_valid_gate"] = 1.0 if gate_valid else 0.0

    if stage == 1:
        if not gate_valid:
            total_loss = _zero_loss()
            log_vars["Loss/total"] = _to_float(total_loss)
            log_vars["Meta/skip_backward"] = 1.0
            return total_loss, log_vars, True
        total_loss = _to_tensor(loss_gate, ref=loss_track)
        log_vars["Loss/total"] = _to_float(total_loss)
        log_vars["Meta/skip_backward"] = 0.0
        return total_loss, log_vars, False

    gate_term = None
    if gate_valid:
        gate_term = float(lambda_gate) * _to_tensor(loss_gate, ref=loss_track)

    if stage == 2:
        if use_track_loss_stage2:
            if _is_valid(loss_track):
                total_loss = _to_tensor(loss_track) + (gate_term if gate_term is not None else 0.0)
                log_vars["Loss/total"] = _to_float(total_loss)
                log_vars["Meta/skip_backward"] = 0.0
                return total_loss, log_vars, False
            if gate_term is None:
                total_loss = _zero_loss()
                log_vars["Loss/total"] = _to_float(total_loss)
                log_vars["Meta/skip_backward"] = 1.0
                return total_loss, log_vars, True
            total_loss = gate_term
            log_vars["Loss/total"] = _to_float(total_loss)
            log_vars["Meta/skip_backward"] = 0.0
            return total_loss, log_vars, False

        if gate_term is None:
            total_loss = _zero_loss()
            log_vars["Loss/total"] = _to_float(total_loss)
            log_vars["Meta/skip_backward"] = 1.0
            return total_loss, log_vars, True
        total_loss = gate_term
        log_vars["Loss/total"] = _to_float(total_loss)
        log_vars["Meta/skip_backward"] = 0.0
        return total_loss, log_vars, False

    if stage == 3 and stage3_freeze_sutrack:
        if gate_term is None:
            total_loss = _zero_loss()
            log_vars["Loss/total"] = _to_float(total_loss)
            log_vars["Meta/skip_backward"] = 1.0
            return total_loss, log_vars, True
        total_loss = gate_term
        log_vars["Loss/total"] = _to_float(total_loss)
        log_vars["Meta/skip_backward"] = 0.0
        return total_loss, log_vars, False

    # stage >= 3 and SUTrack unfreezing is explicitly enabled.
    has_track = _is_valid(loss_track)
    if (not has_track) and (gate_term is None):
        total_loss = _zero_loss()
        log_vars["Loss/total"] = _to_float(total_loss)
        log_vars["Meta/skip_backward"] = 1.0
        return total_loss, log_vars, True

    if has_track:
        total_loss = _to_tensor(loss_track) + (gate_term if gate_term is not None else 0.0)
    else:
        total_loss = gate_term
        log_vars["Meta/warning_track_missing"] = 1.0

    log_vars["Loss/total"] = _to_float(total_loss)
    log_vars["Meta/skip_backward"] = 0.0
    return total_loss, log_vars, False


class SUTrack_Actor(BaseActor):
    """ Actor for training the sutrack"""
    def __init__(self, net, objective, loss_weight, settings, cfg):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg
        self.multi_modal_language = cfg.DATA.MULTI_MODAL_LANGUAGE
        self._last_train_stage = None
        self.lambda_gate = float(getattr(getattr(cfg.TRAIN, 'THREE_STAGE', None), 'LAMBDA_GATE', 0.1))
        self.use_track_loss_stage2 = bool(
            getattr(getattr(cfg.TRAIN, 'THREE_STAGE', None), 'USE_TRACK_LOSS_STAGE2', False)
        )
        self.stage3_freeze_sutrack = bool(
            getattr(getattr(cfg.TRAIN, 'THREE_STAGE', None), 'STAGE3_FREEZE_SUTRACK', True)
        )
        self._last_trainable_params_million = 0.0

    def _unwrap_model(self):
        return self.net.module if multigpu.is_multi_gpu(self.net) else self.net

    def _resolve_train_stage(self, epoch):
        epoch = int(epoch)
        three_stage_cfg = getattr(self.cfg.TRAIN, 'THREE_STAGE', None)
        if (three_stage_cfg is not None) and bool(getattr(three_stage_cfg, 'ENABLED', False)):
            stage1_epochs = int(getattr(three_stage_cfg, 'STAGE1_EPOCHS', 0))
            stage2_epochs = int(getattr(three_stage_cfg, 'STAGE2_EPOCHS', 0))
            if epoch <= stage1_epochs:
                return 1
            if epoch <= (stage1_epochs + stage2_epochs):
                return 2
            return 3

        # Backward compatible fallback: old TWO_STAGE maps to stage1 + stage3.
        two_stage_cfg = getattr(self.cfg.TRAIN, 'TWO_STAGE', None)
        if (two_stage_cfg is None) or (not getattr(two_stage_cfg, 'ENABLED', False)):
            return None
        stage1_epochs = int(getattr(two_stage_cfg, 'STAGE1_EPOCHS', 0))
        return 1 if epoch <= stage1_epochs else 3

    def _apply_trainable_control(self, epoch):
        stage = self._resolve_train_stage(epoch)
        if stage is None:
            return

        model = self._unwrap_model()
        three_stage_cfg = getattr(self.cfg.TRAIN, 'THREE_STAGE', None)
        decoder_last_n_blocks = int(getattr(three_stage_cfg, 'DECODER_LAST_N_BLOCKS', 2)) if three_stage_cfg is not None else 2
        stage3_freeze_sutrack = bool(getattr(three_stage_cfg, "STAGE3_FREEZE_SUTRACK", True))
        set_trainable_state(
            model,
            stage=stage,
            decoder_last_n_blocks=decoder_last_n_blocks,
            stage3_freeze_sutrack=stage3_freeze_sutrack,
        )
        self.stage3_freeze_sutrack = stage3_freeze_sutrack
        if self._last_train_stage != stage:
            trainable_params, _ = summarize_trainable_parameters(model)
            self._last_trainable_params_million = trainable_params / 1e6
        self._last_train_stage = stage

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'search_anno'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        try:
            epoch = data['epoch']
        except KeyError:
            epoch = 1
        self._apply_trainable_control(epoch)
        stage = self._resolve_train_stage(epoch)

        # Stage-aware forward switch:
        # stage >= 2 enables post-decoder training, but training bbox decode
        # stays on baseline/selected peaks rather than dense refined maps.
        model_unwrapped = self._unwrap_model()
        if stage is not None and stage >= 2:
            model_unwrapped.use_refined_for_bbox = True
        else:
            model_unwrapped.use_refined_for_bbox = False

        # Convert to batch-major tensors for forward_train:
        # template_images: [N_t, B, C, H, W] -> [B, N_t, C, H, W]
        # search_images:   [N_s, B, C, H, W] -> [B, N_s, C, H, W]
        template_batch = data['template_images'].permute(1, 0, 2, 3, 4).contiguous()
        search_batch = data['search_images'].permute(1, 0, 2, 3, 4).contiguous()
        template_anno_batch = data['template_anno'].permute(1, 0, 2).contiguous()
        search_anno_batch = data['search_anno'].permute(1, 0, 2).contiguous()

        if self.multi_modal_language:
            text = data['nlp_ids'].permute (1,0)
            text_src = self.net(text_data=text, mode='text')
        else:
            text_src = None

        # task_class
        task_index_batch = [self.cfg.MODEL.TASK_INDEX[key.upper()] for key in data['dataset']]
        task_index_batch = torch.tensor(task_index_batch).cuda() #torch.Size([bs])

        outputs = self.net(
            template_list=template_batch,
            search_list=search_batch,
            template_anno_list=template_anno_batch,
            search_anno_list=search_anno_batch,
            text_src=text_src,
            task_index=task_index_batch,
            mode='train'
        )
        return outputs

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # task classification loss
        task_cls_loss = self.objective['task_cls'](pred_dict['task_class'], pred_dict['task_class_label'])

        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.ENCODER.STRIDE) # list of torch.Size([b, H, W])
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1) # torch.Size([b, 1, H, W])

        # Get boxes
        pred_boxes = pred_dict['pred_boxes'] # torch.Size([b, 1, 4])
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)

        # baseline tracking loss
        loss_track = (
            self.loss_weight['giou'] * giou_loss +
            self.loss_weight['l1'] * l1_loss +
            self.loss_weight['focal'] * location_loss +
            self.loss_weight['task_cls'] * task_cls_loss
        )

        gate_debug = {}

        # Top-K gate/reranker loss with GT-center supervision.
        loss_gate = pred_dict.get('loss_gate', None)
        if loss_gate is None:
            gate_logits = pred_dict.get('target_logits', pred_dict.get('gate_logits', None))
            aux_info = pred_dict.get('disamb_aux', None)
            if (gate_logits is not None) and (aux_info is not None):
                peaks_xy = aux_info.get('peaks_xy', None)      # [B, K, 2]
                is_ambig = aux_info.get('is_ambiguous', None)  # [B]
                if (peaks_xy is not None) and (is_ambig is not None):
                    gate_logits = gate_logits.to(device=gt_bbox.device)
                    peaks_xy = peaks_xy.to(device=gt_bbox.device, dtype=gt_bbox.dtype)
                    is_ambig = is_ambig.to(device=gt_bbox.device).bool()
                    k_peaks = int(gate_logits.size(-1))

                    # map GT center from [0,1] to decoder feature-map coordinates
                    fx_sz = float(pred_dict['score_map'].size(-1))
                    fy_sz = float(pred_dict['score_map'].size(-2))
                    cx = (gt_bbox[:, 0] + 0.5 * gt_bbox[:, 2]) * fx_sz
                    cy = (gt_bbox[:, 1] + 0.5 * gt_bbox[:, 3]) * fy_sz
                    cx = cx.clamp(min=0.0, max=max(fx_sz - 1.0, 0.0))
                    cy = cy.clamp(min=0.0, max=max(fy_sz - 1.0, 0.0))
                    gt_xy = torch.stack([cx, cy], dim=-1)  # [B, 2]

                    dists = torch.norm(peaks_xy - gt_xy[:, None, :], dim=-1)  # [B, K]
                    target_peak_idx = torch.argmin(dists, dim=-1).long()
                    if dists.size(1) > 1:
                        sorted_dists = torch.sort(dists, dim=-1).values
                        dist_gap = sorted_dists[:, 1] - sorted_dists[:, 0]
                    else:
                        dist_gap = torch.zeros_like(target_peak_idx, dtype=gt_bbox.dtype)

                    min_gap = float(getattr(getattr(self.cfg.MODEL, 'POST_DECODER_DISAMBIGUATOR', None), 'GATE_MIN_DIST_GAP', 2.0))
                    valid_gate_mask = is_ambig & (dist_gap > min_gap)

                    valid_logits = gate_logits[valid_gate_mask]
                    valid_targets = target_peak_idx[valid_gate_mask]
                    valid_dist_gap = dist_gap[valid_gate_mask]
                    pred_dict['num_valid_gate'] = int(valid_targets.numel())
                    pred_dict['valid_logits'] = valid_logits
                    pred_dict['valid_targets'] = valid_targets
                    pred_dict['valid_dist_gap'] = valid_dist_gap

                    if valid_targets.numel() > 0:
                        class_counts = torch.bincount(valid_targets, minlength=k_peaks).float()
                        present = class_counts > 0
                        num_present = present.float().sum().clamp_min(1.0)
                        class_weights = torch.zeros_like(class_counts)
                        class_weights[present] = class_counts.sum() / (
                            num_present * class_counts[present].clamp_min(1.0)
                        )
                        class_weights = class_weights.to(device=valid_logits.device, dtype=valid_logits.dtype)
                        ce = F.cross_entropy(
                            valid_logits,
                            valid_targets,
                            reduction="none",
                        )
                        top1_wrong = valid_targets != 0
                        top1_wrong_weight = float(getattr(getattr(self.cfg.MODEL, 'POST_DECODER_DISAMBIGUATOR', None), 'TOP1_WRONG_WEIGHT', 2.0))
                        class_sample_weights = class_weights[valid_targets]
                        sample_weights = torch.ones_like(ce) + top1_wrong.to(ce.dtype) * (top1_wrong_weight - 1.0)
                        final_weights = class_sample_weights * sample_weights
                        loss_gate = (ce * final_weights).sum() / (final_weights.sum() + 1e-6)
                        pred_dict['class_counts'] = class_counts
                        pred_dict['class_weights'] = class_weights
                        pred_dict['final_weights'] = final_weights
                        gate_debug = {
                            'valid_logits': valid_logits,
                            'valid_targets': valid_targets,
                            'valid_dist_gap': valid_dist_gap,
                            'class_counts': class_counts,
                            'class_weights': class_weights,
                            'final_weights': final_weights,
                        }

        gate_valid_count = pred_dict.get('num_valid_gate', None)
        if gate_valid_count is None:
            gate_info = pred_dict.get('gate_loss_inputs', None)
            if isinstance(gate_info, dict) and ('is_ambiguous' in gate_info):
                try:
                    gate_valid_count = gate_info['is_ambiguous'].float().sum().item()
                except Exception:
                    gate_valid_count = None

        # ========== Gate loss logging metrics ==========
        # Only log valid_acc / valid_p_correct for batches that contain valid
        # ambiguous samples; otherwise epoch averages are diluted by zeros.
        gate_metrics = {
            'Gate/num_valid': 0.0,
            'Gate/valid_ratio': 0.0,
            'Gate/has_valid': 0.0,
            'Gate/topk': float(pred_dict.get('gate_logits', torch.empty(0, 0)).size(-1)) if isinstance(pred_dict.get('gate_logits', None), torch.Tensor) else 0.0,
            'Gate/valid_acc': 0.0,
            'Gate/valid_p_correct': 0.0,
            'Gate/pred_peak0_ratio': 0.0,
            'Gate/target_peak0_ratio': 0.0,
            'Gate/top1_wrong_ratio': 0.0,
            'Gate/top1_wrong_acc': 0.0,
            'Gate/class_counts': 0.0,
            'Gate/class0_count': 0.0,
            'Gate/class1_count': 0.0,
            'Gate/dist_gap_mean': 0.0,
            'Gate/effective_peak_count_mean': 0.0,
            'Gate/class_weight_min': 0.0,
            'Gate/class_weight_max': 0.0,
            'Gate/final_weight_mean': 0.0,
        }
        try:
            valid_logits = None
            valid_targets = None
            # 优先从pred_dict获取
            if False and 'gate_logits' in pred_dict and 'disamb_aux' in pred_dict:
                gate_logits = pred_dict['gate_logits']
                aux_info = pred_dict['disamb_aux']
                is_ambig = aux_info.get('is_ambiguous', None)
                if is_ambig is not None and gate_logits is not None:
                    is_ambig = is_ambig.bool()
                    if is_ambig.any():
                        valid_logits = gate_logits[is_ambig]
                        # gate_target 计算逻辑同上
                        top1_xy = aux_info.get('top1_xy', None)
                        top2_xy = aux_info.get('top2_xy', None)
                        if top1_xy is not None and top2_xy is not None:
                            fx_sz = float(pred_dict['score_map'].size(-1))
                            fy_sz = float(pred_dict['score_map'].size(-2))
                            cx = (gt_bbox[:, 0] + 0.5 * gt_bbox[:, 2]) * fx_sz
                            cy = (gt_bbox[:, 1] + 0.5 * gt_bbox[:, 3]) * fy_sz
                            gt_xy = torch.stack([cx, cy], dim=-1)
                            top1_xy = top1_xy.to(device=gt_xy.device, dtype=gt_xy.dtype)
                            top2_xy = top2_xy.to(device=gt_xy.device, dtype=gt_xy.dtype)
                            dist1 = torch.sum((top1_xy - gt_xy) ** 2, dim=-1)
                            dist2 = torch.sum((top2_xy - gt_xy) ** 2, dim=-1)
                            gate_target = (dist2 < dist1).long()
                            valid_targets = gate_target[is_ambig]
            # 如果pred_dict有缓存
            if valid_logits is None and 'valid_logits' in pred_dict and 'valid_targets' in pred_dict:
                valid_logits = pred_dict['valid_logits']
                valid_targets = pred_dict['valid_targets']
            num_valid = int(valid_targets.numel()) if valid_targets is not None else 0
            gate_metrics['Gate/num_valid'] = float(num_valid)
            gate_metrics['Gate/valid_ratio'] = float(num_valid) / float(max(gt_bbox.size(0), 1))

            # 日志统计
            if valid_logits is not None and valid_targets is not None and valid_logits.numel() > 0:
                with torch.no_grad():
                    valid_probs = torch.softmax(valid_logits, dim=-1)  # [N, 2]
                    pred = valid_probs.argmax(dim=-1)  # [N]
                    gate_metrics.update({
                        'Gate/valid_acc': (pred == valid_targets).float().mean().item(),
                        'Gate/valid_p_correct': valid_probs.gather(1, valid_targets.view(-1, 1)).mean().item(),
                        'Gate/pred_peak0_ratio': (pred == 0).float().mean().item(),
                        'Gate/target_peak0_ratio': (valid_targets == 0).float().mean().item(),
                        'Gate/has_valid': 1.0,
                    })
        except Exception:
            pass

        try:
            valid_logits = pred_dict.get('valid_logits', gate_debug.get('valid_logits', None))
            valid_targets = pred_dict.get('valid_targets', gate_debug.get('valid_targets', None))
            valid_dist_gap = pred_dict.get('valid_dist_gap', gate_debug.get('valid_dist_gap', None))
            class_weights_for_log = pred_dict.get('class_weights', gate_debug.get('class_weights', None))
            final_weights_for_log = pred_dict.get('final_weights', gate_debug.get('final_weights', None))
            num_valid = int(valid_targets.numel()) if valid_targets is not None else 0
            gate_metrics['Gate/num_valid'] = float(num_valid)
            gate_metrics['Gate/valid_ratio'] = float(num_valid) / float(max(gt_bbox.size(0), 1))

            aux_info = pred_dict.get('disamb_aux', None)
            if isinstance(aux_info, dict) and isinstance(aux_info.get('effective_peak_count', None), torch.Tensor):
                gate_metrics['Gate/effective_peak_count_mean'] = aux_info['effective_peak_count'].detach().float().mean().item()
            if isinstance(valid_dist_gap, torch.Tensor) and valid_dist_gap.numel() > 0:
                gate_metrics['Gate/dist_gap_mean'] = valid_dist_gap.detach().float().mean().item()

            if valid_logits is not None and valid_targets is not None and valid_logits.numel() > 0:
                with torch.no_grad():
                    valid_probs = torch.softmax(valid_logits, dim=-1)
                    pred = valid_probs.argmax(dim=-1)
                    top1_wrong = valid_targets != 0
                    top1_wrong_acc = (
                        (pred[top1_wrong] == valid_targets[top1_wrong]).float().mean().item()
                        if top1_wrong.any() else 0.0
                    )
                    class_counts = torch.bincount(valid_targets.detach().long(), minlength=valid_logits.size(-1)).float()
                    present_weights = None
                    if isinstance(class_weights_for_log, torch.Tensor):
                        class_weights_for_log = class_weights_for_log.detach().float()
                        present_weights = class_weights_for_log[class_counts.to(class_weights_for_log.device) > 0]
                    if isinstance(present_weights, torch.Tensor) and present_weights.numel() > 0:
                        gate_metrics['Gate/class_weight_min'] = present_weights.min().item()
                        gate_metrics['Gate/class_weight_max'] = present_weights.max().item()
                    if isinstance(final_weights_for_log, torch.Tensor) and final_weights_for_log.numel() > 0:
                        gate_metrics['Gate/final_weight_mean'] = final_weights_for_log.detach().float().mean().item()
                    gate_metrics.update({
                        'Gate/topk': float(valid_logits.size(-1)),
                        'Gate/valid_acc': (pred == valid_targets).float().mean().item(),
                        'Gate/valid_p_correct': valid_probs.gather(1, valid_targets.view(-1, 1)).mean().item(),
                        'Gate/pred_peak0_ratio': (pred == 0).float().mean().item(),
                        'Gate/target_peak0_ratio': (valid_targets == 0).float().mean().item(),
                        'Gate/top1_wrong_ratio': top1_wrong.float().mean().item(),
                        'Gate/top1_wrong_acc': top1_wrong_acc,
                        'Gate/class_counts': class_counts.sum().item(),
                        'Gate/has_valid': 1.0,
                    })
                    for cls_idx, cls_count in enumerate(class_counts.tolist()):
                        gate_metrics[f'Gate/class_count_{cls_idx}'] = float(cls_count)
                        gate_metrics[f'Gate/class{cls_idx}_count'] = float(cls_count)
        except Exception:
            pass

        stage = self._last_train_stage if self._last_train_stage is not None else 3
        loss_inputs = {
            'loss_track': loss_track,
            'loss_total_original': loss_track,
            'loss_giou': giou_loss,
            'loss_l1': l1_loss,
            'loss_location': location_loss,
            'loss_gate': loss_gate,
        }
        if gate_valid_count is not None:
            loss_inputs['num_valid_gate'] = gate_valid_count
        loss, stage_logs, skip_backward = build_total_loss_by_stage(
            stage=stage,
            losses=loss_inputs,
            lambda_gate=self.lambda_gate,
            use_track_loss_stage2=self.use_track_loss_stage2,
            stage3_freeze_sutrack=self.stage3_freeze_sutrack,
            model=self.net,
            fallback_tensor=loss_track,
        )

        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {
                "Loss/giou": giou_loss.item(),
                "Loss/l1": l1_loss.item(),
                "Loss/location": location_loss.item(),
                "Loss/task_class": task_cls_loss.item(),
                "Loss/track": loss_track.item(),
                "IoU": mean_iou.item(),
                "Stage/id": float(stage),
                "Stage/stage3_freeze_sutrack": 1.0 if self.stage3_freeze_sutrack else 0.0,
                "Stage/use_refined_for_bbox": 1.0 if getattr(self._unwrap_model(), "use_refined_for_bbox", False) else 0.0,
                "Stage/trainable_params_million": float(self._last_trainable_params_million),
                "Meta/skip_backward": 1.0 if skip_backward else 0.0,
            }
            if isinstance(loss_gate, torch.Tensor):
                status["Loss/gate"] = loss_gate.detach().item()
            if loss is not None:
                status["Loss/total"] = loss.item()
            else:
                status["Loss/total"] = 0.0
            for key in (
                "V4/use_rerank_safe",
                "V4/use_refined_safe",
                "V4/fallback_to_baseline",
                "V4/fallback_reason",
                "V4/selected_peak_idx",
                "V4/rerank_idx",
                "V4/rerank_confidence",
                "V4/dist_base_prev",
                "V4/dist_rerank_prev",
                "V4/dist_ref_prev",
                "V4/dist_rerank_base",
                "V4/dist_ref_base",
                "V4/dist_selected_base",
                "V4/gate_confidence",
                "V4/ambiguity_ratio",
                "V4/top1_score",
                "V4/effective_peak_count",
            ):
                value = pred_dict.get(key, None)
                if isinstance(value, torch.Tensor):
                    status[key] = float(value.detach().float().mean().item())
            for key, value in stage_logs.items():
                if key in status:
                    continue
                if value is not None:
                    status[key] = float(value)
            # 写入 Gate 日志
            status.update(gate_metrics)
            return loss, status
        else:
            return loss
