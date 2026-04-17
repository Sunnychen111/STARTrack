from lib.test.tracker.basetracker import BaseTracker
import torch
import torch.nn.functional as F
from lib.test.tracker.utils import sample_target, transform_image_to_crop
import cv2
import numpy as np
from lib.utils.box_ops import box_xywh_to_xyxy, box_xyxy_to_cxcywh, clip_box
from lib.test.utils.hann import hann2d
from lib.models.sutrack import build_sutrack
from lib.test.tracker.utils import Preprocessor
import clip
import os

# 尝试导入 Depth Anything V2
try:
    from lib.models.DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2
    DEPTH_MODEL_AVAILABLE = True
except ImportError:
    print("Warning: Depth Anything V2 not found. Running in RGB-only mode.")
    DEPTH_MODEL_AVAILABLE = False

class SUTRACK(BaseTracker):
    def __init__(self, params, dataset_name):
        super(SUTRACK, self).__init__(params)
        
        # 1. Build & Load Network
        network = build_sutrack(params.cfg)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu', weights_only=False)['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        
        self.preprocessor = Preprocessor()
        self.state = None

        self.fx_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.ENCODER.STRIDE
        if self.cfg.TEST.WINDOW == True: 
            self.output_window = hann2d(torch.tensor([self.fx_sz, self.fx_sz]).long(), centered=True).cuda()

        self.num_template = self.cfg.TEST.NUM_TEMPLATES
        self.debug = params.debug
        self.frame_id = 0
        self.search_history = []
        self.temporal_history_size = max(0, getattr(self.cfg.DATA.SEARCH, 'NUMBER', 1) - 1)
        self.depth_infer_policy = getattr(self.cfg.TEST, 'DEPTH_INFER_POLICY', 'bipeak')
        self.depth_reinfer_cooldown = getattr(self.cfg.TEST, 'DEPTH_REINFER_COOLDOWN', 1)
        self.last_depth_infer_frame = -10**9
        ratio_th = getattr(self.cfg.TEST, 'BIPEAK_RATIO_TH', 0.75)
        dist_th = getattr(self.cfg.TEST, 'BIPEAK_DIST_TH', 6)
        self.bipeak_ratio_th = ratio_th
        self.bipeak_dist_th = max(1, int(round(self.fx_sz * dist_th))) if dist_th <= 1 else int(dist_th)

        # Load Depth Model
        self.depth_model = None
        if DEPTH_MODEL_AVAILABLE:
            self.depth_model = self._load_depth_model()

        DATASET_NAME = dataset_name.upper()
        if hasattr(self.cfg.TEST.UPDATE_INTERVALS, DATASET_NAME):
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS[DATASET_NAME]
        else:
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS.DEFAULT
        
        if hasattr(self.cfg.TEST.UPDATE_THRESHOLD, DATASET_NAME):
            self.update_threshold = self.cfg.TEST.UPDATE_THRESHOLD[DATASET_NAME]
        else:
            self.update_threshold = self.cfg.TEST.UPDATE_THRESHOLD.DEFAULT

        if 'GOT10K' in DATASET_NAME: DATASET_NAME = 'GOT10K'
        if 'LASOT' in DATASET_NAME: DATASET_NAME = 'LASOT'
        if 'OTB' in DATASET_NAME: DATASET_NAME = 'TNL2K'

        if hasattr(self.cfg.TEST.MULTI_MODAL_VISION, DATASET_NAME):
            self.multi_modal_vision = self.cfg.TEST.MULTI_MODAL_VISION[DATASET_NAME]
        else:
            self.multi_modal_vision = self.cfg.TEST.MULTI_MODAL_VISION.DEFAULT

        if hasattr(self.cfg.TEST.MULTI_MODAL_LANGUAGE, DATASET_NAME):
            self.multi_modal_language = self.cfg.TEST.MULTI_MODAL_LANGUAGE[DATASET_NAME]
        else:
            self.multi_modal_language = self.cfg.TEST.MULTI_MODAL_LANGUAGE.DEFAULT

        if hasattr(self.cfg.TEST.USE_NLP, DATASET_NAME):
            self.use_nlp = self.cfg.TEST.USE_NLP[DATASET_NAME]
        else:
            self.use_nlp = self.cfg.TEST.USE_NLP.DEFAULT

        self.task_index_batch = None

    def _load_depth_model(self):
        try:
            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            }
            encoder_type = 'vitb' 
            model = DepthAnythingV2(**model_configs[encoder_type])
            ckpt_path = r"/home/cps/czl/RGBDL/lib/models/DepthAnythingV2/checkpoints/depth_anything_v2_vitb.pth" 
            
            if os.path.exists(ckpt_path):
                model.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=False))
                model = model.cuda().eval()
                print(f"[SUTRACK] Depth Model ({encoder_type}) Loaded Successfully.")
                return model
            else:
                return None
        except Exception as e:
            print(f"[SUTRACK] Error loading Depth Model: {e}")
            return None

    def _infer_depth(self, image):
        if self.depth_model is None:
            return None
        
        with torch.no_grad():
            # DepthAnything 输出的是 [H, W] 的浮点数绝对深度图
            depth = self.depth_model.infer_image(image) 

        # ==============================================================
        # 【方案 A 核心实现：深度梯度化 (Sobel 边缘提取)】
        # ==============================================================
        # 1. 计算 X 方向和 Y 方向的深度落差（梯度）
        sobelx = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)
        
        # 2. 勾股定理计算综合的梯度幅值（勾勒出深度的边缘轮廓）
        depth_gradient = np.sqrt(sobelx**2 + sobely**2)

        # 3. 对“梯度图”进行全局归一化，拉伸到 [0, 1] 喂给网络
        g_min = depth_gradient.min()
        g_max = depth_gradient.max()
        if g_max - g_min > 1e-5:
            final_depth_feature = (depth_gradient - g_min) / (g_max - g_min)
        else:
            final_depth_feature = np.zeros_like(depth_gradient)

        # 返回 [H, W, 1] 的深度梯度图，彻底抛弃绝对深度值！
        return final_depth_feature[:, :, None]

    def _process_rgbd(self, patch_arr):
        img_rgb = patch_arr[:, :, :3]
        img_depth = patch_arr[:, :, 3:]
        
        rgb_tensor = self.preprocessor.process(img_rgb) # [1, 3, H, W]
        
        # ==============================================================
        # 【核心修复2：移除局部归一化】直接透传全局深度的 Tensor
        # ==============================================================
        depth_tensor = torch.tensor(img_depth).float().permute(2, 0, 1).unsqueeze(0).cuda()
        
        return torch.cat([rgb_tensor, depth_tensor], dim=1)
        
    def detect_double_peak(self, score_map, peak_ratio_thresh=0.75, dist_thresh_pixel=6):
        """双峰检测算法内嵌"""
        b, c, h, w = score_map.shape
        flat_map = score_map.view(-1)
        
        p1_val, p1_idx = torch.max(flat_map, dim=0)
        p1_y, p1_x = p1_idx // w, p1_idx % w
        
        masked_map = score_map.clone().squeeze()
        device = score_map.device
        y_grid, x_grid = torch.meshgrid(
            torch.arange(h, device=device), 
            torch.arange(w, device=device), 
            indexing='ij'
        )
        dist_sq = (y_grid - p1_y)**2 + (x_grid - p1_x)**2
        mask = dist_sq < (dist_thresh_pixel**2)
        masked_map[mask] = -1e4 
        
        p2_val, p2_idx = torch.max(masked_map.view(-1), dim=0)
        
        ratio = (p2_val / p1_val).item() if p1_val.item() > 0 else 1.0
        is_double_peak = ratio > peak_ratio_thresh
        
        return is_double_peak, ratio

    def initialize(self, image, info: dict):
        # Crop template region (RGB only), then infer depth on the small crop
        z_patch_arr, resize_factor = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                   output_sz=self.params.template_size)
        init_depth = self._infer_depth(z_patch_arr)

        if init_depth is not None:
            template = self._process_rgbd(np.concatenate([z_patch_arr, init_depth], axis=2))
        else:
            template = self.preprocessor.process(z_patch_arr)
            if self.multi_modal_vision and (template.size(1) == 3):
                template = torch.cat((template, template), axis=1)

        self.template_list = [template] * self.num_template
        self.state = info['init_bbox']
        
        prev_box_crop = transform_image_to_crop(torch.tensor(info['init_bbox']),
                                                torch.tensor(info['init_bbox']),
                                                resize_factor,
                                                torch.Tensor([self.params.template_size, self.params.template_size]),
                                                normalize=True)
        self.template_anno_list = [prev_box_crop.to(template.device).unsqueeze(0)]
        self.frame_id = 0
        self.search_history = []
        self.last_depth_infer_frame = -10**9

        if self.multi_modal_language:
            init_nlp = info.get("init_nlp") if self.use_nlp else None
            text_data, _ = self.extract_token_from_nlp_clip(init_nlp)
            text_data = text_data.unsqueeze(0).to(template.device)
            with torch.no_grad():
                self.text_src = self.network.forward_textencoder(text_data=text_data)
        else:
            self.text_src = None

    def _should_trigger_depth_rescue(self, is_double_peak):
        if self.depth_infer_policy == 'never':
            return False
        if self.depth_infer_policy == 'always':
            return True
        if not is_double_peak:
            return False
        return (self.frame_id - self.last_depth_infer_frame) >= self.depth_reinfer_cooldown

    def _build_temporal_search_list(self, current_search):
        if self.temporal_history_size <= 0:
            return [current_search]
        history = list(self.search_history[-self.temporal_history_size:])
        return history + [current_search]

    def _push_search_history(self, search_tensor):
        if self.temporal_history_size <= 0:
            return
        self.search_history.append(search_tensor.detach())
        if len(self.search_history) > self.temporal_history_size:
            self.search_history = self.search_history[-self.temporal_history_size:]

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1

        # ==========================================================
        # Stage 1: Scout (RGB + temporal history, no depth inference)
        # ==========================================================
        x_patch_arr_fast, resize_factor = sample_target(image, self.state, self.params.search_factor,
                                                   output_sz=self.params.search_size)

        search_fast = self.preprocessor.process(x_patch_arr_fast)
        dummy_depth_tensor = torch.zeros_like(search_fast[:, :1, :, :])
        search_fast = torch.cat((search_fast, dummy_depth_tensor), dim=1)

        # Use temporal history so TemporalMambaBlock fires every frame
        temporal_search_list_fast = self._build_temporal_search_list(search_fast)

        with torch.no_grad():
            enc_opt_fast = self.network.forward_encoder(self.template_list,
                                                        temporal_search_list_fast,
                                                        self.template_anno_list,
                                                        self.text_src,
                                                        self.task_index_batch)
            out_dict_fast = self.network.forward_decoder(feature=enc_opt_fast)

        pred_score_map_fast = out_dict_fast['score_map']

        is_double_peak, ratio = self.detect_double_peak(
            pred_score_map_fast,
            peak_ratio_thresh=self.bipeak_ratio_th,
            dist_thresh_pixel=self.bipeak_dist_th,
        )
        current_search_for_history = search_fast
        depth_rescued = False

        # ==========================================================
        # Stage 2: Rescue (RGB-D + temporal, on demand)
        # ==========================================================
        if not self._should_trigger_depth_rescue(is_double_peak):
            out_dict = out_dict_fast
        else:
            # Infer depth only on the already-cropped search patch (much faster than full image)
            depth_crop = self._infer_depth(x_patch_arr_fast)

            search_heavy = self.preprocessor.process(x_patch_arr_fast)
            if depth_crop is not None:
                depth_tensor = torch.tensor(depth_crop).float().permute(2, 0, 1).unsqueeze(0).cuda()
                depth_rescued = True
            else:
                depth_tensor = torch.zeros_like(search_heavy[:, :1, :, :])
            search_heavy = torch.cat([search_heavy, depth_tensor], dim=1)
            current_search_for_history = search_heavy
            temporal_search_list = self._build_temporal_search_list(search_heavy)

            with torch.no_grad():
                enc_opt_heavy = self.network.forward_encoder(self.template_list,
                                                             temporal_search_list,
                                                             self.template_anno_list,
                                                             self.text_src,
                                                             self.task_index_batch)
                out_dict = self.network.forward_decoder(feature=enc_opt_heavy)
            self.last_depth_infer_frame = self.frame_id


        # ==========================================================
        # 统一后续处理：解码 BBox 与 模板更新
        # ==========================================================
        pred_score_map = out_dict['score_map']
        if self.cfg.TEST.WINDOW == True:
            response = self.output_window * pred_score_map
        else:
            response = pred_score_map
            
        if 'size_map' in out_dict.keys():
            pred_boxes, conf_score = self.network.decoder.cal_bbox(response, out_dict['size_map'],
                                                                   out_dict['offset_map'], return_score=True)
        else:
            pred_boxes, conf_score = self.network.decoder.cal_bbox(response, out_dict['offset_map'], return_score=True)
            
        pred_boxes = pred_boxes.view(-1, 4)
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        self._push_search_history(current_search_for_history)

        # --- Template update (crop-based depth) ---
        if self.num_template > 1:
            base_condition = (self.frame_id % self.update_intervals == 0) and (conf_score > self.update_threshold)

            if base_condition:
                final_double_peak, _ = self.detect_double_peak(pred_score_map, peak_ratio_thresh=0.75)
                semantic_reliable = (conf_score > 0.65)

                if not final_double_peak and semantic_reliable:
                    # Crop template (RGB only), optionally infer depth on the small crop
                    z_patch_rgb, resize_factor_z = sample_target(image, self.state, self.params.template_factor,
                                                                  output_sz=self.params.template_size)
                    z_depth = self._infer_depth(z_patch_rgb) if depth_rescued else None

                    depth_reliable = True
                    if z_depth is not None:
                        depth_var = np.var(z_depth)
                        if depth_var < 0.00001 or depth_var > 0.5:
                            depth_reliable = False

                    if depth_reliable:
                        template = self.preprocessor.process(z_patch_rgb)
                        if z_depth is not None:
                            depth_tensor = torch.tensor(z_depth).float().permute(2, 0, 1).unsqueeze(0).cuda()
                        else:
                            depth_tensor = torch.zeros_like(template[:, :1, :, :])
                        template = torch.cat([template, depth_tensor], dim=1)

                        self.template_list.append(template)
                        if len(self.template_list) > self.num_template:
                            self.template_list.pop(1)

                        prev_box_crop = transform_image_to_crop(torch.tensor(self.state),
                                                                torch.tensor(self.state),
                                                                resize_factor_z,
                                                                torch.Tensor([self.params.template_size, self.params.template_size]),
                                                                normalize=True)
                        self.template_anno_list.append(prev_box_crop.to(template.device).unsqueeze(0))
                        if len(self.template_anno_list) > self.num_template:
                            self.template_anno_list.pop(1)

        # debug visualization
        if self.debug == 1:
            image_show = image[:,:,:3] if image.shape[-1] in [4, 6] else image
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image_show, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
            cv2.imshow('vis', image_BGR)
            cv2.waitKey(1)

        return {"target_bbox": self.state,
                "best_score": conf_score}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

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