"""
SUTrack Model
"""
import torch
import math
import os
from torch import nn
import torch.nn.functional as F
from .encoder import build_encoder
from .clip import build_textencoder
from .decoder import build_decoder
from .task_decoder import build_task_decoder
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.utils.pos_embed import get_sinusoid_encoding_table, get_2d_sincos_pos_embed



class SUTRACK(nn.Module):
    """ This is the base class for SUTrack """
    def __init__(self, text_encoder, encoder, decoder, task_decoder,
                 num_frames=1, num_template=1,
                 decoder_type="CENTER", task_feature_type="average"):
        """ Initializes the model.
        """
        super().__init__()
        self.encoder = encoder
        self.text_encoder = text_encoder
        self.decoder_type = decoder_type

        self.class_token = False if (encoder.body.cls_token is None) else True
        self.task_feature_type = task_feature_type

        self.num_patch_x = self.encoder.body.num_patches_search
        self.num_patch_z = self.encoder.body.num_patches_template
        self.fx_sz = int(math.sqrt(self.num_patch_x))
        self.fz_sz = int(math.sqrt(self.num_patch_z))

        self.task_decoder = task_decoder
        self.decoder = decoder

        self.num_frames = num_frames
        self.num_template = num_template
        self.active_num_search = num_frames
        self.active_num_template = num_template


    def forward(self, text_data=None,
                template_list=None, search_list=None, template_anno_list=None,
                text_src=None, task_index=None,
                feature=None, mode="encoder"):
        if mode == "text":
            return self.forward_textencoder(text_data)
        elif mode == "encoder":
            return self.forward_encoder(template_list, search_list, template_anno_list, text_src, task_index)
        elif mode == "decoder":
            return self.forward_decoder(feature), self.forward_task_decoder(feature)
        else:
            raise ValueError

    def forward_textencoder(self, text_data):
        # Forward the encoder
        text_src = self.text_encoder(text_data)
        return text_src

    def forward_encoder(self, template_list, search_list, template_anno_list, text_src, task_index):
        # Forward the encoder
        self.active_num_search = len(search_list) if search_list is not None else self.num_frames
        self.active_num_template = len(template_list) if template_list is not None else self.num_template
        xz = self.encoder(template_list, search_list, template_anno_list, text_src, task_index)
        self.active_num_search = getattr(self.encoder, "last_num_search", self.active_num_search)
        self.active_num_template = getattr(self.encoder, "last_num_template", self.active_num_template)
        return xz

    def forward_decoder(self, feature, gt_score_map=None):

        feature = feature[0]
        actual_num_search = max(1, getattr(self, "active_num_search", self.num_frames))
        search_offset = 1 if self.class_token else 0
        last_search_start = search_offset + self.num_patch_x * (actual_num_search - 1)
        last_search_end = last_search_start + self.num_patch_x
        feature = feature[:, last_search_start:last_search_end]

        if feature.size(1) != self.num_patch_x:
            raise ValueError(
                f"Decoder expects {self.num_patch_x} tokens for the current search frame, got {feature.size(1)}."
            )

        bs, HW, C = feature.size()
        if self.decoder_type in ['CORNER', 'CENTER']:
            feature = feature.permute((0, 2, 1)).contiguous()
            feature = feature.view(bs, C, self.fx_sz, self.fx_sz)
        if self.decoder_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.decoder(feature, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.decoder_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.decoder(feature, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        elif self.decoder_type == "MLP":
            # run the mlp head
            score_map, bbox, offset_map = self.decoder(feature, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError

    def forward_task_decoder(self, feature):
        feature = feature[0]
        if self.task_feature_type == 'class':
            feature = feature[:, 0:1]
        elif self.task_feature_type == 'text':
            feature = feature[:, -1:]
        elif self.task_feature_type == 'average':
            feature = feature.mean(1).unsqueeze(1)
        else:
            raise NotImplementedError('task_feature_type must be choosen from class, text, and average')
        feature = self.task_decoder(feature)
        return feature
    
# =========================================================================
# [Contribution 2] 专门用于加载 MambaVision 预训练权重的辅助函数
# =========================================================================
def load_mamba_weights(model, mamba_path):
    
    if not os.path.exists(mamba_path):
        print(f"⚠️ [Warning] Mamba pretrain file not found at: {mamba_path}")
        print(">> MambaFusionBlock will be initialized randomly (Not Recommended!)")
        return

    print(f"🚀 Loading Mamba weights from: {mamba_path}")
    try:
        # MambaVision checkpoint 通常包含 'model' 键
        checkpoint = torch.load(mamba_path, map_location='cpu',weights_only=False)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        
        # 构造新的 state_dict，只提取 Mamba 层
        # 目标: model.encoder.mamba_fusion.layers.0.mixer...
        # 源: layers.0.mixer...
        new_state_dict = {}
        for k, v in state_dict.items():
            # 我们只需要 'layers' 部分，这是 Mamba 的核心
            if "layers" in k:
                # 映射规则: 源权重 -> encoder.mamba_fusion.源权重
                # 注意：这里假设 SUTRACK 类里叫 self.encoder
                new_key = f"encoder.mamba_fusion.{k}" 
                new_state_dict[new_key] = v
                
        # 加载权重 (strict=False 是必须的，因为我们只加载 Mamba 部分，忽略其他部分)
        msg = model.load_state_dict(new_state_dict, strict=False)
        print(f"✅ Mamba weights loaded successfully!")
        # print(f"Missing keys (expected): {len(msg.missing_keys)}") 
        
    except Exception as e:
        print(f"❌ Failed to load Mamba weights: {e}")

# =========================================================================
# 修改后的 build_sutrack 函数
# =========================================================================
def build_sutrack(cfg, training=True): # 增加 training 参数
    # 1. 构建各组件
    encoder = build_encoder(cfg)

    # 1. 计算总参数量
    total_params = sum(p.numel() for p in encoder.parameters())

    # 2. 计算可训练参数量 (被 freeze 的不计入)
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)

    print(f"总参数量 (Total Parameters): {total_params / 1e6:.2f} M (百万)")
    print(f"可训练参数量 (Trainable Parameters): {trainable_params / 1e6:.2f} M (百万)")
    print(f"冻结参数量 (Frozen Parameters): {(total_params - trainable_params) / 1e6:.2f} M (百万)")
    
    if cfg.DATA.MULTI_MODAL_LANGUAGE:
        text_encoder = build_textencoder(cfg, encoder)
    else:
        text_encoder = None
        
    decoder = build_decoder(cfg, encoder)
    task_decoder = build_task_decoder(cfg, encoder)
    
    # 2. 实例化模型
    model = SUTRACK(
        text_encoder,
        encoder,
        decoder,
        task_decoder,
        num_frames = cfg.DATA.SEARCH.NUMBER,
        num_template = cfg.DATA.TEMPLATE.NUMBER,
        decoder_type=cfg.MODEL.DECODER.TYPE,
        task_feature_type=cfg.MODEL.TASK_DECODER.FEATURE_TYPE
    )

    #3. [关键步骤] 如果是训练模式，且开启了 Mamba，则加载权重
    # if training:
    #     # 检查是否开启了 Mamba (防止报错，先用 hasattr)
    #     mamba_enabled = hasattr(cfg.MODEL, 'MAMBA') and getattr(cfg.MODEL.MAMBA, 'ENABLE', False)
        
    #     if mamba_enabled:
    #         # 这里的路径需要你手动指定，或者在 yaml 里配
    #         # 假设你使用的是 FastITPN-Base (Dim 512)，对应 MambaVision-Small
    #         # 请确保你下载了 mambavision_small_1k.pth.tar 并放在这个路径
    #         mamba_weight_path = "pretrained/mambavision/mambavision_small_1k.pth"
            
    #         # 如果是 Tiny 模型 (Dim 256/384)，路径可能是 mambavision_tiny_1k.pth.tar
    #         if cfg.MODEL.MAMBA.DIM < 512:
    #              mamba_weight_path = "pretrained/mambavision/mambavision_tiny_1k.pth"

    #         load_mamba_weights(model, mamba_weight_path)
    if training:
        mamba_enabled = hasattr(cfg.MODEL, 'MAMBA') and getattr(cfg.MODEL.MAMBA, 'ENABLE', False)
    
        if mamba_enabled:
            # ... (路径定义的代码保持不变) ...
            mamba_weight_path = "pretrained/mambavision/mambavision_small_1k.pth"
            if cfg.MODEL.MAMBA.DIM < 512:
                mamba_weight_path = "pretrained/mambavision/mambavision_tiny_1k.pth"

            # === 关键修改在这里 ===
            import os
            if os.path.exists(mamba_weight_path):
                print(f"Loading Mamba pretrained weights from {mamba_weight_path}")
                load_mamba_weights(model, mamba_weight_path)
            else:
                # 文件不存在，但是不报错，继续往下走！
                print(f"Warning: Mamba weights not found at {mamba_weight_path}. Skipping initialization (Will rely on unified checkpoint).")

    return model
   