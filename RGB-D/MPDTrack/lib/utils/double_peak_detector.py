import torch
import json
import os
from tqdm import tqdm
# 这里的 import 路径请根据你的实际工程修改
from lib.models.sutrack.sutrack import build_sutrack
from utils.double_peak_detector import detect_double_peak # 上文我们写的函数

def extract_hard_dense_sequences(filtered_dataset_loader, weight_path, save_path):
    # 1. 初始化模型并加载预训练权重
    model = build_sutrack(cfg) # 传入你的网络配置
    checkpoint = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(checkpoint['net'], strict=True)
    model.cuda().eval()
    
    hard_samples = []

    print("开始扫描离线难例序列...")
    with torch.no_grad():
        for data in tqdm(filtered_dataset_loader):
            # data 应该包含 template (模板帧) 和 search (搜索帧)
            template = data['template_images'].cuda()
            search = data['search_images'].cuda()
            seq_name = data['seq_name'][0]
            frame_id = data['frame_id'][0]

            # 2. 剥离追踪逻辑，仅做网络的前向传播
            # 现代 SOT 模型（如 OSTrack 变体）通常有专门提取特征和分类头的接口
            # 你需要找到类似下面这样的代码逻辑：
            x = model.forward_backbone(template, search)
            out_dict = model.forward_head(x) 
            
            # 提取响应图 (Classification / Score Map)
            # 形状通常是 [B, 1, H, W] 或 [B, H*W, 1]
            score_map = out_dict['score_map'] 
            
            # 如果是 transformer 出来的 1D 序列，需要 reshape 成 2D 特征图
            if score_map.dim() == 3:
                hw = int(score_map.shape[1] ** 0.5)
                score_map = score_map.view(-1, 1, hw, hw)

            # 3. 传入我们的检测函数
            is_double_peak, info = detect_double_peak(score_map, peak_ratio_thresh=0.75)
            
            if is_double_peak:
                # 记录详细信息
                hard_samples.append({
                    "sequence": seq_name,
                    "frame_id": int(frame_id),
                    "ratio": round(info['ratio'], 3),
                    "p1": round(info['p1'], 3),
                    "p2": round(info['p2'], 3)
                })

    # 4. 保存为 JSON，供后续伪深度生成脚本使用
    with open(save_path, 'w') as f:
        json.dump(hard_samples, f, indent=4)
    print(f"提取完成！共发现 {len(hard_samples)} 个双峰难例帧，已保存至 {save_path}")