import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthGuidedContrastiveLossV2(nn.Module):
    def __init__(self, top_k=10, tau_d=0.15, temperature=0.07):
        """
        Args:
            top_k (int): 候选困难负样本的数量（按外观相似度排序）
            tau_d (float): 标量深度差异阈值 (相对深度差)
            temperature (float): InfoNCE 温度系数
        """
        super().__init__()
        self.top_k = top_k
        self.tau_d = tau_d
        self.temperature = temperature

    def forward(self, aux_info, targets):
        # 直接拿 encoder 切好的特征，避免任何 Sequence Offset Bug
        search_fused = aux_info['search_fused']           # [B, Hs*Ws, C]
        template_fused = aux_info['template_fused']       # [B, Ht*Wt*N_t, C]
        search_depth_scalar = aux_info['search_depth_scalar']  # [B, Hs, Ws] 
        Hs, Ws = aux_info['search_hw']
        Ht, Wt = aux_info['template_hw']
        
        B = search_fused.shape[0]
        device = search_fused.device
        loss = 0.0
        valid_batches = 0
        
        for i in range(B):
            # ---------------------------------------------------------
            # 1. 解析 GT Bbox (假设传入的是 Search 区归一化坐标 cx, cy, w, h)
            # ---------------------------------------------------------
            cx, cy, w, h = targets['boxes'][i]
            x1 = max(0, int((cx - w / 2) * Ws))
            y1 = max(0, int((cy - h / 2) * Hs))
            x2 = min(Ws - 1, int((cx + w / 2) * Ws))
            y2 = min(Hs - 1, int((cy + h / 2) * Hs))
            
            gt_mask_2d = torch.zeros((Hs, Ws), dtype=torch.bool, device=device)
            if x2 > x1 and y2 > y1:
                gt_mask_2d[y1:y2, x1:x2] = True
            else:
                cx_idx = min(Ws - 1, max(0, int(cx * Ws)))
                cy_idx = min(Hs - 1, max(0, int(cy * Hs)))
                gt_mask_2d[cy_idx, cx_idx] = True
                
            gt_mask = gt_mask_2d.flatten() # [Hs*Ws]
            
            # ---------------------------------------------------------
            # 2. 提取 Target Query (f_q) 和 正样本 (f_pos)
            # ---------------------------------------------------------
            # 改动 2：Template Query 提取 - 仅使用中心 50% 区域，拒绝背景污染
            c_h, c_w = Ht // 2, Wt // 2
            dh, dw = max(1, Ht // 4), max(1, Wt // 4)
            template_mask_2d = torch.zeros((Ht, Wt), dtype=torch.bool, device=device)
            template_mask_2d[c_h-dh : c_h+dh, c_w-dw : c_w+dw] = True
            num_tokens_per_template = template_mask_2d.numel()
            f_q = template_fused[i][:num_tokens_per_template][template_mask_2d.flatten()].mean(dim=0) # [C]
            
            f_pos = search_fused[i][gt_mask].mean(dim=0) # [C]
            
            # ---------------------------------------------------------
            # 3. 计算 Target 的标量深度 (使用中位数抵抗边界噪声)
            # ---------------------------------------------------------
            # 改动 1 (续)：使用 patch-level scalar depth 的中位数
            target_depth_region = search_depth_scalar[i][gt_mask_2d]
            if target_depth_region.numel() > 0:
                d_pos_scalar = torch.median(target_depth_region)
            else:
                d_pos_scalar = search_depth_scalar[i, cy_idx, cx_idx] # Fallback 中心点
                
            # ---------------------------------------------------------
            # 4. Top-K 负样本挖掘 (结合外观与深度)
            # ---------------------------------------------------------
            sim_map = F.cosine_similarity(f_q.unsqueeze(0), search_fused[i], dim=-1) # [Hs*Ws]
            
            # 排除 GT 区域，防止误伤
            sim_map[gt_mask] = -2.0 
            
            # 改动 3：先取外观最相似的 Top-K 候选
            k_val = min(self.top_k, sim_map.numel() - gt_mask.sum().item())
            if k_val <= 0:
                continue
                
            topk_sims, topk_indices = torch.topk(sim_map, k_val)
            
            # 提取这 Top-K 个位置的深度差异
            depth_diff_map = torch.abs(search_depth_scalar[i].flatten() - d_pos_scalar)
            topk_depth_diffs = depth_diff_map[topk_indices]
            
            # 过滤出深度差异足够大的作为 Hard Negatives
            hard_neg_mask = topk_depth_diffs > self.tau_d
            
            if hard_neg_mask.any():
                final_neg_indices = topk_indices[hard_neg_mask]
            else:
                # 降级策略 (Fallback)：如果没有通过深度筛选的，说明都在同一深度层或本帧没有难样本
                # 退化为普通的 Contrastive Learning，只推开最相似的 2 个 token 避免 Loss 空转
                fallback_k = min(2, k_val)
                final_neg_indices = topk_indices[:fallback_k]
                
            f_neg = search_fused[i][final_neg_indices] # [N_neg, C]
            
            # ---------------------------------------------------------
            # 5. InfoNCE Loss
            # ---------------------------------------------------------
            pos_logit = F.cosine_similarity(f_q, f_pos, dim=0) / self.temperature
            neg_logits = F.cosine_similarity(f_q.unsqueeze(0), f_neg, dim=-1) / self.temperature
            
            logits = torch.cat([pos_logit.unsqueeze(0), neg_logits])
            log_den = torch.logsumexp(logits, dim=0)
            
            loss += (-pos_logit + log_den)
            valid_batches += 1
            
        if valid_batches > 0:
            return loss / valid_batches
        else:
            return (search_fused * 0).sum()