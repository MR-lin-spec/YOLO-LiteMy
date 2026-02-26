import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional, Union

# ==========================================
# 1. 动态分配器 (DynamicSoftLabelAssigner)
# 核心逻辑：基于 Cost Matrix 的动态标签分配
# ==========================================
class DynamicSoftLabelAssigner(nn.Module):
    def __init__(self, topk: int = 13, iou_factor: float = 3.0, num_classes: int = 80):
        super().__init__()
        self.topk = topk
        self.iou_factor = iou_factor
        self.num_classes = num_classes

    def forward(self, pred_scores: torch.Tensor, pred_bboxes: torch.Tensor, 
                gt_labels: torch.Tensor, gt_bboxes: torch.Tensor, 
                gt_mask: Optional[torch.Tensor] = None):
        """
        Args:
            pred_scores: [B, N, C] - Student 预测分数
            pred_bboxes: [B, N, 4] - Student 预测框 (必须是 xyxy 格式)
            gt_labels:   [B, M]   - Teacher 伪标签类别
            gt_bboxes:   [B, M, 4]- Teacher 伪标签框 (必须是 xyxy 格式)
            gt_mask:     [B, M]   - 有效伪标签掩码
        
        Returns:
            assigned_labels: [B, N]
            assigned_scores: [B, N, C]
            fg_mask:         [B, N]
            matched_gt_indices: [B, N] - 关键：返回匹配的 GT 索引
        """
        bs, n_boxes, _ = pred_scores.shape
        device = pred_scores.device
        
        # 初始化输出张量
        assigned_labels = torch.full((bs, n_boxes), -1, dtype=torch.long, device=device)
        assigned_scores = torch.zeros_like(pred_scores)
        fg_mask = torch.zeros((bs, n_boxes), dtype=torch.bool, device=device)
        matched_gt_indices = torch.full((bs, n_boxes), -1, dtype=torch.long, device=device)

        for b in range(bs):
            p_score = pred_scores[b]      # [N, C]
            p_bbox = pred_bboxes[b]       # [N, 4]
            
            g_label = gt_labels[b]        # [M]
            g_bbox = gt_bboxes[b]         # [M, 4]
            
            # 应用掩码筛选有效 GT
            if gt_mask is not None:
                valid_idx = gt_mask[b]
                if valid_idx.sum() == 0:
                    continue
                g_label = g_label[valid_idx]
                g_bbox = g_bbox[valid_idx]
            
            num_gt = g_bbox.shape[0]
            if num_gt == 0:
                continue

            # --- 1. 计算 IoU Cost ---
            tl = torch.max(p_bbox[:, None, :2], g_bbox[None, :, :2])
            br = torch.min(p_bbox[:, None, 2:], g_bbox[None, :, 2:])
            wh = (br - tl).clamp(min=0)
            inter = wh[:, :, 0] * wh[:, :, 1]
            area_p = (p_bbox[:, 2] - p_bbox[:, 0]) * (p_bbox[:, 3] - p_bbox[:, 1])
            area_g = (g_bbox[:, 2] - g_bbox[:, 0]) * (g_bbox[:, 3] - g_bbox[:, 1])
            union = area_p[:, None] + area_g[None, :] - inter
            ious = (inter / (union + 1e-9)).detach()  # [N, M]

            # --- 2. 计算 Classification Cost (BCE) ---
            gt_onehot = F.one_hot(g_label, self.num_classes).float()  # [M, C]
            # 【关键修复 1】确保预测分数经过 Sigmoid 转换到 (0, 1)
            # 假设输入的 p_score 是 logits 或未严格约束的概率
            p_score_sigmoid = torch.sigmoid(p_score) 
            
            # 【关键修复 2】防止数值溢出，强制裁剪到 [eps, 1-eps]
            eps = 1e-7
            p_score_sigmoid = p_score_sigmoid.clamp(min=eps, max=1.0 - eps)
            
            # 广播以匹配维度 [N, M, C]
            bce_preds = p_score_sigmoid.unsqueeze(1).expand(-1, num_gt, -1)
            bce_targets = gt_onehot.unsqueeze(0).expand(n_boxes, -1, -1)
            
            # 现在输入严格在 (0, 1) 之间，不会再报 CUDA Assertion 错误
            cls_cost = F.binary_cross_entropy(bce_preds, bce_targets, reduction='none').sum(dim=-1) # [N, M]

            # --- 3. 计算 Distance Cost ---
            p_center = (p_bbox[:, :2] + p_bbox[:, 2:]) / 2.0
            g_center = (g_bbox[:, :2] + g_bbox[:, 2:]) / 2.0
            dist = ((p_center[:, None] - g_center[None, :]) ** 2).sum(dim=-1).sqrt()
            # 归一化距离成本
            dis_cost = (dist / (dist.max() + 1e-9)) * 10.0

            # 总 Cost
            cost_matrix = cls_cost + ious * self.iou_factor + dis_cost

            # --- 4. Dynamic K Matching (OTA Style) ---
            matching_matrix = torch.zeros_like(cost_matrix)
            candidate_topk = min(self.topk, n_boxes)
            # 每个 GT 选择 Cost 最小的 TopK 个预测框
            topk_ious, _ = torch.topk(ious, candidate_topk, dim=0)
            dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
            
            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(cost_matrix[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
                matching_matrix[:, gt_idx][pos_idx] = 1.0
            
            # 处理一个预测框匹配多个 GT 的情况 -> 保留 Cost 最小的那个
            match_gt_mask = matching_matrix.sum(1) > 1
            if match_gt_mask.sum() > 0:
                cost_min, cost_argmin = torch.min(cost_matrix[match_gt_mask, :], dim=1)
                matching_matrix[match_gt_mask, :] *= 0.0
                matching_matrix[match_gt_mask, cost_argmin] = 1.0
            
            # 确定前景掩码和匹配索引
            fg_mask_b = matching_matrix.sum(1) > 0.0
            if fg_mask_b.sum() > 0:
                matched_inds = matching_matrix[fg_mask_b, :].argmax(dim=1)
                
                assigned_labels[b, fg_mask_b] = g_label[matched_inds]
                assigned_scores[b, fg_mask_b, :] = gt_onehot[matched_inds]
                matched_gt_indices[b, fg_mask_b] = matched_inds 
                fg_mask[b] = fg_mask_b

        return assigned_labels, assigned_scores, fg_mask, matched_gt_indices

# ==========================================
# 2. 严格的数据提取与坐标转换工具
# ==========================================

def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    将 (cx, cy, w, h) 转换为 (x1, y1, x2, y2)
    输入形状：[..., 4]
    """
    if boxes.shape[-1] != 4:
        raise ValueError(f"Expected box dimension to be 4, got {boxes.shape[-1]}")
    
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)

def extract_predictions_strict(preds: Union[Dict, Tuple], branch_name: str = 'one2one') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    【严格模式】从 YOLO 输出结构中提取 boxes 和 scores。
    如果键不存在，直接抛出 KeyError，拒绝任何隐式假设或占位符。
    
    支持结构:
    1. Dict: {'one2one': {'boxes': ..., 'scores': ...}}
    2. Tuple: (raw_tensor, {'one2one': {'boxes': ..., 'scores': ...}})
    """
    data_dict = None
    
    # 情况 A: 输入是 Dict (Student)
    if isinstance(preds, dict):
        if branch_name not in preds:
            raise KeyError(f"Branch '{branch_name}' not found in student prediction dict. Keys: {preds.keys()}")
        data_dict = preds[branch_name]
        
    # 情况 B: 输入是 Tuple (Teacher)
    elif isinstance(preds, tuple):
        if len(preds) < 2:
            raise ValueError(f"Teacher prediction tuple expected at least 2 elements, got {len(preds)}")
        # 通常第二个元素是处理后的字典
        processed_dict = preds[1]
        if not isinstance(processed_dict, dict):
            raise TypeError(f"Expected second element of teacher tuple to be dict, got {type(processed_dict)}")
        
        if branch_name not in processed_dict:
            raise KeyError(f"Branch '{branch_name}' not found in teacher prediction dict. Keys: {processed_dict.keys()}")
        data_dict = processed_dict[branch_name]
        
    else:
        raise TypeError(f"Unsupported prediction type: {type(preds)}. Expected dict or tuple.")

    if not isinstance(data_dict, dict):
        raise TypeError(f"Branch data must be a dict, got {type(data_dict)}")

    # 【严格检查】必须同时存在 boxes 和 scores
    if 'boxes' not in data_dict:
        raise KeyError(f"'boxes' key missing in branch '{branch_name}'. Available keys: {data_dict.keys()}")
    if 'scores' not in data_dict:
        raise KeyError(f"'scores' key missing in branch '{branch_name}'. Available keys: {data_dict.keys()}")

    boxes = data_dict['boxes']
    scores = data_dict['scores']

    # 检查 boxes 维度：如果是 [B, 4, N]，则转换为 [B, N, 4]
    if boxes.dim() == 3 and boxes.shape[1] == 4:
        boxes = boxes.permute(0, 2, 1).contiguous()
    elif boxes.dim() != 3 or boxes.shape[-1] != 4:
        # 如果既不是 [B, 4, N] 也不是 [B, N, 4]，则报错
        raise ValueError(f"Unexpected boxes shape: {boxes.shape}. Expected [B, 4, N] or [B, N, 4].")
    
    # 检查 scores 维度：如果是 [B, C, N]，则转换为 [B, N, C]
    # 注意：这里假设 C (类别数) 不等于 N (锚点数)。通常 N=8400, C=80。
    if scores.dim() == 3 and scores.shape[1] != scores.shape[2] and scores.shape[1] < 1000: 
        # 简单判断：如果中间维度远小于最后维度，且中间维度大概是类别数 (如 80)，则认为是 [B, C, N]
        # 更严谨的判断是检查 shape[1] 是否等于 num_classes，但这里我们没有 num_classes 变量传入
        # 最稳妥的方式：如果 shape[1] == 4 (不可能，那是 box) 或者 shape[1] 很小 (如 80)，而 shape[2] 很大 (8400)
        if scores.shape[2] > scores.shape[1]: 
             scores = scores.permute(0, 2, 1).contiguous()
    
    # 最终校验
    if boxes.dim() != 3 or boxes.shape[-1] != 4:
        raise ValueError(f"Final boxes shape check failed: Expected [B, N, 4], got {boxes.shape}")
    if scores.dim() != 3:
        raise ValueError(f"Final scores shape check failed: Expected [B, N, C], got {scores.shape}")
    
    return boxes, scores
    
    return boxes, scores

# ==========================================
# 3. 主 Loss 类 (YOLO26ConsistencyLoss)
# ==========================================

class YOLO26ConsistencyLoss(nn.Module):
    def __init__(self, box_weight: float = 1.0, cls_weight: float = 1.0, 
                 temperature: float = 1.0, confidence_threshold: float = 0.25, 
                 num_classes: int = 80, topk: int = 13):
        super().__init__()
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold
        self.num_classes = num_classes
        
        self.iou_loss = IOUloss(reduction="none", iou_type="ciou", xyxy=True)
        self.assigner = DynamicSoftLabelAssigner(topk=topk, num_classes=num_classes)

    def forward(self, student_pred: Union[Dict, Tuple], teacher_pred: Union[Dict, Tuple]) -> Tuple[torch.Tensor, Dict]:
        """
        计算无监督一致性损失。
        流程：严格提取 -> 坐标转换 -> 逐图动态分配 -> 计算 Loss
        """
        # 1. 【严格提取】如果结构不对，这里会直接报错，方便调试
        s_boxes_raw, s_scores = extract_predictions_strict(student_pred, branch_name='one2one')
        t_boxes_raw, t_scores = extract_predictions_strict(teacher_pred, branch_name='one2one')
        
        # 2. 坐标格式统一：YOLO 内部通常是 cxcywh，IoU 需要 xyxy
        s_boxes = cxcywh_to_xyxy(s_boxes_raw)
        t_boxes = cxcywh_to_xyxy(t_boxes_raw)
        
        B, N, _ = s_boxes.shape
        device = s_boxes.device
        
        total_box_loss = 0.0
        total_cls_loss = 0.0
        valid_samples_count = 0

        # 3. 逐图处理 (解决 Batch Mismatch 问题)
        for b in range(B):
            sb_box = s_boxes[b:b+1]      # [1, N, 4]
            sb_score = s_scores[b:b+1]   # [1, N, C]
            tb_box = t_boxes[b:b+1]
            tb_score = t_scores[b:b+1]

            # 生成伪标签 (Pseudo-GT)
            # 获取 Teacher 的最大置信度和对应类别
            t_conf, t_cls = tb_score.max(dim=-1) # [1, N]
            
            # 阈值筛选
            pseudo_gt_mask = (t_conf.squeeze(0) > self.confidence_threshold)
            
            if pseudo_gt_mask.sum() == 0:
                continue
            
            # 提取有效的 GT 数据和对应的 Box
            gt_bboxes = tb_box.squeeze(0)[pseudo_gt_mask] # [M, 4]
            gt_labels = t_cls.squeeze(0)[pseudo_gt_mask]  # [M]
            
            if gt_bboxes.shape[0] == 0:
                continue

           # 【修复点】构造与筛选后 GT 数量匹配的掩码 (全 True)，或者直接传 None
            # 原代码错误地传入了长度为 N 的 pseudo_gt_mask
            # 正确做法：既然已经筛选了，传入的 GT 都是有效的，不需要掩码，或者传入长度为 M 的全 True 掩码
            current_gt_mask = torch.ones_like(gt_labels, dtype=torch.bool) # 形状 [M]

            # 4. 动态标签分配 (ASA Module)
            assigned_labels, assigned_scores, fg_mask, matched_gt_indices = self.assigner(
                pred_scores=sb_score,             # [1, N, C]
                pred_bboxes=sb_box,               # [1, N, 4]
                gt_labels=gt_labels.unsqueeze(0), # [1, M]
                gt_bboxes=gt_bboxes.unsqueeze(0), # [1, M, 4]
                # gt_mask=pseudo_gt_mask.unsqueeze(0) # ❌ 错误：形状是 [1, N]，与 gt_labels [1, M] 不匹配
                gt_mask=current_gt_mask.unsqueeze(0) # ✅ 正确：形状是 [1, M]，与 gt_labels 匹配
            )
            fg_mask_b = fg_mask[0] # [N]
            if fg_mask_b.sum() == 0:
                continue
            
            # 5. 获取匹配的 Target Boxes (利用返回的 indices)
            matched_indices = matched_gt_indices[0][fg_mask_b]
            
            # 安全检查
            if matched_indices.numel() == 0 or matched_indices.min() < 0:
                continue
                
            target_boxes = gt_bboxes[matched_indices]       # [K, 4]
            # target_labels = gt_labels[matched_indices]    # 如果需要分类硬标签
            
            s_pos_boxes = sb_box.squeeze(0)[fg_mask_b]      # [K, 4]
            s_pos_scores = sb_score.squeeze(0)[fg_mask_b]   # [K, C]
            t_soft_scores = assigned_scores[0][fg_mask_b]   # [K, C]

            # 6. 计算损失
            # Box Loss (CIoU)
            box_loss_val = self.iou_loss(s_pos_boxes, target_boxes)
            total_box_loss += box_loss_val.sum()
            
            # Class Loss (KL Divergence with Soft Targets)
            eps = 1e-9
            log_student = F.log_softmax(s_pos_scores / self.temperature, dim=-1)
            kl_div = (t_soft_scores * (torch.log(t_soft_scores + eps) - log_student)).sum(dim=-1)
            total_cls_loss += kl_div.sum()
            
            valid_samples_count += fg_mask_b.sum()

        # 7. 归一化
        if valid_samples_count == 0:
            zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return zero_loss, {
                "cons_box": 0.0, "cons_cls": 0.0, "cons_total": 0.0, "valid_ratio": 0.0
            }

        box_loss = total_box_loss / valid_samples_count
        cls_loss = total_cls_loss / valid_samples_count
        total_loss = self.box_weight * box_loss + self.cls_weight * cls_loss

        return total_loss, {
            "cons_box": box_loss.item(),
            "cons_cls": cls_loss.item(),
            "cons_total": total_loss.item(),
            "valid_ratio": float(valid_samples_count) / (B * N)
        }

class IOUloss(nn.Module):
    def __init__(self, reduction: str = "none", iou_type: str = "ciou", xyxy: bool = True):
        super().__init__()
        self.reduction = reduction
        self.iou_type = iou_type
        self.xyxy = xyxy

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1, 4).float()
        target = target.view(-1, 4).float()
        
        # 假设输入已经是 xyxy (由 forward 中的转换保证)
        tl = torch.max(pred[:, :2], target[:, :2])
        br = torch.min(pred[:, 2:], target[:, 2:])
        hw = (br - tl).clamp(min=0)
        
        area_i = torch.prod(hw, 1)
        area_p = torch.prod(pred[:, 2:] - pred[:, :2], 1)
        area_g = torch.prod(target[:, 2:] - target[:, 2:], 1)
        
        iou = area_i / (area_p + area_g - area_i + 1e-16)

        if self.iou_type == "ciou":
            c_tl = torch.min(pred[:, :2], target[:, :2])
            c_br = torch.max(pred[:, 2:], target[:, 2:])
            convex_dis = torch.pow(c_br[:, 0]-c_tl[:, 0], 2) + torch.pow(c_br[:, 1]-c_tl[:, 1], 2) + 1e-7
            center_dis = torch.pow(pred[:, 0]-target[:, 0], 2) + torch.pow(pred[:, 1]-target[:, 1], 2)
            
            w_pred = pred[:, 2] - pred[:, 0]
            h_pred = pred[:, 3] - pred[:, 1]
            w_tgt = target[:, 2] - target[:, 0]
            h_tgt = target[:, 3] - target[:, 1]
            
            v = (4 / math.pi ** 2) * torch.pow(
                torch.atan(w_tgt / torch.clamp(h_tgt, min=1e-7)) - 
                torch.atan(w_pred / torch.clamp(h_pred, min=1e-7)), 2)
            
            with torch.no_grad():
                alpha = v / ((1 + 1e-7) - iou + v)
            
            ciou = iou - (center_dis / convex_dis + alpha * v)
            loss = 1 - ciou.clamp(min=-1.0, max=1.0)
        else:
            loss = 1 - iou

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss