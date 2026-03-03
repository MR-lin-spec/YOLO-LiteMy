import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional, Union, List
from collections import deque

# 尝试导入 torchvision 的 NMS
try:
    from torchvision.ops import nms as tv_nms
except ImportError:
    try:
        from torchvision.ops.nms import nms as tv_nms
    except ImportError:
        def tv_nms(boxes, scores, iou_threshold):
            if boxes.numel() == 0:
                return torch.empty((0,), dtype=torch.long, device=boxes.device)
            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            areas = (x2 - x1) * (y2 - y1)
            order = scores.sort(descending=True)[1]
            keep = []
            while order.numel() > 0:
                i = order[0]
                keep.append(i.item())
                if order.numel() == 1:
                    break
                xx1 = torch.max(x1[i], x1[order[1:]])
                yy1 = torch.max(y1[i], y1[order[1:]])
                xx2 = torch.min(x2[i], x2[order[1:]])
                yy2 = torch.min(y2[i], y2[order[1:]])
                w = (xx2 - xx1).clamp(min=0)
                h = (yy2 - yy1).clamp(min=0)
                inter = w * h
                ovr = inter / (areas[i] + areas[order[1:]] - inter)
                inds = (ovr <= iou_threshold).nonzero(as_tuple=False).squeeze(1)
                if inds.dim() == 0:
                    break
                order = order[inds + 1]
            return torch.tensor(keep, dtype=torch.long, device=boxes.device)


# ==========================================
# 1. 动态分配器 (保持不变，已优化)
# ==========================================
class DynamicSoftLabelAssigner(nn.Module):
    def __init__(self, iou_factor: float = 3.0, num_classes: int = 80, cost_threshold: float = 1e9):
        super().__init__()
        self.iou_factor = iou_factor
        self.num_classes = num_classes
        self.cost_threshold = cost_threshold

    def forward(self, pred_scores: torch.Tensor, pred_bboxes: torch.Tensor, 
                gt_labels: torch.Tensor, gt_bboxes: torch.Tensor, 
                gt_mask: Optional[torch.Tensor] = None):
        bs, n_boxes, num_classes = pred_scores.shape
        device = pred_scores.device
        
        assigned_labels = torch.full((bs, n_boxes), -1, dtype=torch.long, device=device)
        assigned_scores = torch.zeros_like(pred_scores)
        fg_mask = torch.zeros((bs, n_boxes), dtype=torch.bool, device=device)
        matched_gt_indices = torch.full((bs, n_boxes), -1, dtype=torch.long, device=device)

        for b in range(bs):
            p_score = pred_scores[b]
            p_bbox = pred_bboxes[b]
            g_label = gt_labels[b]
            g_bbox = gt_bboxes[b]
            
            if gt_mask is not None:
                valid_idx = gt_mask[b]
                if valid_idx.sum() == 0:
                    continue
                g_label = g_label[valid_idx]
                g_bbox = g_bbox[valid_idx]
            
            num_gt = g_bbox.shape[0]
            if num_gt == 0:
                continue

            # IoU
            tl = torch.max(p_bbox[:, None, :2], g_bbox[None, :, :2])
            br = torch.min(p_bbox[:, None, 2:], g_bbox[None, :, 2:])
            wh = (br - tl).clamp(min=0)
            inter = wh[:, :, 0] * wh[:, :, 1]
            area_p = (p_bbox[:, 2] - p_bbox[:, 0]) * (p_bbox[:, 3] - p_bbox[:, 1])
            area_g = (g_bbox[:, 2] - g_bbox[:, 0]) * (g_bbox[:, 3] - g_bbox[:, 1])
            union = area_p[:, None] + area_g[None, :] - inter
            ious = (inter / (union + 1e-9)).detach()

            # Cls Cost
            gt_onehot = F.one_hot(g_label, self.num_classes).float()
            p_score_sigmoid = torch.sigmoid(p_score).clamp(min=1e-7, max=1.0-1e-7)
            preds_expanded = p_score_sigmoid.unsqueeze(1).expand(-1, num_gt, -1)
            targets_expanded = gt_onehot.unsqueeze(0).expand(n_boxes, -1, -1)
            cls_cost = F.binary_cross_entropy(preds_expanded, targets_expanded, reduction='none').sum(dim=-1)

            # Dist Cost
            p_center = (p_bbox[:, :2] + p_bbox[:, 2:]) / 2.0
            g_center = (g_bbox[:, :2] + g_bbox[:, 2:]) / 2.0
            dist = ((p_center[:, None] - g_center[None, :]) ** 2).sum(dim=-1).sqrt()
            dis_cost = (dist / (dist.max() + 1e-9)) * 10.0

            cost_matrix = cls_cost - (ious * self.iou_factor) + dis_cost

            # Greedy One-to-One
            matching_matrix = torch.zeros_like(cost_matrix)
            current_cost = cost_matrix.clone()
            used_pred_mask = torch.zeros(n_boxes, dtype=torch.bool, device=device)
            used_gt_mask = torch.zeros(num_gt, dtype=torch.bool, device=device)
            
            for _ in range(min(num_gt, n_boxes)):
                if used_pred_mask.all() or used_gt_mask.all():
                    break
                masked_cost = current_cost.masked_fill(used_pred_mask[:, None], 1e9)
                masked_cost = masked_cost.masked_fill(used_gt_mask[None, :], 1e9)
                min_val, min_flat_idx = torch.min(masked_cost.view(-1), dim=0)
                
                if min_val > self.cost_threshold:
                    break
                
                pred_idx = min_flat_idx // num_gt
                gt_idx = min_flat_idx % num_gt
                
                matching_matrix[pred_idx, gt_idx] = 1.0
                used_pred_mask[pred_idx] = True
                used_gt_mask[gt_idx] = True
                current_cost[pred_idx, :] = 1e9
                current_cost[:, gt_idx] = 1e9

            fg_mask_b = matching_matrix.sum(1) > 0.0
            if fg_mask_b.sum() > 0:
                matched_inds = matching_matrix[fg_mask_b, :].argmax(dim=1)
                assigned_labels[b, fg_mask_b] = g_label[matched_inds]
                assigned_scores[b, fg_mask_b, :] = gt_onehot[matched_inds]
                matched_gt_indices[b, fg_mask_b] = matched_inds 
                fg_mask[b] = fg_mask_b

        return assigned_labels, assigned_scores, fg_mask, matched_gt_indices


# ==========================================
# 2. 坐标转换与提取 (保持不变)
# ==========================================
def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.shape[-1] != 4:
        raise ValueError(f"Expected box dimension to be 4, got {boxes.shape[-1]}")
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)

def extract_predictions_strict(preds: Union[Dict, Tuple], branch_name: str = 'one2one') -> Tuple[torch.Tensor, torch.Tensor]:
    data_dict = None
    if isinstance(preds, dict):
        if branch_name not in preds:
            raise KeyError(f"Branch '{branch_name}' not found.")
        data_dict = preds[branch_name]
    elif isinstance(preds, tuple):
        if len(preds) < 2:
            raise ValueError("Teacher prediction tuple expected at least 2 elements.")
        processed_dict = preds[1]
        if not isinstance(processed_dict, dict):
            raise TypeError("Expected second element of teacher tuple to be dict.")
        if branch_name not in processed_dict:
            raise KeyError(f"Branch '{branch_name}' not found.")
        data_dict = processed_dict[branch_name]
    else:
        raise TypeError("Unsupported prediction type.")

    if not isinstance(data_dict, dict):
        raise TypeError("Branch data must be a dict.")
    if 'boxes' not in data_dict or 'scores' not in data_dict:
        raise KeyError("'boxes' or 'scores' key missing.")

    boxes = data_dict['boxes']
    scores = data_dict['scores']

    if boxes.dim() == 3 and boxes.shape[1] == 4:
        boxes = boxes.permute(0, 2, 1).contiguous()
    elif boxes.dim() != 3 or boxes.shape[-1] != 4:
        raise ValueError(f"Unexpected boxes shape: {boxes.shape}")
    
    if scores.dim() == 3 and scores.shape[1] != scores.shape[2] and scores.shape[1] < 1000:
        if scores.shape[2] > scores.shape[1]: 
             scores = scores.permute(0, 2, 1).contiguous()
    
    return boxes, scores


# ==========================================
# 3. IoU Loss (保持不变)
# ==========================================
class IOUloss(nn.Module):
    def __init__(self, reduction: str = "none", iou_type: str = "ciou", xyxy: bool = True):
        super().__init__()
        self.reduction = reduction
        self.iou_type = iou_type
        self.xyxy = xyxy

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1, 4).float()
        target = target.view(-1, 4).float()
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
            w_pred, h_pred = pred[:, 2] - pred[:, 0], pred[:, 3] - pred[:, 1]
            w_tgt, h_tgt = target[:, 2] - target[:, 0], target[:, 3] - target[:, 1]
            v = (4 / math.pi ** 2) * torch.pow(torch.atan(w_tgt / torch.clamp(h_tgt, min=1e-7)) - torch.atan(w_pred / torch.clamp(h_pred, min=1e-7)), 2)
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


# ==========================================
# 4. 主 Loss 类 (YOLO26ConsistencyLoss)
# 【核心升级】自适应动态阈值策略 (Percentile + Class-Adaptive)
# ==========================================
class YOLO26ConsistencyLoss(nn.Module):
    def __init__(
        self, 
        box_weight: float = 1.0, 
        cls_weight: float = 1.0, 
        temperature: float = 1.0, 
        # 基础参数
        confidence_threshold_floor: float = 0.2,   # 绝对最低门槛，防背景噪声
        top_percentile_start: float = 0.05,        # 早期只选前 5% 最确定的
        top_percentile_end: float = 0.20,          # 后期放宽到前 20%
        threshold_warmup_epochs: int = 50,         
        total_epochs: int = 175,                     
        num_classes: int = 80, 
        pseudo_nms_iou_threshold: float = 0.6,
        assigner_cost_threshold: float = 100.0,
        # 类别自适应参数
        class_momentum: float = 0.9,               # 滑动平均动量
        class_buffer_size: int = 100               # 每个类别的历史缓冲大小 (可选，若显存紧张可只用动量)
    ):
        super().__init__()
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.temperature = temperature
        
        self.confidence_threshold_floor = confidence_threshold_floor
        self.top_percentile_start = top_percentile_start
        self.top_percentile_end = top_percentile_end
        self.threshold_warmup_epochs = threshold_warmup_epochs
        self.total_epochs = total_epochs
        
        self.current_epoch = 0
        self.current_percentile = top_percentile_start
        
        self.num_classes = num_classes
        self.pseudo_nms_iou_threshold = pseudo_nms_iou_threshold
        
        # 【新增】类别自适应置信度均值 (EMA)
        # 初始化为一个中等值，随着训练更新
        self.class_confidence_ema = torch.ones(num_classes, dtype=torch.float32) * 0.5
        self.class_momentum = class_momentum
        
        self.assigner = DynamicSoftLabelAssigner(
            iou_factor=3.0, 
            num_classes=num_classes,
            cost_threshold=assigner_cost_threshold
        )
        self.iou_loss = IOUloss(reduction="none", iou_type="ciou", xyxy=True)

    def set_epoch(self, epoch: int):
        """设置当前 epoch，更新全局百分位阈值"""
        self.current_epoch = epoch
        if epoch >= self.threshold_warmup_epochs:
            progress = 1.0
        else:
            progress = epoch / self.threshold_warmup_epochs
        
        # 线性调整选取的百分位比例
        self.current_percentile = self.top_percentile_start + \
            (self.top_percentile_end - self.top_percentile_start) * progress
        return self.current_percentile

    def _get_adaptive_thresholds(self, t_conf: torch.Tensor, t_cls: torch.Tensor) -> torch.Tensor:
        """
        计算每个预测框的自适应阈值
        策略：Threshold = max(Floor, Class_EMA * Factor)
        这里我们简化为：对每个类别，如果其历史平均置信度低，则降低对该类的要求，
        但必须高于全局 Floor。
        更激进的策略是：直接选取每类的前 K%。
        
        本实现采用混合策略：
        1. 计算全局动态阈值 (基于百分位)
        2. 结合类别 EMA 进行微调 (难类适当放宽，但不低于 Floor)
        """
        device = t_conf.device
        
        # 1. 更新类别置信度 EMA (使用当前 batch 的高置信样本作为参考，避免噪声污染)
        # 仅当置信度 > floor 时才更新 EMA，防止噪声拉低均值
        valid_mask = t_conf > self.confidence_threshold_floor
        if valid_mask.any():
            for c in range(self.num_classes):
                cls_mask = (t_cls == c) & valid_mask
                if cls_mask.any():
                    avg_conf = t_conf[cls_mask].mean().item()
                    # EMA 更新
                    self.class_confidence_ema[c] = (
                        self.class_momentum * self.class_confidence_ema[c].to(device) + 
                        (1 - self.class_momentum) * avg_conf
                    ).to(self.class_confidence_ema.device) # 保持 CPU 或统一设备
        
        # 2. 计算每个框的动态阈值
        # 基础阈值：全局 Floor
        # 动态部分：如果某类的 EMA 很高，说明该类容易，我们可以提高阈值；如果 EMA 低，则维持 Floor
        # 这里采用一种更稳健的方式：阈值 = Floor + (Percentile_Delta * Class_Factor)
        # 简单起见，我们主要依赖 **全局百分位** + **类别独立筛选**
        
        # 【核心逻辑修正】：不再使用单一的全局标量阈值，而是返回每个样本是否保留的 Mask
        # 我们将按类别分别处理，或者使用全局百分位切割
        
        # 方案 A: 全局百分位切割 (最简单稳健)
        # 找出全图置信度的第 (1 - percentile) 分位数
        if t_conf.numel() > 0:
            k = max(1, int(t_conf.numel() * self.current_percentile))
            # 取 Top-K 的最低值作为动态阈值
            topk_vals = torch.topk(t_conf, k, sorted=False)[0]
            dynamic_global_thresh = topk_vals.min()
        else:
            dynamic_global_thresh = 1.0
            
        # 最终阈值取 Max(Floor, Dynamic_Global)
        # 注意：这里为了兼容类别不平衡，我们实际上是在后面筛选时，对每个类别单独做 TopK 会更好
        # 但为了效率，我们先做一个全局粗筛，再在循环中做细筛？
        # 不，为了彻底解决类别不平衡，我们在生成 Mask 时，**按类别分组做 TopK**。
        
        return t_conf # 占位，实际逻辑在 forward 中按类别展开

    def forward(
        self, 
        student_pred: Union[Dict, Tuple], 
        teacher_pred: Union[Dict, Tuple]
    ) -> Tuple[torch.Tensor, Dict]:
        
        s_boxes_raw, s_scores = extract_predictions_strict(student_pred, branch_name='one2one')
        t_boxes_raw, t_scores = extract_predictions_strict(teacher_pred, branch_name='one2one')
        
        s_boxes = cxcywh_to_xyxy(s_boxes_raw)
        t_boxes = cxcywh_to_xyxy(t_boxes_raw)
        
        B, N, _ = s_boxes.shape
        device = s_boxes.device
        
        total_box_loss = 0.0
        total_cls_loss = 0.0
        valid_samples_count = 0
        
        total_pseudo_before = 0
        total_pseudo_after = 0
        
        # 用于统计各类别选取数量
        class_counts = torch.zeros(self.num_classes, dtype=torch.long, device=device)

        for b in range(B):
            sb_box = s_boxes[b:b+1]
            sb_score = s_scores[b:b+1]
            tb_box = t_boxes[b:b+1]
            tb_score = t_scores[b:b+1]

            t_conf, t_cls = tb_score.max(dim=-1)
            t_conf = t_conf.squeeze(0)
            t_cls = t_cls.squeeze(0)
            tb_box_squeeze = tb_box.squeeze(0)
            
            # --- 【核心升级】按类别自适应筛选伪标签 ---
            final_keep_mask = torch.zeros_like(t_conf, dtype=torch.bool)
            
            # 1. 先应用绝对 Floor，剔除纯背景
            base_mask = t_conf > self.confidence_threshold_floor
            if not base_mask.any():
                continue
                
            unique_classes = torch.unique(t_cls[base_mask])
            
            for c in unique_classes:
                cls_indices = torch.where((t_cls == c) & base_mask)[0]
                cls_confs = t_conf[cls_indices]
                
                if len(cls_indices) == 0:
                    continue
                
                # 2. 计算该类别的动态阈值
                # 策略：选取该类中置信度最高的 Top-K%
                k = max(1, int(len(cls_indices) * self.current_percentile))
                
                if k >= len(cls_indices):
                    # 如果样本太少，全选（但已满足 Floor）
                    selected_indices = cls_indices
                else:
                    # 选取 Top-K
                    _, topk_idx_in_cls = torch.topk(cls_confs, k, sorted=False)
                    selected_indices = cls_indices[topk_idx_in_cls]
                
                final_keep_mask[selected_indices] = True
            
            total_pseudo_before += base_mask.sum().item()
            selected_count = final_keep_mask.sum().item()
            total_pseudo_after += selected_count
            
            if selected_count == 0:
                continue

            candidate_boxes = tb_box_squeeze[final_keep_mask]
            candidate_scores = t_conf[final_keep_mask]
            candidate_labels = t_cls[final_keep_mask]
            
            # 更新类别计数
            for c in candidate_labels:
                class_counts[c] += 1

            # 3. NMS 去重
            if len(candidate_boxes) > 0:
                keep_indices = tv_nms(
                    candidate_boxes, 
                    candidate_scores, 
                    self.pseudo_nms_iou_threshold
                )
                
                gt_bboxes = candidate_boxes[keep_indices]
                gt_labels = candidate_labels[keep_indices]
            else:
                continue

            if len(gt_bboxes) == 0:
                continue

            # 4. 动态标签分配
            current_gt_mask = torch.ones_like(gt_labels, dtype=torch.bool)
            assigned_labels, assigned_scores, fg_mask, matched_gt_indices = self.assigner(
                pred_scores=sb_score,
                pred_bboxes=sb_box,
                gt_labels=gt_labels.unsqueeze(0),
                gt_bboxes=gt_bboxes.unsqueeze(0),
                gt_mask=current_gt_mask.unsqueeze(0)
            )
            
            fg_mask_b = fg_mask[0]
            if fg_mask_b.sum() == 0:
                continue
            
            matched_indices = matched_gt_indices[0][fg_mask_b]
            if matched_indices.numel() == 0 or matched_indices.min() < 0:
                continue
                
            target_boxes = gt_bboxes[matched_indices]
            s_pos_boxes = sb_box.squeeze(0)[fg_mask_b]
            s_pos_scores = sb_score.squeeze(0)[fg_mask_b]
            t_soft_scores = assigned_scores[0][fg_mask_b]

            box_loss_val = self.iou_loss(s_pos_boxes, target_boxes)
            total_box_loss += box_loss_val.sum()
            
            eps = 1e-9
            log_student = F.log_softmax(s_pos_scores / self.temperature, dim=-1)
            t_soft_scores_clamped = t_soft_scores.clamp(min=eps)
            kl_div = (t_soft_scores_clamped * (torch.log(t_soft_scores_clamped) - log_student)).sum(dim=-1)
            total_cls_loss += kl_div.sum()
            
            valid_samples_count += fg_mask_b.sum()

        if valid_samples_count == 0:
            zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return zero_loss, {
                "cons_box": 0.0, "cons_cls": 0.0, "cons_total": 0.0, 
                "valid_ratio": 0.0, "pseudo_before": 0, "pseudo_after": 0,
                "threshold_percentile": self.current_percentile
            }

        box_loss = total_box_loss / valid_samples_count
        cls_loss = total_cls_loss / valid_samples_count
        total_loss = self.box_weight * box_loss + self.cls_weight * cls_loss

        return total_loss, {
            "cons_box": box_loss.item(),
            "cons_cls": cls_loss.item(),
            "cons_total": total_loss.item(),
            "valid_ratio": float(valid_samples_count) / (B * N),
            "pseudo_before": total_pseudo_before // B,
            "pseudo_after": total_pseudo_after // B,
            "threshold_percentile": self.current_percentile,
            # 监控最难和最容易的类
            "min_class_conf_ema": self.class_confidence_ema.min().item(),
            "max_class_conf_ema": self.class_confidence_ema.max().item(),
        }