import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional, Union


# ==========================================
# 1. 动态分配器 (DynamicSoftLabelAssigner)
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
            matched_gt_indices: [B, N]
        """
        bs, n_boxes, _ = pred_scores.shape
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

            tl = torch.max(p_bbox[:, None, :2], g_bbox[None, :, :2])
            br = torch.min(p_bbox[:, None, 2:], g_bbox[None, :, 2:])
            wh = (br - tl).clamp(min=0)
            inter = wh[:, :, 0] * wh[:, :, 1]
            area_p = (p_bbox[:, 2] - p_bbox[:, 0]) * (p_bbox[:, 3] - p_bbox[:, 1])
            area_g = (g_bbox[:, 2] - g_bbox[:, 0]) * (g_bbox[:, 3] - g_bbox[:, 1])
            union = area_p[:, None] + area_g[None, :] - inter
            ious = (inter / (union + 1e-9)).detach()

            gt_onehot = F.one_hot(g_label, self.num_classes).float()
            p_score_sigmoid = torch.sigmoid(p_score)
            eps = 1e-7
            p_score_sigmoid = p_score_sigmoid.clamp(min=eps, max=1.0 - eps)
            
            bce_preds = p_score_sigmoid.unsqueeze(1).expand(-1, num_gt, -1)
            bce_targets = gt_onehot.unsqueeze(0).expand(n_boxes, -1, -1)
            cls_cost = F.binary_cross_entropy(bce_preds, bce_targets, reduction='none').sum(dim=-1)

            p_center = (p_bbox[:, :2] + p_bbox[:, 2:]) / 2.0
            g_center = (g_bbox[:, :2] + g_bbox[:, 2:]) / 2.0
            dist = ((p_center[:, None] - g_center[None, :]) ** 2).sum(dim=-1).sqrt()
            dis_cost = (dist / (dist.max() + 1e-9)) * 10.0

            cost_matrix = cls_cost + ious * self.iou_factor + dis_cost

            matching_matrix = torch.zeros_like(cost_matrix)
            candidate_topk = min(self.topk, n_boxes)
            topk_ious, _ = torch.topk(ious, candidate_topk, dim=0)
            dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
            
            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(cost_matrix[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
                matching_matrix[:, gt_idx][pos_idx] = 1.0
            
            match_gt_mask = matching_matrix.sum(1) > 1
            if match_gt_mask.sum() > 0:
                cost_min, cost_argmin = torch.min(cost_matrix[match_gt_mask, :], dim=1)
                matching_matrix[match_gt_mask, :] *= 0.0
                matching_matrix[match_gt_mask, cost_argmin] = 1.0
            
            fg_mask_b = matching_matrix.sum(1) > 0.0
            if fg_mask_b.sum() > 0:
                matched_inds = matching_matrix[fg_mask_b, :].argmax(dim=1)
                
                assigned_labels[b, fg_mask_b] = g_label[matched_inds]
                assigned_scores[b, fg_mask_b, :] = gt_onehot[matched_inds]
                matched_gt_indices[b, fg_mask_b] = matched_inds 
                fg_mask[b] = fg_mask_b

        return assigned_labels, assigned_scores, fg_mask, matched_gt_indices


# ==========================================
# 2. 坐标转换工具
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
            raise KeyError(f"Branch '{branch_name}' not found in student prediction dict. Keys: {preds.keys()}")
        data_dict = preds[branch_name]
        
    elif isinstance(preds, tuple):
        if len(preds) < 2:
            raise ValueError(f"Teacher prediction tuple expected at least 2 elements, got {len(preds)}")
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

    if 'boxes' not in data_dict:
        raise KeyError(f"'boxes' key missing in branch '{branch_name}'. Available keys: {data_dict.keys()}")
    if 'scores' not in data_dict:
        raise KeyError(f"'scores' key missing in branch '{branch_name}'. Available keys: {data_dict.keys()}")

    boxes = data_dict['boxes']
    scores = data_dict['scores']

    if boxes.dim() == 3 and boxes.shape[1] == 4:
        boxes = boxes.permute(0, 2, 1).contiguous()
    elif boxes.dim() != 3 or boxes.shape[-1] != 4:
        raise ValueError(f"Unexpected boxes shape: {boxes.shape}. Expected [B, 4, N] or [B, N, 4].")
    
    if scores.dim() == 3 and scores.shape[1] != scores.shape[2] and scores.shape[1] < 1000:
        if scores.shape[2] > scores.shape[1]: 
             scores = scores.permute(0, 2, 1).contiguous()
    
    if boxes.dim() != 3 or boxes.shape[-1] != 4:
        raise ValueError(f"Final boxes shape check failed: Expected [B, N, 4], got {boxes.shape}")
    if scores.dim() != 3:
        raise ValueError(f"Final scores shape check failed: Expected [B, N, C], got {scores.shape}")
    
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


# ==========================================
# 4. 修改后的主 Loss 类 (YOLO26ConsistencyLoss)
# 关键修改：动态阈值 + epoch 感知
# ==========================================

class YOLO26ConsistencyLoss(nn.Module):
    def __init__(
        self, 
        box_weight: float = 1.0, 
        cls_weight: float = 1.0, 
        temperature: float = 1.0, 
        # 动态阈值参数：早期高阈值减少噪声，后期降低增加召回
        confidence_threshold_start: float = 0.5,    # 初始高阈值
        confidence_threshold_end: float = 0.25,      # 最终低阈值
        threshold_warmup_epochs: int = 50,           # 阈值过渡周期
        total_epochs: int = 175,                     # 总训练轮数
        num_classes: int = 80, 
        topk: int = 13
    ):
        super().__init__()
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.temperature = temperature
        
        # 动态阈值相关参数
        self.confidence_threshold_start = confidence_threshold_start
        self.confidence_threshold_end = confidence_threshold_end
        self.threshold_warmup_epochs = threshold_warmup_epochs
        self.total_epochs = total_epochs
        
        # 当前状态
        self.current_epoch = 0
        self.confidence_threshold = confidence_threshold_start  # 初始值
        
        self.num_classes = num_classes
        
        self.iou_loss = IOUloss(reduction="none", iou_type="ciou", xyxy=True)
        self.assigner = DynamicSoftLabelAssigner(topk=topk, num_classes=num_classes)

    def set_epoch(self, epoch: int):
        """
        设置当前 epoch，更新动态阈值
        应在每个 epoch 开始时调用
        """
        self.current_epoch = epoch
        
        # 计算阈值过渡进度 (0 到 1)
        if epoch >= self.threshold_warmup_epochs:
            progress = 1.0
        else:
            progress = epoch / self.threshold_warmup_epochs
        
        # 线性降低阈值：从高阈值过渡到低阈值
        # 也可以使用余弦退火：progress = (1 - math.cos(progress * math.pi)) / 2
        self.confidence_threshold = self.confidence_threshold_start + \
            (self.confidence_threshold_end - self.confidence_threshold_start) * progress
        
        return self.confidence_threshold

    def get_current_threshold(self) -> float:
        """获取当前阈值，用于日志记录"""
        return self.confidence_threshold

    def forward(
        self, 
        student_pred: Union[Dict, Tuple], 
        teacher_pred: Union[Dict, Tuple]
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算无监督一致性损失，使用动态阈值
        """
        # 严格提取预测
        s_boxes_raw, s_scores = extract_predictions_strict(student_pred, branch_name='one2one')
        t_boxes_raw, t_scores = extract_predictions_strict(teacher_pred, branch_name='one2one')
        
        # 坐标转换
        s_boxes = cxcywh_to_xyxy(s_boxes_raw)
        t_boxes = cxcywh_to_xyxy(t_boxes_raw)
        
        B, N, _ = s_boxes.shape
        device = s_boxes.device
        
        total_box_loss = 0.0
        total_cls_loss = 0.0
        valid_samples_count = 0
        total_pseudo_labels = 0  # 统计伪标签数量，用于监控

        for b in range(B):
            sb_box = s_boxes[b:b+1]
            sb_score = s_scores[b:b+1]
            tb_box = t_boxes[b:b+1]
            tb_score = t_scores[b:b+1]

            # 生成伪标签
            t_conf, t_cls = tb_score.max(dim=-1)
            
            # 使用动态阈值筛选
            pseudo_gt_mask = (t_conf.squeeze(0) > self.confidence_threshold)
            num_pseudo = pseudo_gt_mask.sum().item()
            total_pseudo_labels += num_pseudo
            
            if num_pseudo == 0:
                continue
            
            gt_bboxes = tb_box.squeeze(0)[pseudo_gt_mask]
            gt_labels = t_cls.squeeze(0)[pseudo_gt_mask]

            # 构造有效掩码（与筛选后的GT匹配）
            current_gt_mask = torch.ones_like(gt_labels, dtype=torch.bool)

            # 动态标签分配
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

            # 计算损失
            box_loss_val = self.iou_loss(s_pos_boxes, target_boxes)
            total_box_loss += box_loss_val.sum()
            
            eps = 1e-9
            log_student = F.log_softmax(s_pos_scores / self.temperature, dim=-1)
            kl_div = (t_soft_scores * (torch.log(t_soft_scores + eps) - log_student)).sum(dim=-1)
            total_cls_loss += kl_div.sum()
            
            valid_samples_count += fg_mask_b.sum()

        # 归一化
        if valid_samples_count == 0:
            zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return zero_loss, {
                "cons_box": 0.0, 
                "cons_cls": 0.0, 
                "cons_total": 0.0, 
                "valid_ratio": 0.0,
                "pseudo_labels": 0,
                "threshold": self.confidence_threshold
            }

        box_loss = total_box_loss / valid_samples_count
        cls_loss = total_cls_loss / valid_samples_count
        total_loss = self.box_weight * box_loss + self.cls_weight * cls_loss

        return total_loss, {
            "cons_box": box_loss.item(),
            "cons_cls": cls_loss.item(),
            "cons_total": total_loss.item(),
            "valid_ratio": float(valid_samples_count) / (B * N),
            "pseudo_labels": total_pseudo_labels // B,  # 平均每张图的伪标签数
            "threshold": self.confidence_threshold
        }