import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional, Union


# ==========================================
# 1. [已移除] DynamicSoftLabelAssigner 
# 该类已被移除，其功能被下方的 _simple_iou_assign 替代
# ==========================================


# ==========================================
# 2. 坐标转换工具 (保持不变)
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
# 关键修改：
# 1. 移除了 self.assigner 初始化
# 2. 新增了 _simple_iou_assign 静态方法作为替代
# 3. forward 中调用新的静态方法
# ==========================================

class YOLO26ConsistencyLoss(nn.Module):
    def __init__(
        self, 
        box_weight: float = 1.0, 
        cls_weight: float = 1.0, 
        temperature: float = 1.0, 
        # 动态阈值参数
        confidence_threshold_start: float = 0.9,    
        confidence_threshold_end: float = 0.55,      
        threshold_warmup_epochs: int = 50,           
        total_epochs: int = 175,                     
        num_classes: int = 80, 
        topk: int = 13  # 此参数在消融版本中不再用于动态分配，但保留以兼容构造函数签名
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
        # [已移除] self.assigner = DynamicSoftLabelAssigner(...)

    @staticmethod
    def _simple_iou_assign(
        pred_bboxes: torch.Tensor, 
        gt_bboxes: torch.Tensor, 
        gt_labels: torch.Tensor, 
        num_classes: int
    ):
        """
        简化的基于 IoU 的贪婪分配策略 (消融实验用)
        替代 DynamicSoftLabelAssigner
        
        Args:
            pred_bboxes: [N, 4]
            gt_bboxes: [M, 4]
            gt_labels: [M]
            num_classes: int
            
        Returns:
            assigned_labels: [N]
            assigned_scores: [N, C]
            fg_mask: [N]
            matched_gt_indices: [N]
        """
        n_boxes = pred_bboxes.shape[0]
        device = pred_bboxes.device
        
        # 初始化输出
        assigned_labels = torch.full((n_boxes,), -1, dtype=torch.long, device=device)
        assigned_scores = torch.zeros((n_boxes, num_classes), dtype=torch.float32, device=device)
        fg_mask = torch.zeros((n_boxes,), dtype=torch.bool, device=device)
        matched_gt_indices = torch.full((n_boxes,), -1, dtype=torch.long, device=device)
        
        if gt_bboxes.shape[0] == 0:
            return assigned_labels, assigned_scores, fg_mask, matched_gt_indices

        # 1. 计算 IoU 矩阵 [N, M]
        # 扩展维度以便广播
        p_min = pred_bboxes[:, None, :2]
        p_max = pred_bboxes[:, None, 2:]
        g_min = gt_bboxes[None, :, :2]
        g_max = gt_bboxes[None, :, 2:]
        
        inter_min = torch.max(p_min, g_min)
        inter_max = torch.min(p_max, g_max)
        wh = (inter_max - inter_min).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        
        area_p = (p_max[:, :, 0] - p_min[:, :, 0]) * (p_max[:, :, 1] - p_min[:, :, 1])
        area_g = (g_max[:, :, 0] - g_min[:, :, 0]) * (g_max[:, :, 1] - g_min[:, :, 1])
        
        union = area_p + area_g - inter
        ious = inter / (union + 1e-9)
        
        # 2. 贪婪匹配：对每个预测框，找 IoU 最大的 GT
        # max_ious: [N], max_indices: [N]
        max_ious, max_indices = torch.max(ious, dim=1)
        
        # 3. 设定阈值，只有 IoU > 0 的才视为正样本 (也可以设更高阈值如 0.5，这里保持宽松以匹配原逻辑意图)
        # 原逻辑中只要匹配上就算 FG，这里我们只要 IoU > 0 即认为有重叠
        valid_match = max_ious > 0.0
        
        fg_mask = valid_match
        matched_gt_indices[valid_match] = max_indices[valid_match]
        assigned_labels[valid_match] = gt_labels[max_indices[valid_match]]
        
        # 生成 One-hot 分数
        if valid_match.any():
            matched_labels = gt_labels[max_indices[valid_match]]
            assigned_scores[valid_match] = F.one_hot(matched_labels, num_classes).float()
            
        return assigned_labels, assigned_scores, fg_mask, matched_gt_indices

    def set_epoch(self, epoch: int):
        """
        设置当前 epoch，更新动态阈值
        """
        self.current_epoch = epoch
        
        if epoch >= self.threshold_warmup_epochs:
            progress = 1.0
        else:
            progress = epoch / self.threshold_warmup_epochs
        
        self.confidence_threshold = self.confidence_threshold_start + \
            (self.confidence_threshold_end - self.confidence_threshold_start) * progress
        
        return self.confidence_threshold

    def get_current_threshold(self) -> float:
        return self.confidence_threshold

    def forward(
        self, 
        student_pred: Union[Dict, Tuple], 
        teacher_pred: Union[Dict, Tuple]
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算无监督一致性损失，使用简化的 IoU 分配策略
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
        total_pseudo_labels = 0

        for b in range(B):
            sb_box = s_boxes[b:b+1]   # [1, N, 4]
            sb_score = s_scores[b:b+1] # [1, N, C]
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
            
            gt_bboxes = tb_box.squeeze(0)[pseudo_gt_mask] # [M, 4]
            gt_labels = t_cls.squeeze(0)[pseudo_gt_mask]  # [M]

            # [修改点] 调用简化的静态分配方法，替代 self.assigner
            assigned_labels, assigned_scores, fg_mask, matched_gt_indices = self._simple_iou_assign(
                pred_bboxes=sb_box.squeeze(0),      # [N, 4]
                gt_bboxes=gt_bboxes,                # [M, 4]
                gt_labels=gt_labels,                # [M]
                num_classes=self.num_classes
            )
            
            # fg_mask 已经是 [N]
            fg_mask_b = fg_mask
            if fg_mask_b.sum() == 0:
                continue
            
            # 获取匹配到的 GT 索引
            matched_indices = matched_gt_indices[fg_mask_b]
            
            if matched_indices.numel() == 0 or matched_indices.min() < 0:
                continue
                
            target_boxes = gt_bboxes[matched_indices]
            s_pos_boxes = sb_box.squeeze(0)[fg_mask_b]
            s_pos_scores = sb_score.squeeze(0)[fg_mask_b]
            t_soft_scores = assigned_scores[fg_mask_b]

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
            "pseudo_labels": total_pseudo_labels // B,
            "threshold": self.confidence_threshold
        }