import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple


class IOUloss(nn.Module):
    """直接从loss_ssod.py借鉴的IoU损失"""
    def __init__(self, reduction="none", iou_type="ciou", xyxy=True):
        super().__init__()
        self.reduction = reduction
        self.iou_type = iou_type
        self.xyxy = xyxy

    def forward(self, pred, target):
        pred = pred.view(-1, 4).float()
        target = target.view(-1, 4).float()
        
        if self.xyxy:
            tl = torch.max(pred[:, :2], target[:, :2])
            br = torch.min(pred[:, 2:], target[:, 2:])
            area_p = torch.prod(pred[:, 2:] - pred[:, :2], 1)
            area_g = torch.prod(target[:, 2:] - target[:, :2], 1)
        else:
            tl = torch.max((pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2))
            br = torch.min((pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2))
            area_p = torch.prod(pred[:, 2:], 1)
            area_g = torch.prod(target[:, 2:], 1)

        hw = (br - tl).clamp(min=0)
        area_i = torch.prod(hw, 1)
        iou = area_i / (area_p + area_g - area_i + 1e-16)

        if self.iou_type == "iou":
            loss = 1 - iou ** 2
        elif self.iou_type == "giou":
            c_tl = torch.min(pred[:, :2], target[:, :2])
            c_br = torch.max(pred[:, 2:], target[:, 2:])
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_i) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)
        elif self.iou_type == "diou":
            c_tl = torch.min(pred[:, :2], target[:, :2])
            c_br = torch.max(pred[:, 2:], target[:, 2:])
            convex_dis = torch.pow(c_br[:, 0]-c_tl[:, 0], 2) + torch.pow(c_br[:, 1]-c_tl[:, 1], 2) + 1e-7
            center_dis = torch.pow(pred[:, 0]-target[:, 0], 2) + torch.pow(pred[:, 1]-target[:, 1], 2)
            diou = iou - (center_dis / convex_dis)
            loss = 1 - diou.clamp(min=-1.0, max=1.0)
        elif self.iou_type == "ciou":
            c_tl = torch.min(pred[:, :2], target[:, :2])
            c_br = torch.max(pred[:, 2:], target[:, 2:])
            convex_dis = torch.pow(c_br[:, 0]-c_tl[:, 0], 2) + torch.pow(c_br[:, 1]-c_tl[:, 1], 2) + 1e-7
            center_dis = torch.pow(pred[:, 0]-target[:, 0], 2) + torch.pow(pred[:, 1]-target[:, 1], 2)
            v = (4 / math.pi ** 2) * torch.pow(
                torch.atan(target[:, 2] / torch.clamp(target[:, 3], min=1e-7)) - 
                torch.atan(pred[:, 2] / torch.clamp(pred[:, 3], min=1e-7)), 2)
            with torch.no_grad():
                alpha = v / ((1 + 1e-7) - iou + v)
            ciou = iou - (center_dis / convex_dis + alpha * v)
            loss = 1 - ciou.clamp(min=-1.0, max=1.0)
        elif self.iou_type == "siou":
            # SIoU简化版，完整实现参考loss_ssod.py
            loss = 1 - iou

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


class YOLO26ConsistencyLoss(nn.Module):
    """
    YOLO26半监督一致性损失
    输入格式完全兼容V8DetectionLoss的pred格式
    """
    
    def __init__(
        self,
        box_weight: float = 1.0,
        cls_weight: float = 1.0,
        obj_weight: float = 0.5,
        temperature: float = 1.0,
        confidence_threshold: float = 0.25,
        iou_type: str = "ciou",
    ):
        super().__init__()
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.obj_weight = obj_weight
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold
        
        # IoU损失（借鉴loss_ssod.py）
        self.iou_loss = IOUloss(reduction="none", iou_type=iou_type, xyxy=True)

    def forward(
        self,
        student_pred: Dict[str, torch.Tensor],
        teacher_pred: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算一致性损失
        
        Args:
            student_pred: 学生模型输出，格式同V8DetectionLoss输入
                {
                    "boxes": [B, N, 4],      # xyxy format (dist2bbox解码后)
                    "scores": [B, N, num_classes],  # cls logits
                }
            teacher_pred: 教师模型输出，格式同上（已detach）
        
        Returns:
            total_loss: 加权总损失
            loss_dict: 各组件损失字典
        """
        #提取必要参数
        student_pred=student_pred["one2one"]
        teacher_pred=teacher_pred[1]["one2one"]

        device = student_pred["boxes"].device
        
        # 获取teacher置信度作为伪标签质量评估
        teacher_scores = teacher_pred["scores"]  # [B, N, num_classes]
        teacher_conf, teacher_cls = teacher_scores.max(dim=-1)  # [B, N]
        
        # 高置信度掩码（借鉴DomainFocalLoss的阈值思想）
        valid_mask = teacher_conf > self.confidence_threshold  # [B, N]
        
        if valid_mask.sum() == 0:
            zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return zero_loss, {
                "cons_box": 0.0, "cons_cls": 0.0, "cons_obj": 0.0,
                "cons_total": 0.0, "valid_ratio": 0.0
            }

        # ========== 1. Box一致性损失 ==========
        # 借鉴loss_ssod.py的IoUloss + smooth_l1混合策略
        s_boxes = student_pred["boxes"].view(-1, 4)   # [B*N, 4]
        t_boxes = teacher_pred["boxes"].view(-1, 4)   # [B*N, 4]
        
        # CIoU损失（主要）
        iou_loss = self.iou_loss(s_boxes, t_boxes)  # [B*N]
        
        # Smooth L1辅助（针对小目标，借鉴loss_ssod.py line 260+）
        l1_loss = F.smooth_l1_loss(s_boxes, t_boxes, reduction='none').mean(dim=-1)
        
        # 组合并掩码
        box_losses = iou_loss + 0.1 * l1_loss  # [B*N]
        box_loss = (box_losses * valid_mask.view(-1)).sum() / (valid_mask.sum() + 1e-6)

        # ========== 2. Classification一致性损失 ==========
        # KL散度（借鉴DomainFocalLoss的softmax处理）
        s_scores = student_pred["scores"] / self.temperature  # [B, N, C]
        t_scores = teacher_pred["scores"] / self.temperature  # [B, N, C]
        
        s_probs = F.softmax(s_scores, dim=-1)
        t_probs = F.softmax(t_scores, dim=-1)
        
        # KL(s||t) = sum(s * log(s/t))
        kl_div = (s_probs * (s_probs.log() - t_probs.log())).sum(dim=-1)  # [B, N]
        cls_loss = (kl_div * valid_mask).sum() / (valid_mask.sum() + 1e-6)

        # ========== 3. Objectness一致性损失（可选） ==========
        obj_loss = torch.tensor(0.0, device=device)
        if "obj" in student_pred and "obj" in teacher_pred:
            # BCE with soft targets（借鉴ComputeLoss的objectness处理）
            s_obj = torch.sigmoid(student_pred["obj"])  # [B, N]
            t_obj = torch.sigmoid(teacher_pred["obj"])    # [B, N]
            obj_bce = F.binary_cross_entropy(s_obj, t_obj, reduction='none')  # [B, N]
            obj_loss = (obj_bce * valid_mask).sum() / (valid_mask.sum() + 1e-6)

        # ========== 总损失 ==========
        total_loss = (
            self.box_weight * box_loss + 
            self.cls_weight * cls_loss + 
            self.obj_weight * obj_loss
        )

        loss_dict = {
            "cons_box": box_loss.item(),
            "cons_cls": cls_loss.item(),
            "cons_obj": obj_loss.item(),
            "cons_total": total_loss.item(),
            "valid_ratio": valid_mask.float().mean().item(),
        }

        return total_loss, loss_dict
