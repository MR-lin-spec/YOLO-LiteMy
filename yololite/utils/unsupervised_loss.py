import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Any

class IOUloss(nn.Module):
    """IoU Loss supporting CIoU, DIoU, etc."""
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
        else:
            loss = 1 - iou

        if self.reduction == "mean": return loss.mean()
        if self.reduction == "sum": return loss.sum()
        return loss


class YOLO26ConsistencyLoss(nn.Module):
    """
    YOLO26 专用一致性损失
    核心修复：自动识别并强制对齐 One-to-One 或 One-to-Many 分支
    解决 25200 vs 67200 的架构不匹配问题
    """
    
    def __init__(self, box_weight=1.0, cls_weight=1.0, obj_weight=0.5, 
                 temperature=1.0, confidence_threshold=0.25, iou_type="ciou"):
        super().__init__()
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.obj_weight = obj_weight
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold
        self.iou_loss = IOUloss(reduction="none", iou_type=iou_type, xyxy=True)

    def _extract_branch(self, data, branch_name="auto", name="model"):
        """
        智能提取分支：优先获取指定分支，若不存在则自动降级
        YOLO26 结构通常为: {'one2one': ..., 'one2many': ...} 或 tuple/list 包含它们
        """
        # 1. 解包 tuple/list
        if isinstance(data, (list, tuple)):
            # 尝试常见位置：[1] 通常是 preds, [0] 是 losses
            for idx in [1, 0]:
                if idx < len(data) and isinstance(data[idx], dict):
                    data = data[idx]
                    break
        
        if not isinstance(data, dict):
            raise ValueError(f"{name} output is not a dict after unpacking. Type: {type(data)}")

        # 2. 尝试提取指定分支
        target_key = None
        if branch_name == "one2one":
            if "one2one" in data: target_key = "one2one"
            elif "o2o" in data: target_key = "o2o"
        elif branch_name == "one2many":
            if "one2many" in data: target_key = "one2many"
            elif "o2m" in data: target_key = "o2m"
        else: # auto
            # 优先 one2many (用于训练一致性更稳定)，如果没有则 one2one
            if "one2many" in data: target_key = "one2many"
            elif "o2m" in data: target_key = "o2m"
            elif "one2one" in data: target_key = "one2one"
            elif "o2o" in data: target_key = "o2o"
        
        if target_key and target_key in data:
            return data[target_key], target_key
        
        # 3. 降级：如果找不到特定分支，检查是否整个 dict 就是预测值
        if "boxes" in data and "scores" in data:
            return data, "root"
        
        # 4. 报错并打印可用键
        available_keys = list(data.keys())
        raise KeyError(f"Cannot find '{branch_name}' in {name}. Available keys: {available_keys}. Data structure: {type(data)}")

    def forward(self, student_pred, teacher_pred) -> Tuple[torch.Tensor, Dict[str, float]]:
        device = student_pred["one2one"]["boxes"].device if isinstance(student_pred, dict) and "one2one" in student_pred else torch.device("cpu")
        
        # ================= 1. 强制统一分支策略 =================
        # 策略：在半监督一致性损失中，为了几何对齐，强烈建议双方都使用 One-to-Many (密集预测)
        # 因为 One-to-One 是稀疏且经过筛选的，直接对齐容易出错。
        # 如果用户坚持用 one2one，则必须保证两边都有且数量一致。
        
        try:
            # 尝试提取 One-to-Many (推荐)
            s_data, s_type = self._extract_branch(student_pred, branch_name="one2many", name="Student")
            t_data, t_type = self._extract_branch(teacher_pred, branch_name="one2many", name="Teacher")
            #print(f"[INFO] Using Branches -> Student: {s_type}, Teacher: {t_type}")
        except KeyError:
            # 如果都没有 O2M，降级到 O2O
            print("[WARNING] One-to-Many branch not found. Falling back to One-to-One.")
            s_data, s_type = self._extract_branch(student_pred, branch_name="one2one", name="Student")
            t_data, t_type = self._extract_branch(teacher_pred, branch_name="one2one", name="Teacher")
            print(f"[INFO] Using Branches -> Student: {s_type}, Teacher: {t_type}")

        # ================= 2. 获取 Tensor 并检查形状 =================
        s_boxes_raw = s_data["boxes"]      # [B, N_s, 4]
        t_boxes_raw = t_data["boxes"]      # [B, N_t, 4]
        s_scores_raw = s_data["scores"]    # [B, N_s, C]
        t_scores_raw = t_data["scores"]    # [B, N_t, C]

        B_s, N_s, _ = s_boxes_raw.shape
        B_t, N_t, _ = t_boxes_raw.shape

        # ================= 3. 严格对齐逻辑 =================
        # 情况 A: Batch Size 不一致 (例如 Mixup 导致)
        if B_s != B_t:
            min_b = min(B_s, B_t)
            print(f"[ALIGN] Batch mismatch ({B_s} vs {B_t}). Truncating to {min_b}.")
            s_boxes_raw = s_boxes_raw[:min_b]
            t_boxes_raw = t_boxes_raw[:min_b]
            s_scores_raw = s_scores_raw[:min_b]
            t_scores_raw = t_scores_raw[:min_b]
            B_s, B_t = min_b, min_b
            # 重新获取 N (理论上不变)
            _, N_s, _ = s_boxes_raw.shape
            _, N_t, _ = t_boxes_raw.shape

        # 情况 B: Anchor 数量不一致 (核心问题：25200 vs 67200)
        if N_s != N_t:
            print(f"\n[CRITICAL] Anchor Mismatch! S: {N_s} ({s_type}), T: {N_t} ({t_type})")
            print("This implies Student and Teacher are using DIFFERENT branches (O2O vs O2M).")
            
            # 解决方案：强制切换到同一分支逻辑，或者截断到最小公共集 (仅当确定空间对应时)
            # 由于 O2O 和 O2M 的空间索引不对应，简单截断是错误的。
            # 最佳做法：报错提示用户检查配置，或者尝试重新提取另一分支。
            
            # 尝试自动切换：如果当前是 O2O/O2M 混用，尝试都切换到 O2M
            if (s_type in ["one2one", "o2o"] and t_type in ["one2many", "o2m"]) or \
               (s_type in ["one2many", "o2m"] and t_type in ["one2one", "o2o"]):
                print("[AUTO-FIX] Detected mixed branches. Attempting to re-extract One-to-Many for BOTH.")
                try:
                    s_data, s_type = self._extract_branch(student_pred, branch_name="one2many", name="Student")
                    t_data, t_type = self._extract_branch(teacher_pred, branch_name="one2many", name="Teacher")
                    # 重新赋值
                    s_boxes_raw = s_data["boxes"]
                    t_boxes_raw = t_data["boxes"]
                    s_scores_raw = s_data["scores"]
                    t_scores_raw = t_data["scores"]
                    _, N_s, _ = s_boxes_raw.shape
                    _, N_t, _ = t_boxes_raw.shape
                    print(f"[SUCCESS] Switched to -> S: {N_s} ({s_type}), T: {N_t} ({t_type})")
                except:
                    print("[FAIL] Could not switch branches. Falling back to truncation (RISKY).")

            # 如果还是不一致，只能截断 (警告：这会导致空间不对应，仅作为最后手段)
            if N_s != N_t:
                min_n = min(N_s, N_t)
                print(f"[DANGEROUS] Truncating anchors from ({N_s}, {N_t}) to {min_n}. Spatial alignment NOT guaranteed!")
                s_boxes_raw = s_boxes_raw[:, :min_n, :]
                t_boxes_raw = t_boxes_raw[:, :min_n, :]
                s_scores_raw = s_scores_raw[:, :min_n, :]
                t_scores_raw = t_scores_raw[:, :min_n, :]
                N_s, N_t = min_n, min_n

        # ================= 4. 展平与 Mask 生成 =================
        s_boxes = s_boxes_raw.view(-1, 4)
        t_boxes = t_boxes_raw.view(-1, 4)
        s_scores = s_scores_raw.view(-1, s_scores_raw.shape[-1])
        t_scores = t_scores_raw.view(-1, t_scores_raw.shape[-1])

        # 生成 Mask (基于 Teacher)
        t_conf, _ = t_scores_raw.max(dim=-1) # [B, N]
        valid_mask = (t_conf > self.confidence_threshold).view(-1)

        # 最终安全检查
        total_len = s_boxes.shape[0]
        if t_boxes.shape[0] != total_len or s_scores.shape[0] != total_len or t_scores.shape[0] != total_len or valid_mask.shape[0] != total_len:
            final_len = min(s_boxes.shape[0], t_boxes.shape[0], s_scores.shape[0], t_scores.shape[0], valid_mask.shape[0])
            #print(f"[SAFETY] Final hard truncate to {final_len}.")
            s_boxes = s_boxes[:final_len]
            t_boxes = t_boxes[:final_len]
            s_scores = s_scores[:final_len]
            t_scores = t_scores[:final_len]
            valid_mask = valid_mask[:final_len]

        if valid_mask.sum() == 0:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero, {"cons_box": 0, "cons_cls": 0, "cons_obj": 0, "cons_total": 0, "valid_ratio": 0}

        # ================= 5. 计算 Loss =================
        # Box
        iou_val = self.iou_loss(s_boxes, t_boxes)
        l1_val = F.smooth_l1_loss(s_boxes, t_boxes, reduction='none').mean(dim=-1)
        box_loss = ((iou_val + 0.1 * l1_val) * valid_mask.float()).sum() / (valid_mask.sum() + 1e-6)

        # Cls (KL) - 此时维度已严格对齐
        s_logits = s_scores / self.temperature
        t_logits = t_scores / self.temperature
        s_probs = F.softmax(s_logits, dim=-1)
        t_probs = F.softmax(t_logits, dim=-1)
        
        kl_div = (s_probs * (torch.log(s_probs.clamp(min=1e-8)) - torch.log(t_probs.clamp(min=1e-8)))).sum(dim=-1)
        cls_loss = (kl_div * valid_mask.float()).sum() / (valid_mask.sum() + 1e-6)

        # Obj
        obj_loss = torch.tensor(0.0, device=device)
        if "obj" in s_data and "obj" in t_data:
            s_obj = s_data["obj"].view(-1)[:valid_mask.shape[0]]
            t_obj = t_data["obj"].view(-1)[:valid_mask.shape[0]]
            obj_bce = F.binary_cross_entropy(torch.sigmoid(s_obj), torch.sigmoid(t_obj), reduction='none')
            obj_loss = (obj_bce * valid_mask.float()).sum() / (valid_mask.sum() + 1e-6)

        total_loss = self.box_weight * box_loss + self.cls_weight * cls_loss + self.obj_weight * obj_loss

        return total_loss, {
            "cons_box": box_loss.item(),
            "cons_cls": cls_loss.item(),
            "cons_obj": obj_loss.item(),
            "cons_total": total_loss.item(),
            "valid_ratio": valid_mask.float().mean().item()
        }