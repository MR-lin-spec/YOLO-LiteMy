import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import warnings
from typing import Dict, Tuple, Optional, Union
from sklearn.mixture import GaussianMixture as skm_GMM
from sklearn.exceptions import ConvergenceWarning

# 抑制 sklearn 的收敛警告，避免刷屏，我们在逻辑里处理失败情况
warnings.filterwarnings("ignore", category=ConvergenceWarning)

try:
    from torchvision.ops import nms as tv_nms
except ImportError:
    def tv_nms(boxes, scores, iou_threshold):
        if boxes.numel() == 0: return torch.empty((0,), dtype=torch.long, device=boxes.device)
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.sort(descending=True)[1]
        keep = []
        while order.numel() > 0:
            i = order[0]
            keep.append(i.item())
            if order.numel() == 1: break
            xx1 = torch.max(x1[i], x1[order[1:]])
            yy1 = torch.max(y1[i], y1[order[1:]])
            xx2 = torch.min(x2[i], x2[order[1:]])
            yy2 = torch.min(y2[i], y2[order[1:]])
            w = (xx2 - xx1).clamp(min=0)
            h = (yy2 - yy1).clamp(min=0)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = (ovr <= iou_threshold).nonzero(as_tuple=False).squeeze(1)
            if inds.dim() == 0: break
            order = order[inds + 1]
        return torch.tensor(keep, dtype=torch.long, device=boxes.device)

class DynamicSoftLabelAssigner(nn.Module):
    def __init__(self, iou_factor=3.0, num_classes=80, cost_threshold=1e9):
        super().__init__()
        self.iou_factor = iou_factor
        self.num_classes = num_classes
        self.cost_threshold = cost_threshold

    def forward(self, pred_scores, pred_bboxes, gt_labels, gt_bboxes, gt_mask=None):
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
                if valid_idx.sum() == 0: continue
                g_label = g_label[valid_idx]
                g_bbox = g_bbox[valid_idx]
            num_gt = g_bbox.shape[0]
            if num_gt == 0: continue

            tl = torch.max(p_bbox[:, None, :2], g_bbox[None, :, :2])
            br = torch.min(p_bbox[:, None, 2:], g_bbox[None, :, 2:])
            wh = (br - tl).clamp(min=0)
            inter = wh[:, :, 0] * wh[:, :, 1]
            area_p = (p_bbox[:, 2] - p_bbox[:, 0]) * (p_bbox[:, 3] - p_bbox[:, 1])
            area_g = (g_bbox[:, 2] - g_bbox[:, 0]) * (g_bbox[:, 3] - g_bbox[:, 1])
            union = area_p[:, None] + area_g[None, :] - inter
            ious = (inter / (union + 1e-9)).detach()

            gt_onehot = F.one_hot(g_label, self.num_classes).float()
            p_score_sigmoid = torch.sigmoid(p_score).clamp(min=1e-7, max=1.0-1e-7)
            preds_expanded = p_score_sigmoid.unsqueeze(1).expand(-1, num_gt, -1)
            targets_expanded = gt_onehot.unsqueeze(0).expand(n_boxes, -1, -1)
            cls_cost = F.binary_cross_entropy(preds_expanded, targets_expanded, reduction='none').sum(dim=-1)

            p_center = (p_bbox[:, :2] + p_bbox[:, 2:]) / 2.0
            g_center = (g_bbox[:, :2] + g_bbox[:, 2:]) / 2.0
            dist = ((p_center[:, None] - g_center[None, :]) ** 2).sum(dim=-1).sqrt()
            dis_cost = (dist / (dist.max() + 1e-9)) * 10.0

            cost_matrix = cls_cost - (ious * self.iou_factor) + dis_cost
            matching_matrix = torch.zeros_like(cost_matrix)
            current_cost = cost_matrix.clone()
            used_pred_mask = torch.zeros(n_boxes, dtype=torch.bool, device=device)
            used_gt_mask = torch.zeros(num_gt, dtype=torch.bool, device=device)
            
            for _ in range(min(num_gt, n_boxes)):
                if used_pred_mask.all() or used_gt_mask.all(): break
                masked_cost = current_cost.masked_fill(used_pred_mask[:, None], 1e9)
                masked_cost = masked_cost.masked_fill(used_gt_mask[None, :], 1e9)
                min_val, min_flat_idx = torch.min(masked_cost.view(-1), dim=0)
                if min_val > self.cost_threshold: break
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

def cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - 0.5*w, cy - 0.5*h, cx + 0.5*w, cy + 0.5*h], dim=-1)

def extract_predictions_strict(preds, branch_name='one2one'):
    data_dict = preds[branch_name] if isinstance(preds, dict) else preds[1][branch_name]
    boxes, scores = data_dict['boxes'], data_dict['scores']
    if boxes.dim() == 3 and boxes.shape[1] == 4: boxes = boxes.permute(0, 2, 1).contiguous()
    if scores.dim() == 3 and scores.shape[1] != scores.shape[2] and scores.shape[1] < 1000:
        if scores.shape[2] > scores.shape[1]: scores = scores.permute(0, 2, 1).contiguous()
    return boxes, scores

class IOUloss(nn.Module):
    def __init__(self, reduction="none", iou_type="ciou", xyxy=True):
        super().__init__()
        self.reduction = reduction
        self.iou_type = iou_type
        self.xyxy = xyxy
    def forward(self, pred, target):
        pred, target = pred.view(-1, 4).float(), target.view(-1, 4).float()
        tl = torch.max(pred[:, :2], target[:, :2])
        br = torch.min(pred[:, 2:], target[:, 2:])
        hw = (br - tl).clamp(min=0)
        area_i = torch.prod(hw, 1)
        area_p = torch.prod(pred[:, 2:] - pred[:, :2], 1)
        area_g = torch.prod(target[:, 2:] - target[:, 2:], 1)
        iou = area_i / (area_p + area_g - area_i + 1e-16)
        if self.iou_type == "ciou":
            c_tl, c_br = torch.min(pred[:, :2], target[:, :2]), torch.max(pred[:, 2:], target[:, 2:])
            convex_dis = torch.pow(c_br[:, 0]-c_tl[:, 0], 2) + torch.pow(c_br[:, 1]-c_tl[:, 1], 2) + 1e-7
            center_dis = torch.pow(pred[:, 0]-target[:, 0], 2) + torch.pow(pred[:, 1]-target[:, 1], 2)
            w_pred, h_pred = pred[:, 2] - pred[:, 0], pred[:, 3] - pred[:, 1]
            w_tgt, h_tgt = target[:, 2] - target[:, 0], target[:, 3] - target[:, 1]
            v = (4 / math.pi ** 2) * torch.pow(torch.atan(w_tgt / torch.clamp(h_tgt, min=1e-7)) - torch.atan(w_pred / torch.clamp(h_pred, min=1e-7)), 2)
            alpha = v / ((1 + 1e-7) - iou + v)
            ciou = iou - (center_dis / convex_dis + alpha * v)
            loss = 1 - ciou.clamp(min=-1.0, max=1.0)
        else:
            loss = 1 - iou
        return loss.mean() if self.reduction == "mean" else loss.sum() if self.reduction == "sum" else loss

class YOLO26ConsistencyLoss(nn.Module):
    def __init__(
        self, 
        box_weight=1.0, cls_weight=1.0, temperature=1.0,
        # 策略切换点
        gmm_start_epoch=40,           # 【关键】前 20 个 epoch 不用 GMM，用百分位，保证冷启动
        # GMM 参数 (优化后)
        gmm_min_samples=10,           # 提高样本门槛，太少不用 GMM
        gmm_max_iter=50,              # 增加迭代次数，防止不收敛
        gmm_tol=1e-4,                 # 放宽收敛容忍度
        base_percentile=0.25,         # 冷启动阶段使用的百分位 (Top 15%)
        base_threshold_floor=0.05,    # 绝对最低门槛
        num_classes=80, 
        pseudo_nms_iou_threshold=0.6,
        assigner_cost_threshold=100.0,
        debug_mode=False
    ):
        super().__init__()
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.temperature = temperature
        
        self.gmm_start_epoch = gmm_start_epoch
        self.gmm_min_samples = gmm_min_samples
        self.gmm_max_iter = gmm_max_iter
        self.gmm_tol = gmm_tol
        
        self.base_percentile = base_percentile
        self.base_threshold_floor = base_threshold_floor
        
        self.current_epoch = 0
        self.num_classes = num_classes
        self.pseudo_nms_iou_threshold = pseudo_nms_iou_threshold
        self.debug_mode = debug_mode
        
        self.assigner = DynamicSoftLabelAssigner(iou_factor=3.0, num_classes=num_classes, cost_threshold=assigner_cost_threshold)
        self.iou_loss = IOUloss(reduction="none", iou_type="ciou", xyxy=True)

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def _get_threshold_percentile(self, scores: torch.Tensor) -> float:
        """简单的百分位阈值"""
        if len(scores) == 0: return 1.0
        k = max(1, int(len(scores) * self.base_percentile))
        topk_vals = torch.topk(scores, k, sorted=False)[0]
        return float(topk_vals.min())

    def _gmm_filter_robust(self, scores: torch.Tensor, labels: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        鲁棒的 GMM 筛选：
        1. 检查样本量
        2. 尝试拟合，如果失败或分布不像双峰，自动回退到百分位
        """
        keep_mask = torch.zeros_like(scores, dtype=torch.bool, device=device)
        unique_classes = torch.unique(labels)
        
        # 当前是否启用 GMM?
        use_gmm = (self.current_epoch >= self.gmm_start_epoch)

        for c in unique_classes:
            cls_mask = (labels == c)
            cls_scores = scores[cls_mask]
            cls_indices = torch.where(cls_mask)[0]
            num_samples = len(cls_scores)
            
            # 默认策略：百分位 + Floor
            thr_percentile = self._get_threshold_percentile(cls_scores)
            final_thr = max(self.base_threshold_floor, thr_percentile)
            local_keep = cls_scores >= final_thr
            
            if not use_gmm or num_samples < self.gmm_min_samples:
                # 冷启动期 或 样本太少 -> 直接用百分位
                keep_mask[cls_indices[local_keep]] = True
                continue
            
            # --- 尝试 GMM ---
            scores_np = cls_scores.cpu().numpy().reshape(-1, 1)
            
            # 更稳健的初始化：使用 KMeans 结果或分位数初始化，而不是 min/max
            # 这里简化为使用 25% 和 75% 分位数作为初始均值，避免极端值影响
            q25 = float(np.percentile(scores_np, 25))
            q75 = float(np.percentile(scores_np, 75))
            means_init = [[q25], [q75]]
            
            try:
                gmm = skm_GMM(
                    n_components=2,
                    weights_init=[0.5, 0.5],
                    means_init=means_init,
                    precisions_init=[[[1.0]], [[1.0]]],
                    covariance_type='full',
                    max_iter=self.gmm_max_iter,
                    tol=self.gmm_tol,
                    random_state=42
                )
                gmm.fit(scores_np)
                
                assignments = gmm.predict(scores_np)
                means = gmm.means_.squeeze()
                covars = gmm.covariances_.squeeze()
                
                # 检查是否退化 (方差太小) 或 两个中心太近 (不是双峰)
                dist_means = abs(means[0] - means[1])
                if dist_means < 0.02: # 两个中心距离太近，说明是单峰
                    raise ValueError("Single mode detected")
                
                high_id = 1 if means[1] > means[0] else 0
                
                # 只有当高分簇有一定数量时才采纳
                if (assignments == high_id).sum() < 2:
                    raise ValueError("High cluster too small")
                
                # 取高分簇的最小值作为阈值
                high_scores = scores_np[assignments == high_id]
                gmm_thr = float(np.min(high_scores))
                
                # 【安全网】GMM 阈值不能比百分位阈值高太多，也不能低得离谱
                # 如果 GMM 算出的阈值比百分位还高很多，说明 GMM 可能把噪声当信号了，保守起见取两者较小值？
                # 不，通常 GMM 更准。但如果 GMM 失效，我们要有回退。
                # 这里策略：如果 GMM 成功，优先用 GMM，但必须 > floor
                final_thr = max(self.base_threshold_floor, gmm_thr)
                
                if self.debug_mode and c == 0:
                    print(f"[GMM OK] Class {c}: Means={means}, Thr={final_thr:.3f} (Percentile was {thr_percentile:.3f})")
                    
                local_keep = cls_scores >= final_thr
                keep_mask[cls_indices[local_keep]] = True
                
            except Exception as e:
                # GMM 失败 -> 无缝回退到百分位策略
                if self.debug_mode and c == 0:
                    print(f"[GMM Fail] Class {c}: {str(e)}, fallback to Percentile Thr={final_thr:.3f}")
                keep_mask[cls_indices[local_keep]] = True
                
        return keep_mask

    def forward(self, student_pred: Union[Dict, Tuple], teacher_pred: Union[Dict, Tuple]) -> Tuple[torch.Tensor, Dict]:
        """
        计算一致性损失，包含动态 GMM 伪标签筛选和详细统计日志。
        """
        # 1. 提取预测结果
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
        
        # --- 统计信息收集器 ---
        total_pseudo_selected = 0   # GMM/Percentile 筛选后的数量
        total_pseudo_base = 0       # 仅大于 base_threshold_floor 的数量
        batch_stats_list = []       # 用于记录每个 batch 的详细过滤情况

        for b in range(B):
            sb_box = s_boxes[b:b+1]
            sb_score = s_scores[b:b+1]
            tb_box = t_boxes[b:b+1]
            tb_score = t_scores[b:b+1]

            # 获取教师模型的置信度和类别
            t_conf, t_cls = tb_score.max(dim=-1)
            t_conf = t_conf.squeeze(0)
            t_cls = t_cls.squeeze(0)
            tb_box_squeeze = tb_box.squeeze(0)
            
            # --- 核心：动态筛选 (GMM 或 Percentile) ---
            gmm_keep_mask = self._gmm_filter_robust(t_conf, t_cls, device)
            
            selected_count = int(gmm_keep_mask.sum().item())
            base_count = int((t_conf > self.base_threshold_floor).sum().item())
            
            total_pseudo_selected += selected_count
            total_pseudo_base += base_count
            
            # 记录当前 Batch 的统计信息 (至少记录第一个 batch，或者开启 debug 时记录所有)
            if b == 0 or self.debug_mode:
                mode_str = "GMM" if self.current_epoch >= self.gmm_start_epoch else "Pct"
                ratio = selected_count / max(base_count, 1)
                fallback_flag = ""
                
                # 如果筛选结果为 0 但后续触发了 Fallback，稍后更新此标记
                batch_stats_list.append({
                    "batch_id": b,
                    "selected": selected_count,
                    "base": base_count,
                    "ratio": ratio,
                    "mode": mode_str,
                    "fallback": False
                })

            # --- Fallback 机制：防止无标签导致 Loss 为 0 ---
            if selected_count == 0:
                if len(t_conf) > 0:
                    # 找出分数最高的一个框，只要它大于极小值 (0.01)
                    max_val, max_idx = torch.max(t_conf, dim=0)
                    if max_val > 0.01:
                        # 强制选中这一个
                        gmm_keep_mask = torch.zeros_like(t_conf, dtype=torch.bool)
                        gmm_keep_mask[max_idx] = True
                        selected_count = 1
                        total_pseudo_selected += 1 # 更新总数
                        
                        # 更新统计记录，标记为 Fallback
                        if batch_stats_list and batch_stats_list[-1]["batch_id"] == b:
                            batch_stats_list[-1]["selected"] = 1
                            batch_stats_list[-1]["fallback"] = True
                    else:
                        # 连一个像样的框都没有，跳过
                        continue
                else:
                    continue

            # 获取筛选后的框、分数和标签
            candidate_boxes = tb_box_squeeze[gmm_keep_mask]
            candidate_scores = t_conf[gmm_keep_mask]
            candidate_labels = t_cls[gmm_keep_mask]

            # --- NMS 去重 ---
            if len(candidate_boxes) > 0:
                keep_indices = tv_nms(candidate_boxes, candidate_scores, self.pseudo_nms_iou_threshold)
                gt_bboxes = candidate_boxes[keep_indices]
                gt_labels = candidate_labels[keep_indices]
            else:
                continue

            if len(gt_bboxes) == 0: 
                continue

            # --- 标签分配 (Dynamic Soft Label Assigner) ---
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
            # 检查匹配有效性
            if matched_indices.numel() == 0 or matched_indices.min() < 0: 
                continue
                
            target_boxes = gt_bboxes[matched_indices]
            s_pos_boxes = sb_box.squeeze(0)[fg_mask_b]
            s_pos_scores = sb_score.squeeze(0)[fg_mask_b]
            t_soft_scores = assigned_scores[0][fg_mask_b]

            # --- 计算 Box Loss (CIoU) ---
            box_loss_val = self.iou_loss(s_pos_boxes, target_boxes)
            total_box_loss += box_loss_val.sum()
            
            # --- 计算 Cls Loss (KL Divergence) ---
            eps = 1e-9
            log_student = F.log_softmax(s_pos_scores / self.temperature, dim=-1)
            kl_div = (t_soft_scores.clamp(min=eps) * (torch.log(t_soft_scores.clamp(min=eps)) - log_student)).sum(dim=-1)
            total_cls_loss += kl_div.sum()
            
            valid_samples_count += fg_mask_b.sum()

        # --- 处理无有效样本的情况 ---
        if valid_samples_count == 0:
            zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return zero_loss, {
                "pseudo_labels": 0,
                "filter_details": "No valid samples after filtering & assigner.",
                "strategy": "GMM" if self.current_epoch >= self.gmm_start_epoch else "Percentile",
                "global_ratio": 0.0
            }

        # --- 归一化 Loss ---
        box_loss = total_box_loss / valid_samples_count
        cls_loss = total_cls_loss / valid_samples_count
        total_loss = self.box_weight * box_loss + self.cls_weight * cls_loss

        # --- 构建详细日志字符串 ---
        debug_str = ""
        if batch_stats_list:
            details = []
            for stat in batch_stats_list:
                mode_tag = stat['mode']
                fb_tag = "*FB" if stat['fallback'] else ""
                # 格式: B0:15/50(30%)[GMM]*FB
                details.append(f"B{stat['batch_id']}:{stat['selected']}/{stat['base']}({stat['ratio']:.1%})[{mode_tag}]{fb_tag}")
            debug_str = " | ".join(details)

        global_ratio = total_pseudo_selected / max(total_pseudo_base, 1)
        current_strategy = "GMM" if self.current_epoch >= self.gmm_start_epoch else "Percentile"

        return total_loss, {
            "cons_box": box_loss.item(),
            "cons_cls": cls_loss.item(),
            "cons_total": total_loss.item(),
            # 兼容旧代码的键
            "pseudo_labels": total_pseudo_selected // B if B > 0 else 0, 
            # 新增详细统计键
            "filter_details": debug_str,
            "total_selected": total_pseudo_selected,
            "total_base": total_pseudo_base,
            "global_ratio": global_ratio,
            "current_epoch": self.current_epoch,
            "strategy": current_strategy
        }