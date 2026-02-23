# YOLO-Lite 🚀

import torch
import torch.nn as nn

from . import LOGGER
from .checks import check_version
from .metrics import bbox_iou, probiou
from .ops import xywhr2xyxyxyxy

TORCH_1_10 = check_version(torch.__version__, "1.10.0")


class TaskAlignedAssigner(nn.Module):
    """
    一个用于目标检测的任务对齐分配器。

    该类根据任务对齐指标将真实对象（gt）分配给锚框，该指标结合了分类和定位信息。

    属性：
        topk (int): 考虑的候选对象数量。
        num_classes (int): 对象类别的数量。
        alpha (float): 任务对齐指标中分类组件的 alpha 参数。
        beta (float): 任务对齐指标中定位组件的 beta 参数。
        eps (float): 一个小值，用于防止除以零。
    """

    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9):
        """使用可自定义的超参数初始化 TaskAlignedAssigner 对象。"""
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        计算任务对齐分配。参考代码可在以下链接找到：
        https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py。

        参数：
            pd_scores (Tensor): 形状为 (bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor): 形状为 (bs, num_total_anchors, 4)
            anc_points (Tensor): 形状为 (num_total_anchors, 2)
            gt_labels (Tensor): 形状为 (bs, n_max_boxes, 1)
            gt_bboxes (Tensor): 形状为 (bs, n_max_boxes, 4)
            mask_gt (Tensor): 形状为 (bs, n_max_boxes, 1)

        返回：
            target_labels (Tensor): 形状为 (bs, num_total_anchors)
            target_bboxes (Tensor): 形状为 (bs, num_total_anchors, 4)
            target_scores (Tensor): 形状为 (bs, num_total_anchors, num_classes)
            fg_mask (Tensor): 形状为 (bs, num_total_anchors)
            target_gt_idx (Tensor): 形状为 (bs, num_total_anchors)
        """
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        device = gt_bboxes.device

        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.bg_idx),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]),
                torch.zeros_like(pd_scores[..., 0]),
            )

        try:
            return self._forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
        except torch.OutOfMemoryError:
            # 将张量移到 CPU，计算后再移回原始设备
            LOGGER.warning("WARNING: CUDA OutOfMemoryError in TaskAlignedAssigner, using CPU")
            cpu_tensors = [t.cpu() for t in (pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)]
            result = self._forward(*cpu_tensors)
            return tuple(t.to(device) for t in result)

    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        计算任务对齐分配。参考代码可在以下链接找到：
        https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py。

        参数：
            pd_scores (Tensor): 形状为 (bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor): 形状为 (bs, num_total_anchors, 4)
            anc_points (Tensor): 形状为 (num_total_anchors, 2)
            gt_labels (Tensor): 形状为 (bs, n_max_boxes, 1)
            gt_bboxes (Tensor): 形状为 (bs, n_max_boxes, 4)
            mask_gt (Tensor): 形状为 (bs, n_max_boxes, 1)

        返回：
            target_labels (Tensor): 形状为 (bs, num_total_anchors)
            target_bboxes (Tensor): 形状为 (bs, num_total_anchors, 4)
            target_scores (Tensor): 形状为 (bs, num_total_anchors, num_classes)
            fg_mask (Tensor): 形状为 (bs, num_total_anchors)
            target_gt_idx (Tensor): 形状为 (bs, num_total_anchors)
        """
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )

        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

        # 指派目标
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # 归一化
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """获取 in_gts 掩码，形状为 (b, max_num_obj, h*w)。"""
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes)
        # 获取 anchor_align 指标，形状为 (b, max_num_obj, h*w)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        # 获取 topk_metric 掩码，形状为 (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        # 合并所有掩码为最终掩码，形状为 (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """给定预测和真实边界框计算对齐指标。"""
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # 获取每个网格对于每个 gt 类别的分数
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w

        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """计算水平边界框的 IoU。"""
        return bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        """
        根据给定指标选择前 k 个候选对象。

        参数：
            metrics (Tensor): 形状为 (b, max_num_obj, h*w) 的张量，其中 b 是批处理大小，
                              max_num_obj 是最大对象数，h*w 表示锚点的总数。
            largest (bool): 如果为 True，则选择最大值；否则选择最小值。
            topk_mask (Tensor): 形状为 (b, max_num_obj, topk) 的可选布尔张量，其中
                                topk 是考虑的前 k 个候选对象的数量。如果未提供，
                                则根据给定指标自动计算前 k 个值。

        返回：
            (Tensor): 形状为 (b, max_num_obj, h*w) 的张量，包含选定的前 k 个候选对象。
        """
        # (b, max_num_obj, topk)
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        # (b, max_num_obj, topk)
        topk_idxs.masked_fill_(~topk_mask, 0)

        # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)
        for k in range(self.topk):
            # 为每个 k 扩展 topk_idxs，并在指定位置加 1
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k : k + 1], ones)
        # 筛选无效的边界框
        count_tensor.masked_fill_(count_tensor > 1, 0)

        return count_tensor.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """
        计算正锚点的目标标签、目标边界框和目标分数。

        参数：
            gt_labels (Tensor): 形状为 (b, max_num_obj, 1) 的真实标签，其中 b 是
                                批处理大小，max_num_obj 是最大对象数。
            gt_bboxes (Tensor): 形状为 (b, max_num_obj, 4) 的真实边界框。
            target_gt_idx (Tensor): 正锚点分配的真实对象索引，形状为 (b, h*w)，
                                    其中 h*w 是锚点的总数。
            fg_mask (Tensor): 形状为 (b, h*w) 的布尔张量，指示正
                              （前景）锚点。

        返回：
            (Tuple[Tensor, Tensor, Tensor]): 一个包含以下张量的元组：
                - target_labels (Tensor): 形状为 (b, h*w)，包含正锚点的目标标签。
                - target_bboxes (Tensor): 形状为 (b, h*w, 4)，包含正锚点的目标边界框。
                - target_scores (Tensor): 形状为 (b, h*w, num_classes)，包含正锚点的目标分数，
                                          其中 num_classes 是对象类别的数量。
        """
        # 指派目标标签，形状为 (b, 1)
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w)
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)

        # 指派目标框，形状为 (b, max_num_obj, 4) -> (b, h*w, 4)
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]

        # 指派目标分数
        target_labels.clamp_(0)

        # 比 F.one_hot() 快 10 倍
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=torch.int64,
            device=target_labels.device,
        )  # (b, h*w, 80)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)

        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_bboxes, target_scores

    @staticmethod
    def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
        """
        在真实边界框内选择正锚点中心。

        参数：
            xy_centers (torch.Tensor): 锚点中心坐标，形状为 (h*w, 2)。
            gt_bboxes (torch.Tensor): 真实边界框，形状为 (b, n_boxes, 4)。
            eps (float, optional): 用于数值稳定性的小值。默认为 1e-9。

        返回：
            (torch.Tensor): 正锚点的布尔掩码，形状为 (b, n_boxes, h*w)。

        注意：
            b: 批处理大小，n_boxes: 真实框数量，h: 高度，w: 宽度。
            边界框格式：[x_min, y_min, x_max, y_max]。
        """
        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # 左上角，右下角
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
        # return (bbox_deltas.min(3)[0] > eps).to(gt_bboxes.dtype)
        return bbox_deltas.amin(3).gt_(eps)

    @staticmethod
    def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
        """
        在分配给多个真实框时选择具有最高 IoU 的锚框。

        参数：
            mask_pos (torch.Tensor): 正掩码，形状为 (b, n_max_boxes, h*w)。
            overlaps (torch.Tensor): IoU 重叠，形状为 (b, n_max_boxes, h*w)。
            n_max_boxes (int): 最大真实框数量。

        返回：
            target_gt_idx (torch.Tensor): 分配的真实框索引，形状为 (b, h*w)。
            fg_mask (torch.Tensor): 前景掩码，形状为 (b, h*w)。
            mask_pos (torch.Tensor): 更新后的正掩码，形状为 (b, n_max_boxes, h*w)。

        注意：
            b: 批处理大小，h: 高度，w: 宽度。
        """
        # 转换 (b, n_max_boxes, h*w) -> (b, h*w)
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:  # 一个锚点分配给多个 gt_bboxes
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)  # (b, n_max_boxes, h*w)
            max_overlaps_idx = overlaps.argmax(1)  # (b, h*w)

            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)

            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()  # (b, n_max_boxes, h*w)
            fg_mask = mask_pos.sum(-2)
        # 找到每个网格对应的 gt（索引）
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
        return target_gt_idx, fg_mask, mask_pos


class RotatedTaskAlignedAssigner(TaskAlignedAssigner):
    """使用任务对齐指标将真实对象分配给旋转边界框。"""

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """计算旋转边界框的 IoU。"""
        return probiou(gt_bboxes, pd_bboxes).squeeze(-1).clamp_(0)

    @staticmethod
    def select_candidates_in_gts(xy_centers, gt_bboxes):
        """
        在真实框中选择旋转边界框的正锚点中心。

        参数：
            xy_centers (Tensor): 形状为 (h*w, 2)
            gt_bboxes (Tensor): 形状(b, n_boxes, 5)

        返回：
            (Tensor): 形状为 (b, n_boxes, h*w)
        """
        # (b, n_boxes, 5) --> (b, n_boxes, 4, 2)
        corners = xywhr2xyxyxyxy(gt_bboxes)
        # (b, n_boxes, 1, 2)
        a, b, _, d = corners.split(1, dim=-2)
        ab = b - a
        ad = d - a

        # (b, n_boxes, h*w, 2)
        ap = xy_centers - a
        norm_ab = (ab * ab).sum(dim=-1)
        norm_ad = (ad * ad).sum(dim=-1)
        ap_dot_ab = (ap * ab).sum(dim=-1)
        ap_dot_ad = (ap * ad).sum(dim=-1)
        return (ap_dot_ab >= 0) & (ap_dot_ab <= norm_ab) & (ap_dot_ad >= 0) & (ap_dot_ad <= norm_ad)  # is_in_box


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """从特征生成锚点。"""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # 偏移 x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # 偏移 y
        sy, sx = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_10 else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """将距离 (ltrb) 转换为边界框 (xywh 或 xyxy)。"""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh 边界框
    return torch.cat((x1y1, x2y2), dim)  # xyxy 边界框


def bbox2dist(anchor_points, bbox, reg_max=1):
    """将边界框 (xyxy) 转换为距离 (ltrb)。"""
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)  # 距离 (lt, rb)


def dist2rbox(pred_dist, pred_angle, anchor_points, dim=-1):
    """
    从锚点和分布解码预测的旋转边界框坐标。

    参数：
        pred_dist (torch.Tensor): 预测的旋转距离，形状为 (bs, h*w, 4)。
        pred_angle (torch.Tensor): 预测的角度，形状为 (bs, h*w, 1)。
        anchor_points (torch.Tensor): 锚点，形状为 (h*w, 2)。
        dim (int, optional): 用于拆分的维度。默认为 -1。

    返回：
        (torch.Tensor): 预测的旋转边界框，形状为 (bs, h*w, 4)。
    """
    lt, rb = pred_dist.split(2, dim=dim)
    cos, sin = torch.cos(pred_angle), torch.sin(pred_angle)
    # (bs, h*w, 1)
    xf, yf = ((rb - lt) / 2).split(1, dim=dim)
    x, y = xf * cos - yf * sin, xf * sin + yf * cos
    xy = torch.cat([x, y], dim=dim) + anchor_points
    return torch.cat([xy, lt + rb], dim=dim)