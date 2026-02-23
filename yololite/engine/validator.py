# YOLO-Lite 🚀

import os
import json
import time
from pathlib import Path
import numpy as np
import torch

from yololite.cfg import get_cfg, get_save_dir
from yololite.data.utils import check_det_dataset
from yololite.nn.autobackend import AutoBackend
from yololite.utils import LOGGER, TQDM, colorstr, emojis, ops
from yololite.utils.checks import check_imgsz, check_requirements
from yololite.utils.ops import Profile
from yololite.utils.torch_utils import de_parallel, select_device, smart_inference_mode
from yololite.data import build_dataloader, build_yolo_dataset
from yololite.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from yololite.utils.plotting import output_to_target, plot_images


class DetectionValidator:
    """
    属性：
        args (SimpleNamespace): 验证器的配置。
        dataloader (DataLoader): 用于验证的数据加载器。
        pbar (tqdm): 在验证过程中更新的进度条。
        model (nn.Module): 要验证的模型。
        data (dict): 数据字典。
        device (torch.device): 用于验证的设备。
        batch_i (int): 当前批次索引。
        training (bool): 模型是否处于训练模式。
        names (dict): 类名。
        seen: 记录到目前为止在验证过程中看到的图像数量。
        stats: 验证过程中的统计信息占位符。
        confusion_matrix: 混淆矩阵的占位符。
        nc: 类别数量。
        iouv: (torch.Tensor): 从 0.50 到 0.95 的 IoU 阈值，间隔为 0.05。
        jdict (dict): 用于存储 JSON 验证结果的字典。
        speed (dict): 包含键 'preprocess'、'inference'、'loss'、'postprocess' 及其各自的
                      批处理时间（毫秒）的字典。
        save_dir (Path): 保存结果的目录。
        plots (dict): 用于存储可视化的图表的字典。
        callbacks (dict): 用于存储各种回调函数的字典。
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None):
        """
        参数：
            dataloader (torch.utils.data.DataLoader): 用于验证的数据加载器。
            save_dir (Path, optional): 保存结果的目录。
            pbar (tqdm.tqdm): 显示进度的进度条。
            args (SimpleNamespace): 验证器的配置。
        """
        self.args = get_cfg(overrides=args)
        self.dataloader = dataloader
        self.pbar = pbar
        self.stride = None
        self.data = None
        self.device = None
        self.batch_i = None
        self.training = True
        self.names = None
        self.seen = None
        self.stats = None
        self.confusion_matrix = None
        self.nc = None
        self.iouv = None
        self.jdict = None
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        self.nt_per_class = None
        self.nt_per_image = None
        self.is_coco = False
        self.is_lvis = False
        self.class_map = None
        self.args.task = "detect"
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU 向量用于 mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.lb = []  # 用于自动标注
        if self.args.save_hybrid:
            LOGGER.warning(
                "WARNING ⚠️ 'save_hybrid=True' 将把真实值附加到预测中用于自动标注。\n"
                "WARNING ⚠️ 'save_hybrid=True' 将导致不正确的 mAP。\n"
            )
        self.save_dir = save_dir or get_save_dir(self.args)
        (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
        if self.args.conf is None:
            self.args.conf = 0.001  # 默认 conf=0.001
        self.args.imgsz = check_imgsz(self.args.imgsz, max_dim=1)
        self.plots = {}
        self.metrics = DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """执行验证过程，对数据加载器运行推理并计算性能指标。"""
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            # 在训练期间强制 FP16 验证
            self.args.half = self.device.type != "cpu" and trainer.amp
            model = trainer.ema.ema or trainer.model
            model = model.half() if self.args.half else model.float()
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
            model.eval()
        else:
            if str(self.args.model).endswith(".yaml"):
                LOGGER.warning("WARNING ⚠️ 验证未训练的模型 YAML 将导致 0 mAP。")

            model = AutoBackend(
                weights=model or self.args.model,
                device=select_device(self.args.device, self.args.batch),
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.half,
            )
            self.device = model.device  # 更新设备
            self.args.half = model.fp16  # 更新半精度
            stride, pt = model.stride, model.pt
            imgsz = check_imgsz(self.args.imgsz, stride=stride)

            if str(self.args.data).split(".")[-1] in {"yaml", "yml"}:
                self.data = check_det_dataset(self.args.data)
            else:
                raise FileNotFoundError(emojis(f"数据集 '{self.args.data}' 未找到，任务={self.args.task} ❌"))

            if self.device.type in {"cpu", "mps"}:
                self.args.workers = 0  # 更快的 CPU 验证，因为时间主要由推理主导，而不是数据加载
            if not pt:
                self.args.rect = False
            self.stride = model.stride  # 用于 get_dataloader() 的填充
            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)

            model.eval()
            model.warmup(imgsz=(1 if pt else self.args.batch, 3, imgsz, imgsz))

        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(de_parallel(model))
        self.jdict = []  # 每次验证前为空
        for batch_i, batch in enumerate(bar):
            self.batch_i = batch_i
            # 预处理
            with dt[0]:
                batch = self.preprocess(batch)

            # 推理
            with dt[1]:
                preds = model(batch["img"], augment=augment)

            # 损失
            with dt[2]:
                if self.training:
                    self.loss += model.loss(batch, preds)[1]

            # 后处理
            with dt[3]:
                preds = self.postprocess(preds)

            self.update_metrics(preds, batch)
            if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)

        stats = self.get_stats()
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
        self.finalize_metrics()
        self.print_results()
        if self.training:
            model.float()
            results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}
            return {k: round(float(v), 5) for k, v in results.items()}  # 返回结果保留 5 位小数
        else:
            LOGGER.info(
                "速度: {:.1f}ms 预处理, {:.1f}ms 推理, {:.1f}ms 损失, {:.1f}ms 后处理每张图像".format(
                    *tuple(self.speed.values())
                )
            )
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / "predictions.json"), "w") as f:
                    LOGGER.info(f"保存 {f.name}...")
                    json.dump(self.jdict, f)  # 扁平化并保存
                stats = self.eval_json(stats)  # 更新统计数据
            if self.args.plots or self.args.save_json:
                LOGGER.info(f"结果保存到 {colorstr('bold', self.save_dir)}")
            return stats

    def match_predictions(self, pred_classes, true_classes, iou, use_scipy=False):
        """
        使用 IoU 将预测与真实对象（pred_classes，true_classes）匹配。

        参数：
            pred_classes (torch.Tensor): 预测类索引，形状为 (N,)。
            true_classes (torch.Tensor): 目标类索引，形状为 (M,)。
            iou (torch.Tensor): NxM 张量，包含预测和真实值的成对 IoU 值。
            use_scipy (bool): 是否使用 scipy 进行匹配（更精确）。

        返回：
            (torch.Tensor): 形状为 (N,10) 的正确张量，表示 10 个 IoU 阈值。
        """
        # Dx10 矩阵，其中 D - 检测，10 - IoU 阈值
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        # LxD 矩阵，其中 L - 标签（行），D - 检测（列）
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class  # 将错误类别置为零
        iou = iou.cpu().numpy()
        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            if use_scipy:
                import scipy  # 作用域导入以避免所有命令导入

                cost_matrix = iou * (iou >= threshold)
                if cost_matrix.any():
                    labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix)
                    valid = cost_matrix[labels_idx, detections_idx] > 0
                    if valid.any():
                        correct[detections_idx[valid], i] = True
            else:
                matches = np.nonzero(iou >= threshold)  # IoU > 阈值且类别匹配
                matches = np.array(matches).T
                if matches.shape[0]:
                    if matches.shape[0] > 1:
                        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)

    def _prepare_batch(self, si, batch):
        """准备用于验证的图像和注释批次。"""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # 目标框
            ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad)  # 本地空间标签
        return {"cls": cls, "bbox": bbox, "ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad}

    def _prepare_pred(self, pred, pbatch):
        """准备用于验证的图像和注释批次。"""
        predn = pred.clone()
        ops.scale_boxes(
            pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
        )  # 本地空间预测
        return predn

    def build_dataset(self, img_path, mode="val", batch=None):
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)

    def get_dataloader(self, dataset_path, batch_size):
        """构建并返回数据加载器。"""
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False)  # 返回数据加载器

    def preprocess(self, batch):
        """对 YOLO 训练的图像批次进行预处理。"""
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        for k in ["batch_idx", "cls", "bboxes"]:
            batch[k] = batch[k].to(self.device)

        if self.args.save_hybrid:
            height, width = batch["img"].shape[2:]
            nb = len(batch["img"])
            bboxes = batch["bboxes"] * torch.tensor((width, height, width, height), device=self.device)
            self.lb = [
                torch.cat([batch["cls"][batch["batch_idx"] == i], bboxes[batch["batch_idx"] == i]], dim=-1)
                for i in range(nb)
            ]
        return batch

    def postprocess(self, preds):
        """对预测输出应用非最大抑制。"""
        return ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
        )

    def init_metrics(self, model):
        """为 YOLO 初始化评估指标。"""
        val = self.data.get(self.args.split, "")  # 验证路径
        self.is_coco = (
            isinstance(val, str)
            and "coco" in val
            and (val.endswith(f"{os.sep}val2017.txt") or val.endswith(f"{os.sep}test-dev2017.txt"))
        )  # 是 COCO
        self.is_lvis = isinstance(val, str) and "lvis" in val and not self.is_coco  # 是 LVIS
        self.class_map = [i for i in range(1, 91)] if self.is_coco else list(range(len(model.names)))
        self.args.save_json |= self.args.val and (self.is_coco or self.is_lvis) and not self.training  # 运行最终验证
        
        # ✅ 修改开始：确保 self.names 是字典格式
        if isinstance(model.names, list):
            self.names = {i: name for i, name in enumerate(model.names)}
        else:
            self.names = model.names
        # ✅ 修改结束
        
        self.nc = len(self.names)
        self.metrics.names = self.names
        self.metrics.plot = self.args.plots
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf)
        self.seen = 0
        self.jdict = []
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])
    def update_metrics(self, preds, batch):
        """更新指标。"""
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            nl = len(cls)
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # 预测
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = self._prepare_pred(pred, pbatch)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            # 评估
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
            if self.args.plots:
                self.confusion_matrix.process_batch(predn, bbox, cls)
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            # 保存
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            if self.args.save_txt:
                self.save_one_txt(
                    predn,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f'{Path(batch["im_file"][si]).stem}.txt',
                )

    def finalize_metrics(self, *args, **kwargs):
        """设置指标速度和混淆矩阵的最终值。"""
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix

    def get_stats(self):
        """返回指标统计和结果字典。"""
        stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}  # 转为 numpy
        self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=self.nc)
        self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=self.nc)
        stats.pop("target_img", None)
        if len(stats) and stats["tp"].any():
            self.metrics.process(**stats)
        return self.metrics.results_dict

    def print_results(self):
        """打印训练/验证集的每类指标。"""
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # 打印格式
        LOGGER.info(pf % ("all", self.seen, self.nt_per_class.sum(), *self.metrics.mean_results()))
        if self.nt_per_class.sum() == 0:
            LOGGER.warning(f"WARNING ⚠️ 验证集未找到标签，无法计算指标")

        # 打印每类结果
        if self.args.verbose and not self.training and self.nc > 1 and len(self.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                LOGGER.info(
                    pf % (self.names[c], self.nt_per_image[c], self.nt_per_class[c], *self.metrics.class_result(i))
                )

        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(
                    save_dir=self.save_dir, names=self.names.values(), normalize=normalize, on_plot=self.on_plot
                )

    def get_desc(self):
        """返回格式化字符串，概述 YOLO 模型的类指标。"""
        return ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)")

    @property
    def metric_keys(self):
        """返回 YOLO 训练/验证中使用的指标键。"""
        return []

    def on_plot(self, name, data=None):
        """注册图表（例如，在回调中使用）。"""
        self.plots[Path(name)] = {"data": data, "timestamp": time.time()}

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """
        返回正确的预测矩阵。

        参数：
            detections (torch.Tensor): 形状为 (N, 6) 的张量，表示检测，其中每个检测为
                (x1, y1, x2, y2, conf, class)。
            gt_bboxes (torch.Tensor): 形状为 (M, 4) 的张量，表示真实边界框坐标。每个
                边界框格式为: (x1, y1, x2, y2)。
            gt_cls (torch.Tensor): 形状为 (M,) 的张量，表示目标类索引。

        返回：
            (torch.Tensor): 形状为 (N, 10) 的正确张量，用于 10 个 IoU 级别。

        注意：
            该函数不返回任何直接可用于指标计算的值。相反，它提供了
            用于评估预测与真实值之间的中间表示。
        """
        iou = box_iou(gt_bboxes, detections[:, :4])
        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def plot_predictions(self, batch, preds, ni):
        """在输入图像上绘制预测边界框并保存结果。"""
        plot_images(
            batch["img"],
            *output_to_target(preds, max_det=self.args.max_det),
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # 预测

    def plot_val_samples(self, batch, ni):
        """绘制验证图像样本。"""
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def save_one_txt(self, predn, save_conf, shape, file):
        """以规范化坐标的特定格式将 YOLO 检测结果保存到 txt 文件中。"""
        from yololite.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=predn[:, :6],
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn, filename):
        """将 YOLO 预测序列化为 COCO json 格式。"""
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # 从中心到左上角
        for p, b in zip(predn.tolist(), box.tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(p[5])]
                                   + (1 if self.is_lvis else 0),  # 如果是 lvis，索引从 1 开始
                    "bbox": [round(x, 3) for x in b],
                    "score": round(p[4], 5),
                }
            )

    def eval_json(self, stats):
        """以 JSON 格式评估 YOLO 输出并返回性能统计信息。"""
        if self.args.save_json and (self.is_coco or self.is_lvis) and len(self.jdict):
            pred_json = self.save_dir / "predictions.json"  # 预测
            anno_json = (
                    self.data["path"]
                    / "annotations"
                    / ("instances_val2017.json" if self.is_coco else f"lvis_v1_{self.args.split}.json")
            )  # 注释
            pkg = "pycocotools" if self.is_coco else "lvis"
            LOGGER.info(f"\n评估 {pkg} mAP 使用 {pred_json} 和 {anno_json}...")
            try:
                for x in pred_json, anno_json:
                    assert x.is_file(), f"{x} 文件未找到"
                check_requirements("pycocotools>=2.0.6" if self.is_coco else "lvis>=0.5.3")
                if self.is_coco:
                    from pycocotools.coco import COCO  # noqa
                    from pycocotools.cocoeval import COCOeval  # noqa

                    anno = COCO(str(anno_json))  # 初始化注释 API
                    pred = anno.loadRes(str(pred_json))  # 初始化预测 API（必须传递字符串，而不是 Path）
                    val = COCOeval(anno, pred, "bbox")
                else:
                    from lvis import LVIS, LVISEval
                    anno = LVIS(str(anno_json))  # 初始化注释 API
                    pred = anno._load_json(str(pred_json))  # 初始化预测 API（必须传递字符串，而不是 Path）
                    val = LVISEval(anno, pred, "bbox")
                val.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # 要评估的图像
                val.evaluate()
                val.accumulate()
                val.summarize()
                if self.is_lvis:
                    val.print_results()  # 明确调用 print_results
                # 更新 mAP50-95 和 mAP50
                stats[self.metrics.keys[-1]], stats[self.metrics.keys[-2]] = (
                    val.stats[:2] if self.is_coco else [val.results["AP50"], val.results["AP"]]
                )
            except Exception as e:
                LOGGER.warning(f"{pkg} 无法运行: {e}")
        return stats