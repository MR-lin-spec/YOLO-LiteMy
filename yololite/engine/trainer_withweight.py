# YOLO-Lite 🚀

from functools import partial
import gc
import math
import random
import time
import warnings
from copy import copy, deepcopy
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from torch import nn, optim
from yololite.data import build_dataloader, build_yolo_dataset
from yololite.data.build import build_unlabeleddataloader, build_unlabelyolo_dataset
from yololite.engine.validator import DetectionValidator
from yololite.nn.tasks import DetectionModel
from yololite.utils.muon import MuSGD
from yololite.utils.plotting import plot_images, plot_labels, plot_results
from yololite.cfg import get_cfg, get_save_dir
from yololite.data.utils import check_det_dataset
from yololite.nn.tasks import attempt_load_one_weight, attempt_load_weights
from yololite.utils import (
    DEFAULT_CFG,
    LOGGER,
    TQDM,
    colorstr,
    yaml_save,
)
from yololite.utils.checks import check_imgsz, print_args
from yololite.utils.files import get_latest_run
from yololite.utils.torch_utils import (
    TORCH_2_4,
    de_parallel,
    EarlyStopping,
    ModelEMA,
    autocast,
    convert_optimizer_state_dict_to_fp16,
    init_seeds,
    one_cycle,
    select_device,
    strip_optimizer,
    unwrap_model
)
from yololite.utils.unsupervised_loss import YOLO26ConsistencyLoss,IOUloss
MIN_EPOCH_FOR_UNSUP_FORWARD = 10
class DetectionTrainer:
    """
    属性：
        args (SimpleNamespace): 训练器的配置参数。
        validator (BaseValidator): 验证器实例。
        model (nn.Module): 模型实例。
        save_dir (Path): 保存结果的目录。
        wdir (Path): 保存权重的目录。
        last (Path): 最近检查点的路径。
        best (Path): 最佳检查点的路径。
        save_period (int): 每 x 个 epoch 保存一次检查点（小于 1 时禁用）。
        batch_size (int): 训练的批量大小。
        epochs (int): 训练的总 epoch 数。
        start_epoch (int): 训练开始的 epoch 数。
        device (torch.device): 训练使用的设备。
        amp (bool): 是否启用自动混合精度（AMP）的标志。
        scaler (amp.GradScaler): 用于 AMP 的梯度缩放器。
        data (str): 数据的路径。
        trainset (torch.utils.data.Dataset): 训练数据集。
        testset (torch.utils.data.Dataset): 测试数据集。
        ema (nn.Module): 模型的 EMA（指数移动平均）。
        resume (bool): 是否从检查点恢复训练。
        lf (nn.Module): 损失函数。
        scheduler (torch.optim.lr_scheduler._LRScheduler): 学习率调度器。
        best_fitness (float): 达到的最佳适应度值。
        fitness (float): 当前适应度值。
        loss (float): 当前损失值。
        tloss (float): 总损失值。
        loss_names (list): 损失名称列表。
        csv (Path): 结果 CSV 文件的路径。
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None):
        """
        参数：
            cfg (str, optional): 配置文件的路径。默认为 DEFAULT_CFG。
            overrides (dict, optional): 配置覆盖项。默认为 None。
        """
        self.args = get_cfg(cfg, overrides)  # 加载配置
        self.check_resume(overrides)  # 检查是否需要恢复训练
        self.device = select_device(self.args.device, self.args.batch)  # 选择设备
        self.validator = None  # 验证器初始化
        self.metrics = None  # 评估指标初始化
        self.plots = {}  # 绘图字典初始化
        init_seeds(self.args.seed + 1, deterministic=self.args.deterministic)  # 初始化随机种子

        # 目录初始化
        self.save_dir = get_save_dir(self.args)  # 获取保存目录
        self.args.name = self.save_dir.name  # 更新日志用的名称
        self.wdir = self.save_dir / "weights"  # 权重目录
        self.wdir.mkdir(parents=True, exist_ok=True)  # 创建目录
        self.args.save_dir = str(self.save_dir)  # 保存目录字符串
        yaml_save(self.save_dir / "args.yaml", vars(self.args))  # 保存运行参数
        self.last, self.best = self.wdir / "last.pt", self.wdir / "best.pt"  # 检查点路径
        self.save_period = self.args.save_period  # 保存周期

        # 训练参数
        self.batch_size = self.args.batch  # 批量大小
        self.epochs = self.args.epochs or 100  # 训练的 epoch 数
        self.start_epoch = 0  # 开始的 epoch
        print_args(vars(self.args))  # 打印参数

        # 设备设置
        if self.device.type in {"cpu", "mps"}:
            self.args.workers = 0  # 使用 CPU 时设置工作线程为 0

        # 模型和数据集初始化
        self.model = self.args.model  # 模型初始化
        self.trainset, self.testset = self.get_dataset()  # 获取数据集
        self.ema = None  # EMA 初始化

        # 优化工具初始化
        self.lf = None  # 损失函数初始化
        self.scheduler = None  # 学习率调度器初始化

        # epoch 级别的指标
        self.best_fitness = None  # 最佳适应度
        self.fitness = None  # 当前适应度
        self.loss = None  # 当前损失
        self.tloss = None  # 总损失
        self.loss_names = ["Loss"]  # 损失名称
        self.csv = self.save_dir / "results.csv"  # 结果 CSV 文件路径
        self.plot_idx = [0, 1, 2]  # 绘图索引

    def _setup_scheduler(self):
        """初始化训练学习率调度器。"""
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # 余弦调度
        else:
            self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf  # 线性调度
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)  # 学习率调度器
#-----------------------------------------------------------------新增部分--------------------------------------#
    def get_unlabeldataset(self):
        """检查并获取无标签数据集。"""
        if not hasattr(self.args, "unlabeldata") or not self.args.unlabeldata:
            LOGGER.warning("未提供 unlabeldata 参数，跳过无标签数据集构建。")
            return None
        data = check_det_dataset(self.args.unlabeldata)
        if "yaml_file" in data:
            self.args.unlabeldata = data["yaml_file"]
        self.unlabel_data = data
        return data.get("train")  # 返回无标签训练集路径

    def get_unlabeldataloader(self, dataset_path, batch_size=16):
        """构建并返回无标签数据加载器。"""
        if dataset_path is None:
            return None
        dataset = self.build_unlabeleddataset(dataset_path, mode="unlabel", batch=batch_size)
        shuffle = True
        workers = self.args.workers
        return build_unlabeleddataloader(dataset, batch_size, workers, shuffle)

    
    def get_dynamic_weights(self, epoch):
        """
        计算动态权重，满足以下严格约束：
        1. 有监督权重 + 无监督权重 = 1.0
        2. 有监督权重 > 无监督权重 (即 unsup_weight 永远 < 0.5)
        3. 在指定 epoch (weight_transition_end_epoch) 收敛到固定最大值，之后保持不变
        
        参数:
            epoch (int): 当前 epoch 数
        
        返回:
            tuple: (supervised_weight, unsupervised_weight)
        """
        import math

        # --- 配置区域 ---
        # 无监督权重的最大上限 (必须 < 0.5 以满足条件2)
        # 设置为 0.45 意味着：有监督最小为 0.55，无监督最大为 0.45
        MAX_UNSUP_RATIO = 0.45 
        
        # 确保 epoch 在有效范围内
        current_epoch = max(0, min(epoch, self.epochs))
        
        # 定义过渡区间
        start_ep = self.weight_transition_start_epoch
        end_ep = self.weight_transition_end_epoch
        
        # 如果还没到开始时间，无监督权重为 0 (纯监督)
        if current_epoch < start_ep:
            return 1.0, 0.0
        
        # 如果已经结束过渡期，直接返回收敛后的固定值
        if current_epoch >= end_ep:
            unsup_weight = MAX_UNSUP_RATIO
            sup_weight = 1.0 - unsup_weight
            return sup_weight, unsup_weight
        
        # --- 计算过渡期内的动态权重 (余弦退火) ---
        # 计算当前进度 (0.0 到 1.0)
        if end_ep == start_ep:
            progress = 1.0
        else:
            progress = (current_epoch - start_ep) / (end_ep - start_ep)
            # 严格限制在 [0, 1] 之间，防止浮点数误差
            progress = max(0.0, min(1.0, progress))
        
        # 余弦曲线映射：0 -> 0, 1 -> 1, 中间平滑
        # 公式：(1 - cos(progress * pi)) / 2
        cosine_factor = (1 - math.cos(progress * math.pi)) / 2
        
        # 计算当前的无监督权重 (从 0 线性/曲线增长到 MAX_UNSUP_RATIO)
        unsup_weight = self.unsupervised_weight_start + \
                       (MAX_UNSUP_RATIO - self.unsupervised_weight_start) * cosine_factor
         # 【新增】如果超过过渡期，让无监督权重线性衰减到 0
        if current_epoch >= end_ep:
            decay_progress = (current_epoch - end_ep) / (self.epochs - end_ep)
            unsup_weight = MAX_UNSUP_RATIO * (1.0 - decay_progress) # 从 0.45 线性降到 0
        
        # 再次确保不超标 (防御性编程)
        unsup_weight = min(unsup_weight, MAX_UNSUP_RATIO)
        unsup_weight = max(unsup_weight, 0.0)
        
        # 计算有监督权重 (满足条件1: 和为1)
        sup_weight = 1.0 - unsup_weight
        
        # 双重检查满足条件2: sup > unsup
        # 理论上因为 MAX_UNSUP_RATIO < 0.5，这里永远成立，但加上断言以防配置错误
        assert sup_weight > unsup_weight, f"权重配置错误：sup({sup_weight:.4f}) 必须大于 unsup({unsup_weight:.4f})"
        
        return sup_weight, unsup_weight
    
    
    
    def _setup_train(self):
        """构建数据加载器和优化器。"""
        # 模型设置
        ckpt = self.setup_model()  # 设置模型
        self.model = self.model.to(self.device)  # 将模型移至设备
        self.set_model_attributes()  # 设置模型属性

        # 冻结层设置
        freeze_list = (
            self.args.freeze
            if isinstance(self.args.freeze, list)
            else range(self.args.freeze)
            if isinstance(self.args.freeze, int)
            else []
        )
        self.best_map50 = 0.0  # 初始化最佳 mAP50
        self.current_map50 = 0.0  # 初始化当前 mAP50
        always_freeze_names = [".dfl"]  # 始终冻结的层
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        for k, v in self.model.named_parameters():
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"冻结层 '{k}'")
                v.requires_grad = False  # 冻结参数
            elif not v.requires_grad and v.dtype.is_floating_point:  # 仅浮点类型可以要求梯度
                LOGGER.info(
                    f"警告 ⚠️ 设置 'requires_grad=True' 为冻结层 '{k}'。"
                )
                v.requires_grad = True

        # 检查 AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  # 是否启用 AMP
        if self.amp:  # 单 GPU
            self.model = self.model.to(self.device)  # 移动模型到指定设备
        self.amp = bool(self.amp)  # 转为布尔值
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=self.amp)
        )

        # 检查图像大小
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)  # 网格大小
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)  # 检查图像大小
        self.stride = gs  # 用于多尺度训练

        # 数据加载器
        batch_size = self.batch_size
        self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, mode="train")
        # 【关键新增】创建无标签数据加载器
        unlabel_path = self.get_unlabeldataset()
        if unlabel_path:
            self.unlabelloader = self.get_unlabeldataloader(unlabel_path, batch_size=batch_size)
            LOGGER.info(f"成功创建无标签数据加载器，路径: {unlabel_path}")
        else:
            self.unlabelloader = None
            LOGGER.info("未配置或未找到无标签数据，将以纯监督模式训练。")
        # 测试数据加载器
        self.test_loader = self.get_dataloader(self.testset, batch_size=batch_size * 2, mode="val")
        self.validator = self.get_validator()  # 获取验证器
        metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")  # 指标键
        self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))  # 初始化指标

        self.ema = ModelEMA(self.model)  # 初始化 EMA
        if self.args.plots:
            self.plot_training_labels()  # 绘制训练标签

        # 优化器设置
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # 优化前累积损失
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # 权重衰减
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs  # 迭代次数
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        #权重设置
        # 在 _setup_train() 方法中添加（在 consistent_loss 初始化之后）
        self.supervised_weight_start = 1.0  # 初始监督损失权重
        self.supervised_weight_end = 0.0    # 最终监督损失权重
        self.unsupervised_weight_start = 0.0  # 初始无监督损失权重
        self.unsupervised_weight_end = 1.0    # 最终无监督损失权重
        # 权重过渡策略参数
        self.weight_transition_start_epoch = max(10, int(self.epochs * 0.2)) 
        self.weight_transition_end_epoch = self.epochs-5 # 结束过渡的epoch
        num_classes = self.data.get("nc", 80) 
        #初始化一致性损失
        self.consistent_loss = YOLO26ConsistencyLoss(
        box_weight=1.0,
        cls_weight=1.0,
        obj_weight=0.5,
        temperature=1.0,
        confidence_threshold=0.25,
        iou_type="ciou",
        )
        '''
        self.consistent_loss = YOLO26ConsistencyLoss(
            box_weight=1.0,
            cls_weight=1.0,
            temperature=1.0,
            #confidence_threshold_start=0.5,    # 前20轮用0.5，减少噪声
            #confidence_threshold_end=0.25,      # 后期降到0.25，增加召回
            #threshold_warmup_epochs=50,         # 50轮完成过渡
          #  total_epochs=self.epochs,           # 175
            num_classes=num_classes,
            
        )
        '''
        
        # 设置学习率调度器
        self._setup_scheduler()
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False  # 提前停止
        self.resume_training(ckpt)  # 恢复训练
        self.scheduler.last_epoch = self.start_epoch - 1  # 不移动
    
    def _setup_train_old(self):
        """构建数据加载器和优化器。"""
        # 模型设置
        ckpt = self.setup_model()  # 设置模型
        self.model = self.model.to(self.device)  # 将模型移至设备
        self.set_model_attributes()  # 设置模型属性

        # 冻结层设置
        freeze_list = (
            self.args.freeze
            if isinstance(self.args.freeze, list)
            else range(self.args.freeze)
            if isinstance(self.args.freeze, int)
            else []
        )
        self.best_map50 = 0.0  # 初始化最佳 mAP50
        self.current_map50 = 0.0  # 初始化当前 mAP50
        always_freeze_names = [".dfl"]  # 始终冻结的层
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        for k, v in self.model.named_parameters():
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"冻结层 '{k}'")
                v.requires_grad = False  # 冻结参数
            elif not v.requires_grad and v.dtype.is_floating_point:  # 仅浮点类型可以要求梯度
                LOGGER.info(
                    f"警告 ⚠️ 设置 'requires_grad=True' 为冻结层 '{k}'。"
                )
                v.requires_grad = True

        # 检查 AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  # 是否启用 AMP
        if self.amp:  # 单 GPU
            self.model = self.model.to(self.device)  # 移动模型到指定设备
        self.amp = bool(self.amp)  # 转为布尔值
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=self.amp)
        )

        # 检查图像大小
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)  # 网格大小
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)  # 检查图像大小
        self.stride = gs  # 用于多尺度训练

        # 数据加载器
        batch_size = self.batch_size
        self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, mode="train")
        # 【关键新增】创建无标签数据加载器
        unlabel_path = self.get_unlabeldataset()
        if unlabel_path:
            self.unlabelloader = self.get_unlabeldataloader(unlabel_path, batch_size=batch_size)
            LOGGER.info(f"成功创建无标签数据加载器，路径: {unlabel_path}")
        else:
            self.unlabelloader = None
            LOGGER.info("未配置或未找到无标签数据，将以纯监督模式训练。")
        # 测试数据加载器
        self.test_loader = self.get_dataloader(self.testset, batch_size=batch_size * 2, mode="val")
        self.validator = self.get_validator()  # 获取验证器
        metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")  # 指标键
        self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))  # 初始化指标

        self.ema = ModelEMA(self.model)  # 初始化 EMA
        if self.args.plots:
            self.plot_training_labels()  # 绘制训练标签

        # 优化器设置
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # 优化前累积损失
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # 权重衰减
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs  # 迭代次数
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        #权重设置
        # 在 _setup_train() 方法中添加（在 consistent_loss 初始化之后）
        self.supervised_weight_start = 1.0  # 初始监督损失权重
        self.supervised_weight_end = 0.0    # 最终监督损失权重
        self.unsupervised_weight_start = 0.0  # 初始无监督损失权重
        self.unsupervised_weight_end = 1.0    # 最终无监督损失权重
        # 权重过渡策略参数
        self.weight_transition_start_epoch = max(10, int(self.epochs * 0.2)) 
        self.weight_transition_end_epoch = self.epochs-5 # 结束过渡的epoch
        num_classes = self.data.get("nc", 80) 
        self.consistent_loss = YOLO26ConsistencyLoss(
            box_weight=1.0,
            cls_weight=1.0,
            temperature=1.0,
            confidence_threshold_start=0.5,    # 前20轮用0.5，减少噪声
            confidence_threshold_end=0.25,      # 后期降到0.25，增加召回
            threshold_warmup_epochs=50,         # 50轮完成过渡
            total_epochs=self.epochs,           # 175
            num_classes=num_classes,
            topk=13
        )
        # 设置学习率调度器
        self._setup_scheduler()
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False  # 提前停止
        self.resume_training(ckpt)  # 恢复训练
        self.scheduler.last_epoch = self.start_epoch - 1  # 不移动
    
    def train_old(self):
        self._setup_train()  # 设置训练
        nb = len(self.train_loader)  # 批次数
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # 热身迭代
        last_opt_step = -1  # 最后优化步数
        self.epoch_time = 0  # epoch 时间
        self.epoch_time_start = time.time()  # epoch 开始时间
        self.train_time_start = time.time()  # 训练开始时间
        LOGGER.info(
            f'图像大小 {self.args.imgsz} 训练, {self.args.imgsz} 验证\n'
            f"将结果记录到 {colorstr('bold', self.save_dir)}\n"
            f'开始训练 ' + (f"{self.args.time} 小时..." if self.args.time else f"{self.epochs} 个周期...")
        )
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])  # 更新绘图索引
        epoch = self.start_epoch
        self.optimizer.zero_grad()  # 清零梯度
        while True:
            self.epoch = epoch
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # 忽略警告
                self.scheduler.step()  # 更新学习率

            self.model.train()  # 设置模型为训练模式
            pbar = enumerate(self.train_loader)  # 训练进度条
            # 更新数据加载器属性（可选）
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()  # 关闭马赛克增强
                self.train_loader.reset()  # 重置训练加载器

            LOGGER.info(self.progress_string())  # 打印进度信息
            pbar = TQDM(enumerate(self.train_loader), total=nb)  # 进度条显示
            unlabel_iter = iter(self.unlabelloader) if self.unlabelloader else None
            self.tloss = None  # 总损失初始化
            for i, batch in pbar:
                # 热身
                ni = i + nb * epoch  # 当前迭代次数
                if ni <= nw:
                    xi = [0, nw]  # x 插值
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))  # 更新累积参数
                    for j, x in enumerate(self.optimizer.param_groups):
                        # 学习率调整
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])
                # 获取无标签批次
                unlabelbatch = None
                if unlabel_iter is not None:
                    try:
                        unlabelbatch = next(unlabel_iter)
                    except StopIteration:
                        unlabel_iter = iter(self.unlabelloader)  # 重置迭代器
                        unlabelbatch = next(unlabel_iter)
                    unlabelbatch = self.preprocess_batch(unlabelbatch)  # 归一化等操作

                # 前向传播
                with autocast(self.amp):  # 自动混合精度
                    sup_weight, unsup_weight = self.get_dynamic_weights(epoch) #计算权重
                    #消融实验一：采用固定权重设置，将上行代码注释，下面代码取消注释
                    #sup_weight, unsup_weight = 0.5, 0.5

                    #消融实验二：采用线性过渡策略，将上行代码注释，下面代码取消注释
                    #unsup_weight = min(1.0, epoch / self.epochs)  # 线性过渡进度
                    #sup_weight = 1.0 - unsup_weight  # 监督损失权重从1降到0

                    #学生模型和教师模型预测
                    if unlabelbatch is not None:  # 混合训练
                        teacher_preds= self.ema.ema.predict(unlabelbatch['img']) # 预测
                    batch = self.preprocess_batch(batch)  # 预处理批次
                    #计算无监督损失
                    student_preds=self.model.forward(batch["img"]) # 模型前向传播得到学生模型的预测结果
                    #print("学生模型预测形状：", type(student_preds))
                    #print("教师模型预测形状：", type(teacher_preds))
                    unsupervise_loss, unsupervise_loss_dict = self.consistent_loss(student_preds, teacher_preds)
                    unsupervise_loss=unsupervise_loss * 0.1 # 无监督损失缩放
                    

                    self.loss, self.loss_items = self.model(batch)  # 计算损失
                    #print("模型输出：", self.model(batch))

                    #记录损失函数梯度,用于消融实验
                    #staticname="静态权重"
                    #dynamicname="动态权重"
                    #linearname="线性过渡权重"
                    #LOGGER.info(f"无监督损失梯度: {unsupervise_loss.grad}")
                    #LOGGER.info(f"监督损失梯度: {self.loss.grad}")
                    #LOGGER.info(f"XX权重消融实验，其中XX填入上面你的消融实验名称：{}")
                   
                    self.tloss = (
                        (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                    )
                    
                    #print("模型测试损失：", test_result_loss)
                    #update_ema=self.ema.update(self.model)# 由于ema的update方法是没有返回值的，所以是None
                # 反向传播
                if not self.loss.dim() == 0:  # 检查是否为标量
                    self.loss = self.loss.sum()  # 或 .mean()
               
                total_unsup_loss = 0.0
                if isinstance(unsupervise_loss, torch.Tensor):
                    if unsupervise_loss.numel() > 0:
                        total_unsup_loss = unsupervise_loss.sum() # 确保是标量
                    else:
                        total_unsup_loss = torch.tensor(0.0, device=self.device)
                else:
                    total_unsup_loss = torch.tensor(unsupervise_loss, device=self.device)

                self.scaler.scale(self.loss*sup_weight + total_unsup_loss*unsup_weight).backward()  # 加上无监督损失
                #self.ema.update(self.model)
                # 优化
                if ni - last_opt_step >= self.accumulate:  # 检查是否达到优化条件
                    self.optimizer_step()  # 执行优化步
                    last_opt_step = ni  # 更新最后优化步数

                    # 时间停止检查
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if self.stop:  # 超过训练时间
                            break

                # 日志记录
                loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
            # 在pbar.set_description之前添加动态权重信息
                pbar.set_description(
                    ("%11s" * 2 + "%11.4g" * (2 + loss_length) + " sup_w:%.3f unsup_w:%.3f unsup_loss:%.4g")
                    % (
                        f"{epoch + 1}/{self.epochs}",
                        f"{self._get_memory():.3g}G",
                        *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),
                        batch["cls"].shape[0],
                        batch["img"].shape[-1],
                        sup_weight,
                        unsup_weight,
                        unsupervise_loss.item() if isinstance(unsupervise_loss, torch.Tensor) else unsupervise_loss,
                    )
                )
                if self.args.plots and ni in self.plot_idx:
                    self.plot_training_samples(batch, ni)  # 绘制训练样本

            # 学习率记录
            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # 记录学习率
            final_epoch = epoch + 1 >= self.epochs  # 检查是否为最后一个 epoch
            self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])  # 更新 EMA

            # 验证
            if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                self.metrics, self.fitness = self.validate()  # 验证模型
                # 提取 mAP50(B)
            # 注意：根据你的日志头 "metrics/mAP50(B)"，key 可能是 'metrics/mAP50' 或直接是 'mAP50'
            # 请根据你实际 metrics 字典的 key 进行调整
            current_mAP50 = self.metrics.get('metrics/mAP50(B)', self.metrics.get('mAP50', 0.0))
            
            # 更新实例变量，供 get_dynamic_weights 使用
            self.current_map50 = current_mAP50
            
            # 初始化 best_map50 (如果是第一个 epoch)
            if not hasattr(self, 'best_map50'):
                self.best_map50 = 0.0
                self.unsup_freeze_triggered = False
                
            # 打印调试信息
            if hasattr(self, 'unsup_freeze_triggered') and self.unsup_freeze_triggered:
                LOGGER.info(f"🔒 无监督权重已冻结，当前 mAP: {current_mAP50:.4f}")
            self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})  # 保存指标
            self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch  # 检查是否停止训练
            if self.args.time:
                self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)  # 时间停止检查

            # 保存模型
            if self.args.save or final_epoch:
                self.save_model()  # 保存模型

            # 学习率调度
            t = time.time()
            self.epoch_time = t - self.epoch_time  # 计算 epoch 时间
            self.epoch_time_start = t  # 更新开始时间
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)  # 平均 epoch 时间
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)  # 更新总 epoch
                self._setup_scheduler()  # 设置学习率调度器
                self.scheduler.last_epoch = self.epoch  # 不移动
                self.stop |= epoch >= self.epochs  # 检查是否超过总 epoch
            self._clear_memory()  # 清理内存

            # 提前停止检查
            if self.stop:
                break  # 结束训练
            epoch += 1  # 更新 epoch 计数

        # 最终验证
        seconds = time.time() - self.train_time_start
        LOGGER.info(f"\n{epoch - self.start_epoch + 1} 个周期完成，耗时 {seconds / 3600:.3f} 小时。")
        self.final_eval()  # 最终评估
        if self.args.plots:
            self.plot_metrics()  # 绘制指标
        self._clear_memory()  # 清理内存


    def train_old2(self):
        """优化后的半监督训练方法，确保无标签数据充分利用，并记录详细日志"""
        self._setup_train()
        
        # 【关键修改1】计算两个数据集的实际长度
        labeled_len = len(self.train_loader.dataset)
        unlabeled_len = len(self.unlabelloader.dataset) if self.unlabelloader else 0
        
        # 判断哪个数据集更大，以大数据集为epoch基准
        if self.unlabelloader is not None and unlabeled_len > labeled_len:
            primary_loader = self.unlabelloader
            secondary_loader = self.train_loader
            primary_is_labeled = False
            nb = len(primary_loader)
            self.training_mode = "unlabeled_dominant"
            LOGGER.info(
                f"{colorstr('yellow', '【训练模式】')} 无标签主导模式 | "
                f"有标签: {labeled_len} | 无标签: {unlabeled_len} | "
                f"比例: 1:{unlabeled_len/max(labeled_len,1):.1f} | "
                f"每epoch迭代: {nb}次"
            )
        else:
            primary_loader = self.train_loader
            secondary_loader = self.unlabelloader
            primary_is_labeled = True
            nb = len(primary_loader)
            self.training_mode = "standard"
            LOGGER.info(
                f"{colorstr('green', '【训练模式】')} 标准模式 | "
                f"有标签: {labeled_len} | 无标签: {unlabeled_len or 0} | "
                f"每epoch迭代: {nb}次"
            )
        
        # 热身迭代数计算
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1
        
        last_opt_step = -1
        self.epoch_time = 0
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        
        # 记录训练配置摘要
        LOGGER.info(
            f"\n{colorstr('bold', '========== 训练配置摘要 ==========')}\n"
            f"图像大小: {self.args.imgsz} (训练/验证)\n"
            f"批次大小: {self.batch_size}\n"
            f"总轮数: {self.epochs}\n"
            f"学习率: 初始={self.args.lr0}, 最终={self.args.lrf}\n"
            f"优化器: {type(self.optimizer).__name__}\n"
            f"保存目录: {self.save_dir}\n"
            f"{'='*40}\n"
        )
        
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
            LOGGER.info(f"Mosaic增强将在第 {self.epochs - self.args.close_mosaic} 轮后关闭")
        
        epoch = self.start_epoch
        self.optimizer.zero_grad()
        
        # 记录最佳指标
        best_metrics = {
            'fitness': 0.0,
            'mAP50': 0.0,
            'mAP50-95': 0.0,
            'epoch': 0
        }
        
        while True:
            self.epoch = epoch
            
            # 【关键修改2】每个epoch开始：更新动态阈值并记录
            current_thr = 0.25
            if hasattr(self, 'consistent_loss') and hasattr(self.consistent_loss, 'set_epoch'):
                current_thr = self.consistent_loss.set_epoch(epoch)
            
            # 获取当前权重
            sup_weight, unsup_weight = self.get_dynamic_weights(epoch)
            
            # Epoch开始日志
            LOGGER.info(
                f"\n{colorstr('blue', '╔' + '═'*60 + '╗')}\n"
                f"{colorstr('blue', '║')} {colorstr('bold', f'Epoch {epoch + 1}/{self.epochs}')} 开始\n"
                f"{colorstr('blue', '╠' + '═'*60 + '╣')}\n"
                f"{colorstr('blue', '║')} 学习率策略: {self.scheduler.__class__.__name__}\n"
                f"{colorstr('blue', '║')} 当前LR: {self.optimizer.param_groups[0]['lr']:.6f}\n"
                f"{colorstr('blue', '║')} 监督权重: {sup_weight:.3f} | 无监督权重: {unsup_weight:.3f}\n"
                f"{colorstr('blue', '║')} 伪标签阈值: {current_thr:.3f}\n"
                f"{colorstr('blue', '║')} 数据模式: {self.training_mode}\n"
                f"{colorstr('blue', '╚' + '═'*60 + '╝')}"
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.scheduler.step()

            self.model.train()
            
            # 创建循环迭代器用于secondary数据
            secondary_iter = iter(secondary_loader) if secondary_loader else None
            
            # 记录迭代统计
            epoch_stats = {
                'total_loss': 0.0,
                'sup_loss': 0.0,
                'unsup_loss': 0.0,
                'unsup_box': 0.0,
                'unsup_cls': 0.0,
                'pseudo_labels': 0,
                'valid_ratio': 0.0,
                'num_batches': 0,
                'num_unsup_batches': 0
            }
            
            LOGGER.info(self.progress_string())
            pbar = TQDM(enumerate(primary_loader), total=nb)
            
            self.tloss = None
            
            for i, primary_batch in pbar:
                ni = i + nb * epoch
                
                # 获取secondary数据
                secondary_batch = None
                if secondary_iter is not None:
                    try:
                        secondary_batch = next(secondary_iter)
                    except StopIteration:
                        secondary_iter = iter(secondary_loader)
                        secondary_batch = next(secondary_iter)
                
                # 根据主数据集类型分配batch
                if primary_is_labeled:
                    batch = primary_batch
                    unlabelbatch = secondary_batch
                else:
                    batch = secondary_batch
                    unlabelbatch = primary_batch
                
                # 热身学习率调整
                if ni <= nw:
                    xi = [0, nw]
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])
                
                # 预处理
                has_unlabeled = unlabelbatch is not None
                if has_unlabeled:
                    unlabelbatch = self.preprocess_batch(unlabelbatch)
                    epoch_stats['num_unsup_batches'] += 1
                
                if batch is None:
                    LOGGER.warning(f"批次 {i} 为None，跳过")
                    continue
                    
                batch = self.preprocess_batch(batch)
                
                # 前向传播
                with autocast(self.amp):
                    # 重新获取权重（可能已更新）
                    sup_weight, unsup_weight = self.get_dynamic_weights(epoch)
                    
                    # 教师模型预测
                    teacher_preds = None
                    if has_unlabeled:
                        with torch.no_grad():
                            teacher_preds = self.ema.ema.predict(unlabelbatch['img'])
                    
                    # 学生模型预测
                    student_preds = self.model.forward(batch["img"])
                    
                    # 计算无监督损失
                    unsupervise_loss = torch.tensor(0.0, device=self.device)
                    unsupervise_loss_dict = {
                        "cons_box": 0.0, 
                        "cons_cls": 0.0, 
                        "cons_total": 0.0, 
                        "valid_ratio": 0.0,
                        "pseudo_labels": 0,
                        "threshold": current_thr
                    }
                    
                    if has_unlabeled and teacher_preds is not None:
                        student_unsup_preds = self.model.forward(unlabelbatch['img'])
                        unsupervise_loss, unsupervise_loss_dict = self.consistent_loss(
                            student_unsup_preds, 
                            teacher_preds
                        )
                        unsupervise_loss = unsupervise_loss * 0.1
                    
                    # 监督损失
                    self.loss, self.loss_items = self.model(batch)
                    
                    # 更新显示损失
                    self.tloss = (
                        (self.tloss * i + self.loss_items) / (i + 1) 
                        if self.tloss is not None else self.loss_items
                    )
                    
                    # 确保标量
                    if not self.loss.dim() == 0:
                        self.loss = self.loss.sum()
                    
                    total_unsup_loss = torch.tensor(0.0, device=self.device)
                    if isinstance(unsupervise_loss, torch.Tensor):
                        if unsupervise_loss.numel() > 0:
                            total_unsup_loss = unsupervise_loss.sum()
                        else:
                            total_unsup_loss = torch.tensor(0.0, device=self.device)
                    else:
                        total_unsup_loss = torch.tensor(unsupervise_loss, device=self.device)
                    
                    # 合并损失
                    total_loss = self.loss * sup_weight + total_unsup_loss * unsup_weight
                
                # 反向传播
                self.scaler.scale(total_loss).backward()
                
                # 统计
                epoch_stats['num_batches'] += 1
                epoch_stats['total_loss'] += total_loss.item()
                epoch_stats['sup_loss'] += self.loss.item()
                epoch_stats['unsup_loss'] += total_unsup_loss.item()
                epoch_stats['unsup_box'] += unsupervise_loss_dict.get("cons_box", 0.0)
                epoch_stats['unsup_cls'] += unsupervise_loss_dict.get("cons_cls", 0.0)
                epoch_stats['pseudo_labels'] += unsupervise_loss_dict.get("pseudo_labels", 0)
                epoch_stats['valid_ratio'] += unsupervise_loss_dict.get("valid_ratio", 0.0)
                
                # 优化步骤
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni
                    
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if self.stop:
                            break
                
                # 进度条显示
                loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                desc_format = (
                    "%11s" * 2 + 
                    "%11.4g" * (2 + loss_length) + 
                    " | sw:%.2f uw:%.2f thr:%.2f pse:%d unsup:%.4g"
                )
                desc_values = (
                    f"{epoch + 1}/{self.epochs}",
                    f"{self._get_memory():.3g}G",
                    *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),
                    batch["cls"].shape[0],
                    batch["img"].shape[-1],
                    sup_weight,
                    unsup_weight,
                    unsupervise_loss_dict.get("threshold", 0.25),
                    unsupervise_loss_dict.get("pseudo_labels", 0),
                    total_unsup_loss.item(),
                )
                pbar.set_description(desc_format % desc_values)
                
                # 绘制样本
                if self.args.plots and ni in self.plot_idx:
                    self.plot_training_samples(batch, ni)
            
            # 计算epoch平均统计
            num_batches = max(epoch_stats['num_batches'], 1)
            avg_total_loss = epoch_stats['total_loss'] / num_batches
            avg_sup_loss = epoch_stats['sup_loss'] / num_batches
            avg_unsup_loss = epoch_stats['unsup_loss'] / max(epoch_stats['num_unsup_batches'], 1)
            avg_unsup_box = epoch_stats['unsup_box'] / max(epoch_stats['num_unsup_batches'], 1)
            avg_unsup_cls = epoch_stats['unsup_cls'] / max(epoch_stats['num_unsup_batches'], 1)
            avg_pseudo_labels = epoch_stats['pseudo_labels'] / max(epoch_stats['num_unsup_batches'], 1)
            avg_valid_ratio = epoch_stats['valid_ratio'] / max(epoch_stats['num_unsup_batches'], 1)
            
            # Epoch结束处理
            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}
            final_epoch = epoch + 1 >= self.epochs
            self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])
            
            # 验证
            val_start_time = time.time()
            if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                self.metrics, self.fitness = self.validate()
            val_time = time.time() - val_start_time
            
            # 保存指标
            self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
            
            # 提取关键验证指标
            current_map50 = self.metrics.get('metrics/mAP50(B)', 0.0)
            current_map50_95 = self.metrics.get('metrics/mAP50-95(B)', 0.0)
            current_precision = self.metrics.get('metrics/precision(B)', 0.0)
            current_recall = self.metrics.get('metrics/recall(B)', 0.0)
            
            # 更新最佳指标
            if self.fitness > best_metrics['fitness']:
                best_metrics.update({
                    'fitness': self.fitness,
                    'mAP50': current_map50,
                    'mAP50-95': current_map50_95,
                    'epoch': epoch + 1
                })
            
            # 【关键修改3】详细的Epoch结束日志
            LOGGER.info(
                f"\n{colorstr('cyan', '╔' + '═'*70 + '╗')}\n"
                f"{colorstr('cyan', '║')} {colorstr('bold', f'Epoch {epoch + 1}/{self.epochs}')} 完成\n"
                f"{colorstr('cyan', '╠' + '═'*70 + '╣')}\n"
                f"{colorstr('cyan', '║')} {colorstr('bold', '【训练损失】')}\n"
                f"{colorstr('cyan', '║')}   平均总损失: {avg_total_loss:.4f}\n"
                f"{colorstr('cyan', '║')}   监督损失:   {avg_sup_loss:.4f} (权重: {sup_weight:.2f})\n"
                f"{colorstr('cyan', '║')}   无监督损失: {avg_unsup_loss:.4f} (权重: {unsup_weight:.2f})\n"
                f"{colorstr('cyan', '║')}     └─ Box: {avg_unsup_box:.4f} | Cls: {avg_unsup_cls:.4f}\n"
                f"{colorstr('cyan', '║')}   平均每批伪标签数: {avg_pseudo_labels:.1f}\n"
                f"{colorstr('cyan', '║')}   平均有效比例: {avg_valid_ratio:.3f}\n"
                f"{colorstr('cyan', '╠' + '═'*70 + '╣')}\n"
                f"{colorstr('cyan', '║')} {colorstr('bold', '【验证指标】')} (耗时: {val_time:.1f}s)\n"
                f"{colorstr('cyan', '║')}   Fitness:  {self.fitness:.4f} {'(最佳)' if self.fitness == best_metrics['fitness'] else ''}\n"
                f"{colorstr('cyan', '║')}   mAP50:    {current_map50:.4f}\n"
                f"{colorstr('cyan', '║')}   mAP50-95: {current_map50_95:.4f}\n"
                f"{colorstr('cyan', '║')}   Precision: {current_precision:.4f}\n"
                f"{colorstr('cyan', '║')}   Recall:   {current_recall:.4f}\n"
                f"{colorstr('cyan', '╠' + '═'*70 + '╣')}\n"
                f"{colorstr('cyan', '║')} {colorstr('bold', '【最佳记录】')}\n"
                f"{colorstr('cyan', '║')}   最佳Fitness: {best_metrics['fitness']:.4f} (Epoch {best_metrics['epoch']})\n"
                f"{colorstr('cyan', '║')}   最佳mAP50:   {best_metrics['mAP50']:.4f}\n"
                f"{colorstr('cyan', '║')}   最佳mAP50-95: {best_metrics['mAP50-95']:.4f}\n"
                f"{colorstr('cyan', '╚' + '═'*70 + '╝')}"
            )
            
            # 早停检查
            self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
            
            if self.args.time:
                self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)
            
            # 保存模型
            if self.args.save or final_epoch:
                self.save_model()
                if self.fitness == best_metrics['fitness']:
                    LOGGER.info(f"{colorstr('green', '✓')} 保存最佳模型 (fitness={self.fitness:.4f})")
            
            # 学习率调度
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch
                self.stop |= epoch >= self.epochs
            
            self._clear_memory()
            
            if self.stop:
                LOGGER.info(f"{colorstr('yellow', '训练停止')} (原因: {'早停' if self.stopper.possible_stop else '完成/超时'})")
                break
            
            epoch += 1
        
        # 训练结束总结
        total_time = time.time() - self.train_time_start
        LOGGER.info(
            f"\n{colorstr('bold', '╔' + '═'*70 + '╗')}\n"
            f"{colorstr('bold', '║')} 训练完成总结\n"
            f"{colorstr('bold', '╠' + '═'*70 + '╣')}\n"
            f"{colorstr('bold', '║')} 总轮数: {epoch - self.start_epoch + 1}\n"
            f"{colorstr('bold', '║')} 总耗时: {total_time/3600:.2f}小时 ({total_time/60:.1f}分钟)\n"
            f"{colorstr('bold', '║')} 平均每轮: {total_time/(epoch - self.start_epoch + 1)/60:.1f}分钟\n"
            f"{colorstr('bold', '╠' + '═'*70 + '╣')}\n"
            f"{colorstr('bold', '║')} {colorstr('green', '【最终最佳指标】')}\n"
            f"{colorstr('bold', '║')}   Epoch: {best_metrics['epoch']}\n"
            f"{colorstr('bold', '║')}   Fitness:  {best_metrics['fitness']:.4f}\n"
            f"{colorstr('bold', '║')}   mAP50:    {best_metrics['mAP50']:.4f}\n"
            f"{colorstr('bold', '║')}   mAP50-95: {best_metrics['mAP50-95']:.4f}\n"
            f"{colorstr('bold', '╚' + '═'*70 + '╝')}"
        )
        
        self.final_eval()
        if self.args.plots:
            self.plot_metrics()
        self._clear_memory()

    def train_successfully1(self):
        """
        修复版训练循环：
        1. 严格根据权重 (sup_weight/unsup_weight) 决定是否计算无监督损失。
        2. 优化前向传播路径，避免不必要的计算。
        3. 修正梯度累积与 Loss 标量化处理。
        """
        self._setup_train()  # 设置训练
        nb = len(self.train_loader)  # 批次数
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # 热身迭代
        last_opt_step = -1  # 最后优化步数
        
        self.epoch_time = 0  # epoch 时间
        self.epoch_time_start = time.time()  # epoch 开始时间
        self.train_time_start = time.time()  # 训练开始时间
        
        # 检查无标签加载器
        has_unlabel = self.unlabelloader is not None
        unlabel_iter = iter(self.unlabelloader) if has_unlabel else None

        LOGGER.info(
            f'图像大小 {self.args.imgsz} 训练, {self.args.imgsz} 验证\n'
            f"将结果记录到 {colorstr('bold', self.save_dir)}\n"
            f'开始训练 ' + (f"{self.args.time} 小时..." if self.args.time else f"{self.epochs} 个周期...")
        )
        
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])  # 更新绘图索引
            
        epoch = self.start_epoch
        self.optimizer.zero_grad()  # 清零梯度
        
        while True:
            self.epoch = epoch
            
            # 【关键修改 1】提前获取动态权重
            sup_weight, unsup_weight = self.get_dynamic_weights(epoch)
            
            # 严格判断是否需要计算无监督分支 (避免无效计算)
            compute_unsupervised = (
                (unsup_weight > 1e-6) and 
                has_unlabel and 
                (epoch >= MIN_EPOCH_FOR_UNSUP_FORWARD)  # <--- 新增这一行
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # 忽略警告
                self.scheduler.step()  # 更新学习率

            self.model.train()  # 设置模型为训练模式
            
            # 更新数据加载器属性（可选）
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()  # 关闭马赛克增强
                self.train_loader.reset()
                if has_unlabel:
                    self.unlabelloader.reset()
                    unlabel_iter = iter(self.unlabelloader)

            LOGGER.info(self.progress_string())  # 打印进度信息
            pbar = TQDM(enumerate(self.train_loader), total=nb)  # 进度条显示
            
            self.tloss = None  # 总损失初始化
            
            for i, batch in pbar:
                # 热身逻辑
                ni = i + nb * epoch  # 当前迭代次数
                if ni <= nw:
                    xi = [0, nw]  # x 插值
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))  # 更新累积参数
                    for j, x in enumerate(self.optimizer.param_groups):
                        # 学习率调整
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])
                
                # 获取无标签批次 (仅在需要时获取)
                unlabelbatch = None
                if compute_unsupervised and unlabel_iter is not None:
                    try:
                        unlabelbatch = next(unlabel_iter)
                    except StopIteration:
                        unlabel_iter = iter(self.unlabelloader)  # 重置迭代器
                        unlabelbatch = next(unlabel_iter)
                    # 预处理无标签数据
                    unlabelbatch = self.preprocess_batch(unlabelbatch)

                # 预处理有标签数据
                batch = self.preprocess_batch(batch)

                # 前向传播
                with autocast(self.amp):  # 自动混合精度
                    # 1. 计算有监督损失 (必须)
                    # 注意：这里假设 self.model(batch) 返回 (loss, loss_items) 或者需要在内部计算
                    # 如果 self.model 仅返回 preds，则需手动计算 loss。此处沿用旧代码逻辑假设 model 可计算 loss
                    preds = self.model(batch["img"]) 
                    self.loss, self.loss_items = self.model(batch)  # 兼容旧接口：获取 loss 和 items
                    
                    # 确保 loss 是标量
                    if not self.loss.dim() == 0:
                        self.loss = self.loss.sum()

                    # 2. 计算无监督损失 (仅在权重>0 时)
                    unsupervise_loss = torch.tensor(0.0, device=self.device)
                    unsupervise_loss_val = 0.0 # 用于日志显示的数值
                    
                    if compute_unsupervised and unlabelbatch is not None:
                        LOGGER.info(f"启用无监督分支")
                        # 教师模型预测 (No Grad)
                        with torch.no_grad():
                            teacher_preds = self.ema.ema.predict(unlabelbatch['img'])
                        
                        # 学生模型无监督前向
                        student_unsup_preds = self.model(unlabelbatch['img'])
                        
                        # 计算一致性损失
                        raw_unsup_loss, unsup_dict = self.consistent_loss(student_unsup_preds, teacher_preds)
                        
                        # 应用缩放系数 (保持与原代码一致)
                        unsupervise_loss = raw_unsup_loss * 0.1
                        
                        # 确保标量
                        if not unsupervise_loss.dim() == 0:
                            unsupervise_loss = unsupervise_loss.sum()
                            
                        unsupervise_loss_val = unsupervise_loss.item()

                    # 3. 加权合并总损失
                    # 公式：Total = Sup_Loss * sup_weight + Unsup_Loss * unsup_weight
                    total_loss = self.loss * sup_weight + unsupervise_loss * unsup_weight
                
                # 反向传播
                # 使用 scaler 进行梯度缩放 (AMP)
                self.scaler.scale(total_loss).backward()
                
                # 更新平均损失记录
                self.tloss = (
                    (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                )
                
                # 优化步骤 (基于 accumulate 策略)
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()  # 执行优化步
                    self.optimizer.zero_grad() # 显式清零梯度，防止累积错误
                    last_opt_step = ni  # 更新最后优化步数

                    # 时间停止检查
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if self.stop:
                            break

                # EMA 更新 (通常在优化步之后)
                # self.ema.update(self.model) # 如果 update 内部包含 step 逻辑，需确认位置，通常放在 optimizer.step 后

                # 日志记录与进度条更新
                loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                
                # 安全获取 unsup_loss 用于显示
                display_unsup = unsupervise_loss_val if compute_unsupervised else 0.0

                pbar.set_description(
                    ("%11s" * 2 + "%11.4g" * (2 + loss_length) + " sup_w:%.3f unsup_w:%.3f unsup_l:%.4g")
                    % (
                        f"{epoch + 1}/{self.epochs}",
                        f"{self._get_memory():.3g}G",
                        *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),
                        batch["cls"].shape[0],
                        batch["img"].shape[-1],
                        sup_weight,
                        unsup_weight,
                        display_unsup
                    )
                )
                
                if self.args.plots and ni in self.plot_idx:
                    self.plot_training_samples(batch, ni)  # 绘制训练样本

            # Epoch 结束处理
            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # 记录学习率
            final_epoch = epoch + 1 >= self.epochs  # 检查是否为最后一个 epoch
            
            # 更新 EMA 属性
            self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

            # 验证
            if self.args.val or final_epoch or getattr(self, 'stopper', None) and self.stopper.possible_stop or self.stop:
                self.metrics, self.fitness = self.validate()  # 验证模型
            
            self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})  # 保存指标
            
            # 停止条件检查
            if hasattr(self, 'stopper'):
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
            else:
                self.stop |= final_epoch
                
            if self.args.time:
                self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

            # 保存模型
            if self.args.save or final_epoch:
                self.save_model()  # 保存模型

            # 学习率调度与时间更新
            t = time.time()
            self.epoch_time = t - self.epoch_time_start  # 计算 epoch 时间
            self.epoch_time_start = t  # 更新开始时间
            
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)  # 平均 epoch 时间
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)  # 更新总 epoch
                self._setup_scheduler()  # 设置学习率调度器
                if hasattr(self.scheduler, 'last_epoch'):
                    self.scheduler.last_epoch = self.epoch
                self.stop |= epoch >= self.epochs
            
            self._clear_memory()  # 清理内存

            # 提前停止检查
            if self.stop:
                break  # 结束训练
            epoch += 1  # 更新 epoch 计数

        # 最终验证
        seconds = time.time() - self.train_time_start
        LOGGER.info(f"\n{epoch - self.start_epoch + 1} 个周期完成，耗时 {seconds / 3600:.3f} 小时。")
        self.final_eval()  # 最终评估
        if self.args.plots:
            self.plot_metrics()  # 绘制指标
        self._clear_memory()  # 清理内存
   


    def train(self):
        """
        详细日志版训练循环：
        1. 严格根据权重 (sup_weight/unsup_weight) 决定是否计算无监督损失。
        2. 使用 LOGGER.info 详细记录权重变化、损失构成、梯度状态及计算模式。
        3. 优化前向传播路径，避免不必要的计算。
        """
        self._setup_train()  # 设置训练
        nb = len(self.train_loader)  # 批次数
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # 热温迭代
        last_opt_step = -1  # 最后优化步数
        
        self.epoch_time = 0
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        
        has_unlabel = self.unlabelloader is not None
        unlabel_iter = iter(self.unlabelloader) if has_unlabel else None

        # [LOG] 训练启动概览
        LOGGER.info(
            f"\n{colorstr('bold', '═'*60)}\n"
            f"{colorstr('bold', 'YOLO 半监督训练启动')}\n"
            f"{colorstr('bold', '═'*60)}\n"
            f'图像大小：{self.args.imgsz} (训练/验证)\n'
            f"结果目录：{colorstr('bold', self.save_dir)}\n"
            f"无标签数据加载器：{'已启用' if has_unlabel else '未启用'}\n"
            f'训练模式：{f"{self.args.time} 小时限时" if self.args.time else f"{self.epochs} 个周期"}\n'
        )

        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
            
        epoch = self.start_epoch
        self.optimizer.zero_grad()
        
        while True:
            self.epoch = epoch
            
            # [LOG] 1. 获取并记录权重策略
            sup_weight, unsup_weight = self.get_dynamic_weights(epoch)
            
            # 获取一致性损失的阈值（如果有）
            current_thr = 0.25
            if hasattr(self, 'consistent_loss') and hasattr(self.consistent_loss, 'set_epoch'):
                current_thr = self.consistent_loss.set_epoch(epoch)
            
            # 严格判断是否需要计算无监督分支
            compute_unsupervised = (unsup_weight > 1e-6) and has_unlabel and (epoch >= MIN_EPOCH_FOR_UNSUP_FORWARD)
            
            # [LOG] Epoch 开始详细日志
            LOGGER.info(
                f"\n{colorstr('blue', '╔' + '═'*58 + '╗')}\n"
                f"{colorstr('blue', '║')} {colorstr('bold', f'Epoch {epoch + 1}/{self.epochs}')} 开始\n"
                f"{colorstr('blue', '║')} 学习率：{self.optimizer.param_groups[0]['lr']:.6f}\n"
                f"{colorstr('blue', '║')} 权重配置 -> 监督：{colorstr('green', f'{sup_weight:.4f}')}, 无监督：{colorstr('magenta', f'{unsup_weight:.4f}')}\n"
#                f"{colorstr('blue', '║')} 置信度阈值：{current_thr:.2f}\n"
                f"{colorstr('blue', '║')} 计算策略：{colorstr('green', '启用无监督分支') if compute_unsupervised else colorstr('red', '仅监督模式 (权重过低或无数据)')}\n"
                f"{colorstr('blue', '╚' + '═'*58 + '╝')}"
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.scheduler.step()

            self.model.train()
            
            # 重置无标签迭代器如果需要
            if has_unlabel and (epoch == self.start_epoch or (epoch == (self.epochs - self.args.close_mosaic))):
                 if epoch == (self.epochs - self.args.close_mosaic):
                    self._close_dataloader_mosaic()
                    self.train_loader.reset()
                    self.unlabelloader.reset()
                    unlabel_iter = iter(self.unlabelloader)

            LOGGER.info(self.progress_string())
            pbar = TQDM(enumerate(self.train_loader), total=nb)
            
            self.tloss = None
            
            # 用于 Epoch 统计
            epoch_unsup_count = 0
            epoch_total_unsup_loss = 0.0
            
            for i, batch in pbar:
                ni = i + nb * epoch
                
                # 热身逻辑
                if ni <= nw:
                    xi = [0, nw]
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])
                
                # 获取无标签批次
                unlabelbatch = None
                if compute_unsupervised and unlabel_iter is not None:
                    try:
                        unlabelbatch = next(unlabel_iter)
                    except StopIteration:
                        unlabel_iter = iter(self.unlabelloader)
                        unlabelbatch = next(unlabel_iter)
                    unlabelbatch = self.preprocess_batch(unlabelbatch)

                batch = self.preprocess_batch(batch)

                # 前向传播
                with autocast(self.amp):
                    # --- 1. 监督损失 ---
                    preds = self.model(batch["img"])
                    self.loss, self.loss_items = self.model(batch)
                    
                    if not self.loss.dim() == 0:
                        self.loss = self.loss.sum()
                    
                    # --- 2. 无监督损失 (条件计算) ---
                    unsupervise_loss = torch.tensor(0.0, device=self.device)
                    unsupervise_loss_val = 0.0
                    pseudo_label_count = 0
                    
                    if compute_unsupervised and unlabelbatch is not None:
                        # 使用弱增强图像作为教师模型输入
                        teacher_weak_img = unlabelbatch['weak_img']
                        # 使用强增强图像作为学生模型输入
                        student_strong_img = unlabelbatch['strong_img']
                        
                        # 教师模型预测
                        with torch.no_grad():
                            teacher_preds = self.ema.ema.predict(teacher_weak_img)
                        
                        # 学生模型预测
                        student_unsup_preds = self.model.forward(student_strong_img)
                        
                        # 计算一致性损失
                        raw_unsup_loss, unsup_dict = self.consistent_loss(student_unsup_preds, teacher_preds)
                        
                        # 统计伪标签数量 (如果字典中有)
                        pseudo_label_count = unsup_dict.get("pseudo_labels", 0) if isinstance(unsup_dict, dict) else 0
                        
                        # 缩放
                        unsupervise_loss = raw_unsup_loss
                        
                        if not unsupervise_loss.dim() == 0:
                            unsupervise_loss = unsupervise_loss.sum()
                            
                        unsupervise_loss_val = unsupervise_loss.item()
                        epoch_unsup_count += 1
                        epoch_total_unsup_loss += unsupervise_loss_val
                        
                        # [LOG] 每 N 个 batch 或第一个 batch 记录一次无监督细节
                        if i == 0 or i % (nb // 3) == 0:
                             LOGGER.info(
                                f"  [Batch {i}/{nb}] 无监督详情 -> "
                                f"Loss: {unsupervise_loss_val:.4f}, "
                                f"伪标签数：{pseudo_label_count}, "
                                f"加权后贡献：{unsupervise_loss_val * unsup_weight:.4f}"
                            )

                    # --- 3. 总损失合并 ---
                    sup_loss_contrib = self.loss.item() * sup_weight
                    unsup_loss_contrib = unsupervise_loss_val * unsup_weight
                    total_loss = self.loss * sup_weight + unsupervise_loss * unsup_weight
                    
                    # [LOG] 首个 Batch 详细损失分解
                    if i == 0:
                        LOGGER.info(
                            f"  [Batch 0] 损失分解 -> "
                            f"Sup_Loss(Raw): {self.loss.item():.4f} * {sup_weight:.4f} = {sup_loss_contrib:.4f} | "
                            f"Unsup_Loss(Raw): {unsupervise_loss_val:.4f} * {unsup_weight:.4f} = {unsup_loss_contrib:.4f} | "
                            f"Total: {total_loss.item():.4f}"
                        )

                # 反向传播
                self.scaler.scale(total_loss).backward()
                
                # 更新平均损失
                self.tloss = (
                    (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                )
                
                # 优化步骤
                if ni - last_opt_step >= self.accumulate:
                    # [LOG] 优化步前的梯度检查 (可选，调试用)
                    # grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0) 
                    
                    self.optimizer_step()
                    self.optimizer.zero_grad()
                    last_opt_step = ni
                #临时进行测试
                    #self.validator(mode="val")
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if self.stop:
                            LOGGER.info(f"\n{colorstr('yellow', '达到限时训练时间，停止训练。')}")
                            break

                # 进度条显示
                loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                display_unsup = unsupervise_loss_val if compute_unsupervised else 0.0

                pbar.set_description(
                    ("%11s" * 2 + "%11.4g" * (2 + loss_length) + " sw:%.3f uw:%.3f ul:%.4g")
                    % (
                        f"{epoch + 1}/{self.epochs}",
                        f"{self._get_memory():.3g}G",
                        *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),
                        batch["cls"].shape[0],
                        batch["img"].shape[-1],
                        sup_weight,
                        unsup_weight,
                        display_unsup
                    )
                )
                
                if self.args.plots and ni in self.plot_idx:
                    self.plot_training_samples(batch, ni)

            # [LOG] Epoch 结束总结
            avg_unsup_loss = (epoch_total_unsup_loss / epoch_unsup_count) if epoch_unsup_count > 0 else 0.0
            LOGGER.info(
                f"\n{colorstr('cyan', '═'*40)}\n"
                f"{colorstr('cyan', f'Epoch {epoch + 1} 总结')}\n"
                f"{colorstr('cyan', '═'*40)}\n"
                f"无监督分支计算次数：{epoch_unsup_count}/{nb}\n"
                f"平均无监督损失 (Raw)：{avg_unsup_loss:.4f}\n"
                f"当前权重状态 -> Sup: {sup_weight:.4f}, Unsup: {unsup_weight:.4f}\n"
            )

            # 常规 Epoch 结束处理
            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}
            final_epoch = epoch + 1 >= self.epochs
            
            self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

            if self.args.val or final_epoch or getattr(self, 'stopper', None) and self.stopper.possible_stop or self.stop:
                self.metrics, self.fitness = self.validate()
            
            self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
            
            if hasattr(self, 'stopper'):
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
            else:
                self.stop |= final_epoch
                
            if self.args.time:
                self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

            if self.args.save or final_epoch:
                self.save_model()
                LOGGER.info(f"模型已保存至：{self.save_dir / 'weights' / ('last.pt' if not final_epoch else 'best.pt')}")

            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                if hasattr(self.scheduler, 'last_epoch'):
                    self.scheduler.last_epoch = self.epoch
                self.stop |= epoch >= self.epochs
            
            self._clear_memory()

            if self.stop:
                break
            epoch += 1

        # [LOG] 训练完全结束
        seconds = time.time() - self.train_time_start
        LOGGER.info(
            f"\n{colorstr('bold', '═'*60)}\n"
            f"{colorstr('bold', '训练完成!')}\n"
            f"{colorstr('bold', '═'*60)}\n"
            f"总周期数：{epoch - self.start_epoch + 1}\n"
            f"总耗时：{seconds / 3600:.3f} 小时\n"
            f"最终评估即将开始...\n"
        )
        self.final_eval()
        if self.args.plots:
            self.plot_metrics()
        self._clear_memory()



    def _get_memory(self):
        """获取加速器的内存利用率（单位：GB）。"""
        if self.device.type == "mps":
            memory = torch.mps.driver_allocated_memory()  # MPS 设备内存
        elif self.device.type == "cpu":
            memory = 0  # CPU 内存
        else:
            memory = torch.cuda.memory_reserved()  # CUDA 设备内存
        return memory / 1e9  # 返回内存以 GB 为单位

    def _clear_memory(self):
        """在不同平台上清理加速器内存。"""
        gc.collect()  # 垃圾回收
        if self.device.type == "mps":
            torch.mps.empty_cache()  # 清空 MPS 缓存
        elif self.device.type == "cpu":
            return  # CPU 不需要操作
        else:
            torch.cuda.empty_cache()  # 清空 CUDA 缓存

    def read_results_csv(self):
        """读取 results.csv 并返回字典格式的数据。"""
        import pandas as pd  # 延迟导入以加快速度

        return pd.read_csv(self.csv).to_dict(orient="list")  # 将 CSV 转为字典

    def save_model(self):
        """保存模型训练检查点及附加元数据。"""
        import io

        # 将检查点序列化到字节缓存中（比重复调用 torch.save() 更快）
        buffer = io.BytesIO()
        torch.save(
            {
                "epoch": self.epoch,  # 当前 epoch
                "best_fitness": self.best_fitness,  # 最佳适应度
                "model": None,  # 由 EMA 派生的检查点
                "ema": deepcopy(self.ema.ema).half(),  # EMA 模型
                "updates": self.ema.updates,  # EMA 更新次数
                "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),  # 优化器状态
                "train_args": vars(self.args),  # 保存参数字典
                "train_metrics": {**self.metrics, **{"fitness": self.fitness}},  # 训练指标
                "train_results": self.read_results_csv(),  # 读取训练结果
                "date": datetime.now().isoformat(),  # 当前日期时间
            },
            buffer,
        )
        serialized_ckpt = buffer.getvalue()  # 获取序列化内容

        # 保存检查点
        self.last.write_bytes(serialized_ckpt)  # 保存 last.pt
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)  # 保存 best.pt
        if (self.save_period > 0) and (self.epoch % self.save_period == 0):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)  # 保存当前 epoch
    #通过数据文件获取数据集路径
    def get_dataset(self):
        """检查并获取数据集。"""
        data = check_det_dataset(self.args.data)  # 检查数据集
        if "yaml_file" in data:
            self.args.data = data["yaml_file"]  # 验证 'yolo train data=url.zip' 使用
        self.data = data
        return data["train"], data.get("val") or data.get("test")  # 返回训练集和验证集

    def setup_model(self):
        """加载/创建/下载模型以用于任何任务。"""
        if isinstance(self.model, torch.nn.Module):  # 如果模型已加载，则无需设置
            return

        cfg, weights = self.model, None
        ckpt = None
        if str(self.model).endswith(".pt"):
            weights, ckpt = attempt_load_one_weight(self.model)  # 尝试加载权重
            cfg = weights.yaml  # 获取配置
        elif isinstance(self.args.pretrained, (str, Path)):
            weights, _ = attempt_load_one_weight(self.args.pretrained)  # 尝试加载预训练权重
        self.model = self.get_model(cfg=cfg, weights=weights, verbose=True)  # 创建模型
        return ckpt

    def optimizer_step(self):
        """执行一次训练优化器步骤，包含梯度裁剪和 EMA 更新。"""
        self.scaler.unscale_(self.optimizer)  # 取消缩放梯度
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # 裁剪梯度
        self.scaler.step(self.optimizer)  # 执行优化器步骤
        self.scaler.update()  # 更新缩放器
        self.optimizer.zero_grad()  # 清零梯度
        if self.ema:
            self.ema.update(self.model)  # 更新 EMA

    def preprocess_batch_old(self, batch):
        """预处理一批图像，进行缩放并转换为浮点数。"""
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255  # 转换图像
        if self.args.multi_scale:
            imgs = batch["img"]
            sz = (
                    random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))
                    // self.stride
                    * self.stride
            )  # 随机大小
            sf = sz / max(imgs.shape[2:])  # 缩放因子
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # 新形状（拉伸到网格倍数）
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)  # 重新调整图像大小
            batch["img"] = imgs  # 更新批次图像
        return batch




    def preprocess_batch(self, batch):
        """预处理一批图像，进行缩放并转换为浮点数。"""
        # 检查是否为无标签数据批次（包含强增强和弱增强图像）
        if 'strong_img' in batch and 'weak_img' in batch:
            # 预处理强增强图像
            batch["strong_img"] = batch["strong_img"].to(self.device, non_blocking=True).float() / 255
            # 预处理弱增强图像
            batch["weak_img"] = batch["weak_img"].to(self.device, non_blocking=True).float() / 255
        else:
            # 有标签数据的预处理
            batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        
        # 多尺度训练
        if self.args.multi_scale:
            if 'strong_img' in batch and 'weak_img' in batch:
                # 处理无标签数据的多尺度
                strong_imgs = batch["strong_img"]
                weak_imgs = batch["weak_img"]
                
                sz = (
                    random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))
                    // self.stride
                    * self.stride
                )  # 随机大小
                sf = sz / max(strong_imgs.shape[2:])  # 缩放因子
                if sf != 1:
                    ns = [
                        math.ceil(x * sf / self.stride) * self.stride for x in strong_imgs.shape[2:]
                    ]  # 新形状（拉伸到网格倍数）
                    strong_imgs = nn.functional.interpolate(strong_imgs, size=ns, mode="bilinear", align_corners=False)  # 重新调整图像大小
                    weak_imgs = nn.functional.interpolate(weak_imgs, size=ns, mode="bilinear", align_corners=False)  # 重新调整图像大小
                batch["strong_img"] = strong_imgs  # 更新批次图像
                batch["weak_img"] = weak_imgs  # 更新批次图像
            else:
                # 处理有标签数据的多尺度
                imgs = batch["img"]
                sz = (
                    random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))
                    // self.stride
                    * self.stride
                )  # 随机大小
                sf = sz / max(imgs.shape[2:])  # 缩放因子
                if sf != 1:
                    ns = [
                        math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                    ]  # 新形状（拉伸到网格倍数）
                    imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)  # 重新调整图像大小
                batch["img"] = imgs  # 更新批次图像
        return batch













    def validate(self):
        """验证模型并返回指标。"""
        metrics = self.validator(self)  # 使用验证器进行验证
        fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())  # 使用损失作为适应度
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness  # 更新最佳适应度
        return metrics, fitness  # 返回指标和适应度

    def get_model(self, cfg=None, weights=None, verbose=True):
        """创建并返回检测模型。"""
        model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose)  # 创建检测模型
        if weights:
            model.load(weights)  # 加载权重
        return model

    def get_validator(self):
        """返回用于 YOLO 模型验证的 DetectionValidator。"""
        if 'yolo26' in str(self.args.model).lower():
            self.loss_names = "box_loss", "cls_loss"  # 移除 dfl_loss
        # 或在模型中设置 self.model.use_dfl = False
        else:
            self.loss_names = "box_loss", "cls_loss"
        return DetectionValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))  # 创建验证器

    def get_dataloader(self, dataset_path, batch_size=16, mode="train"):
        """构建并返回数据加载器。"""
        assert mode in {"train", "val"}, f"模式必须为 'train' 或 'val'，而不是 {mode}。"
        dataset = self.build_dataset(dataset_path, mode, batch_size)  # 构建数据集
        shuffle = mode == "train"  # 训练时打乱数据
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("警告 ⚠️ 'rect=True' 与 DataLoader 打乱不兼容，设置 shuffle=False")
            shuffle = False  # 如果矩形模式，关闭打乱
        workers = self.args.workers if mode == "train" else self.args.workers * 2  # 设置工作线程数
        return build_dataloader(dataset, batch_size, workers, shuffle)  # 返回数据加载器


    def build_dataset(self, img_path, mode="train", batch=None):
        """构建并返回 YOLO 数据集。"""
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)  # 网格大小
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)  # 返回数据集



    def build_unlabeleddataset(self, img_path, mode="train", batch=None):
        """构建并返回 YOLO 数据集。"""
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)  # 网格大小
        return build_unlabelyolo_dataset(self.args, img_path, batch, self.data, rect=mode == "val", stride=gs)  # 返回数据集
    
    def set_model_attributes(self):
        """设置模型属性，如类别数量和名称。"""
        self.model.nc = self.data["nc"]  # 附加类别数量到模型
        self.model.names = self.data["names"]  # 附加类别名称到模型
        self.model.args = self.args  # 附加超参数到模型

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        返回带标签的损失字典。

        对于分类不需要，但对于分割和检测是必需的。
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]  # 创建带前缀的损失名称
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # 将张量转换为 5 位小数
            return dict(zip(keys, loss_items))  # 返回字典
        else:
            return keys  # 返回损失名称

    def progress_string(self):
        """返回格式化的训练进度字符串。"""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",  # 当前周期
            "GPU_mem",  # GPU 内存
            *self.loss_names,  # 损失名称
            "Instances",  # 实例数量
            "Size",  # 图像大小
        )

    def plot_training_samples(self, batch, ni):
        """绘制带有注释的训练样本。"""
        plot_images(
            images=batch["img"],  # 图像
            batch_idx=batch["batch_idx"],  # 批次索引
            cls=batch["cls"].squeeze(-1),  # 类别
            bboxes=batch["bboxes"],  # 边界框
            paths=batch["im_file"],  # 文件路径
            fname=self.save_dir / f"train_batch{ni}.jpg",  # 保存文件名
            on_plot=self.on_plot,  # 绘图回调
        )

    def plot_metrics(self):
        """绘制来自 CSV 文件的指标。"""
        plot_results(file=self.csv, on_plot=self.on_plot)  # 保存结果图

    def plot_training_labels(self):
        """创建 YOLO 模型的标记训练图。"""
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)  # 合并所有边界框
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)  # 合并所有类别
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)  # 绘制标签

    def save_metrics(self, metrics):
        """将训练指标保存到 CSV 文件。"""
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 2  # 列数
        s = "" if self.csv.exists() else (("%s," * n % tuple(["epoch", "time"] + keys)).rstrip(",") + "\n")  # 头部
        t = time.time() - self.train_time_start  # 计算经过时间
        with open(self.csv, "a") as f:
            f.write(s + ("%.6g," * n % tuple([self.epoch + 1, t] + vals)).rstrip(",") + "\n")  # 写入数据

    def on_plot(self, name, data=None):
        """注册绘图（例如，供回调使用）。"""
        path = Path(name)  # 将名称转换为路径
        self.plots[path] = {"data": data, "timestamp": time.time()}  # 存储绘图数据和时间戳

    def final_eval(self):
        """执行最终评估和验证 YOLO 模型。"""
        ckpt = {}
        for f in self.last, self.best:
            if f.exists():
                if f is self.last:
                    ckpt = strip_optimizer(f)  # 从 last.pt 中移除优化器
                elif f is self.best:
                    k = "train_results"  # 从 last.pt 更新 best.pt 的训练指标
                    strip_optimizer(f, updates={k: ckpt[k]} if k in ckpt else None)
                    LOGGER.info(f"\n验证 {f}...")
                    self.validator.args.plots = self.args.plots  # 更新绘图参数
                    self.metrics = self.validator(model=f)  # 验证模型
                    self.metrics.pop("fitness", None)  # 移除适应度

    def check_resume(self, overrides):
        """检查是否存在恢复检查点，并相应更新参数。"""
        resume = self.args.resume  # 恢复标志
        if resume:
            try:
                exists = isinstance(resume, (str, Path)) and Path(resume).exists()  # 检查恢复路径是否存在
                last = Path(resume if exists else get_latest_run())  # 获取最新检查点

                # 检查恢复数据 YAML 是否存在，否则强制重新下载数据集
                ckpt_args = attempt_load_weights(last).args  # 获取检查点参数
                if not Path(ckpt_args["data"]).exists():
                    ckpt_args["data"] = self.args.data  # 更新数据路径

                resume = True
                self.args = get_cfg(ckpt_args)  # 重新加载配置
                self.args.model = self.args.resume = str(last)  # 重新设置模型
                for k in (
                    "imgsz",
                    "batch",
                    "device",
                    "close_mosaic",
                ):  # 允许参数更新
                    if k in overrides:
                        setattr(self.args, k, overrides[k])  # 更新参数

            except Exception as e:
                raise FileNotFoundError("恢复检查点未找到。") from e
        self.resume = resume  # 设置恢复标志

    def resume_training(self, ckpt):
        """从检查点恢复训练。"""
        if ckpt is None or not self.resume:
            return
        best_fitness = 0.0  # 最佳适应度初始化
        start_epoch = ckpt.get("epoch", -1) + 1  # 恢复的起始 epoch
        if ckpt.get("optimizer", None) is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])  # 恢复优化器状态
            best_fitness = ckpt["best_fitness"]  # 恢复最佳适应度
        if self.ema and ckpt.get("ema"):
            self.ema.ema.load_state_dict(ckpt["ema"].float().state_dict())  # 恢复 EMA
            self.ema.updates = ckpt["updates"]  # 恢复更新次数
        assert start_epoch > 0, (
            f"{self.args.model} 训练到 {self.epochs} 个周期已完成，无需恢复。\n"
            f"开始新训练，而不是恢复，i.e. 'yolo train model={self.args.model}'"
        )
        LOGGER.info(f"恢复训练 {self.args.model} 从第 {start_epoch + 1} 个周期到总共 {self.epochs} 个周期")
        if self.epochs < start_epoch:
            LOGGER.info(
                f"{self.model} 已训练 {ckpt['epoch']} 个周期。微调 {self.epochs} 个周期。"
            )
            self.epochs += ckpt["epoch"]  # 微调额外的周期
        self.best_fitness = best_fitness  # 更新最佳适应度
        self.start_epoch = start_epoch  # 更新起始 epoch
        if start_epoch > (self.epochs - self.args.close_mosaic):
            self._close_dataloader_mosaic()  # 关闭马赛克增强

    def _close_dataloader_mosaic(self):
        """更新数据加载器以停止使用马赛克增强。"""
        if hasattr(self.train_loader.dataset, "mosaic"):
            self.train_loader.dataset.mosaic = False  # 关闭马赛克
        if hasattr(self.train_loader.dataset, "close_mosaic"):
            LOGGER.info("关闭数据加载器马赛克")
            self.train_loader.dataset.close_mosaic(hyp=copy(self.args))  # 关闭马赛克增强

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """
        构建优化器。

        参数：
            model (torch.nn.Module): 要为其构建优化器的模型。
            name (str, optional): 使用的优化器名称。如果为 'auto'，则根据迭代次数选择优化器。默认值：'auto'。
            lr (float, optional): 优化器的学习率。默认值：0.001。
            momentum (float, optional): 优化器的动量因子。默认值：0.9。
            decay (float, optional): 优化器的权重衰减。默认值：1e-5。
            iterations (float, optional): 迭代次数，决定优化器类型（如果名称为 'auto'）。默认值：1e5。
        """
        g = [], [], []  # 优化器参数组
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # 归一化层，例如 BatchNorm2d()
        if name == "auto":
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' 被找到，"
                f"忽略 'lr0={self.args.lr0}' 和 'momentum={self.args.momentum}'，"
                f"自动确定最佳 'optimizer'、'lr0' 和 'momentum'... "
            )
            nc = getattr(model, "nc", 10)  # 类别数量
            lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 适配方程
            name, lr, momentum = ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0  # Adam 的学习率不高于 0.01

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname =f"{module_name}.{param_name}" if module_name else param_name  # 完整参数名称
                if "bias" in fullname:  # 偏置参数（不衰减）
                    g[2].append(param)  # 添加到偏置参数组
                elif isinstance(module, bn):  # 归一化层（不衰减）
                    g[1].append(param)  # 添加到归一化参数组
                else:  # 权重参数（衰减）
                    g[0].append(param)  # 添加到权重参数组

        # 可用的优化器
        optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD", "auto"}
        name = {x.lower(): x for x in optimizers}.get(name.lower())  # 将名称转为小写
        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)  # 创建 Adam 类优化器
        elif name == "RMSProp":
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)  # 创建 RMSProp 优化器
        elif name == "SGD":
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)  # 创建 SGD 优化器
        else:
            raise NotImplementedError(f"优化器 '{name}' 未找到。")  # 抛出未实现错误

        optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # 添加 g0（带权重衰减的权重参数）
        optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # 添加 g1（不带衰减的归一化层权重）
        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) 带参数组 "
            f'{len(g[1])} 权重(衰减=0.0)，{len(g[0])} 权重(衰减={decay})，{len(g[2])} 偏置(衰减=0.0)'
        )
        return optimizer  # 返回优化器实例