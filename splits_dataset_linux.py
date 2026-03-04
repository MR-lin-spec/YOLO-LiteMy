import os
import argparse
import shutil
import random
import yaml
from collections import defaultdict
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="半监督数据集划分工具 (带自动清理功能)")
    
    # 核心参数
    parser.add_argument('--config', type=str, required=True, help='原始数据集 YAML 配置文件路径')
    parser.add_argument('--ratio', type=float, default=0.2, help='划分为 unlabel 的比例 (0.0 - 1.0)')
    # 注意：虽然参数名叫 ratio，但在你的新逻辑中，它被用作计算基准 a%
    
    # 目录命名
    parser.add_argument('--labeled_dir_name', type=str, default='labeled_data', help='有标签数据子目录名')
    parser.add_argument('--unlabeled_dir_name', type=str, default='unlabel_image', help='无标签数据子目录名')
    
    # 输出文件名
    parser.add_argument('--labeled_yaml_name', type=str, default='data_labeled.yaml', help='输出的有标签 YAML 文件名')
    parser.add_argument('--unlabeled_yaml_name', type=str, default='data_unlabel.yaml', help='输出的无标签 YAML 文件名')
    
    # 安全开关
    parser.add_argument('--no_clean', action='store_true', help='如果不加此标志，脚本会在运行前自动删除旧的输出目录以防残留数据。加上此标志则保留旧数据（仅追加/覆盖）。')
    
    return parser.parse_args()

def normalize_path(p):
    """将路径统一转换为 Linux 风格的正斜杠字符串"""
    if p is None:
        return ""
    return str(p).replace("\\", "/")

def load_config(config_path):
    """加载并解析原始 YAML 配置"""
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在：{config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    if 'path' not in data or 'train' not in data:
        raise ValueError("YAML 文件缺少 'path' 或 'train' 字段")
    
    root_path = Path(data['path']).resolve()
    train_sub = data['train']
    val_sub = data.get('val', '')
    
    train_abs = root_path / train_sub if train_sub else None
    val_abs = root_path / val_sub if val_sub else None
    
    return {
        'root': root_path,
        'train_abs': train_abs,
        'val_abs': val_abs,
        'val_sub': val_sub, 
        'names': data.get('names', {}),
        'nc': data.get('nc', len(data.get('names', {}))),
    }

def get_image_classes(labels_dir):
    """读取标签，建立 {图片名: [类别 ID 列表]} 映射"""
    img_classes = defaultdict(list)
    if not labels_dir or not labels_dir.exists():
        return img_classes

    label_files = list(labels_dir.glob("*.txt"))
    for label_file in label_files:
        img_name = label_file.stem 
        classes = []
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) > 0:
                        class_id = int(parts[0])
                        classes.append(class_id)
        except Exception:
            continue
            
        if classes:
            img_classes[img_name] = list(set(classes))
        else:
            img_classes[img_name] = [-1] 
    return img_classes

def split_by_user_logic(train_img_list, val_img_list, ratio):
    """
    严格按照用户指定的逻辑进行划分：
    假设制定比例为 a% (ratio)，训练集一共有 x 张，验证集有 z 张。
    1. 计算基准数 y = x * a%。
    2. 构建 labeled_data (有标签集):
       - 目标是从验证集取 y/2，从训练集取 y/2。
       - 如果 z < y/2，则验证集全取 (z)，剩下 (y/2 - z) 从训练集取。
       - 最终 labeled_data 包含：选中的验证集图片 + 选中的训练集图片。
    3. 构建 unlabeled_data (无标签集):
       - 来源：仅从【剩余的训练集】中抽取。
       - 剩余训练集数量 x' = x - (实际从训练集取走的数量)。
       - 抽取数量 = x' * (1 - a%)。
    
    返回:
        labeled_imgs: 包含 (部分 train + 部分 val) 的文件名列表
        unlabeled_imgs: 包含 (部分 train) 的文件名列表
        (注意：为了兼容后续 copy 逻辑，后续代码需要知道哪些来自 train 哪些来自 val，
         但原 copy 逻辑只针对 train_src_abs。
         **重要调整**：原脚本逻辑似乎只从 train_src_abs 复制图片到 labeled/unlabeled 目录。
         如果要把验证集的图片也混入 labeled_data，我们需要确保能访问验证集的图片路径。
         
         观察原脚本 copy_files_with_labels 函数，它只接受一个 source_images_dir。
         如果 labeled_data 混合了 train 和 val 的图片，而它们物理路径不同，原复制逻辑会失效。
         
         **重新审视需求与代码架构的冲突**：
         用户要求：“从验证集中抽取...复制到代码中的 labeled_data”。
         原代码结构：
            copy_files_with_labels(source_images_dir=train_src_abs, ..., img_list=labeled_imgs, ...)
         如果 labeled_imgs 里包含了 val 的图片名，而 train_src_abs 下没有这些图（通常 train 和 val 是分开的文件夹），复制会失败。
         
         **解决方案**：
         通常 YOLO 数据集结构中，images/train 和 images/val 是分开的。
         如果要合并到 labeled_data，必须同时从两个源复制。
         但用户强调“只要修改数据划分部分，其他接口和代码尽量都严格保持不变”。
         
         这可能意味着：
         情况 A: 用户的 train 和 val 图片都在同一个大目录下（通过 yaml 的 path 和子目录区分，但物理上可能混在一起？不太可能）。
         情况 B: 用户希望逻辑上划分，但物理复制时，如果原脚本只支持从 train 目录复制，那么从 val 抽取的图片将无法被原脚本复制。
         
         **最合理的推断**：
         用户可能希望将 val 的一部分**移动**或**合并**到 labeled 集合中。
         为了在不大幅重构 `copy_files_with_labels` (它只接受一个源目录) 的前提下实现这一点，
         我们需要修改 `copy_files_with_labels` 或者在调用时做特殊处理。
         但用户说“其他接口和代码尽量都严格保持不变”。
         
         让我们再看一眼 `copy_files_with_labels` 的定义：
         def copy_files_with_labels(source_images_dir, source_labels_dir, img_list, dest_root, subdir_name):
         
         如果我把 val 的图片放进 labeled_imgs 列表，传给这个函数，它会去 source_images_dir (即 train_src_abs) 找图。找不到就会跳过。
         这会导致 labeled_data 图片数量不足。
         
         **必须做的最小改动以支持功能**：
         我需要修改 `copy_files_with_labels` 使其支持多个源目录，或者在 main 函数中分别调用。
         既然用户说“只要修改数据划分部分...严格遵循我给你的信息”，这可能是一个陷阱。
         如果我不改复制逻辑，功能无法实现。
         但如果我改了复制逻辑，就违反了“其他代码尽量严格保持不变”。
         
         **折中方案**：
         在 `main` 函数中，针对 labeled_data，分别调用两次复制逻辑（一次针对 train 源，一次针对 val 源），或者修改 `copy_files_with_labels` 使其更通用。
         考虑到 `copy_files_with_labels` 是内部 helper 函数，修改它以支持多源或分别调用是必要的，否则逻辑不通。
         为了最小侵入，我将在 `main` 函数中处理 labeled 数据的复制逻辑，将其拆分为 train 部分和 val 部分分别复制。
         
         等等，用户说“只要修改数据划分部分”。
         也许用户的数据集结构是 train 和 val 图片都在同一个 `images` 文件夹下？
         不，yaml 里有 train: ... val: ... 通常是子目录。
         
         让我们假设用户愿意接受在 `main` 函数中对复制流程做微小的适配，以容纳新的划分逻辑带来的多源需求。
         或者，更可能的是，用户希望把 val 的图片**当作**训练集的一部分来处理。
         
         **决定**：
         1. 修改 `split_by_user_logic` 返回三个列表：
            - labeled_from_train
            - labeled_from_val
            - unlabeled_from_train
         2. 在 `main` 函数中，调用复制函数时：
            - 对 labeled_from_train: 源是 train_src_abs
            - 对 labeled_from_val: 源是 val_src_abs (需要从 config 获取)
            - 对 unlabeled_from_train: 源是 train_src_abs
         
         这需要我在 `load_config` 中确保返回了 `val_abs` (已存在)，并在 `main` 中稍微调整复制调用。这是实现用户逻辑的唯一可行路径。
    """
    
    x = len(train_img_list)
    z = len(val_img_list)
    a = ratio
    
    # 1. 计算基准数 y
    y = int(x * a)
    if y == 0 and a > 0: y = 1 # 至少取一张，如果比例极小
    
    # 2. 确定 labeled_data 的组成
    # 目标：从 val 取 y/2, 从 train 取 y/2
    target_from_val = y / 2.0
    target_from_train = y / 2.0
    
    actual_from_val = []
    actual_from_train = []
    
    # 处理验证集部分
    if z >= target_from_val:
        # 验证集足够，取一半
        count_v = int(target_from_val)
        if count_v == 0 and target_from_val > 0: count_v = 1
        # 随机打乱验证集
        random.shuffle(val_img_list)
        actual_from_val = val_img_list[:count_v]
    else:
        # 验证集不够，全取
        actual_from_val = val_img_list[:] # 全部
        # 缺少的部分从训练集补
        missing = target_from_val - z
        target_from_train += missing
        
    # 处理训练集部分 (用于 labeled)
    # 注意：这里需要先打乱训练集，然后截取
    # 但要小心，后面 unlabeled 还要从剩下的里面取，所以不能直接切片改变原列表引用，要拷贝或记录索引
    # 为了简单，我们先生成索引或副本
    
    # 重新整理逻辑，避免状态依赖错误
    # Step A: 确定 labeled 需要的 train 数量
    needed_from_train_for_lab = int(target_from_train)
    if needed_from_train_for_lab == 0 and target_from_train > 0: needed_from_train_for_lab = 1
    if needed_from_train_for_lab > x: needed_from_train_for_lab = x
    
    # 打乱训练集
    random.shuffle(train_img_list)
    
    # 选取 labeled 用的 train 图片
    labeled_train_subset = train_img_list[:needed_from_train_for_lab]
    
    # 剩余的训练集图片 (用于计算 unlabeled)
    remaining_train_imgs = train_img_list[needed_from_train_for_lab:]
    
    # 3. 构建 unlabeled_data
    # 来源：remaining_train_imgs
    # 数量：len(remaining) * (1 - a%)
    rem_count = len(remaining_train_imgs)
    count_unlab = int(rem_count * (1 - a))
    
    # 再次打乱剩余部分以确保随机性 (虽然上面已经打乱了整体，切片后顺序相对固定，再打乱一次更保险)
    random.shuffle(remaining_train_imgs)
    
    unlabeled_subset = remaining_train_imgs[:count_unlab]
    
    return labeled_train_subset, actual_from_val, unlabeled_subset

def safe_remove_dir(dir_path):
    """安全删除目录，如果存在的话"""
    if dir_path.exists():
        print(f"🧹 检测到旧目录 {dir_path}，正在清除以确保数据纯净...")
        shutil.rmtree(dir_path)
        print(f"   -> 已清除 {dir_path}")

def copy_files_with_labels(source_images_dir, source_labels_dir, img_list, dest_root, subdir_name):
    """复制图片和标签"""
    if not img_list:
        return 0, 0
        
    dest_images = Path(dest_root) / subdir_name / "images"
    dest_labels = Path(dest_root) / subdir_name / "labels"
    
    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)
    
    count_img = 0
    count_lbl = 0
    
    for img_name in img_list:
        src_img = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG']:
            potential_src = Path(source_images_dir) / f"{img_name}{ext}"
            if potential_src.exists():
                src_img = potential_src
                break
        
        if src_img:
            shutil.copy2(str(src_img), str(dest_images / src_img.name))
            count_img += 1
            
            # 标签目录可能为 None (如果是无标签模式)
            if source_labels_dir:
                src_lbl = Path(source_labels_dir) / f"{img_name}.txt"
                if src_lbl.exists():
                    shutil.copy2(str(src_lbl), str(dest_labels / f"{img_name}.txt"))
                    count_lbl += 1
    return count_img, count_lbl

def generate_yaml(output_dir, root_path, train_sub, val_sub, names, nc, yaml_name, is_unlabel=False):
    """生成 YAML 文件"""
    r_path = normalize_path(root_path)
    t_sub = normalize_path(train_sub)
    v_sub = normalize_path(val_sub) if val_sub else ""
    
    lines = [f"path: {r_path}"]
    lines.append(f"train: {t_sub}")
    lines.append(f"val: {v_sub}")
    
    if not is_unlabel:
        lines.append(f"nc: {nc}")
        lines.append("names:")
        if isinstance(names, list):
            names_dict = {str(i): name for i, name in enumerate(names)}
            for k, v in sorted(names_dict.items(), key=lambda x: int(x[0])):
                lines.append(f"  {k}: {v}")
        elif isinstance(names, dict):
            for k, v in sorted(names.items()):
                lines.append(f"  {k}: {v}")
    else:
        lines.append("is_unlabel: True")
        
    content = "\n".join(lines) + "\n"
    yaml_path = Path(output_dir) / yaml_name
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return yaml_path, content

def main():
    args = parse_args()
    random.seed(42)
    
    script_dir = Path(__file__).resolve().parent
    output_yaml_dir = script_dir 

    print(f"--- YOLO 半监督数据集划分工具 (带清理模式 - 用户定制逻辑) ---")
    
    try:
        cfg = load_config(args.config)
    except Exception as e:
        print(f"❌ 错误：无法加载配置文件 - {e}")
        return

    root = cfg['root']
    train_src_abs = cfg['train_abs']
    val_src_abs = cfg['val_abs'] # 获取验证集绝对路径
    val_sub = cfg['val_sub']
    
    print(f"📂 数据集根目录：{root}")
    
    if not train_src_abs or not train_src_abs.exists():
        print(f"❌ 错误：原始训练集图片目录不存在")
        return

    # === 智能查找标签目录 (保持原逻辑) ===
    labels_src_abs = None
    if "images" in str(train_src_abs):
        potential_labels = Path(str(train_src_abs).replace("/images", "/labels"))
        if potential_labels.exists():
            labels_src_abs = potential_labels
            print(f"🔍 自动推断标签目录 (images->labels): {labels_src_abs}")

    if not labels_src_abs:
        candidate = root / "labels" / "train"
        if candidate.exists():
            labels_src_abs = candidate
            print(f"🔍 找到标准标签目录: {labels_src_abs}")

    if not labels_src_abs:
        candidate = root / "train" / "labels"
        if candidate.exists():
            labels_src_abs = candidate
            print(f"🔍 找到旧式标签目录: {labels_src_abs}")

    if not labels_src_abs:
        candidate = root / "labels"
        if candidate.exists():
            if list(candidate.glob("*.txt")):
                labels_src_abs = candidate
                print(f"🔍 找到扁平标签目录: {labels_src_abs}")

    # === 获取图片列表 ===
    # 训练集图片
    train_all_imgs = [f.stem for f in train_src_abs.glob("*.*") if f.suffix.lower() in ['.jpg','.jpeg','.png','.bmp','.webp']]
    # 验证集图片 (如果存在)
    val_all_imgs = []
    if val_src_abs and val_src_abs.exists():
        val_all_imgs = [f.stem for f in val_src_abs.glob("*.*") if f.suffix.lower() in ['.jpg','.jpeg','.png','.bmp','.webp']]
    
    if not train_all_imgs:
        print("❌ 错误：未找到任何训练集图片。")
        return

    print(f"📊 统计：训练集 {len(train_all_imgs)} 张，验证集 {len(val_all_imgs)} 张")

    # === 执行新的划分逻辑 ===
    print(f"🔄 按用户定制逻辑划分 (比例 a={args.ratio})...")
    
    # 如果没有找到标签目录，依然构建 img_classes 占位，但新逻辑主要依赖文件名列表
    # 为了兼容性，我们保留 img_classes 的构建，但划分主要靠上面的函数
    if labels_src_abs:
        img_classes = get_image_classes(labels_src_abs)
        # 过滤掉没有对应图片的标签（以防万一）
        valid_train_imgs = [img for img in train_all_imgs if img in img_classes]
        # 如果验证集也有标签，可以类似处理，这里简化处理，假设验证集图片都是有效的
    else:
        img_classes = {}
        valid_train_imgs = train_all_imgs
        print("⚠️ 未找到标签目录，按纯文件名划分。")
        valid_train_imgs = train_all_imgs

    # 调用新划分函数
    # 注意：传入 valid_train_imgs (可能被标签过滤过) 和 val_all_imgs
    labeled_train_list, labeled_val_list, unlabeled_train_list = split_by_user_logic(
        valid_train_imgs, 
        val_all_imgs, 
        args.ratio
    )
    
    # 合并 labeled 列表用于统计，但复制时要分开源
    total_labeled_count = len(labeled_train_list) + len(labeled_val_list)
    total_unlabeled_count = len(unlabeled_train_list)
    
    print(f"✅ 划分结果:")
    print(f"   - Labeled (有标签): {total_labeled_count} 张 (训练集来源:{len(labeled_train_list)}, 验证集来源:{len(labeled_val_list)})")
    print(f"   - Unlabeled (无标签): {total_unlabeled_count} 张 (均来自训练集剩余)")

    # === 清理旧目录 ===
    labeled_dest_path = root / args.labeled_dir_name
    unlabel_dest_path = root / args.unlabeled_dir_name

    if not args.no_clean:
        if total_labeled_count > 0:
            safe_remove_dir(labeled_dest_path)
        if total_unlabeled_count > 0:
            safe_remove_dir(unlabel_dest_path)
    else:
        print("⚠️ 检测到 --no_clean 标志，将跳过清理步骤。")

    # === 复制有标签数据 (分两部分复制) ===
    if total_labeled_count > 0:
        # 1. 复制来自训练集的部分
        c1_img, c1_lbl = copy_files_with_labels(
            train_src_abs, labels_src_abs, labeled_train_list, 
            root, args.labeled_dir_name
        )
        # 2. 复制来自验证集的部分 (需要找到验证集对应的标签目录)
        # 推断验证集标签目录逻辑同训练集
        val_labels_src = None
        if labels_src_abs:
            # 尝试根据 train labels 推导 val labels
            # 假设结构一致：labels/train -> labels/val
            if "train" in str(labels_src_abs):
                cand = Path(str(labels_src_abs).replace("/train", "/val"))
                if cand.exists(): val_labels_src = cand
            # 如果没找到，尝试 root/labels/val
            if not val_labels_src:
                cand = root / "labels" / "val"
                if cand.exists(): val_labels_src = cand
        
        c2_img, c2_lbl = 0, 0
        if labeled_val_list:
            if val_src_abs:
                c2_img, c2_lbl = copy_files_with_labels(
                    val_src_abs, val_labels_src, labeled_val_list,
                    root, args.labeled_dir_name
                )
            else:
                print("⚠️ 警告：需要从验证集复制图片，但未找到验证集图片目录。")
        
        print(f"💾 有标签数据已存入：{labeled_dest_path} (图:{c1_img+c2_img}, 标:{c1_lbl+c2_lbl})")

    # === 复制无标签数据 (仅图片，通常无标签或标签被移除，这里按原逻辑只复制图片) ===
    # 原代码中 unlabeled 也会尝试复制标签，但半监督中通常 unlabeled 不需要标签文件或标签文件为空
    # 保持原代码逻辑：如果有标签文件也复制过去（或者用户可以手动删除）
    if unlabeled_train_list:
        # 无标签数据通常只需要图片，但为了保持目录结构一致性，我们调用原函数
        # 注意：原函数会尝试复制标签。如果希望彻底无标签，可以在复制后删除 labels 目录，
        # 但用户要求“其他代码尽量严格保持不变”，所以保留原行为。
        dest_images = Path(root) / args.unlabeled_dir_name / "images"
        dest_images.mkdir(parents=True, exist_ok=True)
        
        # 为了利用已有的 copy_files_with_labels 函数保持一致性，我们调用它
        # 但要注意，unlabeled 数据通常放在单独的目录，且可能不需要 labels 子目录
        # 原脚本在 main 的下半部分有一段专门处理 unlabeled 的循环，没有用 copy_files_with_labels
        # 让我们看原脚本最后一段：
        # """
        # if unlabel_imgs:
        #     dest_images = Path(root) / args.unlabeled_dir_name / "images"
        #     dest_images.mkdir(parents=True, exist_ok=True)
        #     ... (手动循环复制)
        # """
        # 为了严格遵守“其他代码尽量严格保持不变”，我应该恢复那段手动循环逻辑，或者直接用那个逻辑。
        # 原脚本中 `copy_files_with_labels` 主要用于 labeled。
        # 而 unlabeled 在原文中是单独写的循环。
        # 我将沿用原文中 unlabeled 的处理方式（只复制图片到 images 目录），以最小化变动。
        
        count_u = 0
        for img_name in unlabeled_train_list:
            src_img = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG']:
                potential_src = Path(train_src_abs) / f"{img_name}{ext}"
                if potential_src.exists():
                    src_img = potential_src
                    break
            if src_img:
                shutil.copy2(str(src_img), str(dest_images / src_img.name))
                count_u += 1
        print(f"💾 无标签数据已存入：{unlabel_dest_path}/images (图:{count_u})")

    # === 生成 YAML ===
    # 注意：labeled_data 现在混合了 train 和 val 的图片，但物理上都放在了 labeled_dir_name/images 下
    # 所以 yaml 中的 train 路径指向 labeled_dir_name/images 是正确的
    yaml_labeled_path, _ = generate_yaml(
        output_dir=output_yaml_dir, root_path=str(root),
        train_sub=f"{args.labeled_dir_name}/images", val_sub=val_sub, # val_sub 保留原始验证集配置？
        # 在半监督场景下，通常 labeled_data 充当新的 train，原始的 val 可能仍然作为 val
        # 或者用户希望用原始 val 的剩余部分作为 val？
        # 既然用户没特别说明，保持 yaml 中的 val 指向原始验证集目录（如果有的话）
        # 但如果我们把原始 val 的一部分挪走了，原始 val 目录里的图片还在，只是逻辑上被挪用了。
        # 这可能会导致数据泄露（同一张图既在 labeled train 又在 original val）。
        # **严重问题**：如果把 val 的图片物理复制到了 labeled_data，原始 val 目录下的图片并没有被删除。
        # 训练时如果同时用 labeled_data (train) 和 original val，会有重复数据。
        # 但用户只要求“复制”，没要求“移动”。
        # 按照用户指令“复制到代码中的 labeled_data”，我们只做复制。
        # YAML 配置保持原样，指向新生成的目录。
        names=cfg['names'], nc=cfg['nc'],
        yaml_name=args.labeled_yaml_name, is_unlabel=False
    )
    print(f"📄 已生成：{yaml_labeled_path}")

    yaml_unlabel_path, _ = generate_yaml(
        output_dir=output_yaml_dir, root_path=str(root),
        train_sub=f"{args.unlabeled_dir_name}/images", val_sub="",
        names={}, nc=0,
        yaml_name=args.unlabeled_yaml_name, is_unlabel=True
    )
    print(f"📄 已生成：{yaml_unlabel_path}")

    print("\n🎉 完成！数据已按定制逻辑划分并复制。")
    print("⚠️ 注意：由于是从验证集复制图片到 labeled_data，请确保训练时验证集配置不会导致数据重复（如需移动而非复制，请手动删除原验证集对应图片）。")

if __name__ == "__main__":
    main()