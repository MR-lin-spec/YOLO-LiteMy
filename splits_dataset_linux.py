import os
import argparse
import shutil
import random
import yaml
from collections import defaultdict
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="半监督数据集划分工具 (带类别均衡采样)")
    
    # 核心参数
    parser.add_argument('--config', type=str, required=True, help='原始数据集 YAML 配置文件路径')
    parser.add_argument('--ratio', type=float, default=0.2, help='划分为 unlabel 的比例 (0.0 - 1.0)')
    
    # 目录命名
    parser.add_argument('--labeled_dir_name', type=str, default='labeled_data', help='有标签数据子目录名')
    parser.add_argument('--unlabeled_dir_name', type=str, default='unlabel_image', help='无标签数据子目录名')
    
    # 输出文件名
    parser.add_argument('--labeled_yaml_name', type=str, default='data_labeled.yaml', help='输出的有标签 YAML 文件名')
    parser.add_argument('--unlabeled_yaml_name', type=str, default='data_unlabel.yaml', help='输出的无标签 YAML 文件名')
    
    # 安全开关
    parser.add_argument('--no_clean', action='store_true', help='如果不加此标志，脚本会在运行前自动删除旧的输出目录以防残留数据。')
    
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
            # 去重并排序，保证一致性
            img_classes[img_name] = sorted(list(set(classes)))
        else:
            # 如果没有检测到类别，标记为 -1 (背景或未知)
            img_classes[img_name] = [-1] 
    return img_classes

def split_stratified_by_class(train_img_list, val_img_list, img_classes, ratio):
    """
    【核心修改】基于类别的分层采样逻辑
    确保 labeled 数据集中，每个类别的分布比例与原始数据集尽可能一致。
    """
    x = len(train_img_list)
    z = len(val_img_list)
    a = ratio
    
    # 1. 计算 labeled 数据的总目标数量 y
    y = int(x * a)
    if y == 0 and a > 0 and x > 0: 
        y = 1 
    
    # 2. 确定 labeled_data 的组成策略
    # 目标：从 val 取一半，从 train 取一半 (保持原逻辑框架，但内部改为分层)
    target_from_val = y / 2.0
    target_from_train = y / 2.0
    
    # --- 处理验证集部分 (分层采样) ---
    actual_from_val = []
    if z > 0:
        # 将验证集图片按类别分组
        val_class_groups = defaultdict(list)
        for img in val_img_list:
            # 获取主类别 (取标签列表的第一个，若无则为-1)
            cls = img_classes.get(img, [-1])[0]
            val_class_groups[cls].append(img)
        
        # 对每个类别独立采样
        count_v_total = int(target_from_val)
        if count_v_total == 0 and target_from_val > 0: count_v_total = 1
        
        # 计算验证集每个类别应出的数量
        for cls, imgs in val_class_groups.items():
            random.shuffle(imgs)
            # 按比例从该类中抽取，或者如果总数太少则尽量全取
            # 这里简化处理：按全局比例 a 从验证集的该类中抽取，直到凑够 count_v_total
            # 更严谨的做法是计算该类在val中的占比，然后乘以 count_v_total
            cls_ratio = len(imgs) / z if z > 0 else 0
            needed = max(1, int(count_v_total * cls_ratio)) if len(imgs) > 0 else 0
            
            # 确保不超过该类实际数量
            take_count = min(needed, len(imgs))
            actual_from_val.extend(imgs[:take_count])
            
        # 如果因为取整导致数量不足，随机补齐 (防止死循环，简单补齐)
        if len(actual_from_val) < count_v_total:
            remaining = [img for img in val_img_list if img not in actual_from_val]
            random.shuffle(remaining)
            needed_more = count_v_total - len(actual_from_val)
            actual_from_val.extend(remaining[:needed_more])
            
    else:
        # 没有验证集
        pass

    # --- 处理训练集部分 (分层采样) ---
    # 计算需要从训练集拿多少张来补足 labeled 的缺口
    # 如果验证集给够了，这里就少拿；如果验证集不够，这里多拿
    current_lab_count = len(actual_from_val)
    needed_from_train_total = max(0, y - current_lab_count)
    
    # 如果验证集超发了（比如验证集很小但强行取了全部），这里可能为0
    if needed_from_train_total == 0:
        labeled_train_subset = []
    else:
        # 将训练集图片按类别分组
        train_class_groups = defaultdict(list)
        for img in train_img_list:
            cls = img_classes.get(img, [-1])[0]
            train_class_groups[cls].append(img)
        
        labeled_train_subset = []
        
        # 计算训练集每个类别应出的数量
        # 策略：保持原始训练集中各类别的比例
        for cls, imgs in train_class_groups.items():
            random.shuffle(imgs)
            cls_ratio = len(imgs) / x if x > 0 else 0
            needed = max(1, int(needed_from_train_total * cls_ratio)) if len(imgs) > 0 else 0
            
            # 关键：对于极少样本的类别，尽量保留至少一个（如果该类总数<=2且需要>0）
            # 防止小类别在 labeled 中完全消失
            if len(imgs) <= 2 and needed == 0 and needed_from_train_total > 0:
                needed = 1
            
            take_count = min(needed, len(imgs))
            labeled_train_subset.extend(imgs[:take_count])
        
        # 同样，如果因为取整导致数量不足，随机补齐
        if len(labeled_train_subset) < needed_from_train_total:
            remaining = [img for img in train_img_list if img not in labeled_train_subset]
            random.shuffle(remaining)
            needed_more = needed_from_train_total - len(labeled_train_subset)
            labeled_train_subset.extend(remaining[:needed_more])

    # 3. 构建 unlabeled_data
    # 来源：训练集中未被选入 labeled 的部分
    # 注意：为了保持随机性，我们从剩余池中再随机抽取一部分作为 unlabeled
    # 原逻辑：len(remaining) * (1 - a)
    
    # 先找出所有未被选入 labeled 的训练集图片
    labeled_set = set(labeled_train_subset)
    remaining_train_imgs = [img for img in train_img_list if img not in labeled_set]
    
    rem_count = len(remaining_train_imgs)
    # 原逻辑中的 (1-a) 其实有点奇怪，因为 a 是 labeled 比例。
    # 通常 unlabeled 就是剩下的所有，或者剩下的一部分。
    # 依照原代码逻辑：count_unlab = int(rem_count * (1 - a))
    # 这意味着 unlabeled 也是剩余数据的一个子集，而不是全部剩余。
    count_unlab = int(rem_count * (1 - a))
    
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
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.webp']:
            potential_src = Path(source_images_dir) / f"{img_name}{ext}"
            if potential_src.exists():
                src_img = potential_src
                break
        
        if src_img:
            shutil.copy2(str(src_img), str(dest_images / src_img.name))
            count_img += 1
            
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

    print(f"--- YOLO 半监督数据集划分工具 (类别均衡采样版) ---")
    
    try:
        cfg = load_config(args.config)
    except Exception as e:
        print(f"❌ 错误：无法加载配置文件 - {e}")
        return

    root = cfg['root']
    train_src_abs = cfg['train_abs']
    val_src_abs = cfg['val_abs']
    val_sub = cfg['val_sub']
    
    print(f"📂 数据集根目录：{root}")
    
    if not train_src_abs or not train_src_abs.exists():
        print(f"❌ 错误：原始训练集图片目录不存在")
        return

    # === 智能查找标签目录 ===
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
            print(f"🔍 找到标准标签目录：{labels_src_abs}")

    if not labels_src_abs:
        candidate = root / "train" / "labels"
        if candidate.exists():
            labels_src_abs = candidate
            print(f"🔍 找到旧式标签目录：{labels_src_abs}")

    if not labels_src_abs:
        candidate = root / "labels"
        if candidate.exists():
            if list(candidate.glob("*.txt")):
                labels_src_abs = candidate
                print(f"🔍 找到扁平标签目录：{labels_src_abs}")

    # === 获取图片列表 ===
    train_all_imgs = [f.stem for f in train_src_abs.glob("*.*") if f.suffix.lower() in ['.jpg','.jpeg','.png','.bmp','.webp']]
    val_all_imgs = []
    if val_src_abs and val_src_abs.exists():
        val_all_imgs = [f.stem for f in val_src_abs.glob("*.*") if f.suffix.lower() in ['.jpg','.jpeg','.png','.bmp','.webp']]
    
    if not train_all_imgs:
        print("❌ 错误：未找到任何训练集图片。")
        return

    print(f"📊 统计：训练集 {len(train_all_imgs)} 张，验证集 {len(val_all_imgs)} 张")

    # === 构建类别映射 (关键步骤) ===
    img_classes = {}
    valid_train_imgs = train_all_imgs
    
    if labels_src_abs:
        print("🏷️  正在读取标签以进行类别均衡采样...")
        img_classes = get_image_classes(labels_src_abs)
        # 过滤掉没有对应标签的图片（可选，视情况而定，这里保留原逻辑）
        # 如果图片在 train_all_imgs 但不在 img_classes 中，说明没标签，归为 -1 类或跳过
        # 这里我们给没标签的图片分配 -1 类，保证它们也能被采样到
        for img in train_all_imgs:
            if img not in img_classes:
                img_classes[img] = [-1]
        valid_train_imgs = train_all_imgs
    else:
        print("⚠️ 未找到标签目录，将退化为普通随机采样（无法保证类别均衡）。")
        for img in train_all_imgs:
            img_classes[img] = [-1]

    # === 执行新的分层划分逻辑 ===
    print(f"🔄 按类别均衡逻辑划分 (比例 a={args.ratio})...")
    
    labeled_train_list, labeled_val_list, unlabeled_train_list = split_stratified_by_class(
        valid_train_imgs, 
        val_all_imgs, 
        img_classes, 
        args.ratio
    )
    
    total_labeled_count = len(labeled_train_list) + len(labeled_val_list)
    total_unlabeled_count = len(unlabeled_train_list)
    
    print(f"✅ 划分结果:")
    print(f"   - Labeled (有标签): {total_labeled_count} 张 (训练集来源:{len(labeled_train_list)}, 验证集来源:{len(labeled_val_list)})")
    print(f"   - Unlabeled (无标签): {total_unlabeled_count} 张")
    
    # 简单的类别分布检查打印
    if labels_src_abs:
        def count_dist(img_list, name):
            dist = defaultdict(int)
            for img in img_list:
                cls = img_classes.get(img, [-1])[0]
                dist[cls] += 1
            total = sum(dist.values())
            print(f"   [{name}] 类别分布示例 (前5类): ")
            sorted_dist = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:5]
            for c, count in sorted_dist:
                print(f"      类 {c}: {count} ({count/total:.2%})")
        
        if labeled_train_list: count_dist(labeled_train_list, "Labeled-Train")
        if unlabeled_train_list: count_dist(unlabeled_train_list, "Unlabeled-Train")

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

    # === 复制有标签数据 ===
    if total_labeled_count > 0:
        c1_img, c1_lbl = copy_files_with_labels(
            train_src_abs, labels_src_abs, labeled_train_list, 
            root, args.labeled_dir_name
        )
        
        val_labels_src = None
        if labels_src_abs and val_src_abs:
            if "train" in str(labels_src_abs):
                cand = Path(str(labels_src_abs).replace("/train", "/val"))
                if cand.exists(): val_labels_src = cand
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
        
        print(f"💾 有标签数据已存入：{labeled_dest_path} (图:{c1_img+c2_img}, 标:{c1_lbl+c2_lbl})")

    # === 复制无标签数据 ===
    if unlabeled_train_list:
        dest_images = Path(root) / args.unlabeled_dir_name / "images"
        dest_images.mkdir(parents=True, exist_ok=True)
        
        count_u = 0
        for img_name in unlabeled_train_list:
            src_img = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.webp']:
                potential_src = Path(train_src_abs) / f"{img_name}{ext}"
                if potential_src.exists():
                    src_img = potential_src
                    break
            if src_img:
                shutil.copy2(str(src_img), str(dest_images / src_img.name))
                count_u += 1
        print(f"💾 无标签数据已存入：{unlabel_dest_path}/images (图:{count_u})")

    # === 生成 YAML ===
    yaml_labeled_path, _ = generate_yaml(
        output_dir=output_yaml_dir, root_path=str(root),
        train_sub=f"{args.labeled_dir_name}/images", val_sub=val_sub,
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

    print("\n🎉 完成！数据已按类别均衡逻辑划分并复制。")

if __name__ == "__main__":
    main()