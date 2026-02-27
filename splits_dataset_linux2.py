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
    """读取标签，建立 {图片名: [类别ID列表]} 映射"""
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

def split_by_class(img_classes, ratio):
    """按类别均匀抽取图片"""
    class_to_imgs = defaultdict(list)
    for img, classes in img_classes.items():
        if not classes: continue
        main_class = classes[0]
        class_to_imgs[main_class].append(img)

    selected_unlabel_imgs = set()
    for cls, imgs in class_to_imgs.items():
        if not imgs: continue
        if cls == -1:
            count = int(len(imgs) * ratio)
        else:
            count = max(1, int(len(imgs) * ratio)) if len(imgs) > 0 and ratio > 0 else 0
        
        if count >= len(imgs):
            selected = imgs
        else:
            random.shuffle(imgs)
            selected = imgs[:count]
        selected_unlabel_imgs.update(selected)

    unlabel_list = list(selected_unlabel_imgs)
    labeled_list = [img for img in img_classes.keys() if img not in selected_unlabel_imgs]
    return unlabel_list, labeled_list

def safe_remove_dir(dir_path):
    """安全删除目录，如果存在的话"""
    if dir_path.exists():
        print(f"🧹 检测到旧目录 {dir_path}，正在清除以确保数据纯净...")
        shutil.rmtree(dir_path)
        print(f"   -> 已清除 {dir_path}")

def copy_files_with_labels(source_images_dir, source_labels_dir, img_list, dest_root, subdir_name):
    """复制图片和标签"""
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
        # --- 修改开始 ---
        if isinstance(names, list):
            # 如果 names 是列表，转换为字典格式输出，或者直接按列表逻辑处理
            # 这里为了兼容原有逻辑，我们把它转成字典再遍历，或者直接写列表格式到 yaml
            # 原脚本似乎倾向于输出字典格式的 yaml，所以我们这里做个转换
            names_dict = {str(i): name for i, name in enumerate(names)}
            for k, v in sorted(names_dict.items(), key=lambda x: int(x[0])):
                lines.append(f"  {k}: {v}")
        elif isinstance(names, dict):
            # 原有的字典处理逻辑
            for k, v in sorted(names.items()):
                lines.append(f"  {k}: {v}")
    else:
        lines.append("is_unlabel: True")  # 如果 names 既不是列表也不是字典，标记为无标签数据集
        
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

    print(f"--- YOLO 半监督数据集划分工具 (带清理模式) ---")
    
    try:
        cfg = load_config(args.config)
    except Exception as e:
        print(f"❌ 错误：无法加载配置文件 - {e}")
        return

    root = cfg['root']
    train_src_abs = cfg['train_abs']
    val_sub = cfg['val_sub']
    
    # ... (前文代码：加载 config, 确定 root, train_src_abs 不变) ...

    print(f"📂 数据集根目录：{root}")
    
    if not train_src_abs or not train_src_abs.exists():
        print(f"❌ 错误：原始训练集图片目录不存在")
        return

    # === 修改开始：智能查找标签目录 ===
    labels_src_abs = None
    
    # 策略 1: 根据图片路径推断 (例如 images/train -> labels/train)
    # 假设结构是 root/images/train 和 root/labels/train
    if "images" in str(train_src_abs):
        # 将路径中的 'images' 替换为 'labels'
        potential_labels = Path(str(train_src_abs).replace("/images", "/labels"))
        if potential_labels.exists():
            labels_src_abs = potential_labels
            print(f"🔍 自动推断标签目录 (images->labels): {labels_src_abs}")

    # 策略 2: 如果策略1失败，尝试 root/labels/train (常见结构)
    if not labels_src_abs:
        candidate = root / "labels" / "train"
        if candidate.exists():
            labels_src_abs = candidate
            print(f"🔍 找到标准标签目录: {labels_src_abs}")

    # 策略 3: 尝试 root/train/labels (旧式结构)
    if not labels_src_abs:
        candidate = root / "train" / "labels"
        if candidate.exists():
            labels_src_abs = candidate
            print(f"🔍 找到旧式标签目录: {labels_src_abs}")

    # 策略 4: 尝试 root/labels (扁平结构)
    if not labels_src_abs:
        candidate = root / "labels"
        if candidate.exists():
            # 检查里面是否有 txt 文件，如果没有，可能还是不对
            if list(candidate.glob("*.txt")):
                labels_src_abs = candidate
                print(f"🔍 找到扁平标签目录: {labels_src_abs}")

    # === 修改结束 ===
    
    if not labels_src_abs:
        print(f"⚠️ 警告：未找到标签目录 (尝试了多种路径)，按无标签模式处理。")
        img_classes = {}
        all_imgs = [f.stem for f in train_src_abs.glob("*.*") if f.suffix.lower() in ['.jpg','.jpeg','.png','.bmp','.webp']]
        for img in all_imgs: img_classes[img] = [-1]
    else:
        print(f"🔍 分析标签中...")
        img_classes = get_image_classes(labels_src_abs)

    # ... (后文代码不变) ...
    if not img_classes:
        print("❌ 错误：未找到任何图片。")
        return

    # 划分
    print(f"🔄 按 {args.ratio*100}% 比例划分...")
    unlabel_imgs, labeled_imgs = split_by_class(img_classes, args.ratio)
    print(f"✅ 划分结果：有标签 {len(labeled_imgs)} 张，无标签 {len(unlabel_imgs)} 张")

    # === 关键步骤：清理旧目录 ===
    labeled_dest_path = root / args.labeled_dir_name
    unlabel_dest_path = root / args.unlabeled_dir_name

    if not args.no_clean:
        # 除非用户指定 --no_clean，否则总是先删除旧目录
        if labeled_imgs:
            safe_remove_dir(labeled_dest_path)
        if unlabel_imgs:
            safe_remove_dir(unlabel_dest_path)
    else:
        print("⚠️ 检测到 --no_clean 标志，将跳过清理步骤（可能导致旧数据残留）。")

    # 复制有标签数据
    if labeled_imgs:
        count_img, count_lbl = copy_files_with_labels(
            train_src_abs, labels_src_abs, labeled_imgs, 
            root, args.labeled_dir_name
        )
        print(f"💾 有标签数据已存入：{labeled_dest_path} (图:{count_img}, 标:{count_lbl})")

    # 复制无标签数据
    if unlabel_imgs:
        dest_images = Path(root) / args.unlabeled_dir_name / "images"
        dest_images.mkdir(parents=True, exist_ok=True)
        count_u = 0
        for img_name in unlabel_imgs:
            src_img = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG']:
                potential_src = Path(train_src_abs) / f"{img_name}{ext}"
                if potential_src.exists():
                    src_img = potential_src
                    break
            if src_img:
                shutil.copy2(str(src_img), str(dest_images / src_img.name))
                count_u += 1
        print(f"💾 无标签数据已存入：{unlabel_dest_path} (图:{count_u})")

    # 生成 YAML
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

    print("\n🎉 完成！旧数据已清理，新数据已生成。")

if __name__ == "__main__":
    main()