import os
import argparse
import shutil
import random
from collections import defaultdict
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="按类别均匀划分未标注数据，生成半监督学习YAML")
    parser.add_argument('--source_root', type=str, default='.', help='数据集根目录 (包含 train/images 和 train/labels)')
    parser.add_argument('--ratio', type=float, default=0.2, help='划分为 unlabel 的比例 (0.0 - 1.0)')
    parser.add_argument('--output_dir', type=str, default='unlabel_image', help='未标注图片存放目录名 (相对于 source_root)')
    parser.add_argument('--yaml_name', type=str, default='data_unlabel.yaml', help='输出的yaml文件名')
    return parser.parse_args()

def get_image_classes(labels_dir):
    """读取标签，建立 {图片名: [类别ID列表]} 映射"""
    img_classes = defaultdict(list)
    if not os.path.exists(labels_dir):
        print(f"错误：标签目录 {labels_dir} 不存在")
        return img_classes

    for label_file in Path(labels_dir).glob("*.txt"):
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
            img_classes[img_name] = [-1] # 无目标图片
            
    return img_classes

def split_by_class(img_classes, ratio):
    """按类别均匀抽取图片作为未标注集"""
    class_to_imgs = defaultdict(list)
    
    # 分组：以第一个检测到的类别为主类别
    for img, classes in img_classes.items():
        if not classes:
            continue
        main_class = classes[0]
        class_to_imgs[main_class].append(img)

    selected_unlabel_imgs = set()
    
    for cls, imgs in class_to_imgs.items():
        if cls == -1:
            count = int(len(imgs) * ratio)
        else:
            # 确保每个有目标的类别至少有一张被选入未标注集（如果比例允许）
            count = max(1, int(len(imgs) * ratio)) if len(imgs) > 0 and ratio > 0 else 0
        
        if count >= len(imgs):
            selected = imgs
        else:
            random.shuffle(imgs)
            selected = imgs[:count]
            
        selected_unlabel_imgs.update(selected)

    return list(selected_unlabel_imgs)

def move_files(source_images_dir, img_list, dest_dir):
    """将图片复制到目标目录 dest_dir/images"""
    dest_path = Path(dest_dir) / "images"
    dest_path.mkdir(parents=True, exist_ok=True)
    
    moved_count = 0
    for img_name in img_list:
        src_file = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG']:
            potential_src = Path(source_images_dir) / f"{img_name}{ext}"
            if potential_src.exists():
                src_file = potential_src
                break
        
        if src_file:
            # 使用 shutil.copy2 保留元数据，路径转换由 Path 对象自动处理
            shutil.copy2(str(src_file), str(dest_path / src_file.name))
            moved_count += 1
    return moved_count

def generate_yaml(script_dir, root_path, unlabel_subdir, yaml_name):
    """
    生成特定格式的 YAML，并保存到脚本所在目录 (script_dir)
    【关键修复】：确保所有路径使用 Linux 风格的正斜杠 '/'
    """
    # 1. 确保 root_path 是字符串且使用正斜杠
    # pathlib 在 Linux 下默认就是 '/'，但为了保险起见（如果传入的是 Windows 风格字符串），我们统一替换
    r_path = str(root_path).replace("\\", "/")
    
    # 2. 确保子路径使用正斜杠
    u_sub = str(unlabel_subdir).replace("\\", "/")
    
    # 注意：train 字段应该是相对于 path 的路径。
    # 如果 unlabel_subdir 已经包含了 '/images' (如 main 中构造的)，则直接使用
    # 如果 unlabel_subdir 只是目录名，这里可能需要拼接。根据 main 函数逻辑，传入的是 f"{args.output_dir}/images"
    
    # YAML 内容：train指向未标注目录，val留空
    content = f"""path: {r_path}
train: {u_sub}
val: ""
is_unlabel: True
"""
    
    # yaml_path 基于 script_dir (脚本所在目录)
    yaml_path = Path(script_dir) / yaml_name
    
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"YAML 文件已生成 (脚本目录下): {yaml_path}")
    print("--- YAML 内容预览 ---")
    print(content)
    print("-------------------")

def main():
    args = parse_args()
    
    # 1. 获取脚本所在的绝对目录
    script_dir = Path(__file__).resolve().parent
    
    # 2. 解析数据根目录
    root = Path(args.source_root).resolve()
    
    images_dir = root / "train" / "images"
    labels_dir = root / "train" / "labels"
    output_unlabel_dir = root / args.output_dir
    
    if not images_dir.exists():
        print(f"错误：找不到图片目录 {images_dir}")
        print(f"请检查 --source_root 参数。当前解析路径：{root}")
        return

    print(f"数据集根目录：{root}")
    print(f"脚本所在目录：{script_dir}")
    print(f"正在分析标签文件以进行类别均衡划分...")
    
    img_classes = get_image_classes(labels_dir)
    
    if not img_classes:
        print("错误：未找到任何有效的标签文件 (.txt)。")
        return

    print(f"发现 {len(img_classes)} 张带标签的图片。")
    print(f"正在按类别均匀抽取 {args.ratio*100}% 作为未标注数据...")
    
    # 获取未标注图片列表
    unlabel_imgs = split_by_class(img_classes, args.ratio)
    
    print(f"划分完成：共选出 {len(unlabel_imgs)} 张图片作为未标注数据。")

    if unlabel_imgs:
        # 移动/复制图片到 unlabel_image/images (位于数据根目录下)
        count = move_files(images_dir, unlabel_imgs, output_unlabel_dir)
        print(f"操作成功：已复制 {count} 张图片到 {output_unlabel_dir}/images")
        
        # 【关键修复】构建相对路径时，强制使用正斜杠 '/'
        # 原来代码：f"{args.output_dir}\\images" -> 错误，Linux 不识别
        # 修复代码：f"{args.output_dir}/images"
        train_rel = f"{args.output_dir}/images"
        
        # 生成 YAML
        generate_yaml(
            script_dir=script_dir,
            root_path=str(root),
            unlabel_subdir=train_rel,
            yaml_name=args.yaml_name
        )
    else:
        print("警告：没有图片被选中。未生成YAML。")

if __name__ == "__main__":
    random.seed(42) # 保证结果可复现
    main()