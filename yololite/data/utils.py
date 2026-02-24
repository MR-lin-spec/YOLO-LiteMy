# YOLO-Lite 🚀

import hashlib
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from yololite.utils import (
    DATASETS_DIR,
    LOGGER,
    emojis,
    is_dir_writeable,
    yaml_load,
)

HELP_URL = "See https://docs.ultralytics.com/datasets for dataset formatting guidance."
IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm", "heic"}  # image suffixes
VID_FORMATS = {"asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv", "webm"}  # video suffixes
PIN_MEMORY = str(os.getenv("PIN_MEMORY", True)).lower() == "true"  # global pin_memory for dataloaders
FORMATS_HELP_MSG = f"Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}"


def img2label_paths(img_paths):
    """Define label paths as a function of image paths."""
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in img_paths]


def get_hash(paths):
    """Returns a single hash value of a list of paths (files or dirs)."""
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.sha256(str(size).encode())  # hash sizes
    h.update("".join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def exif_size(img: Image.Image):
    """Returns exif-corrected PIL size."""
    s = img.size  # (width, height)
    if img.format == "JPEG":  # only support JPEG images
        try:
            exif = img.getexif()
            if exif:
                rotation = exif.get(274, None)  # the EXIF key for the orientation tag is 274
                if rotation in {6, 8}:  # rotation 270 or 90
                    s = s[1], s[0]
        except Exception:
            pass
    return s

def verify_image_label(args):
    im_file, lb_file, ndim = args
    nm, nf, ne, nc, msg, = 0, 0, 0, 0, ""
    try:
        # Verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        shape = (shape[1], shape[0])  # hw
        assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
        assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}. {FORMATS_HELP_MSG}"
        if im.format.lower() in {"jpg", "jpeg"}:
            with open(im_file, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                    msg = f"WARNING ⚠️ {im_file}: corrupt JPEG restored and saved"

        # Verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                lb = np.array(lb, dtype=np.float32)
            nl = len(lb)
            if nl:
                assert lb.shape[1] == 5, f"labels require 5 columns, {lb.shape[1]} columns detected"
                assert lb.min() >= 0, f"negative label values {lb[lb < 0]}"

                # All labels
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    msg = f"WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed"
            else:
                ne = 1  # label empty
                lb = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, 5), dtype=np.float32)
        lb = lb[:, :5]
        return im_file, lb, shape,  nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f"WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}"
        return [None, None, None, None, None, nm, nf, ne, nc, msg]


def find_dataset_yaml(path: Path) -> Path:
    """
    Find and return the YAML file associated with a Detect dataset.

    This function searches for a YAML file at the root level of the provided directory first, and if not found, it
    performs a recursive search. It prefers YAML files that have the same stem as the provided path. An AssertionError
    is raised if no YAML file is found or if multiple YAML files are found.

    Args:
        path (Path): The directory path to search for the YAML file.

    Returns:
        (Path): The path of the found YAML file.
    """
    files = list(path.glob("*.yaml")) or list(path.rglob("*.yaml"))  # try root level first and then recursive
    assert files, f"No YAML file found in '{path.resolve()}'"
    if len(files) > 1:
        files = [f for f in files if f.stem == path.stem]  # prefer *.yaml files that match
    assert len(files) == 1, f"Expected 1 YAML file in '{path.resolve()}', but found {len(files)}.\n{files}"
    return files[0]


def check_det_dataset(dataset):
    """
    下载、验证和/或解压数据集（如果在本地未找到）。

    此函数检查指定数据集的可用性，如果未找到，则可以选择下载并解压数据集。然后读取并解析附带的 YAML 数据，确保满足关键要求，同时解析与数据集相关的路径。

    参数：
        dataset (str): 数据集的路径或数据集描述符（如 YAML 文件）。
        autodownload (bool, optional): 如果未找到数据集，是否自动下载。默认为 True。

    返回：
        (dict): 解析的数据集信息和路径。
    """
    file = dataset
    extract_dir = ""

    # 读取 YAML 文件
    data = yaml_load(file, append_filename=True)  # 加载 YAML 文件，返回字典

    # 检查关键字段
    for k in "train", "val":  # 对于训练和验证
        if k not in data:  # 检查是否包含必需的关键字
            if k != "val" or "validation" not in data:  # 验证只有在数据缺失时才抛出异常
                raise SyntaxError(
                    emojis(f"{dataset} '{k}:' 键缺失 ❌.\n'train' 和 'val' 在所有数据 YAML 中都是必需的。")
                )
            LOGGER.info("警告 ⚠️ 将数据 YAML 中的 'validation' 键重命名为 'val' 以匹配 YOLO 格式。")
            data["val"] = data.pop("validation")  # 将 'validation' 键替换为 'val' 键
    if "names" not in data and not data.get("is_unlabel", False):  # 检查名称,如果为无标签数据集则跳过
        data["names"] = [f"class_{i}" for i in range(data["nc"])]  # 如果没有，生成默认名称
    elif not data.get("is_unlabel", False):
        data["nc"] = len(data["names"])  # 更新类别数量

    # 解析路径
    path = Path(extract_dir or data.get("path") or Path(data.get("yaml_file", "")).parent)  # 数据集根目录
    if not path.is_absolute():  # 如果路径不是绝对路径
        path = (DATASETS_DIR / path).resolve()  # 解析为绝对路径

    # 设置路径
    data["path"] = path  # 设置数据集路径
    for k in "train", "val", "test", "minival":  # 处理训练、验证、测试等路径
        if data.get(k):  # 如果有路径
            if isinstance(data[k], str):
                x = (path / data[k]).resolve()  # 解析为绝对路径
                if not x.exists() and data[k].startswith("../"):  # 如果路径不存在且以 ../ 开头
                    x = (path / data[k][3:]).resolve()  # 尝试去掉前缀解析
                data[k] = str(x)  # 更新路径为字符串
            else:
                data[k] = [str((path / x).resolve()) for x in data[k]]  # 更新为绝对路径列表

    # 解析 YAML
    val = data.get("val")  # 获取验证集路径

    if val:
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # 解析验证路径
    return data  # 返回包含数据集信息的字典


def load_dataset_cache_file(path):
    """Load an Ultralytics *.cache dictionary from path."""
    import gc

    gc.disable()  # reduce pickle load time https://github.com/ultralytics/ultralytics/pull/1585
    cache = np.load(str(path), allow_pickle=True).item()  # load dict
    gc.enable()
    return cache


def save_dataset_cache_file(prefix, path, x):
    """Save an Ultralytics dataset *.cache dictionary x to path."""
    if is_dir_writeable(path.parent):
        if path.exists():
            path.unlink()  # remove *.cache file if exists
        np.save(str(path), x)  # save cache for next time
        path.with_suffix(".cache.npy").rename(path)  # remove .npy suffix
        LOGGER.info(f"{prefix}New cache created: {path}")
    else:
        LOGGER.warning(f"{prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable, cache not saved.")
