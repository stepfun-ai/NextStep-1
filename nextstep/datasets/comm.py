import random
from nextstep.datasets.utils import make_seed

FILTERED_PATTERNS = [
    "The image depicts ",
    "The image shows ",
    "The image features ",
    "The image is ",
    "The image displays ",
    "The image captures ",
    "The image consists of ",
    "In the image, ",
    "In this image, ",
    "这幅图像展示了",
    "这幅图片展示了",
    "这幅图像描绘了",
    "这张图片展示了",
    "这张图像展示了",
    "这张图片显示了",
    "这幅图像展示",
    "这幅图片展示",
    "这幅图像描绘",
    "这张图片展示",
    "这张图像展示",
    "这张图片显示",
    "图片中有",
    "这张图片是",
    "图片展示了",
    "图片中展示了",
    "图像展示了",
    "图像中展示了",
    "图片描绘了",
    "图片中描绘了",
    "图像描绘了",
    "图像中描绘了",
    "这是一张",
    "这张图片捕捉了",
    "这幅图像捕捉了",
    "这张图片包含",
    "在这张图片中，",
    "在图片中，",
    "图片中，",
]


def filter_hw(info, h, w):
    if info["height"] < h or info["width"] < w:
        return True
    return False


def filter_area(info, area):
    if isinstance(area, int) or isinstance(area, float):
        if info["height"] * info["width"] < area:
            return True
    elif isinstance(area, tuple) or isinstance(area, list):
        min_area, max_area = area
        if info["height"] * info["width"] < min_area or info["height"] * info["width"] > max_area:
            return True
    else:
        raise ValueError(f"Invalid area: {area}, should be int or tuple/list")
    return False


def filter_aspect_ratio(info, aspect_ratio):
    if info["aspect_ratio"] > aspect_ratio:
        return True
    return False


def filter_keywords(info, kw_and_drop_prob_list: list[tuple[str, float]], seed: int):
    _caption = info["caption"].lower()
    for kw, drop_prob in kw_and_drop_prob_list:
        rng = random.Random(make_seed(seed, _caption))
        if kw.lower() in _caption and rng.random() < drop_prob:
            return True
    return False


def filter_other(info, filter_dict):
    for k, v in filter_dict.items():
        if k not in info:
            continue
        if v > 0:
            if info[k] < v:
                return True
        else:
            if info[k] > -v:
                return True
    return False