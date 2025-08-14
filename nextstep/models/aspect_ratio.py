import numpy as np

ANY_ASPECT_RATIO = (0, 0)

HW_ASPECT_RATIOS = [
    (8, 32),  # 256
    (9, 28),  # 252
    (10, 25),  # 250
    (11, 23),  # 253
    (12, 21),  # 252
    (13, 19),  # 247
    (14, 18),  # 252
    (15, 17),  # 255
    (16, 16),  # 256
    (17, 15),  # 255
    (18, 14),  # 252
    (19, 13),  # 247
    (21, 12),  # 252
    (23, 11),  # 253
    (25, 10),  # 250
    (28, 9),  # 252
    (32, 8),  # 256
]


def get_hw_base(hws: list[tuple[int, int]] = HW_ASPECT_RATIOS):
    sqrt_products = [round(np.sqrt(h * w)) for h, w in hws]
    return round(np.mean(sqrt_products))


def hw2str(h: int, w: int) -> str:
    return f"{h}*{w}"


def str2hw(s: str) -> tuple[int, int]:
    return tuple(map(int, s.split("*")))
