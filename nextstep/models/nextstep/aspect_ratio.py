import numpy as np
import PIL.Image

# From SDXL
# HW_ASPECT_RATIOS = [
#     (8, 32),  # 256
#     (8, 31),  # 248
#     (8, 30),  # 240
#     (8, 29),  # 232
#     (9, 28),  # 252
#     (9, 27),  # 243
#     (9, 26),  # 240
#     (10, 25),  # 250
#     (10, 24),  # 240
#     (11, 23),  # 253
#     (11, 22),  # 242
#     (11, 21),  # 231
#     (12, 21),  # 252
#     (12, 20),  # 240
#     (13, 19),  # 247
#     (13, 18),  # 246
#     (14, 18),  # 252
#     (14, 17),  # 238
#     (15, 17),  # 255
#     (15, 16),  # 240
#     (16, 16),  # 256
#     (16, 15),  # 240
#     (17, 15),  # 255
#     (17, 14),  # 238
#     (18, 14),  # 252
#     (18, 13),  # 246
#     (19, 13),  # 247
#     (20, 12),  # 240
#     (21, 12),  # 252
#     (22, 11),  # 242
#     (23, 11),  # 253
#     (24, 10),  # 240
#     (25, 10),  # 250
#     (26, 9),  # 234
#     (27, 9),  # 243
#     (28, 9),  # 252
#     (29, 8),  # 232
#     (30, 8),  # 240
#     (31, 8),  # 255
#     (32, 8),  # 256
# ]

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


def get_ar_base(ars: list[tuple[int, int]] = HW_ASPECT_RATIOS):
    sqrt_products = [round(np.sqrt(h * w)) for h, w in ars]
    return round(np.mean(sqrt_products))


def ar2str(h: int, w: int) -> str:
    return f"{h}*{w}"


def str2ar(s: str) -> tuple[int, int]:
    return tuple(map(int, s.split("*")))

def center_crop_arr(pil_image, image_size, crop=True):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    if crop:
        pil_image = resize_image(pil_image, image_size)
        arr = np.array(pil_image)
        crop_y = (arr.shape[0] - image_size) // 2
        crop_x = (arr.shape[1] - image_size) // 2
        return PIL.Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])
    else:
        # Pad image to square
        width, height = pil_image.size
        if width != height:
            # Create a square canvas with size equal to the larger dimension
            max_dim = max(width, height)
            padded_img = PIL.Image.new(pil_image.mode, (max_dim, max_dim), (0, 0, 0))
            # Paste original image centered on square canvas
            padded_img.paste(pil_image, ((max_dim - width) // 2, (max_dim - height) // 2))
            pil_image = padded_img
        pil_image = resize_image(pil_image, image_size)
        return pil_image
    

def center_crop_arr_with_ar(pil_image, image_size: int, ars: list[tuple[int, int]] = HW_ASPECT_RATIOS, crop=True):
    """
    Center crop the image to match the closest aspect ratio from the provided list.

    Args:
        pil_image: PIL Image to be cropped
        buckets_for_super_multi_aspect: Target size for the smaller dimension
        ars: List of aspect ratios as (height, width) tuples

    Returns:
        PIL Image cropped to the closest aspect ratio
    """

    ar_base = get_ar_base(ars)
    assert image_size % ar_base == 0, f"image_size must be divisible by {ar_base}"

    # Get current image dimensions
    width, height = pil_image.size
        
    current_ar = height / width

    # Find the closest aspect ratio
    closest_ar_idx = np.argmin([abs(current_ar - (h / w)) for h, w in ars])
    target_h, target_w = ars[closest_ar_idx]

    if crop:
        target_h, target_w = round(image_size / ar_base * target_h), round(image_size / ar_base * target_w)

        # First, resize the image while maintaining aspect ratio to ensure the smaller dimension is at least the target size
        scale = max(target_h / height, target_w / width)
        new_height = round(height * scale)
        new_width = round(width * scale)
        pil_image = pil_image.resize((new_width, new_height), resample=PIL.Image.LANCZOS)

        arr = np.array(pil_image)
        # Then perform center crop to the target dimensions
        crop_y = (new_height - target_h) // 2
        crop_x = (new_width - target_w) // 2

        return PIL.Image.fromarray(arr[crop_y : crop_y + target_h, crop_x : crop_x + target_w])
    else:
        scale = image_size // ar_base
        return pil_image.resize((round(target_w * scale), round(target_h * scale)), resample=PIL.Image.LANCZOS)


def center_crop_arr_with_buckets(pil_image, ars: list[tuple[int, int]] = HW_ASPECT_RATIOS, crop=True, buckets: list[int] = [256, 512, 768, 1024]):
    """
    Center crop the image to match the closest aspect ratio from the provided list.

    Args:
        pil_image: PIL Image to be cropped
        image_size: Target size for the smaller dimension
        ars: List of aspect ratios as (height, width) tuples

    Returns:
        PIL Image cropped to the closest aspect ratio
    """
    # ar_base = get_ar_base(ars)
    # Get current image dimensions
    width, height = pil_image.size
    
    buckets = sorted(buckets, reverse=True)
    image_size = buckets[-1]

    for bucket in buckets:
        if width * height >= bucket * bucket:
            image_size = bucket
            break

    return center_crop_arr_with_ar(pil_image, image_size, ars, crop)
