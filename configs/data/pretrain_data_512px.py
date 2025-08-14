"""
Pretraining data configuration for 256px resolution models.

This module defines the data sources and their configurations for pretraining,
including NLP, image generation, image editing, and interleaved multimodal data.
"""

from nextstep.datasets.image_text_wds import ImageTextWDS
from nextstep.lazy_config.registry import _convert_target_to_string
from nextstep.utils.misc import LargeInt
from nextstep.datasets.image_editing_interleave import ImageEditingInterleave
from nextstep.datasets.video_interleave import VideoInterleave
from nextstep.datasets.nlp_itd import NLPITD


def scale_to(data_info_list: list, max_samples: int):
    """
    Scale the number of samples in each data info to match the target total.
    
    Args:
        data_info_list: List of data info dictionaries, each containing a "samples" key.
        max_samples: Target total number of samples after scaling.
    
    Returns:
        The same list with scaled sample counts, maintaining relative proportions.
    """
    num_samples = sum([data_info["samples"] for data_info in data_info_list])
    for data_info in data_info_list:
        data_info["samples"] = int(data_info["samples"] * max_samples / num_samples)
    return data_info_list

# NLP (text-only) data sources
nlp_data_info_list = [
    {
        "cls": _convert_target_to_string(NLPITD),
        "data_type": "nlp",
        "name": "text_only/allenai_dolma_v16/books",
        "caption_keys": ["text"],
        "samples": LargeInt("55K"),
    },
]

# Image generation (text-to-image) data sources
gen_data_info_list = [
    {
        "cls": _convert_target_to_string(ImageTextWDS),
        "data_type": "image_text_pair",
        "name": "text2image/BLIP3o-60k",
        "caption_keys": ["caption"],
        "caption_ratio": [1],
        "filter": {
            "area": [512*512, 1024*1024*2],  # Image area range in pixels
            "aspect_ratio": 6,  # Maximum aspect ratio (width/height or height/width)
        },
        "samples": LargeInt("58K"),
    },
]


# Image editing (image-to-image) data sources
editing_data_info_list = [
    {
        "cls": _convert_target_to_string(ImageEditingInterleave),
        "data_type": "image_editing",
        "name": "image2image/GPT-Image-Edit-1.5M/hqedit",
        "caption_keys": ["caption", "caption_all"],  # Multiple caption fields
        "caption_ratio": [1, 1],  # Ratio for each caption key
        "filter": {
            "area": [512*512, 1024*1024*2],  # Image area range in pixels
            "aspect_ratio": 6,  # Maximum aspect ratio
        },
        "samples": LargeInt("93K"),
    },
]

# Interleaved multimodal data sources (e.g., video with text)
interleave_data_info_list = [
    {
        "cls": _convert_target_to_string(VideoInterleave),
        "data_type": "interleave",
        "name": "interleave/multimodal_textbook",
        "caption_keys": ["caption","caption_i2t"],  # Image-to-text captions
        "caption_ratio": [4,1],  # Weight ratio for each caption type
        "filter": {
            "area": [512*512, 1024*1024*2],  # Image area range in pixels
            "aspect_ratio": 6,  # Maximum aspect ratio
        },
        "samples": LargeInt("32K"),
    },
]

# Validation data sources (used for evaluation, not training)
val_data_info_list = [
    {
        "cls": _convert_target_to_string(ImageTextWDS),
        "data_type": "image_text_pair",
        "name": "image_val_loss",
        "caption_keys": ["caption"],
        "caption_ratio": [1],
        "samples": 45717,
    },
]

# Combine and scale all training data sources to target sample counts
# Total: 20M (NLP) + 40M (Generation) + 10M (Editing) + 30M (Interleave) = 100M samples
data_info_list = scale_to(nlp_data_info_list, LargeInt("20M")) + scale_to(gen_data_info_list, LargeInt("40M")) + scale_to(editing_data_info_list, LargeInt("10M")) + scale_to(interleave_data_info_list, LargeInt("30M"))

if __name__ == "__main__":
    """
    Print statistics about the data configuration when run as a script.
    Shows total samples, distribution by dataset prefix, and per-dataset breakdown.
    """
    from tabulate import tabulate
    
    # Calculate total number of samples
    total_samples = 0
    for data_info in data_info_list:
        total_samples += data_info["samples"]
    
    # Build table with per-dataset statistics
    data_table = []
    for data_info in data_info_list:
        percentage = data_info["samples"] / total_samples * 100
        samples = data_info["samples"]
        # Format sample count with appropriate unit (M, K, or raw number)
        if samples >= 1_000_000:
            samples_str = f"{samples/1_000_000:.2f}M"
        elif samples >= 1_000:
            samples_str = f"{samples/1_000:.2f}K"
        else:
            samples_str = str(samples)
        data_table.append([data_info["name"], f"{percentage:.2f}%", samples_str])
    
    # Aggregate statistics by dataset prefix (first part of name before '/')
    prefix_dict = {}
    for data_info in data_info_list:
        prefix = data_info["name"].split("/")[0]
        if prefix not in prefix_dict:
            prefix_dict[prefix] = 0
        prefix_dict[prefix] += data_info["samples"]
    
    # Build table with prefix-level statistics
    prefix_table = []
    for prefix, samples in prefix_dict.items():
        prefix_percentage = samples / total_samples * 100
        prefix_table.append([prefix, f"{prefix_percentage:.2f}%"])
    
    # Print statistics
    print(f"Total samples: {LargeInt(total_samples)}")
    print()
    print(tabulate(prefix_table, headers=["Dataset", "Percentage", "Samples"], tablefmt="grid"))
    print()
    print(tabulate(data_table, headers=["Dataset Name", "Percentage", "Samples"], tablefmt="grid"))