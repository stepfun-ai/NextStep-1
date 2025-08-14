"""
Build WebDataset format from JSON data with images.

This script processes JSON data containing image paths and converts them into
WebDataset (tar) format. Each dataset may have different input formats, but
the output format is standardized with caption fields describing image-text
positions and images stored as bytes.
"""

import os
import io
import json
import uuid
from PIL import Image
import numpy as np
from tqdm import tqdm
import webdataset as wds
from multiprocessing import Pool, cpu_count
from functools import partial




def pil_to_bytes(pil_img):
    """
    Convert PIL Image or numpy array to bytes.
    
    Args:
        pil_img: PIL Image object or numpy array
        
    Returns:
        bytes: Image data as bytes
    """
    if isinstance(pil_img, np.ndarray):
        pil_img = Image.fromarray(pil_img.astype(np.uint8))
    
    # Use original format if available, otherwise default to JPEG
    img_format = pil_img.format or 'JPEG'
    
    img_byte_arr = io.BytesIO()
    pil_img.save(img_byte_arr, format=img_format)
    return img_byte_arr.getvalue()


def load_data(file_path):
    """
    Load JSON data from file and normalize to list format.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        list: List of data dictionaries
    """
    with open(file_path) as f:
        data = json.load(f)
    
    # Normalize to list: wrap single dict in list, keep list as is
    if isinstance(data, dict):
        return [data]
    elif isinstance(data, list):
        return data
    else:
        return []


def create_example(data_info):
    """
    Create a standardized example from input data info.
    
    Args:
        data_info: Dictionary containing data fields:
            - input_image: Path to input image
            - output_image: Path to output image
            - instruction: Instruction text
            - input_description: Input description text
            
    Returns:
        dict: Standardized example with caption, caption_all, metadata, and images
    """
    input_image = data_info["input_image"]
    output_image = data_info["output_image"]
    instruction = data_info["instruction"]
    input_description = data_info.get("input_description", "")
    
    example = {}
    
    # Required: caption field describes image content and text positions
    # For single image tasks, <image_x> placeholders are not needed
    # For image editing or interleaved image-text tasks, <image_x> placeholders
    # indicate where images should be inserted in the text
    example["caption"] = f"<image_0>{instruction}<image_1>"
    
    # Optional: caption_all contains alternative captions for training
    # This field is a dictionary that can store multiple captions (e.g., short caption, long caption, LLM-rewritten captions, different tasks and so on). During training, one caption is randomly
    # selected based on given weights
    example["caption_all"] = {
        "subject_driven": f"{input_description}<image_0>{instruction}<image_1>"
    }
    
    # Optional: store original metadata
    example["metadata"] = data_info
    
    # Required: images field stores image bytes (not needed for pure text tasks), the order of images is important, the first image is the <image_0> placeholder, the second image is the <image_1> placeholder, the other images are for future use.
    example["images"] = [
        pil_to_bytes(Image.open(input_image)),
        pil_to_bytes(Image.open(output_image))
    ]
    
    return example


def _write_batch_to_webdataset(args):
    """
    Write a batch of examples to WebDataset format (tar files).
    This function is designed to be called by multiprocessing workers.
    
    Args:
        args: Tuple containing:
            - batch_examples: List of example dictionaries for this batch
            - output_dir: Output directory for tar files
            - max_samples_per_tar: Maximum number of samples per tar file
            - batch_idx: Index of this batch (for unique tar file naming)
            - unused: Placeholder for compatibility (not used)
    
    Returns:
        tuple: (number of samples written, list of tar files created)
    """
    batch_examples, output_dir, max_samples_per_tar, batch_idx, _ = args
    
    os.makedirs(output_dir, exist_ok=True)
    
    current_sink = None
    sample_count = 0
    tar_file_idx = 0
    tar_files_created = []
    
    for example in batch_examples:
        # Create new tar file if needed
        if current_sink is None:
            # Use batch_idx and tar_file_idx to ensure unique tar file names across processes
            # Format: idx_batch_tar.tar (e.g., idx_0000_0000.tar)
            tar_path = os.path.join(output_dir, f"idx_{batch_idx:04d}_{tar_file_idx:04d}.tar")
            current_sink = wds.TarWriter(tar_path)
            tar_files_created.append(tar_path)
        
        # Generate unique key for this sample
        key_str = str(uuid.uuid4().hex)
        
        # Extract images and write JSON metadata
        # Make a copy to avoid modifying the original
        example_copy = example.copy()
        images = example_copy.pop("images")
        current_sink.write({
            "__key__": key_str,
            "json": example_copy
        })
        
        # Write each image with a unique key
        for img_idx, image in enumerate(images):
            current_sink.write({
                "__key__": f"{key_str}-{img_idx}",
                "jpg": image
            })
        
        sample_count += 1
        
        # Close current tar and start new one if max samples reached
        if sample_count % max_samples_per_tar == 0:
            current_sink.close()
            current_sink = None
            tar_file_idx += 1
    
    # Close final tar file
    if current_sink is not None:
        current_sink.close()
    
    return len(batch_examples), tar_files_created


def write_to_webdataset(examples, output_dir, max_samples_per_tar, num_workers=None):
    """
    Write examples to WebDataset format (tar files) using multiprocessing.
    
    Args:
        examples: List of example dictionaries
        output_dir: Output directory for tar files
        max_samples_per_tar: Maximum number of samples per tar file
        num_workers: Number of parallel workers. If None, uses cpu_count()
    """
    if num_workers is None:
        num_workers = max(1, cpu_count())
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Split examples into batches for parallel processing
    # Each batch will be processed by one worker
    batch_size = max(1, len(examples) // num_workers)
    batches = []
    batch_idx = 0
    for i in range(0, len(examples), batch_size):
        batch = examples[i:i + batch_size]
        batches.append((batch, output_dir, max_samples_per_tar, batch_idx, 0))
        batch_idx += 1
    
    # Process batches in parallel
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(_write_batch_to_webdataset, batches),
            desc="Building WDS",
            total=len(batches)
        ))
    
    # Print summary
    total_samples = sum(r[0] for r in results)
    all_tar_files = []
    for r in results:
        all_tar_files.extend(r[1])
    
    print(f"Created {len(all_tar_files)} tar files with {total_samples} total samples in output directory {output_dir}")


if __name__ == "__main__":
    # Configuration
    EXAMPLE_PATH = "assets/example.jsonl"
    OUTPUT_DIR = "assets"
    MAX_SAMPLES_PER_TAR = 100


    # Load and normalize input data
    # Process input JSON and image paths, convert image paths to image bytes
    # Different datasets may have different original formats, but the output format
    # is standardized: caption field describes image-text positions, images field
    # stores image bytes
    data_list = load_data(EXAMPLE_PATH)
    
    # Process each data entry into standardized examples
    examples = []
    for data_info in data_list:
        example = create_example(data_info)
        examples.append(example)
    
    # Write preprocessed data to WebDataset format
    write_to_webdataset(examples, OUTPUT_DIR, MAX_SAMPLES_PER_TAR)
