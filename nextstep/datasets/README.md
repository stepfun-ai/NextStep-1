# NextStep Dataset System Guide

Welcome to the NextStep dataset system! This guide will help you understand how the system unifies different sample types into training-ready batches.

---

## 📖 Table of Contents

1. [Introduction](#introduction)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
4. [Dataset Types](#dataset-types)
5. [Batch Output Format](#batch-output-format)
6. [Checkpoint Resumption](#checkpoint-resumption)
7. [Usage Examples](#usage-examples)
8. [Troubleshooting](#troubleshooting)
9. [Related Documentation](#related-documentation)

---

## Introduction

The `nextstep/datasets/` directory is responsible for **unifying different sample types into training-ready batches**. This layer handles:

- ✅ **Multi-source sampling**: Mix multiple data sources according to specified sampling ratios
- ✅ **Batch construction**: Organize samples into batches with proper padding and alignment
- ✅ **Aspect ratio grouping**: Group images by aspect ratio for efficient batch processing
- ✅ **State management**: Track and restore data sampling state for checkpoint resumption
- ✅ **Timeout handling**: Monitor data loading and alert on timeouts

---

## System Architecture

### Data Flow

```
Multiple Data Sources (ImageTextWDS, VideoInterleave, NLPITD, etc.)
    ↓
MixedDataset (weighted sampling)
    ↓
Buffers (NLPBuffer, ImageBuffer)
    ↓
BatchBucket (aspect ratio grouping)
    ↓
MixedDataloader (state tracking, timeout monitoring)
    ↓
Training-ready Batches
```

### Component Hierarchy

```
MixedDataloader
    ├── MixedDataset
    │   ├── Multiple IndexedTarDataset instances
    │   │   ├── ImageTextWDS
    │   │   ├── VideoInterleave
    │   │   ├── ImageEditingInterleave
    │   │   └── NLPITD
    │   ├── NLPBuffer (for text-only datasets)
    │   ├── ImageBuffer (for image datasets)
    │   └── BatchBucket (batch organization)
    └── MixingStatus (state management)
```

---

## Core Components

### `mixed_dataset.py` - Core Mixing Logic

This file contains the core components for mixing multiple data sources and constructing batches.

#### `MixedDataset` - Multi-Source Dataset Mixer

**Purpose**: Combines multiple data sources with weighted sampling and constructs unified batches.

**Key Features**:

| Feature | Description |
|---------|-------------|
| **Weighted sampling** | Samples from multiple datasets according to specified ratios |
| **Buffer management** | Uses `NLPBuffer` for text data and `ImageBuffer` for image data |
| **Batch construction** | Organizes samples into batches via `BatchBucket` |
| **Aspect ratio grouping** | Groups images by aspect ratio for efficient processing |
| **State tracking** | Maintains state for checkpoint resumption |

**Key Parameters**:

| Parameter | Description |
|-----------|-------------|
| `data_info_list` | List of dataset configurations with sampling ratios |
| `batch_size` | Target batch size |
| `tokenizer` | Tokenizer for text processing |
| `hw_aspect_ratios_ids` | Mapping of aspect ratio strings to token IDs |
| `max_len` | Maximum sequence length |
| `drop_text_prob` | Probability of dropping text tokens when sequence is too long |

**Sampling Mechanism**:

- Sampling ratios are normalized from `samples` field in `data_info_list`
- Each dataset is sampled proportionally to its ratio
- Samples are buffered and then organized into batches

#### `MixedDataloader` - State-Aware DataLoader

**Purpose**: Wraps `MixedDataset` with state tracking, timeout monitoring, and checkpoint resumption support.

**Key Features**:

| Feature | Description |
|---------|-------------|
| **State tracking** | Tracks `mixing_status`, `dataset_status`, and `hw_aspect_ratio_status` |
| **Timeout monitoring** | Monitors data loading and alerts on timeouts |
| **Checkpoint resumption** | Saves and restores data sampling state |
| **Memory monitoring** | Optional memory usage monitoring per worker |
| **Data monitoring** | Periodic data statistics reporting |

**Persisted State**:

| State | Description |
|------|-------------|
| `mixing_status` | Mixing ratios and dataset states per worker |
| `dataset_status` | Progress tracking for each dataset |
| `hw_aspect_ratio_status` | Statistics for different aspect ratios |

**Usage**:

```python
from nextstep.datasets.mixed_dataset import MixedDataset, MixedDataloader

# Create mixed dataset
mixed_dataset = MixedDataset(
    data_info_list=data_info_list,
    batch_size=4,
    tokenizer=tokenizer,
    # ... other parameters
)

# Create dataloader
dataloader = MixedDataloader(
    dataset=mixed_dataset,
    num_workers=8,
    timeout=30.0,
    data_monitor_interval=30.0,
)
```

#### `NLPBuffer` - Text Data Buffer

**Purpose**: Buffers text-only (NLP) samples before batch construction.

**Key Features**:

- Stores `input_ids` for text samples
- Provides `enqueue()` to add samples and `dequeue()` to retrieve batches
- Maintains state for checkpoint resumption

**Usage**: Automatically used by `MixedDataset` for datasets with `data_type="nlp"`.

#### `ImageBuffer` - Image Data Buffer

**Purpose**: Buffers image samples with aspect ratio handling before batch construction.

**Key Features**:

- Stores `pixel_values` and `input_ids` for image samples
- Groups images by aspect ratio
- Handles image placeholder tokens (`<image_0>`, `<image_1>`, etc.)
- Maintains state for checkpoint resumption

**Usage**: Automatically used by `MixedDataset` for datasets with image data types.

#### `BatchBucket` - Batch Organization Container

**Purpose**: Organizes samples into batches grouped by aspect ratio.

**Key Features**:

| Feature | Description |
|---------|-------------|
| **Aspect ratio grouping** | Groups samples by image aspect ratio for efficient batching |
| **Batch accumulation** | Accumulates samples until batch size is reached |
| **Safety threshold** | Prevents data accumulation beyond `batch_size * 2` |

**Aspect Ratio Handling**:

- Samples with same aspect ratio are grouped together
- Mixed aspect ratios are placed in `ANY_ASPECT_RATIO` bucket
- NLP samples can be added to any bucket with available space

---

## Dataset Types

### `ImageTextWDS` - Image-Text Pair Dataset

**Purpose**: Handles single image with caption pairs (text-to-image generation data).

**Data Format**:

- Single image file (e.g., `sample1.jpg`)
- Caption text (e.g., `sample1.txt` or in `sample1.json`)

**Key Features**:

| Feature | Description |
|---------|-------------|
| **Caption handling** | Supports multiple caption keys with sampling ratios |
| **Image filtering** | Filters by area, aspect ratio, height, width, keywords, etc. |
| **Post-processing** | Optional image post-processing (cropping, resizing, etc.) |

**Configuration Example**:

```python
{
    "cls": "nextstep.datasets.image_text_wds.ImageTextWDS",
    "data_type": "image_text_pair",
    "name": "text2image/BLIP3o-60k",
    "caption_keys": ["caption"],
    "caption_ratio": [1],
    "filter": {
        "area": [256*256, 1024*1024],
        "aspect_ratio": 6,
    },
    "samples": LargeInt("58K"),
}
```

### `VideoInterleave` - Interleaved Multimodal Dataset

**Purpose**: Handles multiple images interleaved with text (video understanding, story data).

**Data Format**:

- Multiple image files (e.g., `sample1-0.jpg`, `sample1-1.jpg`, `sample1-2.jpg`)
- Caption text with image placeholders (e.g., `"<image_0>Text<image_1>More text<image_2>"`)

**Key Features**:

| Feature | Description |
|---------|-------------|
| **Multi-image support** | Handles sequences of images with interleaved text |
| **Placeholder handling** | Processes `<image_n>` placeholders in captions |
| **Story caption support** | Special handling for story-style captions (title, summary, content) |

**Configuration Example**:

```python
{
    "cls": "nextstep.datasets.video_interleave.VideoInterleave",
    "data_type": "interleave",
    "name": "video/StoryDataset",
    "caption_keys": ["caption"],
    "caption_ratio": [1],
    "samples": LargeInt("100K"),
}
```

### `ImageEditingInterleave` - Image Editing Dataset

**Purpose**: Handles image editing tasks with input image, instruction, and output image.

**Data Format**:

- Input image (e.g., `sample1-0.jpg`)
- Instruction text (e.g., in `sample1.json`)
- Output image (e.g., `sample1-1.jpg`)

**Key Features**:

| Feature | Description |
|---------|-------------|
| **Input-output pairs** | Handles image editing with before/after images |
| **Instruction processing** | Processes editing instructions in captions |
| **Image filtering** | Similar filtering capabilities as `ImageTextWDS` |

**Configuration Example**:

```python
{
    "cls": "nextstep.datasets.image_editing_interleave.ImageEditingInterleave",
    "data_type": "image_editing",
    "name": "editing/InstructPix2Pix",
    "caption_keys": ["caption"],
    "caption_ratio": [1],
    "samples": LargeInt("50K"),
}
```

### `NLPITD` - Text-Only Dataset

**Purpose**: Handles pure text data for language model pretraining.

**Data Format**:

- Text file (e.g., `sample1.txt`) or text in JSON (e.g., `sample1.json`)

**Key Features**:

| Feature | Description |
|---------|-------------|
| **Text-only processing** | No image handling, pure text tokenization |
| **Single caption key** | Only supports one caption key (unlike image datasets) |

**Configuration Example**:

```python
{
    "cls": "nextstep.datasets.nlp_itd.NLPITD",
    "data_type": "nlp",
    "name": "nlp/CommonCrawl",
    "caption_keys": ["text"],
    "samples": LargeInt("1M"),
}
```

---

## Batch Output Format

All dataset classes output batches in a unified format for the training loop.

### Standard Batch Fields

| Field | Type | Description |
|-------|------|-------------|
| `input_ids` | `torch.LongTensor` | Token IDs for the input sequence |
| `attention_mask` | `torch.LongTensor` | Attention mask (1 for valid tokens, 0 for padding) |
| `labels` | `torch.LongTensor` | Labels for loss computation (same as `input_ids` with `IGNORE_INDEX` for padding) |
| `pixel_values` | `torch.Tensor` | Image pixel values (shape: `[batch, channels, height, width]`) |
| `image_filtered_idx` | `list[int]` | Indices of filtered images (for statistics) |
| `waste_token_num` | `list[int]` | Number of wasted tokens due to sequence length limits |

### Batch Structure

```python
batch = {
    "input_ids": torch.LongTensor([batch_size, seq_len]),
    "attention_mask": torch.LongTensor([batch_size, seq_len]),
    "labels": torch.LongTensor([batch_size, seq_len]),
    "pixel_values": torch.Tensor([batch_size, num_images, channels, height, width]),
    "image_filtered_idx": [int, ...],  # List of filtered image indices
    "waste_token_num": [int, ...],      # List of wasted token counts per sample
}
```

### Special Tokens

| Token | Description |
|-------|-------------|
| `<image_0>`, `<image_1>`, ... | Image placeholder tokens (replaced with image embeddings during training) |
| `<BOI>`, `<EOI>` | Beginning/End of Image tokens |
| `<EOL>` | End of Line token (optional) |

> 💡 **Note**: The `labels` field uses `IGNORE_INDEX` for padding tokens, which are ignored during loss computation.

---

## Checkpoint Resumption

The `MixedDataloader` supports full checkpoint resumption, restoring not only model state but also data sampling state.

### Persisted State

`MixedDataloader` persists three types of state:

#### 1. `mixing_status`

Tracks the mixing state across all workers:

```python
{
    "num_workers": int,
    "data_status": {
        "dataset_name": {
            "worker_id": int  # Number of samples processed
        }
    },
    "dataset_state_dict": {
        "worker_id": {
            "rng_state": ...,
            "buffer_state": ...,
            # ... other dataset state
        }
    },
    "last_worker_id": int
}
```

#### 2. `dataset_status`

Tracks progress for each dataset:

```python
{
    "dataset_name": {
        "hit": int,      # Successfully processed samples
        "miss": int,     # Skipped samples
        "total": int     # Total samples in dataset
    }
}
```

#### 3. `hw_aspect_ratio_status`

Tracks statistics for different aspect ratios:

```python
{
    "1:1": int,      # Number of 1:1 aspect ratio images
    "4:3": int,      # Number of 4:3 aspect ratio images
    # ... other aspect ratios
}
```

### Resumption Flow

1. **Save State**: `MixedDataloader` saves state periodically or at checkpoint save
2. **Load State**: On resumption, state is loaded from checkpoint
3. **Restore Dataset State**: Each `IndexedTarDataset` restores its `DataStatus`
4. **Restore Buffer State**: `NLPBuffer` and `ImageBuffer` restore their internal state
5. **Continue Sampling**: Sampling continues from the restored state

> ⚠️ **Important**: When resuming training, ensure:
> - The same number of workers is used (or state will be invalid)
> - Dataset configurations haven't changed
> - Checkpoint contains the data state (saved by `MixedDataloader`)

---

## Usage Examples

### Basic Usage

```python
from nextstep.datasets.mixed_dataset import MixedDataset, MixedDataloader
from transformers import AutoTokenizer

# Prepare data configuration
data_info_list = [
    {
        "cls": "nextstep.datasets.image_text_wds.ImageTextWDS",
        "data_type": "image_text_pair",
        "name": "text2image/BLIP3o-60k",
        "caption_keys": ["caption"],
        "caption_ratio": [1],
        "samples": LargeInt("58K"),
    },
    {
        "cls": "nextstep.datasets.nlp_itd.NLPITD",
        "data_type": "nlp",
        "name": "nlp/CommonCrawl",
        "caption_keys": ["text"],
        "samples": LargeInt("100K"),
    },
]

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("your-tokenizer")

# Create mixed dataset
mixed_dataset = MixedDataset(
    data_info_list=data_info_list,
    batch_size=4,
    tokenizer=tokenizer,
    down_factor=8,  # VAE downsampling factor
    hw_aspect_ratios_ids={
        "1:1": [100, 101],
        "4:3": [102, 103],
        # ... more aspect ratios
    },
    max_len=1280,
)

# Create dataloader
dataloader = MixedDataloader(
    dataset=mixed_dataset,
    num_workers=8,
    timeout=30.0,
)

# Use in training loop
for batch in dataloader:
    # batch contains: input_ids, attention_mask, labels, pixel_values, etc.
    loss = model(**batch)
    # ... training code
```

### Checkpoint Resumption

```python
# Save checkpoint (including data state)
checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "data_state": dataloader.mixing_status.state_dict(),
}

# Load checkpoint
checkpoint = torch.load("checkpoint.pt")
model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])

# Restore data state
dataloader.mixing_status.load_state_dict(checkpoint["data_state"])

# Continue training
for batch in dataloader:
    # ... training continues from restored state
```

---

## Troubleshooting

### Slow Data Loading

**Problem**: Data loading is slow or batches are delayed.

**Solutions**:
- Increase `num_workers` (but not exceeding CPU cores / number of GPUs)
- Check if data is on slow storage (consider using local cache)
- Reduce `batch_size` if memory is constrained
- Check `timeout` parameter - increase if data loading takes longer

### Out of Memory

**Problem**: Out of memory errors during batch construction.

**Solutions**:
- Reduce `batch_size`
- Reduce `max_len` to decrease sequence length
- Reduce `lru_size` in dataset initialization
- Check if `BatchBucket.max_accumulated_samples` threshold is too high

### Aspect Ratio Mismatch

**Problem**: Images with different aspect ratios in the same batch.

**Solutions**:
- This is expected behavior - mixed aspect ratios go to `ANY_ASPECT_RATIO` bucket
- Ensure `hw_aspect_ratios_ids` includes all needed aspect ratios
- Check that `BatchBucket` is correctly grouping by aspect ratio

### State Restoration Issues

**Problem**: Data state not restored correctly after resumption.

**Solutions**:
- Ensure same `num_workers` is used
- Check that checkpoint contains `data_state`
- Verify dataset configurations haven't changed
- Check `mixing_status.state_dict()` is being saved correctly

### Missing Samples

**Problem**: Some samples are being skipped.

**Solutions**:
- Check `dataset_status` for "miss" counts
- Review filtering conditions in dataset configuration
- Check logs for specific error messages
- Verify data format matches expected schema

---

## Related Documentation

- **Data Loading System**: `nextstep/data/README.md` - How tar files are indexed and loaded
- **Configuration System**: `configs/README.md` - How to configure datasets for training
- **Training Engine**: `nextstep/engine/train_nextstep_ds.py` - Training script that uses datasets
- **Base Dataset Class**: `nextstep/data/indexed_tar_dataset.py` - Base class for all datasets

---

## Summary

Core concepts of the dataset system:

1. **MixedDataset**: Combines multiple data sources with weighted sampling
2. **Buffers**: `NLPBuffer` and `ImageBuffer` temporarily store samples before batching
3. **BatchBucket**: Organizes samples into batches grouped by aspect ratio
4. **MixedDataloader**: Wraps dataset with state tracking and checkpoint resumption
5. **Unified Output**: All datasets output batches in the same format for training

The system is designed for efficiency and flexibility, supporting multiple data types, aspect ratio grouping, and full state restoration for seamless checkpoint resumption.
