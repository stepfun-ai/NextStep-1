# NextStep Data Loading System Guide

Welcome to the NextStep data loading system! This guide will help you understand how the system efficiently reads tar-format data and builds indexes.

---

## 📖 Table of Contents

1. [Introduction](#introduction)
2. [Core Components](#core-components)
3. [Key Files](#key-files)
4. [Common Workflows](#common-workflows)
5. [Extension Guide](#extension-guide)
6. [Data Flow Overview](#data-flow-overview)
7. [Troubleshooting](#troubleshooting)
8. [Related Documentation](#related-documentation)

---

## Introduction

The `nextstep/data/` directory is the **core foundation layer** of the data loading system, responsible for efficiently reading tar-format data and building indexes. This layer addresses the following key challenges:

- ✅ **Fast sample location**: Quickly locate specific samples in tar files without decompressing the entire archive
- ✅ **Efficient index caching**: Cache index information to avoid repeated parsing
- ✅ **Distributed training support**: Handle data sharding and worker management for multi-GPU training
- ✅ **Unified data handling**: Process different data field formats (images, text, JSON, etc.) uniformly

---

## Core Components

### Architecture Overview

The data loading system consists of three main layers:

```
┌─────────────────────────────────────┐
│   IndexedTarDataset (Base Class)    │  ← Distributed training support
│   - Worker management               │
│   - Data sharding                   │
│   - State recovery                  │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│   IndexedTarSamples                 │  ← Sample organization
│   - Group files by key              │
│   - Return sample dictionaries      │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│   IndexedTar                       │  ← Core indexing
│   - Parse tar headers               │
│   - Build and cache indexes         │
│   - Support mmap/memory backends    │
└─────────────────────────────────────┘
```

---

## Key Files

### `indexed_tar.py` - Core Tar File Indexing

**Purpose**: The foundation of the entire data loading system, responsible for direct tar file operations.

**Key Features**:

| Feature | Description |
|---------|-------------|
| **Parse tar headers** | Read tar file headers to get file positions and sizes |
| **Build indexes** | Create mappings from filenames to positions (`by_name`) and from indices to positions (`by_index`) |
| **Cache indexes** | Save index information as `.index` files for fast loading on subsequent use |
| **Dual backends** | Support both `mmap` (memory-mapped, low memory) and `memory` (full file in memory) backends |

**Key Classes**:

| Class | Description |
|-------|-------------|
| `IndexedTar` | Basic tar file indexing class, provides access by index or filename |
| `IndexedTarSamples` | Extends `IndexedTar` to organize multiple files with the same key (e.g., `sample1.jpg` and `sample1.txt`) into a sample dictionary |

**Backend Selection**:

- **`mmap`**: Uses memory mapping, suitable for local files with low memory usage
- **`memory`**: Loads entire file into memory, suitable for remote files (e.g., S3) or small files

**Usage Scenario**: Use this module when you need to quickly read specific samples from tar files.

---

### `indexed_tar_dataset.py` - Iterable Dataset Base Class

**Purpose**: Provides an iterable dataset base class for distributed training scenarios, handling complex multi-process and multi-GPU logic.

**Key Features**:

| Feature | Description |
|---------|-------------|
| **Iterable dataset** | Inherits from `torch.utils.data.IterableDataset`, supports streaming reads for large datasets |
| **Distributed support** | Automatically handles `rank` (current GPU ID) and `world_size` (total GPU count) |
| **Sharding modes** | Supports two sharding modes: by sample or by URL (tar file) |
| **Worker management** | Manages worker state during multi-process data loading to ensure each worker reads different data |
| **State recovery** | Supports checkpoint resumption via `DataStatus` class, tracking samples processed by each worker |
| **Unified output** | All subclasses return `RetSample` objects containing sample data, metadata, worker info, etc. |

**Key Classes**:

| Class | Description |
|-------|-------------|
| `IndexedTarDataset` | Abstract base class, subclasses must implement `data_generator()` method |
| `RetSample` | Unified data return structure containing sample data, index info, missing sample statistics, etc. |
| `DataStatus` | Used for recording and recovering training state |

**Sharding Modes**:

- **Sample mode**: Split samples across workers (used when there are fewer URLs than `world_size * 10`)
- **URL mode**: Split tar files across workers (used when there are many URLs, reduces file lock contention)

**Usage Scenario**: Inherit this base class when implementing a dataset that supports distributed training.

---

### `data_handler.py` - Data Field Decoder

**Purpose**: Automatically decodes different data field formats based on file extensions.

**Key Features**:

- **Extension-based decoding**: Automatically selects the corresponding decoder function based on file extension (e.g., `.jpg`, `.txt`, `.json`)

**Supported Data Formats**:

| Format | Extension | Decoded Output |
|--------|-----------|----------------|
| Text | `.txt` | UTF-8 string |
| JSON | `.json` | Python dictionary |
| Images | `.jpg`, `.jpeg`, `.png`, `.webp` | PIL Image object (converted to RGB) |
| NumPy | `.npy`, `.npz` | NumPy array |
| Pickle | `.pickle` | Python object |
| Video | `.mp4` | Raw bytes |

**Extension Method**: To support new data formats, add new extensions and corresponding decoder functions to the `default_handlers` dictionary.

**Usage Scenario**: `IndexedTarSamples` automatically uses these decoders when reading samples. You typically don't need to call them directly.

---

### `_gen_meta.py` - Generate Tar File Metadata

**Purpose**: CLI tool to batch scan tar files and generate metadata JSON files.

**Key Features**:

| Feature | Description |
|---------|-------------|
| **Scan tar files** | Recursively scan all `.tar` files in the specified directory |
| **Extract metadata** | For each tar file, extract: `num_samples`, `size` (bytes), `checksum` |
| **Generate metadata file** | Aggregate metadata for all tar files into a JSON file |
| **Parallel processing** | Supports multi-process parallel processing to speed up scanning |

**Output Files**:

| File | Description |
|------|-------------|
| `meta.json` | Contains relative paths and metadata for all tar files |
| `meta_stats.json` | Overall dataset statistics (total samples, total files, etc.) |

**Usage**:

```bash
# Basic usage (meta.json will be saved in dataset root directory)
gen_meta /path/to/dataset/root_dir

# Specify save path
gen_meta /path/to/dataset/root_dir --meta_save_path /path/to/meta.json

# Specify number of parallel jobs
gen_meta /path/to/dataset/root_dir --meta_save_path /path/to/meta.json --n_jobs 32
```

**Parameters**:

| Parameter | Description |
|-----------|-------------|
| `dir` | Root directory where tar files are located (required) |
| `--meta_save_path` | Save path for meta.json (default: `{dir}/meta.json`) |
| `--n_jobs` | Number of parallel jobs (default: 64, set to 0 to auto-detect CPU cores) |

> ⚠️ **Important**: This tool must be run before using any dataset for the first time. The data loading system depends on the generated metadata file.

---

### `_warmup_data.py` - Warm Up Data Indexes

**Purpose**: CLI tool to pre-build indexes (`.index` files) for tar files, avoiding delays during the first read in training.

**Key Features**:

| Feature | Description |
|---------|-------------|
| **Match datasets** | Match datasets to warm up based on name patterns (supports wildcards) |
| **Check indexes** | Check which tar files don't have `.index` index files yet |
| **Batch index building** | Build and cache indexes for all matched tar files in parallel |
| **Parallel processing** | Supports multi-process parallel processing |

**Why Warm Up?**:

- First read of a tar file requires parsing the entire file to build an index, which can be time-consuming
- After warming up, indexes are cached as `.index` files, and subsequent reads directly load the index, which is much faster
- Warming up before training starts avoids delays during training

**Usage**:

```bash
# Warm up matching datasets (supports wildcards, multiple datasets separated by commas)
warmup_data "text2image/*,interleave/*" --n_jobs 32
```

**Parameters**:

| Parameter | Description |
|-----------|-------------|
| `names` | Dataset name patterns (supports wildcards, comma-separated) |
| `--n_jobs` | Number of parallel jobs (default: 0, auto-detect CPU cores) |

> 💡 **Tip**: Especially useful when reading data from remote storage like S3, where first-read latency can be significant.

---

### `build_wds.py` - WebDataset Packing Example

**Purpose**: Example script demonstrating how to pack raw data into WebDataset format (tar files).

**Key Features**:

| Feature | Description |
|---------|-------------|
| **Data format conversion** | Convert JSON-format raw data to WebDataset standard format |
| **Image processing** | Convert image paths to image byte data |
| **Standardized output** | Generate tar files conforming to NextStep data format requirements |
| **Parallel packing** | Supports multi-process parallel processing to speed up packing |

**Output Format**:

Each tar file contains multiple samples, and each sample contains:

- `key.json`: Contains `caption` field (using `<image_n>` placeholders to describe image-text positions) and optional `caption_all`, `metadata` fields
- `key-{i}.jpg`: Image files, corresponding sequentially to `<image_0>`, `<image_1>`, etc. placeholders

**Usage**:

```python
# Modify the configuration in the script
EXAMPLE_PATH = "/path/to/your/data.json"  # Input data path
OUTPUT_DIR = "/path/to/output/tar/files"  # tar file output directory
MAX_SAMPLES_PER_TAR = 100  # Maximum number of samples per tar file

# Run the script
python nextstep/data/build_wds.py
```

> ⚠️ **Important Notes**:
> - This is an example script. You must modify `load_data` and `create_example` functions according to your data source format
> - The `<image_n>` placeholders in the `caption` field must correspond to the image file order
> - Sample `key` cannot contain periods (`.`) or hyphens (`-`)

---

## Common Workflows

### Workflow 1: Preparing a New Dataset

**Step 1**: Pack raw data into tar format

```bash
# Reference build_wds.py and modify according to your data format
python nextstep/data/build_wds.py
```

**Step 2**: Generate tar file metadata

```bash
gen_meta /path/to/dataset/root --meta_save_path /path/to/meta.json --n_jobs 32
```

**Step 3**: (Optional) Warm up indexes to speed up first read

```bash
warmup_data "your_dataset_name" --n_jobs 32
```

### Workflow 2: Daily Usage

```bash
# If dataset indexes haven't been built yet, warm them up in advance
warmup_data "text2image/*,interleave/*" --n_jobs 32
```

---

## Extension Guide

### Adding New Data Field Types

If your dataset contains new file formats (e.g., `.yaml`, `.csv`), register decoder functions in `data_handler.py`:

```python
# Add to the default_handlers dictionary in data_handler.py
import yaml
import pandas as pd
import io

default_handlers = {
    # ... existing handlers ...
    ".yaml": lambda data: yaml.safe_load(data.decode("utf-8")),
    ".csv": lambda data: pd.read_csv(io.BytesIO(data)),
}
```

### Implementing New Dataset Types

1. Create a new dataset class in the `datasets/` directory
2. Inherit from the `IndexedTarDataset` base class
3. Implement the `data_generator()` method, returning `RetSample` objects
4. Implement the `__repr__()` method for debugging

**Example Structure**:

```python
from nextstep.data.indexed_tar_dataset import IndexedTarDataset, RetSample

class MyCustomDataset(IndexedTarDataset):
    def __repr__(self):
        return f"MyCustomDataset(name={self.name}, ...)"
    
    def data_generator(self):
        # Your data loading logic here
        for index in self.index_generator():
            # Load and process sample
            sample = self._load_sample(index)
            yield RetSample(
                name=self.name,
                data_type=self.data_type,
                missing_indices=[],
                missing_urls_indices=[],
                missing_samples=0,
                index=index,
                url_index=self.index2indices(index),
                sample=sample,
                worker_id=self.worker_id,
                num_workers=self.num_workers,
            )
```

> ⚠️ **Important**: When data schema changes, adapt in `datasets/` subclasses first. Do not directly modify the base class protocol.

---

## Data Flow Overview

```
Raw Data (JSON/images/etc.)
    ↓
build_wds.py (packing)
    ↓
WebDataset Format (tar files)
    ↓
gen_meta (generate metadata)
    ↓
warmup_data (warm up indexes, optional)
    ↓
IndexedTarDataset subclass (reading)
    ↓
Training Usage
```

**Detailed Flow**:

1. **Packing**: Convert raw data to WebDataset tar format using `build_wds.py`
2. **Metadata Generation**: Run `gen_meta` to generate `meta.json` with sample statistics
3. **Index Warming** (Optional): Run `warmup_data` to pre-build `.index` files
4. **Data Loading**: Dataset classes inherit from `IndexedTarDataset` and use the indexes to efficiently load samples
5. **Training**: Samples are streamed to the training loop

---

## Troubleshooting

### Slow First Read

**Problem**: First read of a tar file is very slow.

**Solutions**:
- Run `warmup_data` to pre-build indexes before training
- Ensure `.index` files are in the same directory as tar files
- Check if tar files are on slow storage (consider using local cache)

### Index Cache Issues

**Problem**: Index cache is out of date or corrupted.

**Solutions**:
- Delete `.index` files and let the system rebuild them
- Ensure tar files haven't been modified after index creation
- Check disk space and permissions for index file creation

### Memory Issues

**Problem**: Out of memory when loading data.

**Solutions**:
- Use `mmap` backend instead of `memory` backend (requires local files)
- Reduce `lru_size` parameter in dataset initialization
- Check if multiple workers are loading the same large tar files simultaneously

### Distributed Training Data Overlap

**Problem**: Multiple workers or GPUs are reading the same samples.

**Solutions**:
- Ensure you're using `IndexedTarDataset` base class correctly
- Don't manually shuffle or split data in subclasses
- Check that `rank` and `world_size` are set correctly

### Missing Samples

**Problem**: Some samples are skipped during training.

**Solutions**:
- Check `RetSample.missing_samples` to see how many samples were skipped
- Use `skip_sample()` when a sample fails to load
- Use `success_sample()` to reset the missing sample counter
- Check logs for specific error messages

---

## Related Documentation

- **Dataset Classes**: `nextstep/datasets/` - Concrete dataset implementations
- **Data Configuration**: `configs/README.md` - How to configure datasets for training
- **Training Engine**: `nextstep/engine/train_nextstep_ds.py` - Training script that uses datasets
- **Data Zoos**: `nextstep/data_zoos.py` - Dataset registry and metadata management

---

## Summary

Core concepts of the data loading system:

1. **IndexedTar**: Core indexing engine that parses tar files and builds position indexes
2. **IndexedTarSamples**: Organizes files into samples based on keys
3. **IndexedTarDataset**: Base class for distributed training datasets with worker management
4. **Metadata**: `meta.json` files contain sample statistics for efficient data access
5. **Index Caching**: `.index` files cache parsed indexes to avoid repeated parsing

The system is designed for efficiency and scalability, supporting both local and remote storage, with automatic handling of distributed training complexities.
