# NextStep Configuration System Guide

Welcome to the NextStep configuration system! This guide will help you quickly get started and master the configuration system.

---

## 📖 Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Configuration System Structure](#configuration-system-structure)
4. [Creating New Configurations](#creating-new-configurations)
5. [Data Configuration Details](#data-configuration-details)
6. [Advanced Usage](#advanced-usage)
7. [Troubleshooting](#troubleshooting)
8. [Related Documentation](#related-documentation)
9. [Summary](#summary)

---

## Introduction

The `configs/` directory manages all configuration files for NextStep training experiments. The configuration system adopts a **"base configuration + derived configuration override"** design pattern, enabling you to:

- ✅ Quickly create new experiment configurations
- ✅ Reuse existing configurations to avoid code duplication
- ✅ Flexibly override configuration parameters via command line
- ✅ Clearly manage configurations for different resolutions and datasets

---

## Quick Start

### Launching Training

#### Using smartrun (Recommended)

```bash
smartrun -m configs.nextstep_qwen14b_512px
```

#### Using torchrun

**Single-node training**:
```bash
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
    -m configs.nextstep_qwen14b_512px
```

**Multi-node training**:
```bash
# Node 0 (master node)
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
    --master_addr=<master_node_ip> --master_port=29500 \
    -m configs.nextstep_qwen14b_512px

# Node 1
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \
    --master_addr=<master_node_ip> --master_port=29500 \
    -m configs.nextstep_qwen14b_512px
```

### Command-Line Parameter Override

Override parameters directly from the command line without modifying configuration files:

```bash
# Override a single parameter
smartrun -m configs.nextstep_qwen14b_512px training.max_steps=5000

# Override multiple parameters
smartrun -m configs.nextstep_qwen14b_512px \
    training.max_steps=5000 \
    training.learning_rate=2e-4 \
    data.image_size=512
```

> 💡 **Tip**: For more command-line override usage, refer to the [Advanced Usage](#advanced-usage) section.

---

## Configuration System Structure

### Directory Structure

```
configs/
├── README.md                          # This file (usage guide)
├── nextstep_qwen14b_256px.py         # Base configuration (256px, most complete)
├── nextstep_qwen14b_512px.py         # Derived configuration (512px, inherits from 256px)
└── data/                              # Data configuration directory
    ├── pretrain_data_256px.py        # 256px data source list
    ├── pretrain_data_512px.py        # 512px data source list
    └── pretrain_data.json            # Dataset registry (paths, metadata)
```

### File Descriptions

| File | Description |
|------|-------------|
| `nextstep_qwen14b_256px.py` | **Main base configuration**, contains complete model, data, and training parameters; serves as a template for creating new configurations |
| `nextstep_qwen14b_512px.py` | **Derived configuration**, created by importing and overriding the 256px configuration; only modifies resolution-related parameters |
| `data/pretrain_data_*.py` | **Data source configuration**, defines training and validation dataset lists, sampling ratios, filtering conditions, etc. |
| `data/pretrain_data.json` | **Dataset registry**, stores dataset paths, metadata, and other information |

### Three Main Components of Configuration

Each configuration file contains three main parts:

```python
config = OmegaConf.create(flags={"allow_objects": True})

# 1. Model Configuration - Define model architecture and paths
config.model = dict(
    model_type="NextStepModel",
    lm_model_name_or_path="...",      # Language model path
    vae_name_or_path="...",           # VAE model path
    # ... more model parameters
)

# 2. Data Configuration - Define datasets and data processing methods
config.data = dict(
    datasets=data_info_list,          # Training dataset list
    val_datasets=val_data_info_list,  # Validation dataset list
    image_size=256,                   # Image resolution
    # ... more data parameters
)

# 3. Training Configuration - Define training hyperparameters and strategies
config.training = dict(
    output_dir="./outputs/...",       # Output directory
    learning_rate=1e-4,                # Learning rate
    max_steps=500_000,                # Maximum training steps
    # ... more training parameters
)
```

> 💡 **Tip**: Detailed descriptions of all parameters are in the English comments in the configuration files. You can directly view the configuration files to understand the meaning of each parameter.

### Configuration Inheritance Mechanism

Derived configurations are created by importing and modifying base configurations to avoid code duplication:

```python
# Import base configuration in derived configuration
from .nextstep_qwen14b_256px import config

# Only override the configuration items that need to be modified
config.data.image_size = 512
config.training.per_device_train_batch_size = 1
config.training.max_steps = 10_000
```

---

## Creating New Configurations

### Scenario 1: Creating a Completely New Experiment Configuration

If you need to create a completely different experiment configuration:

1. **Copy the base configuration file**
   ```bash
   cp configs/nextstep_qwen14b_256px.py configs/my_experiment.py
   ```

2. **Modify the configuration**
   - Update model paths (`LM_MODEL_PATH`, `VAE_MODEL_PATH`, etc.)
   - Update output directory (`OUTPUT_DIR`)
   - Adjust training parameters (batch size, learning rate, training steps, etc.)
   - If you need to use different data configurations, create new data configuration files

3. **Test run**
   ```bash
   # Small-scale test run
   smartrun -m configs.my_experiment training.max_steps=100
   ```

### Scenario 2: Creating Variants Based on Existing Configurations

If you only need to modify a few parameters (such as resolution, batch size, etc.):

1. **Create a derived configuration file**
   ```python
   # configs/my_experiment.py
   from .nextstep_qwen14b_256px import config
   from .data.pretrain_data_256px import data_info_list, val_data_info_list
   
   # Override configuration
   config.data.image_size = 512
   config.training.per_device_train_batch_size = 1
   config.training.max_steps = 10_000
   config.training.output_dir = "./outputs/my_experiment"
   
   # Launch code (must be included)
   if __name__ == "__main__":
       from nextstep.engine.train_nextstep_ds import Arguments, main
       from nextstep.lazy_config import LazyLaunch
       from nextstep.lazyrun import setup_config, setup_environ
       setup_config()
       setup_environ()
       LazyLaunch(main, Arguments)
   ```

2. **Test run**
   ```bash
   smartrun -m configs.my_experiment
   ```

### Configuration Checklist

When creating a new configuration, ensure you check the following items:

- [ ] All model paths are correct (`LM_MODEL_PATH`, `VAE_MODEL_PATH`, etc.)
- [ ] Output directory path is correct (`OUTPUT_DIR`)
- [ ] Data configuration is correct (dataset list, sampling ratios, etc.)
- [ ] Batch size is suitable for GPU memory (`per_device_train_batch_size`)
- [ ] Training steps are reasonable (`max_steps`)
- [ ] Learning rate and warmup steps match (`learning_rate`, `warmup_steps`)
- [ ] Checkpoint save interval is reasonable (`save_steps`)

---

## Data Configuration Details

### ⚠️ Data Preparation (Important)

**Before using any new data configuration, you must complete the following three steps:**

#### Step 1: Build WebDataset (tar) Format Archives

All datasets that enter the data configuration must first be converted to WebDataset (tar) format using the `nextstep/data/build_wds.py` script.

**Usage**:

Refer to the `nextstep/data/build_wds.py` script and modify the `load_data` and `create_example` functions according to your data source format:

```python
# Modify the configuration in the script
EXAMPLE_PATH = "/path/to/your/data.json"  # Input data path
OUTPUT_DIR = "/path/to/output/tar/files"  # tar file output directory
MAX_SAMPLES_PER_TAR = 100  # Maximum number of samples per tar file

# Run the script
python nextstep/data/build_wds.py
```

**Parameter Descriptions**:

| Parameter | Description |
|-----------|-------------|
| `EXAMPLE_PATH` | Input JSON data file path (can be JSON or JSONL format) |
| `OUTPUT_DIR` | tar file output directory |
| `MAX_SAMPLES_PER_TAR` | Maximum number of samples per tar file (recommended: 100-1000) |

**Output Format**:

Each tar file contains multiple samples, and each sample contains:
- `key.json`: Contains a `caption` field (using `<image_n>` placeholders to describe image-text positions) and optional `caption_all`, `metadata` fields
- `key-{i}.jpg`: Image files, corresponding sequentially to `<image_0>`, `<image_1>`, etc. placeholders

> ⚠️ **Important Notes**:
> - You must modify the `load_data` and `create_example` functions according to your data source format
> - The `<image_n>` placeholders in the `caption` field must correspond to the image file order
> - Sample `key` cannot contain periods (`.`) or hyphens (`-`)

#### Step 2: Generate meta.json File

All datasets that enter the data configuration must first use the `gen_meta` tool to generate a `meta.json` file. This file contains sample statistics for each tar file and is key for indexing data during training.

**Usage**:

```bash
# Basic usage (meta.json will be saved in the dataset root directory)
gen_meta /path/to/your/dataset/root_dir

# Specify save path
gen_meta /path/to/your/dataset/root_dir --meta_save_path /path/to/meta.json

# Specify number of parallel jobs (to speed up processing)
gen_meta /path/to/your/dataset/root_dir --meta_save_path /path/to/meta.json --n_jobs 32
```

**Parameter Descriptions**:

| Parameter | Description |
|-----------|-------------|
| `dir` | Root directory where tar files are located (required) |
| `--meta_save_path` | Save path for meta.json (default: `{dir}/meta.json`) |
| `--n_jobs` | Number of parallel jobs (default: 64, set to 0 to auto-detect CPU cores) |

**Generated Files**:

- `meta.json`: Contains sample counts, checksums, and other information for each tar file
- `meta_stats.json`: Contains overall dataset statistics (total samples, total files, etc.)

#### Step 3: Register to pretrain_data.json

After generating `meta.json`, you need to register the dataset information in `configs/data/pretrain_data.json`.

**Registration Format**:

```json
{
    "dataset_name": {
        "data_type": "image_text|interleave|nlp|...",
        "dir": "/path/to/tar/files",
        "tar_meta_path": "/path/to/meta.json",
        "num_samples": 1000000,
        "description": "Dataset description information"
    }
}
```

**Field Descriptions**:

| Field | Description |
|-------|-------------|
| `dataset_name` | Name used in data configuration files (e.g., `"text2image/BLIP3o-60k"`), must match the `name` field in data configuration files |
| `data_type` | Data type (`image_text`, `interleave`, `nlp`, etc.) |
| `dir` | Directory path where tar files are located |
| `tar_meta_path` | Path to the meta.json file (usually `{dir}/meta.json`) |
| `num_samples` | Total number of samples in the dataset (can be obtained from `meta_stats.json`) |
| `description` | Dataset description information (optional, recommended to fill in data source information, etc.) |

**Example**:

```json
{
    "text2image/BLIP3o-60k": {
        "data_type": "image_text",
        "dir": "./nextstep_data/BLIP3o-60k",
        "tar_meta_path": "./nextstep_data/BLIP3o-60k/meta.json",
        "num_samples": 58859,
        "description": "filtered from https://huggingface.co/datasets/BLIP3o/BLIP3o-60k"
    }
}
```

**Data Preview and Validation**:

After completing dataset registration, you can use the data preview tool to check if tar packages or registered datasets are normal:

```bash
streamlit run nextstep/service/_preview.py --server.port 8501
```

This tool can:
- Preview sample content in tar files (images, text, etc.)
- Check if datasets registered in `pretrain_data.json` are normal
- Verify that data format meets requirements

> ⚠️ **Important Notes**:
> - All datasets must complete these three steps before they can be used by data configuration files
> - `tar_meta_path` must point to the correct `meta.json` file path
> - `num_samples` should match `total_samples` in `meta_stats.json`
> - Dataset names need to match the names used in data configuration files (e.g., `pretrain_data_256px.py`)

### Data Source Types

Data configuration files support multiple data source types:

| Data Source Type | Dataset Class | Purpose |
|-----------------|---------------|---------|
| **NLP Data** (text-only) | `NLPITD` | Language model pretraining |
| **Image Generation Data** (text-to-image) | `ImageTextWDS` | Text-to-image generation training |
| **Image Editing Data** (image-to-image) | `ImageEditingInterleave` | Image editing task training |
| **Interleaved Multimodal Data** (video and text) | `VideoInterleave` | Video understanding task training |

### Data Source Configuration Format

Each data source is a dictionary containing the following fields:

```python
{
    "cls": _convert_target_to_string(ImageTextWDS),  # Dataset class
    "data_type": "image_text_pair",                  # Data type
    "name": "text2image/BLIP3o-60k",                 # Dataset name
    "caption_keys": ["caption"],                     # Caption field list
    "caption_ratio": [1],                            # Weight ratio for each caption field
    "filter": {                                      # Data filtering conditions
        "area": [256*256, 1024*1024],               # Image area range (pixels)
        "aspect_ratio": 6,                           # Maximum aspect ratio
    },
    "samples": LargeInt("58K"),                      # Number of samples
}
```

### Sampling Ratio Control

- **`samples`**: Number of samples for each data source, using `LargeInt` to support large numbers
  - Examples: `LargeInt("58K")`, `LargeInt("20M")`
- **`scale_to()`**: Function used to scale data source lists to target total sample count while maintaining relative proportions
- **`MixedDataset`**: Automatically normalizes `samples` to sampling probabilities

### Viewing Data Statistics

Run the data configuration file to view data statistics:

```bash
python configs/data/pretrain_data_256px.py
```

Output includes:
- Total number of samples
- Distribution statistics by dataset prefix
- Detailed statistics for each dataset

---

## Advanced Usage

### Command-Line Configuration Override

You can override any configuration item in the configuration file via command-line parameters without modifying the configuration file:

#### Basic Usage

```bash
# Override a single parameter
smartrun -m configs.nextstep_qwen14b_256px training.max_steps=200

# Override multiple parameters
smartrun -m configs.nextstep_qwen14b_256px \
    training.max_steps=200 \
    training.learning_rate=2e-4 \
    data.image_size=512
```

#### Overriding Nested Configurations

Use dots (`.`) to separate and override nested configurations:

```bash
# Override training configuration
training.max_steps=10000
training.learning_rate=2e-4
training.per_device_train_batch_size=4

# Override model configuration
model.lm_loss_weight=0.02
model.freeze_lm_model=True

# Override data configuration
data.image_size=512
data.num_workers=8
```

#### Overriding List and Dictionary Configurations

For complex types (lists, dictionaries), use quotes:

```bash
# Override list configuration
training.report_to='["wandb"]'

# Override dictionary configuration
training.eval_kwargs='{"temperature": 0.8, "cfg": 5.0}'
```

---

## Troubleshooting

### Out of Memory (OOM)

**Solutions**:
- Reduce `per_device_train_batch_size` to decrease the number of batches being trained
- Reduce `model_max_length` to decrease sequence length within the same batch
- Increase `grad_accumulation_steps` to maintain effective batch size
- Reduce `image_size` or use a smaller model

### Slow Data Loading

**Solutions**:
- Increase `num_workers` (but not exceeding CPU cores / number of GPUs)
- Increase `prefetch_factor`
- Enable `pin_memory` (if the environment supports it)
- Check data storage location (local SSD is faster than network storage)

### Configuration Override Not Working

**Checklist**:
- Configuration path is correct (use dots to separate, e.g., `training.max_steps`)
- Configuration item name exactly matches the one in the configuration file (case-sensitive)
- For complex types (lists, dictionaries), use quotes

### Resuming Training

**Method 1**: Manually specify the checkpoint directory to resume from
```python
config.training.resume = "./outputs/checkpoint-10000"
```

**Method 2**: Enable automatic resume
```python
config.training.auto_resume = True  # Automatically resume from the latest checkpoint
```

**Notes**:
- Ensure `output_dir` matches the checkpoint directory
- Can also override via command line: `training.resume=./outputs/checkpoint-10000`

### Data Sampling Ratio

**Modification Methods**:
- Modify the `samples` field in the data configuration file
- Use the `scale_to()` function for uniform scaling
- `MixedDataset` automatically normalizes sampling probabilities

---

## Related Documentation

- **Training Script**: `nextstep/engine/train_nextstep_ds.py`
- **Training Arguments Definition**: `nextstep/engine/training_args.py`
- **Model Definition**: `nextstep/models/nextstep/modeling_nextstep.py`
- **Dataset Classes**: `nextstep/datasets/`
- **Configuration System**: `nextstep/lazy_config/`

---

## Summary

Core concepts of the configuration system:

1. **Base Configuration**: `nextstep_qwen14b_256px.py` is a complete template containing all parameter descriptions
2. **Derived Configuration**: Create new configurations by importing and overriding to avoid code duplication
3. **Command-Line Override**: Flexibly modify parameters without modifying configuration files
4. **Data Separation**: Data configuration is managed independently for easy reuse and modification

If you have questions, please check the detailed comments in the configuration files or refer to the related documentation.
