# NextStep Utils Module

The `nextstep/utils` module provides a collection of utility functions and classes used throughout the NextStep project for common operations.

---

## Overview

This module contains utilities for:

- ✅ Image and video processing
- ✅ Distributed training communication
- ✅ Logging and debugging
- ✅ Memory monitoring
- ✅ Optimizer and scheduler configuration
- ✅ Training utilities (seeding, profiling, etc.)
- ✅ Configuration management
- ✅ Package availability checks

---

## Module Structure

```
nextstep/utils/
├── README.md              # This document
├── __init__.py            # Module initialization
├── image_utils.py         # Image processing utilities
├── video_utils.py         # Video format constants
├── comm.py                # Distributed communication utilities
├── loguru.py              # Logging configuration
├── mem_utils.py           # Memory monitoring
├── optim_utils.py         # Optimizer parameter grouping
├── scheduler_utils.py     # Learning rate schedulers
├── training_utils.py      # Training helpers (seeding, etc.)
├── misc.py                # Miscellaneous utilities (LargeInt, etc.)
├── timer.py               # Timing utilities
├── debug.py               # Debugging helpers
├── import_utils.py        # Package availability checks
├── omegaconf_utils.py     # OmegaConf utilities
├── deepspeed_utils.py     # DeepSpeed configuration
├── compile_utils.py       # Torch compilation utilities
├── torch_profiler.py      # PyTorch profiling
├── general.py             # General system utilities (NUMA, etc.)
└── proxy.py               # Retry decorators
```

---

## Key Utilities

### Image Processing (`image_utils.py`)

Provides image format conversion and manipulation:

- **Format conversion**: PIL ↔ NumPy ↔ PyTorch Tensor
- **Image loading/saving**: Support for various formats (JPEG, PNG, WebP)
- **Image normalization**: Multiple data formats (0-255, 0-1, -1-1)
- **Image operations**: Resize, pad, grid layout, etc.

```python
from nextstep.utils.image_utils import load_image, to_pil, to_pt, normalize_pt

# Load and convert image
img = load_image("path/to/image.jpg")
img_tensor = to_pt(img)  # Convert to PyTorch tensor
img_normalized = normalize_pt(img_tensor, image_mode="11")  # Normalize to [-1, 1]
```

### Distributed Communication (`comm.py`)

Utilities for distributed training:

- **Process group management**: Initialize and manage distributed groups
- **Rank and world size**: Get current rank, local rank, world size
- **Communication primitives**: All-gather, gather, reduce, broadcast
- **Synchronization**: Barrier operations

```python
from nextstep.utils.comm import init_distributed, get_rank, get_world_size, synchronize

init_distributed()
rank = get_rank()
world_size = get_world_size()
synchronize()  # Barrier synchronization
```

### Logging (`loguru.py`)

Enhanced logging with custom logger:

- **Custom logger**: Extended loguru logger with `*_once` methods
- **Logging setup**: Configure logging for training scripts
- **Output redirection**: Redirect stdout/stderr to logger

```python
from nextstep.utils.loguru import get_logger, setup_logger

logger = get_logger()
logger.info("Training started")
logger.warning_once("This warning appears only once")
```

### Memory Monitoring (`mem_utils.py`)

Monitor GPU and CPU memory usage:

- **MemoryMonitor**: Track memory usage over time
- **PeriodicMemoryMonitor**: Periodic memory monitoring with callbacks

### Optimizer Utilities (`optim_utils.py`)

Parameter grouping and learning rate scaling:

- **Parameter grouping**: Group parameters for different learning rates
- **LR scaling functions**: Vision encoder and LLM-specific scaling
- **Gradient norm**: Compute gradient norms

### Scheduler Utilities (`scheduler_utils.py`)

Learning rate scheduler implementations:

- **Constant schedule**: Constant learning rate
- **Linear warmup**: Linear warmup schedule
- **Cosine schedule**: Cosine annealing with warmup
- **Polynomial decay**: Polynomial decay schedule
- **Inverse sqrt**: Inverse square root schedule

### Training Utilities (`training_utils.py`)

Training helper functions:

- **Seeding**: Set random seeds for reproducibility
- **Seed generation**: Generate seeds from arguments

```python
from nextstep.utils.training_utils import set_seed, make_seed

set_seed(42)  # Set seed for reproducibility
seed = make_seed("experiment_name", 100)  # Generate seed from arguments
```

### Miscellaneous (`misc.py`)

Common utilities:

- **LargeInt**: Integer class supporting K/M/B/T suffixes (e.g., "58K", "20M")
- **State dict comparison**: Compare model state dictionaries
- **Model downloading**: Download models from HuggingFace Hub

```python
from nextstep.utils.misc import LargeInt

samples = LargeInt("58K")  # 58000
samples = LargeInt("20M")  # 20000000
```

### Timer Utilities (`timer.py`)

Timing and timeout utilities:

- **TimerManager**: Manage multiple timers
- **Timeout decorator**: Add timeout to functions
- **Shell command timer**: Time shell command execution

### Import Utilities (`import_utils.py`)

Check package availability and versions:

- **Package checks**: Check if packages are installed
- **Version comparison**: Compare package versions
- **Feature detection**: Detect available features (Flash Attention, xFormers, etc.)

```python
from nextstep.utils.import_utils import is_torch_available, is_flash_attn_2_available

if is_torch_available():
    import torch

if is_flash_attn_2_available():
    # Use Flash Attention 2
    pass
```

### DeepSpeed Utilities (`deepspeed_utils.py`)

DeepSpeed configuration helpers:

- **Training config**: Generate DeepSpeed training configuration
- **Inference config**: Generate DeepSpeed inference configuration

### Compilation Utilities (`compile_utils.py`)

Torch compilation management:

- **CompileManager**: Manage torch.compile settings
- **Smart compile**: Compile functions with automatic fallback

---

## Common Usage Patterns

### Image Processing Pipeline

```python
from nextstep.utils.image_utils import load_image, to_pt, normalize_pt

# Load image
img = load_image("path/to/image.jpg")

# Convert to tensor and normalize
img_tensor = to_pt(img)
img_normalized = normalize_pt(img_tensor, image_mode="11")
```

### Distributed Training Setup

```python
from nextstep.utils.comm import init_distributed, get_rank, is_main_process
from nextstep.utils.loguru import setup_logger

# Initialize distributed training
init_distributed()

# Setup logging (only on main process)
if is_main_process():
    setup_logger()

rank = get_rank()
```

### Reproducible Training

```python
from nextstep.utils.training_utils import set_seed
from nextstep.utils.comm import get_rank

seed = 42
set_seed(seed, rank=get_rank())
```

---

## Notes

- **Image formats**: Supports PIL, NumPy arrays, and PyTorch tensors with automatic conversion
- **Distributed training**: All communication utilities assume NCCL backend
- **Logging**: Uses loguru for enhanced logging capabilities
- **Memory monitoring**: Useful for debugging OOM issues
- **LargeInt**: Used in configuration files for readable large numbers

---

## Related Documentation

- **Model Training**: `nextstep/engine/` - Training engine using these utilities
- **Data Processing**: `nextstep/data/` - Data loading and processing
- **Configuration**: `configs/` - Configuration files using LargeInt and other utilities
