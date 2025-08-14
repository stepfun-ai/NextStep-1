# NextStep Training Engine Guide

Welcome to the NextStep training engine! This guide will help you understand the main training workflow and how to customize it for your needs.

---

## 📖 Table of Contents

1. [Introduction](#introduction)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
4. [Training Workflow](#training-workflow)
5. [Checkpoint Management](#checkpoint-management)
6. [Common Customizations](#common-customizations)
7. [Troubleshooting](#troubleshooting)
8. [Related Documentation](#related-documentation)

---

## Introduction

The `nextstep/engine/` directory contains the **main training workflow** for NextStep. This layer orchestrates:

- ✅ **Distributed training**: Multi-GPU and multi-node training support via DeepSpeed
- ✅ **Model initialization**: Tokenizer, model, and VAE setup
- ✅ **Data pipeline**: Training and validation data loading
- ✅ **Optimization**: Optimizer, scheduler, and gradient accumulation
- ✅ **Checkpoint management**: Automatic checkpoint saving and resumption
- ✅ **Logging and monitoring**: TensorBoard, WandB, and performance metrics

---

## System Architecture

### Training Flow

```
Configuration (Arguments)
    ↓
main(config)
    ├── Initialize Distributed Environment
    ├── Build Tokenizer & Special Tokens
    ├── Build Model & VAE
    ├── Build Data Pipeline (MixedDataset + MixedDataloader)
    ├── Initialize Optimizer + Scheduler + DeepSpeed
    └── Training Loop
        ├── Checkpoint Saving (periodic)
        ├── Evaluation (periodic)
        └── Training Step
            ├── Data Loading
            ├── VAE Preprocessing
            ├── Forward Pass
            └── Backward Pass + Optimization
```

### Component Interaction

```
Arguments (ModelArguments, DataArguments, TrainingArguments)
    ↓
main()
    ├── Model Components (Tokenizer, Model, VAE)
    ├── Data Components (MixedDataset, MixedDataloader)
    ├── Optimization Components (Optimizer, Scheduler, DeepSpeed)
    └── Training Loop
        ├── training_step() → Model Forward/Backward
        ├── val_step() → Model Evaluation
        └── save_checkpoint() → State Persistence
```

---

## Core Components

### `train_nextstep_ds.py` - Main Training Script

This file contains the core training logic and workflow orchestration.

#### `main(config: Arguments)` - Training Entry Point

**Purpose**: Orchestrates the entire training process from initialization to completion.

**Key Responsibilities**:

| Responsibility | Description |
|----------------|-------------|
| **Distributed initialization** | Sets up multi-GPU/multi-node training environment |
| **Component initialization** | Initializes tokenizer, model, VAE, and data loaders |
| **Optimization setup** | Configures optimizer, scheduler, and DeepSpeed |
| **Training loop** | Executes training, evaluation, and checkpoint saving |
| **Resource cleanup** | Closes loggers and cleans up resources |

**Workflow Steps**:

1. **Initialize Distributed Environment**
   - Set up CUDA device
   - Initialize distributed communication (NCCL)
   - Set random seeds for reproducibility

2. **Build Tokenizer & Special Tokens**
   - Load tokenizer from configuration
   - Add special tokens (`<image_0>`, `<BOI>`, `<EOI>`, etc.)
   - Configure tokenizer settings

3. **Build Model & VAE**
   - Initialize NextStep model with language model and image head
   - Load VAE for image encoding/decoding
   - Configure model components (freezing, initialization, etc.)

4. **Build Data Pipeline**
   - Create `MixedDataset` with training and validation datasets
   - Initialize `MixedDataloader` with proper worker configuration
   - Set up data state restoration if resuming

5. **Initialize Optimizer + Scheduler + DeepSpeed**
   - Configure parameter groups with learning rate scaling
   - Set up optimizer (AdamW) with weight decay
   - Initialize learning rate scheduler
   - Initialize DeepSpeed engine with ZeRO optimization

6. **Training Loop**
   - Execute training steps with periodic checkpoint saving
   - Run evaluation at specified intervals
   - Log metrics to TensorBoard/WandB
   - Handle graceful shutdown and cleanup

#### `training_step()` - Single Training Step

**Purpose**: Executes one training step including forward pass, backward pass, and optimization.

**Key Operations**:

| Operation | Description |
|-----------|-------------|
| **Data loading** | Get next batch from `MixedDataloader` |
| **VAE preprocessing** | Encode images to latent space via VAE |
| **Forward pass** | Compute model outputs and loss |
| **Backward pass** | Compute gradients via DeepSpeed |
| **Optimization** | Update model parameters (handled by DeepSpeed) |
| **Logging** | Record loss, gradients, and timing metrics |

**Loss Components**:

- **Total Loss**: Weighted combination of language model loss and image loss
- **LM Loss**: Language modeling loss on text tokens
- **Image Loss**: Loss on image tokens (if applicable)

**Gradient Accumulation**:

- Supports gradient accumulation for effective larger batch sizes
- Handled automatically by DeepSpeed engine

#### `val_step()` - Validation Step

**Purpose**: Evaluates the model on validation data without gradient computation.

**Key Operations**:

| Operation | Description |
|-----------|-------------|
| **Model evaluation mode** | Set model to `eval()` mode |
| **Validation loop** | Iterate through validation batches |
| **Loss computation** | Compute validation loss |
| **Metric logging** | Log validation metrics to TensorBoard/WandB |
| **Model training mode** | Restore model to `train()` mode |

**Validation Metrics**:

- Validation loss (total, LM, image)
- Per-dataset validation statistics
- Timing information

#### `save_checkpoint()` - Checkpoint Saving

**Purpose**: Saves model, optimizer, scheduler, and data state to disk.

**Saved Components**:

| Component | Description |
|-----------|-------------|
| **Model state** | Model weights (via DeepSpeed checkpoint) |
| **Optimizer state** | Optimizer states (via DeepSpeed checkpoint) |
| **Scheduler state** | Learning rate scheduler state |
| **Data state** | Data sampling state (via `MixedDataloader.save_state_dict()`) |
| **Tokenizer** | Tokenizer configuration and vocabulary |
| **Model config** | Model configuration file |

**Checkpoint Structure**:

```
checkpoints/checkpoint-{step}/
    ├── deepspeed_shards/          # DeepSpeed ZeRO shards
    ├── config.json                # Model configuration
    ├── tokenizer_config.json      # Tokenizer configuration
    ├── vocab.json                 # Tokenizer vocabulary
    └── mixing_status.json         # Data sampling state
```

#### `auto_resume()` - Automatic Checkpoint Resumption

**Purpose**: Automatically finds and resumes from the latest checkpoint.

**Resumption Logic**:

1. Scan checkpoint directory for `checkpoint-*` folders
2. Sort checkpoints by step number and modification time
3. Select the latest complete checkpoint
4. Load model, optimizer, scheduler, and data state
5. Resume training from the saved step

> ⚠️ **Important**: Automatic resumption will fail if more than 20 automatic resumes have occurred. In this case, manually specify the checkpoint path.

---

### `training_args.py` - Argument Definitions

This file defines the configuration dataclasses used throughout training.

#### `ModelArguments` - Model Configuration

**Purpose**: Defines model-related parameters.

**Key Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `lm_model_name_or_path` | `str` | Path to language model |
| `vae_name_or_path` | `str` | Path to VAE model |
| `tokenizer_name_or_path` | `str` | Path to tokenizer |
| `freeze_lm_model` | `bool` | Whether to freeze language model weights |
| `freeze_lm_head` | `bool` | Whether to freeze language model head |
| `lm_loss_weight` | `float` | Weight for language modeling loss |
| `image_loss_weight` | `float` | Weight for image loss |
| `model_max_length` | `int` | Maximum sequence length |

**Usage Example**:

```python
model_args = ModelArguments(
    lm_model_name_or_path="Qwen/Qwen2.5-7B",
    vae_name_or_path="stabilityai/sd-vae-ft-mse",
    freeze_lm_model=False,
    lm_loss_weight=1.0,
    image_loss_weight=1.0,
)
```

#### `DataArguments` - Data Configuration

**Purpose**: Defines data-related parameters.

**Key Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `datasets` | `list[str \| dict]` | Training dataset configurations |
| `val_datasets` | `list[str \| dict]` | Validation dataset configurations |
| `image_size` | `int` | Target image resolution |
| `use_multi_aspect` | `bool` | Enable multi-aspect ratio training |
| `drop_text_prob` | `float` | Probability of dropping text when sequence is too long |
| `num_workers` | `int` | Number of data loading workers |
| `pin_memory` | `bool` | Whether to pin memory for faster data transfer |

**Usage Example**:

```python
data_args = DataArguments(
    datasets=["text2image/BLIP3o-60k", "nlp/CommonCrawl"],
    val_datasets=["text2image/COCO"],
    image_size=256,
    use_multi_aspect=True,
    num_workers=8,
)
```

#### `TrainingArguments` - Training Configuration

**Purpose**: Defines training hyperparameters and settings.

**Key Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `output_dir` | `str` | Output directory for checkpoints and logs |
| `per_device_train_batch_size` | `int` | Batch size per device |
| `grad_accumulation_steps` | `int` | Gradient accumulation steps |
| `learning_rate` | `float` | Learning rate |
| `max_steps` | `int \| LargeInt` | Maximum training steps |
| `warmup_steps` | `int` | Number of warmup steps |
| `save_steps` | `int` | Checkpoint saving frequency |
| `eval_steps` | `int` | Evaluation frequency |
| `logging_steps` | `int` | Logging frequency |
| `bf16` / `fp16` | `bool` | Mixed precision training |
| `resume` | `str` | Checkpoint path for resumption |

**Usage Example**:

```python
training_args = TrainingArguments(
    output_dir="./outputs/experiment",
    per_device_train_batch_size=4,
    grad_accumulation_steps=4,
    learning_rate=5e-5,
    max_steps=100_000,
    save_steps=1000,
    eval_steps=1000,
    bf16=True,
)
```

---

## Training Workflow

### Complete Training Flow

```python
# 1. Initialize distributed environment
torch.cuda.set_device(dist_ctx.local_rank)
dist.init_process_group(...)

# 2. Build tokenizer and special tokens
tokenizer = AutoTokenizer.from_pretrained(...)
add_special_tokens(tokenizer)

# 3. Build model and VAE
model = NextStepModel.from_pretrained(...)
vae = AutoencoderKL.from_pretrained(...)

# 4. Build data pipeline
train_dataset = MixedDataset(...)
train_dataloader = MixedDataloader(dataset=train_dataset, ...)

# 5. Initialize optimizer, scheduler, and DeepSpeed
optimizer = torch.optim.AdamW(...)
scheduler = get_scheduler(...)
model, optimizer, _, scheduler = deepspeed.initialize(...)

# 6. Training loop
for step in range(start_step, max_steps + 1):
    # Save checkpoint
    if step % save_steps == 0:
        save_checkpoint(step, ...)
    
    # Evaluate
    if step % eval_steps == 0:
        val_step(step, ...)
    
    # Training step
    if step < max_steps:
        training_step(step, ...)
```

### Step-by-Step Breakdown

#### Step 1: Distributed Initialization

- Set CUDA device for current process
- Initialize process group for distributed communication
- Set random seeds for reproducibility

#### Step 2: Component Building

- **Tokenizer**: Load and configure tokenizer with special tokens
- **Model**: Initialize NextStep model with language model and image head
- **VAE**: Load VAE for image encoding/decoding
- **Data**: Create `MixedDataset` and `MixedDataloader` instances

#### Step 3: Optimization Setup

- **Parameter Groups**: Group parameters by learning rate scaling
- **Optimizer**: Initialize AdamW optimizer with weight decay
- **Scheduler**: Set up learning rate scheduler (cosine, linear, etc.)
- **DeepSpeed**: Initialize DeepSpeed engine with ZeRO optimization

#### Step 4: Training Loop

- **Checkpoint Saving**: Save model and data state periodically
- **Evaluation**: Run validation at specified intervals
- **Training Step**: Execute forward/backward pass and optimization
- **Logging**: Record metrics to TensorBoard/WandB

---

## Checkpoint Management

### Saving Checkpoints

Checkpoints are saved automatically at specified intervals:

```python
# In training loop
if step % config.training.save_steps == 0:
    save_checkpoint(step, config, tokenizer, model, dataloader_train)
```

**Saved Components**:

- Model weights (DeepSpeed ZeRO shards)
- Optimizer states
- Learning rate scheduler state
- Data sampling state (`mixing_status`, `dataset_status`, etc.)
- Tokenizer and model configuration

### Resuming Training

#### Manual Resumption

```python
# In configuration
config.training.resume = "./outputs/experiment/checkpoints/checkpoint-10000"
```

#### Automatic Resumption

```python
# If resume is set to output directory, auto_resume() finds latest checkpoint
config.training.resume = "./outputs/experiment"
# Automatically finds: ./outputs/experiment/checkpoints/checkpoint-10000
```

**Resumption Process**:

1. Load data state from checkpoint (`MixedDataloader.load_state_dict()`)
2. Load model checkpoint via DeepSpeed (`model.load_checkpoint()`)
3. Restore optimizer and scheduler states
4. Resume training from saved step

> ⚠️ **Important**: 
> - Ensure the same number of workers is used when resuming
> - Checkpoint must contain data state for proper resumption
> - Model configuration should match between training and resumption

---

## Common Customizations

### Adjusting Loss Weights

**Location**: `config.model.lm_loss_weight` and `config.model.image_loss_weight`

**Example**:

```python
# In configuration file
config.model.lm_loss_weight = 1.0  # Language model loss weight
config.model.image_loss_weight = 2.0  # Image loss weight (higher priority)
```

**Effect**: Controls the relative importance of language modeling vs. image generation loss.

### Adjusting Evaluation Frequency

**Location**: `config.training.eval_steps`

**Example**:

```python
# Evaluate every 500 steps
config.training.eval_steps = 500

# Disable evaluation
config.training.skip_eval = True
```

**Effect**: Controls how often validation is performed during training.

### Adjusting Checkpoint Saving Frequency

**Location**: `config.training.save_steps`

**Example**:

```python
# Save checkpoint every 2000 steps
config.training.save_steps = 2000

# Save only at the end
config.training.save_steps = config.training.max_steps
```

**Effect**: Controls checkpoint saving frequency (more frequent = more disk usage but better recovery).

### Adjusting Mixed Precision

**Location**: `config.training.bf16`, `config.training.fp16`, `config.training.amp`

**Example**:

```python
# Use bfloat16 (recommended for modern GPUs)
config.training.bf16 = True
config.training.fp16 = False

# Use float16 (for older GPUs)
config.training.bf16 = False
config.training.fp16 = True

# Use automatic mixed precision
config.training.amp = True
```

**Effect**: Reduces memory usage and speeds up training, with potential numerical stability trade-offs.

### Customizing Learning Rate Schedule

**Location**: `config.training.lr_scheduler_type` and `config.training.lr_scheduler_kwargs`

**Example**:

```python
# Cosine annealing
config.training.lr_scheduler_type = "cosine"
config.training.lr_scheduler_kwargs = {"min_lr": 1e-6}

# Linear decay
config.training.lr_scheduler_type = "linear"
config.training.lr_scheduler_kwargs = {"min_lr": 0.0}
```

**Effect**: Controls how learning rate changes during training.

### Adjusting Batch Size and Gradient Accumulation

**Location**: `config.training.per_device_train_batch_size` and `config.training.grad_accumulation_steps`

**Example**:

```python
# Effective batch size = per_device_train_batch_size * num_gpus * grad_accumulation_steps
config.training.per_device_train_batch_size = 2
config.training.grad_accumulation_steps = 8
# Effective batch size = 2 * 8 * 8 = 128 (assuming 8 GPUs)
```

**Effect**: Controls effective batch size for training (larger batch = more stable gradients but more memory).

---

## Troubleshooting

### Out of Memory (OOM)

**Problem**: Training runs out of GPU memory.

**Solutions**:
- Reduce `per_device_train_batch_size`
- Increase `grad_accumulation_steps` to maintain effective batch size
- Reduce `model_max_length` to decrease sequence length
- Enable gradient checkpointing in DeepSpeed config
- Use mixed precision training (`bf16=True` or `fp16=True`)

### Slow Training

**Problem**: Training is slower than expected.

**Solutions**:
- Increase `num_workers` for data loading (but not exceeding CPU cores)
- Enable `pin_memory=True` if supported
- Check data loading speed (use `run_data_only=True` to profile)
- Reduce `logging_steps` to decrease logging overhead
- Enable `torch.compile` if using PyTorch 2.0+

### Checkpoint Loading Errors

**Problem**: Cannot load checkpoint or resume training.

**Solutions**:
- Verify checkpoint path is correct
- Ensure checkpoint contains all required files
- Check that model configuration matches between training and resumption
- Verify DeepSpeed ZeRO configuration matches
- Check for file corruption or incomplete checkpoints

### Distributed Training Issues

**Problem**: NCCL errors or deadlocks in distributed training.

**Solutions**:
- Ensure all processes can communicate (check network)
- Verify `MASTER_ADDR` and `MASTER_PORT` are set correctly
- Check for synchronization issues (barriers, all-reduce)
- Reduce batch size if causing memory issues
- Check for data loading imbalance across workers

### Loss Not Decreasing

**Problem**: Training loss is not decreasing or is unstable.

**Solutions**:
- Check learning rate (may be too high or too low)
- Verify data quality and preprocessing
- Check loss weights (`lm_loss_weight`, `image_loss_weight`)
- Ensure gradient clipping is enabled (`max_grad_norm`)
- Check for numerical issues (NaN/Inf) in loss computation

---

## Related Documentation

- **Configuration System**: `configs/README.md` - How to configure training experiments
- **Dataset System**: `nextstep/datasets/README.md` - How datasets are constructed
- **Data Loading**: `nextstep/data/README.md` - How data is loaded from tar files
- **Model Definition**: `nextstep/models/nextstep/modeling_nextstep.py` - Model architecture
- **DeepSpeed Utils**: `nextstep/utils/deepspeed_utils.py` - DeepSpeed configuration helpers

---

## Summary

Core concepts of the training engine:

1. **main()**: Orchestrates the entire training workflow from initialization to completion
2. **training_step()**: Executes one training step with forward/backward pass
3. **val_step()**: Evaluates model on validation data
4. **save_checkpoint()**: Saves model, optimizer, and data state
5. **Arguments**: Configuration dataclasses for model, data, and training parameters

The system is designed for scalability and flexibility, supporting distributed training, automatic checkpoint management, and extensive customization options for different training scenarios.
