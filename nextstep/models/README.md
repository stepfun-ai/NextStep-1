# NextStep Model Architecture Guide

Welcome to the NextStep model architecture guide! This guide will help you understand the model structure and multimodal token system.

---

## 📖 Table of Contents

1. [Introduction](#introduction)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
4. [Special Tokens](#special-tokens)
5. [Image Token System](#image-token-system)
6. [Loss Functions](#loss-functions)
7. [Aspect Ratio Handling](#aspect-ratio-handling)
8. [Integration with Training](#integration-with-training)
9. [Related Documentation](#related-documentation)

---

## Introduction

The `nextstep/models/` directory defines the **model architecture and multimodal token system** for NextStep. This layer provides:

- ✅ **Model architecture**: NextStep model based on Qwen2 backbone with Flow Matching Head
- ✅ **Multimodal token system**: Special tokens for images, text, and control
- ✅ **Aspect ratio support**: Dynamic image token allocation based on aspect ratios
- ✅ **Dual loss training**: Language modeling loss + image generation loss
- ✅ **VAE integration**: Image encoding/decoding via VAE

---

## System Architecture

### Model Structure

```
NextStepModel
    ├── Qwen2Model (Backbone)
    │   ├── Embedding Layer
    │   ├── Transformer Layers
    │   └── Output Layer
    ├── Image Input Projector
    ├── Image Output Projector
    └── Flow Matching Head
        ├── Timestep Embedder
        ├── ResBlocks (with Adaptive Layer Norm)
        └── Final Layer
```

### Data Flow

```
Input (Text + Images)
    ↓
Tokenization (Special Tokens)
    ↓
Embedding (Text + Image Tokens)
    ↓
Qwen2Model (Backbone)
    ↓
Flow Matching Head (Image Generation)
    ↓
Output (Text Tokens + Image Tokens)
    ↓
Loss Computation (LM Loss + Image Loss)
```

---

## Core Components

### `nextstep/modeling_nextstep.py` - Core Model Implementation

This file contains the main model architecture and training logic.

#### `NextStepModel` - Main Model Class

**Purpose**: Multimodal model combining language modeling and image generation capabilities.

**Architecture**:

| Component | Description |
|-----------|-------------|
| **Backbone** | Qwen2Model (transformer-based language model) |
| **Image Input Projector** | Projects VAE-encoded images to model hidden dimension |
| **Image Output Projector** | Projects model hidden states to image token dimension |
| **Flow Matching Head** | Generates image tokens via flow matching process |
| **LM Head** | Language modeling head for text generation |

**Key Features**:

- **Multimodal processing**: Handles both text and image inputs
- **Dynamic image tokens**: Image token count varies by aspect ratio
- **Gradient checkpointing**: Supports memory-efficient training
- **Generation support**: Includes sampling methods for inference

**Inheritance**:

```python
NextStepModel(NextStepMixin, Qwen2Model, GenerationMixin)
```

#### `NextStepConfig` - Model Configuration

**Purpose**: Configuration class extending Qwen2Config with NextStep-specific parameters.

**Key Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `image_size` | `int` | Base image size (e.g., 256) |
| `patch_size` | `int` | Patch size for image tokens |
| `num_channels` | `int` | Number of VAE latent channels |
| `hw_aspect_ratios_ids` | `dict[str, list[int]]` | Mapping of aspect ratios to token IDs |
| `image_placeholder_id` | `int` | Token ID for image placeholder |
| `boi` / `eoi` | `int` | Begin/End of Image token IDs |
| `lm_loss_weight` | `float` | Weight for language modeling loss |
| `image_loss_weight` | `float` | Weight for image generation loss |
| `fm_head_dim` | `int` | Flow Matching Head dimension |
| `fm_head_layers` | `int` | Number of layers in Flow Matching Head |

#### `FlowMatchingHead` - Image Generation Head

**Purpose**: Generates image tokens using flow matching (continuous normalizing flows).

**Components**:

| Component | Description |
|-----------|-------------|
| **TimestepEmbedder** | Embeds diffusion timesteps into vector representations |
| **ResBlocks** | Residual blocks with adaptive layer normalization |
| **FinalLayer** | Output layer for image token prediction |

**Key Features**:

- **Flow matching**: Uses continuous normalizing flows for image generation
- **Adaptive normalization**: Modulates activations based on conditions
- **Multi-step sampling**: Supports various ODE/SDE solvers for inference

#### `NextStepOutputWithPast` - Model Output

**Purpose**: Extended output structure containing both language and image losses.

**Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `loss` | `torch.FloatTensor` | Total weighted loss |
| `lm_loss` | `torch.FloatTensor` | Language modeling loss |
| `image_loss` | `torch.FloatTensor` | Image generation loss |
| `logits` | `torch.FloatTensor` | Model output logits |
| `past_key_values` | `Cache` | Cached key-value pairs for generation |

---

### `nextstep/tokenization_nextstep.py` - Special Tokens

**Purpose**: Defines all special tokens used in the multimodal token system.

#### Special Token Definitions

| Token | Constant | Description |
|-------|----------|-------------|
| **Padding** | `DEFAULT_PAD_TOKEN` | `[PAD]` - Padding token |
| **Begin of Image** | `DEFAULT_BOI_TOKEN` | `<\|begin_of_image\|>` - Marks start of image |
| **End of Image** | `DEFAULT_EOI_TOKEN` | `<\|end_of_image\|>` - Marks end of image |
| **Image Placeholder** | `DEFAULT_IMAGE_PLACEHOLDER_TOKEN` | `<\|image_placeholder\|>` - Placeholder for image position |
| **Image Area** | `DEFAULT_IMAGE_AREA_TOKEN` | `<\|image_area\|>` - Specifies image area |
| **Begin of Text** | `DEFAULT_BOS_TOKEN` | `<\|beginoftext\|>` - Marks start of text |
| **End of Line** | `DEFAULT_EOL_TOKEN` | `<\|image_end_of_line\|>` - End of line marker |
| **Begin of Prompt Refinement** | `DEFAULT_BOPR_TOKEN` | `<\|begin_of_prompt_refinement\|>` - Prompt refinement start |
| **End of Prompt Refinement** | `DEFAULT_EOPR_TOKEN` | `<\|end_of_prompt_refinement\|>` - Prompt refinement end |
| **Begin of Thinking** | `DEFAULT_BOT_TOKEN` | `<\|begin_of_thinking\|>` - Thinking process start |
| **End of Thinking** | `DEFAULT_EOT_TOKEN` | `<\|end_of_thinking\|>` - Thinking process end |

#### Token Registration

```python
special_tokens_dict = dict(
    pad_token=DEFAULT_PAD_TOKEN,
    additional_special_tokens=[
        DEFAULT_IMAGE_AREA_TOKEN,
        DEFAULT_BOI_TOKEN,
        DEFAULT_EOI_TOKEN,
        DEFAULT_IMAGE_PLACEHOLDER_TOKEN,
        # ... other tokens
    ],
)
```

**Usage**: These tokens are added to the tokenizer vocabulary during model initialization.

---

### `nextstep/aspect_ratio.py` - Aspect Ratio Utilities

**Purpose**: Provides utilities for handling multiple aspect ratios in image processing.

#### Key Functions

| Function | Description |
|----------|-------------|
| `ar2str(h, w)` | Converts height/width tuple to string format (e.g., `"16*16"`) |
| `str2ar(s)` | Converts string format to height/width tuple |
| `get_ar_base(ars)` | Computes base size from aspect ratio list |
| `center_crop_arr()` | Center crops image to square |
| `center_crop_arr_with_ar()` | Crops image to closest aspect ratio |
| `center_crop_arr_with_buckets()` | Crops image with bucket-based sizing |

#### Supported Aspect Ratios

The system supports multiple aspect ratios defined in `HW_ASPECT_RATIOS`:

```python
HW_ASPECT_RATIOS = [
    (8, 32),   # Portrait
    (9, 28),   # Portrait
    (16, 16),  # Square
    (28, 9),   # Landscape
    (32, 8),   # Landscape
    # ... more ratios
]
```

**Aspect Ratio Handling**:

- Images are cropped/resized to match the closest supported aspect ratio
- Image token count is determined by aspect ratio (e.g., `16*16` = 256 tokens)
- Different aspect ratios use different token ID prefixes

---

### `nextstep/modeling_nextstep_vae.py` - VAE Integration

**Purpose**: Provides VAE (Variational Autoencoder) integration for image encoding/decoding.

**Key Features**:

- **Image encoding**: Encodes images to latent space for model processing
- **Image decoding**: Decodes latent representations back to images
- **VAE wrapper**: Wraps diffusers AutoencoderKL for NextStep integration

**Usage**: VAE is used in the training loop to preprocess images before model input.

---

### `modeling_outputs.py` - Output Structures

**Purpose**: Defines extended output structures for NextStep model.

**Key Classes**:

- `BaseModelOutputWithPast`: Base output with past key values
- `CausalLMOutputWithPast`: Causal LM output with past key values
- Extended by `NextStepOutputWithPast` for dual loss support

---

## Special Tokens

### Token Roles

Special tokens serve different roles in the multimodal system:

#### Image Control Tokens

- **`<|begin_of_image|>` (BOI)**: Marks the start of an image sequence
- **`<|end_of_image|>` (EOI)**: Marks the end of an image sequence
- **`<|image_placeholder|>`**: Placeholder token indicating where an image should be inserted
- **`<|image_area|>`**: Specifies the area/size of an image

#### Text Control Tokens

- **`<|beginoftext|>` (BOS)**: Marks the start of text
- **`[PAD]`**: Padding token for sequence alignment

#### Advanced Control Tokens

- **`<|begin_of_prompt_refinement|>` / `<|end_of_prompt_refinement|>`**: For prompt refinement tasks
- **`<|begin_of_thinking|>` / `<|end_of_thinking|>`**: For chain-of-thought reasoning
- **`<|image_end_of_line|>`**: End of line marker for image sequences

### Token Usage in Sequences

**Example Sequence**:

```
<|beginoftext|>A cat sitting on a <|image_placeholder|>.<|begin_of_image|><image_tokens><|end_of_image|>
```

**Token Flow**:

1. Text tokens are processed normally
2. `<|image_placeholder|>` indicates image position
3. `<|begin_of_image|>` marks image start
4. Image tokens follow (number depends on aspect ratio)
5. `<|end_of_image|>` marks image end

---

## Image Token System

### Dynamic Token Allocation

Image token count is **dynamically determined** based on aspect ratio:

| Aspect Ratio | Grid Size | Token Count |
|--------------|-----------|-------------|
| `16*16` | 16×16 | 256 tokens |
| `8*32` | 8×32 | 256 tokens |
| `32*8` | 32×8 | 256 tokens |
| `12*21` | 12×21 | 252 tokens |

**Formula**: `token_count = height * width` (after patch size normalization)

### Aspect Ratio Token IDs

Each aspect ratio has associated token IDs used as prefixes:

```python
hw_aspect_ratios_ids = {
    "16*16": [100, 101],      # Token IDs for square images
    "8*32": [102, 103],        # Token IDs for portrait images
    "32*8": [104, 105],        # Token IDs for landscape images
    # ... more aspect ratios
}
```

**Usage**: These token IDs are prepended to image sequences to indicate aspect ratio.

### Image Token Processing

1. **VAE Encoding**: Images are encoded to latent space via VAE
2. **Projection**: Latent vectors are projected to model hidden dimension
3. **Token Generation**: Flow Matching Head generates image tokens
4. **Decoding**: Image tokens are decoded back to images via VAE

---

## Loss Functions

### Dual Loss Training

NextStep uses **two loss components** for training:

#### 1. Language Modeling Loss (`lm_loss`)

**Purpose**: Trains the model to predict text tokens.

**Computation**:
- Standard cross-entropy loss on text token predictions
- Only computed on text tokens (image tokens are masked)

**Weight**: Controlled by `config.lm_loss_weight` (default: 1.0)

#### 2. Image Generation Loss (`image_loss`)

**Purpose**: Trains the model to generate image tokens.

**Computation**:
- Flow matching loss on image token predictions
- Computed via `forward_genloss()` method
- Uses continuous normalizing flow objective

**Weight**: Controlled by `config.image_loss_weight` (default: 1.0)

### Total Loss

```python
total_loss = lm_loss_weight * lm_loss + image_loss_weight * image_loss
```

**Loss Masking**:

- Padding tokens are masked (using `IGNORE_INDEX = -100`)
- Image tokens use image loss, text tokens use LM loss
- Loss is computed only on valid positions

---

## Aspect Ratio Handling

### Multi-Aspect Ratio Training

NextStep supports training with multiple aspect ratios simultaneously:

**Benefits**:
- More flexible image generation
- Better handling of different image shapes
- Reduced cropping artifacts

**Implementation**:
- Images are grouped by aspect ratio during batch construction
- Each aspect ratio uses its own token ID prefix
- Flow Matching Head handles variable-length image sequences

### Aspect Ratio Selection

**During Training**:
- Images are cropped to closest supported aspect ratio
- Aspect ratio is determined by image dimensions
- Token count is computed based on aspect ratio

**During Inference**:
- Aspect ratio can be specified via `hw_aspect_ratio` parameter
- Model generates image tokens for specified aspect ratio
- Token count matches aspect ratio requirements

---

## Integration with Training

### Model Initialization

**In Training Script** (`nextstep/engine/train_nextstep_ds.py`):

```python
from nextstep.models.nextstep.modeling_nextstep import NextStepConfig, NextStepModel

# Create configuration
config = NextStepConfig.from_pretrained(...)

# Initialize model
model = NextStepModel.from_pretrained(
    config.model_name_or_path,
    config=config,
)
```

### Tokenizer Extension

**Special tokens are added to tokenizer**:

```python
from nextstep.models.nextstep.tokenization_nextstep import special_tokens_dict

# Add special tokens
tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))
```

### Image Preprocessing

**VAE encoding in training loop**:

```python
# In training_step()
data = preprocess_pixel_values(data, vae, config)
# Images are encoded to latent space before model input
```

### Loss Computation

**Dual loss is computed automatically**:

```python
outputs = model(**data)
# outputs.loss = total weighted loss
# outputs.lm_loss = language modeling loss
# outputs.image_loss = image generation loss
```

---

## Related Documentation

- **Training Engine**: `nextstep/engine/README.md` - How the model is used in training
- **Dataset System**: `nextstep/datasets/README.md` - How data is prepared for the model
- **Configuration System**: `configs/README.md` - How to configure model parameters
- **Qwen2 Model**: [Qwen2 Documentation](https://github.com/QwenLM/Qwen2) - Backbone model documentation

---

## Summary

Core concepts of the NextStep model:

1. **Architecture**: Qwen2 backbone + Flow Matching Head for multimodal generation
2. **Special Tokens**: Comprehensive token system for text, images, and control
3. **Image Tokens**: Dynamic token allocation based on aspect ratios
4. **Dual Loss**: Language modeling loss + image generation loss
5. **Aspect Ratios**: Support for multiple aspect ratios with efficient token usage

The model is designed for efficient multimodal training and generation, supporting flexible image sizes and aspect ratios while maintaining high-quality text and image generation capabilities.
