import os

from omegaconf import OmegaConf

from nextstep.models.nextstep.tokenization_nextstep import special_tokens_dict

from .data.pretrain_data_256px import data_info_list, val_data_info_list
from nextstep.model_zoos import MODEL_ZOOS

NEXTSTEP_MODEL_PATH = None
LM_MODEL_PATH = MODEL_ZOOS["Qwen/Qwen2.5-14B"]
VAE_MODEL_PATH = MODEL_ZOOS["stepfun-ai/NextStep-1-f8ch16-Tokenizer"]
TOKENIZER_MODEL_PATH = MODEL_ZOOS["stepfun-ai/NextStep-1-f8ch16-Tokenizer"]
OUTPUT_DIR = "./outputs/nextstep_qwen14b_256px"
RESUME_DIR = OUTPUT_DIR

# ===== base config (copied from llama8b, then modified for Qwen-14B) =====
config = OmegaConf.create(flags={"allow_objects": True})

# ===== Model Configuration =====
config.model = dict(
    # Model type
    model_type="NextStepModel",
    # Full model path (including LM model and image head), set this if you have a complete NextStep model
    model_name_or_path=NEXTSTEP_MODEL_PATH,
    # Language model path (LM model only), set this if you have an initial Qwen model
    lm_model_name_or_path=LM_MODEL_PATH,
    # VAE model path, used for image encoding and decoding
    vae_name_or_path=VAE_MODEL_PATH,
    # Tokenizer path, used for text tokenization
    tokenizer_name_or_path=TOKENIZER_MODEL_PATH,
    # Whether to reinitialize the image head (Flow Matching Head)
    reinit_image_head=False,
    # Whether to use MLP layer before LM head
    use_mlp_before_lm_head=False,
    # Whether to freeze the token embedding layer (embed_tokens)
    freeze_embed_tokens=False,
    # Whether to freeze the language model body (transformer layers)
    freeze_lm_model=False,
    # Whether to freeze the language model output head (lm_head)
    freeze_lm_head=False,
    # Whether to train language model from scratch (without pretrained weights)
    lm_model_from_scratch=False,
    # Whether to weight loss by token length (True: weight by token count, False: direct sum)
    use_token_length_weight=False,
    # Weight for language model loss
    lm_loss_weight=0.01,
    # Weight for image generation loss
    image_loss_weight=1.0,
    # Noise strength for data augmentation during training
    noise_strength=0.0,
    # Maximum sequence length of the model
    model_max_length=8192,
    # Whether to use fast tokenizer (Rust implementation, faster)
    use_fast_tokenizer=True,
    # Special tokens dictionary for adding custom special tokens
    special_tokens_dict=special_tokens_dict,
    # Whether to use local files only (no network download)
    local_files_only=True,
    # Hidden dimension of Flow Matching Head
    fm_head_dim=1536,
    # Number of layers in Flow Matching Head
    fm_head_layers=12,
    # MLP expansion ratio in Flow Matching Head
    fm_head_mlp_ratio=1.0,
    # Batch multiplier for Flow Matching Head (for adjusting batch size)
    fm_head_batch_mul=4,
    # Whether to use output bias in attention layers (o_proj bias)
    o_attention_bias=False,
    # Whether to use QK normalization in attention layers
    attention_use_qk_norm=False,
)

# ===== Data Configuration =====
config.data = dict(
    # Training dataset list
    datasets=data_info_list,
    # Validation dataset list
    val_datasets=val_data_info_list,
    # Whether to use multi-aspect ratio training (supports images with different aspect ratios)
    use_multi_aspect=True,
    # Input image size (in pixels)
    image_size=256,
    # Whether to use super multi-aspect ratio training (supports images with different aspect ratios)
    use_super_multi_aspect=False,
    # Image sizes for super multi-aspect ratio training, only used when use_super_multi_aspect is True
    buckets_for_super_multi_aspect=[256, 512, 768, 1024],
    # Probability of randomly dropping text (for training with text randomly removed, keeping only images)
    drop_text_prob=0.3,
    # Shift factor for VAE-encoded latents (for VAE normalization)
    shift_factor=0.0,
    # Scaling factor for VAE-encoded latents (for VAE normalization)
    scaling_factor=1.0,
    # VAE downsampling factor (image_size / vae_f = latent_size)
    vae_f=8,
    # VAE channel number (number of channels in latent)
    vae_ch=16,
    # Image patch size (size for splitting images into patches)
    patch_size=2,
    # Number of data loading workers
    # Increase worker count to improve parallelism and reduce data loading bottlenecks
    # Recommended value: CPU cores / GPU count, but not exceeding 8 (too many workers cause process switching overhead)
    num_workers=4,
    # Whether to use pin_memory (accelerates GPU transfer, but may have thread issues in containers)
    # If container environment allows, enabling pin_memory can accelerate CPU to GPU data transfer
    # If encountering thread issues, try setting to False
    pin_memory=True,  # Try enabling to accelerate data transfer, disable if issues occur
    # Number of data batches to prefetch per worker
    # Increase prefetch batch count to let workers prepare more data in advance, reducing wait time
    prefetch_factor=8,
    # Whether to keep worker processes persistent (avoids overhead of recreating processes)
    persistent_workers=True,
)

# ===== Training Configuration =====
config.training = dict(
    # Random seed for experiment reproducibility
    seed=42,
    # Output directory for saving checkpoints and logs
    output_dir=OUTPUT_DIR,
    # List of logging/reporting tools (wandb: Weights & Biases, tensorboard: TensorBoard)
    report_to=["wandb", "tensorboard"],
    # W&B project name
    run_project="nextstep",
    # Run name (timestamp will be automatically added in output_dir)
    run_name="nextstep_qwen14b_256px",
    # Checkpoint directory for resuming training
    resume=RESUME_DIR,
    # Whether to automatically resume from the latest checkpoint
    auto_resume=True,
    # Training batch size per device
    per_device_train_batch_size=2,
    # Validation batch size per device
    per_device_val_batch_size=4,
    # Validation steps per device
    per_device_val_steps=8,
    # Learning rate
    learning_rate=1e-4,
    # Weight decay (L2 regularization coefficient)
    weight_decay=0.1,
    # Adam optimizer beta1 parameter
    adam_beta1=0.9,
    # Adam optimizer beta2 parameter
    adam_beta2=0.95,
    # Adam optimizer epsilon parameter (for numerical stability)
    adam_epsilon=1e-8,
    # Maximum gradient norm for gradient clipping
    max_grad_norm=1.0,
    # Maximum number of training steps
    max_steps=500_000,
    # Number of warmup steps for learning rate (LR linearly increases from 0 to learning_rate during warmup)
    warmup_steps=5000,
    # Logging interval (log every N steps)
    logging_steps=1,
    # Data statistics logging interval (log data stats every N steps)
    data_stats_logging_steps=100,
    # Whether to skip the first step (for debugging)
    skip_first_step=True,
    # Checkpoint saving interval (save checkpoint every N steps)
    save_steps=2500,
    # Evaluation interval (run evaluation every N steps)
    eval_steps=2500,
    # Generation parameters for evaluation
    eval_kwargs=dict(
        temperature=1.0,  # Sampling temperature (controls randomness)
        cfg_schedule="constant",  # CFG schedule type (constant: fixed, linear: linear)
        cfg=7.5,  # Classifier-Free Guidance strength
        positive_prompt=None,  # Positive prompt
        negative_prompt=None,  # Negative prompt
    ),
    # Prompt dictionary for evaluation
    eval_prompts=dict(),
    # Number of times to repeat generation for each evaluation prompt
    eval_repeat=4,
    # Learning rate scheduler type (cosine: cosine annealing)
    lr_scheduler_type="cosine",
    # Additional parameters for learning rate scheduler
    lr_scheduler_kwargs=dict(scale_lr=1.0),  # Learning rate scaling factor
    # Whether to use different learning rate scaling for different parameter groups
    use_lr_scale=False,
    # Whether to use Automatic Mixed Precision (AMP)
    amp=True,
    # Whether to use bfloat16 precision (BF16)
    bf16=True,
    # Whether to use float16 precision (FP16, mutually exclusive with bf16)
    fp16=False,
    # Whether to use torch.compile to compile the model (accelerates training)
    torch_compile=False,
    # Whether to enable TF32 (TensorFloat-32, acceleration feature for Ampere architecture GPUs)
    tf32=True,
    # DeepSpeed ZeRO optimization level (0/1/2/3, higher number means more memory optimization)
    deepspeed=2,
    # Whether to enable gradient checkpointing (trades computation time for memory)
    gradient_checkpointing=True,
    # Additional parameters for gradient checkpointing
    gradient_checkpointing_kwargs={"use_reentrant": False},  # Whether to use reentrant checkpoint
    # Gradient accumulation steps
    grad_accumulation_steps=1,
    # Memory monitoring interval (in seconds, -1 to disable)
    memory_monitor_interval=-1,
    # Whether to use PyTorch Profiler for performance analysis
    use_torch_profiler=False,
    # Whether to run data loading only (no training, for testing data pipeline)
    run_data_only=False,
)


# sudo torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 -m configs.nextstep_qwen14b_256px
if __name__ == "__main__":
    from nextstep.engine.train_nextstep_ds import Arguments, main
    from nextstep.lazy_config import LazyLaunch
    from nextstep.lazyrun import setup_config, setup_environ

    # Setup smartrun environment variables (takes effect even when launched with torchrun)
    setup_config()
    setup_environ()

    LazyLaunch(main, Arguments)