import re
from dataclasses import dataclass, field

import torch

from nextstep.utils.import_utils import is_torch_available, is_torch_tf32_available
from nextstep.utils.loguru import logger
from nextstep.utils.misc import LargeInt

@dataclass
class DataArguments:
    datasets: list[str | dict] = field(default_factory=list)
    val_datasets: list[str | dict] = field(default_factory=list)
    use_multi_aspect: bool = False
    image_size: int = 256
    use_super_multi_aspect: bool = False
    buckets_for_super_multi_aspect: list[int] = field(default_factory=list)
    drop_text_prob: float = 0.1
    shift_factor: float = 0.0609  # FIXME: hardcoded, SD3 VAE, # FIXME: Deprecated
    scaling_factor: float = 1.5305  # FIXME: hardcoded, SD3 VAE, # FIXME: Deprecated
    vae_f: int = 8  # FIXME: hardcoded, SD3 VAE, # FIXME: Deprecated
    vae_ch: int = 16  # FIXME: hardcoded, SD3 VAE, # FIXME: Deprecated
    patch_size: int = 1
    data_balance: bool = False
    num_workers: int = 4  # Reduced from 8 to avoid resource conflicts
    pin_memory: bool = False  # Disabled due to container pin memory thread issues
    prefetch_factor: int = 8
    persistent_workers: bool = True

    def __post_init__(self):
        assert self.drop_text_prob >= 0 and self.drop_text_prob <= 1, "`drop_text_prob` should be in the range of [0, 1]."
        assert (
            sum([self.use_multi_aspect, self.use_super_multi_aspect]) <= 1
        ), "Only one of `use_multi_aspect`, `use_super_multi_aspect` can be True."



@dataclass
class ModelArguments:
    model_type: str = "NextStepModel"
    lm_model_name_or_path: str | None = None
    model_name_or_path: str | None = None
    tokenizer_name_or_path: str | None = None
    vae_name_or_path: str | None = None
    reinit_image_head: bool = False
    use_mlp_before_lm_head: bool = False

    freeze_embed_tokens: bool = False
    freeze_lm_model: bool = False
    freeze_lm_head: bool = False
    lm_model_from_scratch: bool = False
    model_max_length: int = 4096
    use_fast_tokenizer: bool = True
    special_tokens_dict: dict = field(default_factory=dict)
    local_files_only: bool = False

    use_token_length_weight: bool = False
    lm_loss_weight: float = 1.0
    image_loss_weight: float = 1.0
    noise_strength: float = 0.0

    fm_head_dim: int = 1536
    fm_head_layers: int = 12
    fm_head_mlp_ratio: float = 1.0
    fm_head_batch_mul: int = 4
    o_attention_bias: bool = False
    attention_use_qk_norm: bool = False
    create_kwargs: dict = field(default_factory=dict)


@dataclass
class TrainingArguments:
    seed: int = 42
    output_dir: str | None = None
    report_to: str | list[str] | None = None
    run_project: str | None = None
    run_name: str | None = None
    resume: str | None = None

    per_device_train_batch_size: int = 16
    per_device_val_batch_size: int = 16
    per_device_val_steps: int = 8
    
    grad_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    max_steps: str | int | LargeInt = LargeInt("100K")
    warmup_steps: int = 1000
    logging_steps: int = 1
    save_steps: int = 1000
    eval_steps: int = 1000
    eval_kwargs: dict = field(default_factory=dict)

    lr_scheduler_type: str = "cosine"
    lr_scheduler_kwargs: dict = field(default_factory=dict)

    amp: bool = True
    bf16: bool = False
    fp16: bool = False

    torch_compile: bool = False
    tf32: bool | None = None

    deepspeed: int | None = None
    enable_flops_profiler: bool = False
    enable_comms_logger: bool = False

    gradient_checkpointing: bool = False
    gradient_checkpointing_kwargs: dict | None = None
    
    skip_first_step: bool = False
    eval_prompts: dict = field(default_factory=dict)
    eval_repeat: int = 4
    use_lr_scale: bool = False
    auto_resume: bool = False

    data_stats_logging_steps: int = 100
    run_data_only: bool = False
    eval_only: bool = False
    skip_eval: bool = False
    skip_val: bool = False

    memory_monitor_interval: int = 1800

    use_torch_profiler: bool = False

    def __post_init__(self):
        assert self.output_dir is not None, "`output_dir` is required."

        assert self.run_project is not None, "run_project shouldn't be None!"
        assert self.run_name is not None, "run_name shouldn't be None!"

        if isinstance(self.report_to, str):
            self.report_to = [self.report_to]

        pattern = r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}"
        match = re.search(pattern, self.output_dir)
        timestamp = match.group() if match else ""
        self.run_name = f"{timestamp}_{self.run_name}"

        self.max_steps = int(LargeInt(self.max_steps))

        if self.per_device_val_batch_size is None:
            self.per_device_val_batch_size = self.per_device_train_batch_size

        assert self.per_device_train_batch_size > 0, "`per_device_train_batch_size` should be > 0."
        assert self.per_device_val_batch_size > 0, "`per_device_val_batch_size` should be > 0."

        if self.fp16 and self.bf16:
            raise ValueError("Cannot use both bf16 and fp16 at the same time.")

        if self.torch_compile:
            if is_torch_available() and is_torch_tf32_available():
                if self.tf32 is None and not self.fp16 or self.bf16:
                    logger.info(
                        "Setting TF32 in CUDA backends to speedup torch compile, you won't see any improvement otherwise."
                    )
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
            else:
                logger.warning(
                    "The speedups for torchdynamo mostly come wih GPU Ampere or higher and which is not detected here."
                )
        if is_torch_available() and self.tf32 is not None:
            if self.tf32:
                if is_torch_tf32_available():
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                else:
                    raise ValueError("--tf32 requires Ampere or a newer GPU arch, cuda>=11 and torch>=1.7")
            else:
                if is_torch_tf32_available():
                    torch.backends.cuda.matmul.allow_tf32 = False
                    torch.backends.cudnn.allow_tf32 = False
                # no need to assert on else

    @property
    def dtype(self) -> torch.dtype:
        dtype = torch.float32
        if self.fp16:
            dtype = torch.float16
        elif self.bf16:
            dtype = torch.bfloat16
        return dtype
