from nextstep.engine.training_args import TrainingArguments
from nextstep.utils.comm import dist_ctx


def get_train_ds_config(
    config: TrainingArguments,
    stage: int,
    offload: bool = False,
    enable_hybrid_engine: bool = False,
    max_out_tokens: int = 512,
    inference_tp_size: int = 1,
    release_inference_cache: bool = False,
    pin_parameters: bool = True,
    tp_gather_partition_size: int = 8,
    enable_flops_profiler: bool = False,
    enable_comms_logger: bool = False,
) -> dict:
    device = "cpu" if offload else "none"
    assert stage in [1, 2, 3], f"Invalid stage: {stage}"
    if stage == 3:
        # some trick is needed when using stage 3, see:
        # https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/dschat/utils/model/model_utils.py
        raise ValueError("Stage 3 is not recommended for now.")

    zero_opt_dict = {
        "stage": stage,
        "overlap_comm": True,
        "contiguous_gradients": False,  # should be False if model is small,
        "sub_group_size": 1e9,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
        "allgather_bucket_size": 5e8 if stage != 3 else 5e4,
        "round_robin_gradients": False,
        # offload
        "offload_param": {"device": device},
        "offload_optimizer": {"device": device},
        # zero3
        "stage3_param_persistence_threshold": 1e4,
        "stage3_max_live_parameters": 3e7,
        "stage3_max_reuse_distance": 3e7,
        "stage3_prefetch_bucket_size": 0,
        "memory_efficient_linear": False,
    }

    deepspeed_config = {
        "train_batch_size": config.per_device_train_batch_size * dist_ctx.world_size * config.grad_accumulation_steps,
        "train_micro_batch_size_per_gpu": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.grad_accumulation_steps,
        "gradient_accumulation_dtype": "fp32",
        "zero_optimization": zero_opt_dict,
        "zero_allow_untested_optimizer": True,
        "zero_force_ds_cpu_optimizer": False,
        "fp16": {
            "enabled": config.fp16,
            "loss_scale": 0,
            "initial_scale_power": 16,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1,
        },
        "bf16": {"enabled": config.bf16},
        "amp": {"enabled": False, "opt_level": "O2"},
        "gradient_clipping": config.max_grad_norm,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "hybrid_engine": {
            "enabled": enable_hybrid_engine,
            "max_out_tokens": max_out_tokens,
            "inference_tp_size": inference_tp_size,
            "release_inference_cache": release_inference_cache,
            "pin_parameters": pin_parameters,
            "tp_gather_partition_size": tp_gather_partition_size,
        },
        "flops_profiler": {
            "enabled": enable_flops_profiler,
            "profile_step": 1,
            "module_depth": -1,
            "top_modules": 1,
            "detailed": True,
        },
        "comms_logger": {
            "enabled": enable_comms_logger,
            "verbose": False,
            "prof_all": True,
            "debug": False,
        },
        "compile": {
            "deepcompile": True,
        },
    }

    return deepspeed_config


def get_inference_ds_config(
    config: TrainingArguments = None,
    stage: int = 1,
    offload: bool = False,
    enable_hybrid_engine: bool = False,
    max_out_tokens: int = 512,
    inference_tp_size: int = 1,
    release_inference_cache: bool = False,
    pin_parameters: bool = True,
    tp_gather_partition_size: int = 8,
    enable_flops_profiler: bool = False,
) -> dict:
    device = "cpu" if offload else "none"
    assert stage in [1, 2, 3], f"Invalid stage: {stage}"
    if stage == 3:
        # some trick is needed when using stage 3, see:
        # https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/dschat/utils/model/model_utils.py
        raise ValueError("Stage 3 is not recommended for now.")

    ds_config = {
        # 1. Basic configuration
        "dtype": "bf16",  # or "bf16", "fp32"
        # "fp16": {
        #     "enabled": True,
        #     "auto_cast": True,
        #     "loss_scale": 0,
        #     "initial_scale_power": 16,
        #     "loss_scale_window": 1000,
        # },
        # # or use bf16
        # "bf16": {
        #     "enabled": True,
        # },
        "tensor_parallel": {
            "tp_size": inference_tp_size,  # Tensor parallel size
        },
        # 2. ZeRO configuration
        "zero": {
            "stage": stage,  # inference usually uses 0
            "stage3_param_persistence_threshold": 1e4,
            "offload_param": {
                "device": "cpu",  # optional "cpu" or "nvme"
                "pin_memory": True,
            },
        },
        # 3. Memory optimization
        "replace_with_kernel_inject": True,
        # "replace_method": "auto",
        # 4. Batch processing configuration
        "max_out_tokens": 1024,
        "min_out_tokens": 1,
        # # 5. Inference-specific configuration
        # "injection_policy": {
        #     "inf_block": "auto",  # or specify specific module
        #     "attention": "auto",
        # },
    }
    return ds_config
