import asyncio
import gc
import math
import os
import threading
from dataclasses import asdict, dataclass, field

import deepspeed
import deepspeed.comm
import megfile
import torch
import torch.distributed as dist
import torchvision.transforms as transforms
import wandb
from deepspeed.runtime.engine import DeepSpeedEngine
from diffusers import AutoencoderKL
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

from nextstep.datasets.mixed_dataset import MixedDataloader, MixedDataset
from nextstep.engine.training_args import DataArguments, ModelArguments, TrainingArguments
from nextstep.lazy_config import LazyArguments, LazyLaunch
from nextstep.model_zoos import MODEL_ZOOS
from nextstep.models.nextstep.tokenization_nextstep import (
    DEFAULT_BOI_TOKEN,
    DEFAULT_EOI_TOKEN,
    DEFAULT_IMAGE_AREA_TOKEN,
    DEFAULT_IMAGE_PLACEHOLDER_TOKEN,
    DEFAULT_PAD_TOKEN,
)
from nextstep.models.nextstep.aspect_ratio import (
    HW_ASPECT_RATIOS,
    ar2str,
    center_crop_arr,
    center_crop_arr_with_ar,
    center_crop_arr_with_buckets,
    get_ar_base,
)
from nextstep.models.nextstep.modeling_nextstep import NextStepConfig, NextStepModel, NextStepOutputWithPast
from nextstep.models.nextstep.modeling_nextstep_vae import AutoencoderKL
from nextstep.utils.comm import dist_ctx
from nextstep.utils.compile_utils import compile_manager
from nextstep.utils.deepspeed_utils import get_train_ds_config
from nextstep.utils.image_utils import to_pil
from nextstep.utils.loguru import logger
from nextstep.utils.mem_utils import PeriodicMemoryMonitor
from nextstep.utils.misc import LargeInt, pretty_format
from nextstep.utils.optim_utils import get_decay_parameter_names, get_grouped_parameters, get_num_grouped_parameters
from nextstep.utils.scheduler_utils import get_scheduler
from nextstep.utils.timer import TimerManager, format_time
from nextstep.utils.torch_profiler import TorchProfiler
from nextstep.utils.general import set_numa_affinity


@dataclass
class Arguments(LazyArguments):
    model: ModelArguments = field(default_factory=ModelArguments)
    data: DataArguments = field(default_factory=DataArguments)
    training: TrainingArguments = field(default_factory=TrainingArguments)


def check_training_data(
    step: int,
    config: Arguments,
    tokenizer: PreTrainedTokenizer,
    vae: AutoencoderKL,
    data: dict,
    image_samples_list=None,
):
    """decode input_ids to text, and decode pixel_values to image."""
    input_ids = data["input_ids"]
    pixel_values = data["pixel_values"]
    image_filtered_idx = data["image_filtered_idx"] if "image_filtered_idx" in data else ""

    # 1. define the save folder
    save_folder = os.path.join(config.training.output_dir, "decoded_output", f"steps{step}_rank{dist_ctx.rank}")
    logger.info(f"Save decoded results to: {save_folder}")
    os.makedirs(save_folder, exist_ok=True)

    # 2. decode input_ids and build sample information
    try:
        responses = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
    except Exception as e:
        logger.error(f"Error when decoding input_ids: {e}")
        return
    try:
        image_samples_list = [url.url_index for url in image_samples_list]
    except Exception as e:
        image_samples_list = []

    samples = []
    image_count = 0
    for i, response in enumerate(responses):
        boi_count = response.count(DEFAULT_BOI_TOKEN)
        if boi_count > 0:
            clean_text = response
            clean_text = clean_text.replace(DEFAULT_BOI_TOKEN, "[IMAGE]")
            clean_text = clean_text.replace(DEFAULT_EOI_TOKEN, "")
            clean_text = clean_text.replace(DEFAULT_IMAGE_PLACEHOLDER_TOKEN, "")
            clean_text = clean_text.replace(DEFAULT_PAD_TOKEN, "")
            if tokenizer.bos_token is not None:
                clean_text = clean_text.replace(tokenizer.bos_token, f"\n\n{tokenizer.bos_token}")
            clean_text = clean_text.replace(tokenizer.eos_token, f"{tokenizer.eos_token}\n\n")

            samples.append(
                {
                    "text": clean_text
                    + "\n\nimage_filtered_idx: "
                    + str(image_filtered_idx[image_count : image_count + boi_count])
                    + "\n\n"
                    + str(image_samples_list),
                    "image_ids": list(range(image_count, image_count + boi_count)),
                }
            )
            image_count += boi_count
    del responses

    # 3. build the mapping from img_id to pixel_values location
    if pixel_values is None:
        logger.error("No pixel_values provided for decoding images")
        # if no image data but text marked images, only save text
        pixel_values_available = False
    else:
        pixel_values_available = True
        image_location_map = {}
        current_img_idx = 0
        try:
            if isinstance(pixel_values, list):
                for source_idx, pv_batch in enumerate(pixel_values):
                    if pv_batch is None:
                        continue  # Skip if an element in the list is None
                    for inner_idx in range(pv_batch.shape[0]):
                        # store enough information to retrieve and decode later
                        image_location_map[current_img_idx] = {
                            "type": "list",
                            "source_idx": source_idx,
                            "inner_idx": inner_idx,
                        }
                        current_img_idx += 1
            elif isinstance(pixel_values, torch.Tensor):
                for idx in range(pixel_values.shape[0]):
                    image_location_map[current_img_idx] = {
                        "type": "tensor",
                        "source_idx": idx,
                    }  # Index within the single tensor
                    current_img_idx += 1
            else:
                logger.error(f"Unknown pixel_values type: {type(pixel_values)}")
                pixel_values_available = False

            if current_img_idx != image_count and pixel_values_available:
                # This may indicate a problem with data preparation or batching
                logger.warning(
                    f"The number of mapped images ({current_img_idx}) does not match the number of text-marked images ({image_count})."
                )

        except Exception as e:
            logger.error(f"Error when building image location mapping: {e}")
            pixel_values_available = False

    # 4. save samples (text and decoded images)
    saved_samples_count = 0
    total_images_saved = 0
    for idx, sample in enumerate(samples):
        sample_folder = os.path.join(save_folder, f"sample_{idx}")
        os.makedirs(sample_folder, exist_ok=True)

        # save text
        text_path = os.path.join(sample_folder, "text.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(sample["text"])

        images_saved_in_sample = 0
        if pixel_values_available:
            for i, img_id in enumerate(sample["image_ids"]):
                loc = image_location_map.get(img_id)
                if loc:
                    try:
                        # a. extract the corresponding latent tensor
                        if loc["type"] == "list":
                            pv_tensor_latent = pixel_values[loc["source_idx"]][loc["inner_idx"] : loc["inner_idx"] + 1]
                        elif loc["type"] == "tensor":
                            pv_tensor_latent = pixel_values[loc["source_idx"] : loc["source_idx"] + 1]
                        else:
                            # Should not happen if map build is correct
                            continue

                        # b. reverse scaling/shifting
                        pv_tensor_scaled = (pv_tensor_latent / config.data.scaling_factor) + config.data.shift_factor

                        # c. decode with VAE
                        # Ensure tensor is on the same device as VAE and correct dtype
                        pv_tensor_scaled = pv_tensor_scaled.to(device=vae.device, dtype=vae.dtype)
                        # Use torch.no_grad() for inference
                        with torch.no_grad():
                            decoded_image_tensor = vae.decode(pv_tensor_scaled).sample

                        # d. convert to PIL Image
                        # Move to CPU, convert dtype, take the single image [0], convert
                        current_pil_image = to_pil(decoded_image_tensor.detach().cpu().to(torch.float32)[0], "11")

                        # e. save image
                        img_path = os.path.join(sample_folder, f"image_{i}.png")
                        current_pil_image.save(img_path)
                        images_saved_in_sample += 1
                        total_images_saved += 1

                        # clean up temporary variables
                        del pv_tensor_latent
                        del pv_tensor_scaled
                        del decoded_image_tensor
                        del current_pil_image

                    except Exception as e:
                        logger.error(f"Error when saving image {i} (img_id {img_id}) of sample {idx}: {e}")
                else:
                    logger.warning(f"Sample {idx}: cannot find location information for img_id {img_id}, skipping image {i}")

        # record sample saving status
        if images_saved_in_sample > 0:
            saved_samples_count += 1
            logger.debug(f"Sample {idx}: saved text and {images_saved_in_sample} images")
        elif len(sample["image_ids"]) > 0 and not pixel_values_available:
            logger.warning(
                f"Sample {idx}: text marked images, but cannot decode/save images (pixel_values unavailable or mapping failed)"
            )
        elif len(sample["image_ids"]) == 0:
            logger.debug(f"Sample {idx}: only contains text, saved")
        else:  # there are image marked, but all failed to save
            logger.warning(
                f"Sample {idx}: tried to save {len(sample['image_ids'])} images, but all failed. Check above errors."
            )

    # clean up after completion
    del samples
    if pixel_values_available:
        del image_location_map
    torch.cuda.empty_cache()

    logger.info(
        f"[Decoding Step {step}/{config.training.max_steps}] Successfully saved text of {saved_samples_count} samples and {total_images_saved} images."
    )


def get_mean_and_max(value: float | torch.Tensor) -> tuple[float, float]:
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value, dtype=torch.float32, device="cuda")

    # clone avoid in-place modification
    value_tensor = value.clone().detach()

    # use all_reduce(max) and all_reduce(sum)
    mean_tensor = value_tensor.clone()
    max_tensor = value_tensor.clone()

    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(mean_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(max_tensor, op=dist.ReduceOp.MAX)

        world_size = dist.get_world_size()
        value_mean = mean_tensor.item() / world_size
        value_max = max_tensor.item()
    else:
        value_mean = value_tensor.item()
        value_max = value_tensor.item()

    return value_mean, value_max


def preprocess_pixel_values(
    data: dict,
    vae: AutoencoderKL,
    config: Arguments,
) -> dict:
    """
    Unified VAE preprocessing function for both training and validation.
    Encodes raw RGB images to latents if needed, otherwise just moves to device and converts dtype.
    
    Args:
        data: Dictionary containing input data, will be modified in-place
        vae: VAE model for encoding images
        config: Configuration object containing data preprocessing parameters
    
    Returns:
        Modified data dictionary with preprocessed pixel_values
    """
    for k, v in data.items():
        if k == "pixel_values" and data[k] is not None:
            if isinstance(data[k], list):
                pixel_values = []
                for pixel_value in data[k]:
                    pixel_value = pixel_value.to(dist_ctx.device, non_blocking=True)
                    # Only encode if it's raw RGB image (shape[1] == 3), not already encoded latent
                    if pixel_value.shape[1] == 3:
                        posterior = vae.encode(pixel_value.to(vae.dtype)).latent_dist
                        pixel_value = (posterior.sample() - config.data.shift_factor) * config.data.scaling_factor
                        # Release intermediate variables from VAE encoding
                        del posterior
                    pixel_value = pixel_value.to(config.training.dtype)
                    pixel_values.append(pixel_value)
                data[k] = pixel_values
            elif data[k].shape[1] == 3:
                pixel_values = data[k].to(dist_ctx.device, non_blocking=True)
                posterior = vae.encode(pixel_values.to(vae.dtype)).latent_dist
                pixel_values = (posterior.sample() - config.data.shift_factor) * config.data.scaling_factor
                # Release intermediate variables from VAE encoding
                del posterior
                pixel_values = pixel_values.to(config.training.dtype)
                data[k] = pixel_values
            else:
                # Already encoded latent, just move to device and convert dtype
                data[k] = data[k].to(dist_ctx.device, non_blocking=True).to(config.training.dtype)
        else:
            data[k] = v.to(dist_ctx.device, non_blocking=True) if v is not None else None
    return data


# fmt: off
def training_step(
    step: int,
    config: Arguments,
    vae: AutoencoderKL,
    model: DeepSpeedEngine,
    dataloader_train: MixedDataloader,
    timer_manager: TimerManager,
    tb_writer: SummaryWriter | None = None,
    tokenizer: PreTrainedTokenizer | None = None,
    start_step: int = 0,
):
    timer_manager.beat("training_step", "start")
    model.train()

    total_loss = torch.tensor(0.0, device=dist_ctx.device)
    total_lm_loss = torch.tensor(0.0, device=dist_ctx.device)
    total_image_loss = torch.tensor(0.0, device=dist_ctx.device)
    for micro_step in range(model.gradient_accumulation_steps()):
        timer_manager.beat("before_backward", "start")
        ##############################################################
        # Data
        ##############################################################
        timer_manager.beat("data", "start")
        batch = dataloader_train.next_batch(step)
        data, image_samples_list = batch.batch_data, batch.image_samples_list

        if step < start_step + 5:
            # Add a barrier for the first few steps to synchronize processes
            # This prevents NCCL waiting issues due to different data loading speeds
            torch.cuda.empty_cache()
            logger.info(f"[Warmup Step {step}/{start_step + 5}] Rank {dist_ctx.rank} get corresponding data and wait for all ranks.")
            dist.barrier()
            logger.info(f"[Warmup Step {step}/{start_step + 5}] All ranks get corresponding data and ready to forward.")
        timer_manager.beat("data", "end")

        ##############################################################
        # VAE Preprocessing
        ##############################################################
        timer_manager.beat("vae", "start")
        data = preprocess_pixel_values(data, vae, config)

        if step % 2500 == 0:
            check_training_data(step, config, tokenizer=tokenizer, vae=vae, data=data, image_samples_list=image_samples_list)

        if step < start_step + 5:
            logger.info(f"[Warmup Step {step}/{start_step + 5}] VAE Preprocessing done")
        timer_manager.beat("vae", "end")

        ##############################################################
        # Forward
        ##############################################################
        timer_manager.beat("forward", "start")
        if not config.training.run_data_only:
            outputs = model(**data)
        else:
            outputs = NextStepOutputWithPast(
                loss=torch.tensor(0.0, device=dist_ctx.device),
                lm_loss=torch.tensor(0.0, device=dist_ctx.device),
                image_loss=torch.tensor(0.0, device=dist_ctx.device),
            )
        loss: torch.Tensor = outputs.loss
        total_loss += loss
        total_lm_loss += outputs.lm_loss
        total_image_loss += outputs.image_loss
        if step < start_step + 5:
            logger.info(f"[Warmup Step {step}/{start_step + 5}] Forward done")
        timer_manager.beat("forward", "end")
        timer_manager.beat("before_backward", "end")

        ##############################################################
        # Backward
        ##############################################################
        timer_manager.beat("backward", "start")
        if not config.training.run_data_only:
            model.backward(loss)
            # DeepSpeed handles grad_norm, optim.step and optim.zero_grad internally
            model.step()
            grad_norm = model.get_global_grad_norm()
        else:
            grad_norm = torch.tensor(0.0, device=dist_ctx.device)
        if step < start_step + 5:
            logger.info(f"[Warmup Step {step}/{start_step + 5}] Backward done")
        timer_manager.beat("backward", "end")

    timer_manager.beat("training_step", "end")
    timer_manager.beat("training", "end")


    timer_manager.beat("log", "start")

    iter_time, max_iter_time = get_mean_and_max(timer_manager.training_step.value)
    data_time, max_data_time = get_mean_and_max(timer_manager.data.acc_value(config.training.grad_accumulation_steps))
    vae_time, max_vae_time = get_mean_and_max(timer_manager.vae.acc_value(config.training.grad_accumulation_steps))
    forward_time, max_forward_time = get_mean_and_max(timer_manager.forward.acc_value(config.training.grad_accumulation_steps))
    backward_time, max_backward_time = get_mean_and_max(timer_manager.backward.acc_value(config.training.grad_accumulation_steps))

    _, max_before_backward_time = get_mean_and_max(timer_manager.before_backward.acc_value(config.training.grad_accumulation_steps))
    # backward wait time = max_before_backward_time - before_backward_time
    # actually backward time = backward_time - backward_wait_time
    backward_wait_time = max_before_backward_time - timer_manager.before_backward.acc_value(config.training.grad_accumulation_steps)
    act_backward_time = timer_manager.backward.acc_value(config.training.grad_accumulation_steps) - backward_wait_time
    act_backward_time, max_act_backward_time = get_mean_and_max(act_backward_time)

    progress_infos = dict(
        iter=iter_time,
        max_iter=max_iter_time,
        data=data_time,
        max_data=max_data_time,
        vae=vae_time,
        max_vae=max_vae_time,
        forward=forward_time,
        max_forward=max_forward_time,
        backward=backward_time,
        max_backward=max_backward_time,
        act_backward=act_backward_time,
        max_act_backward=max_act_backward_time,
    )

    eta_str = format_time(timer_manager.training_step.moving_avg * (config.training.max_steps - step))
    progress_str = f"{step}/{config.training.max_steps}, {format_time(timer_manager.training.value)}<{eta_str}"
    for k, v in progress_infos.items():
        progress_str = progress_str + f", {v:.4f} s/{k}"

    lr_str = f"{model.get_lr()[0]:.8f}"
    loss_str = f"{dist_ctx.all_reduce_mean(total_loss / config.training.grad_accumulation_steps):.4f}"
    lm_loss_str = f"{dist_ctx.all_reduce_mean(total_lm_loss / config.training.grad_accumulation_steps):.4f}"
    image_loss_str = f"{dist_ctx.all_reduce_mean(total_image_loss / config.training.grad_accumulation_steps):.4f}"
    grad_norm_str = f"{dist_ctx.all_reduce_mean(grad_norm / config.training.grad_accumulation_steps):.4f}"
    max_gpu_mem_str = f"{torch.cuda.max_memory_allocated() / (1024.0 * 1024.0):.4f}"

    waste_token_num = dist_ctx.all_reduce_sum(data["waste_token_num"])
    bsz, seq_len = data["input_ids"].shape
    waste_token_rate = f"{waste_token_num / (bsz * seq_len * dist_ctx.world_size) * 100:.2f}"
    # waste_token_rate = torch.tensor(0.0, device=dist_ctx.device)

    pv_num = len(data["pixel_values"]) if isinstance(data["pixel_values"], list) else data["pixel_values"].shape[0]

    max_pv_num = torch.tensor(pv_num, device=dist_ctx.device)
    dist.all_reduce(max_pv_num, op=dist.ReduceOp.MAX)
    max_pv_num = max_pv_num.item()

    min_pv_num = torch.tensor(pv_num, device=dist_ctx.device)
    dist.all_reduce(min_pv_num, op=dist.ReduceOp.MIN)
    min_pv_num = min_pv_num.item()

    pv_diff = max_pv_num - min_pv_num

    training_infos = dict(
        lr=lr_str,
        loss=loss_str,
        lm_loss=lm_loss_str,
        image_loss=image_loss_str,
        grad_norm=grad_norm_str,
        waste_token_rate=waste_token_rate,
        max_pv_num=max_pv_num,
        min_pv_num=min_pv_num,
        pv_diff=pv_diff,
        max_gpu_mem=max_gpu_mem_str,
    )
    info_str = ""
    for k, v in training_infos.items():
        info_str = info_str + f", {k}: {v}"

    if step % config.training.data_stats_logging_steps == 0:
        logger.info(dataloader_train.status)

    timer_manager.beat("log", "end")
    _, max_log_time = get_mean_and_max(timer_manager.log.value)
    logger.info(f"[{progress_str}, {max_log_time:.4f} s/max_log_time]" + info_str)

    if dist_ctx.is_main_process:
        if "tensorboard" in config.training.report_to:
            for k, v in training_infos.items():
                tb_writer.add_scalar(f"train/{k}", float(v), step)
            for k, v in progress_infos.items():
                tb_writer.add_scalar(f"time/{k}", float(v), step)
            tb_writer.add_scalar("time/max_log_time", float(max_log_time), step)

        if "wandb" in config.training.report_to:
            wandb.log({f"train/{k}": float(v) for k, v in training_infos.items()}, step=step)
            wandb.log({f"time/{k}": float(v) for k, v in progress_infos.items()}, step=step)
            wandb.log({"time/max_log_time": float(max_log_time)}, step=step)
            # Disabled: sample_loss histogram logging may cause torch.compile to fail
            # for dataset_name, loss_values in sample_loss_dict.items():
            #     wandb.log({f"dataset_loss/{dataset_name}": wandb.Histogram(loss_values)}, step=step)

    torch.cuda.synchronize()
    dist.barrier()
    if config.training.enable_comms_logger:
        deepspeed.comm.log_summary()
# fmt: on


@torch.no_grad()
def val_step(
    step: int,
    config: Arguments,
    vae: AutoencoderKL,
    model: DeepSpeedEngine,
    dataloader_val: MixedDataloader,
    timer_manager: TimerManager,
    tb_writer: SummaryWriter | None = None,
):
    timer_manager.beat("val_step", "start")
    model.eval()
    vae.eval()  # Ensure VAE is also in eval mode

    total_loss = torch.tensor(0.0, device=dist_ctx.device)
    total_lm_loss = torch.tensor(0.0, device=dist_ctx.device)
    total_image_loss = torch.tensor(0.0, device=dist_ctx.device)
    
    # Reset validation iterator so each validation starts from the same sample sequence.
    dataloader_val.reset()
    dataloader_val.init()
    
    # Ensure all processes start validation synchronously to avoid data loading inconsistencies
    dist.barrier()
    
    # Track actual validation steps executed (may be less than n_val_steps if data runs out early)
    actual_val_steps = 0
    n_val_steps = config.training.per_device_val_steps
    
    # Clear cache once before validation starts to release memory that may remain from training
    torch.cuda.empty_cache()
    
    for val_idx in tqdm(range(n_val_steps), desc="Validation Step", total=n_val_steps, disable=not dist_ctx.is_main_process):
        ##############################################################
        # Data
        ##############################################################
        timer_manager.beat("val_data", "start")
        try:
            batch = dataloader_val.next_batch(step)
        except StopIteration:
            # If validation data is exhausted, stop validation early
            logger.warning(f"Validation data exhausted at step {val_idx}/{n_val_steps}, stopping validation early.")
            break
        data = batch.batch_data
        timer_manager.beat("val_data", "end")

        ##############################################################
        # VAE Preprocessing
        ##############################################################
        timer_manager.beat("val_vae", "start")
        data = preprocess_pixel_values(data, vae, config)
        timer_manager.beat("val_vae", "end")

        ##############################################################
        # Forward
        ##############################################################
        timer_manager.beat("val_forward", "start")
        outputs = model(force_train_mode=True, **data)
        loss: torch.Tensor = outputs.loss
        total_loss += loss
        total_lm_loss += outputs.lm_loss
        total_image_loss += outputs.image_loss
        timer_manager.beat("val_forward", "end")
        
        # Clean up intermediate variables and CUDA cache to prevent memory accumulation
        # Loss values have been extracted to total_* variables above
        del outputs
        del data
        # Synchronize GPU operations to ensure all computations complete before clearing cache
        torch.cuda.synchronize()
        if val_idx < n_val_steps - 1:  # No need to clean up on the last step
            torch.cuda.empty_cache()
        
        actual_val_steps += 1

    timer_manager.beat("val_step", "end")
    
    # Use actual executed steps to calculate average (if data runs out early, actual_val_steps may be less than n_val_steps)
    actual_val_steps = max(1, actual_val_steps)  # Ensure at least 1 to avoid division by zero error

    val_time, max_val_time = get_mean_and_max(timer_manager.val_step.value)
    val_data_time, max_val_data_time = get_mean_and_max(timer_manager.val_data.acc_value(actual_val_steps))
    val_vae_time, max_val_vae_time = get_mean_and_max(timer_manager.val_vae.acc_value(actual_val_steps))
    val_forward_time, max_val_forward_time = get_mean_and_max(timer_manager.val_forward.acc_value(actual_val_steps))
    
    progress_infos = dict(
        val=val_time,
        max_val=max_val_time,
        val_data=val_data_time,
        max_val_data=max_val_data_time,
        val_vae=val_vae_time,
        max_val_vae=max_val_vae_time,
        val_forward=val_forward_time,
        max_val_forward=max_val_forward_time,
    )

    progress_str = f"Validation Step {step}/{config.training.max_steps}"
    for k, v in progress_infos.items():
        progress_str = progress_str + f", {v:.4f} s/{k}"
    progress_str = f"[{progress_str}]"

    loss_str = f"{dist_ctx.all_reduce_mean(total_loss / actual_val_steps):.4f}"
    lm_loss_str = f"{dist_ctx.all_reduce_mean(total_lm_loss / actual_val_steps):.4f}"
    image_loss_str = f"{dist_ctx.all_reduce_mean(total_image_loss / actual_val_steps):.4f}"
    max_gpu_mem_str = f"{torch.cuda.max_memory_allocated() / (1024.0 * 1024.0):.4f}"

    val_infos = dict(
        loss=loss_str,
        lm_loss=lm_loss_str,
        image_loss=image_loss_str,
        max_gpu_mem=max_gpu_mem_str,
    )
    info_str = ""
    for k, v in val_infos.items():
        info_str = info_str + f", {k}: {v}"

    logger.info(progress_str + info_str)
    
    # Log performance bottleneck warnings
    if dist_ctx.is_main_process:
        if val_data_time > val_forward_time * 0.5:
            logger.warning(f"Validation data loading is slow: {val_data_time:.4f}s vs forward {val_forward_time:.4f}s. Consider increasing num_workers or optimizing data loading.")
        if val_vae_time > val_forward_time * 0.5:
            logger.warning(f"Validation VAE encoding is slow: {val_vae_time:.4f}s vs forward {val_forward_time:.4f}s. This is expected but may be a bottleneck.")

    if dist_ctx.is_main_process:
        if "tensorboard" in config.training.report_to:
            for k, v in val_infos.items():
                tb_writer.add_scalar(f"val/{k}", float(v), step)
            for k, v in progress_infos.items():
                tb_writer.add_scalar(f"time/{k}", float(v), step)

        if "wandb" in config.training.report_to:
            wandb.log({f"val/{k}": float(v) for k, v in val_infos.items()}, step=step)
            wandb.log({f"time/{k}": float(v) for k, v in progress_infos.items()}, step=step)

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()
    dist.barrier()


def save_checkpoint(
    step: int,
    config: Arguments,
    tokenizer: PreTrainedTokenizer,
    model: DeepSpeedEngine,
    dataloader: MixedDataloader,
):
    logger.info(f"Save checkpoint at step {step} ...")

    checkpoint_name = f"checkpoint-{step}"
    checkpoint_folder = os.path.join(config.training.output_dir, "checkpoints", checkpoint_name)
    megfile.smart_makedirs(checkpoint_folder, exist_ok=True)

    dataloader.save_state_dict(checkpoint_folder)

    client_state = {"step": step}
    model.save_checkpoint(save_dir=checkpoint_folder, tag="deepspeed_shards", client_state=client_state)

    if dist_ctx.is_main_process:
        model.module.config.save_pretrained(checkpoint_folder)
        """
        BUG: RuntimeError: The weights trying to be saved contained shared tensors [{'lm_head.weight', 'embed_tokens.weight'}]
        that are mismatching the transformers base configuration. Try saving using `safe_serialization=False` or remove this tensor sharing.
        """
        tokenizer.save_pretrained(checkpoint_folder)

    # Optional: convert zero checkpoint to fp32 state dict (may cause nccl timeout error)
    # from nextstep.deepspeed.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
    # if dist_ctx.is_main_process:
    #     logger.info("Convert zero checkpoint to fp32 state dict ...")
    #     convert_zero_checkpoint_to_fp32_state_dict(checkpoint_folder, checkpoint_folder)

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()
    dist.barrier()


def extract_checkpoint_step(checkpoint_path):
    import re

    match = re.search(r"checkpoint-(\d+)", checkpoint_path)
    if match:
        return int(match.group(1))
    return 0  # Return 0 as default if step number cannot be extracted


def auto_resume(resume_dir):
    if len([d for d in os.listdir(resume_dir) if os.path.isdir(os.path.join(resume_dir, d))]) > 20:
        raise ValueError(f"Auto-resume has been executed more than 20 times in {resume_dir}, please manually specify the resume checkpoint")
    checkpoint_dirs = []

    for cur_dir in megfile.smart_glob(os.path.join(resume_dir, "**/checkpoints*/checkpoint-*"), recursive=True):
        if os.path.isdir(cur_dir) and "checkpoint-" in cur_dir:
            checkpoint_dirs.append(cur_dir)

    if checkpoint_dirs:
        # Sort by folder name, usually checkpoint folder names contain step information
        checkpoint_dirs.sort(key=lambda x: (extract_checkpoint_step(x), os.path.getmtime(x)))

        # Count files in each checkpoint directory
        checkpoint_file_counts = [len(os.listdir(checkpoint_dir)) for checkpoint_dir in checkpoint_dirs]
        # Find the last checkpoint with the maximum file count
        # max_count = max(checkpoint_file_counts)
        # max_idx = [i for i, count in enumerate(checkpoint_file_counts) if count == max_count][-1]

        # Use the latest checkpoint folder
        resume_dir = checkpoint_dirs[-1]

        logger.info(f"Found latest checkpoint: {resume_dir}")
    else:
        logger.error(
            f"No folders starting with 'checkpoint' found in {resume_dir}, please manually specify the resume checkpoint. Auto-resume failed!"
        )
        resume_dir = None

    return resume_dir

# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 -m configs.nextstep.nextstep_qwen14b_256px
def main(config: Arguments):
        
    torch.cuda.set_device(dist_ctx.local_rank)
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    deepspeed.init_distributed()
    dist.barrier()

    try:
        set_numa_affinity()
    except Exception as e:
        logger.warning(f"Failed to set NUMA affinity: {e}")
        logger.warning("Continuing without NUMA affinity optimization...")
    
    
    logger.info(
        f"Initializing main process (ip: {os.environ.get('MASTER_ADDR', '127.0.0.1')}, pid: {os.getpid()}) "
        f"You can use `py-spy top --pid {os.getpid()} profile` to profile the process."
    )
    
    # from nextstep.utils.debug import debug
    # debug(rank=0, stop=True)

    ###########################################################################
    # Tokenizer
    ###########################################################################
    tokenizer_name_or_path = (
        config.model.model_name_or_path if config.model.model_name_or_path is not None else config.model.lm_model_name_or_path
    )
    if config.model.tokenizer_name_or_path is not None:
        tokenizer_name_or_path = config.model.tokenizer_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        local_files_only=config.model.local_files_only,
        model_max_length=config.model.model_max_length,
        padding_side="right",
        use_fast=config.model.use_fast_tokenizer,
    )
    logger.info(
        f"Successfully load ({'fast' if config.model.use_fast_tokenizer else 'slow'}) tokenizer from {tokenizer_name_or_path}, "
        f"with `len(tokenizer)` {len(tokenizer)}, `model_max_length` {config.model.model_max_length} and padding right."
    )

    # add special tokens
    num_added_tokens = 0
    if len(config.model.special_tokens_dict) > 0:
        num_added_tokens = tokenizer.add_special_tokens(config.model.special_tokens_dict)
    if num_added_tokens > 0:
        logger.info(
            f"Successfully add {num_added_tokens} special tokens, which are:\n{pretty_format(config.model.special_tokens_dict)}"
        )
    else:
        logger.info("No special tokens need to be added.")
    
            
    logger.info(f"The final tokenizer is:\n{pretty_format(tokenizer)}")

    image_placeholder_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_PLACEHOLDER_TOKEN)
    boi = tokenizer.convert_tokens_to_ids(DEFAULT_BOI_TOKEN)
    eoi = tokenizer.convert_tokens_to_ids(DEFAULT_EOI_TOKEN)
    pad_token_id = tokenizer.pad_token_id

    ###########################################################################
    # Model
    ###########################################################################
    down_factor = config.data.vae_f * config.data.patch_size
    image_grid_size = round(config.data.image_size / down_factor)
    ar_base = get_ar_base(HW_ASPECT_RATIOS)  # 16

    if config.data.use_multi_aspect:
        hw_aspect_ratios = [
            (round(h * (image_grid_size / ar_base)), round(w * (image_grid_size / ar_base))) for h, w in HW_ASPECT_RATIOS
        ]
    elif config.data.use_super_multi_aspect:
        buckets_for_super_multi_aspect = config.data.buckets_for_super_multi_aspect
        hw_aspect_ratios = []
        for image_size in buckets_for_super_multi_aspect:
            image_grid_size = round(image_size / down_factor)
            hw_aspect_ratios.extend(
                [(round(h * (image_grid_size / ar_base)), round(w * (image_grid_size / ar_base))) for h, w in HW_ASPECT_RATIOS]
            )
    else:  # if not use multi aspect, only keep square aspect ratios
        hw_aspect_ratios = [(image_grid_size, image_grid_size)]

    hw_aspect_ratios_ids = {
        ar2str(h, w): (
            tokenizer(f"{DEFAULT_IMAGE_AREA_TOKEN}{ar2str(h, w)}").input_ids[1:]
            if tokenizer.bos_token is not None
            else tokenizer(f"{DEFAULT_IMAGE_AREA_TOKEN}{ar2str(h, w)}").input_ids
        )
        for h, w in hw_aspect_ratios
    }

    model_config = NextStepConfig.from_pretrained(
        MODEL_ZOOS[
            config.model.lm_model_name_or_path if config.model.model_name_or_path is None else config.model.model_name_or_path
        ],
        attn_implementation="sdpa",
        image_size=round(config.data.image_size / config.data.vae_f),
        patch_size=config.data.patch_size,
        num_channels=config.data.vae_ch,
        hw_aspect_ratios_ids=hw_aspect_ratios_ids,
        image_placeholder_id=image_placeholder_id,
        boi=boi,
        eoi=eoi,
        pad_token_id_added=pad_token_id,
        vae_name_or_path=config.model.vae_name_or_path,
        use_mlp_before_lm_head=config.model.use_mlp_before_lm_head,
        use_token_length_weight=config.model.use_token_length_weight,
        lm_loss_weight=config.model.lm_loss_weight,
        image_loss_weight=config.model.image_loss_weight,
        noise_strength=config.model.noise_strength,
        fm_head_dim=config.model.fm_head_dim,
        fm_head_layers=config.model.fm_head_layers,
        fm_head_mlp_ratio=config.model.fm_head_mlp_ratio,
        fm_head_batch_mul=config.model.fm_head_batch_mul,
        o_attention_bias=config.model.o_attention_bias,
        attention_use_qk_norm=config.model.attention_use_qk_norm,
    )
    
    model_type = config.model.model_type
    model_init_dtype = torch.bfloat16
    logger.info(f"Model init dtype: {model_init_dtype}")

    prev_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(model_init_dtype)
    try:
        if config.model.model_name_or_path is None:
            model = NextStepModel(model_config)
            # Manually load pretrained parameters
            if config.model.lm_model_name_or_path is not None and not config.model.lm_model_from_scratch:
                model.load_llama_model(config.model.lm_model_name_or_path, config.model.local_files_only)
        else:
            try:
                logger.info(f"Loading model from {config.model.model_name_or_path} ...")
                model = NextStepModel.from_pretrained(
                    MODEL_ZOOS[config.model.model_name_or_path],
                    config=model_config,
                    local_files_only=True,
                    low_cpu_mem_usage=True,
                    torch_dtype=model_init_dtype,
                )
            except Exception as e:
                logger.warning(f"Failed to load model from {config.model.model_name_or_path}, error: {e}")
                logger.warning("Try to load model with ignore_mismatched_sizes=False ...")
                model = NextStepModel.from_pretrained(
                    MODEL_ZOOS[config.model.model_name_or_path],
                    config=model_config,
                    ignore_mismatched_sizes=False,
                    local_files_only=True,
                    low_cpu_mem_usage=True,
                    torch_dtype=model_init_dtype,
                )
    finally:
        torch.set_default_dtype(prev_default_dtype)

    if config.model.reinit_image_head:
        model.reinit_image_head()
        logger.info("Image head is reinitialized.")
    # Check tokenizer and model vocabulary size
    if len(tokenizer) > model.config.vocab_size:
        logger.info(
            f"The tokenizer vocabulary size {len(tokenizer)} is larger than the model vocabulary size {model.config.vocab_size}. "
            f"Resizing token embedding of model..."
        )
        model.resize_token_embeddings(len(tokenizer))
        # keep the size of `model.vocab_size` consistent with `tokenizer` and `model.vocab_size`
        model.vocab_size = len(tokenizer)
    elif len(tokenizer) < model.config.vocab_size:
        logger.warning(
            f"The tokenizer vocabulary size {len(tokenizer)} is smaller than the model vocabulary size {model.config.vocab_size}. "
            f"Carefully check the configuration to avoid potential issues."
        )
    logger.info(f"Now, the tokenizer and model vocabulary sizes are both {len(tokenizer)}.")

    if config.training.resume is not None and os.path.isdir(config.training.resume):
        resume_dir = config.training.resume
        if config.training.auto_resume and os.path.isdir(resume_dir) and not ("checkpoint-" in resume_dir):
            # Traverse all folders starting with 'checkpoint' under resume_dir
            resume_dir = auto_resume(resume_dir)    
        config.training.resume = resume_dir if resume_dir and "checkpoint-" in resume_dir else None
        
        if resume_dir is not None:
            logger.info(f"Resuming tokenizer from {resume_dir}")
            tokenizer = AutoTokenizer.from_pretrained(resume_dir, local_files_only=True)

    if config.model.freeze_embed_tokens:
        model.embed_tokens.requires_grad_(False)
        model.embed_tokens.eval()
        logger.info("Embed tokens are frozen.")

    if config.model.freeze_lm_model:
        model.layers.requires_grad_(False)
        model.layers.eval()
        model.norm.requires_grad_(False)
        model.norm.eval()
        logger.info("LM model is frozen.")

    if config.model.freeze_lm_head:
        model.lm_head.requires_grad_(False)
        model.lm_head.eval()
        logger.info("LM head is frozen.")

        if config.model.use_mlp_before_lm_head:
            model.before_lm_head.requires_grad_(False)
            model.before_lm_head.eval()
            logger.info("MLP before LM head is frozen.")

    logger.info("Model = %s" % str(model))
    logger.info(f"Number of trainable parameters: {model.trainable_params}")

    if config.training.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=config.training.gradient_checkpointing_kwargs)

    ###########################################################################
    # Data
    ###########################################################################
    # For evaluation and training preprocessing
    vae = AutoencoderKL.from_pretrained(config.model.vae_name_or_path).to(device=dist_ctx.device,dtype=model_init_dtype)
    vae.requires_grad_(False)
    vae.eval()
    
    # fmt: off
    def get_image_transform(crop=True):
        """
        Unified image transform function for both training and validation.
        This ensures consistent data preprocessing between train and val sets.
        
        Args:
            crop: Whether to crop the image. Default True for both train and val.
        
        Returns:
            A composed transform pipeline.
        """
        transform_list = []
        if config.data.use_multi_aspect:
            transform_list.append(
                transforms.Lambda(lambda pil_image: center_crop_arr_with_ar(pil_image, config.data.image_size, ars=HW_ASPECT_RATIOS, crop=crop))
            )
        elif config.data.use_super_multi_aspect:
            transform_list.append(
                transforms.Lambda(lambda pil_image: center_crop_arr_with_buckets(pil_image, ars=HW_ASPECT_RATIOS, crop=crop, buckets=config.data.buckets_for_super_multi_aspect))
            )
        else:
            transform_list.append(
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, config.data.image_size, crop=crop))
            )
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
        return transforms.Compose(transform_list)
    
    # Use the same transform for both training and validation to ensure data distribution consistency
    train_transform = get_image_transform(crop=True)
    eval_transform = get_image_transform(crop=True)
    # fmt: on
    if not config.training.eval_only:
        dataset_train = MixedDataset(
            data_info_list=config.data.datasets,
            batch_size=config.training.per_device_train_batch_size,
            tokenizer=tokenizer,
            down_factor=down_factor,
            hw_aspect_ratios_ids=hw_aspect_ratios_ids,
            image_seq_len=model.image_seq_len,
            max_len=config.model.model_max_length,
            drop_text_prob=config.data.drop_text_prob,
            shuffle_size=LargeInt("50K"),
            lru_size=128,
            data_balance=config.data.data_balance,
            seed=config.training.seed,
            dataset_kwargs={"post_processing": train_transform},
        )
        dataloader_train = MixedDataloader(
            dataset_train,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            prefetch_factor=config.data.prefetch_factor,
            persistent_workers=config.data.persistent_workers,
            memory_monitor_interval=config.training.memory_monitor_interval,
            data_monitor_interval=30.0,
        )
        logger.info(f"Dataset train:\n{dataset_train}")

    from nextstep.datasets.image_text_wds import ImageTextWDS
    from nextstep.lazy_config.registry import _convert_target_to_string
    
    samples_val = sum([data_info["samples"] for data_info in config.data.val_datasets])
    if config.data.drop_text_prob > 0:
        logger.info(f"Force val drop_text_prob to 0.0 (train drop_text_prob={config.data.drop_text_prob}).")

    
    # Validation data should not be shuffled, use a very large shuffle_size to minimize shuffle impact
    # When shuffle_size >= total samples, all samples will be in one shuffle buffer, but there will still be some shuffling
    # Use a very large value to minimize shuffle impact as much as possible
    dataset_val = MixedDataset(
        data_info_list=config.data.val_datasets,
        batch_size=config.training.per_device_val_batch_size,
        tokenizer=tokenizer,
        down_factor=down_factor,
        hw_aspect_ratios_ids=hw_aspect_ratios_ids,
        image_seq_len=model.image_seq_len,
        max_len=config.model.model_max_length,
        drop_text_prob=0.0,
        shuffle_size=LargeInt("50K"),  # Use a very large value to disable shuffling (validation data should maintain fixed order)
        lru_size=16,
        seed=config.training.seed,
        dataset_kwargs={"post_processing": eval_transform},
    )
    dataloader_val = MixedDataloader(
        dataset_val,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        prefetch_factor=config.data.prefetch_factor,
        persistent_workers=config.data.persistent_workers,
        memory_monitor_interval=config.training.memory_monitor_interval,
        data_monitor_interval=30.0,
    )

    # Eagerly create DataLoader worker processes BEFORE starting any background
    # threads (wandb/tensorboard/monitors). Forking after threads start can
    # crash/hang DataLoader even when num_workers=1.
    if not config.training.eval_only:
        dataloader_train.init()
    dataloader_val.init()

    
    ###########################################################################
    # Optimizer
    ###########################################################################
    if config.training.use_lr_scale:
        def lr_scale_func(name: str):
            if "image" not in name:
                llm_base_lr = 1e-5
                return llm_base_lr / config.training.learning_rate
            return 1
    else:
        def lr_scale_func(name: str):
            return 1

    decay_parameter_names = get_decay_parameter_names(model)
    grouped_parameters = get_grouped_parameters(
        model,
        config.training.learning_rate,
        config.training.weight_decay,
        lr_scale_func,
        decay_parameter_names,
    )
    optimizer = torch.optim.AdamW(
        grouped_parameters,
        lr=config.training.learning_rate,
        betas=(config.training.adam_beta1, config.training.adam_beta2),
        eps=config.training.adam_epsilon,
        weight_decay=config.training.weight_decay,
    )
    logger.info(optimizer)
    logger.info(f"Trainable parameters in rank {dist_ctx.rank}: {get_num_grouped_parameters(grouped_parameters)}")

    ###########################################################################
    # Scheduler
    ###########################################################################
    lr_scheduler = get_scheduler(
        config.training.lr_scheduler_type,
        optimizer,
        num_warmup_steps=config.training.warmup_steps,
        num_training_steps=config.training.max_steps,
        **config.training.lr_scheduler_kwargs,
    )

    
    ###########################################################################
    # DeepSpeed Prepare
    ###########################################################################
    ds_config = get_train_ds_config(
        config.training,
        config.training.deepspeed,
        enable_flops_profiler=config.training.enable_flops_profiler,
        enable_comms_logger=config.training.enable_comms_logger,
    )
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        config=ds_config,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dist_init_required=False,
    )
    model: DeepSpeedEngine = model
    logger.info(
        f"Trainable parameters in rank {dist_ctx.rank} after deepspeed init: {get_num_grouped_parameters(grouped_parameters)}"
    )
        
        
    compile_manager.set_compile_enabled(config.training.torch_compile)
    ###########################################################################
    # Resume
    ###########################################################################
    start_step = 0
    skip_once_in_resume = False
    
    if config.training.resume is not None and os.path.isdir(config.training.resume):
        logger.info(f"Resuming model, optimizer, scheduler and dataloader_state_dict from {config.training.resume}")

        dataloader_train.load_state_dict(config.training.resume)

        def custom_load_fn(src, dst):
            modified_state_dict = {}
            for k, v in src.items():
                if k.startswith('_orig_mod.'):
                    new_key = k[len('_orig_mod.'):]
                    modified_state_dict[new_key] = v
                else:
                    modified_state_dict[k] = v
            dst.load_state_dict(modified_state_dict, strict=False)

        _, client_state = model.load_checkpoint(
            config.training.resume,
            custom_load_fn=custom_load_fn,
        )
        start_step = client_state["step"]
        skip_once_in_resume = True

        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()
        dist.barrier()
    elif config.training.resume is not None and not os.path.exists(config.training.resume):
        logger.error(f"Resume directory {config.training.resume} does not exist, please check the path, we will start from scratch")
    else:
        logger.info(f"No resume directory provided, starting from scratch")

    ###########################################################################
    # Training
    ###########################################################################
    tb_writer = None
    if dist_ctx.is_main_process:
        if "tensorboard" in config.training.report_to:
            tb_writer = SummaryWriter(os.path.join(config.training.output_dir, "tensorboard"))
        if "wandb" in config.training.report_to:
            wandb.init(
                mode=os.environ.get("WANDB_MODE", "offline"),
                project=config.training.run_project,
                name=config.training.run_name,
                config=asdict(config),
                dir=config.training.output_dir,
            )

    memory_monitor = PeriodicMemoryMonitor(name="main_process", interval=config.training.memory_monitor_interval)
    memory_monitor.start()

    logger.info(f"Start training from step {start_step}, total {config.training.max_steps} steps")
    timer_manager = TimerManager()
    timer_manager.beat("training", "start")

    if config.training.use_torch_profiler:
        profiler = TorchProfiler(os.path.join(config.training.output_dir, "torch_profiler"))
        profiler.start()

    try:
        # Last step is used for saving checkpoint and evaluation
        for step in range(start_step, config.training.max_steps + 1):
            if (
                step % config.training.save_steps == 0
                and not skip_once_in_resume
                and (not config.training.skip_first_step or step > 0)
                and not config.training.run_data_only
            ) and not config.training.eval_only:
                save_checkpoint(step, config, tokenizer, model, dataloader_train)
            
            if (
                step % config.training.eval_steps == 0
                and not config.training.run_data_only
                and not config.training.skip_eval
                and (not config.training.skip_first_step or step > 0)
            ) or config.training.eval_only:
                val_step(step, config, vae, model, dataloader_val, timer_manager, tb_writer)

            skip_once_in_resume = False

            if step < config.training.max_steps:
                training_step(
                    step,
                    config,
                    vae,
                    model,
                    dataloader_train,
                    timer_manager,
                    tb_writer,
                    tokenizer=tokenizer,
                    start_step=start_step,
                )
                if config.training.use_torch_profiler:
                    profiler.step()
    finally:
        if config.training.use_torch_profiler:
            profiler.stop()

        if dist_ctx.is_main_process and not config.training.run_data_only:
            if "tensorboard" in config.training.report_to:
                tb_writer.close()
            if "wandb" in config.training.report_to:
                wandb.finish()

        timer_manager.beat("training", "end")
        logger.info(f"Training time {format_time(timer_manager.training.value)}.")

        memory_monitor.stop()


if __name__ == "__main__":
    LazyLaunch(main, Arguments)
