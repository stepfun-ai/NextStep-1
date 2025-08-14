import re
import copy
import os
import json
import inspect
import random
from typing import Literal, TypeAlias
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from loguru import logger

from transformers import AutoTokenizer
from transformers.cache_utils import Cache, StaticCache
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL as DiffusersAutoencoderKL
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput

try:
    from .nextstep_model import NextStep
except ImportError:
    from nextstep_model import NextStep

DEFAULT_IMAGE_AREA_TOKEN = "<|image_area|>"
@dataclass
class AutoEncoderParams:
    resolution: int = 256
    in_channels: int = 3
    ch: int = 128
    out_ch: int = 3
    ch_mult: list[int] = field(default_factory=lambda: [1, 2, 4, 4])
    num_res_blocks: int = 2
    z_channels: int = 16
    scaling_factor: float = 1
    shift_factor: float = 0
    deterministic: bool = True
    encoder_norm: bool = True
    psz: int | None = 1


class AutoencoderKL(DiffusersAutoencoderKL):
    """
    Inherits from diffusers' AutoencoderKL, adding patchify and encoder_norm functionality.
    This allows reusing diffusers' complete implementation while only extending necessary features.
    """

    def __init__(self, params: AutoEncoderParams):
        """
        Initialize model from AutoEncoderParams, converting to diffusers format
        """
        # Convert parameter format to diffusers AutoencoderKL format
        down_block_types = ["DownEncoderBlock2D"] * len(params.ch_mult)
        up_block_types = ["UpDecoderBlock2D"] * len(params.ch_mult)
        block_out_channels = [params.ch * m for m in params.ch_mult]

        # Call parent class initialization to create diffusers encoder and decoder
        super().__init__(
            in_channels=params.in_channels,
            out_channels=params.out_ch,
            down_block_types=tuple(down_block_types),
            up_block_types=tuple(up_block_types),
            block_out_channels=tuple(block_out_channels),
            layers_per_block=params.num_res_blocks,
            latent_channels=params.z_channels,
            norm_num_groups=32,
            sample_size=params.resolution,
            scaling_factor=params.scaling_factor,
            shift_factor=params.shift_factor,
            act_fn="silu",
            mid_block_add_attention=True,
            use_quant_conv=False,  # Old VAE does not use quant_conv
            use_post_quant_conv=False,
        )

        # Save custom parameters
        self.params = params
        self.encoder_norm = params.encoder_norm
        self.psz = params.psz
        self.deterministic = params.deterministic

    def layer_norm_2d(self, input: torch.Tensor, normalized_shape: torch.Size, eps: float = 1e-6) -> torch.Tensor:
        """Layer normalization for 2D spatial features."""
        # input.shape = (bsz, c, h, w)
        return F.layer_norm(
            input.permute(0, 2, 3, 1), normalized_shape, None, None, eps
        ).permute(0, 3, 1, 2)

    def patchify(self, img: torch.Tensor):
        """
        img: (bsz, C, H, W)
        x: (bsz, patch_size**2 * C, H / patch_size, W / patch_size)
        """
        bsz, c, h, w = img.shape
        p = self.psz
        h_, w_ = h // p, w // p

        img = img.reshape(bsz, c, h_, p, w_, p)
        img = torch.einsum("nchpwq->ncpqhw", img)
        x = img.reshape(bsz, c * p**2, h_, w_)
        return x

    def unpatchify(self, x: torch.Tensor):
        """
        x: (bsz, patch_size**2 * C, H / patch_size, W / patch_size)
        img: (bsz, C, H, W)
        """
        bsz = x.shape[0]
        p = self.psz
        c = self.config.latent_channels
        h_, w_ = x.shape[2], x.shape[3]

        x = x.reshape(bsz, c, p, p, h_, w_)
        x = torch.einsum("ncpqhw->nchpwq", x)
        img = x.reshape(bsz, c, h_ * p, w_ * p)
        return img

    def encode(self, x: torch.Tensor, return_dict: bool = True):
        """
        Override encode method to support patchify and encoder_norm
        """
        # Use parent class encoder
        h = self.encoder(x)

        # Use parent class quant_conv (if exists)
        moments = self.quant_conv(h) if self.config.use_quant_conv else h

        # Apply patchify and normalization (if enabled)
        mean, logvar = torch.chunk(moments, 2, dim=1)

        if self.psz is not None:
            mean = self.patchify(mean)
            if self.encoder_norm:
                mean = self.layer_norm_2d(mean, (mean.size(1),))
            mean = self.unpatchify(mean)

        moments = torch.cat([mean, logvar], dim=1).contiguous()
        posterior = DiagonalGaussianDistribution(moments, deterministic=self.deterministic)

        return (posterior,) if not return_dict else AutoencoderKLOutput(latent_dist=posterior)

    @staticmethod
    def _convert_old_keys_to_diffusers(state_dict, num_resolutions=4):
        """
        Convert old custom VAE key names to diffusers AutoencoderKL key names
        Old format: encoder.down.X.block.Y -> New format: encoder.down_blocks.X.resnets.Y

        Note: decoder's up_blocks order needs to be reversed because old implementation uses insert(0, ...)
        """
        import re
        new_state_dict = {}

        # Define replacement rules
        replacements = [
            # General replacements
            (".nin_shortcut.", ".conv_shortcut."),
            (".norm_out.", ".conv_norm_out."),
            # Encoder down blocks
            (".down.", ".down_blocks."),
            (".block.", ".resnets."),
            (".downsample.", ".downsamplers.0."),
            # Encoder mid blocks
            (".mid.block_1.", ".mid_block.resnets.0."),
            (".mid.block_2.", ".mid_block.resnets.1."),
            (".mid.attn_1.norm.", ".mid_block.attentions.0.group_norm."),
            (".mid.attn_1.q.", ".mid_block.attentions.0.to_q."),
            (".mid.attn_1.k.", ".mid_block.attentions.0.to_k."),
            (".mid.attn_1.v.", ".mid_block.attentions.0.to_v."),
            (".mid.attn_1.proj_out.", ".mid_block.attentions.0.to_out.0."),
            # Decoder up blocks
            (".upsample.", ".upsamplers.0."),
        ]

        for key, value in state_dict.items():
            new_key = key

            # Skip keys that don't need conversion
            if any(skip in key for skip in ["conv_in", "conv_out"]) and ("encoder." in key or "decoder." in key):
                new_state_dict[new_key] = value
                continue

            # Encoder conversion
            if key.startswith("encoder."):
                for old_pattern, new_pattern in replacements:
                    new_key = new_key.replace(old_pattern, new_pattern)

            # Decoder conversion
            elif key.startswith("decoder."):
                # Handle up blocks index reversal
                if ".up." in key:
                    match = re.search(r'\.up\.(\d+)\.', key)
                    if match:
                        old_idx = int(match.group(1))
                        new_idx = num_resolutions - 1 - old_idx
                        new_key = re.sub(r'\.up\.\d+\.', f'.up_blocks.{new_idx}.', key)
                    else:
                        new_key = new_key.replace(".up.", ".up_blocks.")

                # Apply other replacement rules
                for old_pattern, new_pattern in replacements:
                    new_key = new_key.replace(old_pattern, new_pattern)

            # Handle Conv2d (1x1) -> Linear weight shape conversion
            if "attentions" in new_key and "weight" in new_key and len(value.shape) == 4:
                if value.shape[2] == 1 and value.shape[3] == 1:
                    value = value.squeeze(-1).squeeze(-1)

            new_state_dict[new_key] = value

        return new_state_dict

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        """
        Load model from local path (compatible with old format)
        """
        config_path = os.path.join(model_path, "config.json")
        ckpt_path = os.path.join(model_path, "checkpoint.pt")

        if not os.path.isdir(model_path) or not os.path.isfile(ckpt_path):
            raise ValueError(f"Invalid model path: {model_path}. Missing config.json or checkpoint.pt files.")

        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)

        # Load configuration
        config = {}
        if os.path.isfile(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
        config.update(kwargs)

        # Filter parameters in AutoEncoderParams
        param_signature = inspect.signature(AutoEncoderParams.__init__).parameters
        valid_kwargs = {k: v for k, v in config.items() if k in param_signature}

        # Record ignored parameters
        ignored_params = [k for k in config.keys() if k not in param_signature]
        if ignored_params:
            logger.debug(f"Ignoring parameters: {ignored_params}")

        params = AutoEncoderParams(**valid_kwargs)
        model = cls(params)

        # Convert old format key names to diffusers format
        logger.info("Converting old VAE keys to diffusers format...")
        state_dict = cls._convert_old_keys_to_diffusers(state_dict, num_resolutions=len(params.ch_mult))

        try:
            msg = model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded state_dict from {ckpt_path}")
            if msg.missing_keys:
                logger.warning(f"Missing keys: {msg.missing_keys}")
            # if msg.unexpected_keys:
            #     logger.warning(f"Unexpected keys: {msg.unexpected_keys}")
        except Exception as e:
            logger.error(f"Failed to load state_dict: {e}, using random initialization")

        return model


class NextStepPipeline:
    def __init__(
        self,
        model_name_or_path: str | None = None,
        vae_name_or_path: str | None = None,
        tokenizer: AutoTokenizer | None = None,
        model: nn.Module | None = None,
        vae: AutoencoderKL | None = None,
        enable_gradient_checkpointing: bool = False,
        attn_implementation: str | None = None,  # "sdpa", "flash_attention_2", "eager"
        device: str | None = "cuda",
        dtype: torch.dtype | None = torch.bfloat16,
        local_files_only: bool = False,  # Allow downloading from network by default
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
            padding_side="left",
            use_fast=True,
        )
        self.model: NextStep = NextStep.from_pretrained(model_name_or_path, local_files_only=local_files_only, enable_gradient_checkpointing=enable_gradient_checkpointing)
        self.model.to(device=device,dtype=dtype)


        self.tokenizer.add_eos_token = False
        if vae_name_or_path is None:
            vae_name_or_path = getattr(self.model.config, "vae_name_or_path", None)
        self.vae = AutoencoderKL.from_pretrained(vae_name_or_path, local_files_only=local_files_only).to(device=device,dtype=dtype)

        vae_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.down_factor = vae_factor * self.model.config.latent_patch_size
        self.shift_factor = getattr(self.vae.config, "shift_factor", 0.0)
        self.scaling_factor = getattr(self.vae.config, "scaling_factor", 1.0)

        self.boi = self.model.config.boi
        self.eoi = self.model.config.eoi

        self.image_placeholder_id = self.model.config.image_placeholder_id
        self.pil2tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.device = self.model.device
        self.dtype = self.model.dtype

    def set_seed(self, seed: int, rank: int = 0):
        random.seed(seed + rank)
        np.random.seed(seed + rank)
        torch.manual_seed(seed + rank)
        torch.cuda.manual_seed_all(seed + rank)
        torch.backends.cudnn.deterministic = True
        os.environ["PYTHONHASHSEED"] = str(seed + rank)

    def to_pil(self, image: torch.Tensor) -> Image.Image:
        """Convert PyTorch tensor to PIL image"""
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(image)}")

        # Normalize to [0,1] and convert to uint8
        image = (image / 2 + 0.5).clamp(0, 1).mul(255).round().to(torch.uint8)
        image = image.cpu().permute(1, 2, 0).numpy()

        # Handle single channel images
        if image.shape[-1] == 1:
            image = image[:, :, 0]
            mode = "L"
        else:
            mode = "RGB"

        return Image.fromarray(image, mode=mode)

    def to(self, device: str | None = None, dtype: torch.dtype | None = None):
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        self.model.to(self.device, dtype=self.dtype)
        self.vae.to(self.device, dtype=self.dtype)
        return self

    def hw2str(self, h: int, w: int) -> str:
        return f"{h}*{w}"

    def _image_str(self, hw: tuple[int, int] = (256, 256)):
        latent_hw = (hw[0] // self.down_factor, hw[1] // self.down_factor)
        image_ids = [self.boi] + [self.image_placeholder_id] * (latent_hw[0] * latent_hw[1]) + [self.eoi]
        image_str = DEFAULT_IMAGE_AREA_TOKEN + self.hw2str(*latent_hw) + self.tokenizer.decode(image_ids)
        return image_str

    def _check_input(
        self, captions: str | list[str], images: Image.Image | list[Image.Image] | None
    ) -> tuple[list[str], list[Image.Image] | None]:
        if not isinstance(captions, list):
            captions = [captions]
        if images is not None:
            if not isinstance(images, list):
                images = [images]
            # Verify image count matches <image> tokens
            image_token_count = sum(len(re.findall(r"<image>", caption)) for caption in captions)
            if image_token_count != len(images):
                raise ValueError(f"Image count ({len(images)}) does not match image token count ({image_token_count})")
            # Replace <image> tokens with image strings
            hws = [(img.size[1], img.size[0]) for img in images]
            processed_captions = []
            image_idx = 0

            for caption in captions:
                processed_caption = caption
                while "<image>" in processed_caption:
                    processed_caption = processed_caption.replace("<image>", self._image_str(hws[image_idx]), 1)
                    image_idx += 1
                processed_captions.append(processed_caption)
            captions = processed_captions
        return captions, images

    def _build_captions(
        self,
        captions: str | list[str],
        images: list[Image.Image] | None = None,
        num_images_per_caption: int = 1,
        positive_prompt: str | None = None,
        negative_prompt: str | None = None,
        cfg: float = 1.0,
        cfg_img: float = 1.0,
    ):
        # Normalize input
        if not isinstance(captions, list):
            captions = [captions]

        # Repeat captions and images
        captions = [caption for caption in captions for _ in range(num_images_per_caption)]
        if images is not None:
            images = [img for img in images for _ in range(num_images_per_caption)]

        # Add positive prompt
        if positive_prompt:
            captions = [f"{caption} {positive_prompt}" for caption in captions]

        # Set negative prompt default value
        negative_prompt = negative_prompt or ""

        num_samples = len(captions)

        # Simplify CFG logic
        if cfg != 1.0:
            if cfg_img != 1.0:  # Use image and text CFG
                w, h = images[0].size
                captions = captions + [self._image_str((h, w)) + negative_prompt] * num_samples
                images = images + images
            captions = captions + [negative_prompt] * num_samples

        return captions, images

    def _add_prefix_ids(self, hw: tuple[int, int], input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Add image area prefix IDs"""
        prefix_str = DEFAULT_IMAGE_AREA_TOKEN + self.hw2str(hw[0] // self.down_factor, hw[1] // self.down_factor)
        prefix_output = self.tokenizer(prefix_str, truncation=False, add_special_tokens=True, return_tensors="pt")

        prefix_input_ids = prefix_output.input_ids.to(input_ids.device, dtype=input_ids.dtype)
        prefix_attention_mask = prefix_output.attention_mask.to(attention_mask.device, dtype=attention_mask.dtype)

        # Remove BOS token and add BOI token
        if self.tokenizer.bos_token is not None:
            prefix_input_ids = prefix_input_ids[:, 1:]
            prefix_attention_mask = prefix_attention_mask[:, 1:]

        boi_token = prefix_input_ids.new_tensor([self.model.config.boi]).unsqueeze(0)
        prefix_input_ids = torch.cat([prefix_input_ids, boi_token], dim=1)
        prefix_attention_mask = torch.cat([prefix_attention_mask, prefix_attention_mask.new_ones((1, 1))], dim=1)

        # Expand to batch dimension and concatenate
        bsz = input_ids.shape[0]
        input_ids = torch.cat([input_ids, prefix_input_ids.expand(bsz, -1)], dim=1)
        attention_mask = torch.cat([attention_mask, prefix_attention_mask.expand(bsz, -1)], dim=1)

        return input_ids, attention_mask


    def layer_norm(self, input: torch.Tensor, normalized_shape: torch.Size, eps: float = 1e-6) -> torch.Tensor:
        """Simplified layer norm implementation using PyTorch built-in function"""
        return F.layer_norm(input, normalized_shape, eps=eps)

    @torch.no_grad()
    def decoding(
        self,
        c: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Cache,
        max_new_len: int,
        num_images_per_caption: int,
        noise: torch.Tensor = None,
        use_norm: bool = False,
        cfg: float = 1.0,
        cfg_img: float = 1.0,
        cfg_schedule: Literal["linear", "constant"] = "constant",
        timesteps_shift: float = 1.0,
        num_sampling_steps: int = 20,
        progress: bool = True,
        hw: tuple[int, int] = (256, 256),
        step: int = 0,
        sde_solver: bool = False,
        sde_type: str = "sde",
    ):
        indices = list(range(max_new_len))
        indices = tqdm(indices, unit="tokens") if progress else indices
        tokens = None
        for step in indices:
            cur_noise = None
            if noise is not None:
                cur_noise = noise[:,step:step+1,:][0]

            # Simplify CFG scheduling logic
            if cfg_schedule == "linear":
                tokens_len = 0 if tokens is None else tokens.shape[1]
                cfg_iter = max(cfg / 2, 1 + (cfg - 1) * tokens_len / max_new_len)
                cfg_img_iter = max(cfg_img / 2, 1 + (cfg_img - 1) * tokens_len / max_new_len)
            else:  # constant or other
                cfg_iter = cfg
                cfg_img_iter = cfg_img

            c = self.model.image_out_projector(c)
            token_sampled = self.model.image_head.sample(
                c=c.squeeze(1),
                noise=cur_noise,
                cfg=cfg_iter,
                cfg_img=cfg_img_iter,
                timesteps_shift=timesteps_shift,
                num_sampling_steps=num_sampling_steps,
                noise_repeat=num_images_per_caption,
                sde_solver=sde_solver,
                sde_type=sde_type,
            )

            if use_norm:
                token_sampled = self.layer_norm(token_sampled, normalized_shape=token_sampled.size()[1:])
            if tokens is not None:
                tokens = torch.cat([tokens, token_sampled.unsqueeze(1)], dim=1)
            else:
                tokens = token_sampled.unsqueeze(1)

            cur_inputs_embeds = self.model.image_in_projector(tokens[:, -1:])
            # Simplify CFG processing logic
            if cfg != 1.0:
                repeat_count = 3 if cfg_img != 1.0 else 2
                cur_inputs_embeds = torch.cat([cur_inputs_embeds] * repeat_count, dim=0)

            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
            outputs = self.model.forward(
                inputs_embeds=cur_inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            c = outputs.last_hidden_state[:, -1:]


        return tokens

    @torch.no_grad()
    def generate_image(
        self,
        captions: str | list[str],
        images: list[Image.Image] | None = None,
        num_images_per_caption: int = 1,
        positive_prompt: str | None = None,
        negative_prompt: str | None = None,
        hw: tuple[int, int] = (256, 256),
        use_norm: bool = False,
        cfg: float = 1.0,
        cfg_img: float = 1.0,
        cfg_schedule: Literal["linear", "constant"] = "constant",
        num_sampling_steps: int = 20,
        timesteps_shift: float = 1.0,
        seed: int = 42,
        progress: bool = True,
        sde_type: str = "sde",
    ) -> list[Image.Image]:
        # 0. set seed

        # 1. check input
        captions, images = self._check_input(captions, images)

        # 2. build captions
        captions, images = self._build_captions(
            captions, images, num_images_per_caption, positive_prompt, negative_prompt, cfg, cfg_img
        )

        # 3. encode images
        latents = None
        if images is not None:
            pixel_values = torch.stack([self.pil2tensor(img) for img in images]).to(self.device)
            posterior = self.vae.encode(pixel_values.to(self.vae.dtype)).latent_dist
            latents = (posterior.sample() - self.shift_factor) * self.scaling_factor

        # Add BOS token
        if self.tokenizer.bos_token is not None:
            captions = [self.tokenizer.bos_token + caption for caption in captions]

        if seed is not None:
            self.set_seed(seed)

        # 4. Tokenize caption & add prefix ids
        output = self.tokenizer(
            captions,
            padding="longest",
            truncation=False,
            add_special_tokens=True,
            return_tensors="pt",
            padding_side="left"
        )
        input_ids = output.input_ids.to(self.device)
        attention_mask = output.attention_mask.to(self.device)
        input_ids, attention_mask = self._add_prefix_ids(hw, input_ids, attention_mask)

        # 5. LLM prefill
        max_new_len = (hw[0] // self.down_factor) * (hw[1] // self.down_factor)
        max_cache_len = input_ids.shape[1] + max_new_len
        # past_key_values = StaticCache(
        #     config=self.model.config,
        #     max_batch_size=input_ids.shape[0],
        #     max_cache_len=max_cache_len,
        #     device=self.device,
        #     dtype=self.dtype,
        # )
        inputs_embeds = self.model.prepare_inputs_embeds(input_ids, latents)

        outputs = self.model.forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        c = outputs.last_hidden_state[:, -1:]


        # 6. decoding
        tokens = self.decoding(
            c=c,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            max_new_len=max_new_len,
            num_images_per_caption=num_images_per_caption,
            use_norm=use_norm,
            cfg=cfg,
            cfg_img=cfg_img,
            cfg_schedule=cfg_schedule,
            timesteps_shift=timesteps_shift,
            num_sampling_steps=num_sampling_steps,
            progress=progress,
            hw=hw,
            sde_type=sde_type,
        )

        # 7. Unpatchify
        latents = self.model.unpatchify(tokens, h=hw[0] // self.down_factor, w=hw[1] // self.down_factor)
        latents = (latents / self.scaling_factor) + self.shift_factor

        # 8. Decode latents
        sampled_images = self.vae.decode(latents.to(self.vae.dtype)).sample
        sampled_images = sampled_images.detach().cpu().to(torch.float32)
        pil_images = [self.to_pil(img) for img in sampled_images]

        return pil_images

