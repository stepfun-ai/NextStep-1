import gc
import glob
import inspect
import json
import math
import os
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
from tqdm.auto import tqdm
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import ModelOutput
from transformers import Qwen2Model, Qwen2Config

from nextstep.model_zoos import MODEL_ZOOS
from nextstep.models.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from nextstep.models.nextstep.tokenization_nextstep import IGNORE_INDEX, DEFAULT_EOL_TOKEN, DEFAULT_BOS_TOKEN
from nextstep.models.nextstep.aspect_ratio import str2ar
from nextstep.utils.compile_utils import compile_manager, smart_compile
from nextstep.utils.loguru import logger
from nextstep.utils.misc import LargeInt
from nextstep.utils.training_utils import set_seed



@dataclass
class NextStepOutputWithPast(CausalLMOutputWithPast):
    lm_loss: torch.FloatTensor | None = None
    image_loss: torch.FloatTensor | None = None
    sample_loss: list[torch.FloatTensor] | None = None


@dataclass
class ImageTokensOutput(ModelOutput):
    image_tokens: torch.FloatTensor | None = None
    conds: torch.FloatTensor | None = None
    refine_image_tokens: torch.FloatTensor | None = None
    tokens_id_sample: torch.LongTensor | None = None
    tokens_id_argmax: torch.LongTensor | None = None
    mean: torch.FloatTensor | None = None
    std: torch.FloatTensor | None = None
    mean_per_step: torch.FloatTensor | None = None
    std_per_step: torch.FloatTensor | None = None


# ============================================================================
# Flow Matching Head Components
# ============================================================================

def modulate(x, shift, scale=None):
    """Adaptive layer normalization modulation"""
    if shift is None:
        return x * (1 + scale)
    return x * (1 + scale) + shift


def expand_t(t, x):
    """Function to reshape time t to broadcastable dimension of x
    Args:
        t: [bsz,], time vector
        x: [bsz,...], data point
    """
    return t.view(-1, *([1] * (x.ndim - 1)))


class ResBlock(nn.Module):
    def __init__(self, channels, mlp_ratio=1.0):
        super().__init__()
        self.channels = channels
        self.intermediate_size = int(channels * mlp_ratio)

        self.in_ln = nn.LayerNorm(self.channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.channels, self.intermediate_size),
            nn.SiLU(),
            nn.Linear(self.intermediate_size, self.channels),
        )

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(channels, 3 * channels, bias=True))

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(model_channels, 2 * model_channels, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    
    Similar to diffusers.models.embeddings.TimestepEmbedding but with a different interface.
    This implementation follows the Glide-style timestep embedding.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element. These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq.to(self.mlp[0].weight.dtype))
        return t_emb


class SimpleMLPAdaLN(nn.Module):
    def __init__(self, input_dim, cond_dim, dim=1536, layers=12, mlp_ratio=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.dim = dim
        self.layers = layers
        self.mlp_ratio = mlp_ratio

        self.time_embed = TimestepEmbedder(dim)
        self.cond_embed = nn.Linear(cond_dim, dim)
        self.input_proj = nn.Linear(input_dim, dim)

        res_blocks = []
        for _ in range(layers):
            res_blocks.append(ResBlock(dim, mlp_ratio))
        self.res_blocks = nn.ModuleList(res_blocks)

        self.final_layer = FinalLayer(dim, input_dim)

        self.grad_checkpointing = False

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, c):
        """
        x.shape = (bsz, input_dim)
        t.shape = (bsz,)
        c.shape = (bsz, cond_dim)
        """

        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(c)

        y = t + c

        for block in self.res_blocks:
            if self.grad_checkpointing and self.training:
                x = checkpoint(block, x, y, use_reentrant=False)
            else:
                x = block(x, y)

        return self.final_layer(x, y)


class FlowMatchingHead(nn.Module):

    def __init__(self, input_dim, cond_dim, dim=1536, layers=12, mlp_ratio=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.net = SimpleMLPAdaLN(input_dim=input_dim, cond_dim=cond_dim, dim=dim, layers=layers, mlp_ratio=mlp_ratio)

    @property
    def dtype(self):
        return self.net.input_proj.weight.dtype

    @property
    def device(self):
        return self.net.input_proj.weight.device

    def forward(self, z: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None):
        noise = torch.randn_like(target)
        u = torch.normal(mean=0.0, std=1.0, size=(len(target),))
        t = (1 / (1 + torch.exp(-u))).to(target)

        xt, ut = expand_t(t, target) * target + (1 - expand_t(t, target)) * noise, target - noise

        model_output = self.net(xt, t, z)
        B, *_, C = xt.shape
        if model_output.size() != (B, *xt.size()[1:-1], C):
            raise ValueError(
                f"Model output shape ({model_output.size()}) does not match the expected shape ({B, *xt.size()[1:-1], C})"
            )

        loss = (model_output.float() - ut.float()) ** 2
        loss = torch.mean(loss, dim=list(range(1, len(loss.size()))))  # Take the mean over all non-batch dimensions.
        loss = (loss * mask).sum() / mask.sum() if mask is not None else loss.mean()
        return loss

    def get_score_from_velocity(self, velocity, x, t):
        """Wrapper function: transfrom velocity prediction model to score
        Args:
            velocity: [bsz, ...] shaped tensor; velocity model output
            x:        [bsz, ...] shaped tensor; x_t data point
            t:        [bsz,] time tensor
        """
        t = expand_t(t, x)
        alpha_t, d_alpha_t = t, 1
        sigma_t, d_sigma_t = 1 - t, -1
        mean = x
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
        score = (reverse_alpha_ratio * velocity - mean) / var
        return score

    def get_velocity_from_cfg(self, velocity, cfg, cfg_img, cfg_mul, normalize=False):
        if cfg_mul == 2:
            cond_v, uncond_v = torch.chunk(velocity, 2, dim=0)
            velocity = uncond_v + cfg * (cond_v - uncond_v)
            if normalize and cfg > 1.0:
                norm_cond = torch.norm(cond_v, dim=-1, keepdim=True)
                norm_velocity = torch.norm(velocity, dim=-1, keepdim=True)
                scale_factor = norm_cond / (norm_velocity + 1e-6)
                scale_factor = torch.clamp(scale_factor, min=0, max=1)
                velocity = velocity * scale_factor
        elif cfg_mul == 3:
            cond_v, uncond_v1, uncond_v2 = torch.chunk(velocity, 3, dim=0)
            velocity = uncond_v2 + cfg_img * (uncond_v1 - uncond_v2) + cfg * (cond_v - uncond_v1)
            if normalize and cfg > 1.0 and cfg_img > 1.0:
                norm_cond = torch.norm(cond_v, dim=-1, keepdim=True)
                norm_velocity = torch.norm(velocity, dim=-1, keepdim=True)
                scale_factor = norm_cond / (norm_velocity + 1e-6)
                scale_factor = torch.clamp(scale_factor, min=0, max=1)
                velocity = velocity * scale_factor
        return velocity

    def _compute_cfg_mult(self, cfg: float, cfg_img: float) -> int:
        """Calculate CFG multiplier"""
        cfg_mul = 1
        if cfg > 1.0:
            cfg_mul += 1
        if cfg_img > 1.0:
            cfg_mul += 1
        return cfg_mul

    def _compute_velocity_and_basics(self, x, c, ti, cfg, cfg_img, cfg_mul):
        """Calculate velocity and basic values (cur_t, next_t, x0, x1)"""
        combined = torch.cat([x] * cfg_mul, dim=0)
        velocity = self.net(combined.to(c.dtype), ti.expand(c.shape[0]).to(c), c)
        velocity = velocity.to(torch.float32)
        velocity = self.get_velocity_from_cfg(velocity, cfg, cfg_img, cfg_mul)
        
        cur_t = ti.view(-1, *([1] * (len(x.shape) - 1))).to(x.device)
        x0 = x - velocity * cur_t  # noise
        x1 = x + velocity * (1 - cur_t)  # ode image
        
        return velocity, cur_t, x0, x1

    @smart_compile(options={"triton.cudagraphs": True}, fullgraph=True)
    @torch.no_grad()
    def sample(
        self,
        c: torch.Tensor,
        noise: torch.Tensor = None,
        cfg: float = 1.0,
        cfg_img: float = 1.0,
        timesteps_shift: float = 1.0,
        num_sampling_steps: int = 20,
        last_step_size: float = 0.0,
        noise_repeat: int = 1,
        sde_solver: bool = False,
        sde_type: str = "sde",
        noise_level: float = 0.8,
    ):
        # """c.shape = (bsz, cond_dim)"""
        cfg_mul = self._compute_cfg_mult(cfg, cfg_img)

        if noise is None:
            noise = torch.randn((c.shape[0] // cfg_mul, self.input_dim), device=c.device, dtype=c.dtype)

        x = noise
        xs = []

        t0, t1 = 0, 1
        timesteps = torch.linspace(t0, t1, num_sampling_steps + 1, device=c.device)[:-1]
        timesteps = timesteps / (timesteps_shift - (timesteps_shift - 1) * timesteps)
        timesteps = torch.cat([timesteps, torch.ones(1, device=c.device)])
        sigma_max = timesteps[-2]
        for ti, tj in zip(timesteps[:-1], timesteps[1:]):
            velocity, cur_t, x0, x1 = self._compute_velocity_and_basics(x, c, ti, cfg, cfg_img, cfg_mul)
            next_t = tj.view(-1, *([1] * (len(x.shape) - 1))).to(x.device)
                
            if sde_type == "cps":
                # Flow-CPS
                std_dev_t = (1 - next_t)  * math.sin(noise_level * math.pi / 2) # sigma_t in paper
                sde_noise = torch.randn((c.shape[0] // cfg_mul, self.input_dim), device=c.device, dtype=c.dtype)
                
                x = x0 * torch.sqrt((1 - next_t)**2 - std_dev_t**2) + x1 * next_t + std_dev_t * sde_noise
            elif sde_type == "sde":
                sigma = 1 - cur_t
                dt = tj - ti
                std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma)))*noise_level
                
                variance_noise = torch.randn((c.shape[0] // cfg_mul, self.input_dim), device=c.device, dtype=c.dtype)
                
                x = x0 * (1 - next_t) + x1 * next_t - std_dev_t ** 2 * dt / (2 * sigma) * x0
                x = x + std_dev_t * torch.sqrt(dt) * variance_noise
            else:
                # ODE
                x = x0 * (1 - next_t) + x1 * next_t
            
            xs.append(x)

        if len(xs) != num_sampling_steps:
            raise ValueError(f"Samples ({len(xs)}) does not match the number of steps ({num_sampling_steps})")

        return xs[-1].to(c.dtype)


# ============================================================================
# NextStep Mixin and Config
# ============================================================================

class NextStepMixin:
    """Mixin base class containing methods shared by NextStep and NextStepModel"""
    
    def _create_linear(self, in_features: int, out_features: int, std: float) -> nn.Linear:
        """Create and initialize linear layer"""
        linear = nn.Linear(in_features, out_features)
        linear.weight.data.normal_(mean=0.0, std=std)
        linear.bias.data.zero_()
        return linear
    
    def _get_patch_size(self):
        """Get patch size, unified access interface"""
        if hasattr(self, 'image_patch_size'):
            return self.image_patch_size
        elif hasattr(self.config, 'latent_patch_size'):
            return self.config.latent_patch_size
        else:
            return self.config.patch_size
    
    def _get_num_channels(self):
        """Get number of channels, unified access interface"""
        if hasattr(self, 'image_num_channels'):
            return self.image_num_channels
        elif hasattr(self.config, 'latent_channels'):
            return self.config.latent_channels
        else:
            return self.config.num_channels
    
    def patchify(self, img: torch.Tensor):
        """
        img: (bsz, C, H, W)
        x: (bsz, H * W / patch_size**2, patch_size**2 * C)
        """
        bsz, c, h, w = img.shape
        p = self._get_patch_size()
        h_, w_ = h // p, w // p

        img = img.reshape(bsz, c, h_, p, w_, p)
        img = torch.einsum("nchpwq->nhwcpq", img)
        x = img.reshape(bsz, h_ * w_, c * p**2)
        return x

    def unpatchify(self, x: torch.Tensor, h: int = None, w: int = None):
        """
        x: (bsz, H * W / patch_size**2, patch_size**2 * C)
        img: (bsz, C, H, W)
        """
        bsz = x.shape[0]
        p = self._get_patch_size()
        c = self._get_num_channels()
        if h is None and w is None:
            h_ = w_ = int(x.shape[1] ** 0.5)
        else:
            h_, w_ = h, w
        
        # Use better error handling
        try:
            assert h_ * w_ == x.shape[1], f"Invalid sequence length {x.shape[1]}."
            x = x.reshape(bsz, h_, w_, c, p, p)
        except:
            assert h_ * (w_ + 1) == x.shape[1], f"Invalid sequence length {x.shape[1]}."
            logger.warning(f"Invalid sequence length {x.shape[1]}, reshape to {h_} * {w_ + 1}.")
            x = x.reshape(bsz, h_, w_ + 1, c, p, p)
            x = x[:, :, :-1, :, :, :]
        
        x = torch.einsum("nhwcpq->nchpwq", x)
        img = x.reshape(bsz, c, h_ * p, w_ * p)
        return img
    
    def gradient_checkpointing_enable(self, **kwargs):
        """Enable gradient checkpointing"""
        super().gradient_checkpointing_enable(**kwargs)
        if hasattr(self, 'image_head') and hasattr(self.image_head, 'net'):
            self.image_head.net.grad_checkpointing = True
        if hasattr(self, 'gradient_checkpointing'):
            self.gradient_checkpointing = True


class NextStepConfig(Qwen2Config):

    model_type = "nextstep"

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 1,
        num_channels: int = 16,
        hw_aspect_ratios_ids: dict[str, list[int]] | None = None,
        image_placeholder_id: int | None = None,
        boi: int | None = None,
        eoi: int | None = None,
        pad_token_id_added: int | None = None,
        vae_name_or_path: str | None = None,
        use_mlp_before_lm_head: bool = False,
        use_token_length_weight: bool = False,
        lm_loss_weight: float = 1.0,
        image_loss_weight: float = 1.0,
        noise_strength: float = 0.0,
        fm_head_dim: int = 1536,
        fm_head_layers: int = 12,
        fm_head_mlp_ratio: float = 1.0,
        fm_head_batch_mul: int = 4,
        create_kwargs: dict = {},
        o_attention_bias: bool = False,
        attention_use_qk_norm: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        image_grid_prefix_ids = kwargs.pop("image_grid_prefix_ids", None)
        if image_grid_prefix_ids is not None:
            logger.warning("`image_grid_prefix_ids` is deprecated, will use `hw_aspect_ratios_ids` instead.")

        self.image_size = image_size
        self.patch_size = patch_size
        self.base_image_grid_size = image_size // patch_size
        self.num_channels = num_channels

        self.hw_aspect_ratios_ids = {}
        if image_grid_prefix_ids is not None:
            self.hw_aspect_ratios_ids.update({f"{image_size//patch_size}*{image_size//patch_size}": image_grid_prefix_ids})
        if hw_aspect_ratios_ids is not None:
            self.hw_aspect_ratios_ids.update(hw_aspect_ratios_ids)

        self.image_placeholder_id = image_placeholder_id
        self.boi = boi
        self.eoi = eoi
        self.pad_token_id_added = pad_token_id_added
        self.vae_name_or_path = vae_name_or_path
        self.use_mlp_before_lm_head = use_mlp_before_lm_head
        self.use_token_length_weight = use_token_length_weight
        self.lm_loss_weight = lm_loss_weight
        self.image_loss_weight = image_loss_weight
        self.noise_strength = noise_strength
        self.fm_head_dim = fm_head_dim
        self.fm_head_layers = fm_head_layers
        self.fm_head_mlp_ratio = fm_head_mlp_ratio
        self.fm_head_batch_mul = fm_head_batch_mul
        self.create_kwargs = create_kwargs

        self.o_attention_bias = o_attention_bias
        self.attention_use_qk_norm = attention_use_qk_norm


# ============================================================================
# NextStep Model
# ============================================================================

class NextStepModel(NextStepMixin, Qwen2Model, GenerationMixin):
    config_class = NextStepConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    @property
    def trainable_params(self) -> float:
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return LargeInt(n_params)

    def __init__(self, config: NextStepConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.gradient_checkpointing = True

        # Initialize weights and apply final processing
        self.post_init()

        if config.use_mlp_before_lm_head:
            self.before_lm_head = nn.Linear(config.hidden_size, config.hidden_size)
            self.before_lm_head.weight.data.normal_(mean=0.0, std=config.initializer_range)
            self.before_lm_head.bias.data.zero_()
        else:
            self.before_lm_head = nn.Identity()

        image_in_dim = self.image_token_embed_dim

        self.image_in_projector = self._create_linear(image_in_dim, config.hidden_size, config.initializer_range)
        self.image_out_projector = self._create_linear(config.hidden_size, config.hidden_size, config.initializer_range)
        
        self.image_head = FlowMatchingHead(
            input_dim=image_in_dim,
            cond_dim=config.hidden_size,
            dim=config.fm_head_dim,
            layers=config.fm_head_layers,
            mlp_ratio=config.fm_head_mlp_ratio,
        )

    @property
    def image_size(self):
        return self.config.image_size

    @property
    def image_patch_size(self):
        return self.config.patch_size

    @property
    def image_grid_size(self):
        return round(self.image_size / self.image_patch_size)

    @property
    def image_num_channels(self):
        return self.config.num_channels

    @property
    def image_seq_len(self) -> dict[str, int]:
        return {k: str2ar(k)[0] * str2ar(k)[1] for k in self.config.hw_aspect_ratios_ids.keys()}

    @property
    def image_token_embed_dim(self):
        return self.image_num_channels * self.image_patch_size**2

    def reinit_image_head(self):
        self.image_head.net.initialize_weights()

    @staticmethod
    def _resolve_weight_files(model_dir: str) -> list[str]:
        index_files = ("model.safetensors.index.json", "pytorch_model.bin.index.json")
        for index_file in index_files:
            index_path = os.path.join(model_dir, index_file)
            if os.path.exists(index_path):
                with open(index_path, "r") as fp:
                    index_data = json.load(fp)
                weight_map = index_data.get("weight_map", {})
                shard_files = list(dict.fromkeys(weight_map.values()))
                return [os.path.join(model_dir, shard) for shard in shard_files]

        direct_files = ("model.safetensors", "pytorch_model.bin")
        for file_name in direct_files:
            file_path = os.path.join(model_dir, file_name)
            if os.path.exists(file_path):
                return [file_path]

        shard_patterns = ("model-*.safetensors", "pytorch_model-*.bin", "*.safetensors", "*.bin")
        for pattern in shard_patterns:
            shard_files = sorted(glob.glob(os.path.join(model_dir, pattern)))
            if len(shard_files) > 0:
                return shard_files
        return []

    @staticmethod
    def _load_weight_file(weight_file: str) -> dict[str, torch.Tensor]:
        if weight_file.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file as safe_load_file
            except ImportError as e:
                raise ImportError(
                    "Loading .safetensors checkpoint requires `safetensors` package, please install it first."
                ) from e
            return safe_load_file(weight_file, device="cpu")

        try:
            return torch.load(weight_file, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(weight_file, map_location="cpu")

    def load_llama_model(self, lm_model_name_or_path: str, local_files_only: bool = False):
        model_dir = MODEL_ZOOS[lm_model_name_or_path]
        if not os.path.isdir(model_dir):
            logger.warning(
                f"`{model_dir}` is not a local directory, fallback to `from_pretrained` loading. "
                "This fallback may consume more peak memory."
            )
            from transformers import LlamaForCausalLM

            temp_llama_model = LlamaForCausalLM.from_pretrained(
                model_dir,
                local_files_only=local_files_only,
                low_cpu_mem_usage=True,
            )
            sd = temp_llama_model.state_dict()
            sd = {k[6:] if k.startswith("model.") else k: v for k, v in sd.items()}
            msg = self.load_state_dict(sd, strict=False)
            logger.info(f"Loaded llama model from {lm_model_name_or_path}.")
            ignore_keys = [
                "image_in_projector",
                "image_encoder",
                "image_out_projector",
                "image_head",
            ]
            missing_keys = [k for k in msg.missing_keys if all(not k.startswith(key) for key in ignore_keys)]
            logger.info(f"Missing keys:\n{missing_keys}")
            logger.info(f"Unexpected keys:\n{msg.unexpected_keys}")
            del temp_llama_model
            del sd
            gc.collect()
            return

        weight_files = self._resolve_weight_files(model_dir)
        if len(weight_files) == 0:
            raise FileNotFoundError(f"No model weight file found in `{model_dir}`.")

        target_state = self.state_dict()
        loaded_keys: set[str] = set()
        unexpected_keys: set[str] = set()

        for weight_file in tqdm(
            weight_files,
            total=len(weight_files),
            desc=f"Loading llama shards from {lm_model_name_or_path}",
        ):
            shard_state_dict = self._load_weight_file(weight_file)
            mapped_state_dict: dict[str, torch.Tensor] = {}

            for key, value in shard_state_dict.items():
                mapped_key = key[6:] if key.startswith("model.") else key
                if mapped_key not in target_state:
                    unexpected_keys.add(mapped_key)
                    continue

                target_dtype = target_state[mapped_key].dtype
                if value.is_floating_point() and value.dtype != target_dtype:
                    value = value.to(dtype=target_dtype)
                mapped_state_dict[mapped_key] = value

            self.load_state_dict(mapped_state_dict, strict=False)
            loaded_keys.update(mapped_state_dict.keys())

            del shard_state_dict
            del mapped_state_dict
            gc.collect()

        logger.info(f"Loaded llama model from {lm_model_name_or_path}.")
        ignore_keys = [
            "image_in_projector",
            "image_encoder",
            "image_out_projector",
            "image_head",
        ]
        missing_keys = [k for k in target_state.keys() if k not in loaded_keys and all(not k.startswith(key) for key in ignore_keys)]
        logger.info(f"Missing keys:\n{missing_keys}")
        logger.info(f"Unexpected keys:\n{sorted(unexpected_keys)}")



    def prepare_inputs_embeds(self, input_ids: torch.LongTensor | None = None, pixel_values: torch.FloatTensor | None = None):
        if pixel_values is None:
            if not self.training:
                return self.embed_tokens(input_ids)
            else:  # dummy forward for image pass, for the consistent shape of gradient.
                # raise NotImplementedError("Dummy forward for image pass is not implemented.")
                return self.embed_tokens(input_ids)
        else:
            bs, seq_length = input_ids.shape
            inputs_embeds = torch.zeros(
                (bs, seq_length, self.config.hidden_size),
                device=self.embed_tokens.weight.device,
                dtype=self.embed_tokens.weight.dtype,
            )
            image_indices = input_ids == self.config.image_placeholder_id
            token_indices = ~image_indices

            if isinstance(pixel_values, list):
                tokens = []
                for pixel_value in pixel_values:
                    tokens.append(self.patchify(pixel_value))
            else:
                tokens = self.patchify(pixel_values)
            if isinstance(tokens, list):
                image_embeds = []
                for token in tokens:
                    image_embeds.append(self.image_in_projector(token)[0])
                image_embeds = torch.cat(image_embeds, dim=0)
            else:
                image_embeds = self.image_in_projector(tokens)
                image_embeds = image_embeds.view(-1, self.config.hidden_size)

            token_embeds = self.embed_tokens(input_ids[token_indices])

            inputs_embeds[image_indices] = image_embeds.to(inputs_embeds.dtype)
            inputs_embeds[token_indices] = token_embeds

            return inputs_embeds

    def get_rope_index(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor | list[torch.FloatTensor] = None,
        attention_mask: torch.Tensor | None = None,
    ):
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        _, _, h, w = pixel_values.shape
        h, w = h // self.image_patch_size, w // self.image_patch_size
        if pixel_values is not None:
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            position_ids = torch.ones(2, bsz, seq_len, dtype=input_ids.dtype, device=device)
            for i, (_input_ids, _attention_mask) in enumerate(zip(input_ids, attention_mask)):
                _input_ids = _input_ids[_attention_mask == 1]
                num_images = (_input_ids == self.config.boi).sum().item()
                input_ids_list = _input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images = num_images
                for _ in range(num_images):
                    if remain_images > 0:
                        ed = input_ids_list.index(self.config.boi, st)
                    else:
                        ed = len(input_ids_list) + 1
                    remain_images -= 1

                    text_len = ed - st
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(2, -1) + st_idx)

                    hpos_ids = torch.arange(h).view(-1, 1).expand(-1, w).flatten()
                    wpos_ids = torch.arange(w).view(1, -1).expand(h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([hpos_ids, wpos_ids]) + text_len + st_idx)
                    st = ed + h * w

                if st < len(input_ids_list):
                    text_len = len(input_ids_list) - st
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(2, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(2, -1)
                position_ids[..., i, _attention_mask == 1] = llm_positions.to(device)
            return position_ids
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(2, -1, -1).to(device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            else:
                position_ids = torch.arange(input_ids.shape[1], device=device).view(1, 1, -1).expand(2, input_ids.shape[0], -1)
            return position_ids

    @smart_compile
    def forward_model(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor | list[torch.FloatTensor] = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple | BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    **kwargs,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def forward_genloss(
        self,
        z: torch.Tensor | list[torch.Tensor],
        target: torch.Tensor | list[torch.Tensor],
        hw: list[tuple[int, int]] | None = None,
    ):
        if isinstance(z, list):
            loss = torch.tensor(0.0, device=z[0].device)
            z_list = []
            target_list = torch.cat(target, dim=1)
            for i in range(len(z)):
                bsz, seq_len, _ = target[i].shape
                z_single = self.image_out_projector(z[i])
                z_list.append(z_single)

            z_list = torch.cat(z_list, dim=1)
            target = target_list.repeat(1, self.config.fm_head_batch_mul, 1).reshape(
                target_list.shape[1] * self.config.fm_head_batch_mul, -1
            )
            z = z_list.repeat(1, self.config.fm_head_batch_mul, 1).reshape(
                z_list.shape[1] * self.config.fm_head_batch_mul, -1
            )

            # FlowMatchingHead.forward(self, target, z, mask=None)
            # Use positional arguments as defined here to avoid _forward_unimplemented errors when keyword arguments are wrapped/replaced
            loss = self.image_head(z=z, target=target.to(z.dtype)).mean()
            # Alternative implementation for sample_loss (currently unused):
            # loss = self.image_head(z=z, target=target.to(z.dtype))
            # loss = loss.reshape(bsz, seq_len, self.config.fm_head_batch_mul)
            return loss
        else:
            bsz, seq_len, _ = target.shape
            z = self.image_out_projector(z)

            target = target.repeat(1, self.config.fm_head_batch_mul, 1).reshape(
                bsz * seq_len * self.config.fm_head_batch_mul, -1
            )
            z = z.repeat(1, self.config.fm_head_batch_mul, 1).reshape(bsz * seq_len * self.config.fm_head_batch_mul, -1)

            # Same as above, use positional arguments aligned with FlowMatchingHead.forward
            loss = self.image_head(z=z, target=target.to(z.dtype)).mean()
            # Alternative implementation for sample_loss (currently unused):
            # loss = self.image_head(z=z, target=target.to(z.dtype))
            # loss = loss.reshape(bsz, seq_len, self.config.fm_head_batch_mul)
            return loss

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor | list[torch.FloatTensor] = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        force_train_mode: bool = False,
        image_filtered_idx: torch.LongTensor | None = None,
        *args,
        **kwargs,
    ) -> tuple | CausalLMOutputWithPast:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        if self.training and self.config.noise_strength > 0.0 and pixel_values is not None:
            if isinstance(pixel_values, list):
                noise_pixel_values = []
                for pixel_value in pixel_values:
                    _, _, h, w = pixel_value.shape
                    noise_strength = (
                        torch.linspace(0, self.config.noise_strength, steps=h * w)
                        .view(1, 1, h, w)
                        .to(pixel_value.device, pixel_value.dtype)
                    )
                    noise = torch.randn_like(pixel_value, device=pixel_value.device, dtype=pixel_value.dtype)
                    noise_pixel_values.append(pixel_value + noise * noise_strength)
            else:
                _, _, h, w = pixel_values.shape
                noise_strength = (
                    torch.linspace(0, self.config.noise_strength, steps=h * w)
                    .view(1, 1, h, w)
                    .to(pixel_values.device, pixel_values.dtype)
                )
                noise = torch.randn_like(pixel_values, device=pixel_values.device, dtype=pixel_values.dtype)
                noise_pixel_values = pixel_values + noise * noise_strength
        else:  # not training or no noise
            noise_pixel_values = pixel_values

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.prepare_inputs_embeds(input_ids, noise_pixel_values)

        outputs = super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs
        )

        hidden_states = outputs[0]

        # For generation, we only need the last token logits
        if not self.training and not force_train_mode:
            logits = self.lm_head(hidden_states[:, -1:, :])
            return CausalLMOutputWithPast(
                loss=None,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        # last hidden state will not be calculated to next token prediction loss
        hidden_states = hidden_states[..., :-1, :].contiguous()

        image_indices = input_ids == self.config.image_placeholder_id
        # label will determine which head (`lm_head` or `image_head`) to use
        shift_image_indices = image_indices[..., 1:].contiguous()
        shift_token_indices = ~shift_image_indices

        ###########################################################################
        # LM Loss
        ###########################################################################
        lm_hidden_states = hidden_states[shift_token_indices].contiguous()
        lm_hidden_states = self.before_lm_head(lm_hidden_states)
        shift_lm_logits = self.lm_head(lm_hidden_states)
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        shift_lm_logits = shift_lm_logits.float()
        shift_lm_logits = shift_lm_logits.view(-1, self.config.vocab_size)

        # Shift so that tokens < n predict n
        shift_labels = labels[..., 1:].contiguous()

        shift_lm_labels = shift_labels[shift_token_indices]
        shift_lm_labels = shift_lm_labels.view(-1)
        shift_lm_labels = shift_lm_labels.to(shift_lm_logits.device)

        loss_fct = CrossEntropyLoss()
        lm_loss = loss_fct(shift_lm_logits, shift_lm_labels)

        ###########################################################################
        # Image Loss
        ###########################################################################
        if pixel_values is not None and self.config.image_loss_weight > 0.0:
            image_hidden_states = hidden_states[shift_image_indices].contiguous()
            if isinstance(pixel_values, list):
                gt_latents = [self.patchify(pixel_value).clone().detach() for pixel_value in pixel_values]
            else:
                gt_latents = self.patchify(pixel_values).clone().detach()

            if isinstance(gt_latents, list):
                image_hidden_states_reordered = []
                start_idx = 0
                for gt_latent in gt_latents:
                    image_hidden_state = image_hidden_states[start_idx : start_idx + gt_latent.shape[1]].unsqueeze(0)
                    start_idx += gt_latent.shape[1]
                    image_hidden_states_reordered.append(image_hidden_state)
                image_hidden_states = image_hidden_states_reordered

                image_hidden_states = [
                    image_hidden_state for i, image_hidden_state in enumerate(image_hidden_states) if image_filtered_idx[i] != 0
                ]
                gt_latents = [gt_latent for i, gt_latent in enumerate(gt_latents) if image_filtered_idx[i] != 0]
                pixel_values = [pixel_value for i, pixel_value in enumerate(pixel_values) if image_filtered_idx[i] != 0]
            else:
                image_hidden_states = image_hidden_states.reshape(
                    gt_latents.shape[0], gt_latents.shape[1], image_hidden_states.shape[-1]
                )

                image_hidden_states = image_hidden_states[image_filtered_idx != 0]
                gt_latents = gt_latents[image_filtered_idx != 0]
                pixel_values = pixel_values[image_filtered_idx != 0]

            assert (
                len(pixel_values) == len(image_hidden_states) == len(gt_latents)
            ), f"The length of pixel_values: {len(pixel_values)}, image_hidden_states: {len(image_hidden_states)}, gt_latents: {len(gt_latents)} are not the same"

            if isinstance(pixel_values, list):
                hw = []
                for i, pixel_value in enumerate(pixel_values):
                    image_hidden_states[i] = image_hidden_states[i].reshape(
                        gt_latents[i].shape[0], gt_latents[i].shape[1], image_hidden_states[i].shape[-1]
                    )
                    _, _, h, w = pixel_values[i].shape
                    hw_single = (h // self.image_patch_size, w // self.image_patch_size)
                    hw.append(hw_single)
            else:
                image_hidden_states = image_hidden_states.reshape(
                    gt_latents.shape[0], gt_latents.shape[1], image_hidden_states.shape[-1]
                )
                _, _, h, w = pixel_values.shape
                hw = (h // self.image_patch_size, w // self.image_patch_size)

            image_loss = self.forward_genloss(image_hidden_states, gt_latents, hw=hw)

        else:
            image_loss = torch.tensor(0.0, device=lm_loss.device)

        if torch.isnan(lm_loss) or torch.isinf(lm_loss):
            raise ValueError(f"LM Loss is {lm_loss}, stopping training!")
        if torch.isnan(image_loss) or torch.isinf(image_loss):
            raise ValueError(f"Image Loss is {image_loss}, stopping training!")

        logits = None  # TODO: implement logits for generation
        # sample_loss = image_loss.clone().detach()
        # sample_loss = sample_loss.reshape(sample_loss.shape[0], -1).mean(dim=-1)
        # image_loss = image_loss.mean()
        if self.config.use_token_length_weight:
            lm_tokens = (shift_lm_labels != IGNORE_INDEX).sum().detach().item()
            image_tokens = shift_image_indices.sum().detach().item()

            loss = (
                (lm_loss * self.config.lm_loss_weight) * lm_tokens + (image_loss * self.config.image_loss_weight) * image_tokens
            ) / (lm_tokens + image_tokens)
        else:
            loss = lm_loss * self.config.lm_loss_weight + image_loss * self.config.image_loss_weight

        return NextStepOutputWithPast(
            loss=loss,
            lm_loss=lm_loss.detach(),
            image_loss=image_loss.detach(),
            # sample_loss=sample_loss.tolist(),
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def sample_image_tokens(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        uncond_text_input_ids: torch.LongTensor | None = None,
        uncond_text_attention_mask: torch.Tensor | None = None,
        uncond_image_input_ids: torch.LongTensor | None = None,
        uncond_image_attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        hw_aspect_ratio: str = "16*16",
        cfg: float = 1.0,
        cfg_img: float = 1.0,
        cfg_schedule: Literal["linear", "constant"] = "constant",
        use_norm: bool = True,
        temperature: float = 1.0,
        num_sampling_steps: int = 20,
        sampler: Literal["ode", "sde"] = "sde",
        ode_sampling_method: Literal["dopri5", "euler", "heun2", "rk4"] = "dopri5",
        return_tokens_id: bool = False,
        return_mean_std: bool = False,
        seed: int | None = None,
        progress: bool = False,
        timesteps_shift: float = 1.0,
        add_eol_token: bool = False,
        eol_token_id: int = 151674, # watchout
        pad_token_id: int = 151665,
        add_bos_token_id = False,
        tokenizer = None,
        default_noise: torch.Tensor | None = None,
        **kwargs,
    ) -> ImageTokensOutput:
        
        def _get_tokens_id(z):
            logits = self.lm_head(z[:, -1, :]).float()
            logits[:, self.config.eoi] = -999999
            # token selection
            probs = nn.functional.softmax(logits, dim=-1)
            next_tokens_sample = torch.multinomial(probs, num_samples=1).squeeze(1)
            next_tokens_max = torch.argmax(logits, dim=-1)
            return next_tokens_sample.unsqueeze(1), next_tokens_max.unsqueeze(1),logits[:,self.config.eoi]

        if seed is not None:
            set_seed(seed)

        if len(self.config.hw_aspect_ratios_ids) == 1 and f"{self.image_grid_size}*{self.image_grid_size}" != hw_aspect_ratio:
            logger.warning(
                f"hw_aspect_ratio is {hw_aspect_ratio}, but only one aspect ratio is provided, using {self.image_grid_size}*{self.image_grid_size} instead."
            )
            hw_aspect_ratio = f"{self.image_grid_size}*{self.image_grid_size}"
        h, w = str2ar(hw_aspect_ratio)
        
        prefix_ids = self.config.hw_aspect_ratios_ids[hw_aspect_ratio] + [self.config.boi]
        
        
        input_ids = torch.cat([input_ids, input_ids.new_tensor(prefix_ids).expand(input_ids.shape[0], -1)], dim=1)
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], len(prefix_ids)))], dim=1)
        # attention_mask = torch.cat([torch.ones_like(attention_mask), attention_mask.new_ones((attention_mask.shape[0], len(prefix_ids)))], dim=1)
        

        if add_eol_token:
            eol_token_ids = input_ids.new_tensor([eol_token_id]).expand(input_ids.shape[0], -1)
            eol_token_embeddings = self.embed_tokens(eol_token_ids)

        # eoi_token_embeddings = self.embed_tokens(input_ids.new_tensor([self.config.eoi]).expand(input_ids.shape[0], -1))
        
        device = input_ids.device
        dtype = input_ids.dtype

        if uncond_text_input_ids is not None:
            uncond_text_input_ids = torch.cat(
                [
                    uncond_text_input_ids,
                    torch.tensor(prefix_ids, device=device, dtype=dtype).expand(uncond_text_input_ids.shape[0], -1),
                ],
                dim=1,
            )
            uncond_text_attention_mask = torch.cat(
                [
                    uncond_text_attention_mask,
                    uncond_text_attention_mask.new_ones((uncond_text_attention_mask.shape[0], len(prefix_ids))),
                ],
                dim=1,
            )
            input_ids = torch.cat([input_ids, uncond_text_input_ids], dim=0)
            attention_mask = torch.cat([attention_mask, uncond_text_attention_mask], dim=0)

        if uncond_image_input_ids is not None:
            uncond_image_input_ids = torch.cat(
                [   
                    uncond_image_input_ids,
                    torch.tensor(prefix_ids, device=device, dtype=dtype).expand(uncond_image_input_ids.shape[0], -1),
                ],
                dim=1,
            )
            uncond_image_attention_mask = torch.cat(
                [
                    uncond_image_attention_mask,
                    uncond_image_attention_mask.new_ones((uncond_image_attention_mask.shape[0], len(prefix_ids))),
                ],
                dim=1,
            )
            
            input_ids = torch.cat([input_ids, uncond_image_input_ids], dim=0)
            attention_mask = torch.cat([attention_mask, uncond_image_attention_mask], dim=0)
        
        if add_bos_token_id:
            bos_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_BOS_TOKEN)
            bsz = input_ids.shape[0]
            input_ids = torch.cat([input_ids.new_tensor([pad_token_id]).expand(bsz, -1), input_ids,], dim=1)
            attention_mask = torch.cat([attention_mask.new_tensor([0]).expand(bsz, -1), attention_mask], dim=1)
            
            # Find the last occurrence of pad_token_id in this row
            for i in range(len(input_ids)):
                pad_positions = (input_ids[i] == pad_token_id).nonzero(as_tuple=True)[0]
                if len(pad_positions) > 0:
                    last_pad_pos = pad_positions[-1]
                    input_ids[i, last_pad_pos] = bos_token_id
                    attention_mask[i, last_pad_pos] = 1
                
        max_new_len = self.image_seq_len[hw_aspect_ratio] if not add_eol_token else self.image_seq_len[hw_aspect_ratio] + h
        max_generated_length = input_ids.shape[1] + max_new_len
        lm_past_key_values = StaticCache(
            config=self.config,
            batch_size=input_ids.shape[0],
            max_cache_len=max_generated_length,
            device=self.device,
            dtype=self.dtype,
        )
        
        inputs_embeds = self.prepare_inputs_embeds(input_ids, pixel_values)        
        with compile_manager.compile_disabled():
            outputs = self.forward_model(
                input_ids=None,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                past_key_values=lm_past_key_values,
                use_cache=True,
            )
        lm_past_key_values = outputs.past_key_values
        z = outputs.last_hidden_state[:, -1:]
        if return_tokens_id:
            tokens_id_sample, tokens_id_max = _get_tokens_id(z)

        indices = list(range(self.image_seq_len[hw_aspect_ratio])) if not add_eol_token else list(range(self.image_seq_len[hw_aspect_ratio] + h - 1))
        
        indices = tqdm(indices) if progress else indices

        img_encoder_past_key_values = None
        img_decoder_past_key_values = None
        tokens = None
        unnormed_tokens = None
        conds = []
        for step in indices:
            cur_tokens = None if tokens is None else tokens.clone()

            # cfg schedule follow Muse
            if cfg_schedule == "linear":
                tokens_len = 0 if tokens is None else tokens.shape[1]
                cfg_iter = (
                    1 + (cfg - 1) * (self.image_seq_len[hw_aspect_ratio] - tokens_len) / self.image_seq_len[hw_aspect_ratio]
                )
                cfg_iter = max(cfg / 2, cfg - (cfg_iter - 1))
                if cfg_img != 1.0:
                    cfg_img_iter = (
                        1 + (cfg_img - 1) * (self.image_seq_len[hw_aspect_ratio] - tokens_len) / self.image_seq_len[hw_aspect_ratio]
                    )
                    cfg_img_iter = max(cfg_img / 2, cfg_img - (cfg_img_iter - 1))
                else:
                    cfg_img_iter = cfg_img
            elif cfg_schedule == "constant":
                cfg_iter = cfg
                cfg_img_iter = cfg_img
            else:
                raise NotImplementedError

            z = self.image_out_projector(z)
            conds.append(z)

            sampled_token_latent = self.image_head.sample_cps(
                z=z.squeeze(1),
                cfg=cfg_iter,
                cfg2=cfg_img_iter,
                temperature=temperature,
                num_sampling_steps=num_sampling_steps,
                sampler=sampler,
                ode_sampling_method=ode_sampling_method,
                timesteps_shift=timesteps_shift,
                default_noise=default_noise[:,step:step+1,:] if default_noise is not None else None,
            )
                
            if cfg != 1.0 and cfg_img == 1.0:
                sampled_token_latent, sampled_token_latent_uncond = sampled_token_latent.chunk(2, dim=0)  # Remove null class samples
                # if use_norm:
                #     norm_v_t = torch.norm(sampled_token_latent_uncond, dim=-1, keepdim=True)
                #     norm_v_t_text = torch.norm(sampled_token_latent, dim=-1, keepdim=True)
                #     scale = (norm_v_t / (norm_v_t_text + 1e-8)).clamp(min=0, max=1.0)
                #     sampled_token_latent = sampled_token_latent * scale
            elif cfg != 1.0 and cfg_img != 1.0:
                sampled_token_latent, _, _ = sampled_token_latent.chunk(3, dim=0)  # Remove null class samples

            unnormed_sampled_token_latent = sampled_token_latent.clone()
            
            if cur_tokens is not None:
                cur_tokens = torch.cat([cur_tokens, sampled_token_latent.unsqueeze(1)], dim=1)
                unnormed_tokens = torch.cat([unnormed_tokens, unnormed_sampled_token_latent.unsqueeze(1)], dim=1)
            else:
                cur_tokens = sampled_token_latent.unsqueeze(1)
                unnormed_tokens = unnormed_sampled_token_latent.unsqueeze(1)

            tokens = cur_tokens.clone()

            if add_eol_token and (step+1) % (w+1) == 0:
                tokens = tokens[:, :-1]

            cur_input_embeds = self.image_in_projector(tokens[:, -1:])

            if add_eol_token and (step+1) % (w+1) == 0:
                cur_input_embeds = eol_token_embeddings

            if cfg != 1.0 and cfg_img == 1.0:
                cur_input_embeds = torch.cat([cur_input_embeds, cur_input_embeds], dim=0)
            elif cfg != 1.0 and cfg_img != 1.0:
                cur_input_embeds = torch.cat([cur_input_embeds, cur_input_embeds, cur_input_embeds], dim=0)

            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
            outputs = self.forward_model(
                inputs_embeds=cur_input_embeds,
                past_key_values=lm_past_key_values,
                pixel_values=None,
                attention_mask=attention_mask,
                use_cache=True,
            )
            lm_past_key_values = outputs.past_key_values
            z = outputs.last_hidden_state[:, -1:]
            
            if return_tokens_id:
                tokens_id_sample = torch.cat([tokens_id_sample, _get_tokens_id(z)[0]], dim=1)
                tokens_id_max = torch.cat([tokens_id_max, _get_tokens_id(z)[1]], dim=1)

        if not return_tokens_id:
            tokens_id_sample = None
            tokens_id_max = None

        mean = None
        std = None
        mean_per_step = None
        std_per_step = None
        if return_mean_std:
            mean = tokens.flatten(-len(tokens.shape) + 1).mean(dim=-1)
            std = tokens.flatten(-len(tokens.shape) + 1).std(dim=-1)
            mean_per_step = tokens.mean(dim=-1)
            std_per_step = tokens.std(dim=-1)

        # unpatchify
        tokens = self.unpatchify(unnormed_tokens, h, w)

        return ImageTokensOutput(
            image_tokens=tokens,
            conds=torch.stack(conds, dim=1),
            tokens_id_sample=tokens_id_sample,
            tokens_id_argmax=tokens_id_max,
            mean=mean,
            std=std,
            mean_per_step=mean_per_step,
            std_per_step=std_per_step,
        )

    @torch.no_grad()
    def generate(self, inputs: torch.LongTensor = None, **kwargs):
        input_ids = kwargs.pop("input_ids")
        pixel_values = kwargs.pop("pixel_values", None)
        inputs_embeds = self.prepare_inputs_embeds(input_ids, pixel_values)
        return super().generate(inputs=inputs, input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)
