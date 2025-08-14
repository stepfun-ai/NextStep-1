import math
import torch
import torch.nn as nn

from torch.utils.checkpoint import checkpoint

from transformers import Qwen2Model, Qwen2Config
from diffusers.utils.torch_utils import randn_tensor

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
        super(FlowMatchingHead, self).__init__()
        self.input_dim = input_dim
        self.net = SimpleMLPAdaLN(input_dim=input_dim, cond_dim=cond_dim, dim=dim, layers=layers, mlp_ratio=mlp_ratio)

    @property
    def dtype(self):
        return self.net.input_proj.weight.dtype

    @property
    def device(self):
        return self.net.input_proj.weight.device

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
        # timesteps = torch.tensor([1.0000, 0.9601, 0.9133, 0.8577, 0.7904, 0.7073, 0.6022, 0.4649, 0.2780, 0.0089, 0.0000], device=c.device)
        # timesteps = 1 - timesteps
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

class NextStepConfig(Qwen2Config):
    model_type = "stepx_llama"

    def __init__(
        self,
        vae_name_or_path: str | None = None,
        latent_size: int = 32,
        latent_patch_size: int = 2,
        latent_channels: int = 16,
        boi: int | None = None,
        eoi: int | None = None,
        image_placeholder_id: int | None = None,
        pad_token_id_added: int | None = None,
        lm_loss_weight: float = 0.01,
        im_loss_weight: float = 1.0,
        fm_head_dim: int = 1536,
        fm_head_layers: int = 12,
        fm_head_batch_mul: int = 4,
        o_attention_bias: bool | None = None,
        attn_implementation: str | None = None,  # Add flash attention support
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Image-related parameters
        self.vae_name_or_path = vae_name_or_path
        self.latent_size = latent_size
        self.latent_patch_size = latent_patch_size
        self.latent_channels = latent_channels

        # Special token IDs
        self.boi = boi
        self.eoi = eoi
        self.image_placeholder_id = image_placeholder_id
        self.pad_token_id_added = pad_token_id_added

        # Loss weights
        self.lm_loss_weight = lm_loss_weight
        self.im_loss_weight = im_loss_weight

        # Flow Matching Head parameters
        self.fm_head_dim = fm_head_dim
        self.fm_head_layers = fm_head_layers
        self.fm_head_batch_mul = fm_head_batch_mul

        # Attention bias
        self.o_attention_bias = o_attention_bias

        # Flash attention support
        self._attn_implementation = attn_implementation

class NextStep(Qwen2Model):
    config_class = NextStepConfig

    def __init__(self, config: NextStepConfig, enable_gradient_checkpointing: bool = False):
        super().__init__(config)

        # Initialize projectors and heads
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        token_dim = config.latent_channels * config.latent_patch_size**2

        # Image projectors
        self.image_in_projector = self._create_linear(token_dim, config.hidden_size, config.initializer_range)
        self.image_out_projector = self._create_linear(config.hidden_size, config.hidden_size, config.initializer_range)

        # Flow Matching Head
        self.image_head = FlowMatchingHead(
            input_dim=token_dim,
            cond_dim=config.hidden_size,
            dim=config.fm_head_dim,
            layers=config.fm_head_layers,
        )

        self.gradient_checkpointing = False
        # Cache first trainable parameter for establishing gradient connection (avoid traversing each time)
        self._first_trainable_param = None

        if enable_gradient_checkpointing:
            try:
                self.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                print("Enabled gradient checkpointing (use_reentrant=False) at init for NextStep model")
            except Exception as e:
                print(f"Enable gradient checkpointing failed at init: {e}")

    def gradient_checkpointing_enable(self, **kwargs):
        super().gradient_checkpointing_enable(**kwargs)
        self.image_head.net.grad_checkpointing = True
        self.gradient_checkpointing = True

    def _get_first_trainable_param(self):
        """Get reference to first trainable parameter (lazy initialization to avoid LoRA not yet added during __init__)"""
        if self._first_trainable_param is None:
            for param in self.parameters():
                if param.requires_grad:
                    self._first_trainable_param = param
                    break
        return self._first_trainable_param

    def _ensure_gradient_connection(self, hidden_states):
        """Ensure hidden_states has gradient connection (important for LoRA + gradient checkpointing)"""
        if self.training and not hidden_states.requires_grad:
            # Establish gradient connection through identity operation with trainable parameter
            # This allows gradients to propagate through LoRA layers even when base layers are frozen
            # Use cached trainable parameter to avoid traversing each time
            param = self._get_first_trainable_param()
            if param is not None:
                # Create an identity operation but establish connection through trainable parameter
                # This connects hidden_states to the computation graph
                # Use flatten()[0] instead of sum() to reduce computational overhead
                hidden_states = hidden_states + 0.0 * param.flatten()[0]
            else:
                # If no trainable parameters, directly set requires_grad
                hidden_states = hidden_states.detach().requires_grad_(True)
        return hidden_states

    def _create_linear(self, in_features: int, out_features: int, std: float) -> nn.Linear:
        """Create and initialize linear layer"""
        linear = nn.Linear(in_features, out_features)
        linear.weight.data.normal_(mean=0.0, std=std)
        linear.bias.data.zero_()
        return linear

    def patchify(self, img: torch.Tensor):
        """
        img: (bsz, C, H, W)
        x: (bsz, H * W / patch_size**2, patch_size**2 * C)
        """
        bsz, c, h, w = img.shape
        p = self.config.latent_patch_size
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
        p = self.config.latent_patch_size
        c = self.config.latent_channels
        if h is None and w is None:
            h_ = w_ = int(x.shape[1] ** 0.5)
        else:
            h_, w_ = h, w
        assert h_ * w_ == x.shape[1], f"Invalid sequence length {x.shape[1]}."

        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum("nhwcpq->nchpwq", x)
        img = x.reshape(bsz, c, h_ * p, w_ * p)
        return img

    def prepare_inputs_embeds(self, input_ids: torch.LongTensor | None = None, latents: torch.FloatTensor | None = None):
        """Prepare input embeddings, supporting mixed text and image tokens"""
        if latents is None:
            return self.embed_tokens(input_ids)

        bs, seq_length = input_ids.shape
        inputs_embeds = torch.zeros(
            (bs, seq_length, self.config.hidden_size),
            device=self.embed_tokens.weight.device,
            dtype=self.embed_tokens.weight.dtype,
        )

        im_indices = input_ids == self.config.image_placeholder_id
        lm_indices = ~im_indices

        # Process image latents
        try:
            if isinstance(latents, list):
                tokens = torch.cat([self.patchify(latent) for latent in latents], dim=1)
            else:
                tokens = self.patchify(latents)
        except Exception as e:
            tokens = latents

        image_embeds = self.image_in_projector(tokens).view(-1, self.config.hidden_size)
        token_embeds = self.embed_tokens(input_ids[lm_indices])

        inputs_embeds[im_indices] = image_embeds.to(inputs_embeds.dtype)
        inputs_embeds[lm_indices] = token_embeds

        return inputs_embeds

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                inputs_embeds=None, past_key_values=None, use_cache=None,
                output_attentions=None, output_hidden_states=None,
                return_dict=None, forward_head=False,normalize=False, **kwargs):
        """Override forward method, add gradient connection logic"""
        # If inputs_embeds is provided, ensure it has gradient connection
        if forward_head:
            x, t, c, cfg, cfg_img, cfg_mul = kwargs["x"], kwargs["t"], kwargs["c"],kwargs["cfg"], kwargs["cfg_img"], kwargs["cfg_mul"]
            return self.forward_head(x, t, c, cfg, cfg_img, cfg_mul, normalize=normalize)

        if inputs_embeds is not None:
            inputs_embeds = self._ensure_gradient_connection(inputs_embeds)

        # Call parent class forward method
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

    def forward_head(self, x, t, c, cfg, cfg_img, cfg_mul, normalize=False, **kwargs):
        # Align tensors with the actual module dtypes so Deepspeed casting does not cause matmul mismatches.
        projector_dtype = next(self.image_out_projector.parameters()).dtype
        c = c.to(projector_dtype)
        x = x.to(projector_dtype)
        t = t.to(projector_dtype)

        c = self.image_out_projector(c)
        c = c.squeeze(1)

        velocity = self.image_head.net(x, t, c)
        velocity = velocity.to(torch.float32)
        velocity = self.image_head.get_velocity_from_cfg(velocity, cfg, cfg_img, cfg_mul, normalize)
        return velocity
