import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from nextstep.utils.compile_utils import smart_compile
from nextstep.utils.misc import LargeInt


def modulate(x, shift, scale=None):
    if shift is None:
        return x * (1 + scale)
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
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

    @smart_compile()
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
                x = checkpoint(block, x, y, use_reentrant=True)
            else:
                x = block(x, y)

        return self.final_layer(x, y)


def expand_t(t, x):
    """Function to reshape time t to broadcastable dimension of x
    Args:
        t: [bsz,], time vector
        x: [bsz,...], data point
    """
    dims = [1] * (len(x.size()) - 1)
    t = t.view(t.size(0), *dims)
    return t


def randn_tensor(shape, noise_repeat, device, dtype=torch.float32):
    bsz = shape[0]
    if bsz % noise_repeat != 0:
        raise ValueError(f"Batch size ({bsz}) must be divisible by noise repeat ({noise_repeat})")
    _shape = (noise_repeat,) + shape[1:]
    _tensor = torch.randn(_shape, device=device, dtype=dtype).repeat(bsz // noise_repeat, 1)
    return _tensor


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

    @property
    def trainable_params(self) -> float:
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return LargeInt(n_params)

    @smart_compile()
    def forward(self, target: torch.Tensor, c: torch.Tensor, mask: torch.Tensor | None = None):
        """
        target.shape = (bsz, input_dim)
        c.shape      = (bsz, cond_dim)
        mask.shape   = (bsz,)
        """
        noise = torch.randn_like(target)

        # sample t from logit-normal distribution
        u = torch.normal(mean=0.0, std=1.0, size=(len(target),))
        t = (1 / (1 + torch.exp(-u))).to(target)

        # linear interpolation between target and noise
        xt, ut = expand_t(t, target) * target + (1 - expand_t(t, target)) * noise, target - noise

        model_output = self.net(xt, t, c)
        loss = (model_output.float() - ut.float()) ** 2
        loss = torch.mean(loss, dim=list(range(1, len(loss.size()))))  # Take the mean over all non-batch dimensions.
        loss = (loss * mask).sum() / (mask.sum() + 1e-8) if mask is not None else loss.mean()
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

    def get_velocity_from_cfg(self, velocity, cfg, cfg2, cfg_mult):
        if cfg_mult == 2:
            cond_v, uncond_v = torch.chunk(velocity, 2, dim=0)
            velocity = uncond_v + cfg * (cond_v - uncond_v)
        elif cfg_mult == 3:
            cond_v, uncond_v1, uncond_v2 = torch.chunk(velocity, 3, dim=0)
            velocity = uncond_v2 + cfg2 * (uncond_v1 - uncond_v2) + cfg * (cond_v - uncond_v1)
        return velocity

    @smart_compile(options={"triton.cudagraphs": True}, fullgraph=True)
    @torch.no_grad()
    def sample(
        self,
        c: torch.Tensor,
        cfg: float = 1.0,
        cfg2: float = 1.0,
        timesteps_shift: float = 1.0,
        num_sampling_steps: int = 20,
        last_step_size: float = 0.04,
        noise_repeat: int = 1,
    ):
        """c.shape = (bsz, cond_dim)"""
        cfg_mult = 1
        if cfg > 1.0:
            cfg_mult += 1
        if cfg2 > 1.0:
            cfg_mult += 1

        noise = randn_tensor((c.shape[0] // cfg_mult, self.input_dim), noise_repeat, self.device)

        mean_x = noise
        x = noise
        xs = []

        t0, t1 = 0, 1 - last_step_size
        timesteps = torch.linspace(t0, t1, num_sampling_steps)
        timesteps = timesteps / (timesteps_shift - (timesteps_shift - 1) * timesteps)
        for ti, tj in zip(timesteps[:-1], timesteps[1:]):
            dt = tj - ti

            combined = torch.cat([x] * cfg_mult, dim=0)
            velocity = self.net(combined.to(c.dtype), ti.expand(c.shape[0]).to(c), c)
            velocity = velocity.to(torch.float32)

            velocity = self.get_velocity_from_cfg(velocity, cfg, cfg2, cfg_mult)
            score = self.get_score_from_velocity(velocity, x, ti.expand(x.shape[0]).to(x))
            drift = velocity + (1 - expand_t(ti.expand(x.shape[0]).to(x), x)) * score

            w_cur = randn_tensor((c.shape[0] // cfg_mult, self.input_dim), noise_repeat, self.device)
            dw = w_cur * torch.sqrt(dt)

            mean_x = x + drift * dt
            x = mean_x + torch.sqrt(2 * (1 - expand_t(ti.expand(x.shape[0]).to(x), x))) * dw
            xs.append(x)

        combined = torch.cat([xs[-1]] * cfg_mult, dim=0)
        velocity = self.net(combined.to(c.dtype), timesteps[-1].expand(c.shape[0]).to(c), c)
        velocity = velocity.to(torch.float32)

        velocity = self.get_velocity_from_cfg(velocity, cfg, cfg2, cfg_mult)
        x = xs[-1] + velocity * last_step_size
        xs.append(x)

        if len(xs) != num_sampling_steps:
            raise ValueError(f"Samples ({len(xs)}) does not match the number of steps ({num_sampling_steps})")

        return xs[-1].to(c.dtype)

    @smart_compile(options={"triton.cudagraphs": True}, fullgraph=True)
    @torch.no_grad()
    def sample_new(
        self,
        c: torch.Tensor,
        cfg: float = 1.0,
        cfg2: float = 1.0,
        timesteps_shift: float = 1.0,
        num_sampling_steps: int = 20,
        last_step_size: float = 0.04,
        noise_repeat: int = 1,
    ):
        z = c
        cfg_mult = 1
        if cfg > 1.0:
            cfg_mult += 1
        if cfg2 > 1.0:
            cfg_mult += 1

        bsz = z.shape[0]
        device, dtype = z.device, z.dtype

        noise = torch.randn(bsz // cfg_mult, self.input_dim, device=device, dtype=dtype)

        noise = torch.cat([noise] * cfg_mult, dim=0)

        mean_x = noise
        x = noise
        xs = []

        sigmas = torch.linspace(0, 1, num_sampling_steps + 1)[:-1]

        delta_timesteps = [tj - ti for ti, tj in zip(sigmas[:-1], sigmas[1:])]

        sigmas = sigmas / (timesteps_shift - (timesteps_shift - 1) * sigmas)

        sigmas = torch.cat([sigmas, torch.ones(1, device=sigmas.device)])

        timesteps = sigmas

        delta_timesteps = [tj - ti for ti, tj in zip(timesteps[:-1], timesteps[1:])]

        for step, (ti, tj) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            dt = tj - ti
            w_cur = torch.randn(x.size(), device=device, dtype=dtype)
            dw = w_cur * torch.sqrt(dt)

            split_x = x[: len(x) // cfg_mult]
            combined = torch.cat([split_x] * cfg_mult, dim=0)
            t = torch.ones(x.size(0)).to(x) * ti
            model_output = self.net(combined, t, z)

            eps, rest = model_output[:, : self.input_dim], model_output[:, self.input_dim :]
            if cfg_mult == 2:
                cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
                half_eps = uncond_eps + cfg * (cond_eps - uncond_eps)
                # eps = torch.cat([half_eps, half_eps], dim=0)
                eps = torch.cat([half_eps, uncond_eps], dim=0)

            elif cfg_mult == 3:
                cond_eps, uncond_eps1, uncond_eps2 = torch.split(eps, len(eps) // 3, dim=0)
                third_eps = uncond_eps2 + cfg2 * (uncond_eps1 - uncond_eps2) + cfg * (cond_eps - uncond_eps1)
                eps = torch.cat([third_eps, third_eps, third_eps], dim=0)

            velocity = torch.cat([eps, rest], dim=1)
            score = self.get_score_from_velocity(velocity, x, t)
            drift = velocity + (1 - expand_t(t, x)) * score

            mean_x = x + drift * dt
            x = mean_x + torch.sqrt(2 * (1 - expand_t(t, x))) * dw
            xs.append(x)

        if len(xs) != num_sampling_steps:
            raise ValueError(f"Samples ({len(xs)}) does not match the number of steps ({num_sampling_steps})")

        return xs[-1][:1]
