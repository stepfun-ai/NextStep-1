import os
import json
import inspect
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from loguru import logger

from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL as DiffusersAutoencoderKL
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from nextstep.utils.compile_utils import compile_manager, smart_compile


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
        Convert old custom VAE key names to diffusers AutoencoderKL key names.
        Old format: encoder.down.X.block.Y -> New format: encoder.down_blocks.X.resnets.Y
        
        Decoder's up_blocks order needs to be reversed because the old implementation uses insert(0, ...).
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
