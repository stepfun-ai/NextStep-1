import copy
import re
from typing import Literal

import torch
import torch.amp
import torch.nn.functional as F
import torchvision.transforms as transforms
from diffusers import AutoencoderKL
from PIL import Image
from torch import nn
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers.cache_utils import Cache, StaticCache

from nextstep.models.aspect_ratio import hw2str
from nextstep.models.modeling_flux_vae import AutoencoderKL
from nextstep.models.modeling_nextstep import NextStep
from nextstep.models.tokenization_nextstep import DEFAULT_IMAGE_AREA_TOKEN
from nextstep.utils.compile_utils import compile_manager
from nextstep.utils.image_utils import ImageType, center_crop_arr, load_image, to_pil
from nextstep.utils.training_utils import set_seed


def layer_norm(input: torch.Tensor, normalized_shape: torch.Size, eps: float = 1e-6) -> torch.Tensor:
    return F.layer_norm(input, normalized_shape, None, None, eps)


class NextStepPipeline:
    def __init__(
        self,
        model_name_or_path: str | None = None,
        vae_name_or_path: str | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        model: nn.Module | None = None,
        vae: AutoencoderKL | None = None,
    ):
        if model is not None:
            self.tokenizer = copy.deepcopy(tokenizer)
            self.model = model

        elif model_name_or_path is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                local_files_only=True,
                model_max_length=512,
                padding_side="left",
                use_fast=True,
            )
            self.model: NextStep = NextStep.from_pretrained(model_name_or_path, local_files_only=True)

        else:
            raise ValueError("model or model_name_or_path is required")

        self.tokenizer.add_eos_token = False

        if vae_name_or_path is None:
            vae_name_or_path = getattr(self.model.config, "vae_name_or_path", None)

        if vae is not None:
            self.vae = vae
        elif vae_name_or_path is not None:
            self.vae = AutoencoderKL.from_pretrained(vae_name_or_path)
        else:
            raise ValueError("vae or vae_name_or_path is required")

        self.model.eval()
        self.vae.eval()

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

        self.__device = self.model.device
        self.__dtype = self.model.dtype
        self.to(self.device, self.dtype)

    @property
    def device(self):
        return self.__device

    @property
    def device_type(self):
        if isinstance(self.__device, str):
            return self.__device
        return self.__device.type

    @property
    def dtype(self):
        return self.__dtype

    def to(self, device: str | None = None, dtype: torch.dtype | None = None):
        if device is not None:
            self.__device = device
        if dtype is not None:
            self.__dtype = dtype
        self.model.to(self.__device, dtype=self.__dtype)
        self.vae.to(self.__device, dtype=self.__dtype)
        return self

    def process_images(
        self,
        images: ImageType | list[ImageType],
        crop_mode: Literal["center", "original"],
        image_size: int | None = None,
    ) -> list[Image.Image]:
        if crop_mode == "center":
            if image_size is None:
                raise ValueError("image_size is required when crop_mode is center")
            if image_size % self.down_factor != 0:
                raise ValueError(f"image_size ({image_size}) is not divisible by down_factor ({self.down_factor})")

        if not isinstance(images, list):
            images = [images]

        images = [load_image(image) for image in images]
        match crop_mode:
            case "center":
                images = [center_crop_arr(image, image_size) for image in images]
            case "original":
                raise NotImplementedError("original crop mode is not implemented")
            case _:
                raise ValueError(f"Invalid crop_mode: {crop_mode}")

        return images

    def _image_str(self, hw: tuple[int, int] = (256, 256)):
        latent_hw = (hw[0] // self.down_factor, hw[1] // self.down_factor)
        image_ids = [self.boi] + [self.image_placeholder_id] * (latent_hw[0] * latent_hw[1]) + [self.eoi]
        image_str = DEFAULT_IMAGE_AREA_TOKEN + hw2str(*latent_hw) + self.tokenizer.decode(image_ids)
        return image_str

    def _check_input(
        self, captions: str | list[str], images: Image.Image | list[Image.Image] | None
    ) -> tuple[list[str], list[Image.Image] | None]:
        if not isinstance(captions, list):
            captions = [captions]

        if images is not None:
            if not isinstance(images, list):
                images = [images]

            # Validate image count matches <image> tokens in captions
            image_token_count = 0
            for caption in captions:
                num_image_token = len(re.findall(r"<image>", caption))
                assert num_image_token == 1, f"Caption `{caption}` has {num_image_token} image tokens, but only 1 is allowed."
                image_token_count += num_image_token
            if image_token_count != len(images):
                raise ValueError(
                    f"Number of images ({len(images)}) does not match number of image tokens ({image_token_count}).\n"
                    f"Captions: {captions}"
                )

            hws = [(image.size[1], image.size[0]) for image in images]

            # Replace <image> tokens sequentially with corresponding image_str based on hw
            processed_captions = []
            image_idx = 0
            for caption in captions:
                # Process each caption
                processed_caption = caption
                num_image_tokens = processed_caption.count("<image>")

                # Replace each <image> token in order
                for _ in range(num_image_tokens):
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
        # 1. repeat captions and images
        if not isinstance(captions, list):
            captions = [captions]
        captions = [caption for caption in captions for _ in range(num_images_per_caption)]
        if images is not None:
            images = [image for image in images for _ in range(num_images_per_caption)]

        # 2. add positive prompt
        if positive_prompt is not None and positive_prompt != "":
            captions = [f"{caption} {positive_prompt}" for caption in captions]

        # 3. add negative prompt
        if negative_prompt is None:
            negative_prompt = ""
        num_samples = len(captions)
        if cfg != 1.0 and cfg_img != 1.0:  # use both image and text CFG
            w, h = images[0].size
            captions = captions + [self._image_str((h, w)) + negative_prompt] * num_samples
            images = images + images
            captions = captions + [negative_prompt] * num_samples
        elif cfg != 1.0 and cfg_img == 1.0:  # use text CFG
            captions = captions + [negative_prompt] * num_samples
        elif cfg == 1.0 and cfg_img == 1.0:
            pass

        return captions, images

    def _add_prefix_ids(self, hw: tuple[int, int], input_ids: torch.Tensor, attention_mask: torch.Tensor):
        prefix_str = DEFAULT_IMAGE_AREA_TOKEN + hw2str(hw[0] // self.down_factor, hw[1] // self.down_factor)
        prefix_output = self.tokenizer(prefix_str, truncation=False, add_special_tokens=True, return_tensors="pt")
        prefix_input_ids = prefix_output.input_ids.to(input_ids.device, dtype=input_ids.dtype)
        prefix_attention_mask = prefix_output.attention_mask.to(attention_mask.device, dtype=attention_mask.dtype)

        # remove bos token
        if self.tokenizer.bos_token is not None:
            prefix_input_ids = prefix_input_ids[:, 1:]
            prefix_attention_mask = prefix_attention_mask[:, 1:]

        # add boi token
        prefix_input_ids = torch.cat(
            [
                prefix_input_ids,
                prefix_input_ids.new_tensor([self.model.config.boi]).unsqueeze(0),
            ],
            dim=1,
        )
        prefix_attention_mask = torch.cat(
            [
                prefix_attention_mask,
                prefix_attention_mask.new_ones((prefix_attention_mask.shape[0], 1)),
            ],
            dim=1,
        )

        bsz = input_ids.shape[0]
        input_ids = torch.cat([input_ids, prefix_input_ids.expand(bsz, -1)], dim=1)
        attention_mask = torch.cat([attention_mask, prefix_attention_mask.expand(bsz, -1)], dim=1)

        return input_ids, attention_mask

    @torch.no_grad()
    def decoding(
        self,
        c: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Cache,
        max_new_len: int,
        num_images_per_caption: int,
        use_norm: bool = False,
        cfg: float = 1.0,
        cfg_img: float = 1.0,
        cfg_schedule: Literal["linear", "constant"] = "constant",
        timesteps_shift: float = 1.0,
        num_sampling_steps: int = 20,
        progress: bool = True,
    ):
        indices = list(range(max_new_len))
        indices = tqdm(indices) if progress else indices

        tokens = None
        for step in indices:
            # cfg schedule follow Muse
            if cfg_schedule == "linear":
                tokens_len = 0 if tokens is None else tokens.shape[1]
                cfg_iter = 1 + (cfg - 1) * (max_new_len - tokens_len) / max_new_len
                cfg_img_iter = 1 + (cfg_img - 1) * (max_new_len - tokens_len) / max_new_len
            elif cfg_schedule == "constant":
                cfg_iter = cfg
                cfg_img_iter = cfg_img
            else:
                raise NotImplementedError

            c = self.model.image_out_projector(c)
            token_sampled = self.model.image_head.sample(
                c=c.squeeze(1),
                cfg=cfg_iter,
                cfg2=cfg_img_iter,
                timesteps_shift=timesteps_shift,
                num_sampling_steps=num_sampling_steps,
                noise_repeat=num_images_per_caption,
            )

            if use_norm:
                token_sampled = layer_norm(token_sampled, normalized_shape=token_sampled.size()[1:])

            if tokens is not None:
                tokens = torch.cat([tokens, token_sampled.unsqueeze(1)], dim=1)
            else:
                tokens = token_sampled.unsqueeze(1)

            cur_inputs_embeds = self.model.image_in_projector(tokens[:, -1:])
            if cfg != 1.0 and cfg_img == 1.0:
                cur_inputs_embeds = torch.cat([cur_inputs_embeds, cur_inputs_embeds], dim=0)
            elif cfg != 1.0 and cfg_img != 1.0:
                cur_inputs_embeds = torch.cat([cur_inputs_embeds, cur_inputs_embeds, cur_inputs_embeds], dim=0)

            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
            outputs = self.model.forward_model(
                inputs_embeds=cur_inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            c = outputs.last_hidden_state[:, -1:]
            if self.model.config.use_gen_pos_embed:
                c = c + self.model.gen_pos_embed[:, step + 1 : step + 2, :]

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
        timesteps_shift: float = 1.0,
        num_sampling_steps: int = 20,
        seed: int = 42,
        progress: bool = True,
    ) -> list[Image.Image]:
        # 0. set seed
        if seed is not None:
            set_seed(seed)

        # 1. check input
        captions, images = self._check_input(captions, images)

        # 2. build captions
        captions, images = self._build_captions(
            captions, images, num_images_per_caption, positive_prompt, negative_prompt, cfg, cfg_img
        )

        # 3. encode images
        # `images` must be processed by `process_images` before calling this function
        latents = None
        if images is not None:
            pixel_values = [self.pil2tensor(image) for image in images]
            pixel_values = torch.stack(pixel_values).to(self.device)
            with compile_manager.compile_disabled():
                posterior = self.vae.encode(pixel_values.to(self.vae.dtype)).latent_dist
            latents = (posterior.sample() - self.shift_factor) * self.scaling_factor

        # 4. tokenize caption & add prefix ids
        output = self.tokenizer(captions, padding="longest", truncation=False, add_special_tokens=True, return_tensors="pt")
        input_ids = output.input_ids.to(self.device)
        attention_mask = output.attention_mask.to(self.device)
        input_ids, attention_mask = self._add_prefix_ids(hw, input_ids, attention_mask)

        # 5. LLM prefill
        max_new_len = (hw[0] // self.down_factor) * (hw[1] // self.down_factor)
        max_cache_len = input_ids.shape[1] + max_new_len
        past_key_values = StaticCache(
            config=self.model.config,
            max_batch_size=input_ids.shape[0],
            max_cache_len=max_cache_len,
            device=self.device,
            dtype=self.dtype,
        )
        inputs_embeds = self.model.prepare_inputs_embeds(input_ids, latents)
        with compile_manager.compile_disabled():
            outputs = self.model.forward_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
        past_key_values = outputs.past_key_values
        c = outputs.last_hidden_state[:, -1:]
        if self.model.config.use_gen_pos_embed:
            c = c + self.model.gen_pos_embed[:, 0:1, :]

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
        )

        # 7. unpatchify
        latents = self.model.unpatchify(tokens)
        latents = (latents / self.scaling_factor) + self.shift_factor

        # 8. decode latents
        with compile_manager.compile_disabled():
            sampled_images = self.vae.decode(latents.to(self.vae.dtype)).sample
        sampled_images = sampled_images.detach().cpu().to(torch.float32)
        pil_images = [to_pil(img, "11") for img in sampled_images]

        return pil_images
