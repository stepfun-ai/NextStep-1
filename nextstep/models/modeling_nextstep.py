import inspect
import json
import os
from dataclasses import dataclass

import numpy as np
import torch
from safetensors.torch import safe_open
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from nextstep.models.configuration_nextstep import NextStepConfig
from nextstep.models.modeling_fm_head import FlowMatchingHead
from nextstep.models.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding
from nextstep.utils.compile_utils import smart_compile
from nextstep.utils.loguru import logger
from nextstep.utils.misc import LargeInt


@dataclass
class NextStepOutputWithPast(CausalLMOutputWithPast):
    lm_loss: torch.FloatTensor | None = None
    im_loss: torch.FloatTensor | None = None


class NextStepPreTrainedModel(PreTrainedModel):
    config_class = NextStepConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def trainable_params(self) -> float:
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return LargeInt(n_params)


class NextStep(NextStepPreTrainedModel):

    def __init__(self, config: NextStepConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        self.layers = nn.ModuleList([LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

        token_dim = self.config.latent_channels * self.config.latent_patch_size**2

        # HACK: this is hack
        self.register_buffer(
            "gen_pos_embed",
            torch.from_numpy(get_2d_sincos_pos_embed(self.config.hidden_size, 32)).float().unsqueeze(0),
        )

        self.image_in_projector = nn.Linear(token_dim, config.hidden_size)
        self.image_in_projector.weight.data.normal_(mean=0.0, std=config.initializer_range)
        self.image_in_projector.bias.data.zero_()

        self.image_out_projector = nn.Linear(config.hidden_size, config.hidden_size)
        self.image_out_projector.weight.data.normal_(mean=0.0, std=config.initializer_range)
        self.image_out_projector.bias.data.zero_()

        self.image_head = FlowMatchingHead(
            input_dim=token_dim,
            cond_dim=config.hidden_size,
            dim=config.fm_head_dim,
            layers=config.fm_head_layers,
        )

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def load_lm_head(self, lm_head_dir: str | None = None):
        index_json_file = os.path.join(lm_head_dir, "model.safetensors.index.json")
        head_weight_name = "lm_head.weight" if not self.config.tie_word_embeddings else "model.embed_tokens.weight"
        if os.path.exists(index_json_file):
            with open(index_json_file, "r") as f:
                index = json.load(f)
            model_name = index["weight_map"][head_weight_name]
        else:
            model_name = "model.safetensors"
        with safe_open(os.path.join(lm_head_dir, model_name), framework="pt") as f:
            loaded_weight = f.get_tensor(head_weight_name)
            loaded_weight = loaded_weight.to(dtype=self.lm_head.weight.dtype, device=self.lm_head.weight.device)
            self.lm_head.weight.data.copy_(loaded_weight)

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
        if latents is None:
            if not self.training:
                return self.embed_tokens(input_ids)
            else:  # dummy forward for image pass, for the consistent shape of gradient.
                raise NotImplementedError("Dummy forward for image pass is not implemented.")
        else:
            bs, seq_length = input_ids.shape
            inputs_embeds = torch.zeros(
                (bs, seq_length, self.config.hidden_size),
                device=self.embed_tokens.weight.device,
                dtype=self.embed_tokens.weight.dtype,
            )
            im_indices = input_ids == self.config.image_placeholder_id
            lm_indices = ~im_indices

            if isinstance(latents, list):
                tokens = torch.cat([self.patchify(latent) for latent in latents], dim=1)
            else:
                tokens = self.patchify(latents)
                tokens = tokens.reshape(1, -1, tokens.shape[-1])

            image_embeds = self.image_in_projector(tokens)
            image_embeds = image_embeds.view(-1, self.config.hidden_size)

            token_embeds = self.embed_tokens(input_ids[lm_indices])

            inputs_embeds[im_indices] = image_embeds.to(inputs_embeds.dtype)
            inputs_embeds[lm_indices] = token_embeds

            return inputs_embeds

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor) else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(causal_mask.device)
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(padding_mask, min_dtype)

        return causal_mask

    @smart_compile()
    def forward_model(
        self,
        inputs_embeds: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
    ) -> tuple | BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        position_ids = cache_position.unsqueeze(0)

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
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
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

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        latents: torch.FloatTensor | list[torch.FloatTensor] = None,
        latents_mask: torch.LongTensor | None = None,  # (bsz,)
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple | CausalLMOutputWithPast:

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.prepare_inputs_embeds(input_ids, latents)

        outputs = self.forward_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]

        if not self.training:
            logits = self.lm_head(hidden_states[:, -1:, :])
            return NextStepOutputWithPast(
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        # last hidden state will not be calculated to next token prediction loss
        hidden_states = hidden_states[..., :-1, :].contiguous()

        im_indices = input_ids == self.config.image_placeholder_id
        # label will determine which head (`lm_head` or `image_head`) to use
        shift_im_indices = im_indices[..., 1:].contiguous()
        shift_lm_indices = ~shift_im_indices

        ###########################################################################
        # LM Loss
        ###########################################################################
        lm_hidden_states = hidden_states[shift_lm_indices].contiguous()
        shift_lm_logits = self.lm_head(lm_hidden_states)
        shift_lm_logits = shift_lm_logits.float().view(-1, self.config.vocab_size)

        shift_labels = labels[..., 1:].contiguous()
        shift_lm_labels = shift_labels[shift_lm_indices]
        shift_lm_labels = shift_lm_labels.view(-1).to(shift_lm_logits.device)

        loss_fct = CrossEntropyLoss()
        lm_loss = loss_fct(shift_lm_logits, shift_lm_labels)

        ###########################################################################
        # IM Loss
        ###########################################################################
        im_loss = torch.tensor(0.0, device=lm_loss.device)
        if latents is not None and self.config.im_loss_weight > 0.0:
            im_hidden_states = hidden_states[shift_im_indices].contiguous()  # bsz*seq_len, dim

            if isinstance(latents, list):
                gt_latents_list = []
                im_loss_mask_list = []
                im_hidden_states_list = []

                sidx = 0
                for index, latent in enumerate(latents):
                    gt_latent = self.patchify(latent).clone().detach()  # 1, seq_len, ch
                    gt_latents_list.append(gt_latent.squeeze(0))

                    im_loss_mask_list.append(latents_mask[index].expand(gt_latent.shape[1]))

                    # bsz=1, seq_len, _
                    im_hidden_state = im_hidden_states[sidx : sidx + gt_latent.shape[1]].unsqueeze(0)
                    sidx += gt_latent.shape[1]

                    im_hidden_states_list.append(im_hidden_state.squeeze(0))

                gt_latents = torch.cat(gt_latents_list, dim=0)
                im_loss_mask = torch.cat(im_loss_mask_list, dim=0)
                im_hidden_states = torch.cat(im_hidden_states_list, dim=0)

            else:
                gt_latents = self.patchify(latents).clone().detach()  # bsz, seq_len, ch
                bsz, seq_len, _ = gt_latents.shape

                im_loss_mask = latents_mask.unsqueeze(-1).expand(-1, seq_len)

                # bsz, seq_len, dim
                im_hidden_states = im_hidden_states.reshape(bsz, seq_len, im_hidden_states.shape[-1])

                gt_latents = gt_latents.reshape(bsz * seq_len, -1)
                im_loss_mask = im_loss_mask.reshape(bsz * seq_len)
                im_hidden_states = im_hidden_states.reshape(bsz * seq_len, -1)

            gt_latents = gt_latents.repeat(self.config.fm_head_batch_mul, 1)
            im_loss_mask = im_loss_mask.repeat(self.config.fm_head_batch_mul)
            im_hidden_states = im_hidden_states.repeat(self.config.fm_head_batch_mul, 1)

            im_loss = self.image_head(target=gt_latents.to(im_hidden_states.dtype), c=im_hidden_states, mask=im_loss_mask)

        if torch.isnan(lm_loss) or torch.isinf(lm_loss):
            raise ValueError(f"LM Loss is {lm_loss}, stopping training!")
        if torch.isnan(im_loss) or torch.isinf(im_loss):
            raise ValueError(f"IM Loss is {im_loss}, stopping training!")

        loss = lm_loss * self.config.lm_loss_weight + im_loss * self.config.im_loss_weight

        return NextStepOutputWithPast(
            loss=loss,
            lm_loss=lm_loss.detach(),
            im_loss=im_loss.detach(),
            logits=None,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Cache | None = None,
        attention_mask: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ):
        """
        Prepare the model inputs for generation. In includes operations like computing the 4D attention mask or
        slicing inputs given the existing cache.

        See the forward pass in the model documentation for expected arguments (different models might have different
        requirements for e.g. `past_key_values`). This function should work as is for most LLMs.
        """

        # 1. Handle BC:
        model_inputs = {}
        # - some models don't have `Cache` support (which implies they don't expect `cache_position` in `forward`)
        if self._supports_cache_class:
            model_inputs["cache_position"] = cache_position
        # - `cache_position` was not a mandatory input in `prepare_inputs_for_generation` for those models, and this
        #   function may be called outside of `generate`. Handle most use cases by creating `cache_position` on the fly
        #   (this alternative is not as robust as calling `generate` and letting it create `cache_position`)
        elif cache_position is None:
            past_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
            cache_position = torch.arange(past_length, input_ids.shape[1], dtype=torch.long, device=input_ids.device)

        # 2. Generic cache-dependent input preparation
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        # Exception 3: with synced GPUs cache_position may go out of bounds, but we only want dummy token in that case
        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values
            if inputs_embeds is not None or cache_position[-1] >= input_ids.shape[1]:  # Exception 1 or Exception 3
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        # 3. Prepare base model inputs
        input_ids_key = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if not self.config.is_encoder_decoder:
            if inputs_embeds is not None and cache_position[0] == 0:
                model_inputs[input_ids_key] = None
                model_inputs["inputs_embeds"] = inputs_embeds
            else:
                # `clone` calls in this function ensure a consistent stride. See #32227
                model_inputs[input_ids_key] = input_ids.clone(memory_format=torch.contiguous_format)
                model_inputs["inputs_embeds"] = None
        else:
            model_inputs[input_ids_key] = input_ids.clone(memory_format=torch.contiguous_format)

        # 4. Create missing `position_ids` on the fly
        if (
            attention_mask is not None
            and kwargs.get("position_ids") is None
            and "position_ids" in set(inspect.signature(self.forward).parameters.keys())
        ):
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            kwargs["position_ids"] = position_ids  # placed in kwargs for further processing (see below)

        # 5. Slice model inputs if it's an input that should have the same length as `input_ids`
        for model_input_name in ["position_ids", "token_type_ids"]:
            model_input = kwargs.get(model_input_name)
            if model_input is not None:
                if past_key_values:
                    model_input = model_input[:, -input_ids.shape[1] :]
                    model_input = model_input.clone(memory_format=torch.contiguous_format)
                model_inputs[model_input_name] = model_input

        # 6. Create 4D attention mask is we are using a `StaticCache` (important for performant compiled forward pass)
        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs[input_ids_key].shape
                device = model_inputs[input_ids_key].device

            # Create the causal mask with fixed shape in advance, to reduce recompilations. If the function to create
            # the 4D causal mask exists, it should be present in the base model (XXXModel class).
            base_model = getattr(self, self.base_model_prefix, None)
            if base_model is None:
                causal_mask_creation_function = getattr(self, "_prepare_4d_causal_attention_mask_with_cache_position", None)
            else:
                causal_mask_creation_function = getattr(
                    base_model, "_prepare_4d_causal_attention_mask_with_cache_position", None
                )
            if causal_mask_creation_function is None:
                logger.warning_once(
                    f"{self.__class__.__name__} has no `_prepare_4d_causal_attention_mask_with_cache_position` method "
                    "defined in its base modeling class. Compiled forward passes will be sub-optimal. If you're "
                    "writing code, see Llama for an example implementation. If you're a user, please report this "
                    "issue on GitHub."
                )
            else:
                attention_mask = causal_mask_creation_function(
                    attention_mask,
                    sequence_length=sequence_length,
                    target_length=past_key_values.get_max_cache_shape(),
                    dtype=self.dtype,
                    device=device,
                    cache_position=cache_position,
                    batch_size=batch_size,
                    config=self.config,
                    past_key_values=past_key_values,
                )
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask

        # 7. Forward ALL kwargs that are uninitialized (e.g. `use_cache`).
        for key, value in kwargs.items():
            if key not in model_inputs:
                model_inputs[key] = value

        # 8. Remove unexpected `generate` inputs (TODO @joao: fix trainer and examples)
        model_inputs.pop("labels", None)
        return model_inputs

    @torch.no_grad()
    def generate(self, inputs: torch.LongTensor = None, **kwargs):
        input_ids = kwargs.pop("input_ids")
        latents = kwargs.pop("latents", None)
        inputs_embeds = self.prepare_inputs_embeds(input_ids, latents)
        return super().generate(inputs=inputs, input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)

    def gradient_checkpointing_enable(self, **kwargs):
        super().gradient_checkpointing_enable(**kwargs)

        self.image_head.net.grad_checkpointing = True


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0, pe_interpolation=1.0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32) / pe_interpolation
    grid_w = np.arange(grid_size, dtype=np.float32) / pe_interpolation
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model_path, pe_key: str = "gen_pos_embed", new_len: int = 4096):
    state_dict = torch.load(model_path, map_location="cpu")

    pos_embed_1d = state_dict[pe_key]
    _, ori_len, embed_dim = pos_embed_1d.shape

    ori_size = int(ori_len**0.5)
    new_size = int(new_len**0.5)

    if ori_size != new_size:
        logger.info("Position interpolate from %dx%d to %dx%d" % (ori_size, ori_size, new_size, new_size))
        pos_embed_2d = pos_embed_1d.reshape(-1, ori_size, ori_size, embed_dim).permute(0, 3, 1, 2)
        pos_embed_2d = torch.nn.functional.interpolate(
            pos_embed_2d, size=(new_size, new_size), mode="bicubic", align_corners=False
        )
        pos_embed_1d = pos_embed_2d.permute(0, 2, 3, 1).flatten(1, 2)
        state_dict[pe_key] = pos_embed_1d

    torch.save(state_dict, model_path)
