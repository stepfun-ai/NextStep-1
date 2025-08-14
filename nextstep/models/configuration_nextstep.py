from transformers.models.llama.configuration_llama import LlamaConfig


class NextStepConfig(LlamaConfig):

    model_type = "nextstep"

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
        use_gen_pos_embed: bool = False,
        o_attention_bias: bool | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vae_name_or_path = vae_name_or_path

        self.latent_size = latent_size
        self.latent_patch_size = latent_patch_size
        self.latent_channels = latent_channels

        self.boi = boi
        self.eoi = eoi
        self.image_placeholder_id = image_placeholder_id
        self.pad_token_id_added = pad_token_id_added

        self.lm_loss_weight = lm_loss_weight
        self.im_loss_weight = im_loss_weight

        self.fm_head_dim = fm_head_dim
        self.fm_head_layers = fm_head_layers
        self.fm_head_batch_mul = fm_head_batch_mul
        self.use_gen_pos_embed = use_gen_pos_embed

        self.o_attention_bias = self.attention_bias if o_attention_bias is None else o_attention_bias
