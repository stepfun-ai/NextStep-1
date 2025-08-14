"""Tokenization classes for NextStep."""

IGNORE_INDEX = -100

# special token
DEFAULT_PAD_TOKEN = "[PAD]"

# additional special token
DEFAULT_IMAGE_AREA_TOKEN = "<|image_area|>"
DEFAULT_BOI_TOKEN = "<|begin_of_image|>"
DEFAULT_EOI_TOKEN = "<|end_of_image|>"
DEFAULT_IMAGE_PLACEHOLDER_TOKEN = "<|image_placeholder|>"
DEFAULT_BOPR_TOKEN = "<|begin_of_prompt_refinement|>"
DEFAULT_EOPR_TOKEN = "<|end_of_prompt_refinement|>"
DEFAULT_BOT_TOKEN = "<|begin_of_thinking|>"
DEFAULT_EOT_TOKEN = "<|end_of_thinking|>"

additional_special_tokens = [
    DEFAULT_IMAGE_AREA_TOKEN,
    DEFAULT_BOI_TOKEN,
    DEFAULT_EOI_TOKEN,
    DEFAULT_IMAGE_PLACEHOLDER_TOKEN,
    DEFAULT_BOPR_TOKEN,
    DEFAULT_EOPR_TOKEN,
    DEFAULT_BOT_TOKEN,
    DEFAULT_EOT_TOKEN,
]

special_tokens_dict = dict(
    pad_token=DEFAULT_PAD_TOKEN,
    additional_special_tokens=additional_special_tokens,
)
