# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tokenization classes for NextStep Llama."""

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
DEFAULT_BOS_TOKEN = "<|beginoftext|>"

DEFAULT_IMAGE_TOKEN_TEMPLATE = "<|image_token_{token_id:0>6d}|>"
DEFAULT_EOL_TOKEN = "<|image_end_of_line|>"

additional_special_tokens = [
    DEFAULT_IMAGE_AREA_TOKEN,
    DEFAULT_BOI_TOKEN,
    DEFAULT_EOI_TOKEN,
    DEFAULT_IMAGE_PLACEHOLDER_TOKEN,
    DEFAULT_BOPR_TOKEN,
    DEFAULT_EOPR_TOKEN,
    DEFAULT_BOT_TOKEN,
    DEFAULT_EOT_TOKEN,
    DEFAULT_BOS_TOKEN
]

special_tokens_dict = dict(
    pad_token=DEFAULT_PAD_TOKEN,
    additional_special_tokens=additional_special_tokens,
)
