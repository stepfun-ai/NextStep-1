# NextStep-1: Toward Autoregressive Image Generation with Continuous Tokens at Scale

<div align="center">

[![Homepage](https://img.shields.io/static/v1?label=Homepage&message=Project%20Page&color=blue&logo=home)](https://stepfun.ai/research/en/nextstep1)&nbsp;[![huggingface weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-StepFun/NextStep1-yellow)](https://huggingface.co/collections/stepfun-ai/nextstep-1-689d80238a01322b93b8a3dc)&nbsp;

</div>

> Autoregressive models—generating content step-by-step like reading a sentence—excel in language but struggle with images. Traditionally, they either depend on costly diffusion models or compress images into discrete, lossy tokens via vector quantization (VQ).
> NextStep-1 takes a different path: a 14B-parameter autoregressive model that works directly with continuous image tokens, preserving the full richness of visual data. It models sequences of discrete text tokens and continuous image tokens jointly—using a standard LM head for text and a lightweight 157M-parameter flow matching head for visuals. This unified next-token prediction framework is simple, scalable, and capable of producing stunningly detailed image
<div align="center">
<img width="720" alt="t2i_demo" src="./assets/t2i_demo.gif">
</div>
<div align="center">
<img width="720" alt="edit_demo" src="./assets/edit_demo.gif">
</div>

## 🔥 News

- Aug 14, 2025: 👋 We release the inference code and [huggingface model weights](https://huggingface.co/collections/stepfun-ai/nextstep-1-689d80238a01322b93b8a3dc) of NextStep-1-Large-Pretrain, NextStep-1-Large and NextStep-1-Large-Edit
- Aug 14, 2025: 👋 We have made our [technical report](./nextstep_1_tech_report.pdf) available as open source.


## 📑 Open-Source Plan

- [ ] Training Code
  - [ ] Data Infrastructure
  - [ ] Image Tokenizer Training Code
  - [ ] Pre-training Code
  - [ ] DPO Code
- [x] Inference Code
- [ ] Evaluation Code
- [x] Model Checkpoints
  - [x] NextStep-1-f8ch16-Tokenizer
  - [x] NextStep-1-Large-Pretrain
  - [x] NextStep-1-Large
  - [x] NextStep-1-Large-Edit

## ⚒️ Quick Start

0️⃣ Requirements

> [!NOTE]
> The codebase has been tested and verified on:
>
> - Python: 3.10
> - PyTorch: 2.5.1+cu121
> - CUDA: 12.1
> - cuDNN: 8.8.1.3
> - Flash-Attention: 2.7.4.post1
> - DeepSpeed: 0.16.3
> - Transformers: 4.49.0

1️⃣ Set up environment

```bash
git clone https://github.com/stepfun-ai/NextStep-1.git
cd NextStep
conda create -n nextstep python=3.10 -y
conda activate nextstep
pip install -e .

# (optional) install flash-attn
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

2️⃣ Download the model weights from [HuggingFace](https://huggingface.co/collections/stepfun-ai/nextstep-1-689d80238a01322b93b8a3dc).

```bash
pip install -U huggingface_hub
huggingface-cli download stepfun-ai/NextStep-1-f8ch16-Tokenizer
huggingface-cli download stepfun-ai/NextStep-1-Large
huggingface-cli download stepfun-ai/NextStep-1-Large-Edit
```

3️⃣ Inference

```python
# python example.py
import torch
from PIL import Image

from nextstep.models.pipeline_nextstep import NextStepPipeline

device = "cuda"
dtype = torch.bfloat16

# Replace with absolute dir
vae_name_or_path = "/path/to/stepfun-ai/NextStep-1-f8ch16-Tokenizer"

## Image Edit
edit_model_name_or_path = "stepfun-ai/NextStep-1-Large-Edit"

edit_pipeline = NextStepPipeline(model_name_or_path=edit_model_name_or_path, vae_name_or_path=vae_name_or_path).to(
    device, dtype
)

cfgs = [7.5, 1.5]
hw = (512, 512)

editing_caption = (
    "<image>"
    + "Add a pirate hat to the dog's head. Change the background to a stormy sea with dark clouds. Include the text 'Captain Paws' in bold white letters at the top portion of the image."
)
positive_prompt = None
negative_prompt = "Copy original image."

input_image = Image.open("./assets/dog.jpg")
input_image = input_image.resize(hw)

output_imgs = edit_pipeline.generate_image(
    captions=editing_caption,
    images=input_image,
    num_images_per_caption=1,
    positive_prompt=positive_prompt,
    negative_prompt=negative_prompt,
    hw=hw,
    use_norm=True,
    cfg=cfgs[0],
    cfg_img=cfgs[1],
    cfg_schedule="constant",
    timesteps_shift=3.2,
    num_sampling_steps=50,
    seed=42,
)
output_imgs = output_imgs[0]
output_imgs.save(f"./test_edit.png")

## Text to image
t2i_model_name_or_path = "stepfun-ai/NextStep-1-Large"
t2_pipeline = NextStepPipeline(model_name_or_path=t2i_model_name_or_path, vae_name_or_path=vae_name_or_path).to(device, dtype)

cfg = 7.5
hw = (512, 512)

caption = "A baby panda wearing an Iron Man mask, holding a board with 'NextStep-1' written on it"
positive_prompt = "masterpiece, film grained, best quality."
negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."

output_imgs = t2_pipeline.generate_image(
    captions=caption,
    images=None,
    num_images_per_caption=4,
    positive_prompt=positive_prompt,
    negative_prompt=negative_prompt,
    hw=hw,
    use_norm=True,
    cfg=cfg,
    cfg_schedule="constant",
    timesteps_shift=1.0,
    num_sampling_steps=50,
    seed=42,
)
for i, img in enumerate(output_imgs):
    img.save(f"./test_t2i_{i}.png")
```

> [!WARNING]
> **Troubleshooting**: If you encounter the following errors during inference:
>
> ```bash
> torch._inductor.exc.InductorError: ImportError: /path/to/triton_cache/cuda_utils.cpython-310-x86_64-linux-gnu.so: undefined symbol: cuModuleGetFunction
> ```
>
> or
>
> ```bash
> Triton cache error: compiled module cuda_utils.so could not be loaded
> ```
>
> You can try the following solutions:
>
> **Solution 1**: Disable torch.compile
>
> ```bash
> ENABLE_TORCH_COMPILE=false python inference.py
> ```
>
> **Solution 2**: Install CUDA version of torch
>
> ```bash
> pip install torch==2.5.1+cu121 torchvision triton --index-url https://download.pytorch.org/whl/cu121
> ```

## 🚀 Train

coming soon!

## 📊 Evaluation

coming soon!

## Acknowledgments

We would like to express our sincere thanks to theWe would like to sincerely thank Tianhong Li and Yonglong Tian for their
insightful discussions.

## LICENSE

NextStep is licensed under the Apache License 2.0. You can find the license files in the respective github and HuggingFace repositories.

## Citation

If you find NextStep useful for your research and applications, please consider starring this repository and citing:

```bibtex
@misc{nextstep_1,
    title={NextStep-1: Toward Autoregressive Image Generation with Continuous Tokens at Scale},
    author={NextStep Team},
    year={2025},
    url={https://github.com/stepfun-ai/NextStep-1},
}
```
