# NextStep-1: Toward Autoregressive Image Generation with Continuous Tokens at Scale

<div align="center">

[![Homepage](https://img.shields.io/static/v1?label=Homepage&message=Project%20Page&color=blue&logo=home)](https://stepfun.ai/research/en/nextstep1)&nbsp;[![huggingface weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-StepFun/NextStep1-yellow)](https://huggingface.co/collections/stepfun-ai/nextstep-1-689d80238a01322b93b8a3dc)&nbsp;[![arXiv:2508.10711](https://img.shields.io/badge/arXiv-2508.10711-b31b1b.svg)](https://arxiv.org/abs/2508.10711)&nbsp;[![Blog](https://img.shields.io/badge/Blog-NextStep1-blue)](https://stepfun-ai.github.io/NextStep-1/nextstep_1_blog/)&nbsp;[![Blog](https://img.shields.io/badge/Blog-NextStep1.1-blue)](https://stepfun-ai.github.io/NextStep-1/nextstep_1p1_blog/)

</div>

> Autoregressive models—generating content step-by-step like reading a sentence—excel in language but struggle with images. Traditionally, they either depend on costly diffusion models or compress images into discrete, lossy tokens via vector quantization (VQ).
>
> NextStep-1 takes a different path: a 14B-parameter autoregressive model that works directly with continuous image tokens, preserving the full richness of visual data. It models sequences of discrete text tokens and continuous image tokens jointly—using a standard LM head for text and a lightweight 157M-parameter flow matching head for visuals. This unified next-token prediction framework is simple, scalable, and capable of producing stunningly detailed images.

<div align="center">
<img width="720" alt="t2i_demo" src="./assets/t2i_demo.gif">
</div>

<div align="center">
<img width="720" alt="edit_demo" src="./assets/edit_demo.gif">
</div>

## 🔥 News

- **Feb. 25, 2026**: **vLLM-Omni** supports high performance inference of NextStep-1.1. Please check [here](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/examples/offline_inference/text_to_image/?h=nextstep#nextstep-models) for details!

- **Feb. 16, 2026**: The training code of NextStep-1 (this repo) and the post-training blogs of NextStep-1.1 ([link](https://stepfun-ai.github.io/NextStep-1/nextstep_1p1_blog/)) have been released. Welcome to discuss and contribute. Happy Chinese New Year!

- **Feb. 6, 2026**: NextStep-1 has been selected as **Oral Presentation** by ICLR 2026! 🎉🎉🎉

- **Dec. 24, 2025**: 🔥 We release **NextStep-1.1**, a text-to-image model that substantially elevates output quality through extended training and a Flow-based Reinforcement Learning (RL) post-training paradigm. Feel free to try with checkpoints hosted on our [HF repo](https://huggingface.co/stepfun-ai/NextStep-1.1)!

  Checkpoints are available on:
  - 🤗 **Hugging Face**:
    - Pretrain: [NextStep-1.1-Pretrain](https://huggingface.co/stepfun-ai/NextStep-1.1-Pretrain)
    - Post-train: [NextStep-1.1](https://huggingface.co/stepfun-ai/NextStep-1.1)
  - 🇨🇳 **ModelScope**:
    - Pretrain: [NextStep-1.1-Pretrain](https://modelscope.cn/models/stepfun-ai/NextStep-1.1-Pretrain)
    - Post-train: [NextStep-1.1](https://modelscope.cn/models/stepfun-ai/NextStep-1.1)

- **Aug. 18, 2025**: 👋 We deploy NextStep-1-Large-Edit on [HuggingFace Spaces](https://huggingface.co/spaces/stepfun-ai/NextStep-1-Large-Edit). Feel free to try it out!

- **Aug. 18, 2025**: 👋 We open the [WeChat Group](./assets/wechat.png). Feel free to join us!

  <div align="center">
  <img width="360" alt="wechat" src="./assets/wechat.png">
  </div>

- **Aug. 14, 2025**: 👋 We release the inference code and [huggingface model weights](https://huggingface.co/collections/stepfun-ai/nextstep-1-689d80238a01322b93b8a3dc) of NextStep-1-Large-Pretrain, NextStep-1-Large and NextStep-1-Large-Edit

- **Aug. 14, 2025**: 👋 We have made our [technical report](https://arxiv.org/abs/2508.10711) available as open source.

---

## 📑 Table of Contents

- [🔥 News](#-news)
- [📦 Installation & Environment](#-installation--environment)
- [📥 Model & Data Preparation](#-model--data-preparation)
  - [2.1 Download Model Weights](#21-download-model-weights)
  - [2.2 Download Training Datasets](#22-download-training-datasets)
  - [2.3 Process Custom Data (Optional)](#23-process-custom-data-optional)
- [🚀 Training](#-training)
  - [3.1 Start Training (via `smartrun`)](#31-start-training-via-smartrun)
  - [3.2 Override Training Parameters](#32-override-training-parameters)
  - [3.3 Inspect and Compare Configurations](#33-inspect-and-compare-configurations)
- [🔮 Inference](#-inference)
  - [4.1 Convert Checkpoint Format](#41-convert-checkpoint-format)
  - [4.2 Run Inference](#42-run-inference)
- [📚 References](#-references)
- [📄 License](#-license)
- [📖 Citation](#-citation)

---

## 📦 Installation & Environment

### 1.1 Clone the Repository

```bash
git clone https://github.com/stepfun-ai/NextStep-1
cd NextStep-1
```

### 1.2 Create Conda Environment

```bash
conda create -n nextstep python=3.10 -y
conda activate nextstep
```

### 1.3 Install Dependencies

> ⚠️ **Note**: Pre-installing PyTorch based on your CUDA version is recommended.

```bash
pip install uv
uv pip install -e .
```

> ☕ **Tip**: This installation may take a while. Grab a cup of coffee and take a break! ☕

### 1.4 Built-in CLI Tools

The following CLI tools are available after installation:

- **`smartrun`**: An intelligent distributed launcher that automatically wraps `torchrun` parameters.
- **`gen_meta`**: Scans datasets to generate metadata indices (sample counts, checksums, etc.).
- **`warmup_data`**: Pre-warms and caches data indices to significantly speed up training startup.
- **`eshow`**: Inspect or compare experiment configurations.
- **`singlegpu_debug` / `multigpu_debug`**: Dedicated debug entries for remote attachment.

---

## 📥 Model & Data Preparation

### 2.1 Download Model Weights

Download models to `./nextstep_models`. Please update the corresponding paths in `nextstep/model_zoos.py`.

```bash
bash download_models.sh
```
> ☕ **Tip**: This download may take a while. Grab a cup of coffee and take a break! ☕

#### Available Models

The following table lists all available models and their training stages:

| Model | Pre-Training 256px | Pre-Training 512px | Annealing | RL | Visual Diversity | Fine-Tunability | Hugging Face |
|-------|-------------------|-------------------|----------|----|-----------|------------------|--------------|
| **NextStep-1-f8ch16-Tokenizer** | ❌ | ❌ | ❌ | ❌ | - | - | [![🤗](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/stepfun-ai/NextStep-1-f8ch16-Tokenizer) |
| **NextStep-1.1-Pretrain-256px** | ✅ | ❌ | ❌ | ❌ | High | Easy | [![🤗](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/stepfun-ai/NextStep-1.1-Pretrain-256px) |
| **NextStep-1.1-Pretrain** | ✅ | ✅ | ✅ | ❌ | Medium | Medium | [![🤗](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/stepfun-ai/NextStep-1.1-Pretrain) |
| **NextStep-1.1** | ✅ | ✅ | ✅ | ✅ | Low | Hard | [![🤗](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/stepfun-ai/NextStep-1.1) |
| **NextStep-1-Large-Pretrain** | ✅ | ✅ | ✅ | ❌ | High | Medium | [![🤗](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/stepfun-ai/NextStep-1-Large-Pretrain) |
| **NextStep-1-Large** | ✅ | ✅ | ✅ | ✅ | Low | Hard | [![🤗](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/stepfun-ai/NextStep-1-Large) |
| **NextStep-1-Large-Edit** | ✅ | ✅ | ✅ | ✅ | Low | Hard | [![🤗](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/stepfun-ai/NextStep-1-Large-Edit) |

> ⚠️ **Note**: The models of NextStep-1 series are from the old version. Their performance is not as good as NextStep-1.1, so we do not recommend using them. Please use NextStep-1.1 series models instead.

> 💡 **Quick Inference**: If you want to quickly inference the model, refer to the inference script below.

```bash
python3 inference/inference.py
```

### 2.2 Download Training Datasets

Download datasets to `./nextstep_data`.

```bash
bash download_datasets.sh
```
> ☕ **Tip**: This download may take a while. Grab a cup of coffee and take a break! ☕

> ⚠️ **Important Note**: The datasets provided in `download_datasets.sh` are only example open-source datasets for demonstration purposes. NextStep's actual training utilized approximately **1 billion images** from proprietary in-house data sources that cannot be open-sourced. To achieve optimal training results, we strongly recommend collecting and preparing your own large-scale datasets following the data processing guidelines in section 2.3.

### 2.3 Process Custom Data (Optional)

> 💡 **Skip this section** if you are only using the default datasets from step 2.2. Follow these steps to process custom data:

#### 2.3.1 Data Processing

Convert raw data into the unified WebDataset (Tar) format.

```bash
python3 nextstep/data/build_wds.py
```

**Data Specification** (generates `assets/idx_0000_0000.tar`):

- **`key.json`**: Must contain a `caption` field using `<image_n>` placeholders to define the interleaved sequence.
- **`key-{i}.png`**: Images must be named `key-0.png`, `key-1.png`, etc., matching the placeholders in the JSON.
- ⚠️ **Important**: The `key` must **NOT** contain dots (`.`) or hyphens (`-`). You must use the `build_wds.py` script to ensure correct indexing. **Modify `load_data` and `create_example` in the script to fit your specific data source.**

#### 2.3.2 Metadata Generation

Calculate sample counts for each Tar file to build training indices.

```bash
gen_meta /path/to/your/dataset/root_dir
```

> 💡 After completion, update `configs/data/pretrain_data.json` and the corresponding Python data config files in `configs/data` with the new data.

#### 2.3.3 Warmup Indices

Recommended for large-scale training to cache indices locally.

```bash
warmup_data /path/to/your/dataset/root_dir --n_jobs 32
```

#### 2.3.4 Data Visualization

Preview data distribution and content in Tar files or configurations.

```bash
streamlit run nextstep/service/_preview.py --server.port 8501
```

#### 2.3.5 W&B Credentials

Create a `.config` file in the root directory for experiment tracking. API key can be found at https://wandb.ai/settings

```text
WANDB_MODE=online
WANDB_API_KEY=YOUR_WANDB_API_KEY
WANDB_BASE_URL=https://api.wandb.ai
```

---

## 🚀 Training

> ⚠️ **Before training**, please carefully review the configurations in the `configs` directory. You may need to modify the model or output paths in the configuration files.

### 3.1 Start Training (via `smartrun`)

**Option 1**: Start with the NextStep-1.1-Pretrain-256px model with small training steps (~10K)

```bash
smartrun -m configs.nextstep_qwen14b_512px
```

> 💡 This command automatically utilizes all available machine resources. If you run this command on a single machine, it is equivalent to: `torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 -m configs.nextstep_qwen14b_512px`

**Option 2**: Start with the Qwen2.5-14B model with very large training steps (~500K)

```bash
smartrun -m configs.nextstep_qwen14b_256px
```

### 3.2 Override Training Parameters

Override specific parameters during training:

```bash
smartrun -m configs.nextstep_qwen14b_512px \
  training.max_steps=1000 \
  training.save_steps=200 \
  data.num_workers=2
```

### 3.3 Inspect and Compare Configurations

**View a single configuration:**

```bash
eshow configs/nextstep_qwen14b_512px.py
```

**Compare differences between two configurations** (e.g., 256px vs 512px):

```bash
eshow configs/nextstep_qwen14b_256px.py configs/nextstep_qwen14b_512px.py
```

> 📌 **Tips**: Adjust specific parameters, configuration files, and data paths according to your situation. For detailed explanations, see [`configs/README.md`](./configs/README.md).

---

## 🔮 Inference

### 4.1 Convert Checkpoint Format

Convert DeepSpeed sharded checkpoints to standard HuggingFace format:

```bash
python3 nextstep/deepspeed/zero_to_fp32.py /path/to/your/trained/checkpoint_dir
```

### 4.2 Run Inference

**Basic inference:**

```bash
python3 inference/inference.py --model_name_or_path /path/to/your/trained/checkpoint_dir
```

**Quick start with default model:**

```bash
python3 inference/inference.py
```

---

## 📖 Documentation

For detailed documentation on specific modules, please refer to:

- [NextStep Package](./nextstep/README.md) - Core package overview
- [Configuration System](./configs/README.md) - Configuration files and training setup
- [Training Engine](./nextstep/engine/README.md) - Training and validation implementation
- [Models](./nextstep/models/README.md) - Model architecture and implementation
- [Datasets](./nextstep/datasets/README.md) - Dataset adapters and mixed sampling
- [Data Processing](./nextstep/data/README.md) - Data loading, indexing, and utilities
- [Service](./nextstep/service/README.md) - Data preview and visualization service
- [Utils](./nextstep/utils/README.md) - Utility functions and helpers

---

## 📚 References

### Core Frameworks

- [DeepSpeed](https://github.com/deepspeedai/DeepSpeed)
- [Transformers](https://github.com/huggingface/transformers)
- [Diffusers](https://github.com/huggingface/diffusers)

### Datasets

- [Dolma](https://huggingface.co/datasets/allenai/dolma)
- [BLIP3o](https://huggingface.co/datasets/BLIP3o/BLIP3o-60k)
- [GPT-Image-Edit](https://huggingface.co/datasets/UCSC-VLAA/GPT-Image-Edit-1.5M)
- [Multimodal Textbook](https://github.com/DAMO-NLP-SG/multimodal_textbook)

---

## 📄 License

NextStep is licensed under the Apache License 2.0. You can find the license files in the respective GitHub and HuggingFace repositories.

---

## 📖 Citation

If you find NextStep useful for your research and applications, please consider starring this repository and citing:

```bibtex
@article{nextstepteam2025nextstep1,
  title={NextStep-1: Toward Autoregressive Image Generation with Continuous Tokens at Scale},
  author={NextStep Team and Chunrui Han and Guopeng Li and Jingwei Wu and Quan Sun and Yan Cai and Yuang Peng and Zheng Ge and Deyu Zhou and Haomiao Tang and Hongyu Zhou and Kenkun Liu and Ailin Huang and Bin Wang and Changxin Miao and Deshan Sun and En Yu and Fukun Yin and Gang Yu and Hao Nie and Haoran Lv and Hanpeng Hu and Jia Wang and Jian Zhou and Jianjian Sun and Kaijun Tan and Kang An and Kangheng Lin and Liang Zhao and Mei Chen and Peng Xing and Rui Wang and Shiyu Liu and Shutao Xia and Tianhao You and Wei Ji and Xianfang Zeng and Xin Han and Xuelin Zhang and Yana Wei and Yanming Xu and Yimin Jiang and Yingming Wang and Yu Zhou and Yucheng Han and Ziyang Meng and Binxing Jiao and Daxin Jiang and Xiangyu Zhang and Yibo Zhu},
  journal={arXiv preprint arXiv:2508.10711},
  year={2025}
}
```
