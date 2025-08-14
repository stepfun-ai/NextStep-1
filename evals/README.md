# Overview

This file guides you through running inference (image generation) using NextStep-1 series models on a variety of text-to-image benchmarks: GenEval, DPG bench, GenAI bench, WISE, and OneIG.

The generated images are formatted according to each benchmark's recommended file structure. After all files are saved, you may refer to these benchmarks' official repo guides for further evaluation.

# Environment Preparation

Clone this repo, and run:

```
# in a conda env or any venv
pip install -r ./evals/requirements.txt
```

Then, edit **RESULTS_PATH** in ``evals/config.py``, this path will host all generation results.

# Checkpoint Downloads

This eval code supports all NextStep-1 models hosted on the Hugging Face [NextStep-1 Collection](https://huggingface.co/collections/stepfun-ai/nextstep-1), which includes:

- NextStep-1.1 (we recommend using this checkpoint for better performance)
- NextStep-1.1-Pretrain
- NextStep-1.1-Pretrain-256px
- NextStep-1-Large
- NextStep-1-Large-Edit
- NextStep-1-Large-Pretrain

For example, to use the NextStep-1.1 checkpoint, run:

```
hf download stepfun-ai/NextStep-1.1 --local-dir path/to/your/ckpt_folder/NextStep_1p1
# it may take a while for downloading
```

# Run Generations

After downloading the checkpoint to ``path/to/your/ckpt_folder/NextStep_1p1``, you can run all benchmark prompts with a single command:

```
torchrun --nproc-per-node=8 evals/run.py --model_name_or_path "path/to/your/ckpt_folder/NextStep_1p1" --model_alias "nextstep_1p1"
```

The generated images will be saved in ``path/to/your/results_folder/nextstep_1p1``.