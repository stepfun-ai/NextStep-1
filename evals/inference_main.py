from PIL.Image import Image
import os
import json
from argparse import ArgumentParser
from glob import glob
from PIL import Image
from loguru import logger
from datetime import timedelta
from functools import partial

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

from inference.gen_pipeline import NextStepPipeline
from evals.gen_images.data_utils import RepeatNoPadDistributedSampler

# inference utils functions
def make_grid(file_paths: list[str]):
    images = [Image.open(file_path) for file_path in file_paths]
    weight, height = images[0].size

    new_img = Image.new("RGB", (weight * 2, height * 2))
    new_img.paste(images[0], (0, 0))
    new_img.paste(images[1], (weight, 0))
    new_img.paste(images[2], (0, height))
    new_img.paste(images[3], (weight, height))

    return new_img


def post_process(result_path: str, rank: int, world_size: int):
    name_set = {}
    for file_path in glob(os.path.join(result_path, "**", "*.png"), recursive=True):
        file_name = file_path.replace(".png", "")[:-8]
        if file_name not in name_set:
            name_set[file_name] = []
        name_set[file_name].append(file_path)

    for idx, (file_name, file_paths) in enumerate(name_set.items()):
        # enabling multi-rank post-process
        if (idx % world_size) != rank:
            continue

        grid = make_grid(file_paths)
        grid.save(f"{file_name}.webp", format="WEBP", lossless=True)

        for file_path in file_paths:
            os.remove(file_path)


class EvalT2IPromptDataset(Dataset):
    def __init__(self, path: str, prompts: dict[str, str] = {}):
        self.path = path
        with open(self.path, "r") as f:
            self.prompts: dict = json.load(f)

        if prompts is not None:
            self.prompts.update(prompts)

        self.prompts = list(self.prompts.items())

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        name, prompt = self.prompts[idx]
        return name, prompt


def get_eval_t2i_prompts_dataloader(
    batch_size,
    path: str,
    prompts: dict[str, str] = {},
    repeat: int = 1,
    dataloader_kwargs: dict = {},
):
    def collate_fn(batch):
        names = [item[0] for item in batch]
        prompts = [item[1] for item in batch]

        return {"names": names, "prompts": prompts}

    dataset = EvalT2IPromptDataset(path=path, prompts=prompts)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=RepeatNoPadDistributedSampler(dataset=dataset, repeat=repeat) if dist.is_initialized() else None,
        collate_fn=collate_fn,
        **dataloader_kwargs,
    )


class InferenceManager:
    manager_name: str = "NextStep"
    methods = {}

    @classmethod
    def benchmark_register(cls, func):
        class_name = func.__qualname__.split('.')[0]
        cls.methods.setdefault(func.__name__, {})[class_name] = func
        return func

    def __init__(self, model_name_or_path: str, output_dir: str):
        self.model_name_or_path = model_name_or_path
        self.output_dir = output_dir

        logger.info(f"Initializing \033[1m\033[31m{self.manager_name}\033[0m inference manager for {self.model_name_or_path}.")

        self.init_parallel()
        self.init_pipeline()

    def init_parallel(self):
        # get distributed environment variables
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.master_addr = os.environ["MASTER_ADDR"]
        self.master_port = os.environ["MASTER_PORT"]
        self.node_rank = int(os.environ["NODE_RANK"])

        torch.cuda.set_device(self.local_rank)
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))

        logger.info(f"Setup parallel for rank {self.rank} of world_size {self.world_size} on {self.master_addr}:{self.master_port}.")

    def init_pipeline(self):
        self.pipeline = NextStepPipeline(
            model_name_or_path=self.model_name_or_path,
            vae_name_or_path=os.path.join(self.model_name_or_path, "./vae"),
        ).to(device="cuda")
        self.IMG_SIZE = 256

    @torch.no_grad()
    def run_t2i_inference(self, prompts: list[str]) -> list[Image.Image]:
        # set prompts
        positive_prompt = "masterpiece, film grained, best quality."
        negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."

        gen_func = partial(
            self.pipeline.generate_image,
            hw=(self.IMG_SIZE, self.IMG_SIZE),
            num_images_per_caption=1,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            cfg=7.5,
            cfg_img=1.0,
            cfg_schedule="constant",
            use_norm=False,
            num_sampling_steps=28,
            timesteps_shift=1.0,
            seed=42 + self.rank,
        )

        if self.IMG_SIZE <= 512:
            images = gen_func(prompts)
        else:
            images = [gen_func(prompt)[0] for prompt in prompts]

        return images

    def run_t2i_benchmark(self, bench_name: str):
        pass

    def finish_and_cleanup(self):
        del self.pipeline
        torch.cuda.empty_cache()

        dist.destroy_process_group()


def main(args):
    manager = InferenceManager(
        model_name_or_path=args.model_name_or_path,
        output_dir=args.output_dir,
    )
    manager.run_t2i_benchmark(args.bench_name)
    manager.finish_and_cleanup()


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--bench_name", type=str, default="T2I_PROMPTS_GEMINI")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
