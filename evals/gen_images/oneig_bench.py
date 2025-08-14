import os
import argparse
import pandas as pd
from loguru import logger
from glob import glob
from PIL import Image

import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

from evals.inference_main import InferenceManager
from evals.gen_images.data_utils import RepeatNoPadDistributedSampler

# category to subfolder name
class_item = {
    "Anime_Stylization" : "anime",
    "Portrait" : "human",
    "General_Object" : "object",
    "Text_Rendering" : "text",
    "Knowledge_Reasoning" : "reasoning",
    "Multilingualism" : "multilingualism"
}

# ONEIG_BENCH utils functions
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
        file_name = file_path.split('.')[0][:-8]
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


# ONEIG_BENCH dataset and dataloader
class ONEIGBenchDataset(Dataset):
    def __init__(self, path: str):
        self.path = path
        self.data = pd.read_csv(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        prompt = row['prompt_en'] if 'prompt_en' in row.keys() else row['prompt_cn']
        category = class_item[row['category']]
        return f"{row['id']:0>3}", prompt, category


def get_oneig_bench_dataloader(
    path: str,
    batch_size: int,
    repeat: int,
    dataloader_kwargs: dict = {},
):
    def collate_fn(batch):
        ids = [item[0] for item in batch]
        prompts = [item[1] for item in batch]
        categories = [item[2] for item in batch]

        return {"ids": ids, "prompts": prompts, "categories": categories}

    dataset = ONEIGBenchDataset(path=path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=RepeatNoPadDistributedSampler(dataset=dataset, repeat=repeat) if dist.is_initialized() else None,
        collate_fn=collate_fn,
        **dataloader_kwargs,
    )


# ONEIG_BENCH inference manager
class ONEIGBenchInferenceManager(InferenceManager):
    @InferenceManager.benchmark_register
    def run_t2i_benchmark(self, bench_name: str):
        from evals.config import T2I_BENCHMARK
        self.benchmark_name = bench_name

        subfolder = "en" if bench_name.endswith("_EN") else "zh"
        self.benchmark_dir = os.path.join(self.output_dir, "oneig", subfolder)
        os.makedirs(self.benchmark_dir, exist_ok=True)

        loader = get_oneig_bench_dataloader(
            path=T2I_BENCHMARK[bench_name],
            batch_size=4,
            repeat=4,
        )

        dist.barrier(device_ids=[self.local_rank])
        for idx, data in enumerate(loader):
            logger.info(f"\033[1m\033[31m{self.benchmark_name}\033[0m Progress: {idx + 1}/{len(loader)}")

            ids, prompts, categories = data['ids'], data['prompts'], data['categories']
            images = self.run_t2i_inference(prompts)
            for i in range(len(images)):
                os.makedirs(os.path.join(self.benchmark_dir, categories[i]), exist_ok=True)
                images[i].save(f"{self.benchmark_dir}/{categories[i]}/{ids[i]}_rank{self.rank:0>3}.png")

        dist.barrier(device_ids=[self.local_rank])
        logger.info(f"Finished \033[1m\033[31m{self.benchmark_name}\033[0m inference at rank {self.rank}.")

        post_process(self.benchmark_dir, self.rank, self.world_size)
        dist.barrier(device_ids=[self.local_rank])


def main(args):
    manager = ONEIGBenchInferenceManager(
        model_name_or_path=args.model_name_or_path,
        output_dir=args.output_dir,
    )
    manager.run_t2i_benchmark(args.bench_name)
    manager.finish_and_cleanup()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--bench_name", type=str, default="ONEIG_BENCH_EN")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)