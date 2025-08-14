import os
import json
import argparse
from loguru import logger

import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

from evals.inference_main import InferenceManager
from evals.gen_images.data_utils import RepeatNoPadDistributedSampler

# GENAI_BENCH utils functions
def post_process(result_path: str):
    name_set = {}
    for file_name in os.listdir(result_path):
        file_path = os.path.join(result_path, file_name)
        file_name = file_name.split('.')[0][:-8]
        if file_name not in name_set:
            name_set[file_name] = 0
        name_set[file_name] += 1
        new_file_name = os.path.join(result_path, f"{file_name}-{name_set[file_name] - 1}.png")
        os.rename(file_path, new_file_name)


# GENAI_BENCH dataset and dataloader
class GENAIBenchDataset(Dataset):
    def __init__(self, path: str):
        self.path = path
        self.data = json.load(open(path, 'r', encoding='utf-8'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        name = list(self.data.keys())[idx]
        prompt = self.data[name]['prompt']
        return name, prompt


def get_genai_bench_dataloader(
    batch_size: int,
    path: str,
    repeat: int = 1,
    dataloader_kwargs: dict = {},
):
    def collate_fn(batch):
        names = [item[0] for item in batch]
        prompts = [item[1] for item in batch]

        return {"names": names, "prompts": prompts}

    dataset = GENAIBenchDataset(path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=RepeatNoPadDistributedSampler(dataset=dataset, repeat=repeat) if dist.is_initialized() else None,
        collate_fn=collate_fn,
        **dataloader_kwargs,
    )


# GENAI_BENCH inference manager
class GENAIBenchInferenceManager(InferenceManager):
    @InferenceManager.benchmark_register
    def run_t2i_benchmark(self, bench_name: str):
        from evals.config import T2I_BENCHMARK
        self.benchmark_name = bench_name

        self.benchmark_dir = os.path.join(self.output_dir, self.benchmark_name.lower())
        os.makedirs(self.benchmark_dir, exist_ok=True)

        loader = get_genai_bench_dataloader(
            path=T2I_BENCHMARK[bench_name],
            batch_size=8,
            repeat=4,
        )

        dist.barrier(device_ids=[self.local_rank])
        for idx, data in enumerate(loader):
            logger.info(f"\033[1m\033[31m{self.benchmark_name}\033[0m Progress: {idx + 1}/{len(loader)}")

            names, prompts = data['names'], data['prompts']
            images = self.run_t2i_inference(prompts)
            for i in range(len(images)):
                images[i].save(f"{self.benchmark_dir}/{names[i]}-rank{self.rank:0>3}.png")

        dist.barrier(device_ids=[self.local_rank])
        logger.info(f"Finished \033[1m\033[31m{self.benchmark_name}\033[0m inference at rank {self.rank}.")

        if self.rank == 0:
            post_process(self.benchmark_dir)
        dist.barrier(device_ids=[self.local_rank])


def main(args):
    manager = GENAIBenchInferenceManager(
        model_name_or_path=args.model_name_or_path,
        output_dir=args.output_dir,
    )
    manager.run_t2i_benchmark(args.bench_name)
    manager.finish_and_cleanup()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--bench_name", type=str, default="GENAI_BENCH")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)