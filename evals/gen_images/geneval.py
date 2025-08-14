import os
import argparse
import json
from loguru import logger

import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

from evals.inference_main import InferenceManager
from evals.gen_images.data_utils import RepeatNoPadDistributedSampler


# GenEval utils functions
def post_process(result_path: str):
    for indice_dir in os.listdir(result_path):
        sample_dir = os.path.join(result_path, indice_dir, "samples")
        file_names = os.listdir(sample_dir)
        for idx, file_name in enumerate(file_names):
            file_path = os.path.join(sample_dir, file_name)
            new_file_path = os.path.join(sample_dir, f"{idx:0>5}.png")
            os.rename(file_path, new_file_path)


# GenEval dataset and dataloader
class GenEvalDataset(Dataset):
    def __init__(self, path: str):
        self.path = path
        with open(self.path, 'r', encoding='utf-8') as fp:
            self.metadatas = [json.loads(line) for line in fp]

    def __len__(self):
        return len(self.metadatas)

    def __getitem__(self, idx):
        metadata = self.metadatas[idx]
        return idx, metadata


def get_geneval_dataloader(
    batch_size: int,
    path: str,
    repeat: int = 1,
    dataloader_kwargs: dict = {},
):
    def collate_fn(batch):
        indices = [item[0] for item in batch]
        metadatas = [item[1] for item in batch]

        return {"indices": indices, "metadatas": metadatas}

    dataset = GenEvalDataset(path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=RepeatNoPadDistributedSampler(dataset=dataset, repeat=repeat) if dist.is_initialized() else None,
        collate_fn=collate_fn,
        **dataloader_kwargs,
    )


# GenEval inference manager
class GenEvalInferenceManager(InferenceManager):
    @InferenceManager.benchmark_register
    def run_t2i_benchmark(self, bench_name: str):
        from evals.config import T2I_BENCHMARK
        self.benchmark_name = bench_name

        self.benchmark_dir = os.path.join(self.output_dir, self.benchmark_name.lower())
        os.makedirs(self.benchmark_dir, exist_ok=True)

        loader = get_geneval_dataloader(
            path=T2I_BENCHMARK[bench_name],
            batch_size=8,
            repeat=4,
        )

        dist.barrier(device_ids=[self.local_rank])
        for idx, data in enumerate(loader):
            logger.info(f"\033[1m\033[31m{self.benchmark_name}\033[0m Progress: {idx + 1}/{len(loader)}")

            indices, metadatas = data['indices'], data['metadatas']
            for index, metadata in zip(indices, metadatas):
                os.makedirs(os.path.join(self.benchmark_dir, f"{index:0>5}"), exist_ok=True)
                os.makedirs(os.path.join(self.benchmark_dir, f"{index:0>5}", "samples"), exist_ok=True)
                with open(os.path.join(self.benchmark_dir, f"{index:0>5}", "metadata.jsonl"), "w") as fp:
                    json.dump(metadata, fp)

            prompts = [metadata["prompt"] for metadata in metadatas]
            images = self.run_t2i_inference(prompts)
            for i in range(len(images)):
                images[i].save(f"{self.benchmark_dir}/{indices[i]:0>5}/samples/rank_{self.rank:0>3}.png")

        dist.barrier(device_ids=[self.local_rank])
        logger.info(f"Finished \033[1m\033[31m{self.benchmark_name}\033[0m inference at rank {self.rank}.")

        if self.rank == 0:
            post_process(self.benchmark_dir)
        dist.barrier(device_ids=[self.local_rank])


def main(args):
    manager = GenEvalInferenceManager(
        model_name_or_path=args.model_name_or_path,
        output_dir=args.output_dir,
    )
    manager.run_t2i_benchmark(args.bench_name)
    manager.finish_and_cleanup()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--bench_name", type=str, default="GENEVAL")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
