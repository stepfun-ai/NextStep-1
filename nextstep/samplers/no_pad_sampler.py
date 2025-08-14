import warnings

import torch.distributed as dist
from torch.utils.data import Dataset


class NoPadDistributedSampler:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.num_replicas = dist.get_world_size()
        self.rank = dist.get_rank()

        self.base_samples = len(self.dataset) // self.num_replicas
        self.remainder = len(self.dataset) % self.num_replicas
        self.num_samples = self.base_samples + (1 if self.rank < self.remainder else 0)

    def __iter__(self):
        start_idx = self.rank * self.base_samples + min(self.rank, self.remainder)
        end_idx = start_idx + self.num_samples
        indices = list(range(start_idx, end_idx))
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples


class RepeatNoPadDistributedSampler(NoPadDistributedSampler):
    def __init__(self, dataset: Dataset, repeat: int = 1):
        super().__init__(dataset)
        self.repeat = repeat

        if self.repeat > self.num_replicas:
            warnings.warn(
                f"repeat {self.repeat} is greater than the number of replicas {self.num_replicas}, "
                f"this will cause some samples to be repeated by same rank."
            )

        self.num_samples = [self.base_samples + (1 if i < self.remainder else 0) for i in range(self.num_replicas)]
        self.indices = []

        for i in range(self.num_replicas):
            start_idx = i * self.base_samples + min(i, self.remainder)
            end_idx = start_idx + self.num_samples[i]
            self.indices.append(list(range(start_idx, end_idx)))

    def __iter__(self):
        indices = []
        for i in range(self.repeat):
            indices.extend(self.indices[(self.rank + i) % self.num_replicas])
        return iter(indices)

    def __len__(self) -> int:
        num_samples = 0
        for i in range(self.repeat):
            num_samples += self.num_samples[i % self.num_replicas]
        return num_samples
