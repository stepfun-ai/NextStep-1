import math

import torch
import torch.distributed as dist
from torch.utils.data.dataset import Dataset


class SkipBatchDistributedSampler:
    def __init__(
        self,
        dataset: Dataset,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.num_replicas = dist.get_world_size()
        self.rank = dist.get_rank()
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil((len(self.dataset) - self.num_replicas) / self.num_replicas)  # type: ignore[arg-type]
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.skip_samples = 0

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        # Skip samples if needed
        if self.skip_samples > 0:
            indices = indices[self.skip_samples :]
            self.skip_samples = 0  # Reset skip count after using it

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples - self.skip_samples

    def set_epoch(self, epoch: int, skip_samples: int = 0) -> None:
        r"""
        Set the epoch for this sampler and optionally skip samples.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
            skip_samples (int, optional): Number of samples to skip at the beginning.
                                        Defaults to 0.
        """
        self.epoch = epoch
        self.skip_samples = skip_samples
