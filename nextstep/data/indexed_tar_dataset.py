import copy
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generator

import torch
from torch.utils.data import IterableDataset

from nextstep.data.lru import LRUShards
from nextstep.data_zoos import DATA_META_PATH
from nextstep.datasets.utils import get_num_workers, get_worker_id, is_match, make_seed
from nextstep.utils.comm import dist_ctx
from nextstep.utils.misc import LargeInt


@dataclass
class RetSample:
    name: str
    data_type: int
    missing_indices: list[int]
    missing_urls_indices: list[tuple[str, int]]
    missing_samples: int
    index: int
    url_index: tuple[str, int]  # (url, sample_index) tuple representing where the sample is physically located
    sample: Any  # The actual returned sample
    worker_id: int
    num_workers: int


class DataStatus:
    def __init__(
        self,
        num_workers: int,
        status: dict[int, int] | None = None,
        last_worker_id: int | None = None,
    ):
        """
        Initialize a DataStatus object to track worker progress.

        Args:
            num_workers (int): Number of workers processing the dataset
            status (dict[int, int] | None, optional): Dictionary mapping Worker IDs to the
                number of samples they've processed. Defaults to None.
            last_worker_id (int | None, optional): The ID of the last worker that processed
                a sample. Defaults to None.

        The DataStatus object helps maintain dataset state across workers, ensuring samples
        are distributed evenly and allowing for resumption of training from checkpoints.
        """
        self.num_workers = max(num_workers, 1)
        if status is None:
            self.status: dict[int, int] = {}
            for i in range(self.num_workers):
                self.status[i] = 0
        else:
            self.status = status
        self.last_worker_id = 0 if last_worker_id is None else last_worker_id

    def update(self, ret_sample: RetSample):
        self.status[ret_sample.worker_id] += 1 + ret_sample.missing_samples
        self.last_worker_id = ret_sample.worker_id


class IndexedTarDataset(IterableDataset, ABC):
    """
    Abstract base class for datasets that read from indexed tar files.
    Subclasses must implement the required abstract methods.

    Variables prefixed with double underscore (__) are protected and should only be
    modified through the implemented methods, not directly from outside the class.
    """

    def __init__(self, data_info: dict, shuffle_size: int = LargeInt("50K"), lru_size: int = 32, seed: int = 42):
        """
        Initialize the dataset with data information.

        Args:
            data_info (dict): A dictionary containing dataset configuration.
                Must include:
                - data_type (str): The type of the dataset
                - name (str): The name of the dataset
                - Other configuration parameters specific to the dataset implementation
            shuffle_size (int, optional): The size of the buffer for shuffling samples.
                Defaults to LargeInt("50K").
            lru_size (int, optional): The size of the LRU cache.
                Defaults to 32.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.

        e.g.:
            data_info = {
                "data_type": "image_text_pair",
                "name": "google_collections/cat",
                "caption_keys": ["caption", "caption_all"],
                "caption_ratio": [1, 1],
                "filter": {
                    "height": 256,
                    "width": 256,
                    "aspect_ratio": 1.5,
                    "aes_score": 5.5,
                    "keywords": ["black and white"],
                    "keywords_drop_prob": [0.9],
                },
            }
        """
        super(IndexedTarDataset, self).__init__()

        self.data_info = data_info
        self.lru_size = lru_size
        self.seed = seed

        if "data_type" not in data_info:
            raise ValueError("data_info must contain a 'data_type' key")

        if "name" not in data_info:
            raise ValueError("data_info must contain a 'name' key")

        self.name = data_info["name"]
        self.data_type = data_info["data_type"]

        # get shard urls
        self.__urls = []
        self.__num_samples = []
        with open(DATA_META_PATH, "r") as f:
            data_meta = json.load(f)
        self.__matched_keys = [k for k in data_meta.keys() if is_match(self.name, k)]
        for k in self.__matched_keys:
            with open(data_meta[k]["tar_meta_path"], "r") as f:
                tar_meta = json.load(f)
            self.__urls.extend([os.path.join(data_meta[k]["dir"], rel_path) for rel_path in tar_meta.keys()])
            self.__num_samples.extend([v["num_samples"] for v in tar_meta.values()])

        self.__lru_shards = LRUShards(lru_size=lru_size)

        self.__missing_indices = []
        self.__missing_urls_indices = []
        self.__missing_samples = 0

        # get distributed and worker information
        self.__rank = dist_ctx.rank
        self.__world_size = dist_ctx.world_size

        # These should be set in __iter__, otherwise workers will inherit information from the main process
        self.__worker_id = None
        self.__num_workers = None

        self.__cur_urls = copy.deepcopy(self.__urls)
        self.__cur_num_samples = copy.deepcopy(self.__num_samples)

        self.base_shuffle_size = shuffle_size

        match self.__split_mode():
            case "sample":
                self.__shuffle_size = min(self.base_shuffle_size, sum(self.__cur_num_samples))
                try:
                    self.__max_sub_ep = sum(self.__cur_num_samples) // self.__shuffle_size
                except Exception as e:
                    raise RuntimeError(
                        f"Error {e}: {self.name}, sum(self.__cur_num_samples)={sum(self.__cur_num_samples)}, self.__shuffle_size={self.__shuffle_size}"
                    )
            case "url":
                self.__shuffle_size = None
                self.__max_sub_ep = None
            case _:
                raise ValueError(f"Invalid split mode: {self.__split_mode()}")

        # data status, for recovery
        self.__data_status = None

    def __split_mode(self) -> str:
        # When reading data from S3, splitting by sample can cause frequent file locks,
        # leading to data loading bottlenecks. If we have many URLs available (at least 10x
        # the world size), we split by URL first to reduce contention and improve throughput.

        # # FIXME: force to use sample split mode
        # return "sample"

        if len(self.__urls) >= self.__world_size * 10:
            return "url"
        return "sample"

    def __set_ep(self, ep: int, sub_ep: int, base_pointer: int = 0):
        # BUG: When dataset size is very small compared to (world_size * num_workers),
        # there might not be enough data to distribute across all workers

        self.__ep = ep
        self.__sub_ep = sub_ep

        # Shuffle URLs deterministically based on epoch
        g = torch.Generator()
        g.manual_seed(make_seed(self.seed, self.__ep))
        indices = torch.randperm(len(self.__urls), generator=g).tolist()
        self.__cur_urls = [self.__urls[i] for i in indices]
        self.__cur_num_samples = [self.__num_samples[i] for i in indices]

        if self.__split_mode() == "url":
            self.__cur_urls = self.__cur_urls[self.__rank :: self.__world_size]
            self.__cur_num_samples = self.__cur_num_samples[self.__rank :: self.__world_size]

            self.__shuffle_size = min(self.base_shuffle_size, sum(self.__cur_num_samples))
            self.__max_sub_ep = sum(self.__cur_num_samples) // self.__shuffle_size

        # Shuffle indices deterministically based on epoch and sub-epoch
        g = torch.Generator()
        g.manual_seed(make_seed(self.seed, self.__ep, self.__sub_ep))
        perm_n = self.__shuffle_size
        # For the last sub-epoch, include both shuffle_size and any remaining samples
        if (self.__sub_ep + 2) * self.__shuffle_size > sum(self.__cur_num_samples):
            perm_n = sum(self.__cur_num_samples) - self.__sub_ep * self.__shuffle_size
        self.__cur_indices = (torch.randperm(perm_n, generator=g) + self.__sub_ep * self.__shuffle_size).tolist()
        if self.__split_mode() == "sample":
            self.__cur_indices = self.__cur_indices[self.__rank : self.__shuffle_size : self.__world_size]

        # pointer to the current index
        self.__base_pointer = base_pointer

        self.indices = None

    def __next_sub_ep(self, pointer: int):
        if self.__sub_ep + 1 == self.__max_sub_ep:
            self.__set_ep(ep=self.__ep + 1, sub_ep=0, base_pointer=pointer)
        else:
            self.__set_ep(ep=self.__ep, sub_ep=self.__sub_ep + 1, base_pointer=pointer)

    @property
    def cur_urls(self):
        return self.__cur_urls

    @property
    def cur_num_samples(self):
        return self.__cur_num_samples

    @property
    def cur_indices(self):
        return self.__cur_indices

    @property
    def ep(self):
        return self.__ep

    @property
    def sub_ep(self):
        return self.__sub_ep

    @property
    def max_sub_ep(self):
        return self.__max_sub_ep

    @property
    def cur_pointer(self):
        return self.__base_pointer + self.__worker_id

    @property
    def cur_index(self):
        return self.cur_indices[self.cur_pointer]

    @property
    def matched_keys(self):
        return self.__matched_keys

    @property
    def shuffle_size(self):
        return self.__shuffle_size

    @property
    def lru_shards(self):
        return self.__lru_shards

    @property
    def missing_indices(self):
        return self.__missing_indices

    @property
    def missing_urls_indices(self):
        return self.__missing_urls_indices

    @property
    def missing_samples(self):
        return self.__missing_samples

    @property
    def rank(self):
        return self.__rank

    @property
    def world_size(self):
        return self.__world_size

    @property
    def worker_id(self):
        return self.__worker_id

    @property
    def num_workers(self):
        return self.__num_workers

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError("Subclasses must implement this method")

    def __len__(self):
        """Return the number of samples in the dataset."""
        return LargeInt(sum(self.__num_samples))

    def __iter__(self):
        """Return an iterator over the dataset."""
        self.__worker_id = get_worker_id()
        self.__num_workers = get_num_workers()

        self.__set_ep(0, 0, 0)
        if self.__data_status is not None:
            assert self.__data_status.num_workers == self.__num_workers, "num_workers mismatch"
            self.__worker_id = (self.__data_status.last_worker_id + self.__worker_id + 1) % self.__num_workers
            skip_samples = self.__data_status.status[self.__worker_id]
            for _ in range(skip_samples):
                self.next_index()

        self.__data_generator = self.data_generator()
        return self.__data_generator

    def __next__(self):
        """Return the next sample from the dataset."""
        return next(self.__data_generator)

    def set_data_status(self, data_status: DataStatus):
        """
        Set the data status for resuming dataset iteration.

        This method sets the data status which manages the recovery state of the dataset.
        When resuming, the __iter__ method will use this status to skip the appropriate
        number of samples based on what was already processed.

        Each time a sample is successfully returned, the DataStatus should be updated
        using its update method with the RetSample, which tracks missing samples and
        worker state.

        Args:
            data_status (DataStatus): The data status object containing recovery information
        """
        self.__data_status = data_status

    def index2indices(self, index: int) -> tuple[str, int]:
        """Return the indices (url_index, sample_index) of the sample at the given index."""
        _index = copy.deepcopy(index)
        for url, num_samples in zip(self.__cur_urls, self.__cur_num_samples):
            if _index < num_samples:
                return url, _index
            _index -= num_samples
        raise IndexError(f"Index {index} is out of range for the dataset")

    def skip_sample(self, index: int):
        """Increment missing samples counter and move to the next sample.

        This method should be called when a sample is skipped due to errors or filtering.
        Proper usage of skip_sample() and success_sample() is critical for accurate
        missing_samples statistics.
        """
        self.__missing_indices.append(index)
        self.__missing_urls_indices.append(self.index2indices(index))
        self.__missing_samples += 1
        return

    def success_sample(self):
        """Reset missing samples counter and move to the next sample.

        This method should be called before a sample is successfully processed.
        Proper usage of skip_sample() and success_sample() is critical for accurate
        missing_samples statistics.
        """
        missing_indices = copy.deepcopy(self.__missing_indices)
        missing_urls_indices = copy.deepcopy(self.__missing_urls_indices)
        missing_samples = copy.deepcopy(self.__missing_samples)
        self.__missing_indices = []
        self.__missing_urls_indices = []
        self.__missing_samples = 0
        return missing_indices, missing_urls_indices, missing_samples

    def index_generator(self):
        while True:
            index = copy.deepcopy(self.cur_index)
            yield index
            self.__base_pointer += self.__num_workers
            if self.cur_pointer >= len(self.cur_indices):
                self.__next_sub_ep(self.__base_pointer - len(self.cur_indices))

    def next_index(self):
        """Return the next index and move the pointer to the next index."""
        if self.indices is None:
            self.indices = self.index_generator()
        return next(self.indices)

    @abstractmethod
    def data_generator(self) -> Generator[RetSample, None, None]:
        """Return a generator of data samples."""
        raise NotImplementedError("Subclasses must implement this method")
