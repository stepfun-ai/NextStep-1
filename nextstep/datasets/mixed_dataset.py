import copy
import gc
import json
import os
import random
import re
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures._base import TimeoutError
from dataclasses import dataclass

import numpy as np
import torch.multiprocessing as mp
import torch.utils.data
from tabulate import tabulate
from torch.utils.data import DataLoader, IterableDataset
from transformers import PreTrainedTokenizer

from nextstep.data.indexed_tar_dataset import DataStatus, IndexedTarDataset, RetSample
from nextstep.datasets.data_logger import data_logger as logger
from nextstep.datasets.utils import get_num_workers, get_worker_id, make_seed
from nextstep.lazy_config.registry import locate
from nextstep.models.nextstep.tokenization_nextstep import (
    DEFAULT_BOI_TOKEN,
    DEFAULT_EOI_TOKEN,
    DEFAULT_IMAGE_PLACEHOLDER_TOKEN,
    IGNORE_INDEX,
    DEFAULT_EOL_TOKEN,
)
from nextstep.models.nextstep.aspect_ratio import ANY_ASPECT_RATIO, ar2str
from nextstep.utils.comm import dist_ctx
from nextstep.utils.mem_utils import PeriodicMemoryMonitor
from nextstep.utils.misc import LargeInt
from nextstep.utils.timer import TimerManager
import gc
import torch
import numpy as np

def _get_hw_aspect_ratio(pixel_value: torch.Tensor, down_factor: int):
    if pixel_value.ndim == 4:
        _, _, h, w = pixel_value.shape
    else:
        _, h, w = pixel_value.shape
    h, w = h // down_factor, w // down_factor
    hw_aspect_ratio = ar2str(h, w)
    return hw_aspect_ratio


@dataclass
class Batch:
    batch_data: dict
    dataset_indices: list[int]
    pv2data: list[str]
    image_samples_list: list[RetSample]
    status: dict[str, dict]  # dataset_name: {worker_id: {hit: int, miss: int}} of cur batch
    state_dict: dict  # worker_id: state_dict of cur batch
    worker_id: int
    num_workers: int

# Load and save state
class MixingStatus:
    def __init__(self, num_workers: int):
        self.num_workers = max(num_workers, 1)
        # dataset_name: {worker_id: data_status}
        # data_status manages the data state of IndexedTarDataset contained in MixedDataset
        # In the __iter__ method, this is used to create DataStatus to help IndexedTarDataset restore data state
        self.data_status: dict[str, dict[int, int]] = {}
        # worker_id: state_dict
        # Manages the state_dict of each worker's MixedDataset, including rng state, buffer state, etc.
        # This is used to restore the dataset state when resuming training
        self.dataset_state_dict: dict[int, dict] = {}
        self.last_worker_id = 0

    def update(self, batch: Batch):
        for name, status in batch.status.items():
            if name not in self.data_status:
                self.data_status[name] = {id: 0 for id in range(self.num_workers)}
            self.data_status[name][batch.worker_id] += status["hit"] + status["miss"]

        self.dataset_state_dict[batch.worker_id] = batch.state_dict
        self.last_worker_id = batch.worker_id

    def state_dict(self):
        return copy.deepcopy(
            {
                "num_workers": self.num_workers,
                "data_status": self.data_status,
                "dataset_state_dict": self.dataset_state_dict,
                "last_worker_id": self.last_worker_id,
            }
        )

    def load_state_dict(self, state_dict: dict):
        _state_dict = copy.deepcopy(state_dict)
        self.num_workers = _state_dict["num_workers"]
        self.data_status = _state_dict["data_status"]
        self.dataset_state_dict = _state_dict["dataset_state_dict"]
        self.last_worker_id = _state_dict["last_worker_id"]


# fmt: off
class BaseBuffer(ABC):

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def enqueue(self):
        pass

    @abstractmethod
    def dequeue(self):
        pass

    @abstractmethod
    def reinit_buffer(self):
        pass


class NLPBuffer(BaseBuffer):

    def __init__(self, name: str):
        super().__init__(name)
        self.__input_ids: list[int] = []

    def __len__(self):
        return len(self.__input_ids)

    @property
    def input_ids(self):
        return self.__input_ids

    def enqueue(self, input_ids: list[int]):
        self.__input_ids.extend(input_ids)

    def dequeue(self, length: int):
        if len(self.__input_ids) < length:
            raise ValueError(f"{self.name} --> The length of buffer ({len(self.__input_ids)}) is less than the length ({length}).")

        input_ids = copy.deepcopy(self.__input_ids[:length])
        attention_mask = [1] * length
        labels = copy.deepcopy(self.__input_ids[:length])

        self.__input_ids = self.__input_ids[length:]

        return input_ids, attention_mask, labels

    def reinit_buffer(self):
        self.__input_ids = []

    def state_dict(self):
        return copy.deepcopy({"name": self.name, "input_ids": self.__input_ids})

    def load_state_dict(self, state_dict: dict):
        _state_dict = copy.deepcopy(state_dict)
        self.name = _state_dict["name"]
        self.__input_ids = _state_dict["input_ids"]


class ImageBuffer(BaseBuffer):

    def __init__(
        self,
        name: str,
        image_placeholder_id: int,
        boi: int,
        eoi: int,
        pad_token_id: int,
        down_factor: int,
        hw_aspect_ratios_str: list[str],
        image_seq_len: dict[str, int],
        sample_length: int,
    ):
        super().__init__(name)
        self.image_placeholder_id = image_placeholder_id
        self.boi = boi
        self.eoi = eoi
        self.pad_token_id = pad_token_id
        self.down_factor = down_factor
        self.hw_aspect_ratios_str = hw_aspect_ratios_str
        self.image_seq_len = image_seq_len

        self.sample_length = sample_length

        self.reinit_buffer()

    def __len__(self):
        return max((len(v) for v in self.__input_ids.values()))

    def get_max_len_key(self):
        max_len = 0
        max_key = None
        for key, value in self.__input_ids.items():
            if len(value) > max_len:
                max_len = len(value)
                max_key = key
        return max_key

    @property
    def input_ids(self):
        return self.__input_ids

    @property
    def labels(self):
        return self.__labels

    @property
    def pixel_values(self):
        return self.__pixel_values

    @property
    def image_filtered_idx(self):
        return self.__image_filtered_idx

    @property
    def indices(self):
        return self.__indices

    @property
    def samples(self):
        return self.__samples

    def indices_status(self):
        """
        Print the status of indices using tabulate.
        """
        from tabulate import tabulate

        table_data = []
        headers = ["Aspect Ratio", "Indices"]

        for aspect_ratio, indices in self.__indices.items():
            table_data.append([
                aspect_ratio,
                indices,
            ])

        table = tabulate(table_data, headers=headers, tablefmt="pipe")
        return table

    def enqueue_sanity_check(self, sample: RetSample, input_ids: list[int], pixel_values: list[torch.Tensor], image_filtered_idx: list[int]):
        url, sample_index = sample.url_index

        num_boi = (torch.tensor(input_ids) == self.boi).sum().item()
        num_eoi = (torch.tensor(input_ids) == self.eoi).sum().item()
        num_placeholder_id = (torch.tensor(input_ids) == self.image_placeholder_id).sum().item()

        is_valid = True

        if num_boi != len(pixel_values):
            is_valid = False
            logger.error(
                f"{self.name} ({url}, {sample_index}) --> number of boi ({num_boi}) is not equal to the number of images ({len(pixel_values)})\n"
                # f"sample: {sample.sample}\n"
                f"input_ids: {input_ids}"
            )

        if num_boi != len(image_filtered_idx):
            is_valid = False
            logger.error(
                f"{self.name} ({url}, {sample_index}) --> number of boi ({num_boi}) is not equal to the number of image_filtered_idx ({len(image_filtered_idx)})\n"
                # f"sample: {sample.sample}\n"
                f"input_ids: {input_ids}"
            )

        if num_eoi != len(pixel_values):
            is_valid = False
            logger.error(
                f"{self.name} ({url}, {sample_index}) --> number of eoi ({num_eoi}) is not equal to the number of images ({len(pixel_values)})\n"
                # f"sample: {sample.sample}\n"
                f"input_ids: {input_ids}"
            )

        if num_eoi != len(image_filtered_idx):
            is_valid = False
            logger.error(
                f"{self.name} ({url}, {sample_index}) --> number of eoi ({num_eoi}) is not equal to the number of image_filtered_idx ({len(image_filtered_idx)})\n"
                # f"sample: {sample.sample}\n"
                f"input_ids: {input_ids}"
            )

        placeholder_num = 0
        for i in range(len(pixel_values)):
            placeholder_num += self.image_seq_len[_get_hw_aspect_ratio(pixel_values[i], self.down_factor)]

        if num_placeholder_id != placeholder_num:
            is_valid = False
            logger.error(
                f"{self.name} ({url}, {sample_index}) --> number of placeholders ({num_placeholder_id}) is not equal to the number of images tokens ({placeholder_num})\n"
                # f"sample: {sample.sample}\n"
                f"input_ids: {input_ids}"
            )

        if len(input_ids) > self.sample_length:
            is_valid = False
            logger.error(
                f"{self.name} ({url}, {sample_index}) --> The length of input_ids ({len(input_ids)}) is greater than the sample_length ({self.sample_length})\n"
                # f"sample: {sample.sample}\n"
                # f"input_ids: {input_ids}"
            )

        return is_valid

    def enqueue(self, sample: RetSample, input_ids: list[int], labels: list[int], pixel_values: list[torch.Tensor], image_filtered_idx: list[int]):
        if not self.enqueue_sanity_check(sample, input_ids, pixel_values, image_filtered_idx):
            return

        hw_aspect_ratios = [_get_hw_aspect_ratio(pixel_value, self.down_factor) for pixel_value in pixel_values]
        if len(pixel_values) == 1 or len(set(hw_aspect_ratios)) == 1:
            hw_aspect_ratio = _get_hw_aspect_ratio(pixel_values[0], self.down_factor)
        else:
            hw_aspect_ratio = ar2str(*ANY_ASPECT_RATIO)

        self.__input_ids[hw_aspect_ratio].extend(input_ids)
        self.__labels[hw_aspect_ratio].extend(labels)
        self.__pixel_values[hw_aspect_ratio].extend(pixel_values)
        self.__image_filtered_idx[hw_aspect_ratio].extend(image_filtered_idx)

        self.__indices[hw_aspect_ratio].append(len(self.__input_ids[hw_aspect_ratio]))
        self.__samples[hw_aspect_ratio].append(sample)

    def find_index(self, length: int, ar_key: str):
        """return the index of the sample that is less than or equal to the length"""
        for i, idx in enumerate(self.__indices[ar_key]):
            if idx > length:
                return i - 1
        return i

    def dequeue(self, length: int | None = None, ar_key: str | None = None):
        if length is None:
            length = self.sample_length

        if ar_key is None:
            ar_key = self.get_max_len_key()

        if len(self.__input_ids[ar_key]) < length:
            raise ValueError(f"{self.name} --> The length of input_ids ({len(self.__input_ids[ar_key])}) is less than the length ({length}).")

        index = self.find_index(length, ar_key)
        pad_len = length - self.__indices[ar_key][index]
        length = self.__indices[ar_key][index]
        self.__indices[ar_key] = self.__indices[ar_key][index:]
        for i in range(len(self.__indices[ar_key])):
            self.__indices[ar_key][i] -= length

        samples = self.__samples[ar_key][:index]
        self.__samples[ar_key] = self.__samples[ar_key][index:]

        input_ids = self.__input_ids[ar_key][:length] + [self.pad_token_id] * pad_len
        attention_mask = [1] * length + [1] * pad_len
        labels = self.__labels[ar_key][:length] + [IGNORE_INDEX] * pad_len

        num_boi = (torch.tensor(input_ids) == self.boi).sum().item()
        num_eoi = (torch.tensor(input_ids) == self.eoi).sum().item()
        # Ensure the number of begin and end image tokens match
        if num_boi != num_eoi:
            raise ValueError(f"Number of BOI tokens ({num_boi}) does not match number of EOI tokens ({num_eoi}) in dequeued sequence.")
        pv_num = num_boi # Use the count of BOI (or EOI) tokens as the number of images

        pixel_values = self.__pixel_values[ar_key][:pv_num]
        image_filtered_idx = self.__image_filtered_idx[ar_key][:pv_num]

        self.__input_ids[ar_key] = self.__input_ids[ar_key][length:]
        self.__labels[ar_key] = self.__labels[ar_key][length:]
        self.__pixel_values[ar_key] = self.__pixel_values[ar_key][pv_num:]
        self.__image_filtered_idx[ar_key] = self.__image_filtered_idx[ar_key][pv_num:]
        return input_ids, attention_mask, labels, pixel_values, samples, image_filtered_idx, pad_len

    def reinit_buffer(self):
        self.__input_ids: dict[str, list[int]] = {_str: [] for _str in self.hw_aspect_ratios_str}
        self.__labels: dict[str, list[int]] = {_str: [] for _str in self.hw_aspect_ratios_str}
        self.__pixel_values: dict[str, list[torch.Tensor]] = {_str: [] for _str in self.hw_aspect_ratios_str}
        self.__image_filtered_idx: dict[str, list[int]] = {_str: [] for _str in self.hw_aspect_ratios_str}

        # [0, len(sample1), len(sample1) + sample2), ...
        self.__indices: dict[str, list[int]] = {_str: [0] for _str in self.hw_aspect_ratios_str}
        self.__samples: dict[str, list[RetSample]] = {_str: [] for _str in self.hw_aspect_ratios_str}

    def state_dict(self):
        """
        Save state dictionary. Use shallow copy for pixel_values to avoid memory leaks from deep copying large tensors.
        For samples, only save lightweight metadata (url_index, etc.), do not save large image data in sample.sample.
        """
        # Use shallow copy: only copy dictionary structure, do not deep copy tensor data
        pixel_values_shallow = {}
        for ar_key, pv_list in self.__pixel_values.items():
            pixel_values_shallow[ar_key] = list(pv_list)  # Shallow copy list, tensors are just references
        
        return {
            "name": self.name,
            "image_placeholder_id": self.image_placeholder_id,
            "boi": self.boi,
            "eoi": self.eoi,
            "pad_token_id": self.pad_token_id,
            "down_factor": self.down_factor,
            "hw_aspect_ratios_str": self.hw_aspect_ratios_str,
            "image_seq_len": self.image_seq_len,
            "sample_length": self.sample_length,
            "input_ids": copy.deepcopy(self.__input_ids),  # Integer lists, deep copy overhead is small
            "labels": copy.deepcopy(self.__labels),  # Integer lists, deep copy overhead is small
            "image_filtered_idx": copy.deepcopy(self.__image_filtered_idx),  # Integer lists, deep copy overhead is small
            "indices": copy.deepcopy(self.__indices),  # Integer lists, deep copy overhead is small
            "pixel_values": pixel_values_shallow,  # Shallow copy, only save references
        }

    def load_state_dict(self, state_dict: dict):
        """
        Load state dictionary. Use shallow copy to restore pixel_values and samples, avoid deep copying large tensors.
        """
        self.name = state_dict["name"]
        self.image_placeholder_id = state_dict["image_placeholder_id"]
        self.boi = state_dict["boi"]
        self.eoi = state_dict["eoi"]
        self.pad_token_id = state_dict["pad_token_id"]
        self.down_factor = state_dict["down_factor"]
        self.hw_aspect_ratios_str = state_dict["hw_aspect_ratios_str"]
        self.image_seq_len = state_dict["image_seq_len"]
        self.sample_length = state_dict["sample_length"]
        self.__input_ids = copy.deepcopy(state_dict["input_ids"])
        self.__labels = copy.deepcopy(state_dict["labels"])
        self.__image_filtered_idx = copy.deepcopy(state_dict["image_filtered_idx"])
        self.__indices = copy.deepcopy(state_dict["indices"])
        
        # Use shallow copy to restore pixel_values
        self.__pixel_values = {
            ar_key: list(pv_list) if isinstance(pv_list, list) else pv_list
            for ar_key, pv_list in state_dict["pixel_values"].items()
        }
        

class BatchBucket:
    def __init__(self, batch_size: int, down_factor: int, hw_aspect_ratios_str: list[str]):
        self.batch_size = batch_size
        self.down_factor = down_factor
        self.hw_aspect_ratios_str = hw_aspect_ratios_str

        self.dataset_indices: dict[str, list[int]] = {_str: [] for _str in hw_aspect_ratios_str}
        self.batch_data: dict[str, dict[str, list]] = {_str: dict(pixel_values=[], input_ids=[], attention_mask=[], labels=[], image_filtered_idx=[], waste_token_num=[]) for _str in hw_aspect_ratios_str}
        self.image_samples_list: dict[str, list[RetSample]] = {_str: [] for _str in hw_aspect_ratios_str}

        # Add safety threshold to prevent data accumulation
        self.max_accumulated_samples = batch_size * 2
        self.rng = None

    def set_rng(self, seed: int, rank: int, worker_id: int):
        self.rng = random.Random(make_seed("BatchBucket", seed, rank, worker_id))

    def enqueue(
        self,
        dataset_idx: int,
        pixel_values: list[torch.Tensor],
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        labels: torch.LongTensor,
        samples: list[RetSample],
        image_filtered_idx: list[int],
        waste_token_num: int,
    ):
        # calculate hw_aspect_ratios
        hw_aspect_ratios = [_get_hw_aspect_ratio(pixel_value, self.down_factor) for pixel_value in pixel_values]
        if len(pixel_values) > 1 and len(set(hw_aspect_ratios)) > 1:
            hw_aspect_ratio = ar2str(*ANY_ASPECT_RATIO)
        elif len(set(hw_aspect_ratios)) == 1:
            hw_aspect_ratio = _get_hw_aspect_ratio(pixel_values[0], self.down_factor)
        else:  # NLP
            indices = list(range(len(self.hw_aspect_ratios_str)))
            available_indices = []
            for i in indices:
                cur_batch_data = self.batch_data[self.hw_aspect_ratios_str[i]]
                if len(cur_batch_data["input_ids"]) == self.batch_size - 1 and len(cur_batch_data["pixel_values"]) == 0:
                    continue
                available_indices.append(i)
            if len(available_indices) == 0:
                return None, None, None, None
            index = self.rng.choice(available_indices)
            hw_aspect_ratio = self.hw_aspect_ratios_str[index]

        self.dataset_indices[hw_aspect_ratio].append(dataset_idx)
        self.batch_data[hw_aspect_ratio]["pixel_values"].extend(pixel_values)
        self.batch_data[hw_aspect_ratio]["input_ids"].append(input_ids)
        self.batch_data[hw_aspect_ratio]["attention_mask"].append(attention_mask)
        self.batch_data[hw_aspect_ratio]["labels"].append(labels)
        self.batch_data[hw_aspect_ratio]["image_filtered_idx"].extend(image_filtered_idx)
        self.batch_data[hw_aspect_ratio]["waste_token_num"].append(waste_token_num)
        self.image_samples_list[hw_aspect_ratio].extend(samples)

        # Directly check if current aspect ratio is full
        if len(self.dataset_indices[hw_aspect_ratio]) == self.batch_size:
            dataset_indices = copy.deepcopy(self.dataset_indices[hw_aspect_ratio])
            batch_data = copy.deepcopy(self.batch_data[hw_aspect_ratio])
            image_samples_list = copy.deepcopy(self.image_samples_list[hw_aspect_ratio])

            self.dataset_indices[hw_aspect_ratio] = []
            self.batch_data[hw_aspect_ratio] = dict(pixel_values=[], input_ids=[], attention_mask=[], labels=[], image_filtered_idx=[], waste_token_num=[])
            self.image_samples_list[hw_aspect_ratio] = []

            return dataset_indices, batch_data, image_samples_list, hw_aspect_ratio

        return None, None, None, None

    def state_dict(self):
        return copy.deepcopy(
            {
                "batch_size": self.batch_size,
                "down_factor": self.down_factor,
                "hw_aspect_ratios_str": self.hw_aspect_ratios_str,
                "dataset_indices": self.dataset_indices,
                "batch_data": self.batch_data,
                "image_samples_list": self.image_samples_list,
                "rng_state": self.rng.getstate(),
            }
        )

    def load_state_dict(self, state_dict: dict):
        _state_dict = copy.deepcopy(state_dict)
        self.batch_size = _state_dict["batch_size"]
        self.down_factor = _state_dict["down_factor"]
        self.hw_aspect_ratios_str = _state_dict["hw_aspect_ratios_str"]
        self.dataset_indices = _state_dict["dataset_indices"]
        self.batch_data = _state_dict["batch_data"]
        self.image_samples_list = _state_dict["image_samples_list"]
        self.rng.setstate(_state_dict["rng_state"])


class MixedDataset(IterableDataset):
    datasets: list[IndexedTarDataset]

    def __init__(
        self,
        data_info_list: list[dict],
        batch_size: int,
        tokenizer: PreTrainedTokenizer,
        down_factor: int,
        hw_aspect_ratios_ids: dict[str, list[int]],
        image_seq_len: dict[str, int] = {},
        max_len: int = 1280,
        drop_text_prob: float = 0.1,
        shuffle_size: int = LargeInt("50K"),
        lru_size: int = 32,
        seed: int = 42,
        dataset_kwargs: dict = {},
        eol_token = False,
        *args,
        **kwargs,
    ):
        self.data_info_list = data_info_list
        self.status = {}
        self._mixing_status = None

        self.dataset_names = [data_info["name"] for data_info in data_info_list]
        self.dataset_types = [data_info["data_type"] for data_info in data_info_list]
        assert all(
            dataset_type in set(["nlp", "image_text_pair", "interleave", "image_editing"]) for dataset_type in self.dataset_types
        ), f"Unknown dataset type: {self.dataset_types}"
        self.sampling_ratios = [LargeInt(data_info["samples"]) for data_info in data_info_list]
        self.sampling_ratios = [sampling_ratio / sum(self.sampling_ratios) for sampling_ratio in self.sampling_ratios]

        self.datasets = []
        for i, data_info in enumerate(data_info_list):
            dataset = locate(data_info["cls"])(
                data_info=data_info,
                shuffle_size=shuffle_size,
                lru_size=max(1, int(lru_size * self.sampling_ratios[i])),
                seed=seed,
                **dataset_kwargs,
            )
            self.datasets.append(dataset)
            self.status[data_info["name"]] = {"hit": 0, "miss": 0, "total": len(dataset)}
        self.total_samples = sum([len(dataset) for dataset in self.datasets])

        self.batch_size = batch_size
        self.tokenizer = copy.deepcopy(tokenizer)
        self.tokenizer.add_eos_token = False
        self.image_placeholder_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_PLACEHOLDER_TOKEN)
        self.boi = self.tokenizer.convert_tokens_to_ids(DEFAULT_BOI_TOKEN)
        self.eoi = self.tokenizer.convert_tokens_to_ids(DEFAULT_EOI_TOKEN)
        self.eol = self.tokenizer.convert_tokens_to_ids(DEFAULT_EOL_TOKEN) if eol_token else None
        self.down_factor = down_factor
        self.hw_aspect_ratios_ids = hw_aspect_ratios_ids
        self.image_seq_len = image_seq_len
        self.max_len = max_len
        self.drop_text_prob = drop_text_prob

        self.nlp_buffer: dict[int, NLPBuffer] = {
            i: NLPBuffer(name=self.dataset_names[i])
            for i, dataset_type in enumerate(self.dataset_types)
            if dataset_type == "nlp"
        }
        self.image_buffer: dict[int, ImageBuffer] = {
            i: ImageBuffer(
                name=self.dataset_names[i],
                image_placeholder_id=self.image_placeholder_id,
                boi=self.boi,
                eoi=self.eoi,
                pad_token_id=self.tokenizer.pad_token_id,
                down_factor=self.down_factor,
                hw_aspect_ratios_str=list(hw_aspect_ratios_ids.keys()) + [ar2str(*ANY_ASPECT_RATIO)],
                image_seq_len=self.image_seq_len,
                sample_length=self.max_len,
            )
            for i, dataset_type in enumerate(self.dataset_types)
            if dataset_type != "nlp"
        }
        self.batch_bucket = BatchBucket(
            batch_size=batch_size,
            down_factor=down_factor,
            hw_aspect_ratios_str=list(hw_aspect_ratios_ids.keys()) + [ar2str(*ANY_ASPECT_RATIO)],
        )

        self.seed = seed
        self.verbose = False

        self.timer_manager = TimerManager()
        self.threading_executor = None

    def __repr__(self):
        headers = ["Dataset ID", "Dataset Name", "Samples", "Actual Ratio", "Sampling Ratio"]
        data = []
        for i, (name, sampling_ratio) in enumerate(zip(self.dataset_names, self.sampling_ratios)):
            dataset_size = len(self.datasets[i])
            actual_ratio = (dataset_size / self.total_samples) * 100 if self.total_samples > 0 else 0
            data.append([
                i,
                name,
                str(LargeInt(dataset_size)),
                f"{actual_ratio:.2f}%",
                f"{sampling_ratio*100:.2f}%",
            ])
        rep = f"{self.__class__.__name__}: {len(self)} samples, {len(self.datasets)} datasets\n"
        rep += tabulate(data, headers=headers, tablefmt="pipe")
        return rep

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        self.__worker_id = get_worker_id()
        self.__num_workers = get_num_workers()

        self.rng = random.Random(make_seed(self.seed, dist_ctx.rank, self.__worker_id))
        self.batch_bucket.set_rng(self.seed, dist_ctx.rank, self.__worker_id)

        if self._mixing_status is not None:
            assert self._mixing_status.num_workers == self.__num_workers, "num_workers mismatch"
            self.__worker_id = (self._mixing_status.last_worker_id + self.__worker_id + 1) % self.__num_workers

            self.rng = random.Random(make_seed(self.seed, dist_ctx.rank, self.__worker_id))  # reset rng if worker_id is remapped
            self.batch_bucket.set_rng(self.seed, dist_ctx.rank, self.__worker_id)

            if self.__worker_id in self._mixing_status.dataset_state_dict:
                self.load_state_dict(self._mixing_status.dataset_state_dict[self.__worker_id])
            for dataset in self.datasets:
                dataset.set_data_status(
                    DataStatus(
                        num_workers=self.__num_workers,
                        status=self._mixing_status.data_status[dataset.name] if dataset.name in self._mixing_status.data_status else None,
                        last_worker_id=self._mixing_status.last_worker_id,
                    )
                )

        self.datasets = [iter(dataset) for dataset in self.datasets]
        self.__data_generator = self.data_generator()
        return self.__data_generator

    def _get_threading_executor(self):
        if self.threading_executor is None:
            self.threading_executor = ThreadPoolExecutor(max_workers=1)
        return self.threading_executor

    def __next__(self):
        # return next(self.__data_generator)
        executor = self._get_threading_executor()
        data_future = executor.submit(next, self.__data_generator)
        self.timer_manager.beat("next", "start")
        timout_triggered = False
        while True:
            try:
                result = data_future.result(timeout=10.0)
                self.timer_manager.beat("next", "end")
                if timout_triggered:
                    dataset_names = [self.dataset_names[dataset_idx] for dataset_idx in result.dataset_indices]
                    logger.warning(f"{dataset_names} Finally get data after {self.timer_manager.next.value:.2f} seconds.")
                    logger.warning(f"{dataset_names} get_sample time: {self.timer_manager.get_sample.total:.2f} seconds, other process time: {self.timer_manager.next.value - self.timer_manager.get_sample.total:.2f} seconds.")
                self.timer_manager.reset("next")
                self.timer_manager.reset("get_sample")
                return result
            except TimeoutError as e:
                self.timer_manager.beat("next", "end")
                logger.warning(f"Data loading is still in progress after {self.timer_manager.next.value:.2f} seconds.")
                timout_triggered = True

    def set_mixing_status(self, mixing_status: MixingStatus):
        self._mixing_status = mixing_status

    def state_dict(self):
        _state_dict: dict = {}

        buffer_state_dict: dict[str, dict] = {}
        for v in self.nlp_buffer.values():
            buffer_state_dict[v.name] = v.state_dict()
        for v in self.image_buffer.values():
            buffer_state_dict[v.name] = v.state_dict()

        _state_dict["buffer_state_dict"] = buffer_state_dict
        _state_dict["batch_bucket"] = self.batch_bucket.state_dict()
        _state_dict["rng_state"] = self.rng.getstate()

        return copy.deepcopy(_state_dict)

    def load_state_dict(self, state_dict: dict):
        _state_dict = copy.deepcopy(state_dict)

        for _, v in self.nlp_buffer.items():
            if v.name in _state_dict["buffer_state_dict"]:
                v.load_state_dict(_state_dict["buffer_state_dict"][v.name])
        for _, v in self.image_buffer.items():
            if v.name in _state_dict["buffer_state_dict"]:
                v.load_state_dict(_state_dict["buffer_state_dict"][v.name])

        self.batch_bucket.load_state_dict(_state_dict["batch_bucket"])

        self.rng.setstate(_state_dict["rng_state"])

    def success_batch(self):
        status_return = copy.deepcopy(self.status)
        for k, v in self.status.items():
            self.status[k] = {"hit": 0, "miss": 0, "total": v["total"]}
        return status_return

    @property
    def image_ids(self) -> dict[str, list[int]]:
        _image_ids = {}
        for k, _len in self.image_seq_len.items():
            if _len == 0:
                _image_ids[k] = None  # never used, just for compatibility
            else:
                if self.eol is not None:
                    h, w = tuple(map(int, k.split("*")))
                    _image_ids[k] = self.hw_aspect_ratios_ids[k] + [self.boi] + ([self.image_placeholder_id] * w + [self.eol]) * h + [self.eoi]
                else:
                    _image_ids[k] = self.hw_aspect_ratios_ids[k] + [self.boi] + [self.image_placeholder_id] * _len + [self.eoi]
        return _image_ids

    def __fill_nlp_buffer(self, length: int, dataset_idx: int):
        while len(self.nlp_buffer[dataset_idx]) < length:
            self.timer_manager.beat("get_sample", "start")
            sample = next(self.datasets[dataset_idx])
            self.timer_manager.beat("get_sample", "end")
            self.status[sample.name]["hit"] += 1
            self.status[sample.name]["miss"] += sample.missing_samples
            text = sample.sample
            input_ids = self.tokenizer(text).input_ids + [self.tokenizer.eos_token_id]
            self.nlp_buffer[dataset_idx].enqueue(input_ids)

    def __fill_image_buffer(self, length: int, dataset_idx: int):
        while len(self.image_buffer[dataset_idx]) < length:
            match self.dataset_types[dataset_idx]:
                case "image_text_pair":
                    self.timer_manager.beat("get_sample", "start")
                    sample = next(self.datasets[dataset_idx])
                    self.timer_manager.beat("get_sample", "end")
                    self.status[sample.name]["hit"] += 1
                    self.status[sample.name]["miss"] += sample.missing_samples
                    image, caption = sample.sample
                    hw_aspect_ratio = _get_hw_aspect_ratio(image, self.down_factor)
                    pixel_values = [image.unsqueeze(0)]
                    
                    del image
                    input_ids = self.tokenizer(caption).input_ids

                    rng = random.Random(make_seed(self.seed, sample.index))
                    if rng.random() < self.drop_text_prob:
                        if self.tokenizer.bos_token is not None:
                            input_ids = input_ids[:1] + self.image_ids[hw_aspect_ratio] + input_ids[1:] + [self.tokenizer.eos_token_id]
                        else:
                            input_ids = self.image_ids[hw_aspect_ratio] + input_ids + [self.tokenizer.eos_token_id]
                    else:
                        input_ids = input_ids + self.image_ids[hw_aspect_ratio] + [self.tokenizer.eos_token_id]

                    labels = torch.tensor(copy.deepcopy(input_ids))
                    boi_idx = (labels == self.boi).nonzero(as_tuple=True)[0][0]
                    labels[boi_idx : boi_idx + self.image_seq_len[hw_aspect_ratio]] = IGNORE_INDEX
                    if input_ids[-2] == self.image_ids[hw_aspect_ratio][-1]:
                        # set eos to -100
                        labels[-1] = IGNORE_INDEX
                    labels = labels.tolist()
                    image_filtered_idx = [1] * len(pixel_values)
                
                case "interleave": 
                    self.timer_manager.beat("get_sample", "start")
                    sample = next(self.datasets[dataset_idx])
                    self.timer_manager.beat("get_sample", "end")
                    self.status[sample.name]["hit"] += 1
                    self.status[sample.name]["miss"] += sample.missing_samples
                    images, caption = sample.sample

                    # Use generator to process images, avoid loading all image aspect ratios simultaneously
                    hw_aspect_ratios = []
                    for image in images:
                        hw_aspect_ratios.append(_get_hw_aspect_ratio(image, self.down_factor))

                    placeholder_pattern = re.compile(r"<image_(\d+)>")
                    new_caption = caption
                    for match in placeholder_pattern.finditer(caption):
                        new_caption = new_caption.replace(match.group(0), DEFAULT_IMAGE_PLACEHOLDER_TOKEN)

                    pixel_values = [image.unsqueeze(0) for image in images]
                    if not pixel_values or len(pixel_values) != len(hw_aspect_ratios):
                        logger.warning(f"No valid images found in sample {sample.index}. Skipping.")
                        continue  # Skip this sample, get next one

                    # Release large objects early
                    del images
                    caption = new_caption
                    del new_caption
                    
                    new_pixel_values = []
                    new_caption = []
                    new_hw_aspect_ratios = []
                    image_seq_len = max([x for x in self.image_seq_len.values()])
                    MAX_INTERLEAVED_LEN = self.max_len // (2 * image_seq_len)
                    if len(pixel_values) > MAX_INTERLEAVED_LEN:
                        start_idx = self.rng.randint(0, len(pixel_values) - MAX_INTERLEAVED_LEN)
                        chosen_list = [x for x in range(start_idx, start_idx + MAX_INTERLEAVED_LEN)]
                        for idx in chosen_list:
                            new_pixel_values.append(pixel_values[idx])
                            new_caption.append(caption.split(DEFAULT_IMAGE_PLACEHOLDER_TOKEN)[idx])
                            new_hw_aspect_ratios.append(hw_aspect_ratios[idx])  
                        pixel_values = new_pixel_values
                        caption = f"{DEFAULT_IMAGE_PLACEHOLDER_TOKEN}".join(new_caption) + f"{DEFAULT_IMAGE_PLACEHOLDER_TOKEN}"
                        hw_aspect_ratios = new_hw_aspect_ratios
                        del new_pixel_values
                        del new_caption
                        del new_hw_aspect_ratios

                    if caption.count(DEFAULT_IMAGE_PLACEHOLDER_TOKEN) != len(pixel_values):
                        logger.warning(f"caption: {caption} has {caption.count(DEFAULT_IMAGE_PLACEHOLDER_TOKEN)} placeholders, but pixel_values has {len(pixel_values)} images")
                        continue
                    
                    if caption.count(DEFAULT_IMAGE_PLACEHOLDER_TOKEN) != len(hw_aspect_ratios):
                        logger.warning(f"caption: {caption} has {caption.count(DEFAULT_IMAGE_PLACEHOLDER_TOKEN)} placeholders, but hw_aspect_ratios has {len(hw_aspect_ratios)} aspect ratios")
                        continue
                    
                    input_ids = []
                    placeholder_count = 0
                    for token_id in self.tokenizer(caption).input_ids:
                        if token_id == self.image_placeholder_id:
                            input_ids.extend(self.image_ids[hw_aspect_ratios[placeholder_count]])
                            placeholder_count += 1
                        else:
                            input_ids.append(token_id)
                    input_ids = input_ids + [self.tokenizer.eos_token_id]
                    labels = torch.from_numpy(np.array(input_ids))
                    
                    if input_ids[-2] == self.image_ids[hw_aspect_ratios[-1]][-1]:
                        # set eos to -100
                        labels[-1] = IGNORE_INDEX
                    
                    boi_indices = (labels == self.boi).nonzero(as_tuple=True)[0]
                    for i, boi_idx in enumerate(boi_indices):
                        labels[boi_idx : boi_idx + self.image_seq_len[hw_aspect_ratios[i]]] = IGNORE_INDEX
                    labels = labels.tolist()
                    image_filtered_idx = [1] * len(pixel_values)
                    

                case "image_editing": 
                    self.timer_manager.beat("get_sample", "start")
                    sample = next(self.datasets[dataset_idx])
                    self.timer_manager.beat("get_sample", "end")
                    self.status[sample.name]["hit"] += 1
                    self.status[sample.name]["miss"] += sample.missing_samples
                    images, caption = sample.sample

                    # Use generator to process images, avoid loading all image aspect ratios simultaneously
                    hw_aspect_ratios = []
                    for image in images:
                        hw_aspect_ratios.append(_get_hw_aspect_ratio(image, self.down_factor))

                    placeholder_pattern = re.compile(r"<image_(\d+)>")
                    new_caption = caption
                    for match in placeholder_pattern.finditer(caption):
                        new_caption = new_caption.replace(match.group(0), DEFAULT_IMAGE_PLACEHOLDER_TOKEN)

                    pixel_values = [image.unsqueeze(0) for image in images]
                    if not pixel_values or len(pixel_values) != len(hw_aspect_ratios):
                        logger.warning(f"No valid images found in sample {sample.index}. Skipping.")
                        continue  # Skip this sample, get next one

                    # Release large objects early
                    del images
                    caption = new_caption
                    del new_caption

                    # Directly build final input_ids, avoid intermediate lists
                    input_ids = []
                    placeholder_count = 0
                    for token_id in self.tokenizer(caption).input_ids:
                        if token_id == self.image_placeholder_id:
                            input_ids.extend(self.image_ids[hw_aspect_ratios[placeholder_count]])
                            placeholder_count += 1
                        else:
                            input_ids.append(token_id)

                    input_ids = input_ids + [self.tokenizer.eos_token_id]
                    
                    # Optimize labels and image_filtered_idx construction
                    labels = torch.from_numpy(np.array(input_ids))
                    
                    if input_ids[-2] == self.image_ids[hw_aspect_ratios[-1]][-1]:
                        # set eos to -100
                        labels[-1] = IGNORE_INDEX
                    
                    boi_indices = (labels == self.boi).nonzero(as_tuple=True)[0]
                    image_filtered_idx = []

                    for i, boi_idx in enumerate(boi_indices):
                        labels[boi_idx : boi_idx + self.image_seq_len[hw_aspect_ratios[i]]] = IGNORE_INDEX
                        image_filtered_idx.append(1 if i == len(boi_indices) - 1 else 0)

                    labels = labels.tolist()
            self.image_buffer[dataset_idx].enqueue(sample, input_ids, labels, pixel_values, image_filtered_idx)

    def get_nlp_buffer(self, length: int, dataset_idx: int):
        self.__fill_nlp_buffer(length, dataset_idx)
        input_ids, attention_mask, labels = self.nlp_buffer[dataset_idx].dequeue(length)
        return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(labels)

    def get_image_buffer(self, length: int, dataset_idx: int):
        self.__fill_image_buffer(length, dataset_idx)
        if self.__worker_id == 0 and self.verbose:
            logger.debug(f"[ENQUEUE] {self.dataset_names[dataset_idx]} --> Indices Status:\n{self.image_buffer[dataset_idx].indices_status()}")
        input_ids, attention_mask, labels, pixel_values, samples, image_filtered_idx, waste_token_num = self.image_buffer[dataset_idx].dequeue(length)
        if self.__worker_id == 0 and self.verbose:
            logger.debug(f"[DEQUEUE] {self.dataset_names[dataset_idx]} --> Indices Status:\n{self.image_buffer[dataset_idx].indices_status()}")
        return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(labels), pixel_values, samples, image_filtered_idx, waste_token_num

    def batch_sanity_check(self, batch_data: dict):
        is_valid = True

        pv_bs = len(batch_data["pixel_values"])
        if pv_bs == 0:
            logger.error(f"`pixel_values` is empty, skip this batch ...")
            is_valid = False
        all_shape = [pixel_value.shape for pixel_value in batch_data["pixel_values"]]
        if len(set(all_shape)) > 1:
            num_placeholder_id = sum((input_ids == self.image_placeholder_id).sum().item() for input_ids in batch_data["input_ids"])
            pv_bs = []
            for pixel_value in batch_data["pixel_values"]:
                hw_aspect_ratio = _get_hw_aspect_ratio(pixel_value, self.down_factor)
                pv_bs.append(self.image_seq_len[hw_aspect_ratio])

            if sum(pv_bs) != num_placeholder_id:
                logger.error(f"`pixel_values` ({pv_bs * sum(pv_bs)}) is not matched with `input_ids` ({num_placeholder_id}), skip this batch ...")
                is_valid = False
        else:
            hw_aspect_ratio = _get_hw_aspect_ratio(batch_data["pixel_values"][0], self.down_factor)
            num_placeholder_id = sum((input_ids == self.image_placeholder_id).sum().item() for input_ids in batch_data["input_ids"])
            if pv_bs * self.image_seq_len[hw_aspect_ratio] != num_placeholder_id:
                logger.error(f"`pixel_values` ({pv_bs * self.image_seq_len[hw_aspect_ratio]}) is not matched with `input_ids` ({num_placeholder_id}), skip this batch ...")
                is_valid = False

        if len(batch_data["input_ids"]) != self.batch_size:
            logger.error(f"The input_ids ({len(batch_data['input_ids'])}) is not matched with batch_size ({self.batch_size}), skip this batch ...")
            is_valid = False

        if len(batch_data["image_filtered_idx"]) != len(batch_data["pixel_values"]):
            logger.error(f"The image_filtered_idx ({len(batch_data['image_filtered_idx'])}) is not matched with pixel_values ({len(batch_data['pixel_values'])}), skip this batch ...")
            is_valid = False

        return is_valid

    def data_generator(self):
        while True:
            # logger.debug(f"rank {dist_ctx.rank}, worker {self.__worker_id}, avg_pv_num: {self.shared_stats.get_avg_pv_num()}")
            dataset_idx = self.rng.choices(range(len(self.datasets)), weights=self.sampling_ratios, k=1)[0]
            match self.dataset_types[dataset_idx]:
                case "nlp":
                    input_ids, attention_mask, labels = self.get_nlp_buffer(self.max_len, dataset_idx)
                    pixel_values = []
                    samples = []
                    image_filtered_idx = []
                    waste_token_num = 0
                case "image_text_pair" | "interleave" | "image_editing":
                    input_ids, attention_mask, labels, pixel_values, samples, image_filtered_idx, waste_token_num = self.get_image_buffer(self.max_len, dataset_idx)
                case _:
                    raise ValueError(f"Unknown dataset type: {self.dataset_types[dataset_idx]}")

            dataset_indices, batch_data, image_samples_list, hw_aspect_ratio = self.batch_bucket.enqueue(
                dataset_idx,
                pixel_values,
                input_ids,
                attention_mask,
                labels, samples,
                image_filtered_idx,
                waste_token_num,
            )

            if batch_data is not None and self.batch_sanity_check(batch_data):
                if hw_aspect_ratio == ar2str(*ANY_ASPECT_RATIO):
                    # Ensure pixel_values list contains independent tensors for multiprocessing
                    batch_data["pixel_values"] = [pv.contiguous() if isinstance(pv, torch.Tensor) else pv for pv in batch_data["pixel_values"]]
                    batch_data["input_ids"] = torch.stack(batch_data["input_ids"]).contiguous()
                    batch_data["attention_mask"] = torch.stack(batch_data["attention_mask"]).contiguous()
                    batch_data["labels"] = torch.stack(batch_data["labels"]).contiguous()
                    batch_data["image_filtered_idx"] = torch.from_numpy(np.array(batch_data["image_filtered_idx"])).contiguous() if len(batch_data["image_filtered_idx"]) > 0 else None
                    batch_data["waste_token_num"] = torch.from_numpy(np.array(sum(batch_data["waste_token_num"]))).contiguous() if len(batch_data["waste_token_num"]) > 0 else torch.tensor(0).contiguous()

                    num_pv = []
                    for input_ids in batch_data["input_ids"]:
                        num_pv.append(
                            ((input_ids == self.boi).sum(dim=-1)).tolist()
                        )
                    pv2data = []
                    for num, dataset_idx in zip(num_pv, dataset_indices):
                        pv2data.extend([self.dataset_names[dataset_idx]] * num)
                else:
                    batch_data["pixel_values"] = torch.cat(batch_data["pixel_values"]).contiguous() if len(batch_data["pixel_values"]) > 0 else None
                    batch_data["input_ids"] = torch.stack(batch_data["input_ids"]).contiguous()
                    batch_data["attention_mask"] = torch.stack(batch_data["attention_mask"]).contiguous()
                    batch_data["labels"] = torch.stack(batch_data["labels"]).contiguous()
                    batch_data["image_filtered_idx"] = torch.from_numpy(np.array(batch_data["image_filtered_idx"])).contiguous() if len(batch_data["image_filtered_idx"]) > 0 else None
                    batch_data["waste_token_num"] = torch.from_numpy(np.array(sum(batch_data["waste_token_num"]))).contiguous() if len(batch_data["waste_token_num"]) > 0 else torch.tensor(0).contiguous()
                    num_pv = (
                        (batch_data["input_ids"] == self.image_placeholder_id).sum(dim=-1) // self.image_seq_len[hw_aspect_ratio]
                    ).tolist()
                    pv2data = []
                    for num, dataset_idx in zip(num_pv, dataset_indices):
                        pv2data.extend([self.dataset_names[dataset_idx]] * num)

                yield Batch(
                    batch_data=batch_data,
                    dataset_indices=dataset_indices,
                    pv2data=pv2data,
                    image_samples_list=image_samples_list,
                    status=self.success_batch(),
                    state_dict=self.state_dict(),
                    worker_id=self.__worker_id,
                    num_workers=self.__num_workers,
                )
# fmt: on


class MixedDataloader:

    def __init__(
        self,
        dataset: MixedDataset,
        *,
        num_workers: int = 0,
        pin_memory: bool = False,
        timeout: float = 0,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        memory_monitor_interval: int = -1,
        data_monitor_interval: float = 30.0,
    ):
        self._dataset = dataset
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.timeout = timeout
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers

        self.memory_monitor_interval = memory_monitor_interval
        self.data_monitor_interval = data_monitor_interval

        self._dataloader = None
        self._dataloader_iter = None

        self.mixing_status = MixingStatus(num_workers=num_workers)
        # For tracking data processing progress
        self.dataset_status = {}
        # For tracking different aspect ratio images
        self.hw_aspect_ratio_status = {key: 0 for key in self._dataset.hw_aspect_ratios_ids.keys()}
        self.hw_aspect_ratio_status[ar2str(*ANY_ASPECT_RATIO)] = 0

        self.timer_manager = TimerManager()
        self.data_monitor_executor = ThreadPoolExecutor(max_workers=1)

    def init(self):
        """
        Eagerly initialize the underlying PyTorch DataLoader iterator.

        This is useful when `num_workers > 0` because worker processes are
        created at `iter(dataloader)` time. Initializing early helps avoid
        forking after background threads (e.g. wandb/tensorboard/monitors)
        have started, which can cause deadlocks or worker crashes.
        """
        if self._dataloader_iter is None and self._dataloader is None:
            self._init_dataloader()

    def reset(self):
        if self._dataloader_iter is not None:
            shutdown_workers = getattr(self._dataloader_iter, "_shutdown_workers", None)
            if callable(shutdown_workers):
                shutdown_workers()
        self._dataloader_iter = None
        self._dataloader = None

    def _init_dataloader(self):
        def _worker_init_fn(worker_id):
            logger.info(
                f"Initializing worker {worker_id} (ip: {os.environ.get('MASTER_ADDR')}, pid: {os.getpid()}) "
                f"You can use `py-spy top --pid {os.getpid()} profile` to profile the process."
            )
            worker_monitor = PeriodicMemoryMonitor(f"Worker-{worker_id}", interval=self.memory_monitor_interval)
            worker_monitor.start()

            def cleanup():
                worker_monitor.stop()

            # Register cleanup handler
            import atexit

            atexit.register(cleanup)

            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                dataset = worker_info.dataset
                dataset.worker_monitor = worker_monitor

        # When num_workers == 0, PyTorch DataLoader does not allow setting prefetch_factor.
        # Must explicitly pass None, otherwise will raise:
        # "prefetch_factor option could only be specified in multiprocessing..."
        _prefetch_factor = self.prefetch_factor if self.num_workers > 0 else None
        # Similarly, persistent_workers can only be enabled when num_workers > 0
        _persistent_workers = self.persistent_workers if self.num_workers > 0 else False

        self._dataloader = DataLoader(
            self._dataset,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            timeout=self.timeout,
            worker_init_fn=_worker_init_fn,
            prefetch_factor=_prefetch_factor,
            persistent_workers=_persistent_workers,
        )
        self._dataloader_iter = iter(self._dataloader)

    def __len__(self):
        return len(self._dataset)

    @property
    def status(self):
        status_str = ""

        ### Dataset Status
        _reduce_stats = {}
        for k, v in self.dataset_status.items():
            _reduce_stats[k] = {
                "hit": dist_ctx.all_reduce_sum(v["hit"]),
                "miss": dist_ctx.all_reduce_sum(v["miss"]),
            }
            _reduce_stats[k]["total"] = v["total"]

        _reduce_stats["TOTAL"] = {
            "hit": sum(v["hit"] for v in _reduce_stats.values()),
            "miss": sum(v["miss"] for v in _reduce_stats.values()),
            "total": sum(v["total"] for v in _reduce_stats.values()),
        }
        # Add tabulate formatting
        table_data = []
        headers = ["Dataset", "Hit", "Miss", "Miss Rate", "Total", "Progress"]

        for dataset_name, _stats in _reduce_stats.items():
            hit = _stats["hit"]
            miss = _stats["miss"]
            total = _stats["total"]
            miss_rate = f"{(miss / (hit + miss)) * 100:.2f}%" if (hit + miss) > 0 else "0.00%"
            progress = f"{(hit + miss) / total * 100:.2f}%" if total > 0 else "0.00%"

            table_data.append(
                [dataset_name, str(LargeInt(hit)), str(LargeInt(miss)), miss_rate, str(LargeInt(total)), progress]
            )

        table = tabulate(table_data, headers=headers, tablefmt="grid")
        status_str += f"\nDataset Status:\n{table}"

        ### HW Aspect Ratio Status
        _reduce_stats = {}
        for k, v in self.hw_aspect_ratio_status.items():
            _reduce_stats[k] = dist_ctx.all_reduce_sum(v)
        _reduce_stats = {k: v / sum(_reduce_stats.values()) for k, v in _reduce_stats.items()}

        table_data = []
        headers = ["Aspect Ratio", "Percentage"]

        for k, v in _reduce_stats.items():
            table_data.append([k, v])

        table = tabulate(table_data, headers=headers, tablefmt="grid")
        status_str += f"\nHW Aspect Ratio Status:\n{table}"

        return status_str

    def _update_status(self, batch: Batch):
        self.mixing_status.update(batch)

        for k, v in batch.status.items():
            if k not in self.dataset_status:
                self.dataset_status[k] = {"hit": 0, "miss": 0, "total": 0}
            self.dataset_status[k]["hit"] += v["hit"]
            self.dataset_status[k]["miss"] += v["miss"]
            self.dataset_status[k]["total"] = v["total"]

        if isinstance(batch.batch_data["pixel_values"], list):
            self.hw_aspect_ratio_status[ar2str(*ANY_ASPECT_RATIO)] += len(batch.batch_data["pixel_values"])
        else:
            bsz, _, h, w = batch.batch_data["pixel_values"].shape
            h, w = h // self._dataset.down_factor, w // self._dataset.down_factor
            self.hw_aspect_ratio_status[ar2str(h, w)] += bsz

    def next_batch(self, step: int):
        
        if self._dataloader_iter is None and self._dataloader is None:
            self._init_dataloader()

        self.timer_manager.beat("data", "start")
        batch_future = self.data_monitor_executor.submit(next, self._dataloader_iter)
        timout_triggered = False
        while True:
            try:
                batch = batch_future.result(timeout=self.data_monitor_interval)
                self.timer_manager.beat("data", "end")
                if timout_triggered:
                    logger.warning(
                        f"[Step {step}, Rank {dist_ctx.rank}] Finally get batch after {self.timer_manager.data.value:.2f} seconds."
                    )
                self._update_status(batch)
                return batch
            except TimeoutError as e:
                self.timer_manager.beat("data", "end")
                logger.warning(
                    f"[Step {step}, Rank {dist_ctx.rank}] Batch loading is still in progress after {self.timer_manager.data.value:.2f} seconds."
                )
                timout_triggered = True

    def save_state_dict(self, dir: str):
        mixing_status_path = os.path.join(dir, "mixing_status", f"mixing_status_rank{dist_ctx.rank}.pt")
        os.makedirs(os.path.dirname(mixing_status_path), exist_ok=True)
        torch.save(self.mixing_status.state_dict(), mixing_status_path)

        dataset_status_path = os.path.join(dir, "dataset_status", f"dataset_status_rank{dist_ctx.rank}.json")
        os.makedirs(os.path.dirname(dataset_status_path), exist_ok=True)
        with open(dataset_status_path, "w") as f:
            json.dump(self.dataset_status, f, indent=4)

        hw_aspect_ratio_status_path = os.path.join(
            dir, "hw_aspect_ratio_status", f"hw_aspect_ratio_status_rank{dist_ctx.rank}.json"
        )
        os.makedirs(os.path.dirname(hw_aspect_ratio_status_path), exist_ok=True)
        with open(hw_aspect_ratio_status_path, "w") as f:
            json.dump(self.hw_aspect_ratio_status, f, indent=4)

    def load_state_dict(self, dir: str):
        mixing_status_path = os.path.join(dir, "mixing_status", f"mixing_status_rank{dist_ctx.rank}.pt")
        self.mixing_status.load_state_dict(torch.load(mixing_status_path))
        self._dataset.set_mixing_status(self.mixing_status)

        dataset_status_path = os.path.join(dir, "dataset_status", f"dataset_status_rank{dist_ctx.rank}.json")
        with open(dataset_status_path, "r") as f:
            self.dataset_status = json.load(f)

        hw_aspect_ratio_status_path = os.path.join(
            dir, "hw_aspect_ratio_status", f"hw_aspect_ratio_status_rank{dist_ctx.rank}.json"
        )
        with open(hw_aspect_ratio_status_path, "r") as f:
            self.hw_aspect_ratio_status = json.load(f)

# sudo torchrun --nproc_per_node=8 --master_port=12345  nextstep/datasets/mixed_dataset.py
if __name__ == "__main__":
    import torchvision.transforms as transforms
    from tqdm.auto import tqdm
    from transformers import AutoTokenizer

    from nextstep.datasets.test_utils import setup_test_environment, test_mixed_dataset_recovery

    from configs.data.pretrain_data_256px import data_info_list
    from nextstep.model_zoos import MODEL_ZOOS
    from nextstep.models.nextstep.tokenization_nextstep import DEFAULT_IMAGE_AREA_TOKEN, special_tokens_dict
    from nextstep.models.nextstep.aspect_ratio import ar2str, get_ar_base, str2ar
    from nextstep.utils.image_utils import center_crop_arr

    setup_test_environment()
    
    HW_ASPECT_RATIOS = [
        (16, 16),  # 256
    ]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ZOOS["Qwen/Qwen2.5-3B"])
    tokenizer.add_special_tokens(special_tokens_dict)

    def train_transform_wrapper(crop = True):
        return transforms.Compose(
            [
                (
                    transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 256, crop=True))
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
    train_transform = train_transform_wrapper

    def get_dataset(num_workers = 0):
        down_factor = 16
        image_tokens_len = 256
        image_grid_size = image_tokens_len / down_factor
        ar_base = get_ar_base(HW_ASPECT_RATIOS)
        hw_aspect_ratios = [
            (round(h / ar_base * image_grid_size), round(w / ar_base * image_grid_size)) for h, w in HW_ASPECT_RATIOS
        ]
        # hw_aspect_ratios = [(h, w) for (h, w) in hw_aspect_ratios if h == w]
        hw_aspect_ratios_ids = {
            ar2str(h, w): (
                tokenizer(f"{DEFAULT_IMAGE_AREA_TOKEN}{ar2str(h, w)}").input_ids[1:]
                if tokenizer.bos_token is not None
                else tokenizer(f"{DEFAULT_IMAGE_AREA_TOKEN}{ar2str(h, w)}").input_ids
            )
            for h, w in hw_aspect_ratios
        }
        dataset = MixedDataset(
            data_info_list=copy.deepcopy(data_info_list),
            batch_size=4,
            tokenizer=tokenizer,
            down_factor=16,
            hw_aspect_ratios_ids=hw_aspect_ratios_ids,
            image_seq_len={k: str2ar(k)[0] * str2ar(k)[1] for k in hw_aspect_ratios_ids.keys()},
            max_len=image_tokens_len * 2,
            drop_text_prob=0.3,
            prefetch_factor=4,
            persistent_workers=True,
            num_workers=num_workers,
            dataset_kwargs={"post_processing": train_transform},
        )
        return dataset

    # from nextstep.utils.debug import debug
    # debug(rank=0, stop=True)
    test_mixed_dataset_recovery(get_dataset, num_workers=2)
    
