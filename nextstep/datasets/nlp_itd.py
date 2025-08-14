import traceback
from typing import Generator

from nextstep.data.indexed_tar_dataset import IndexedTarDataset, RetSample
from nextstep.datasets.data_logger import data_logger as logger
from nextstep.datasets.utils import format_list
from nextstep.utils.misc import LargeInt


class NLPITD(IndexedTarDataset):

    def __init__(
        self,
        data_info: dict | None = None,
        shuffle_size: int = LargeInt("50K"),
        lru_size: int = 32,
        seed: int = 42,
        **kwargs,
    ):
        super(NLPITD, self).__init__(data_info=data_info, shuffle_size=shuffle_size, lru_size=lru_size, seed=seed)

        caption_keys = data_info.get("caption_keys", ["text"])
        assert len(caption_keys) == 1, "Only one caption key is supported"
        self.caption_key = caption_keys[0]

    def __repr__(self):
        __repr = f"""
Dataset: {self.name}
|- Total URLs: {len(self.cur_urls)}
|- Total Samples: {len(self)}
|- Matched Keys: {format_list(self.matched_keys, "|- ", 4)}
|---------------------------------------------------------- """
        return __repr

    def data_generator(self) -> Generator[RetSample, None, None]:
        while True:
            index = self.next_index()
            url, sample_index = self.index2indices(index)

            try:
                shard = self.lru_shards.get_shard(url)
            except Exception as e:
                logger.error(f"{self.name}: Error while opening shard ({url}) --> {e}")
                logger.error_once(f"Traceback:\n{traceback.format_exc()}")
                self.skip_sample(index)
                continue

            try:
                sample = shard[sample_index]
            except Exception as e:
                logger.error(f"{self.name}: Error while get sample from shard ({url}) at index {sample_index} --> {e}")
                logger.error_once(f"Traceback:\n{traceback.format_exc()}")
                self.skip_sample(index)
                continue

            try:
                text = sample[".json"][self.caption_key]
            except Exception as e:
                logger.error(f"{self.name}: Error while get text from sample --> {e}")
                logger.error_once(f"Traceback:\n{traceback.format_exc()}")
                self.skip_sample(index)
                continue

            missing_indices, missing_urls_indices, missing_samples = self.success_sample()
            yield RetSample(
                name=self.name,
                data_type=self.data_type,
                missing_indices=missing_indices,
                missing_urls_indices=missing_urls_indices,
                missing_samples=missing_samples,
                index=index,
                url_index=(url, sample_index),
                sample=text,
                worker_id=self.worker_id,
                num_workers=self.num_workers,
            )
