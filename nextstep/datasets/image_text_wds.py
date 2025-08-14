import copy
import random
import traceback
from typing import Callable, Generator

import PIL.Image
import torch
from transformers import BaseImageProcessor

from nextstep.data.indexed_tar_dataset import IndexedTarDataset, RetSample
from nextstep.datasets.data_logger import data_logger as logger
from nextstep.datasets.utils import find_key, format_list, make_seed
from nextstep.utils.image_utils import IMAGE_EXT
from nextstep.utils.misc import LargeInt
from nextstep.datasets.comm import (
    FILTERED_PATTERNS,
    filter_area,
    filter_aspect_ratio,
    filter_hw,
    filter_keywords,
    filter_other,
)


class ImageTextWDS(IndexedTarDataset):
    def __init__(
        self,
        data_info: dict | None = None,
        shuffle_size: int = LargeInt("50K"),
        lru_size: int = 32,
        seed: int = 42,
        post_processing: Callable[[PIL.Image.Image], PIL.Image.Image] | None = None,
        **kwargs,
    ):
        super(ImageTextWDS, self).__init__(data_info=data_info, shuffle_size=shuffle_size, lru_size=lru_size, seed=seed, **kwargs)
        try:
            self.post_processing = post_processing(crop=True)
        except:
            self.post_processing = post_processing
        # caption keys and ratio
        self.caption_keys = data_info.get("caption_keys", ["caption"])
        self.caption_ratio = data_info.get("caption_ratio", [1])

        # filter rules
        self.other_filter: dict = data_info.get("filter", {})
        self.height_filter: int = self.other_filter.pop("height", 0)
        self.width_filter: int = self.other_filter.pop("width", 0)
        self.area_filter: int = self.other_filter.pop("area", 0)
        self.aspect_ratio_filter: float = self.other_filter.pop("aspect_ratio", 999)
        keywords: list[str] = self.other_filter.pop("keywords", [])
        keywords_drop_prob: list[float] = self.other_filter.pop("keywords_drop_prob", [])
        self.keywords_filter: list[tuple[str, float]] = [(k, float(d)) for k, d in zip(keywords, keywords_drop_prob)]

    def __repr__(self):
        __repr = f"""
Dataset: {self.name}
|- Total URLs: {len(self.cur_urls)}
|- Total Samples: {len(self)}
|- Matched Keys: {format_list(self.matched_keys, "|- ", 4)}
|- Filter Rules:
|  |- Height, Width: ({self.height_filter}, {self.width_filter})
|  |- Aspect Ratio: {self.aspect_ratio_filter}
|  |- Keywords: {format_list(self.keywords_filter, "|  |- ", 4)}
|  |- Other: {self.other_filter}
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
                image, info = self.process_sample(sample)
            except Exception as e:
                logger.error(f"{self.name}: Error while processing sample --> {e}")
                logger.error_once(f"Traceback:\n{traceback.format_exc()}")
                self.skip_sample(index)
                continue

            is_filtered, msg = self.filter_sample(info)
            if is_filtered:
                logger.warning(f"{self.name}: Sample from shard {url} at index {sample_index} is filtered: {msg}")
                self.skip_sample(index)
                continue

            try:
                image = self.process_image(image)
            except Exception as e:
                logger.error(f"{self.name}: Error while processing image --> {e}")
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
                sample=(image, info["caption"]),
                worker_id=self.worker_id,
                num_workers=self.num_workers,
            )

    def filter_sample(self, info: dict) -> tuple[bool, str]:
        is_filtered = False
        msg = ""
        if filter_hw(info, self.height_filter, self.width_filter):
            is_filtered = True
            msg += f"\nHeight, Width: ({info['height']}, {info['width']})"
        if filter_area(info, self.area_filter):
            is_filtered = True
            msg += f"\nArea: ({info['height']} * {info['width']} = {info['height'] * info['width']})"
        if filter_aspect_ratio(info, self.aspect_ratio_filter):
            is_filtered = True
            msg += f"\nAspect Ratio: {info['aspect_ratio']}"
        if filter_keywords(info, self.keywords_filter, self.seed):
            is_filtered = True
            msg += f"\nCaption: {info['caption']}"
        if filter_other(info, self.other_filter):
            is_filtered = True
            for k, _ in self.other_filter.items():
                if k not in info:
                    continue
                msg += f"\n{k}: {info[k]}"
        return is_filtered, msg
    
    def apply_template(self, caption: str, template: str) -> str:
        return template.format(text=caption)

    def process_image(self, image: PIL.Image.Image) -> PIL.Image.Image | torch.Tensor:
        if self.post_processing is None:
            return image
        elif isinstance(self.post_processing, BaseImageProcessor):
            return self.post_processing(image, return_tensors="pt")["pixel_values"].squeeze(0)
        else:  # torch transform
            return self.post_processing(image)

    def process_sample(self, sample: dict) -> tuple[torch.Tensor, dict]:
        """sample is an item from IndexedTarSamples, usually is a dict. Process it to yield a desired output."""
        if find_key(sample, IMAGE_EXT) is not None:  # image
            image = self.__extract_image__(sample, find_key(sample, IMAGE_EXT))
        else:
            raise ValueError("No image data found in the sample!")

        info = self.__extract_info__(sample)
        w, h = image.size
        aspect_ratio = 1.0 * max(h, w) / min(h, w)
        info["height"] = h
        info["width"] = w
        info["aspect_ratio"] = aspect_ratio

        return image, info

    def __extract_image__(self, sample: dict, key: str = "jpg") -> PIL.Image.Image:
        return sample[f".{key}"]

    def __extract_info__(self, sample: dict, **kwargs) -> str:
        rng = random.Random(make_seed(self.seed, self.ep, self.sub_ep, self.cur_index))
        if ".json" in sample:
            meta_info = sample[".json"]

            # check available keys
            available_caption_keys = []
            available_caption_ratio = []
            for key, ratio in zip(self.caption_keys, self.caption_ratio):
                if isinstance(key, list):
                    try:
                        _meta_info = copy.deepcopy(meta_info)
                        for _key in key:
                            _meta_info = _meta_info[_key]
                        if isinstance(_meta_info, str):
                            available_caption_keys.append(key)
                            available_caption_ratio.append(ratio)
                    except:
                        continue
                else:
                    try:
                        caption = meta_info[key]
                        if isinstance(caption, str):
                            available_caption_keys.append(key)
                            available_caption_ratio.append(ratio)
                        elif isinstance(caption, (list, dict)) and len(caption) > 0:
                            if isinstance(caption, dict):
                                caption = list(caption.values())
                            caption = [c for c in caption if isinstance(c, str)]
                            if len(caption) > 0:
                                available_caption_keys.append(key)
                                available_caption_ratio.append(ratio)
                    except:
                        continue

            if len(available_caption_keys) == 0:
                raise ValueError("No available caption keys found in the sample!")

            # random select one caption by caption_ratio
            caption_key = rng.choices(available_caption_keys, weights=available_caption_ratio)[0]
            if isinstance(caption_key, list):
                _meta_info = copy.deepcopy(meta_info)
                for _key in caption_key:
                    _meta_info = _meta_info[_key]
                caption = _meta_info
            else:
                caption = meta_info[caption_key]

            if isinstance(caption, dict):
                caption = list(caption.values())

            if isinstance(caption, list) and len(caption) > 0:
                caption = [c for c in caption if isinstance(c, str)]
                caption = rng.choice(caption)

            # "caption_all", "caption_all_cn"
            # delete all patterns in caption
            if "_all" in caption_key:
                for pattern in FILTERED_PATTERNS:
                    caption = caption.replace(pattern, "")

            if rng.random() < 0.5:
                caption = caption.capitalize()

            if caption == "":
                raise ValueError(f"Caption is empty for dataset {self.name}, meta_info: {meta_info}")
            
            if "templates" in self.data_info and len(self.data_info["templates"]) > 0:
                caption = self.apply_template(caption, rng.choice(self.data_info["templates"]))

            ret_info = {"caption": caption}
            for k in self.other_filter.keys():
                if k not in meta_info:
                    continue
                ret_info[k] = meta_info[k]

            return ret_info
        else:
            raise ValueError("No text data found in the sample!")

# sudo torchrun --nproc_per_node=8 --master_port=12345 nextstep/datasets/image_text_wds.py
if __name__ == "__main__":
    from nextstep.datasets.test_utils import setup_test_environment, run_dataset_tests

    setup_test_environment()
    
    # from nextstep.utils.debug import debug
    # debug(rank=0, stop=True)

    def get_dataset():
        return ImageTextWDS(
            data_info={
                "data_type": "image_text_pair",
                "name": "text2image/BLIP3o-60k",
                "caption_keys": ["caption"],
                "caption_ratio": [1],
                "filter": {
                    "area": [256*256, 1024*1024],
                    "aspect_ratio": 6,
                },
                "samples": LargeInt("58K"),
            },
            lru_size=8,
            shuffle_size=1_000,
        )  # 2106 samples

    run_dataset_tests(get_dataset, num_workers=8)
