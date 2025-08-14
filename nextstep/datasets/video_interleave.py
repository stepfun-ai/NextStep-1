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
import re
from nextstep.datasets.comm import (
    FILTERED_PATTERNS,
    filter_area,
    filter_aspect_ratio,
    filter_hw,
    filter_keywords,
    filter_other,
)


def handle_story_caption(caption: dict, caption_key: str) -> str:
    ret_caption = ""
    if random.random() < 0.5:
        ret_caption += caption["title"]
        ret_caption += "." if "cn" not in caption_key else "。"
    if random.random() < 0.5:
        ret_caption += caption["summary"]
    ret_caption += caption["content"]
    return ret_caption


class VideoInterleave(IndexedTarDataset):
    def __init__(
        self,
        data_info: dict | None = None,
        shuffle_size: int = LargeInt("50K"),
        lru_size: int = 32,
        seed: int = 42,
        post_processing: Callable[[PIL.Image.Image], PIL.Image.Image] | None = None,
        **kwargs,
    ):
        super(VideoInterleave, self).__init__(data_info=data_info, shuffle_size=shuffle_size, lru_size=lru_size, seed=seed, **kwargs)
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
            msg += f"\Area: ({info['height']} * {info['width']} = {info['height'] * info['width']})"
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
    
    def get_sample(self, pixel_values: list[PIL.Image.Image], caption: str) -> tuple[list[PIL.Image.Image], str]:        
        if len(pixel_values) <= MAX_NUM_IMAGES_PER_SEQ:
            return pixel_values, caption
        
        new_caption = []
        
        # TODO: only for t2it2i..., should be generalized
        start_idx = self.rng.randint(0, len(pixel_values) - MAX_NUM_IMAGES_PER_SEQ)
        chosen_indices = list(range(start_idx, start_idx + MAX_NUM_IMAGES_PER_SEQ))
        pixel_values = [pixel_values[idx] for idx in chosen_indices]
        
        for idx in chosen_indices:
            new_caption.append(caption.split(DEFAULT_IMAGE_PLACEHOLDER_TOKEN)[idx])
        caption = f"{DEFAULT_IMAGE_PLACEHOLDER_TOKEN}".join(new_caption) + f"{DEFAULT_IMAGE_PLACEHOLDER_TOKEN}"
        
        return pixel_values, caption
            

    def process_image(self, image: list[PIL.Image.Image]) -> list[PIL.Image.Image] | list[torch.Tensor]:
        if self.post_processing is None:
            return image
        elif isinstance(self.post_processing, BaseImageProcessor):
            return [self.post_processing(_img, return_tensors="pt")["pixel_values"].squeeze(0) for _img in image]
        else:  # torch transform
            return [self.post_processing(_img) for _img in image]

    def process_sample(self, sample: dict) -> tuple[list[PIL.Image.Image], dict]:
        """sample is an item from IndexedTarSamples, usually is a dict. Process it to yield a desired output."""

        info = self.__extract_info__(sample)
        caption = info["caption"]

        rng = random.Random(make_seed(self.seed, self.ep, self.sub_ep, self.cur_index))
        if rng.random() < 0.1:  # FIXME hard code
            # Keep only <image_X> tokens
            caption = "".join(re.findall(r"<image_\d+>", caption))

        placeholder_pattern = re.compile(r"<image_(\d+)>")

        # Collect images that meet the conditions
        valid_matches = []
        for match in placeholder_pattern.finditer(caption):
            img_idx = int(match.group(1))
            valid_matches.append((match, img_idx))

        if find_key(sample, IMAGE_EXT) is not None:  # image
            images = self.__extract_image__(sample, find_key(sample, IMAGE_EXT), valid_matches)
        else:
            raise ValueError("No image data found in the sample!")

        count = len(re.findall(r"<image_\d+>", caption))
        if count != len(images):  # image
            raise ValueError("The number of <image_X> tokens in caption does not match the number of images!")

        if not isinstance(images, list):
            images = [images]

        w_list, h_list = [image.size[0] for image in images], [image.size[1] for image in images]
        w, h = min(w_list), min(h_list)
        aspect_ratio_list = [1.0 * max(h_list[i], w_list[i]) / min(h_list[i], w_list[i]) for i in range(len(w_list))]
        aspect_ratio = max(aspect_ratio_list)
        
        info["height"] = h
        info["width"] = w
        info["aspect_ratio"] = aspect_ratio
        return images, info

    def __extract_image__(
        self, sample: dict, key: str = "jpg", valid_matches: list[tuple[re.Match, int]] = None
    ) -> list[PIL.Image.Image]:
        images = []
        # if len(sample[f".{key}"]) > 50:
        #     raise ValueError(f"interleave {self.name}, sample {sample} has more than 50 images!")
        if not isinstance(sample[f".{key}"], list):
            sample[f".{key}"] = [sample[f".{key}"]]

        for match, img_idx in valid_matches:
            images.append(sample[f".{key}"][img_idx])
        return images

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
                            caption = [c for c in caption if isinstance(c, (str, list))]
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
                for k in caption.keys(): # randomly chose one sample if caption is a list
                    if len(caption[k]) == 0:
                        raise ValueError(f"No captions found in sample with key {caption_key}!")
                    caption[k] = random.choice(caption[k]) if (isinstance(caption[k], list)) else caption[k]
                    assert isinstance(caption[k], str)
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

            ret_info = {"caption": caption}
            for k in self.other_filter.keys():
                if k not in meta_info:
                    continue
                ret_info[k] = meta_info[k]

            return ret_info
        else:
            raise ValueError("No text data found in the sample!")

# sudo torchrun --nproc_per_node=8 --master_port=12345 nextstep/datasets/video_interleave.py
if __name__ == "__main__":
    from nextstep.datasets.test_utils import setup_test_environment, run_dataset_tests

    setup_test_environment()
    
    # from nextstep.utils.debug import debug
    # debug(rank=0, stop=True)

    def get_dataset():
        return VideoInterleave(
            data_info={
                "data_type": "interleave",
                "name": "interleave/multimodal_textbook",
                "caption_keys": ["caption", "caption_i2t"],
                "caption_ratio": [4, 1],
                "filter": {
                    "area": [256 * 256, 1024 * 1024],
                    "aspect_ratio": 6,
                },
                "samples": LargeInt("32K"),
            },
            lru_size=8,
            shuffle_size=1_000,
        )

    run_dataset_tests(get_dataset, num_workers=8)
