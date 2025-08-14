import time

import megfile
import streamlit as st

from nextstep.data.indexed_tar import IndexedTarSamples
from nextstep.data_zoos import get_cache_path
from nextstep.service.utils import resize_image_h
from nextstep.utils.image_utils import IMAGE_EXT
from nextstep.utils.loguru import logger
from nextstep.utils.video_utils import VIDEO_EXT

tar_img_size = 512


@st.cache_data
def preprocessed_indexed_tar_samples(path: str, size: int = 256) -> list[dict]:
    start_time = time.time()
    processed_indexed_tar = []
    with IndexedTarSamples(path, cache_to=get_cache_path(path), delete_cache=False, backend="memory") as indexed_tar:
        for sample in indexed_tar:
            new_sample = {}
            for key, value in sample.items():
                if key[1:] in IMAGE_EXT:
                    if not isinstance(value, list):
                        value = [value]
                    new_sample[key] = [(resize_image_h(img, size), img.size) for img in value]
                elif key[1:] == "json":
                    new_sample[key] = {k: v for k, v in value.items() if "movq" not in k}
                else:
                    new_sample[key] = value
            processed_indexed_tar.append(new_sample)
    logger.info(f"Load {path} time: {time.time() - start_time:.2f} seconds")
    return processed_indexed_tar


def preview_tar():
    # Input tar file path
    tar_path = st.sidebar.text_input("Enter tar file path:", placeholder="Path to your tar file")
    col_num = st.sidebar.number_input("Number of columns", min_value=1, max_value=10, value=2, step=1)

    if tar_path and megfile.smart_exists(tar_path):
        tar = preprocessed_indexed_tar_samples(tar_path, size=tar_img_size)

        st.error(f"Number of samples: {len(tar)}, but we only show 1000 or less samples")
        tar = tar[:1000]

        for i in range(0, len(tar), col_num):
            group = tar[i : i + col_num]
            cols = st.columns(len(group))
            for idx, (show_data, col) in enumerate(zip(group, cols)):
                with col:
                    st.warning(f"Sample Index: {i + idx}")
                    keys = show_data.keys()
                    sorted_keys = sorted(keys, key=lambda x: 0 if x[1:] in IMAGE_EXT + VIDEO_EXT else 1)
                    for key in sorted_keys:
                        if key[1:] in IMAGE_EXT:
                            for i in range(0, len(show_data[key]), 3):
                                inner_group = show_data[key][i : i + 3]
                                inner_cols = st.columns(len(inner_group))
                                for (_img, _size), inner_col in zip(inner_group, inner_cols):
                                    with inner_col:
                                        st.info(f"Image size (WxH): {_size}")
                                        st.image(_img)

                        elif key[1:] in VIDEO_EXT:
                            st.video(show_data[key])

                        elif key[1:] == "json":
                            st.info(f"Key: {key}")
                            st.write({k: v for k, v in show_data[key].items() if "movq" not in k})

                        else:
                            st.info(f"Key: {key}")
                            st.write(show_data[key])
            st.divider()

    else:
        if tar_path:
            st.error("Invalid tar file path or file does not exist")
        st.info("Please enter a valid tar file path in the sidebar")
