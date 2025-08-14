import copy
import json
import os
from functools import partial

import megfile
import pandas as pd
import streamlit as st

from nextstep.data.indexed_tar import IndexedTarSamples, ULockFile
from nextstep.data_zoos import DATA_META_PATH, get_cache_path
from nextstep.service.utils import init_session_state, resize_image_h
from nextstep.utils.image_utils import IMAGE_EXT
from nextstep.utils.loguru import logger
from nextstep.utils.misc import LargeInt
from nextstep.utils.video_utils import VIDEO_EXT
import random

dataset_img_size = 512


# Function to increment the count
def next(samples: list, dataset_name: str):
    if st.session_state.count is None:
        st.session_state.count = 0
    if st.session_state.local_index is None:
        st.session_state.local_index = 0
    if st.session_state.tar_index is None:
        st.session_state.tar_index = 0
    st.session_state.count += 1
    st.session_state.count %= sum(samples)
    st.session_state.local_index += 1
    if st.session_state.local_index >= samples[st.session_state.tar_index]:
        st.session_state.tar_index += 1
        if st.session_state.tar_index >= len(samples):
            st.session_state.tar_index = 0
        st.session_state.local_index = 0
        st.session_state.cur_tar = indexed_tar_samples(
            os.path.join(
                st.session_state.data_meta[dataset_name]["dir"],
                st.session_state[dataset_name]["keys"][st.session_state.tar_index],
            )
        )


# Function to decrement the count
def prev(samples: list, dataset_name: str):
    if st.session_state.count is None:
        st.session_state.count = 0
    if st.session_state.local_index is None:
        st.session_state.local_index = 0
    if st.session_state.tar_index is None:
        st.session_state.tar_index = 0
    st.session_state.count -= 1
    st.session_state.count %= sum(samples)
    st.session_state.local_index -= 1
    if st.session_state.local_index < 0:
        st.session_state.tar_index -= 1
        if st.session_state.tar_index < 0:
            st.session_state.tar_index = len(samples) - 1
        st.session_state.local_index = samples[st.session_state.tar_index] - 1
        st.session_state.cur_tar = indexed_tar_samples(
            os.path.join(
                st.session_state.data_meta[dataset_name]["dir"],
                st.session_state[dataset_name]["keys"][st.session_state.tar_index],
            )
        )


def on_selectbox_change():
    st.session_state.count = 0
    st.session_state.local_index = 0
    st.session_state.tar_index = 0
    st.session_state.cur_tar = None


def on_jump_to_index_change():
    st.session_state.count = None
    st.session_state.local_index = None
    st.session_state.tar_index = None
    st.session_state.cur_tar = None


def count2index(count: int, samples: list):
    count = count % sum(samples)
    for i, sample in enumerate(samples):
        if count < sample:
            return i, count
        count -= sample
    raise ValueError(f"Count {count} is out of range for samples {samples}")


def show_progress_bar_and_navigation_buttons(samples: list, dataset_name: str):
    # Jump to index
    index_to_jump = st.number_input(
        "Jump to Index",
        min_value=0,
        max_value=sum(samples) - 1,
        step=1,
        on_change=on_jump_to_index_change,
    )
    if st.session_state.count is None:
        st.session_state.count = index_to_jump
        st.session_state.tar_index, st.session_state.local_index = count2index(index_to_jump, samples)

    # Progress bar
    st.write(f"Progress: {st.session_state.count}/{sum(samples)}")
    st.progress(st.session_state.count / sum(samples))

    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        st.button("Previous", on_click=partial(prev, samples, dataset_name))
    with col2:
        st.button("Next", on_click=partial(next, samples, dataset_name))


def find_first_tar(directory):
    try:
        for entry in megfile.smart_scandir(directory):
            if entry.is_file() and entry.name.endswith(".tar"):
                return entry.path
            elif entry.is_dir():
                result = find_first_tar(entry.path)
                if result:
                    return result
        return None
    except PermissionError:
        return None


def indexed_tar_samples(path: str) -> IndexedTarSamples:
    return IndexedTarSamples(path, cache_to=get_cache_path(path), delete_cache=False, backend="mmap")


def preview_dataset():
    ################################################################################
    # Initialize session state
    ################################################################################
    if "data_meta" not in st.session_state:
        with open(DATA_META_PATH, "r") as f:
            data_meta = json.load(f)
        st.session_state["data_meta"] = data_meta

    initial_state = dict(count=0, tar_index=0, local_index=0, cur_tar=None)
    init_session_state(**initial_state)

    ################################################################################
    # Load dataset
    ################################################################################
    dataset_name_list = st.session_state.data_meta.keys()
    error_dataset_names = []
    for dataset_name in dataset_name_list:
        try:
            if dataset_name not in st.session_state:
                tar_meta_path = st.session_state.data_meta[dataset_name]["tar_meta_path"]
                if tar_meta_path is not None and megfile.smart_exists(tar_meta_path):
                    with megfile.smart_open(tar_meta_path, "r") as f:
                        tar_meta = json.load(f)

                    tar_meta_keys = [key for key in tar_meta.keys() if tar_meta[key]["num_samples"] > 0]
                    st.session_state[dataset_name] = {
                        "tar_meta": tar_meta,
                        "keys": tar_meta_keys,
                        "samples": [tar_meta[key]["num_samples"] for key in tar_meta_keys],
                        "missing_tar_meta": False,
                    }
                else:
                    first_tar = find_first_tar(st.session_state.data_meta[dataset_name]["dir"])
                    tar_samples = indexed_tar_samples(first_tar)
                    tar_meta = {
                        os.path.relpath(first_tar, st.session_state.data_meta[dataset_name]["dir"]): {
                            "num_samples": len(tar_samples),
                            "size": 0,
                            "checksum": 0,
                        }
                    }
                    del tar_samples
                    st.session_state[dataset_name] = {
                        "tar_meta": tar_meta,
                        "keys": list(tar_meta.keys()),
                        "samples": [tar_meta[key]["num_samples"] for key in tar_meta.keys()],
                        "missing_tar_meta": True,
                    }

        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {e}")
            error_dataset_names.append(dataset_name)
    dataset_name_list = [dataset_name for dataset_name in dataset_name_list if dataset_name not in error_dataset_names]

    ################################################################################
    # Sidebar
    ################################################################################
    dataset_name = st.sidebar.selectbox(
        "Choose a dataset",
        dataset_name_list,
        on_change=on_selectbox_change,
        placeholder="Choose a dataset to preview...",
    )

    num_samples_list = [LargeInt(sum(st.session_state[dataset_name]["samples"])) for dataset_name in dataset_name_list]
    data = pd.DataFrame(
        {
            "Name": [dataset_name for dataset_name in dataset_name_list],
            "Type": [st.session_state.data_meta[dataset_name]["data_type"] for dataset_name in dataset_name_list],
            "Num Samples": [str(num_samples) for num_samples in num_samples_list],
            "Tar Meta": [not st.session_state[dataset_name]["missing_tar_meta"] for dataset_name in dataset_name_list],
        }
    ).set_index("Name")
    st.sidebar.dataframe(data, use_container_width=True, hide_index=False)

    def _check_num_samples(data_meta, dataset_name_list, num_samples_list):
        for dataset_name, _num_samples in zip(dataset_name_list, num_samples_list):
            if data_meta[dataset_name]["num_samples"] != _num_samples:
                return False
        return True

    new_data_meta = copy.deepcopy(st.session_state.data_meta)
    if not _check_num_samples(new_data_meta, dataset_name_list, num_samples_list):
        for dataset_name, _num_samples in zip(dataset_name_list, num_samples_list):
            new_data_meta[dataset_name]["num_samples"] = _num_samples
        with ULockFile(DATA_META_PATH + ".lock"):
            with open(DATA_META_PATH, "w", encoding="utf-8") as f:
                json.dump(new_data_meta, f, indent=4, ensure_ascii=False)

    ################################################################################
    # Main
    ################################################################################
    show_progress_bar_and_navigation_buttons(st.session_state[dataset_name]["samples"], dataset_name)

    if st.session_state.cur_tar is None:
        st.session_state.cur_tar = indexed_tar_samples(
            os.path.join(
                st.session_state.data_meta[dataset_name]["dir"],
                st.session_state[dataset_name]["keys"][st.session_state.tar_index],
            )
        )

    show_data = st.session_state.cur_tar[st.session_state.local_index]
    keys = show_data.keys()
    sorted_keys = sorted(keys, key=lambda x: 0 if x[1:] in IMAGE_EXT + VIDEO_EXT else 1)
    for key in sorted_keys:
        if key[1:] in IMAGE_EXT:
            img = show_data[key]
            if not isinstance(img, list):
                img = [img]

            for i in range(0, len(img), 3):
                group = img[i : i + 3]
                cols = st.columns(len(group))
                for _img, col in zip(group, cols):
                    with col:
                        st.info(f"Image size (WxH): {_img.size}")
                        st.image(resize_image_h(_img, dataset_img_size))

        elif key[1:] in VIDEO_EXT:
            st.video(show_data[key])

        elif key[1:] == "json":
            st.info(f"Key: {key}")
            st.write({k: v for k, v in show_data[key].items() if "movq" not in k})

        else:
            st.info(f"Key: {key}")
            st.write(show_data[key])
