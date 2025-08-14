import argparse
import json
import multiprocessing
import os
import shutil
import traceback
from concurrent.futures import ThreadPoolExecutor

import joblib
from tqdm.auto import tqdm

from nextstep.data.indexed_tar import IndexedTarSamples
from nextstep.data_zoos import DATA_META_PATH, get_cache_path
from nextstep.datasets.utils import is_match
from nextstep.utils.loguru import logger


def load_tar_file(file):
    indexed_tar = None
    try:
        indexed_tar = IndexedTarSamples(file, backend="mmap")
    except Exception as e:
        print(f"Error in processing file {file}: {e}")
        print(f"Traceback:\n{''.join(traceback.format_exc())}")
    finally:
        if indexed_tar is not None:
            del indexed_tar


def warmup_data():
    parser = argparse.ArgumentParser(description="Count samples in tar files")
    parser.add_argument("names", type=str, help="tar file root directory")
    parser.add_argument("--n_jobs", type=int, default=0, help="Number of parallel jobs. Default is 8")
    args = parser.parse_args()

    if args.n_jobs <= 0:
        args.n_jobs = max(1, multiprocessing.cpu_count() // 2)
        logger.info(f"Auto-detected {multiprocessing.cpu_count()} CPU cores, using {args.n_jobs} jobs")

    names: list[str] = args.names.split(",")

    with open(DATA_META_PATH, "r") as f:
        data_meta = json.load(f)

    matched_keys = []
    for name in names:
        matched_keys.extend([k for k in data_meta.keys() if is_match(name, k)])

    urls = []
    for k in matched_keys:
        with open(data_meta[k]["tar_meta_path"], "r") as f:
            tar_meta = json.load(f)
        urls.extend([os.path.join(data_meta[k]["dir"], rel_path) for rel_path in tar_meta.keys()])
    logger.info(f"Find {len(urls)} urls in {len(matched_keys)} datasets:\n[{', '.join(matched_keys)}]")

    with ThreadPoolExecutor() as executor:
        urls = list(tqdm(executor.map(get_cache_path, urls), desc="get cache path", total=len(urls)))

    def is_exists(url):
        if os.path.isdir(url):
            shutil.rmtree(url)
        return os.path.exists(url)

    with ThreadPoolExecutor() as executor:
        urls = list(
            tqdm(executor.map(lambda url: url if is_exists(url) else None, urls), desc="check file exists", total=len(urls))
        )
    urls = [url for url in urls if url is not None]
    logger.info(f"Find {len(urls)} cache paths")

    with ThreadPoolExecutor() as executor:
        urls = list(
            tqdm(
                executor.map(lambda url: url if not os.path.exists(url + ".index") else None, urls),
                desc="check index",
                total=len(urls),
            )
        )
    urls = [url for url in urls if url is not None]
    logger.info(f"Remaining {len(urls)} urls w/o index file")

    # reduce batch size, avoid memory problem
    batch_size = max(1, min(100, len(urls) // args.n_jobs))

    for i in range(0, len(urls), batch_size):
        batch = urls[i : i + batch_size]
        logger.info(f"Processing batch {i // batch_size + 1}/{(len(urls) + batch_size - 1) // batch_size}, size: {len(batch)}")

        joblib.Parallel(n_jobs=args.n_jobs)(
            joblib.delayed(load_tar_file)(file) for file in tqdm(batch, desc=f"Processing batch {i // batch_size + 1}")
        )

    logger.info(f"Successfully warmup {len(urls)} urls in {len(matched_keys)} datasets:\n[{', '.join(matched_keys)}]")


if __name__ == "__main__":
    warmup_data()
