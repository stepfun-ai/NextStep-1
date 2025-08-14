import argparse
import copy
import json
import multiprocessing
import os
import traceback
from concurrent.futures import ThreadPoolExecutor

import joblib
import megfile
from tqdm.auto import tqdm

from nextstep.data.indexed_tar import IndexedTarSamples
from nextstep.data_zoos import TMP_DIR
from nextstep.utils.loguru import logger, setup_logger


def is_s3(path):
    return path.startswith("s3://")


def tar2json_path(tar_file):
    json_file = TMP_DIR + tar_file.replace(".tar", ".json")
    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    return json_file


def process_single_tar_file(file):
    try:
        indexed_tar = IndexedTarSamples(file, backend="memory" if is_s3(file) else "mmap")
        num_samples = indexed_tar.num_samples
        size = indexed_tar.size
        checksum = indexed_tar.checksum
        info_file = tar2json_path(file)
        with open(info_file, "w") as f:
            json.dump({"num_samples": num_samples, "size": size, "checksum": checksum}, f)
        return file, 1
    except ValueError as ve:
        # Handle the case of empty file mmap error
        if "cannot mmap an empty file" in str(ve):
            return file, 2
        else:
            raise
    except Exception as e:
        print(f"Error in processing file {file}: {e}")
        print(f"Traceback:\n{''.join(traceback.format_exc())}")
        return file, 0


def process_tar_files(path, n_jobs=1):
    files_path = TMP_DIR + os.path.join(path, "files.json")

    if not os.path.exists(files_path):
        try:
            files = megfile.smart_glob(f"{path}/**/*.tar", recursive=True)
            with open(files_path, "w") as f:
                json.dump(files, f)
        except Exception as e:
            logger.error(f"Error in generating files.json: {e}")
            return [], [], [], []
    else:
        try:
            with open(files_path, "r") as f:
                files = json.load(f)
        except Exception as e:
            logger.error(f"Error in reading files.json: {e}")
            return [], [], [], []

    files_to_process = copy.deepcopy(files)
    logger.info(f"Found total {len(files_to_process)} tar files")
    # Use ThreadPoolExecutor to check file existence in parallel
    with ThreadPoolExecutor() as executor:
        file_exists_results = list(
            executor.map(
                lambda file: (file, os.path.exists(tar2json_path(file))),
                tqdm(files_to_process, desc="Filtering tar files"),
            )
        )
    files_to_process = [file for file, exists in file_exists_results if not exists]
    logger.info(f"Remaining {len(files_to_process)} tar files to process")

    # reduce batch size, avoid memory problem
    batch_size = max(1, min(100, len(files_to_process) // n_jobs))

    results = []
    for i in range(0, len(files_to_process), batch_size):
        batch = files_to_process[i : i + batch_size]
        logger.info(
            f"Processing batch {i // batch_size + 1}/{(len(files_to_process) + batch_size - 1) // batch_size}, size: {len(batch)}"
        )

        sub_results = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(process_single_tar_file)(file)
            for file in tqdm(batch, desc=f"Processing batch {i // batch_size + 1}")
        )
        results.extend(sub_results)

    successful_files = [result[0] for result in results if result[1] == 1]
    empty_files = [result[0] for result in results if result[1] == 2]
    failed_files = [result[0] for result in results if result[1] == 0]

    files = list(set(files) - set(empty_files))
    files_to_process = list(set(files_to_process) - set(empty_files))

    if empty_files:
        logger.warning(f"Ignored {len(empty_files)} empty tar files")

    if failed_files:
        logger.warning(f"Failed to process {len(failed_files)} files")
        for failed_file in failed_files:
            logger.warning(f"Failed file: {failed_file}")

    return files, files_to_process, successful_files, failed_files

# python -m nextstep.data._gen_meta /path/to/dataset/root
def gen_meta():
    parser = argparse.ArgumentParser(description="Count samples in tar files")
    parser.add_argument("dir", type=str, help="tar file root directory")
    parser.add_argument("--meta_save_path", type=str, default=None, help="meta.json save path")
    parser.add_argument("--n_jobs", type=int, default=64, help="Number of parallel jobs. Default is 8")
    args = parser.parse_args()

    if args.n_jobs <= 0:
        args.n_jobs = max(1, multiprocessing.cpu_count() // 2)
        logger.info(f"Auto-detected {multiprocessing.cpu_count()} CPU cores, using {args.n_jobs} jobs")

    if args.dir[-1] == "/":
        args.dir = args.dir[:-1]

    if args.meta_save_path is None:
        args.meta_save_path = os.path.join(args.dir, "meta.json")

    setup_logger(
        _logger=logger,
        save_dir=TMP_DIR + args.dir,
        filename="counting.log",
        enable_redirect_sys_output=False,
        enable_redirect_logging=False,
    )

    logger.info(f"Generating meta info for {args.dir}")
    files, files_to_process, successful_files, failed_files = process_tar_files(args.dir, n_jobs=args.n_jobs)
    logger.info(f"Total files processed: {len(files_to_process)}")
    logger.info(f"Successful files: {len(successful_files)}")
    logger.info(f"Failed files: {len(failed_files)}")

    if len(failed_files) == 0:
        assert len(successful_files) == len(
            files_to_process
        ), f"successful files count {len(successful_files)} != total files count {len(files_to_process)}"

        json_files = [tar2json_path(file) for file in files]
        meta_info = {}
        loader = megfile.smart_open if is_s3(json_files[0]) else open

        def load_json_file(file_tuple):
            file, json_file, args_dir = file_tuple
            with loader(json_file, "r") as f:
                info = json.load(f)
            return os.path.relpath(file, args_dir), info

        file_tuples = [(file, json_file, args.dir) for file, json_file in zip(files, json_files)]
        with ThreadPoolExecutor(max_workers=args.n_jobs) as executor:
            results = list(
                tqdm(
                    executor.map(load_json_file, file_tuples),
                    total=len(file_tuples),
                    desc="Generating meta info",
                ),
            )

        meta_info = dict(results)
        with loader(args.meta_save_path, "w") as f:
            json.dump(meta_info, f, indent=4)
        logger.info(f"Meta info saved to {args.meta_save_path}")
        
        # Calculate total sample count in dataset
        total_samples = sum(info.get("num_samples", 0) for info in meta_info.values())
        logger.info(f"Total samples in dataset: {total_samples:,}")
        
        # Save total statistics to file
        stats_save_path = args.meta_save_path.replace(".json", "_stats.json")
        stats_info = {
            "total_samples": total_samples,
            "total_files": len(meta_info),
            "dataset_path": args.dir
        }
        with loader(stats_save_path, "w") as f:
            json.dump(stats_info, f, indent=4)
        logger.info(f"Dataset statistics saved to {stats_save_path}")
    else:
        logger.warning(f"Some files failed to process, please run again!")


if __name__ == "__main__":
    gen_meta()
