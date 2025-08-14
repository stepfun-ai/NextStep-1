import os
import re
import time

import torch.utils.data

from nextstep.datasets.data_logger import data_logger as logger


def warn_and_continue(exn):
    """Call in an exception handler to ignore any exception, isssue a warning, and continue."""
    logger.warning(repr(exn))
    time.sleep(0.5)
    return True


def find_key(_dict: dict, ext_keys: str | list[str] | tuple[str]) -> str | None:
    if isinstance(ext_keys, str):
        ext_keys = [ext_keys]
    ext_keys = [key.lower() for key in ext_keys] + [key.upper() for key in ext_keys]
    ext_keys = [key[1:] if key.startswith(".") else key for key in ext_keys]

    for key in ext_keys:
        if key in _dict or f".{key}" in _dict:
            return key
    return None




def is_match(pattern, string):
    """Check if the string (from the dataset) matches the pattern (from the config)."""
    if pattern == string:
        return True
    elif string.startswith(f"{pattern}/"):
        return True
    elif any(c in pattern for c in ".*+?^$[](){}|\\"):
        try:
            return bool(re.match(pattern, string))
        except re.error:
            return False
    return False


def format_list(lst: list, prefix: str = "", indent: int = 4) -> str:
    if len(lst) == 0:
        return "[]"
    if len(lst) == 1:
        return f"[{lst[0]}]"

    string = "["
    for item in lst:
        string += f"\n{prefix}" + " " * indent + str(item)
    string += f"\n{prefix}" + "]"
    return string


def get_num_workers() -> int:
    num_workers = 1
    if "NUM_WORKERS" in os.environ:
        num_workers = int(os.environ["NUM_WORKERS"])
    else:
        try:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                num_workers = worker_info.num_workers
        except ModuleNotFoundError:
            pass
    return num_workers


def get_worker_id() -> int:
    worker_id = 0
    if "WORKER" in os.environ:
        print(f"os.environ['WORKER']: {os.environ['WORKER']}")
        worker_id = int(os.environ["WORKER"])
    else:
        try:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                worker_id = worker_info.id
        except ModuleNotFoundError:
            pass
    return worker_id


def make_seed(*args):
    seed = 0
    # Flatten any list or tuple arguments to handle them individually
    flattened_args = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            flattened_args.extend(arg)
        else:
            flattened_args.append(arg)
    for arg in flattened_args:
        if isinstance(arg, str):
            h = 0
            for c in arg:
                h = (h * 31 + ord(c)) & 0x7FFFFFFF
            seed = (seed * 31 + h) & 0x7FFFFFFF
        elif isinstance(arg, int):
            seed = (seed * 31 + hash(arg)) & 0x7FFFFFFF
        else:
            logger.warning_once(f"Unsupported argument type: {type(arg)}, pass this argument: {arg}")
    return seed
