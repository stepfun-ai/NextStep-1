import argparse
import json
import re
import time
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from functools import wraps

import torch
import yaml
from transformers import PreTrainedTokenizerBase

from nextstep.utils.loguru import logger


def retry(total_tries=5, initial_wait=1, backoff_factor=2, max_wait=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            wait_time = initial_wait
            for i in range(total_tries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # If this was the last attempt
                    if i == total_tries - 1:
                        raise ValueError(f"Function failed after {total_tries} attempts") from e
                    logger.error(f"Function failed with error: `{e}`, retry in {wait_time} seconds...")
                    time.sleep(wait_time)
                    # Exponential backoff
                    wait_time *= backoff_factor
                    if max_wait is not None:
                        if wait_time > max_wait:
                            wait_time = max_wait

        return wrapper

    return decorator


class LargeInt(int):
    def __new__(cls, value):
        if isinstance(value, str):
            units = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
            last_char = value[-1].upper()
            if last_char in units:
                num = float(value[:-1]) * units[last_char]
                return super(LargeInt, cls).__new__(cls, int(num))
            else:
                return super(LargeInt, cls).__new__(cls, int(value))
        else:
            return super(LargeInt, cls).__new__(cls, value)

    def __str__(self):
        value = int(self)
        if abs(value) < 1000:
            return f"{value}"
        for unit in ["", "K", "M", "B", "T"]:
            if abs(value) < 1000:
                return f"{value:.1f}{unit}"
            value /= 1000
        return f"{value:.1f}P"  # P stands for Peta, or 10^15

    def __repr__(self):
        return f'"{self.__str__()}"'  # Ensure repr also returns the string with quotes

    def __json__(self):
        return f'"{self.__str__()}"'

    def __add__(self, other):
        if isinstance(other, int):
            return LargeInt(super().__add__(other))
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)  # This ensures commutativity


class nullcontext(AbstractContextManager, AbstractAsyncContextManager):
    """Context manager that does no additional processing.

    Used as a stand-in for a normal context manager, when a particular
    block of code is only sometimes used with a normal context manager:

    cm = optional_cm if condition else nullcontext()
    with cm:
        # Perform operation, using optional_cm if condition is True
    """

    def __init__(self, enter_result=None, *args, **kwargs):
        self.enter_result = enter_result

    def __enter__(self):
        return self.enter_result

    def __exit__(self, *excinfo):
        pass

    async def __aenter__(self):
        return self.enter_result

    async def __aexit__(self, *excinfo):
        pass


def find_matching_parenthesis(expression, opening_index):
    if expression[opening_index] != "(":
        raise ValueError("The character at the provided index is not '('.")

    stack = 0

    for index in range(opening_index + 1, len(expression)):
        char = expression[index]
        if char == "(":
            stack += 1
        elif char == ")":
            if stack == 0:
                return index
            stack -= 1

    raise ValueError("No matching ')' found for '(' at index {}.".format(opening_index))


def pretty_format(obj, indent: int = 4) -> str:
    if isinstance(obj, dict):
        return yaml.dump(obj, sort_keys=True, indent=indent)
        # return json.dumps(obj, sort_keys=True, indent=indent)
    elif isinstance(obj, PreTrainedTokenizerBase):
        repr_str = obj.__repr__()
        class_name, rest = repr_str.split("(", 1)
        idx = find_matching_parenthesis(f"({rest}", 0)
        other = rest[idx:]
        other = other.strip(",").strip(" ")
        rest = rest[:idx]
        rest = rest.rstrip(")")

        pairs = re.findall(r"(\w+)=({[^}]*}|[^,]*),?", rest)

        formatted_pairs = []
        for k, v in pairs:
            if v.startswith("{") and v.endswith("}"):
                try:
                    v_dict = json.loads(v.replace("'", '"'))
                    v_formatted = json.dumps(v_dict, indent=indent).replace("\n", "\n" + " " * indent)
                except json.JSONDecodeError:
                    v_formatted = v
            else:
                v_formatted = v

            formatted_pairs.append(f"{' ' * indent}{k}={v_formatted},")

        return f"{class_name}(\n" + "\n".join(formatted_pairs) + "\n),\n" + other.replace("\t", " " * indent)
    elif isinstance(obj, argparse.Namespace):
        args_dict = vars(obj)
        return yaml.dump(args_dict, sort_keys=True, indent=indent)
    else:
        return obj


def compare_state_dicts(dict1, dict2, rtol: float = 1e-5, atol: float = 1e-8):
    """
    Compare whether two PyTorch state dicts are completely identical

    Args:
        dict1: The first state dict
        dict2: The second state dict
        rtol: Relative Tolerance (default: 1e-5)
        atol: Absolute Tolerance (default: 1e-8)

    Returns:
        Tuple[bool, List[str]]:
            - Whether the boolean values are completely consistent
            - List of inconsistent parameter names
    """
    if dict1.keys() != dict2.keys():
        missing_in_1 = set(dict2.keys()) - set(dict1.keys())
        missing_in_2 = set(dict1.keys()) - set(dict2.keys())
        differences = []

        if missing_in_1:
            differences.append(f"Keys missing in dict1: {missing_in_1}")
        if missing_in_2:
            differences.append(f"Keys missing in dict2: {missing_in_2}")

        return False, differences

    differences = []

    for key in dict1.keys():
        param1 = dict1[key]
        param2 = dict2[key]

        if type(param1) != type(param2):
            differences.append(f"{key}: Type mismatch - {type(param1)} vs {type(param2)}")
            continue

        if param1.shape != param2.shape:
            differences.append(f"{key}: Shape mismatch - {param1.shape} vs {param2.shape}")
            continue

        if param1.device != param2.device:
            differences.append(f"{key}: Device mismatch - {param1.device} vs {param2.device}")
            continue

        if not torch.allclose(param1, param2, rtol=rtol, atol=atol):
            max_diff = torch.max(torch.abs(param1 - param2)).item()
            rel_diff = torch.max(torch.abs((param1 - param2) / (param2 + 1e-7))).item()
            differences.append(f"{key}: Value mismatch - Max absolute diff: {max_diff:.2e}, Max relative diff: {rel_diff:.2e}")

    return len(differences) == 0, differences
