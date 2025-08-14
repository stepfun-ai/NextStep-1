#!/usr/bin/env python3
import copy
import importlib.util
import os
import re
import shlex
import sys
import uuid
from typing import Literal

import debugpy
import psutil
import torch
import torch.cuda.nccl
import torch.version
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.run import parse_args, run

from nextstep.lazy_config.utils import wait_for_remote_attach
from nextstep.utils.loguru import logger

module_spec = importlib.util.find_spec("nextstep")
init_file = module_spec.origin



CONFIG_FILE = ".config"
CONFIG: dict = {}


def setup_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    CONFIG[key.strip()] = value.strip()
    else:
        # If file does not exist, automatically create and write default content
        default_config = {
            "WANDB_MODE": "offline",
            "WANDB_BASE_URL": "https://api.wandb.ai",
            "WANDB_API_KEY": "",
        }
        CONFIG.update(default_config)
        with open(CONFIG_FILE, "w") as f:
            for key, value in default_config.items():
                f.write(f"{key}={value}\n")


def quote_string(s):
    if "'" in s and '"' in s:  # Both " and '
        # If the string contains both single and double quotes,
        # then use double quotes and escape the double quotes in the string
        return '"' + s.replace('"', '\\"') + '"'

    elif '"' in s:  # Only "
        # If the string only contains double quotes, use single quotes
        return "'" + s + "'"

    else:  # Only ' or no quotes
        # If the string does not contain double quotes, use double quotes
        return '"' + s + '"'


def check_port_usage(port):
    for conn in psutil.net_connections(kind="inet"):
        local_address, local_port = conn.laddr
        if local_port == port:
            return True
    return False


def find_available_port(start_port=5678, end_port=65535):
    for port in range(start_port, end_port + 1):
        if not check_port_usage(port):
            return port
    raise ValueError(f"No available port found between {start_port} and {end_port}")


def get_cmd_list(auto_multigpu=True, return_dict=False):
    cmd = copy.deepcopy(sys.argv)

    default_port = find_available_port()
    
    # Standard torchrun: single node or multi-node based on MASTER_ADDR
    if os.environ.get("MASTER_ADDR") is None:
        # Single node: use all available GPUs
        start_cmd = ["torchrun"]
        dist_args = f"--nproc_per_node {torch.cuda.device_count()} --master_port {default_port}".split()
    else:
        # Multi-node: use standard torchrun environment variables
        start_cmd = ["torchrun"]
        nproc_per_node = os.environ.get("PROC_PER_NODE", str(torch.cuda.device_count()))
        master_addr = os.environ.get("MASTER_ADDR")
        master_port = os.environ.get("MASTER_PORT", str(default_port))
        nnodes = os.environ.get("NODE_COUNT", "1")
        node_rank = os.environ.get("NODE_RANK", "0")
        dist_args = f"--nproc_per_node {nproc_per_node} --master_addr {master_addr} --master_port {master_port} --nnodes {nnodes} --node_rank {node_rank}".split()

    if auto_multigpu:
        cmd = start_cmd + dist_args + cmd[1:]
    else:
        # For single GPU, explicitly set nproc_per_node to 1
        cmd = ["torchrun", "--nproc_per_node", "1"] + cmd[1:]

    last_args_index = -1
    for index, _cmd in enumerate(cmd):
        if _cmd.startswith("--"):
            last_args_index = index
    for i in range(last_args_index + 2, len(cmd)):
        cmd[i] = quote_string(cmd[i])

    if return_dict:
        _dict = {"launch": cmd[0]}
        if auto_multigpu:
            _dict["dist_args"] = cmd[1 : 1 + len(dist_args)]
            _dict["others"] = cmd[1 + len(dist_args) :]
        else:
            _dict["others"] = cmd[1:]
        return _dict
    else:
        return cmd


def setup_environ() -> dict:
    env_vars = {
        "NCCL_TIMEOUT": "7200",
        "NCCL_DEBUG": "WARN",
        "NCCL_PXN_DISABLE": "1",
        # "NCCL_ALGO": "^Ring",  # Commented out: Ring algorithm is required for int8 dtype broadcast
        "NCCL_NET_OVERHEAD": "1000000",
        "NCCL_SHM_DISABLE": "1",
        "WANDB_MODE": CONFIG.get("WANDB_MODE", "offline"),
        "WANDB_BASE_URL": CONFIG.get("WANDB_BASE_URL", ""),
        "WANDB_API_KEY": CONFIG["WANDB_API_KEY"],
        "TOKENIZERS_PARALLELISM": "false",
        "ENABLE_TORCH_COMPILE": "true",
    }

    env_vars["TORCH_HOME"] = os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache/torch"))

    python_path = copy.deepcopy(sys.executable)
    if "conda" in python_path:
        while "envs" in python_path:
            python_path = os.path.dirname(python_path)
        conda_root = os.path.dirname(python_path)
        env_vars["TRITON_CACHE_DIR"] = os.path.join(conda_root, "triton_cache")

    # set environment variables
    for key, value in env_vars.items():
        os.environ[key] = value
        logger.info(f"Set default environment variable: {key}={value}")

    return env_vars


def expandvars(s):
    def replace(match):
        var_name = match.group(1)
        default_value = match.group(2)
        return os.environ.get(var_name, default_value)

    # handle ${VAR:-default}
    pattern = r"\$\{(\w+)(?::-([^}]+))?\}"
    result = re.sub(pattern, replace, s)

    # handle $VAR
    result = re.sub(r"\$(\w+)", lambda m: os.environ.get(m.group(1), ""), result)

    return result


def format_dict(d, indent=4, depth=0):
    formatted_output = ""
    spacing = " " * indent

    for key, value in d.items():
        if isinstance(value, dict):
            formatted_output += f"{spacing * depth}{key}:\n"
            formatted_output += format_dict(value, indent, depth + 1)
        else:
            if isinstance(value, str):
                if "\n" in value:
                    value = value.replace("\n", "\n" + (spacing * depth + " " * (len(key) + 2)))
            formatted_output += f"{spacing * depth}{key}: {value}\n"

    return formatted_output


@logger.catch
def smartrun(args=None, debug_mode: Literal["singlegpu", "multigpu"] | None = None):
    setup_config()
    
    job_id = "sj-" + str(uuid.uuid4())

    auto_multigpu = True
    if debug_mode is not None:
        auto_multigpu = debug_mode != "singlegpu"

    # auto get distributed args
    cmd = get_cmd_list(auto_multigpu=auto_multigpu)

    # if debug_mode is not None, we will add --debug to cmd
    if debug_mode is not None:
        for i, arg in enumerate(cmd):
            if arg.endswith(".py"):
                cmd = cmd[: i + 1] + ["--debug"] + cmd[i + 1 :]
                break

            if arg.startswith("-m"):
                cmd = cmd[: i + 2] + ["--debug"] + cmd[i + 2 :]
                break

    # expand environment variables
    cmd = " ".join(cmd)
    cmd = expandvars(cmd)
    logger.info(f"Modified Command:\n\033[32m{cmd}\033[0m")

    # set git safe directory
    os.system(f"git config --global --add safe.directory {os.getcwd()}")

    # output cuda and nccl version
    logger.info(f"CUDA Version: {torch.version.cuda}")
    logger.info(f"NCCL Version: {torch.cuda.nccl.version()}")

    # imitate the operating system to parse command line parameters
    sys.argv = shlex.split(cmd)

    if debug_mode is not None:
        debugpy.configure(subProcess=True)
        wait_for_remote_attach()

    args = parse_args(args)
    run_func = record(run)
    try:
        logger.info(f"Job {job_id} has started!")
        run_func(args)
    finally:
        logger.info(f"Job {job_id} has finished!")
def singlegpu_debug():
    smartrun(debug_mode="singlegpu")


def multigpu_debug():
    smartrun(debug_mode="multigpu")

# set config
setup_config()
# set environment variables
_ = setup_environ()
