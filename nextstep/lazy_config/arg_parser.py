import argparse
import copy
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, NewType

import dacite
import pygments
import yaml
from colorama import Fore, Style, init
from omegaconf import DictConfig, OmegaConf
from pygments.formatters import Terminal256Formatter
from pygments.lexers import Python3Lexer
from torch.distributed.elastic.multiprocessing.errors import record

from nextstep.datasets.data_logger import data_logger
from nextstep.lazy_config.lazy import CONFIG_KEY, LazyConfig
from nextstep.lazy_config.utils import auto_profiler, copy_codebase, export_requirements
from nextstep.utils.comm import TemporaryProcessGroup, broadcast_object, get_rank, get_world_size, is_main_process
from nextstep.utils.loguru import logger, setup_logger
from nextstep.utils.omegaconf_utils import omageconf_safe_update
from nextstep.utils.timer import timestamp_str
from nextstep.utils.training_utils import set_seed

init(autoreset=True)


# fmt: off
@dataclass
class LazyArguments:
    config_file: str = field(default="", metadata={"help": "Path to config file."})
    run_dir: str = field(default="", metadata={"help": "Path to save logs, config and profiler."})


def default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="", metavar="FILE", help="Path to config file.")
    parser.add_argument("--logger_rank", default="all", choices=["main", "all"], help="Choose from `main` or `all`.")
    parser.add_argument("--use_profiler", action="store_true", help="Use pyinstrument to profile.")
    parser.add_argument("--use_viz_tracer", action="store_true", help="Use viztracer to profile.")
    parser.add_argument("--profile_time_out", type=int, default=None, help="Time out, useful when optimizing performance.")
    parser.add_argument("--copy_codebase", action='store_true', default=True)
    parser.add_argument("--skip_copy_codebase", action='store_false', dest='copy_codebase', help="Skip copying codebase to run_dir.")
    parser.add_argument("--export_requirements", action='store_true',  default=True)
    parser.add_argument("--skip_export_requirements", action='store_false', dest='export_requirements', help="Skip exporting requirements to requirements.txt.")
    parser.add_argument("--debug", action="store_true", help="It will be automatically set by smart debug, or you can manually set.")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="The remaining parameters will override the configuration file.")
    return parser
# fmt: on


def _try_get_key(cfg, *keys, default=None):
    """
    Try select keys from cfg until the first key that exists. Otherwise return default.
    """
    for k in keys:
        none = object()
        p = OmegaConf.select(cfg, k, default=none)
        if p is not none:
            return p
    return default


def _try_set_key(cfg, *keys, value=None):
    """
    Try set keys from cfg until the first key that exists.
    """
    for k in keys:
        none = object()
        p = OmegaConf.select(cfg, k, default=none)
        if p is not none:
            omageconf_safe_update(cfg, k, value)


def _highlight(code):
    lexer = Python3Lexer()
    code = pygments.highlight(code, lexer, Terminal256Formatter(style="monokai"))
    return code


def compare_dicts(d1, d2, parent_key="", diff_keys=None):
    if diff_keys is None:
        diff_keys = []

    for key in d1:
        if key not in d2:
            diff_keys.append(f"{parent_key}.{key}" if parent_key else key)
        else:
            if isinstance(d1[key], (dict, DictConfig)) and isinstance(d2[key], (dict, DictConfig)):
                compare_dicts(d1[key], d2[key], f"{parent_key}.{key}" if parent_key else key, diff_keys)
            elif d1[key] != d2[key]:
                diff_keys.append(f"{parent_key}.{key}" if parent_key else key)
    return diff_keys


def add_color(yaml_str, keys_to_color):
    for key in keys_to_color:
        key = key.split(".")[-1]
        pattern = re.compile(rf"(\b{key}\b:(?:\s+[^\n]+(?:\n\s*-\s+[^\n]+)*|\s*(?:\n\s*-\s+[^\n]+)+))", re.MULTILINE)
        yaml_str = pattern.sub(Fore.RED + r"\1" + Style.RESET_ALL, yaml_str)
    return yaml_str


def default_setup(config: DictConfig, args, config_before_override: DictConfig):
    if args.debug:
        _try_set_key(config, "REPORT_TO", "report_to", "train.report_to", "training.report_to", value=[])

    current_time = timestamp_str()
    with TemporaryProcessGroup(backend="gloo"):
        current_time = broadcast_object(current_time)
    if args.debug:
        current_time = f"debug_{current_time}"

    output_dir = _try_get_key(config, "OUTPUT_DIR", "output_dir", "train.output_dir", "training.output_dir")
    if output_dir is None:
        raise ValueError(
            "OUTPUT_DIR/output_dir/train.output_dir/training.output_dir not found in arguments. Please specify an output directory."
        )

    run_dir = os.path.join(output_dir, current_time)

    eval_only = _try_get_key(config, "EVAL_ONLY", "eval_only", "train.eval_only", "training.eval_only", default=False)
    if eval_only:
        run_dir = f"{run_dir}_eval_only"

    _try_set_key(config, "OUTPUT_DIR", "output_dir", "train.output_dir", "training.output_dir", value=run_dir)
    config.run_dir = run_dir

    rank = get_rank()

    # setup logger and warnings, only first process in distributed training will print info
    setup_logger(
        _logger=logger,
        save_dir=os.path.join(run_dir, "training_logs"),
        filename="training.log",
        distributed_rank=rank,
        logger_rank=args.logger_rank,
        enable_redirect_sys_output=not args.debug,
        enable_redirect_logging=True,
    )

    setup_logger(
        _logger=data_logger,
        save_dir=os.path.join(run_dir, "data_logs"),
        filename="data.log",
        distributed_rank=rank,
        logger_rank=args.logger_rank,
        enable_redirect_sys_output=False,
        enable_redirect_logging=False,
    )

    if is_main_process():
        os.makedirs(run_dir, exist_ok=True)
        if args.copy_codebase:
            copy_codebase(run_dir)
        if args.export_requirements:
            export_requirements(run_dir)

    if not is_main_process():
        warnings.filterwarnings("ignore")

    logger.info(f"World size: {get_world_size()}")

    # command line arguments and content of `args.config_file`
    logger.info("Command line arguments:\n" + yaml.dump(vars(args), indent=4))
    logger.info(f"Config file: {args.config_file}")
    keys = compare_dicts(config_before_override, config)
    config_yaml_str = add_color(OmegaConf.to_yaml(config), keys)
    logger.info("Full config:\n" + config_yaml_str)

    # make sure each worker has a different, yet deterministic seed if specified
    seed = _try_get_key(config, "SEED", "seed", "train.seed", "training.seed", default=42)
    set_seed(seed, rank=rank)

    config.config_file = args.config_file

    def _extract_main_block(file_path):
        """Extract the 'if __name__ == "__main__":' block and all content after it from a Python file."""
        try:
            with open(file_path, "r") as f:
                content = f.read()
                # Find the main block and everything after it
                main_pattern = re.compile(r'if\s+__name__\s*==\s*["\']__main__["\']\s*:.*$', re.MULTILINE | re.DOTALL)
                match = main_pattern.search(content)
                if match:
                    return "\n" + match.group(0)
        except Exception as e:
            pass
        return ""

    if is_main_process() and run_dir:
        path = os.path.join(run_dir, f"{CONFIG_KEY}.py")
        LazyConfig.save(config, path, suffix=_extract_main_block(args.config_file))
        logger.info("Full config saved to {}".format(path))


DataClass = NewType("DataClass", Any)
DataClassType = NewType("DataClassType", Any)


def LazyArgumentParser(dataclass_type: DataClassType, args: argparse.Namespace) -> DataClass:
    """
    Initialize an instance from the `CONFIG_KEY` dict in the config file, and accept modifications from the command line.

    Args:
        dataclass_type (DataClassType): Pass a dataclass type, which contains all arguments you need, including nested dataclass.

    Returns:
        DataClass: Return an instance of dataclass from config file.

    Example:
    ```
    torchrun --nproc-per-node=8 train.py --config_file path_to_config.py \
    "mix_precision=True" \
    "model.max_length=77"
    ```
    """
    if args.config_file == "":
        args.config_file = sys.argv[0]

    # CONFIG_KEY is an Omegaconf variable name in file
    config: DictConfig = LazyConfig.load(args.config_file, CONFIG_KEY)
    config_before_override = copy.deepcopy(config)
    config: DictConfig = LazyConfig.apply_overrides(config, args.opts)
    default_setup(config, args, config_before_override)

    config: dict = OmegaConf.to_container(config, resolve=True)

    dacite_config = dacite.Config(check_types=True, strict=True)
    dataclass_args = dacite.from_dict(data_class=dataclass_type, data=config, config=dacite_config)
    return dataclass_args


@logger.catch(reraise=True)
@record
def LazyLaunch(main_func: Callable, dataclass_type: DataClassType, *args, **kwargs):
    """`main_func` must accept argument `config` of type `dataclass_type`."""
    parsed_args = default_parser().parse_args()
    config = LazyArgumentParser(dataclass_type, parsed_args)

    main_func = auto_profiler(
        dir=config.run_dir,
        seconds=parsed_args.profile_time_out,
        use_profiler=parsed_args.use_profiler,
        use_viz_tracer=parsed_args.use_viz_tracer,
    )(main_func)

    main_func(config, *args, **kwargs)
