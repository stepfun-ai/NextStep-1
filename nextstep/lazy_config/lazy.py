# Copyright (c) Facebook, Inc. and its affiliates.

import ast
import builtins
import collections.abc as abc
import importlib
import os
import uuid
from contextlib import contextmanager

import black
from hydra.core.override_parser.overrides_parser import OverridesParser
from omegaconf import DictConfig, ListConfig, OmegaConf

from nextstep.lazy_config.registry import _convert_target_to_string, locate
from nextstep.utils.omegaconf_utils import omageconf_safe_update


class LazyCall:
    """
    Wrap a callable so that when it's called, the call will not be executed,
    but returns a dict that describes the call.

    LazyCall object has to be called with only keyword arguments. Positional
    arguments are not yet supported.

    Examples:
    ::
        from detectron2.config import instantiate, LazyCall

        layer_cfg = LazyCall(nn.Conv2d)(in_channels=32, out_channels=32)
        layer_cfg.out_channels = 64   # can edit it afterwards
        layer = instantiate(layer_cfg)
    """

    def __init__(self, target):
        if not (callable(target) or isinstance(target, (str, abc.Mapping))):
            raise TypeError(f"target of LazyCall must be a callable or defines a callable! Got {target}")
        self._target = target

    def __call__(self, **kwargs):
        # if is_dataclass(self._target):
        #     # omegaconf object cannot hold dataclass type
        #     # https://github.com/omry/omegaconf/issues/784
        #     target = _convert_target_to_string(self._target)
        # else:
        #     target = self._target

        # HACK: we always convert Clsss (abc.ABCMeta) to str
        if isinstance(self._target, str):
            try:
                locate(self._target)
                kwargs["_target_"] = self._target
            except:
                raise ValueError(f"Cannot locate target {self._target}")
        elif isinstance(self._target, type):
            target = _convert_target_to_string(self._target)
            kwargs["_target_"] = target

        return DictConfig(content=kwargs, flags={"allow_objects": True})


def _visit_dict_config(cfg, func):
    """
    Apply func recursively to all DictConfig in cfg.
    """
    if isinstance(cfg, DictConfig):
        func(cfg)
        for v in cfg.values():
            _visit_dict_config(v, func)
    elif isinstance(cfg, ListConfig):
        for v in cfg:
            _visit_dict_config(v, func)


def _validate_py_syntax(filename):
    # see also https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/config.py
    with open(filename, "r") as f:
        content = f.read()
    try:
        ast.parse(content)
    except SyntaxError as e:
        raise SyntaxError(f"Config file {filename} has syntax error!") from e


def _cast_to_config(obj):
    # if given a dict, return DictConfig instead
    if isinstance(obj, dict):
        return DictConfig(obj, flags={"allow_objects": True})
    return obj


_CFG_PACKAGE_NAME = "nextstep._cfg_loader"
"""
A namespace to put all imported config into.
"""


def _random_package_name(filename):
    # generate a random package name when loading config files
    return _CFG_PACKAGE_NAME + "_" + str(uuid.uuid4())[:4] + "." + os.path.basename(filename)


@contextmanager
def _patch_import():
    """
    Enhance relative import statements in config files, so that they:
    1. locate files purely based on relative location, regardless of packages.
        e.g. you can import file without having __init__
    2. do not cache modules globally; modifications of module states has no side effect
    3. (deperacated) support other storage system through PathManager, so config files can be in the cloud
    4. imported dict are turned into omegaconf.DictConfig automatically
    """
    old_import = builtins.__import__

    def find_relative_file(original_file, relative_import_path, level):
        # "from . import x" is not handled because it's unclear
        # if such import should produce `x` as a python module or DictConfig.
        # This can be discussed further if needed.
        relative_import_err = "Relative import of directories is not allowed within config files. Within a config file, relative import can only import other config files."
        if not len(relative_import_path):
            raise ImportError(relative_import_err)

        cur_file = os.path.dirname(original_file)
        for _ in range(level - 1):
            cur_file = os.path.dirname(cur_file)
        cur_name = relative_import_path.lstrip(".")
        for part in cur_name.split("."):
            cur_file = os.path.join(cur_file, part)
        if not cur_file.endswith(".py"):
            cur_file += ".py"
        if not os.path.isfile(cur_file):
            cur_file_no_suffix = cur_file[: -len(".py")]
            if os.path.isdir(cur_file_no_suffix):
                raise ImportError(f"Cannot import from {cur_file_no_suffix}." + relative_import_err)
            else:
                raise ImportError(
                    f"Cannot import name {relative_import_path} from " f"{original_file}: {cur_file} does not exist."
                )
        return cur_file

    def new_import(name, globals=None, locals=None, fromlist=(), level=0):
        # Only deal with relative imports inside config files
        if level != 0 and globals is not None and (globals.get("__package__", "") or "").startswith(_CFG_PACKAGE_NAME):
            cur_file = find_relative_file(globals["__file__"], name, level)
            _validate_py_syntax(cur_file)
            spec = importlib.machinery.ModuleSpec(_random_package_name(cur_file), None, origin=cur_file)
            module = importlib.util.module_from_spec(spec)
            module.__file__ = cur_file
            with open(cur_file) as f:
                content = f.read()
            exec(compile(content, cur_file, "exec"), module.__dict__)
            # turn imported dict into DictConfig automatically
            for name in fromlist:
                val = _cast_to_config(module.__dict__[name])
                module.__dict__[name] = val
            return module
        return old_import(name, globals, locals, fromlist=fromlist, level=level)

    builtins.__import__ = new_import
    yield new_import
    builtins.__import__ = old_import


CONFIG_KEY = "config"


class LazyConfig:
    """
    Provide methods to save, load, and overrides an omegaconf config object which may contain definition of lazily-constructed objects.
    """

    @staticmethod
    def load(filename: str, key: str = CONFIG_KEY) -> DictConfig:
        """
        Load a config file.

        Args:
            filename: absolute path or relative path w.r.t. the current working directory
            keys: keys to load and return.
        """
        filename = filename.replace("/./", "/")  # redundant

        if os.path.splitext(filename)[1] != ".py":
            raise ValueError(f"Config file {filename} has to be a python file.")
        _validate_py_syntax(filename)

        with _patch_import():
            # Record the filename
            module_namespace = {"__file__": filename, "__package__": _random_package_name(filename)}
            with open(filename) as f:
                content = f.read()
            # Compile first with `filename` to:
            # 1. make `filename` appears in stacktrace
            # 2. make `load_rel` able to find its parent's (possibly remote) location
            exec(compile(content, filename, "exec"), module_namespace)

        if key not in module_namespace.keys():
            raise KeyError(f"Config file `{filename}` must define a dictionary named `{key}`")
        assert isinstance(
            module_namespace[key], (dict, DictConfig)
        ), f"`{key}` in `{filename}` must be a `dict` or `DictConfig`, but got `{type(module_namespace[key])}`."

        return DictConfig(module_namespace[key], flags={"allow_objects": True})

    @staticmethod
    def save(cfg, filename: str, suffix: str = ""):
        """
        Save a config object to a py file.

        Args:
            cfg: an omegaconf config object
            filename: yaml file name to save the config file
        """
        with open(filename, "w") as f:
            f.write(LazyConfig.to_py(cfg))
            if suffix:
                f.write("\n")  # Add blank lines for better readability
                f.write(suffix)

    @staticmethod
    def apply_overrides(cfg: DictConfig, overrides: list[str]) -> DictConfig:
        """
        In-place override contents of cfg.

        Args:
            cfg: an omegaconf config object
            overrides: list of strings in the format of "a=b" to override configs.
                See https://hydra.cc/docs/advanced/override_grammar/basic/ for syntax.

        Returns:
            the cfg object
        """
        parser = OverridesParser.create()
        overrides = parser.parse_overrides(overrides)
        for o in overrides:
            key = o.key_or_group
            value = o.value()
            if o.is_delete():
                # TODO support this
                raise NotImplementedError("deletion is not yet a supported override")
            omageconf_safe_update(cfg, key, value)
        return cfg

    @staticmethod
    def to_py(cfg, prefix: str = f"{CONFIG_KEY}."):
        """
        Try to convert a config object into Python-like psuedo code.

        Note that perfect conversion is not always possible.
        So the returned results are mainly meant to be human-readable, and not meant to be executed.

        Args:
            cfg: an omegaconf config object
            prefix: root name for the resulting code (default: f"{CONFIG_KEY}.")

        Returns:
            str of formatted Python code
        """

        cfg = OmegaConf.to_container(cfg, resolve=True)

        def _to_str(obj, prefix=None, inside_call=False):
            if prefix is None:
                prefix = []
            if isinstance(obj, abc.Mapping) and "_target_" in obj:
                # Dict representing a function call
                target = _convert_target_to_string(obj.pop("_target_"))
                args = []
                for k, v in sorted(obj.items()):
                    args.append(f"{k}={_to_str(v, inside_call=True)}")
                args = ", ".join(args)
                call = f"{target}({args})"
                return "".join(prefix) + call
            elif isinstance(obj, abc.Mapping) and not inside_call:
                # Dict that is not inside a call is a list of top-level config objects that we
                # render as one object per line with dot separated prefixes
                key_list = []
                for k, v in sorted(obj.items()):
                    if isinstance(v, abc.Mapping) and "_target_" not in v:
                        if len(v) == 0:
                            key_list.append(f"{''.join(prefix)}{k}=dict()")
                        else:
                            key_list.append(_to_str(v, prefix=prefix + [k + "."]))
                    else:
                        key = "".join(prefix) + k
                        key_list.append(f"{key}={_to_str(v)}")
                return "\n".join(key_list)
            elif isinstance(obj, abc.Mapping):
                # Dict that is inside a call is rendered as a regular dict
                return "{" + ",".join(f"{repr(k)}: {_to_str(v, inside_call=inside_call)}" for k, v in sorted(obj.items())) + "}"
            elif isinstance(obj, list):
                return "[" + ",".join(_to_str(x, inside_call=True) for x in obj) + "]"
            elif isinstance(obj, type):
                return repr(_convert_target_to_string(obj))
            else:
                return repr(obj)

        def _convert_nested(py_str, prefix):
            # Initialize an empty dictionary to hold the parsed configuration
            config = {}

            # Split the input string into lines and process each line
            for line in py_str.strip().split("\n"):
                # Remove the prefix (e.g. "config.") and strip whitespace
                line = line.replace(prefix, "", 1).strip()
                if line.endswith("()"):  # Skip lines that create objects
                    continue

                # Split the line into key and value parts
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                # Use a recursive function to set the value in the dictionary
                def set_value(d, keys, value):
                    if "." in keys:
                        key, rest = keys.split(".", 1)
                        if key not in d:
                            d[key] = {}
                        set_value(d[key], rest, value)
                    else:
                        try:
                            d[keys] = eval(value)
                        except SyntaxError:
                            # Handle special cases like <StateDictType.FULL_STATE_DICT: 1>
                            if value.startswith("<") and value.endswith(">"):  # Keep it as a string
                                d[keys] = value
                            else:  # If it's not a special case, raise the exception
                                raise

                set_value(config, key, value)

            # Function to convert the dictionary to the desired string format
            def dict_to_str(d, indent=0):
                if not isinstance(d, dict):
                    return repr(d)
                indent_str = " " * indent
                inner_indent_str = " " * (indent + 4)
                lines = ["dict("]
                for key, value in d.items():
                    if isinstance(value, dict):
                        value_str = dict_to_str(value, indent=indent + 4)
                    else:
                        value_str = repr(value)
                    lines.append(f"{inner_indent_str}{key}={value_str},")
                lines.append(indent_str + ")")
                return "\n".join(lines)

            # Convert the parsed configuration dictionary to the desired string format
            output_str = ""
            for key, value in config.items():
                output_str += f"config.{key} = {dict_to_str(value, indent=0)}\n"
            return output_str

        py_prefix = 'from omegaconf import OmegaConf\nconfig = OmegaConf.create(flags={"allow_objects": True})\n'
        py_str = _to_str(cfg, prefix=[prefix])
        py_str = _convert_nested(py_str, prefix)
        py_str = py_prefix + py_str
        try:
            return black.format_str(py_str, mode=black.Mode(line_length=128))
        except black.InvalidInput:
            return py_str
