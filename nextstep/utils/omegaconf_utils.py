from typing import Any

from omegaconf import DictConfig, OmegaConf


def omageconf_safe_update(cfg: DictConfig, key: str, value: Any, merge: bool = True):
    parts = key.split(".")
    for idx in range(1, len(parts)):
        prefix = ".".join(parts[:idx])
        v = OmegaConf.select(cfg, prefix, default=None)
        if v is None:
            break
        if not OmegaConf.is_config(v):
            raise KeyError(f"Trying to update key {key}, but {prefix} is not a config, but has type {type(v)}.")
    OmegaConf.update(cfg, key, value, merge=merge)


def omageconf_safe_update_dict(cfg: DictConfig, _dict: dict, merge: bool = True):
    for k, v in _dict.items():
        omageconf_safe_update(cfg, k, v, merge=merge)
