import re
from typing import Callable

import torch
import yaml
from torch import nn
from torch.nn.utils.clip_grad import _no_grad

from nextstep.utils.loguru import logger
from nextstep.utils.misc import LargeInt

ALL_LAYERNORM_LAYERS = [nn.LayerNorm]


def _pretty_format(obj, indent: int = 4) -> str:
    if isinstance(obj, dict):
        return yaml.dump(obj, sort_keys=True, indent=indent)


def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


def get_decay_parameter_names(model) -> list[str]:
    """
    Get all parameter names that weight decay will be applied to

    Note that some models implement their own layernorm instead of calling nn.LayerNorm, weight decay could still
    apply to those modules since this function only filter out instance of nn.LayerNorm
    """
    logger.info(
        f"Getting decay parameter names for model {model.__class__.__name__}, except for forbidden layers: {ALL_LAYERNORM_LAYERS}"
    )
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    return decay_parameters


def vision_encoder_lr_scale_func(name: str):
    if "vision_encoder" in name:
        return 0.1  # a smaller learning rate for vision encoder
    return 1


def llm_lr_scale_func(name: str, depth: int = 32, decay: float = 0.931):
    if "model.layers" in name:
        in_pp_layer = int(re.findall(f"layers\.(\d+)\.", name)[0])
        scale = decay ** (depth - in_pp_layer - 1)
        return scale
    return 1


def get_grouped_parameters(
    model: nn.Module,
    lr: float,
    wd: float,
    lr_scale_func: Callable | None = None,
    wd_params: list[str] | None = None,
) -> list:
    """
    Create grouped parameters based on lr_scale_func and wd_params.
    lr_scale_func is a function that takes a parameter name and returns a scale factor for the learning rate.
    wd_params is a list of parameter names that should be regularized.
    """
    weight_decay_params: dict[str, list] = {}
    no_weight_decay_params: dict[str, list] = {}

    weight_decay_infos: dict[str, list] = {}
    no_weight_decay_infos: dict[str, list] = {}

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if lr_scale_func is not None:
            lr_scale = lr_scale_func(n)
        else:
            lr_scale = 1

        if wd_params is not None:
            is_wd = n in wd_params
        else:
            # Default behavior: do not regularize biases nor Norm parameters
            is_wd = not ("bias" in n or len(p.shape) == 1)

        info_key = f"LR = initial_lr * scale = {lr} * {lr_scale:.8f} = {lr * lr_scale:.8f}"
        if is_wd:
            weight_decay_params.setdefault(lr_scale, []).append(p)
            weight_decay_infos.setdefault(info_key, []).append(n)
        else:
            no_weight_decay_params.setdefault(lr_scale, []).append(p)
            no_weight_decay_infos.setdefault(info_key, []).append(n)

    grouped_parameters = []
    for lr_scale, params in weight_decay_params.items():
        grouped_parameters.append({"params": params, "lr": lr * lr_scale, "weight_decay": wd})
    for lr_scale, params in no_weight_decay_params.items():
        grouped_parameters.append({"params": params, "lr": lr * lr_scale, "weight_decay": 0.0})

    logger.info(f"->> Number of Optimizer Groups: {len(grouped_parameters)}")
    logger.info(f"Learning Rate: {lr}, Weight Decay: {wd}")
    logger.info(f"Weight Decay Parameters:\n{_pretty_format(weight_decay_infos)}")
    logger.info(f"No Weight Decay Parameters:\n{_pretty_format(no_weight_decay_infos)}")

    return grouped_parameters


def get_num_grouped_parameters(grouped_parameters: dict) -> LargeInt:
    n_params = 0
    for group in grouped_parameters:
        n_params += sum(p.numel() for p in group["params"])
    return LargeInt(n_params)


@_no_grad
def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.0)
    first_device = grads[0].device

    norms = [torch.linalg.vector_norm(g, norm_type) for g in grads]
    total_norm = torch.linalg.vector_norm(torch.stack([norm.to(first_device) for norm in norms]), norm_type)

    return total_norm
