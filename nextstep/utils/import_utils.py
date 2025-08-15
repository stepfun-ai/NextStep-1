import importlib.metadata
import importlib.util
import operator as op
import os

from packaging import version
from packaging.version import Version, parse

from nextstep.utils.loguru import logger

HOME_DIR = os.getenv("HOME", None)
ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})


# TODO: This doesn't work for all packages (`bs4`, `faiss`, etc.) Talk to Sylvain to see how to do with it better.
def _is_package_available(pkg_name: str, return_version: bool = False) -> tuple[bool, str] | bool:
    # Check we're not importing a "pkg_name" directory somewhere but the actual library by trying to grab the version
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            package_version = importlib.metadata.version(pkg_name)
            package_exists = True
        except importlib.metadata.PackageNotFoundError:
            package_exists = False
    if return_version:
        return package_exists, package_version
    else:
        return package_exists


STR_OPERATION_TO_FUNC = {">": op.gt, ">=": op.ge, "==": op.eq, "!=": op.ne, "<=": op.le, "<": op.lt}


# This function was copied from: https://github.com/huggingface/accelerate/blob/874c4967d94badd24f893064cc3bef45f57cadf7/src/accelerate/utils/versions.py#L319
def compare_versions(library_or_version: str | Version, operation: str, requirement_version: str):
    """
    Args:
    Compares a library version to some requirement using a given operation.
        library_or_version (`str` or `packaging.version.Version`):
            A library name or a version to check.
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`.
        requirement_version (`str`):
            The version to compare the library version against
    """
    if operation not in STR_OPERATION_TO_FUNC.keys():
        raise ValueError(f"`operation` must be one of {list(STR_OPERATION_TO_FUNC.keys())}, received {operation}")
    operation = STR_OPERATION_TO_FUNC[operation]
    if isinstance(library_or_version, str):
        library_or_version = parse(importlib.metadata.version(library_or_version))
    return operation(library_or_version, parse(requirement_version))


USE_TF = os.environ.get("USE_TF", "AUTO").upper()
USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()
_torch_available = False
if USE_TORCH in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TF not in ENV_VARS_TRUE_VALUES:
    _torch_available, _torch_version = _is_package_available("torch", return_version=True)
else:
    logger.debug("Disabling PyTorch because USE_TF is set")
    _torch_available = False

_flash_attn_2_available, _flash_attn_version = _is_package_available("flash_attn", return_version=True)
if _flash_attn_2_available:
    if not compare_versions(parse(_flash_attn_version), ">=", "2.0.0"):
        _flash_attn_2_available = False

_xformers_available, _xformers_version = _is_package_available("xformers", return_version=True)
_xformers_available = _xformers_available and _torch_available
# check if the version of torch is >= 1.12
if _xformers_available:
    if compare_versions(parse(_torch_version), "<", "1.12"):
        logger.debug("xformers is available but torch version is < 1.12.")
        logger.debug("Disabling xformers...")
        _xformers_available = False

    if compare_versions(parse(_xformers_version), "==", "0.0.16"):
        logger.debug(
            "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
        )
        logger.debug("Disabling xformers...")
else:
    logger.debug(f"Carefully keep trying to install the `xformers` version that corresponds to the `torch` version.")
    if _torch_available:
        import torch.version

        cuda_version = torch.version.cuda.replace(".", "")
    else:
        cuda_version = "{cuda_version}"
    logger.debug(
        f"You can install with `pip install -U xformers` or `pip install -U xformers --index-url https://download.pytorch.org/whl/cu{cuda_version}`."
    )
    logger.debug(f"More information can be found at https://github.com/facebookresearch/xformers.")

_accelerate_available, _accelerate_version = _is_package_available("accelerate", return_version=True)

_deepspeed_available, _deepspeed_version = _is_package_available("deepspeed", return_version=True)

_peft_available = _is_package_available("peft")

_apex_available = _is_package_available("apex")

# This is the version of torch required to run torch.fx features and torch.onnx with dictionary inputs.
TORCH_FX_REQUIRED_VERSION = version.parse("1.10")
_torch_fx_available = False
if _torch_available:
    torch_version = version.parse(_torch_version)
    _torch_fx_available = (torch_version.major, torch_version.minor) >= (
        TORCH_FX_REQUIRED_VERSION.major,
        TORCH_FX_REQUIRED_VERSION.minor,
    )

_wandb_available = _is_package_available("wandb")

_safetensors_available = _is_package_available("safetensors")

_transformers_available, _transformers_version = _is_package_available("transformers", return_version=True)

_safetensors_available = _is_package_available("safetensors")

_datasets_available = _is_package_available("datasets")

_tokenizers_available, _tokenizers_version = _is_package_available("tokenizers", return_version=True)


def is_torch_available(operation: str = None, version: str = None):
    """
    Args:
    Compares the current PyTorch version to a given reference with an operation.
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A string version of PyTorch
    """
    if version is not None and operation is not None:
        return _torch_available and compare_versions(parse(_torch_version), operation, version)
    return _torch_available


def is_torch_tf32_available():
    if not is_torch_available():
        return False

    import torch

    if not torch.cuda.is_available() or torch.version.cuda is None:
        return False
    if torch.cuda.get_device_properties(torch.cuda.current_device()).major < 8:
        return False
    if int(torch.version.cuda.split(".")[0]) < 11:
        return False
    if version.parse(version.parse(torch.__version__).base_version) < version.parse("1.7"):
        return False

    return True


def is_flash_attn_2_available(operation: str = None, version: str = None):
    if not is_torch_available():
        return False

    # Let's add an extra check to see if cuda is available
    import torch

    if version is not None and operation is not None:
        return (
            _flash_attn_2_available
            and torch.cuda.is_available()
            and compare_versions(parse(_flash_attn_version), operation, version)
        )
    return _flash_attn_2_available and torch.cuda.is_available()


def is_xformers_available(operation: str = None, version: str = None):
    if version is not None and operation is not None:
        return _xformers_available and compare_versions(parse(_xformers_version), operation, version)
    return _xformers_available


def is_accelerate_available(operation: str = None, version: str = None):
    if version is not None and operation is not None:
        return _accelerate_available and compare_versions(parse(_accelerate_version), operation, version)
    return _accelerate_available


def is_deepspeed_available(operation: str = None, version: str = None):
    if version is not None and operation is not None:
        return _deepspeed_available and compare_versions(parse(_deepspeed_version), operation, version)
    return _deepspeed_available


def is_peft_available():
    return _peft_available


def is_torch_compile_available():
    if not is_torch_available():
        return False

    import torch

    # We don't do any version check here to support nighlies marked as 1.14.
    # Ultimately needs to check version against 2.0 but let's do it later.
    return hasattr(torch, "compile")


def is_apex_available():
    return _apex_available


def is_torch_fx_available():
    return _torch_fx_available


def is_wandb_available():
    # any value of WANDB_DISABLED disables wandb
    if os.getenv("WANDB_DISABLED", "").upper() in ENV_VARS_TRUE_VALUES:
        logger.warning(
            "Using the `WANDB_DISABLED` environment variable is deprecated and will be removed in v5. Use the "
            "--report_to flag to control the integrations used for logging result (for instance --report_to none)."
        )
        return False
    return _wandb_available


def is_safetensors_available():
    return _safetensors_available


# Cache this result has it's a C FFI call which can be pretty time-consuming
def is_torch_distributed_available():
    if is_torch_available():
        import torch.distributed

        return torch.distributed.is_available()
    else:
        return False


def is_transformers_available(operation: str = None, version: str = None):
    if version is not None and operation is not None:
        return _transformers_available and compare_versions(parse(_transformers_version), operation, version)
    return _transformers_available


def is_safetensors_available():
    return _safetensors_available


def is_datasets_available():
    return _datasets_available


def is_tokenizers_available(operation: str = None, version: str = None):
    if version is not None and operation is not None:
        return _tokenizers_available and compare_versions(parse(_tokenizers_version), operation, version)
    return _tokenizers_available
