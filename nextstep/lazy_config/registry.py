import pydoc
from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig

from nextstep.utils.loguru import logger


def _convert_target_to_string(t: Any) -> str:
    """
    Inverse of ``locate()``.

    Args:
        t: any object with ``__module__`` and ``__qualname__``
    """
    if t is None:
        return None
    module, qualname = t.__module__, t.__qualname__

    # Compress the path to this object, e.g. ``module.submodule._impl.class``
    # may become ``module.submodule.class``, if the later also resolves to the same
    # object. This simplifies the string, and also is less affected by moving the
    # class implementation.
    module_parts = module.split(".")
    for k in range(1, len(module_parts)):
        prefix = ".".join(module_parts[:k])
        candidate = f"{prefix}.{qualname}"
        try:
            if locate(candidate) is t:
                return candidate
        except ImportError:
            pass
    return f"{module}.{qualname}"


def locate(name: str) -> Any:
    """
    Locate and return an object ``x`` using an input string ``{x.__module__}.{x.__qualname__}``,
    such as "module.submodule.class_name".

    Raise Exception if it cannot be found.
    """
    if name is None:
        return None
    obj = pydoc.locate(name)

    # Some cases (e.g. torch.optim.sgd.SGD) not handled correctly
    # by pydoc.locate. Try a private function from hydra.
    if obj is None:
        try:
            # from hydra.utils import get_method - will print many errors
            from hydra.utils import _locate
        except ImportError as e:
            raise ImportError(f"Cannot dynamically locate object {name}!") from e
        else:
            obj = _locate(name)  # it raises if fails

    return obj


@dataclass
class BaseRegInfo:
    name: str
    cls: type | str | None = None  # NOTE: Only used for compatibility with info without the cls field.


class BaseRegistry:
    """
    A base registry class that manages registration of named components.

    This class provides functionality to register and retrieve components by name,
    supporting both BaseRegInfo and DictConfig objects for registration.
    """

    def __init__(self, registry_name: str, name_len: int = 16):
        """
        Initialize a new registry.

        Args:
            registry_name: The name of the registry
            name_len: Length used for formatting log messages (default: 16)
        """
        self.__registry_name = registry_name
        self.__registry: dict[str, BaseRegInfo] = {}

        # used for formatting the log message
        self.name_len = name_len

    @property
    def registry_name(self):
        return self.__registry_name

    @property
    def registry(self):
        return self.__registry

    def _register(self, info: BaseRegInfo | DictConfig, force: bool = False):
        """
        Internal method to register a single component.

        Args:
            info: Registration information as BaseRegInfo or DictConfig
            force: If True, allows overriding existing registrations (default: False)

        Raises:
            TypeError: If info is neither BaseRegInfo nor DictConfig
            KeyError: If name already exists and force is False
        """
        if not isinstance(info, (BaseRegInfo, DictConfig)):
            raise TypeError(f"The registration object must be a `BaseRegInfo` or `DictConfig`, but got {type(info)}")

        if info.name in self.__registry:
            if not force:
                raise KeyError(f"`{info.name}` is already registered in `{self.__registry_name}`")
            else:
                logger.warning(f"`{info.name}` is already registered in `{self.__registry_name}`, but we force to override it.")
        info.cls = _convert_target_to_string(info.cls)
        self.__registry[info.name] = info

    def register(self, info: BaseRegInfo | list[BaseRegInfo] | DictConfig | list[DictConfig], force: bool = False):
        """
        Register one or multiple components.

        Args:
            info: Single or list of registration information (BaseRegInfo or DictConfig)
            force: If True, allows overriding existing registrations (default: False)
        """
        if isinstance(info, (BaseRegInfo, DictConfig)):
            info = [info]
        for _info in info:
            try:
                self._register(_info, force=force)
            except:
                logger.warning(f"{_info.name} is not registered in {self.__registry_name}.")

    def __len__(self):
        return len(self.__registry)

    def __getitem__(self, idx) -> BaseRegInfo:
        return self.__registry[idx]
