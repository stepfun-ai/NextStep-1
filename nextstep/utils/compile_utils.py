import contextlib
import functools
import os
from typing import Callable, Dict, Optional

import torch

from nextstep.utils.loguru import logger

"""
Usage:

1. Control through environment variable (at startup):
    export TORCH_COMPILE_ENABLE=true
    python your_script.py

2. Control through environment variable (disable):
    export TORCH_COMPILE_ENABLE=false  # or not set
    python your_script.py

3. Dynamically control in code:
    compile_manager.set_compile_enabled(True)   # enable
    compile_manager.set_compile_enabled(False)  # disable

4. Select version at runtime:
    # use the version configured
    result = my_function(args)

    # force use the original version
    result = my_function.original(args)

    # force use the compiled version
    result = my_function.compiled(args)
"""

# Global configuration: control whether to enable compile through environment variables
# Default set this env to true
ENABLE_TORCH_COMPILE = os.getenv("ENABLE_TORCH_COMPILE", "true").lower() == "true"


class CompileManager:
    """Global controller for torch.compile"""

    def __init__(self):
        self.compile_enabled = ENABLE_TORCH_COMPILE
        self.compiled_functions: Dict[str, Callable] = {}
        self.original_functions: Dict[str, Callable] = {}

    def set_compile_enabled(self, enabled: bool):
        """Dynamic setting of whether to enable compile"""
        self.compile_enabled = enabled

    def get_compile_status(self):
        """Get the current compile status"""
        return self.compile_enabled

    @contextlib.contextmanager
    def compile_disabled(self):
        """Temporarily disable compile within the context"""
        original_status = self.compile_enabled
        try:
            self.compile_enabled = False
            yield
        finally:
            self.compile_enabled = original_status


# global instance
compile_manager = CompileManager()


def smart_compile(func: Optional[Callable] = None, **compile_kwargs):
    """
    Smart compile decorator

    Args:
        func: The function to decorate
        **compile_kwargs: Other compile parameters, see https://pytorch.org/docs/stable/generated/torch.compile.html
    """

    def decorator(fn: Callable) -> Callable:
        # save the original function
        original_func = fn
        # Use qualified name to handle functions with same name in different classes
        # Include module name to handle functions with same name in different files
        func_name = f"{fn.__module__}.{fn.__qualname__}"
        compile_manager.original_functions[func_name] = original_func

        # if compile is disabled, return the original function
        if not compile_manager.compile_enabled:
            # add attributes to the original function for later access
            original_func.original = original_func
            original_func.compiled = original_func  # point to itself
            return original_func

        # create the compiled function
        try:
            compiled_func = torch.compile(original_func, **compile_kwargs)
            compile_manager.compiled_functions[func_name] = compiled_func
        except Exception as e:
            logger.warning(f"[WARNING] Failed to compile function {func_name}: {e}")
            # if compile fails, revert to the original function
            compiled_func = original_func

        @functools.wraps(original_func)
        def wrapper(*args, **kwargs):
            # check whether to use the compiled version at runtime
            if compile_manager.compile_enabled:
                return compiled_func(*args, **kwargs)
            else:
                return original_func(*args, **kwargs)

        # add attributes to the wrapper for later access
        wrapper.original = original_func
        wrapper.compiled = compiled_func

        return wrapper

    # support direct use of @smart_compile or @smart_compile(...)
    if func is not None:
        return decorator(func)
    return decorator
