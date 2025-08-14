import os
import re

from .lazyrun import singlegpu_debug, multigpu_debug, smartrun
from .data._gen_meta import gen_meta
from .data._warmup_data import warmup_data
from .lazy_config import _compare

real_path = os.path.realpath(__file__)
pyproject_toml_path = os.path.join(os.path.dirname(os.path.dirname(real_path)), "pyproject.toml")
with open(pyproject_toml_path, "r") as f:
    content = f.read()

__version__ = re.search(r"version = \"(.*)\"", content).group(1)
