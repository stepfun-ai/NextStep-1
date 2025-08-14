import fire
from omegaconf import DictConfig

from nextstep.lazy_config.lazy import CONFIG_KEY, LazyConfig


def cdiff(a, b):
    def colored(text):
        if text.startswith("+"):
            text = f"\033[32m{text}\033[0m"
        elif text.startswith("-"):
            text = f"\033[31m{text}\033[0m"
        elif text.startswith("?"):
            text = f"\033[33m{text}\033[0m"
        return text

    from difflib import ndiff

    ans = "\n".join([colored(x) for x in ndiff([str(x) for x in a], [str(x) for x in b])])
    return ans


def compare(ref: str | DictConfig, exp: str | DictConfig = None):
    """Show an Exp or compare two Exps

    Usage:
    eshow configs/your_config.py
    # or
    eshow configs/your_config1.py configs/your_config2.py
    """
    if isinstance(ref, str):
        ref: DictConfig = LazyConfig.load(ref, CONFIG_KEY)

    if exp is None:
        print(LazyConfig.to_py(ref))
        return

    if isinstance(exp, str):
        exp: DictConfig = LazyConfig.load(exp, CONFIG_KEY)

    diff = cdiff(LazyConfig.to_py(ref).splitlines(), LazyConfig.to_py(exp).splitlines())

    print(diff)


def _compare():
    fire.Fire(compare)


if __name__ == "__main__":
    _compare()
