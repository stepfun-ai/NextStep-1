import atexit as _atexit
import logging
import os
import sys as _sys
from typing import Literal

import pendulum
from loguru import _defaults
from loguru._logger import Core as _Core
from loguru._logger import Logger as _Logger


class CustomLogger(_Logger):
    def __init__(self, core, exception, depth, record, lazy, colors, raw, capture, patchers, extra):
        self._core = core
        self._options = (exception, depth, record, lazy, colors, raw, capture, patchers, extra)

        # Newly added attributes.
        self._critical_messages = set()
        self._infoed_messages = set()
        self._warned_messages = set()
        self._error_messages = set()

    def critical_once(__self, __message, *args, **kwargs):  # noqa: N805
        r"""Log ``message.format(*args, **kwargs)`` with severity ``'CRITICAL'``."""
        if __message not in __self._critical_messages:
            __self._critical_messages.add(__message)
            __self._log("CRITICAL", False, __self._options, __message, args, kwargs)

    def info_once(__self, __message, *args, **kwargs):  # noqa: N805
        r"""Log ``message.format(*args, **kwargs)`` with severity ``'INFO'``."""
        if __message not in __self._infoed_messages:
            __self._infoed_messages.add(__message)
            __self._log("INFO", False, __self._options, __message, args, kwargs)

    def warning_once(__self, __message, *args, **kwargs):  # noqa: N805
        r"""Log ``message.format(*args, **kwargs)`` with severity ``'WARNING'``."""
        if __message not in __self._warned_messages:
            __self._warned_messages.add(__message)
            __self._log("WARNING", False, __self._options, __message, args, kwargs)

    def error_once(__self, __message, *args, **kwargs):  # noqa: N805
        r"""Log ``message.format(*args, **kwargs)`` with severity ``'ERROR'``."""
        if __message not in __self._error_messages:
            __self._error_messages.add(__message)
            __self._log("ERROR", False, __self._options, __message, args, kwargs)


def get_logger():
    return CustomLogger(
        core=_Core(),
        exception=None,
        depth=0,
        record=False,
        lazy=False,
        colors=False,
        raw=False,
        capture=True,
        patchers=[],
        extra={},
    )


# from loguru
logger = get_logger()


def set_datetime(record):
    record["extra"]["datetime"] = str(pendulum.now("Asia/Shanghai")).split(".")[0]


FORMAT = "<green>{extra[datetime]}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"


class StreamToLoguru:
    """
    stream object that redirects writes to a logger instance.
    """

    def __init__(self, level="INFO"):
        """
        Args:
            level(str): log level string of loguru. Default value: "INFO".
        """
        self.level = level

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            # use caller level log
            logger.opt(depth=2).log(self.level, line.rstrip())

    def flush(self):
        # flush is related with CPR(cursor position report) in terminal
        return _sys.__stdout__.flush()

    def isatty(self):
        # when using colab, jax is installed by default and issue like
        # https://github.com/Megvii-BaseDetection/YOLOX/issues/1437 might be raised
        # due to missing attribute like`isatty`.
        # For more details, checked the following link:
        # https://github.com/google/jax/blob/10720258ea7fb5bde997dfa2f3f71135ab7a6733/jax/_src/pretty_printer.py#L54  # noqa
        return _sys.__stdout__.isatty()

    def fileno(self):
        # To solve the issue when using debug tools like pdb
        return _sys.__stdout__.fileno()


def redirect_sys_output(log_level="INFO"):
    redirect_logger = StreamToLoguru(log_level)
    _sys.stderr = redirect_logger
    _sys.stdout = redirect_logger


class InterceptHandler(logging.Handler):
    """Intercepts standard logging and redirects to loguru."""

    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging():
    """Redirect standard logging to loguru."""
    logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO, force=True)

    # Redirect specific loggers
    logger_names = [name for name in logging.root.manager.loggerDict]
    for logger_name in logger_names:
        logging.getLogger(logger_name).handlers = []
        logging.getLogger(logger_name).propagate = True

    logging.getLogger("botocore").setLevel(logging.WARNING)


def setup_logger(
    _logger: CustomLogger,
    save_dir: str,
    filename: str,
    distributed_rank: int = 0,
    logger_rank: Literal["main", "all"] = "main",
    mode: Literal["a", "o"] = "a",
    enable_redirect_sys_output: bool = True,
    enable_redirect_logging: bool = True,
):
    """setup logger for training and testing.
    Args:
        save_dir (str): location to save log file
        filename (str): log save name.
        distributed_rank (int): device rank when multi-gpu environment
        logger_rank (Literal["main", "all"]): `main` means only main process write log, `all` means all process write log. default is `main`.
        mode (str): log file write mode, `append` or `override`. default is `a`.

    Return:
        logger instance.
    """
    save_file = os.path.join(save_dir, filename)

    if mode == "o" and os.path.exists(save_file):
        os.remove(save_file)

    if logger_rank == "main":
        if distributed_rank == 0:
            _logger.configure(patcher=set_datetime)
            _logger.add(save_file, format=FORMAT)
    elif logger_rank == "all":
        save_file_no_ext, ext = os.path.splitext(save_file)
        save_file = f"{save_file_no_ext}_{distributed_rank}{ext}"
        _logger.configure(patcher=set_datetime)
        _logger.add(save_file, format=FORMAT)

    # redirect stdout/stderr to loguru
    if enable_redirect_sys_output:
        redirect_sys_output("INFO")
    # redirect standard logging to loguru
    if enable_redirect_logging:
        setup_logging()


def default_setup(_logger: CustomLogger):
    # make sure only rank0 process write log
    if int(os.environ.get("RANK", 0)) == 0:
        if _defaults.LOGURU_AUTOINIT and _sys.stderr:
            _logger.configure(patcher=set_datetime)
            _logger.add(_sys.stderr, format=FORMAT, enqueue=True)

    _atexit.register(_logger.remove)


default_setup(logger)
