import os
import time
from contextlib import contextmanager
from functools import wraps

from nextstep.utils.loguru import logger


def retry(total_tries=5, initial_wait=1, backoff_factor=2, max_wait=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            wait_time = initial_wait
            for i in range(total_tries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # If this was the last attempt
                    if i == total_tries - 1:
                        raise ValueError(f"Function failed after {total_tries} attempts") from e
                    logger.error(f"Function failed with error: `{e}`, retry in {wait_time} seconds...")
                    time.sleep(wait_time)
                    # Exponential backoff
                    wait_time *= backoff_factor
                    if max_wait is not None:
                        if wait_time > max_wait:
                            wait_time = max_wait

        return wrapper

    return decorator
