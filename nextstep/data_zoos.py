import os
import shutil

TMP_DIR = "./tmp"
DATA_META_PATH = "configs/data/pretrain_data.json"
CACHE_DIRS = [
    "./cache",
]

def get_remaining_space(path=None):
    try:
        usage = shutil.disk_usage(path)
        return usage.free
    except Exception as e:
        raise ValueError(f"Error getting disk usage: {e}")

def url2local_path(cache_dir: str, url: str) -> str:
    return os.path.join(cache_dir, url)


def get_cache_path(url: str) -> str | None:
    """Convert URL to a cache path.

    Args:
        url: Original URL
    Returns:
        Cache file path
    """
    if CACHE_DIRS is None:
        return None

    # local file will not be cached
    if "s3://" not in url:
        return None

    cached_urls = []
    is_cached = []
    for cache_dir in CACHE_DIRS:
        cached_url = url2local_path(cache_dir, url)
        cached_urls.append(cached_url)
        is_cached.append(os.path.exists(cached_url))

    cached_urls = list(set([_cached_url for _is_cached, _cached_url in zip(is_cached, cached_urls) if _is_cached]))

    # If found in one cache directory, return the cache path
    if len(cached_urls) == 1:
        return cached_urls[0]

    # If found in multiple cache directories, remove duplicates
    if len(cached_urls) > 1:
        for cached_url in cached_urls:
            os.remove(cached_url)

    # Select the cache directory with the most remaining space
    cache_dir = CACHE_DIRS[0]
    remaining_space = get_remaining_space(cache_dir)
    for _cache_dir in CACHE_DIRS[1:]:
        if get_remaining_space(_cache_dir) > remaining_space:
            cache_dir = _cache_dir
            remaining_space = get_remaining_space(_cache_dir)

    return url2local_path(cache_dir, url)
