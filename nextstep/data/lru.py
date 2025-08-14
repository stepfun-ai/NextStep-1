from collections import OrderedDict
from typing import Literal

from nextstep.data.indexed_tar import IndexedTarSamples
from nextstep.data_zoos import get_cache_path


class LRUCache:
    def __init__(self, capacity: int, release_handler=None):
        """Initialize a new LRU cache with the given capacity."""
        self.capacity = capacity
        self.cache = OrderedDict()
        self.release_handler = release_handler

    def __getitem__(self, key):
        """Return the value associated with the given key, or None."""
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def __setitem__(self, key, value):
        """Associate the given value with the given key."""
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            key, value = self.cache.popitem(last=False)
            if self.release_handler is not None:
                self.release_handler(key, value)

    def __delitem__(self, key):
        """Remove the given key from the cache."""
        if key in self.cache:
            if self.release_handler is not None:
                value = self.cache[key]
                self.release_handler(key, value)
            del self.cache[key]

    def __len__(self):
        """Return the number of entries in the cache."""
        return len(self.cache)

    def __contains__(self, key):
        """Return whether the cache contains the given key."""
        return key in self.cache

    def items(self):
        """Return an iterator over the keys of the cache."""
        return self.cache.items()

    def keys(self):
        """Return an iterator over the keys of the cache."""
        return self.cache.keys()

    def values(self):
        """Return an iterator over the values of the cache."""
        return self.cache.values()

    def clear(self):
        for key in list(self.keys()):
            value = self.cache[key]
            if self.release_handler is not None:
                self.release_handler(key, value)
            del self[key]

    def __del__(self):
        self.clear()


class LRUShards:
    """A class that manages a cache of shards using LRU (Least Recently Used) policy.
    The cache stores URLs as keys and IndexedTarSamples objects as values.
    When a shard is accessed, it is downloaded and cached if not already present.
    The IndexedTarSamples objects are automatically closed when evicted from the cache.
    Tracks cache statistics including number of accesses and cache misses.
    """

    def __init__(
        self,
        lru_size=64,
        cache=True,
        backend: Literal["mmap", "memory"] = "mmap",
        cache_index: bool = True,
        **kwargs,
    ):
        # the cache contains the local name as the key and the downloaded path as the value
        self.lru = LRUCache(lru_size, release_handler=self.release_handler)
        self.cache = cache
        self.backend = backend
        self.cache_index = cache_index
        # keep statistics
        self.reset_stats()

    def reset_stats(self):
        self.accesses = 0
        self.misses = 0

    def __len__(self):
        return len(self.lru)

    def release_handler(self, key, value):
        value.close()

    def clear(self):
        self.lru.clear()

    def get_shard(self, url) -> IndexedTarSamples:
        assert isinstance(url, str)
        self.accesses += 1
        if url not in self.lru:
            self.lru[url] = (
                IndexedTarSamples(
                    url, cache_to=get_cache_path(url), delete_cache=False, backend=self.backend, cache_index=self.cache_index
                )
                if self.cache
                else IndexedTarSamples(url, backend=self.backend, cache_index=self.cache_index)
            )
            self.misses += 1
            self.last_missed = True
        else:
            self.last_missed = False
        return self.lru[url]
