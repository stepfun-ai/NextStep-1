import collections
import fcntl
import io
import mmap
import os
import pickle
import re
import shutil
import struct
import time
import uuid
import zlib
from typing import Callable, Iterator, Literal

import megfile
from tqdm.auto import tqdm

from nextstep.data.data_handler import default_handlers
from nextstep.datasets.data_logger import data_logger as logger


class Bar:
    def __init__(self, src_path: str):
        self.src_path = src_path
        self.total = megfile.smart_stat(self.src_path).size
        src_path_display = self.src_path
        if len(src_path_display) > 30:
            src_path_display = src_path_display[:15] + "..." + src_path_display[-12:]
        self._bar = tqdm(total=self.total, desc=f"Downloading {src_path_display}", unit="B", unit_scale=True)

    def __call__(self, bytes_num):
        self._bar.update(bytes_num)


TarHeader = collections.namedtuple(
    "TarHeader",
    [
        "name",
        "mode",
        "uid",
        "gid",
        "size",
        "mtime",
        "chksum",
        "typeflag",
        "linkname",
        "magic",
        "version",
        "uname",
        "gname",
        "devmajor",
        "devminor",
        "prefix",
    ],
)


def parse_tar_header(header_bytes):
    header = struct.unpack("!100s8s8s8s12s12s8s1s100s6s2s32s32s8s8s155s", header_bytes)
    return TarHeader(*header)


def next_header(offset, header):
    block_size = 512
    size = header.size.decode("utf-8").strip("\x00")
    if size == "":
        return -1
    size = int(size, 8)
    # compute the file size rounded up to the next block size if it is a partial block
    padded_file_size = (size + block_size - 1) // block_size * block_size
    return offset + block_size + padded_file_size


class IndexedTar:
    def __init__(
        self,
        fname: str | None = None,
        stream: io.IOBase | None = None,
        backend: Literal["mmap", "memory"] = "mmap",
        cache_index: bool = True,
    ):
        if fname is None and stream is None:
            raise ValueError("Either fname or stream must be provided")

        self.fname = fname
        self.stream = stream
        self.backend = backend

        if self.fname is None and self.backend == "mmap":
            self.backend = "memory"
            logger.info("Changed to memory backend for in-memory stream")

        if self.stream is None:
            self.stream = megfile.smart_open(self.fname, "rb")

        if self.backend == "mmap":
            if not megfile.is_fs(self.fname):
                raise ValueError("Only support local file for mmap backend")
            self.file_bytes = mmap.mmap(self.stream.fileno(), 0, access=mmap.ACCESS_READ)

        elif self.backend == "memory":
            self.file_bytes = self.stream.read()

        else:
            raise ValueError(f"Unsupported backend: {backend}")

        self.cache_index = cache_index

        self.by_name = {}
        self.by_index = []

        lock_build_index_start_time = time.time()

        index_cache_path = self.fname + ".index"
        self.lock_build_index_and_cache(index_cache_path)

        if not self._is_index_ok():
            try:
                with open(index_cache_path, "rb") as f:
                    cached_data = pickle.load(f)
                self.by_name = cached_data["by_name"]
                self.by_index = cached_data["by_index"]
            except Exception as e:
                logger.warning(f"Failed to load index cache: {e}, rebuild index and cache...")
                try:
                    os.unlink(index_cache_path)
                except OSError:
                    pass
                self.lock_build_index_and_cache(index_cache_path, force_rebuild=True)

        lock_build_index_time = time.time() - lock_build_index_start_time
        if lock_build_index_time > 1:
            logger.debug(f"[WARNING] lock build index time: {lock_build_index_time:.4f}s --> {self.fname}")

    @property
    def size(self):
        return len(self.file_bytes)

    @property
    def checksum(self):
        return zlib.adler32(self.file_bytes)

    def num_samples(self):
        return len(self.by_index)

    def lock_build_index_and_cache(self, index_cache_path, force_rebuild=False):
        """if not cached, build index and cache, else do nothing"""
        if not self.cache_index:
            self._build_index()
            return
        if not os.path.exists(index_cache_path) or force_rebuild:
            with ULockFile(index_cache_path):
                if not os.path.exists(index_cache_path) or force_rebuild:
                    start_time = time.time()
                    self._build_index()
                    self._safe_cache(index_cache_path)
                    logger.info(f"Build index and cache time: {time.time() - start_time:.4f}s")

    def _safe_cache(self, index_cache_path):
        if megfile.is_s3(index_cache_path):
            raise ValueError(f"Cannot cache to S3: {index_cache_path}")

        temp_path = index_cache_path + f".tmp.{uuid.uuid4()}"
        try:
            with open(temp_path, "wb") as f:
                pickle.dump({"by_name": self.by_name, "by_index": self.by_index}, f)
            try:
                os.unlink(index_cache_path)
            except OSError:
                pass
            os.rename(temp_path, index_cache_path)
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    def _build_index(self):
        offset = 0
        while offset >= 0 and offset < len(self.file_bytes):
            header = parse_tar_header(self.file_bytes[offset : offset + 500])
            name = header.name.decode("utf-8").strip("\x00")
            typeflag = header.typeflag.decode("utf-8").strip("\x00")
            if name != "" and name != "././@PaxHeader" and typeflag in ["0", ""]:
                try:
                    size = int(header.size.decode("utf-8")[:-1], 8)
                except ValueError as exn:
                    print(header)
                    raise exn
                if len(name.split(".")) == 2 and name.split(".")[0] != "" and name.split(".")[1] != "":
                    if name in self.by_name:
                        # If name exists, try adding _1, _2 etc until we find an unused name
                        base, ext = name.rsplit(".", 1)
                        counter = 1
                        new_name = f"{base}_{counter}.{ext}"
                        while new_name in self.by_name:
                            counter += 1
                            new_name = f"{base}_{counter}.{ext}"
                        logger.warning_once(f"{name} already exists in {self.fname}, renamed to {new_name}")
                        name = new_name
                    self.by_name[name] = offset
                    self.by_index.append((name, offset, size))
                else:
                    logger.warning_once(
                        f"Filename {self.fname} has a very strange name ({name}), which may cause problems, ignoring it..."
                    )
            offset = next_header(offset, header)

        assert len(self.by_name) == len(
            self.by_index
        ), f"Number of unique names ({len(self.by_name)}) does not match number of indexed samples ({len(self.by_index)})"

    def _is_index_ok(self):
        return len(self.by_name) != 0 and len(self.by_index) != 0

    def get_at_offset(self, offset) -> tuple[str, bytes]:
        header = parse_tar_header(self.file_bytes[offset : offset + 500])
        name = header.name.decode("utf-8").strip("\x00")
        start = offset + 512
        end = start + int(header.size.decode("utf-8")[:-1], 8)
        return name, self.file_bytes[start:end]

    def get_at_index(self, index) -> tuple[str, bytes]:
        name, offset, size = self.by_index[index]
        return self.get_at_offset(offset)

    def get_by_name(self, name) -> tuple[str, bytes]:
        offset = self.by_name[name]
        return self.get_at_offset(offset)

    def __iter__(self) -> Iterator[tuple[str, bytes]]:
        for name, offset, size in self.by_index:
            yield name, self.file_bytes[offset + 512 : offset + 512 + size]

    def __getitem__(self, key) -> tuple[str, bytes]:
        if isinstance(key, int):
            return self.get_at_index(key)
        else:
            return self.get_by_name(key)

    def get_file(self, i) -> tuple[str, bytes]:
        start_time = time.time()
        fname, data = self.get_at_index(i)
        get_file_time = time.time() - start_time
        if get_file_time > 1:
            logger.debug(f"[WARNING] get sample time: {get_file_time:.4f}s --> {self.fname} ({fname})")
        return fname, data

    def names(self):
        return self.by_name.keys()

    def close(self):
        if getattr(self, "stream", None) is not None:
            if not self.stream.closed:
                self.stream.close()
        if getattr(self, "file_bytes", None) is not None:
            if self.backend == "mmap" and not self.file_bytes.closed:
                self.file_bytes.close()

    def __del__(self):
        self.close()


def splitname(fname):
    """Returns the basename and extension of a filename"""
    assert "." in fname, "Filename must have an extension"
    basename, extension = re.match(r"^((?:.*/)?.*?)(\..*)$", fname).groups()
    return basename, extension


def is_key_equal(key, another_key):
    if key == another_key:
        return True
    # handle None case
    if key is None and another_key is None:
        return True
    if key is None or another_key is None:
        return False
    if another_key.startswith(key + "-"):
        return re.match(f"^{re.escape(key)}-\d+$", another_key) is not None
    return False


def keys_sanity_check(keys, indices, path):
    if len(keys) <= 1:
        return indices
    key0 = keys[0]
    other_keys = keys[1:]
    ids = []
    for key in other_keys:
        if not is_key_equal(key0, key):
            raise ValueError(f"Error in {path}: Key mismatch: {key0} != {key}")
        if key == key0:
            ids.append(0)
        else:
            id = key.split(key0 + "-")[-1]
            ids.append(int(id))

    sorted_ids = sorted(ids)
    id_mapping = {v: i for i, v in enumerate(sorted_ids)}
    remapped_ids = [id_mapping[i] for i in ids]  # from 0

    if sorted(remapped_ids) != remapped_ids:
        # logger.warning(f"[WARNING] ({path}): The id order is inconsistent with the storage order: {keys}")
        new_indices = [indices[0]]
        for i in remapped_ids:
            new_indices.append(indices[i + 1])
        return new_indices
    else:
        return indices


def group_by_key(names, path):
    """Group the file names by key.

    Args:
        names: A list of file names.

    Returns:
        A list of lists of indices, where each sublist contains indices of files
        with the same key.
    """
    groups = []
    last_key = None
    current_keys = []
    current = []
    for i, fname in enumerate(names):
        # Ignore files that are not in a subdirectory.
        if "." not in fname:
            print(f"Warning: Ignoring file {fname} (no '.')")
            continue
        key, ext = splitname(fname)
        if not is_key_equal(last_key, key):
            if current:
                current = keys_sanity_check(current_keys, current, path)
                groups.append(current)
            current_keys = []
            current = []
            last_key = key
        current_keys.append(key)
        current.append(i)
    if current:
        current = keys_sanity_check(current_keys, current, path)
        groups.append(current)
    return groups


class ULockFile:
    """A simple locking class. We don't need any of the third
    party libraries since we rely on POSIX semantics for linking
    below anyway."""

    def __init__(self, path):
        self.lockfile_path = path
        self.lockfile = None

    def __enter__(self):
        os.makedirs(os.path.dirname(self.lockfile_path), exist_ok=True)
        for i in range(4):
            try:
                self.lockfile = open(self.lockfile_path, "w")
                fcntl.lockf(self.lockfile, fcntl.LOCK_EX)
                break
            except:
                pass
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        fcntl.lockf(self.lockfile, fcntl.LOCK_UN)
        self.lockfile.close()
        self.lockfile = None


class IndexedTarSamples:
    """A class that accesses samples in a tar file. The tar file must follow
    WebDataset conventions. The tar file is indexed when the IndexedTarSamples
    object is created. The samples are accessed by index using the __getitem__
    method. The __getitem__ method returns a dictionary containing the files
    for the sample. The key for each file is the extension of the file name.
    The key "__key__" is reserved for the key of the sample (the basename of
    each file without the extension). For example, if the tar file contains
    the files "sample1.jpg" and "sample1.txt", then the sample with key
    "sample1" will be returned as the dictionary {"jpg": ..., "txt": ...}.
    """

    def __init__(
        self,
        path: str,
        *,
        cache_to: str | None = None,
        delete_cache: bool = True,
        backend: Literal["mmap", "memory"] = "mmap",
        cache_index: bool = True,
        handlers: dict[str, Callable] | None = None,
        **kwargs,
    ):
        self.path = path
        self.cache_to = cache_to
        self.delete_cache = delete_cache

        lock_copy_start_time = time.time()

        if cache_to is not None:
            self.local_path = cache_to
            if not os.path.exists(self.local_path):
                with ULockFile(self.local_path + ".lock"):
                    # Check if local_path is a directory and remove it if it is
                    if os.path.isdir(self.local_path):
                        logger.info(f"Maybe misuse rclone, remove the directory before copying file: {self.local_path}")
                        shutil.rmtree(self.local_path)
                    # Multiple processes might simultaneously determine the path doesn't exist,
                    # but only one process will proceed with the download
                    # Therefore, check again if the path exists
                    if not os.path.exists(self.local_path):
                        start_time = time.time()
                        megfile.smart_copy(self.path, self.local_path)
                        # megfile.smart_copy(self.path, self.local_path, callback=Bar(src_path=self.path))
                        logger.info(f"Copy {self.path} to {self.local_path} time: {time.time() - start_time:.4f}s")
        else:
            self.local_path = path

        lock_copy_time = time.time() - lock_copy_start_time
        if lock_copy_time > 1:
            logger.debug(f"[WARNING] lock copy time: {lock_copy_time:.4f}s --> {path} ({self.local_path})")

        # use the mmap based implementation
        self.reader = IndexedTar(fname=self.local_path, stream=None, backend=backend, cache_index=cache_index)

        # Get list of all files in stream
        all_files = self.reader.names()

        # Group files by key into samples
        self.samples = group_by_key(all_files, self.path)

        self.uuid = str(uuid.uuid4())

        self.handlers = default_handlers
        if handlers is not None:
            self.handlers.update(handlers)

    def __reduce__(self):
        # Only serialize the constructor arguments, not the entire object
        return (
            self.__class__,
            (self.path,),  # The first argument is the path
            {
                "cache_to": self.cache_to,
                "delete_cache": self.delete_cache,
                "backend": getattr(self, "backend", "mmap"),
                "handlers": self.handlers if hasattr(self, "handlers") else None,
            },
        )

    @property
    def num_samples(self):
        return len(self.samples)

    @property
    def extensions(self):
        ext_set = set()
        for i, sample in enumerate(self):
            if i > min(10, self.num_samples):
                break
            ext_set.update(sample.keys())
        return ext_set - {"__key__"}

    @property
    def size(self):
        return self.reader.size

    @property
    def checksum(self):
        return self.reader.checksum

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Get indexes of files for the sample at index idx
        indexes = self.samples[idx]
        sample = {}
        key = None
        for i in indexes:
            # Get filename and data for the file at index i
            fname, data = self.reader.get_file(i)
            # Split filename into key and extension
            k, ext = splitname(fname)
            # Make sure all files in sample have same key
            key = key or k
            if not is_key_equal(key, k):
                raise ValueError(f"Key mismatch for file {self.path}({fname}): {key} != {k}")
            if ext not in sample:
                sample[ext] = self.handlers[ext](data)
            else:
                if not isinstance(sample[ext], list):
                    sample[ext] = [sample[ext]]
                sample[ext].append(self.handlers[ext](data))
        # Add key to sample
        sample["__key__"] = key
        return sample

    def __str__(self):
        cache_str = f" (Cached to {self.local_path})" if self.local_path != self.path else ""
        tar_info = f"{self.num_samples} samples, with extensions: {self.extensions}"
        return f"<IndexedTarSamples-{id(self)} {self.path}>" + cache_str + "\n" + tar_info

    def __repr__(self):
        return str(self)

    def close(self):
        if getattr(self, "reader", None) is not None:
            self.reader.close()
        # remove the local file
        if self.cache_to is not None and self.delete_cache:
            with ULockFile(self.local_path + ".lock"):
                megfile.smart_remove(self.local_path, missing_ok=True)

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
