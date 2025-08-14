import functools
import os
from typing import Any

import torch
import torch.distributed as dist

from nextstep.utils.loguru import logger
import datetime


def init_distributed():
    backend = "nccl"
    dist_url = "env://"
    world_size = get_world_size()
    rank = get_rank()

    logger.info(
        f"Initializing the distributed mode, backend: {backend}, init_method: {dist_url}, world size: {world_size}, rank: {rank}"
    )

    torch.cuda.set_device(get_local_rank())
    dist.init_process_group(
        backend=backend,
        init_method=dist_url,
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(seconds=36000)
    )
    dist.barrier()


class TemporaryProcessGroup:
    def __init__(self, backend="nccl", init_method="env://", **kwargs):
        self.backend = backend
        self.init_method = init_method
        self.kwargs = kwargs

    def __enter__(self):
        if not dist.is_initialized():
            dist.init_process_group(backend=self.backend, init_method=self.init_method, **self.kwargs)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if dist.is_initialized():
            dist.destroy_process_group()


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def get_local_world_size() -> int:
    return int(os.environ.get("LOCAL_WORLD_SIZE", 1))


def get_rank() -> int:
    return int(os.environ.get("RANK", 0))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    return get_rank() == 0


def is_local_main_process() -> bool:
    return get_local_rank() == 0


def get_device() -> torch.device:
    return torch.device("cuda", get_local_rank())


class DistributedContext:

    @property
    def is_distributed(self) -> bool:
        return get_world_size() > 1

    @property
    def world_size(self) -> int:
        return get_world_size()

    @property
    def local_world_size(self) -> int:
        return get_local_world_size()

    @property
    def rank(self) -> int:
        return get_rank()

    @property
    def local_rank(self) -> int:
        return get_local_rank()

    @property
    def device(self) -> torch.device:
        return get_device()

    @property
    def device_id(self) -> int:
        return get_local_rank()

    @property
    def is_main_process(self) -> bool:
        return is_main_process()

    @property
    def is_local_main_process(self) -> bool:
        return is_local_main_process()

    def all_reduce_mean(self, x: float | torch.Tensor):
        if isinstance(x, torch.Tensor):
            x = x.item()

        if self.world_size > 1:
            x_reduce = torch.tensor(x, device=self.device)
            dist.all_reduce(x_reduce)
            x_reduce /= self.world_size
            return x_reduce.item()
        else:
            return x

    def all_reduce_sum(self, x: float | torch.Tensor):
        if isinstance(x, torch.Tensor):
            x = x.item()

        if self.world_size > 1:
            x_reduce = torch.tensor(x, device=self.device)
            dist.all_reduce(x_reduce)
            return x_reduce.item()
        else:
            return x

    def all_gather_list(self, x: float | torch.Tensor):
        if isinstance(x, torch.Tensor):
            x = x.item()

        if self.world_size > 1:
            x_list = [None for _ in range(self.world_size)]
            dist.all_gather_object(x_list, x)
            return x_list
        else:
            return [x]

    def all_gather_object(self, x: Any) -> list[Any]:
        if self.world_size > 1:
            x_list = [None for _ in range(self.world_size)]
            dist.all_gather_object(x_list, x)
            return x_list
        else:
            return [x]


dist_ctx = DistributedContext()


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        logger.warning("Cannot synchronize since distributed package is not available.")
        return
    if not dist.is_initialized():
        logger.warning("Cannot synchronize since process group is not initialized.")
        return
    world_size = get_world_size()
    if world_size == 1:
        return
    if dist.get_backend() == dist.Backend.NCCL:
        # This argument is needed to avoid warnings.
        # It's valid only for NCCL backend.
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD


def all_gather(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: list of data gathered from each rank
    """
    if group is None:
        group = _get_global_gloo_group()  # use CPU group by default, to reduce GPU RAM usage.
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return [data]

    device = data.device
    output = [None for _ in range(world_size)]
    dist.all_gather_object(output, data, group=group)
    output = [o.to(device) for o in output]
    return output


def gather(data, dst=0, group=None):
    """
    Run gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        dst (int): destination rank
        group: a torch process group. By default, will use a group which contains all ranks on gloo backend.

    Returns:
        list[data]: on dst, a list of data gathered from each rank. Otherwise, an empty list.
    """
    if group is None:
        group = _get_global_gloo_group()
    world_size = dist.get_world_size(group=group)
    if world_size == 1:
        return [data]
    rank = dist.get_rank(group=group)

    if rank == dst:
        output = [None for _ in range(world_size)]
        dist.gather_object(data, output, dst=dst, group=group)
        return output
    else:
        dist.gather_object(data, None, dst=dst, group=group)
        return []


def reduce_dict(input_dict, average=True):
    """
    Reduce the values in the dictionary from all processes so that process with rank
    0 has the reduced results.

    Args:
        input_dict (dict): inputs to be reduced. All the values must be scalar CUDA Tensor.
        average (bool): whether to do average or sum

    Returns:
        a dict with the same keys as input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.reduce(values, dst=0)
        if dist.get_rank() == 0 and average:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def broadcast_object(obj, src=0):
    # broadcast a pickle-able python object from rank-0 to all ranks
    if get_rank() == src:
        objects = [obj]
    else:
        objects = [None]
    dist.broadcast_object_list(objects, src=src)
    return objects[0]


def all_gather_object(args, obj, dst=0):
    # gather a pickle-able python object across all ranks
    objects = [None for _ in range(get_world_size())]
    dist.all_gather_object(objects, obj)
    return objects
