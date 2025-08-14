import os
from typing import Any

import torch
import torch.distributed as dist

from nextstep.utils.loguru import logger


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
            dist.all_reduce(x_reduce, op=dist.ReduceOp.SUM)
            x_reduce /= self.world_size
            return x_reduce.item()
        else:
            return x

    def all_reduce_sum(self, x: float | torch.Tensor):
        if isinstance(x, torch.Tensor):
            x = x.item()

        if self.world_size > 1:
            x_reduce = torch.tensor(x, device=self.device)
            dist.all_reduce(x_reduce, op=dist.ReduceOp.SUM)
            return x_reduce.item()
        else:
            return x

    def all_gather_object(self, x: Any) -> list[Any]:
        if self.world_size > 1:
            x_list = [None for _ in range(self.world_size)]
            dist.all_gather_object(x_list, x)
            return x_list
        else:
            return [x]

    def all_reduce_mean_dict(self, x: dict[str, float | int]) -> dict[str, float]:
        all_x = self.all_gather_object(x)
        result = {}
        for _x in all_x:
            for k, v in _x.items():
                if k not in result:
                    result[k] = 0.0
                result[k] += v
        for k, v in result.items():
            result[k] = v / self.world_size
        return result


dist_ctx = DistributedContext()


def broadcast_object(obj, src=0):
    # broadcast a pickle-able python object from rank-0 to all ranks
    if get_rank() == src:
        objects = [obj]
    else:
        objects = [None]
    dist.broadcast_object_list(objects, src=src)
    return objects[0]
