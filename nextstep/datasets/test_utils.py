"""
Shared dataset testing utility functions
"""
import random
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from nextstep.data.indexed_tar_dataset import DataStatus
from nextstep.datasets.data_logger import data_logger as logger
from nextstep.utils.comm import dist_ctx, init_distributed


def setup_test_environment():
    """Setup test environment: disable warnings and low-level logs, initialize distributed"""
    # Disable warning and lower level logs for tests
    logger.remove()
    logger.add(lambda msg: None, level="ERROR")  # Only show ERROR and above
    init_distributed()


def test_data_sharding(get_dataset, num_workers: int = 8):
    """
    Test data sharding functionality
    
    Args:
        get_dataset: Function that returns a dataset instance
        num_workers: Number of DataLoader workers
    """
    # Create dataloader with no workers
    dataloader = DataLoader(get_dataset(), batch_size=None, num_workers=num_workers)
    dataloader_iter = iter(dataloader)

    steps = int(len(dataloader) / dist_ctx.world_size * 2)
    for i in tqdm(range(steps), desc=f"Test data sharding with {num_workers} workers", total=steps):
        sample = next(dataloader_iter)
        url_indices = dist_ctx.all_gather_list(sample.url_index)
        
        if len(url_indices) != len(set(url_indices)):
            raise ValueError(f"Duplicate index found across different ranks in batch {i}, url indices: {url_indices}")
            
        dist.barrier()


def test_data_recovery(get_dataset, num_workers: int = 8, random_endpoint = False):
    """
    Test data recovery functionality
    
    Args:
        get_dataset: Function that returns a dataset instance
        num_workers: Number of DataLoader workers
        endpoint0: First recovery point
        endpoint1: Second recovery point
    """
    dataset0 = get_dataset()
    data_status0 = DataStatus(num_workers=num_workers)

    dataset1 = get_dataset()
    data_status1 = DataStatus(num_workers=num_workers)

    dataloader = DataLoader(get_dataset(), batch_size=None, num_workers=num_workers)
    dataloader_iter = iter(dataloader)

    steps = int(len(dataloader) / dist_ctx.world_size * 2)
    if random_endpoint:
        endpoint0 = random.randint(0, steps // 2)
        endpoint1 = random.randint(steps // 2, steps - 1)
    else:
        endpoint0 = 1
        endpoint1 = 10

    print(f"endpoint0: {endpoint0}, endpoint1: {endpoint1}")

    for i in tqdm(range(steps), desc=f"Test data recovery with {num_workers} workers", total=steps):
        sample = next(dataloader_iter)

        if i < endpoint0:
            data_status0.update(sample)
        if i < endpoint1:
            data_status1.update(sample)

        if i == endpoint0:
            print(f"set data status0")
            dataset0.set_data_status(data_status0)
            dataloader0 = DataLoader(dataset0, batch_size=None, num_workers=num_workers)
            dataloader0_iter = iter(dataloader0)
            print("start to load data from data status0")
        if i == endpoint1:
            print(f"set data status1")
            dataset1.set_data_status(data_status1)
            dataloader1 = DataLoader(dataset1, batch_size=None, num_workers=num_workers)
            dataloader1_iter = iter(dataloader1)
            print("start to load data from data status1")

        if i >= endpoint0:
            sample0 = next(dataloader0_iter)
            assert sample0.index == sample.index, f"sample0.index: {sample0.index}, sample.index: {sample.index}, failed to load data from data status0"
            
        if i >= endpoint1:
            sample1 = next(dataloader1_iter)
            assert sample1.index == sample.index, f"sample1.index: {sample1.index}, sample.index: {sample.index}, failed to load data from data status1"


def _assert_batch_match(actual, expected, name: str, step: int):
    """Helper function: Check if input_ids of two batches match"""
    assert torch.allclose(actual.batch_data["input_ids"], expected.batch_data["input_ids"]), (
        f"Step {step}: {name} mismatch. "
        f"Shape: {actual.batch_data['input_ids'].shape} vs {expected.batch_data['input_ids'].shape}"
    )


def test_mixed_dataset_recovery(get_dataset, num_workers: int = 8, steps: int = 100, random_endpoint: bool = False):
    """
    Test MixedDataset data recovery functionality
    
    Args:
        get_dataset: Function that returns a MixedDataset instance
        num_workers: Number of DataLoader workers
        steps: Number of test steps
        random_endpoint: Whether to randomly select recovery points
    """
    from nextstep.datasets.mixed_dataset import MixingStatus
    
    dataset0 = get_dataset(num_workers=num_workers)
    mixing_status0 = MixingStatus(num_workers=num_workers)

    dataset1 = get_dataset(num_workers=num_workers)
    mixing_status1 = MixingStatus(num_workers=num_workers)

    dataloader = DataLoader(get_dataset(), batch_size=None, num_workers=num_workers)
    dataloader_iter = iter(dataloader)

    if random_endpoint:
        endpoint0 = random.randint(0, steps // 2)
        endpoint1 = random.randint(steps // 2, steps - 1)
    else:
        endpoint0 = 1
        endpoint1 = 10

    print(f"endpoint0: {endpoint0}, endpoint1: {endpoint1}")

    for i in tqdm(range(steps), desc=f"Test mixed dataset recovery with {num_workers} workers", total=steps):
        sample = next(dataloader_iter)

        if i < endpoint0:
            mixing_status0.update(sample)
        if i < endpoint1:
            mixing_status1.update(sample)

        if i == endpoint0:
            print(f"set mixing status0")
            dataset0.set_mixing_status(mixing_status0)
            dataloader0 = DataLoader(dataset0, batch_size=None, num_workers=num_workers)
            dataloader0_iter = iter(dataloader0)
            print("start to load data from mixing status0")
        if i == endpoint1:
            print(f"set mixing status1")
            dataset1.set_mixing_status(mixing_status1)
            dataloader1 = DataLoader(dataset1, batch_size=None, num_workers=num_workers)
            dataloader1_iter = iter(dataloader1)
            print("start to load data from mixing status1")

        if i >= endpoint0:
            _assert_batch_match(next(dataloader0_iter), sample, "sample0", i)
        if i >= endpoint1:
            _assert_batch_match(next(dataloader1_iter), sample, "sample1", i)


def run_dataset_tests(get_dataset, num_workers: int = 8, random_endpoint = False):
    """
    Run complete dataset test suite
    
    Args:
        get_dataset: Function that returns a dataset instance
        num_workers: Number of DataLoader workers
    """
    test_data_sharding(get_dataset, num_workers)
    test_data_recovery(get_dataset, num_workers, random_endpoint = random_endpoint)
