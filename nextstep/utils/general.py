"""
Standalone utils that has nothing to do with Steptron.
"""

import os

import torch

_SET_AFFINITY = False


def _get_numa_node_by_gpu_uuid(target_uuid):
    import subprocess

    """
    Get the corresponding NUMA node ID based on GPU UUID.

    Args:
        target_uuid (str): GPU UUID.

    Returns:
        int: NUMA node ID.

    Raises:
        RuntimeError: If command execution fails or file reading error.
        ValueError: If UUID does not exist or PCI format is invalid.
    """
    # Get GPU UUID and PCI bus ID list
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,pci.bus_id", "--format=csv"], universal_newlines=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to execute nvidia-smi: {e}") from e
    except FileNotFoundError:
        raise RuntimeError("nvidia-smi command not found, please ensure NVIDIA driver is installed.")

    # Parse output, build UUID to PCI bus ID mapping
    uuid_pci_map = {}
    for line in output.strip().split("\n")[1:]:  # Skip header line
        if line.strip():
            uuid, pci_bus_id = [s.strip() for s in line.split(",", 1)]
            uuid_pci_map[uuid] = pci_bus_id

    if target_uuid not in uuid_pci_map:
        raise ValueError(f"GPU with UUID {target_uuid} not found")

    pci_bus_id = uuid_pci_map[target_uuid]

    # Format PCI bus ID to sysfs path format (e.g., 0000:00:00.0)
    parts = pci_bus_id.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid PCI bus ID format: {pci_bus_id}")
    domain = parts[0][-4:].zfill(4)  # Extract last 4 digits as domain, zero-pad
    formatted_pci = f"{domain}:{parts[1]}:{parts[2]}".lower()

    # Read NUMA node information
    sysfs_path = f"/sys/bus/pci/devices/{formatted_pci}/numa_node"
    if not os.path.exists(sysfs_path):
        raise RuntimeError(f"PCI device {formatted_pci} does not exist")

    try:
        with open(sysfs_path, "r") as f:
            numa_node = f.read().strip()
    except IOError as e:
        raise RuntimeError(f"Unable to read {sysfs_path}: {e}") from e

    # Parse NUMA node ID
    try:
        numa_node_id = int(numa_node)
    except ValueError:
        raise RuntimeError(f"Invalid NUMA node value: {numa_node}")

    if numa_node_id < 0:
        raise RuntimeError(f"GPU is not associated with any NUMA node (id={numa_node_id})")

    return numa_node_id


def _get_current_gpu_uuid():
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    if hasattr(properties, "uuid"):
        return "GPU-" + str(properties.uuid)
    # Pytorch version does not provide uuid
    # Warning: using an internal method to get the uuid
    nvml_device_hanlder = torch.cuda._get_pynvml_handler(device)
    import pynvml

    return pynvml.nvmlDeviceGetUUID(nvml_device_hanlder)


def set_numa_affinity(numa_node_id: int = None, strict_bind=False):
    global _SET_AFFINITY
    if _SET_AFFINITY:
        return

    import ctypes as ct
    from ctypes.util import find_library

    class bitmask_t(ct.Structure):
        _fields_ = [
            ("size", ct.c_ulong),
            ("maskp", ct.POINTER(ct.c_ulong)),
        ]

    LIBNUMA = ct.CDLL(find_library("numa"))
    LIBNUMA.numa_parse_nodestring.argtypes = [ct.c_char_p]
    LIBNUMA.numa_parse_nodestring.restype = ct.POINTER(bitmask_t)
    LIBNUMA.numa_run_on_node_mask.argtypes = [ct.POINTER(bitmask_t)]
    LIBNUMA.numa_run_on_node_mask.restype = ct.c_int
    if strict_bind:
        LIBNUMA.numa_set_membind.argtypes = [ct.POINTER(bitmask_t)]
        LIBNUMA.numa_set_membind.restype = ct.c_void_p
    else:
        LIBNUMA.numa_set_preferred.argtypes = [ct.POINTER(bitmask_t)]
        LIBNUMA.numa_set_preferred.restype = ct.c_void_p
    LIBNUMA.numa_num_configured_nodes.argtypes = []
    LIBNUMA.numa_num_configured_nodes.restype = ct.c_int

    def numa_bind(nid: int):
        bitmask = LIBNUMA.numa_parse_nodestring(bytes(str(nid), "ascii"))
        LIBNUMA.numa_run_on_node_mask(bitmask)
        if strict_bind:
            LIBNUMA.numa_set_membind(bitmask)
        else:
            LIBNUMA.numa_set_preferred(bitmask)

    if numa_node_id is None:
        gpu_uuid = _get_current_gpu_uuid()
        numa_node_id = _get_numa_node_by_gpu_uuid(gpu_uuid)
        print(f"NUMA node ID for GPU UUID '{gpu_uuid}' is {numa_node_id}")
    try:
        numa_bind(numa_node_id)
        print("BIND_NUMA: success")
    except Exception as e:
        print(f"BIND_NUMA: {e}")
    _SET_AFFINITY = True
