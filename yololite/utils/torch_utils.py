# YOLO-Lite 🚀

from contextlib import contextmanager
from datetime import datetime
import functools
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Union

from yololite.utils.cpu import CPUInfo
__version__ = "8.3.230"
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from yololite.utils import (
    DEFAULT_CFG_DICT,
    DEFAULT_CFG_KEYS,
    LOGGER,
    NUM_THREADS,
    PYTHON_VERSION,
    WINDOWS,
    colorstr,
)
from yololite.utils.checks import check_version
from yololite.utils.patches import torch_load
from yololite.utils.tal import TORCH_1_10

# Version checks (all default to version>=min_version)
TORCH_1_9 = check_version(torch.__version__, "1.9.0")
TORCH_1_13 = check_version(torch.__version__, "1.13.0")
TORCH_2_0 = check_version(torch.__version__, "2.0.0")
TORCH_2_4 = check_version(torch.__version__, "2.4.0")
if WINDOWS and check_version(torch.__version__, "==2.4.0"):  # reject version 2.4.0 on Windows
    LOGGER.warning(
        "WARNING ⚠️ Known issue with torch==2.4.0 on Windows with CPU, recommend upgrading to torch>=2.4.1 to resolve "
        "https://github.com/ultralytics/ultralytics/issues/15049"
    )

def torch_distributed_zero_first(local_rank: int):
    """Ensure all processes in distributed training wait for the local master (rank 0) to complete a task first."""
    initialized = torch.dist.is_available() and torch.dist.is_initialized()
    use_ids = initialized and torch.dist.get_backend() == "nccl"

    if initialized and local_rank not in {-1, 0}:
        torch.dist.barrier(device_ids=[local_rank]) if use_ids else torch.dist.barrier()
    yield
    if initialized and local_rank == 0:
        torch.dist.barrier(device_ids=[local_rank]) if use_ids else torch.dist.barrier()


def smart_inference_mode():
    """Apply torch.inference_mode() decorator if torch>=1.10.0, else torch.no_grad() decorator."""

    def decorate(fn):
        """Apply appropriate torch decorator for inference mode based on torch version."""
        if TORCH_1_9 and torch.is_inference_mode_enabled():
            return fn  # already in inference_mode, act as a pass-through
        else:
            return (torch.inference_mode if TORCH_1_10 else torch.no_grad)()(fn)

    return decorate


def autocast(enabled: bool, device: str = "cuda"):
    """Get the appropriate autocast context manager based on PyTorch version and AMP setting.

    This function returns a context manager for automatic mixed precision (AMP) training that is compatible with both
    older and newer versions of PyTorch. It handles the differences in the autocast API between PyTorch versions.

    Args:
        enabled (bool): Whether to enable automatic mixed precision.
        device (str, optional): The device to use for autocast.

    Returns:
        (torch.amp.autocast): The appropriate autocast context manager.

    Examples:
        >>> with autocast(enabled=True):
        ...     # Your mixed precision operations here
        ...     pass

    Notes:
        - For PyTorch versions 1.13 and newer, it uses `torch.amp.autocast`.
        - For older versions, it uses `torch.cuda.amp.autocast`.
    """
    if TORCH_1_13:
        return torch.amp.autocast(device, enabled=enabled)
    else:
        return torch.cuda.amp.autocast(enabled)


@functools.lru_cache
def get_cpu_info():
    """Return a string with system CPU information, i.e. 'Apple M2'."""
    from ultralytics.utils import PERSISTENT_CACHE  # avoid circular import error

    if "cpu_info" not in PERSISTENT_CACHE:
        try:
            PERSISTENT_CACHE["cpu_info"] = CPUInfo.name()
        except Exception:
            pass
    return PERSISTENT_CACHE.get("cpu_info", "unknown")


@functools.lru_cache
def get_gpu_info(index):
    """Return a string with system GPU information, i.e. 'Tesla T4, 15102MiB'."""
    properties = torch.cuda.get_device_properties(index)
    return f"{properties.name}, {properties.total_memory / (1 << 20):.0f}MiB"


def select_device(device="", newline=False, verbose=True):
    """Select the appropriate PyTorch device based on the provided arguments.

    The function takes a string specifying the device or a torch.device object and returns a torch.device object
    representing the selected device. The function also validates the number of available devices and raises an
    exception if the requested device(s) are not available.

    Args:
        device (str | torch.device, optional): Device string or torch.device object. Options include 'cpu', 'cuda', '0',
            '0,1,2,3', 'mps', or '-1' for auto-select. Defaults to auto-selecting the first available GPU, or CPU if no
            GPU is available.
        newline (bool, optional): If True, adds a newline at the end of the log string.
        verbose (bool, optional): If True, logs the device information.

    Returns:
        (torch.device): Selected device.

    Examples:
        >>> select_device("cuda:0")
        device(type='cuda', index=0)

        >>> select_device("cpu")
        device(type='cpu')

    Notes:
        Sets the 'CUDA_VISIBLE_DEVICES' environment variable for specifying which GPUs to use.
    """
    if isinstance(device, torch.device) or str(device).startswith(("tpu", "intel", "vulkan")):
        return device

    s = f"Ultralytics {__version__} 🚀 Python-{PYTHON_VERSION} torch-{torch.__version__} "