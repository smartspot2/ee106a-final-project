"""
Pytorch utilities.

Mainly for converting to and from numpy.
"""

import numpy as np
import torch

device = None


def init_device(use_gpu=True, gpu_id=0):
    """
    Initialize device to use for pytorch.
    """
    global device

    if use_gpu:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{gpu_id}")
        else:
            raise RuntimeError(f"GPU device id {gpu_id} is not available.")
    else:
        device = torch.device("cpu")

    print(f"Using device {device}")


def np_to_torch(array: np.ndarray, **kwargs) -> torch.Tensor:
    """
    Convert a numpy array to a torch tensor.
    """
    data = torch.from_numpy(array, **kwargs)
    if data.dtype == torch.float64:
        data = data.float()
    return data.to(device)


def torch_to_np(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a torch tensor to a numpy array.
    """
    return tensor.detach().cpu().numpy()
