from typing import Tuple, Union

import numpy as np
import torch

NUMPY_TORCH = Union[np.ndarray, torch.Tensor]
FLOAT_TORCH = Union[float, torch.Tensor]


def to_numpy(arr: NUMPY_TORCH):
    if isinstance(arr, torch.Tensor):
        return arr.clone().detach().cpu().numpy()
    return arr

