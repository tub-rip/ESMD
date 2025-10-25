import logging
from typing import Optional, Tuple

import numpy as np
import torch

from ..types import  NUMPY_TORCH

logger = logging.getLogger(__name__)


def infer_event_timestamps_into_sec(ts_array: np.ndarray) -> np.ndarray:
    """Estimate timestamp order (s, ms, us, ns)

    Args:
        ts_array (np.ndarray): the original timestamp array of events.
            Based on the assumption that events are in us order, it infers the scale (order) of the
            event timestamps

    Returns:
        np.ndarray: _description_
    """
    med_diff = np.percentile(np.diff(ts_array), 99)
    # print('aaaaa', med_diff, med_diff > 1e-1)
    if med_diff > 100:  # 1000~. ns
        return ts_array / 1e9
    elif med_diff > 1e-1:  # 1~, us
        return ts_array / 1e6
    elif med_diff > 1e-4:  # 0.001~, ms
        return ts_array / 1e3
    else:  # sec
        return ts_array


def filter_event(
    events: NUMPY_TORCH, start_time: Optional[float] = None, end_time: Optional[float] = None
) -> NUMPY_TORCH:
    """Filter event based on timestamps.
    event should be sorted based on time.

    Args:
        events (NUMPY_TORCH): [n, 4]
        start_time (Optional[float], optional): _description_. Defaults to None.
        end_time (Optional[float], optional): _description_. Defaults to None.

    Returns:
        NUMPY_TORCH: Filered events
    """
    if start_time is None and end_time is None:
        raise ValueError("Either start_time or end_time should be non-None")

    i1 = np.searchsorted(events[:, 2], start_time) if start_time is not None else 0
    i2 = np.searchsorted(events[:, 2], end_time) if end_time is not None else len(events)

    if i1 >= i2 or i1 >= len(events):
        # logger.warning('No events filtered')
        return np.array([])
    return events[i1:i2]


def crop_event(events: NUMPY_TORCH, x0: int, x1: int, y0: int, y1: int) -> NUMPY_TORCH:
    """Crop events.

    Args:
        events (NUMPY_TORCH): [n x 4]. [x, y, t, p].
        x0 (int): Start of the crop, at row[0]
        x1 (int): End of the crop, at row[0]
        y0 (int): Start of the crop, at row[1]
        y1 (int): End of the crop, at row[1]

    Returns:
        NUMPY_TORCH: Cropped events.
    """
    mask = (
        (x0 <= events[..., 0])
        * (events[..., 0] < x1)
        * (y0 <= events[..., 1])
        * (events[..., 1] < y1)
    )
    cropped = events[mask]
    return cropped



def shift_event(events: NUMPY_TORCH, x0: int, y0: int) -> NUMPY_TORCH:
    """Shift events.

    Args:
        events (NUMPY_TORCH): [n x 4]. [x, y, t, p].
        x0 (int): Start of the shift, at row[0]
        y0 (int): Start of the shift, at row[1]

    Returns:
        NUMPY_TORCH: Cropped events.
    """
    if isinstance(events, np.ndarray):
        return events + np.array([x0, y0, 0, 0])
    return events + torch.tensor([x0, y0, 0, 0]).to(events.device)


# Denoise
def evaluate_denoising(events: NUMPY_TORCH, iwe: NUMPY_TORCH):
    """Calculate ESR

    Args:
        events (NUMPY_TORCH): _description_
        iwe (NUMPY_TORCH): _description_

    Returns:
        _type_: _description_
    """
    if isinstance(events, torch.Tensor):
        events = events.cpu().numpy()
    if isinstance(iwe, torch.Tensor):
        iwe = iwe.cpu().numpy()

    N, M = len(events), int(len(events) * 2 / 3)
    K = iwe.shape[0] * iwe.shape[1]  # n.size
    n = np.copy(iwe)
    # calculate ntss
    ntss = (n * (n - 1)).sum() / (N + np.spacing(1)) / (N - 1 + np.spacing(1))
    
    # calculate ln
    ln = K - ((1 - M / N) ** n).sum()

    # return esr
    return np.sqrt(ntss * ln)

    # n = median_filter(iwe, size=3)
    # N = len(events)
    # K = iwe.shape[0] * iwe.shape[1]  # n.size
    # # calculate ntss
    # ntss = (n * n).sum() / (N * N)
    
    # # calculate ln
    # ln = (K - (0.5 ** n).sum()) / K
    # # return esr
    # esr = 1000 * np.sqrt(ntss * ln)

    # return {"esr": esr}

# Voxel conversion
def create_event_voxel(
    x: torch.Tensor,
    y: torch.Tensor,
    pol: torch.Tensor,
    time: torch.Tensor,
    voxel_shape: tuple,
    normalize: bool = False,
):
    """Create voxel grid with weights.
    Original code is https://github.com/uzh-rpg/DSEC/blob/main/scripts/dataset/representations.py
    This encode positive and negative events together.
    The polarity information is used for the weight of the voxel.

    Args:
        x (torch.Tensor) ... (n_events, ). x is width direction.
        y (torch.Tensor) ... (n_events, ).
        pol (torch.Tensor) ... (n_events, ). The polarity is [-1, +1].
        time (torch.Tensor) ... (n_events, ).
        voxel_shape (tuple) ... [C, H, W].
        normalize (bool) ... True to normalie the output voxel.

    Returns:
        voxel_grid (torch.Tensor) ... (voxel_shape).

    """
    assert x.shape == y.shape == pol.shape == time.shape
    assert x.ndim == 1

    C, H, W = voxel_shape
    with torch.no_grad():
        voxel_grid = x.new_zeros(voxel_shape, dtype=torch.double)

        t_norm = time
        t_norm = (C - 1) * (t_norm - t_norm[0]) / (t_norm[-1] - t_norm[0])

        x0 = x.int()  # int() gives floor
        y0 = y.int()
        t0 = t_norm.int()

        # value = 2 * pol - 1   # for pol in [0, 1]
        value = pol  # for pol already [-1, 1]

        for xlim in [x0, x0 + 1]:
            for ylim in [y0, y0 + 1]:
                for tlim in [t0, t0 + 1]:
                    mask = (
                        (xlim < W)
                        & (xlim >= 0)
                        & (ylim < H)
                        & (ylim >= 0)
                        & (tlim >= 0)
                        & (tlim < C)
                    )
                    interp_weights = (
                        value
                        * (1 - (xlim - x).abs())
                        * (1 - (ylim - y).abs())
                        * (1 - (tlim - t_norm).abs())
                    )

                    index = H * W * tlim.long() + W * ylim.long() + xlim.long()

                    voxel_grid.put_(index[mask], interp_weights[mask], accumulate=True)

        if normalize:
            mask = torch.nonzero(voxel_grid, as_tuple=True)
            if mask[0].size()[0] > 0:
                mean = voxel_grid[mask].mean()
                std = voxel_grid[mask].std()
                if std > 0:
                    voxel_grid[mask] = (voxel_grid[mask] - mean) / std
                else:
                    voxel_grid[mask] = voxel_grid[mask] - mean

    return voxel_grid


# EVFlownet utils
def calc_floor_ceil_delta(x):
    """
    Args:
        x (torch.Tensor)

    Returns
        [floor(x), (floor(x) + 1) - x], [ceil(x), x - floor(x)]
    """
    x_fl = torch.floor(x + 1e-8)
    x_ce = torch.ceil(x - 1e-8)
    x_ce_fake = torch.floor(x) + 1

    dx_ce = x - x_fl
    dx_fl = x_ce_fake - x
    return [x_fl.long(), dx_fl], [x_ce.long(), dx_ce]


def create_update(x, y, t, dt, p, vol_size: tuple):
    """Helper function to create discretized event volume.

    Args:
        x, y, t (torch.Tensor)
        vol_size (tuple) ... (b, x, y). x is height.
    """
    # This is old, when x-width and y-height.
    # assert (x >= 0).byte().all() and (x < vol_size[2]).byte().all()
    # assert (y >= 0).byte().all() and (y < vol_size[1]).byte().all()
    assert (x >= 0).byte().all() and (x < vol_size[1]).byte().all()
    assert (y >= 0).byte().all() and (y < vol_size[2]).byte().all()
    assert (t >= 0).byte().all() and (t < vol_size[0] // 2).byte().all()

    vol_mul = torch.where(
        p < 0,
        torch.ones(p.shape, dtype=torch.long) * vol_size[0] // 2,
        torch.zeros(p.shape, dtype=torch.long),
    )
    # This is old, when x-width and y-height.
    # inds = (vol_size[1] * vol_size[2]) * (t + vol_mul) + (vol_size[2]) * y + x
    inds = (vol_size[1] * vol_size[2]) * (t + vol_mul) + (vol_size[2]) * x + y
    vals = dt
    return inds, vals


def generate_discretized_event_volume(events: torch.Tensor, vol_size: tuple):
    """Create discretized event volume for a given patch.
    Original code is https://github.com/alexzzhu/EventGAN/blob/master/EventGAN/utils/event_utils.py
    This volume encode positive and negative polarity separately.
    This means, [:n_bin // 2] ... Positive, [n_bin // 2:] negative events aggregation.

    Args:
        events (torch.Tensor) ... [n_events, 4]. 4 is [x, y, t, p]
        vol_size (tuple) ... tuple specifing the return volume size.

    Returns:
        volume (torch.tensor) ... [t, x, y]. t is discretized.
    """
    volume = events.new_zeros(vol_size)
    x = events[:, 0].long()
    y = events[:, 1].long()
    t = events[:, 2]

    t_min = t.min()
    t_max = t.max()
    t_scaled = (t - t_min) * ((vol_size[0] // 2 - 1) / (t_max - t_min))
    ts_fl, ts_ce = calc_floor_ceil_delta(t_scaled.squeeze())

    inds_fl, vals_fl = create_update(x, y, ts_fl[0], ts_fl[1], events[:, 3], vol_size)
    volume.view(-1).put_(inds_fl, vals_fl, accumulate=True)
    inds_ce, vals_ce = create_update(x, y, ts_ce[0], ts_ce[1], events[:, 3], vol_size)
    volume.view(-1).put_(inds_ce, vals_ce, accumulate=True)
    return volume


def bilinear_sampling_tensor(image: torch.Tensor, events: torch.Tensor, weight: float=1.0) -> torch.Tensor:
    """Bilinear sample the image value, using the coordinates of events.

    Args:
        image (torch.Tensor) ... [H, W]
        events (torch.Tensor) ... [n_events, 4] Batch of events. 4 is (x, y, t, p). Attention that (x, y) could float.

    Returns:
        torch.Tensor ... [n_events, 1]
    """
    assert len(events.shape) == 2

    h, w = image.shape
    bilinear_sample = torch.zeros_like(events)   # ne, 4 ... four bilinear locations.
    floor_xy = torch.floor(events[..., :2] + 1e-6)
    floor_to_xy = events[..., :2] - floor_xy
    floor_xy = floor_xy.long()

    x1 = floor_xy[..., 1]
    y1 = floor_xy[..., 0]

    lin_image = image.view(-1)
    inds = torch.stack(
        [
            x1 + y1 * w,
            x1 + (y1 + 1) * w,
            (x1 + 1) + y1 * w,
            (x1 + 1) + (y1 + 1) * w,
        ],
    )  # [4 x n_events]
    inds_mask = torch.stack(
        [
            (0 <= x1) * (x1 < w) * (0 <= y1) * (y1 < h),
            (0 <= x1) * (x1 < w) * (0 <= y1 + 1) * (y1 + 1 < h),
            (0 <= x1 + 1) * (x1 + 1 < w) * (0 <= y1) * (y1 < h),
            (0 <= x1 + 1) * (x1 + 1 < w) * (0 <= y1 + 1) * (y1 + 1 < h),
        ],
    )
    inds = (inds * inds_mask).long()  # 4 x ne
    bilinear_indices = torch.stack([
        lin_image.gather(-1, _ind) for _ind in inds
    ])   # 4 x ne

    w_pos0 = (1 - floor_to_xy[..., 0]) * (1 - floor_to_xy[..., 1]) * weight
    w_pos1 = floor_to_xy[..., 0] * (1 - floor_to_xy[..., 1]) * weight
    w_pos2 = (1 - floor_to_xy[..., 0]) * floor_to_xy[..., 1] * weight
    w_pos3 = floor_to_xy[..., 0] * floor_to_xy[..., 1] * weight
   
    bilinear_sample = \
        bilinear_indices[0] * w_pos0 * inds_mask[0] + \
        bilinear_indices[1] * w_pos1 * inds_mask[1] + \
        bilinear_indices[2] * w_pos2 * inds_mask[2] + \
        bilinear_indices[3] * w_pos3 * inds_mask[3]
    # vals = torch.cat([w_pos0, w_pos1, w_pos2, w_pos3], dim=-1)  # [(b,) n_events x 4]
    # vals = vals * inds_mask
    # image.scatter_add_(1, inds, vals)
    # return image.reshape((nb,) + self.image_size).squeeze()
    return bilinear_sample
