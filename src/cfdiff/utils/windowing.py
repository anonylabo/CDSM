from __future__ import annotations

from typing import List, Tuple


def compute_window_positions(
    total_length: int,
    window_length: int,
    stride: int,
    *,
    align_end: bool = False,
) -> List[Tuple[int, int]]:
    """
    Determine start/stop indices for sliding windows over a 1-D sequence.

    Parameters
    ----------
    total_length:
        Total number of timesteps available.
    window_length:
        Length of each extracted window.
    stride:
        Step size between consecutive window starts.
    align_end:
        If True, ensure the final window terminates at ``total_length`` by
        appending an additional window when the stride does not evenly divide
        the available span.

    Returns
    -------
    List[Tuple[int, int]]
        Inclusive-exclusive (start, stop) index pairs.
    """
    if total_length <= 0:
        raise ValueError(f"total_length must be positive; received {total_length}.")
    if window_length <= 0:
        raise ValueError(f"window_length must be positive; received {window_length}.")
    if stride <= 0:
        raise ValueError(f"stride must be positive; received {stride}.")
    if window_length > total_length:
        raise ValueError(
            f"window_length ({window_length}) cannot exceed total_length ({total_length})."
        )

    positions: List[Tuple[int, int]] = []
    last_start = total_length - window_length
    start = 0

    while start <= last_start:
        positions.append((start, start + window_length))
        start += stride

    if align_end and positions:
        final_start = last_start
        if positions[-1][0] != final_start:
            positions.append((final_start, final_start + window_length))

    return positions
