from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class DiffusableBatch:
    """Batch for unconditional diffusion."""

    X: torch.Tensor
    timesteps: Optional[torch.Tensor] = None
    target_time: Optional[torch.Tensor] = None
    X_mask: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return len(self.X)

    @property
    def device(self) -> torch.device:
        return self.X.device


def collate_batch(data: list[dict[str, torch.Tensor]]) -> DiffusableBatch:
    if "X" not in data[0]:
        raise KeyError("Expected key 'X' in dataset samples.")
    X = torch.stack([row["X"] for row in data])
    timesteps = (
        torch.stack([row["timesteps"] for row in data])
        if "timesteps" in data[0]
        else None
    )
    target_time = (
        torch.stack([row["target_time"] for row in data])
        if "target_time" in data[0]
        else None
    )
    X_mask = (
        torch.stack([row["X_mask"] for row in data])
        if "X_mask" in data[0] and row_has_key(data, "X_mask")
        else None
    )
    return DiffusableBatch(X=X, timesteps=timesteps, target_time=target_time, X_mask=X_mask)


def row_has_key(rows: list[dict[str, torch.Tensor]], key: str) -> bool:
    return all(key in row and row[key] is not None for row in rows)
