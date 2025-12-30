from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class DiffusionBatch:
    """Batch holding context (conditioning window) and target segment."""

    context: torch.Tensor
    target: torch.Tensor
    timesteps: Optional[torch.Tensor] = None
    context_time: Optional[torch.Tensor] = None
    target_time: Optional[torch.Tensor] = None
    target_clean: Optional[torch.Tensor] = None
    context_mask: Optional[torch.Tensor] = None
    target_mask: Optional[torch.Tensor] = None
    context_mask_time: Optional[torch.Tensor] = None
    target_mask_time: Optional[torch.Tensor] = None
    cov: Optional[torch.Tensor] = None
    cov_mask: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return len(self.target)

    @property
    def device(self) -> torch.device:
        return self.target.device

    def to(self, device: torch.device) -> "DiffusionBatch":
        kwargs = {
            "context": self.context.to(device),
            "target": self.target.to(device),
        }
        if self.timesteps is not None:
            kwargs["timesteps"] = self.timesteps.to(device)
        if self.context_time is not None:
            kwargs["context_time"] = self.context_time.to(device)
        if self.target_time is not None:
            kwargs["target_time"] = self.target_time.to(device)
        if self.target_clean is not None:
            kwargs["target_clean"] = self.target_clean.to(device)
        if self.context_mask is not None:
            kwargs["context_mask"] = self.context_mask.to(device)
        if self.target_mask is not None:
            kwargs["target_mask"] = self.target_mask.to(device)
        if self.context_mask_time is not None:
            kwargs["context_mask_time"] = self.context_mask_time.to(device)
        if self.target_mask_time is not None:
            kwargs["target_mask_time"] = self.target_mask_time.to(device)
        if self.cov is not None:
            kwargs["cov"] = self.cov.to(device)
        if self.cov_mask is not None:
            kwargs["cov_mask"] = self.cov_mask.to(device)
        return DiffusionBatch(**kwargs)


def collate_diffusion_batch(data: list[dict[str, torch.Tensor]]) -> DiffusionBatch:
    """Collate function for conditional diffusion batches."""

    context = torch.stack([example["context"] for example in data])
    target = torch.stack([example["target"] for example in data])
    extras: dict[str, torch.Tensor] = {}
    if "context_time" in data[0]:
        extras["context_time"] = torch.stack([example["context_time"] for example in data])
    if "target_time" in data[0]:
        extras["target_time"] = torch.stack([example["target_time"] for example in data])
    if "target_clean" in data[0]:
        extras["target_clean"] = torch.stack([example["target_clean"] for example in data])
    if "context_mask" in data[0]:
        extras["context_mask"] = torch.stack([example["context_mask"] for example in data])
    if "target_mask" in data[0]:
        extras["target_mask"] = torch.stack([example["target_mask"] for example in data])
    if "context_mask_time" in data[0]:
        extras["context_mask_time"] = torch.stack([example["context_mask_time"] for example in data])
    if "target_mask_time" in data[0]:
        extras["target_mask_time"] = torch.stack([example["target_mask_time"] for example in data])
    if "cov" in data[0]:
        extras["cov"] = torch.stack([example["cov"] for example in data])
    if "cov_mask" in data[0]:
        extras["cov_mask"] = torch.stack([example["cov_mask"] for example in data])
    if "timesteps" in data[0]:
        extras["timesteps"] = torch.stack([example["timesteps"] for example in data])
    return DiffusionBatch(context=context, target=target, **extras)
