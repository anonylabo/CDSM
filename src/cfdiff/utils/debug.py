from __future__ import annotations

from typing import Optional

import torch


def describe_tensor(
    tensor: torch.Tensor,
    *,
    name: Optional[str] = None,
    max_rows: int = 2,
    max_cols: int = 6,
) -> str:
    """Return a human readable description of a tensor.

    Parameters
    ----------
    tensor:
        Input tensor. Non-tensor inputs are converted with ``torch.as_tensor``.
    name:
        Optional label printed at the beginning of the description.
    max_rows, max_cols:
        How many rows / columns of data to include in the preview.
    """

    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)

    arr = tensor.detach().cpu()
    flat = arr.reshape(-1)

    header = f"[{name}] " if name else ""
    header += f"shape={tuple(arr.shape)}, dtype={arr.dtype}, device={tensor.device}"

    stats = (
        f"    stats -> mean={flat.mean().item():.4f}, "
        f"std={flat.std().item():.4f}, "
        f"min={flat.min().item():.4f}, "
        f"max={flat.max().item():.4f}"
    )

    if arr.ndim >= 2:
        preview = arr.reshape(-1, arr.shape[-1])[:max_rows, :max_cols]
    else:
        preview = arr[:max_cols]

    body = f"    preview:\n{preview}"
    return "\n".join([header, stats, body])


def print_tensor_description(
    tensor: torch.Tensor,
    *,
    name: Optional[str] = None,
    max_rows: int = 2,
    max_cols: int = 6,
) -> None:
    """Convenience wrapper around :func:`describe_tensor` that prints the result."""

    print(
        describe_tensor(
            tensor,
            name=name,
            max_rows=max_rows,
            max_cols=max_cols,
        )
    )


def empirical_covariance(samples: torch.Tensor) -> torch.Tensor:
    """Compute per-sample covariance matrices for real-valued sequences.

    Parameters
    ----------
    samples:
        Tensor shaped ``(batch, length, channels)``.

    Returns
    -------
    torch.Tensor
        Covariance matrices of shape ``(batch, channels, channels)``.
    """

    if samples.ndim != 3:
        raise ValueError(f"Expected samples with shape (batch, length, channels); received {samples.shape}")

    centered = samples - samples.mean(dim=1, keepdim=True)
    cov = torch.matmul(centered.transpose(1, 2), centered)
    denom = max(samples.shape[1] - 1, 1)
    return cov / float(denom)


def stacked_to_complex(x: torch.Tensor) -> torch.Tensor:
    """Convert stacked real/imag representation back to a complex tensor."""

    if x.shape[-1] % 2 != 0:
        raise ValueError("Stacked real/imag tensor must have an even number of channels")
    half = x.shape[-1] // 2
    real = x[..., :half]
    imag = x[..., half:]
    return torch.complex(real, imag)


def complex_covariance(samples: torch.Tensor) -> torch.Tensor:
    """Compute covariance matrices for complex-valued sequences.

    Parameters
    ----------
    samples:
        Complex tensor shaped ``(batch, length, channels)``.

    Returns
    -------
    torch.Tensor
        Hermitian covariance matrices of shape ``(batch, channels, channels)``.
    """

    if samples.ndim != 3:
        raise ValueError(f"Expected samples with shape (batch, length, channels); received {samples.shape}")

    centered = samples - samples.mean(dim=1, keepdim=True)
    cov = torch.matmul(centered.conj().transpose(1, 2), centered)
    denom = max(samples.shape[1] - 1, 1)
    return cov / float(denom)
