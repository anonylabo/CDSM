from __future__ import annotations

import numpy as np
import torch


def _check_even_features(size: int) -> None:
    if size % 2 != 0:
        raise ValueError(f"Stacked real/imag representation requires an even feature dimension; received {size}.")


def numpy_fft_realimag(arr: np.ndarray) -> np.ndarray:
    """Apply real FFT along axis=1 and stack real/imaginary parts."""

    rfft = np.fft.rfft(arr, axis=1)
    real = rfft.real
    imag = rfft.imag
    return np.concatenate([real, imag], axis=-1).astype(np.float32, copy=False)


def numpy_ifft_realimag(arr: np.ndarray, signal_len: int) -> np.ndarray:
    """Inverse real FFT for the stacked representation (NumPy)."""

    if signal_len <= 0:
        raise ValueError(f"signal_len must be positive; received {signal_len}.")
    _check_even_features(arr.shape[-1])
    half = arr.shape[-1] // 2
    complex_vals = arr[..., :half] + 1j * arr[..., half:]
    inv = np.fft.irfft(complex_vals, n=signal_len, axis=1)
    return inv.astype(np.float32, copy=False)


def tensor_fft_realimag(x: torch.Tensor, *, dim: int = 1) -> torch.Tensor:
    """Real FFT along ``dim`` returning stacked real/imaginary parts."""

    rfft = torch.fft.rfft(x, dim=dim)
    return torch.cat([rfft.real, rfft.imag], dim=-1)


def tensor_ifft_realimag(x: torch.Tensor, *, signal_len: int, dim: int = 1) -> torch.Tensor:
    """Inverse real FFT for stacked real/imaginary tensor representation."""

    if signal_len <= 0:
        raise ValueError(f"signal_len must be positive; received {signal_len}.")
    _check_even_features(x.shape[-1])
    half = x.shape[-1] // 2
    real = x[..., :half]
    imag = x[..., half:].clone()
    imag[..., 0, :] = 0.0
    if signal_len % 2 == 0 and imag.size(-2) > 1:
        imag[..., -1, :] = 0.0
    complex_tensor = torch.complex(real, imag)
    inv = torch.fft.irfft(complex_tensor, n=signal_len, dim=dim)
    return inv
