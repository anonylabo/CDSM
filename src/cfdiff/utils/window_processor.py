from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Union

import numpy as np
import torch

from cfdiff.stats import compute_sliding_covariances
from cfdiff.utils.fourier import tensor_fft_realimag


TensorLike = Union[np.ndarray, torch.Tensor]


def _forward_fill_missing_2d(tensor: torch.Tensor, *, fill_value: float = 0.0) -> torch.Tensor:
    """Forward-fill non-finite values along the time axis (dim=0) for a 2D tensor."""

    if tensor.ndim != 2:
        raise ValueError(f"Expected a 2-D tensor (T, C); got {tuple(tensor.shape)}")

    out = tensor.clone()
    T, C = out.shape
    finite = torch.isfinite(out)

    for c in range(C):
        last = torch.as_tensor(fill_value, device=out.device, dtype=out.dtype)
        for t in range(T):
            if finite[t, c]:
                last = out[t, c]
            else:
                out[t, c] = last
    return out


def _interpolate_missing_2d(
    tensor: torch.Tensor,
    *,
    kind: str = "linear",
    fill_value: float = 0.0,
) -> torch.Tensor:
    """Interpolate non-finite values along the time axis (dim=0) for a 2D tensor."""

    if tensor.ndim != 2:
        raise ValueError(f"Expected a 2-D tensor (T, C); got {tuple(tensor.shape)}")

    device = tensor.device
    dtype = tensor.dtype
    arr = tensor.detach().to("cpu").numpy()
    arr = arr.astype(np.float64, copy=True)

    T, C = arr.shape
    x = np.arange(T, dtype=np.float64)
    kind = str(kind or "linear").lower()

    for c in range(C):
        col = arr[:, c]
        finite = np.isfinite(col)
        n_finite = int(finite.sum())
        if n_finite == T:
            continue
        if n_finite == 0:
            col[:] = float(fill_value)
            arr[:, c] = col
            continue
        if n_finite == 1:
            col[:] = float(col[finite][0])
            arr[:, c] = col
            continue

        known_x = x[finite]
        known_y = col[finite]

        missing = ~finite
        left_mask = missing & (x <= known_x[0])
        right_mask = missing & (x >= known_x[-1])
        mid_mask = missing & ~(left_mask | right_mask)

        if left_mask.any():
            col[left_mask] = float(known_y[0])
        if right_mask.any():
            col[right_mask] = float(known_y[-1])

        if mid_mask.any():
            if kind in {"cubic", "spline", "interp_cubic"} and n_finite >= 4:
                try:
                    from scipy.interpolate import CubicSpline  # type: ignore
                except Exception:
                    col[mid_mask] = np.interp(x[mid_mask], known_x, known_y)
                else:
                    cs = CubicSpline(known_x, known_y, extrapolate=False)
                    col[mid_mask] = cs(x[mid_mask])
                    # CubicSpline may return NaN at boundaries when extrapolate=False; fall back to linear there.
                    bad = ~np.isfinite(col[mid_mask])
                    if bad.any():
                        col[mid_mask] = np.interp(x[mid_mask], known_x, known_y)
            else:
                col[mid_mask] = np.interp(x[mid_mask], known_x, known_y)

        arr[:, c] = col

    out = torch.from_numpy(arr.astype(np.float32, copy=False)).to(device=device, dtype=dtype)
    return out


def _lift_time_mask_to_fourier(
    mask_time: torch.Tensor,
    *,
    freq_len: int,
) -> torch.Tensor:
    """
    Convert a time-domain observation mask (T, A) into a diffusion-space mask (F, 2A).

    Note: Fourier features are global mixtures of all time steps; there is no exact
    per-frequency "observed/unobserved" indicator. We use a stable proxy: the per-asset
    observed ratio over the time window, replicated across frequencies and real/imag
    channels.
    """

    if mask_time.ndim != 2:
        raise ValueError(f"Expected time mask shape (T, A); got {tuple(mask_time.shape)}")
    if freq_len <= 0:
        raise ValueError("freq_len must be positive")

    ratios = mask_time.to(dtype=torch.float32).mean(dim=0)  # (A,)
    ratios = ratios.clamp(0.0, 1.0)
    stacked = torch.cat([ratios, ratios], dim=0)  # (2A,)
    return stacked.unsqueeze(0).expand(freq_len, -1)


@dataclass(frozen=True)
class WindowProcessorConfig:
    """Configuration describing how to split and transform rolling windows."""

    context_len: int
    pred_len: int
    apply_fourier: bool = False
    cov_window: Optional[int] = None
    cov_eps: float = 1e-5
    dtype: torch.dtype = torch.float32
    nan_policy: str = "raise"  # one of: "raise", "zero", "ffill", "mask"
    include_missing_mask: bool = False


class WindowProcessor:
    """
    Stateless helper that exposes the preprocessing steps used inside
    ``GluonTSWindowDatamodule`` as a reusable component.
    """

    def __init__(self, config: WindowProcessorConfig) -> None:
        self.config = config

    def _to_tensor(self, window: TensorLike) -> torch.Tensor:
        if isinstance(window, torch.Tensor):
            tensor = window
        elif isinstance(window, np.ndarray):
            tensor = torch.from_numpy(window)
        else:
            raise TypeError(f"Unsupported window type {type(window)!r}")
        if tensor.dtype != self.config.dtype:
            tensor = tensor.to(self.config.dtype)
        return tensor

    def transform(self, window: TensorLike) -> Dict[str, torch.Tensor]:
        tensor = self._to_tensor(window)
        if tensor.ndim != 2:
            raise ValueError(f"Expected window tensor of shape (T, C); got {tuple(tensor.shape)}")

        # Identify missingness before any filling so downstream masks reflect the true pattern.
        tensor_raw = tensor
        obs_mask_time_full = torch.isfinite(tensor_raw)
        has_missing = not bool(obs_mask_time_full.all())

        tensor_filled = tensor_raw
        if has_missing:
            policy = str(self.config.nan_policy or "raise").lower()
            if policy in {"raise", "error"}:
                bad = int((~obs_mask_time_full).sum().item())
                raise ValueError(f"Window contains {bad} non-finite values but nan_policy='{self.config.nan_policy}'.")
            if policy in {"interp", "interpolate", "interp_linear"}:
                # Interpolate context/target separately to avoid leaking target information into the context.
                context_time = _interpolate_missing_2d(tensor_raw[: self.config.context_len, :], kind="linear")
                target_time = _interpolate_missing_2d(tensor_raw[-self.config.pred_len :, :], kind="linear")
                tensor_filled = None
            elif policy in {"interp_cubic", "cubic", "spline"}:
                context_time = _interpolate_missing_2d(tensor_raw[: self.config.context_len, :], kind="cubic")
                target_time = _interpolate_missing_2d(tensor_raw[-self.config.pred_len :, :], kind="cubic")
                tensor_filled = None
            elif policy in {"zero", "mask"}:
                tensor_filled = torch.nan_to_num(tensor_raw, nan=0.0, posinf=0.0, neginf=0.0)
            elif policy in {"ffill", "forward_fill"}:
                tensor_filled = _forward_fill_missing_2d(tensor_raw, fill_value=0.0)
            else:
                raise ValueError(f"Unsupported nan_policy='{self.config.nan_policy}'")

        window_len = tensor_raw.shape[0]
        if self.config.pred_len > window_len:
            raise ValueError("pred_len cannot exceed window length.")

        if tensor_filled is not None:
            context_time = tensor_filled[: self.config.context_len, :].clone().contiguous()
            target_time = tensor_filled[-self.config.pred_len :, :].clone().contiguous()
        else:
            # Interpolation policies constructed context_time/target_time directly.
            context_time = context_time.clone().contiguous()  # type: ignore[has-type]
            target_time = target_time.clone().contiguous()  # type: ignore[has-type]
        target_clean = tensor_raw[-self.config.pred_len :, :].clone().contiguous()

        sample: Dict[str, torch.Tensor] = {}
        if self.config.apply_fourier:
            context = tensor_fft_realimag(context_time.unsqueeze(0)).squeeze(0)
            target = tensor_fft_realimag(target_time.unsqueeze(0)).squeeze(0)
        else:
            context = context_time
            target = target_time

        sample["context"] = context.contiguous()
        sample["target"] = target.contiguous()
        sample["context_time"] = context_time
        sample["target_time"] = target_time
        sample["target_clean"] = target_clean

        if self.config.include_missing_mask:
            context_mask_time = obs_mask_time_full[: self.config.context_len, :].clone().contiguous()
            target_mask_time = obs_mask_time_full[-self.config.pred_len :, :].clone().contiguous()
            sample["context_mask_time"] = context_mask_time
            sample["target_mask_time"] = target_mask_time

            if self.config.apply_fourier:
                sample["context_mask"] = _lift_time_mask_to_fourier(context_mask_time, freq_len=context.size(0))
                sample["target_mask"] = _lift_time_mask_to_fourier(target_mask_time, freq_len=target.size(0))
            else:
                sample["context_mask"] = context_mask_time
                sample["target_mask"] = target_mask_time

        if self.config.cov_window and self.config.cov_window > 0:
            cov, mask = compute_sliding_covariances(target_time, self.config.cov_window, eps=self.config.cov_eps)
            sample["cov"] = cov
            sample["cov_mask"] = mask

        return sample

    def transform_batch(self, windows: Sequence[TensorLike]) -> Iterable[Dict[str, torch.Tensor]]:
        for window in windows:
            yield self.transform(window)
