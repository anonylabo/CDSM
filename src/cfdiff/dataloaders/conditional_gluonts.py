from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from cfdiff.utils.dataclasses import collate_diffusion_batch
from cfdiff.utils.window_processor import WindowProcessor, WindowProcessorConfig
from cfdiff.utils.windowing import compute_window_positions


def _resolve_split_file(root: Path, split: str) -> Path:
    candidates = [
        root / f"{split}.json",
        root / split / f"{split}.json",
        root / split / "data.json",
        root / split / "data.json.gz",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Could not locate {split} split under {root}")


def _iter_jsonl(path: Path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    else:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def _load_gluonts_like(path: Path) -> np.ndarray:
    series: List[List[float]] = []
    for obj in _iter_jsonl(path):
        target = obj.get("target")
        if target is None:
            raise ValueError(f"Missing 'target' field in {path}")
        series.append(list(map(float, target)))
    if not series:
        raise ValueError(f"Empty series in {path}")
    min_len = min(len(s) for s in series)
    series = [s[-min_len:] for s in series]
    return np.stack(series, axis=1).astype(np.float32)


def _build_windows(
    X: np.ndarray,
    window: int,
    stride: int,
    align_end: bool,
) -> np.ndarray:
    T, A = X.shape
    if T < window:
        raise ValueError(f"time length {T} shorter than window {window}")
    idx = compute_window_positions(T, window, stride, align_end=align_end)
    windows = np.stack([X[s:e, :] for (s, e) in idx], axis=0)
    return windows.astype(np.float32, copy=False)


def _extract_stage_windows(
    windows: np.ndarray,
    context_len: int,
    pred_len: int,
    stage: int,
) -> np.ndarray:
    if stage not in {0, 1, 2}:
        raise ValueError(f"Unsupported stage index {stage}; expected 0, 1, or 2.")
    window_len = context_len + pred_len * 3
    if windows.shape[1] != window_len:
        raise ValueError("Unexpected window length when extracting stage windows.")
    target_start = context_len + stage * pred_len
    context_start = max(0, target_start - context_len)
    target_end = target_start + pred_len
    context_end = context_start + context_len
    if context_start < 0 or target_end > window_len:
        raise ValueError("Window does not contain sufficient rows for requested stage.")
    context = windows[:, context_start:context_end, :]
    target = windows[:, target_start:target_end, :]
    return np.concatenate([context, target], axis=1)


def _forward_fill_numpy(
    array: np.ndarray,
    *,
    initial_last: float | np.ndarray = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Forward-fill non-finite values along time axis for a (T, A) array.

    Returns the filled array and the final per-asset "last" values to allow warm-start
    filling across splits (train tail -> test head).
    """

    arr = np.asarray(array, dtype=np.float32).copy()
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D array (T, A); got shape {arr.shape}")
    T, A = arr.shape

    if np.isscalar(initial_last):
        last = np.full(A, float(initial_last), dtype=arr.dtype)
    else:
        init = np.asarray(initial_last, dtype=arr.dtype)
        if init.shape != (A,):
            raise ValueError(f"Expected initial_last shape {(A,)}, got {init.shape}")
        last = init.copy()

    for t in range(T):
        row = arr[t]
        finite = np.isfinite(row)
        if finite.any():
            last[finite] = row[finite]
        missing = ~finite
        if missing.any():
            row[missing] = last[missing]
            arr[t] = row
    return arr, last


class _WindowDataset(Dataset):
    def __init__(
        self,
        windows: np.ndarray,
        context_len: int,
        pred_len: int,
        apply_fourier: bool = False,
        cov_window: Optional[int] = None,
        cov_eps: float = 1e-5,
        nan_policy: str = "raise",
        include_missing_mask: bool = False,
    ):
        self.windows = windows
        self.processor = WindowProcessor(
            WindowProcessorConfig(
                context_len=context_len,
                pred_len=pred_len,
                apply_fourier=apply_fourier,
                cov_window=int(cov_window) if cov_window else None,
                cov_eps=float(cov_eps),
                nan_policy=str(nan_policy),
                include_missing_mask=bool(include_missing_mask),
            )
        )

    def __len__(self) -> int:
        return self.windows.shape[0]

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        window = self.windows[i]
        return self.processor.transform(window)


class ConditionalGluonTSJsonDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 64,
        context_len: int = 120,
        pred_len: int = 30,
        stride: int = 1,
        val_ratio: float = 0.1,
        val_test_gap: int = -1,
        train_val_gap: int = -1,
        standardize: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
        jsonl_train: Optional[str] = None,
        jsonl_test: Optional[str] = None,
        fourier_transform: bool = False,
        estimate_sliding_cov: bool = False,
        sliding_cov_window: int = 8,
        sliding_cov_eps: float = 1e-5,
        align_tail_windows: bool = True,
        nan_policy: str = "raise",
        add_missing_mask: bool = False,
        ffill_warm_start: bool = False,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = int(batch_size)
        self.context_len_time = int(context_len)
        self.pred_len_time = int(pred_len)
        self.sequence_len_time = self.context_len_time + self.pred_len_time
        self.context_len = self.context_len_time
        self.pred_len = self.pred_len_time
        self.sequence_len = self.sequence_len_time
        self.stride = int(stride)
        self.val_ratio = float(val_ratio)
        self.train_val_gap = int(train_val_gap)
        # Optional gap (in window indices) to enforce between val and test.
        # If set to a negative value, the gap is computed automatically as
        # ceil((context_len+pred_len)/stride) - 1, which guarantees that the
        # earliest test window does not overlap in time with any validation
        # window given the current stride.
        self.val_test_gap = int(val_test_gap)
        self.standardize = bool(standardize)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.fourier_transform = bool(fourier_transform)
        self.estimate_sliding_cov = bool(estimate_sliding_cov)
        self.sliding_cov_window = int(sliding_cov_window)
        self.sliding_cov_eps = float(sliding_cov_eps)
        self.align_tail_windows = bool(align_tail_windows)
        self.nan_policy = str(nan_policy or "raise").lower()
        self.add_missing_mask = bool(add_missing_mask)
        self.ffill_warm_start = bool(ffill_warm_start)

        self.jsonl_train = Path(jsonl_train) if jsonl_train else None
        self.jsonl_test = Path(jsonl_test) if jsonl_test else None

        self.feature_mean: Optional[torch.Tensor] = None
        self.feature_std: Optional[torch.Tensor] = None
        self.ds_train: Optional[Dataset] = None
        self.ds_val: Optional[Dataset] = None
        self.ds_test: Optional[Dataset] = None
        # Keep standardised full series for baselines that operate on raw sequences
        self.train_series: Optional[np.ndarray] = None
        self.val_series: Optional[np.ndarray] = None
        self.test_series: Optional[np.ndarray] = None

    def prepare_data(self) -> None:
        if not self.data_dir.exists():
            raise FileNotFoundError(f"data_dir does not exist: {self.data_dir}")
        if self.jsonl_train is None:
            self.jsonl_train = _resolve_split_file(self.data_dir, "train")
        if self.jsonl_test is None:
            try:
                self.jsonl_test = _resolve_split_file(self.data_dir, "test")
            except FileNotFoundError:
                self.jsonl_test = None

    def setup(self, stage: Optional[str] = None) -> None:
        assert self.jsonl_train is not None
        Xtr = _load_gluonts_like(self.jsonl_train)
        Xt = None
        if self.jsonl_test is not None:
            Xt = _load_gluonts_like(self.jsonl_test)
            if Xt.shape[1] != Xtr.shape[1]:
                raise ValueError("train/test asset counts mismatch")

        if self.nan_policy in {"raise", "error"}:
            if not np.isfinite(Xtr).all():
                bad = int((~np.isfinite(Xtr)).sum())
                raise ValueError(f"Training split contains {bad} non-finite values but nan_policy='{self.nan_policy}'.")
            if Xt is not None and not np.isfinite(Xt).all():
                bad = int((~np.isfinite(Xt)).sum())
                raise ValueError(f"Test split contains {bad} non-finite values but nan_policy='{self.nan_policy}'.")
        else:
            # Treat any non-finite values as missing; WindowProcessor will apply the requested nan_policy per-window.
            Xtr = Xtr.copy()
            Xtr[~np.isfinite(Xtr)] = np.nan
            if Xt is not None:
                Xt = Xt.copy()
                Xt[~np.isfinite(Xt)] = np.nan

        if self.standardize:
            mu = np.nanmean(Xtr, axis=0)
            sigma = np.nanstd(Xtr, axis=0, ddof=1)
            mu = np.nan_to_num(mu, nan=0.0)
            sigma = np.nan_to_num(sigma, nan=1.0)
            sigma[sigma < 1e-8] = 1.0
            Xtr = (Xtr - mu) / sigma
            if Xt is not None:
                Xt = (Xt - mu) / sigma
            self.feature_mean = torch.from_numpy(mu.astype(np.float32))
            self.feature_std = torch.from_numpy(sigma.astype(np.float32))
        else:
            self.feature_mean = None
            self.feature_std = None

        if self.nan_policy in {"ffill", "forward_fill"} and self.ffill_warm_start:
            Xtr, last = _forward_fill_numpy(Xtr, initial_last=0.0)
            if Xt is not None:
                Xt, _ = _forward_fill_numpy(Xt, initial_last=last)

        # Expose full standardized series for baselines that operate on raw sequences
        self.train_series = Xtr
        self.test_series = Xt
        self.val_series = None  # optional; set below if available

        window_total = self.context_len + self.pred_len
        train_windows = _build_windows(
            Xtr,
            window=window_total,
            stride=self.stride,
            align_end=self.align_tail_windows,
        )
        if train_windows.shape[0] == 0:
            raise ValueError("No training windows generated; check context_len/pred_len/stride settings.")
        self.original_num_assets = Xtr.shape[1]
        train_time_len = Xtr.shape[0]

        val_ratio = float(self.val_ratio)

        # Branch 1: val from train windows when val_ratio < 0
        if val_ratio < 0:
            abs_ratio = abs(val_ratio)
            total_train = int(train_windows.shape[0])
            if total_train <= 0:
                raise ValueError("No training windows generated; cannot split train/val.")

            val_count = int(round(total_train * abs_ratio))
            min_val_windows = 2 if total_train >= 2 else 1
            val_count = max(min_val_windows, val_count)
            if val_count >= total_train:
                raise ValueError(
                    f"val_ratio={val_ratio} requests {val_count} validation windows but only {total_train} train windows exist. "
                    "Reduce |val_ratio| or increase data length."
                )
            if self.train_val_gap < 0:
                L = int(window_total)
                s = int(self.stride)
                gap = max(0, (L + s - 1) // s - 1)  # ceil(L/s) - 1
            else:
                gap = max(0, int(self.train_val_gap))

            train_end = total_train - val_count - gap
            if train_end <= 0:
                raise ValueError(
                    f"No training windows remain after applying train_val_gap (val_count={val_count}, "
                    f"gap={gap}, total_train_windows={total_train}). Adjust val_ratio/gap or stride/length."
                )

            val_start = total_train - val_count
            train_windows_final = train_windows[:train_end]
            val_windows = train_windows[val_start:]

            self.ds_train = _WindowDataset(
                train_windows_final,
                self.context_len,
                self.pred_len,
                apply_fourier=self.fourier_transform,
                cov_window=self.sliding_cov_window if self.estimate_sliding_cov else None,
                cov_eps=self.sliding_cov_eps,
                nan_policy=self.nan_policy,
                include_missing_mask=(self.nan_policy == "mask") or self.add_missing_mask,
            )
            self.ds_val = _WindowDataset(
                val_windows,
                self.context_len,
                self.pred_len,
                apply_fourier=self.fourier_transform,
                cov_window=self.sliding_cov_window if self.estimate_sliding_cov else None,
                cov_eps=self.sliding_cov_eps,
                nan_policy=self.nan_policy,
                include_missing_mask=(self.nan_policy == "mask") or self.add_missing_mask,
            )

            if Xt is None:
                raise ValueError(
                    "Test split is required for evaluation; provide test.json when using val_ratio<0."
                )
            self.val_series = Xt
            eval_windows = _build_windows(
                Xt,
                window=window_total,
                stride=self.stride,
                align_end=self.align_tail_windows,
            )
            if eval_windows.shape[0] == 0:
                raise ValueError("No test windows generated; check stride and lengths.")
            self.ds_test = _WindowDataset(
                eval_windows,
                self.context_len,
                self.pred_len,
                apply_fourier=self.fourier_transform,
                cov_window=self.sliding_cov_window if self.estimate_sliding_cov else None,
                cov_eps=self.sliding_cov_eps,
                nan_policy=self.nan_policy,
                include_missing_mask=(self.nan_policy == "mask") or self.add_missing_mask,
            )
        else:
            # Branch 2: val/test both from test split (existing behavior)
            self.ds_train = _WindowDataset(
                train_windows,
                self.context_len,
                self.pred_len,
                apply_fourier=self.fourier_transform,
                cov_window=self.sliding_cov_window if self.estimate_sliding_cov else None,
                cov_eps=self.sliding_cov_eps,
                nan_policy=self.nan_policy,
                include_missing_mask=(self.nan_policy == "mask") or self.add_missing_mask,
            )

            if Xt is None:
                raise ValueError(
                    "Test split is required so that validation/test windows do not overlap with training data."
                )
            self.val_series = Xt

            eval_windows = _build_windows(
                Xt,
                window=window_total,
                stride=self.stride,
                align_end=self.align_tail_windows,
            )
            if eval_windows.shape[0] == 0:
                raise ValueError("No evaluation windows generated from test split; check stride and lengths.")

            if val_ratio <= 0:
                val_count = 1
            else:
                val_count = int(round(eval_windows.shape[0] * val_ratio))

            # Need at least two validation windows so epoch-level Fourier metrics
            # and covariance diagnostics have enough samples. Otherwise Lightning
            # callbacks monitoring those keys crash when the metrics are absent.
            min_val_windows = 2 if eval_windows.shape[0] >= 2 else 1
            val_count = max(min_val_windows, val_count)
            val_count = min(val_count, eval_windows.shape[0])

            total_eval = int(eval_windows.shape[0])
            if total_eval <= 0:
                raise ValueError("No evaluation windows generated from test split; check stride and lengths.")

            # Validation block
            val_windows = eval_windows[:val_count]

            # Compute test start index with optional automatic gap
            if self.val_test_gap < 0:
                L = int(self.context_len + self.pred_len)
                s = int(self.stride)
                auto_gap = max(0, (L + s - 1) // s - 1)  # ceil(L/s) - 1
                gap = auto_gap
            else:
                gap = max(0, int(self.val_test_gap))

            test_start = min(total_eval, val_count + gap)
            if test_start >= total_eval:
                raise ValueError(
                    f"No test windows remain after applying val_test_gap (val_count={val_count}, "
                    f"gap={gap}, total_eval={total_eval}). Consider reducing val_ratio, reducing gap, "
                    f"or increasing test length/stride."
                )
            test_windows = eval_windows[test_start:]

            self.ds_val = _WindowDataset(
                val_windows,
                self.context_len,
                self.pred_len,
                apply_fourier=self.fourier_transform,
                cov_window=self.sliding_cov_window if self.estimate_sliding_cov else None,
                cov_eps=self.sliding_cov_eps,
                nan_policy=self.nan_policy,
                include_missing_mask=(self.nan_policy == "mask") or self.add_missing_mask,
            )
            if test_windows is not None and test_windows.shape[0] > 0:
                self.ds_test = _WindowDataset(
                    test_windows,
                    self.context_len,
                    self.pred_len,
                    apply_fourier=self.fourier_transform,
                    cov_window=self.sliding_cov_window if self.estimate_sliding_cov else None,
                    cov_eps=self.sliding_cov_eps,
                    nan_policy=self.nan_policy,
                    include_missing_mask=(self.nan_policy == "mask") or self.add_missing_mask,
                )
            else:
                self.ds_test = None

        self.num_assets = self.original_num_assets * (2 if self.fourier_transform else 1)
        if self.fourier_transform:
            self.context_len_freq = self.context_len_time // 2 + 1
            self.pred_len_freq = self.pred_len_time // 2 + 1
        else:
            self.context_len_freq = self.context_len_time
            self.pred_len_freq = self.pred_len_time
        self.sequence_len = self.context_len_freq + self.pred_len_freq
        self.sequence_len_freq = self.sequence_len
        self.num_training_steps = max(1, math.ceil(len(self.ds_train) / self.batch_size))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_diffusion_batch,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_diffusion_batch,
            drop_last=False,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.ds_test is None:
            return None
        return DataLoader(
            self.ds_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_diffusion_batch,
            drop_last=False,
        )

    @property
    def dataset_parameters(self) -> dict:
        return {
            "n_channels": self.num_assets,
            "sequence_len": self.sequence_len,
            "sequence_len_time": self.sequence_len_time,
            "context_len": self.context_len_freq,
            "context_len_time": self.context_len_time,
            "num_training_steps": self.num_training_steps,
            "pred_len": self.pred_len_freq,
            "pred_len_time": self.pred_len_time,
            "target_channels": self.original_num_assets,
        }

    @property
    def feature_mean_and_std(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.feature_mean is not None and self.feature_std is not None:
            return self.feature_mean, self.feature_std
        channels = self.num_assets
        return torch.zeros(channels, dtype=torch.float32), torch.ones(channels, dtype=torch.float32)
