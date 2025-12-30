from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from cfdiff.utils.window_processor import WindowProcessor, WindowProcessorConfig
from cfdiff.utils.windowing import compute_window_positions
from fdiff.utils.dataclasses import DiffusableBatch, collate_batch


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
) -> tuple[np.ndarray, np.ndarray]:
    T, _ = X.shape
    if T < window:
        raise ValueError(f"time length {T} shorter than window {window}")
    idx = compute_window_positions(T, window, stride, align_end=align_end)
    windows = np.stack([X[s:e, :] for (s, e) in idx], axis=0)
    starts = np.array([s for (s, _) in idx], dtype=np.int64)
    return windows.astype(np.float32, copy=False), starts


def _extract_stage_windows(
    windows: np.ndarray,
    context_len: int,
    pred_len: int,
    stage: int,
) -> np.ndarray:
    if stage not in {0, 1, 2}:
        raise ValueError(f"Unsupported stage index {stage}; expected 0, 1, or 2.")
    window_len = context_len + 3 * pred_len
    if windows.shape[1] != window_len:
        raise ValueError("Unexpected window length when extracting stage windows.")
    target_start = context_len + stage * pred_len
    context_start = max(0, target_start - context_len)
    target_end = target_start + pred_len
    context_end = context_start + context_len
    if target_end > window_len or context_start < 0:
        raise ValueError("Window does not contain sufficient context/target rows for stage extraction.")
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
        nan_policy: str = "raise",
        include_missing_mask: bool = False,
    ):
        self.windows = windows
        self.processor = WindowProcessor(
            WindowProcessorConfig(
                context_len=context_len,
                pred_len=pred_len,
                apply_fourier=apply_fourier,
                cov_window=None,
                nan_policy=str(nan_policy),
                include_missing_mask=bool(include_missing_mask),
            )
        )

    def __len__(self) -> int:
        return self.windows.shape[0]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        batch = self.processor.transform(self.windows[index])
        return {
            "X": batch["target"],
            "X_mask": batch.get("target_mask"),
            "target_time": batch["target_time"],
            # keep additional fields for evaluation parity with the conditional pipeline
            "context": batch["context"],
            "context_time": batch["context_time"],
            "target": batch["target"],
            "target_clean": batch["target_clean"],
        }


class UnconditionalGluonTSJsonDatamodule(pl.LightningDataModule):
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
        self.stride = int(stride)
        self.val_ratio = float(val_ratio)
        self.train_val_gap = int(train_val_gap)
        # Optional gap (in window indices) to enforce between val and test.
        # If negative, compute automatically as ceil((context_len+pred_len)/stride) - 1
        # to ensure no temporal overlap between validation and test windows.
        self.val_test_gap = int(val_test_gap)
        self.standardize = bool(standardize)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.fourier_transform = bool(fourier_transform)
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
        # Standardised full series (time, assets) for raw-series baselines
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

        self.train_series = Xtr
        self.test_series = Xt

        window_total = self.context_len_time + self.pred_len_time
        train_windows, train_starts = _build_windows(
            Xtr,
            window=window_total,
            stride=self.stride,
            align_end=self.align_tail_windows,
        )
        if train_windows.shape[0] == 0:
            raise ValueError("No training windows generated. Check context_len/pred_len/stride configuration.")

        val_ratio = float(self.val_ratio)
        if val_ratio < 0:
            abs_ratio = abs(val_ratio)
            total_train = int(train_windows.shape[0])
            if total_train <= 0:
                raise ValueError("No training windows generated; cannot split train/val.")

            val_count = int(round(total_train * abs_ratio))
            val_count = max(1, val_count)
            if val_count >= total_train:
                raise ValueError(
                    f"val_ratio={val_ratio} requests {val_count} validation windows but only {total_train} train windows exist. "
                    "Reduce |val_ratio| or increase data length."
                )
            if self.train_val_gap < 0:
                L = int(window_total)
                s = int(self.stride)
                gap = max(0, (L + s - 1) // s - 1)
            else:
                gap = max(0, int(self.train_val_gap))

            train_end = total_train - val_count - gap
            if train_end <= 0:
                raise ValueError(
                    f"No training windows remain after applying train_val_gap (val_count={val_count}, gap={gap}, "
                    f"total_train_windows={total_train}). Adjust val_ratio/gap or stride/length."
                )
            val_start = total_train - val_count

            self.ds_train = _WindowDataset(
                train_windows[:train_end],
                context_len=self.context_len_time,
                pred_len=self.pred_len_time,
                apply_fourier=self.fourier_transform,
                nan_policy=self.nan_policy,
                include_missing_mask=(self.nan_policy == "mask") or self.add_missing_mask,
            )
            self.ds_val = _WindowDataset(
                train_windows[val_start:],
                context_len=self.context_len_time,
                pred_len=self.pred_len_time,
                apply_fourier=self.fourier_transform,
                nan_policy=self.nan_policy,
                include_missing_mask=(self.nan_policy == "mask") or self.add_missing_mask,
            )
            # Validation series slice (from training split)
            start_val = train_starts[val_start]
            end_val = start_val + window_total + (val_count - 1) * self.stride
            self.val_series = Xtr[start_val:end_val]

            if Xt is None:
                raise ValueError("Test split is required so that validation/test data remain disjoint from training.")
            eval_windows, eval_starts = _build_windows(
                Xt,
                window=window_total,
                stride=self.stride,
                align_end=self.align_tail_windows,
            )
            if eval_windows.shape[0] == 0:
                raise ValueError("No evaluation windows generated from test split. Check context_len/pred_len/stride.")
            self.ds_test = _WindowDataset(
                eval_windows,
                context_len=self.context_len_time,
                pred_len=self.pred_len_time,
                apply_fourier=self.fourier_transform,
                nan_policy=self.nan_policy,
                include_missing_mask=(self.nan_policy == "mask") or self.add_missing_mask,
            )
        else:
            self.ds_train = _WindowDataset(
                train_windows,
                context_len=self.context_len_time,
                pred_len=self.pred_len_time,
                apply_fourier=self.fourier_transform,
                nan_policy=self.nan_policy,
                include_missing_mask=(self.nan_policy == "mask") or self.add_missing_mask,
            )

            if Xt is None:
                raise ValueError("Test split is required so that validation/test data remain disjoint from training.")

            eval_windows, eval_starts = _build_windows(
                Xt,
                window=window_total,
                stride=self.stride,
                align_end=self.align_tail_windows,
            )
            if eval_windows.shape[0] == 0:
                raise ValueError("No evaluation windows generated from test split. Check context_len/pred_len/stride.")

            if val_ratio <= 0:
                val_count = 1
            else:
                val_count = int(round(eval_windows.shape[0] * val_ratio))
                val_count = max(1, min(val_count, eval_windows.shape[0]))

            total_eval = int(eval_windows.shape[0])
            if total_eval <= 0:
                raise ValueError("No evaluation windows generated from test split. Check context_len/pred_len/stride.")

            val_windows = eval_windows[:val_count]

            if self.val_test_gap < 0:
                L = int(self.context_len_time + self.pred_len_time)
                s = int(self.stride)
                gap = max(0, (L + s - 1) // s - 1)  # ceil(L/s) - 1
            else:
                gap = max(0, int(self.val_test_gap))

            test_start = min(total_eval, val_count + gap)
            if test_start >= total_eval:
                raise ValueError(
                    f"No test windows remain after applying val_test_gap (val_count={val_count}, "
                    f"gap={gap}, total_eval={total_eval}). Adjust val_ratio/gap/stride or dataset length."
                )
            test_windows = eval_windows[test_start:]

            self.ds_val = _WindowDataset(
                val_windows,
                context_len=self.context_len_time,
                pred_len=self.pred_len_time,
                apply_fourier=self.fourier_transform,
                nan_policy=self.nan_policy,
                include_missing_mask=(self.nan_policy == "mask") or self.add_missing_mask,
            )
            if test_windows is not None and test_windows.shape[0] > 0:
                self.ds_test = _WindowDataset(
                    test_windows,
                    context_len=self.context_len_time,
                    pred_len=self.pred_len_time,
                    apply_fourier=self.fourier_transform,
                    nan_policy=self.nan_policy,
                    include_missing_mask=(self.nan_policy == "mask") or self.add_missing_mask,
                )
            else:
                self.ds_test = None

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate,
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
            collate_fn=self._collate,
        )

    def _collate(self, batch: List[Dict[str, torch.Tensor]]) -> DiffusableBatch:
        return collate_batch(batch)

    @property
    def dataset_parameters(self) -> Dict[str, int]:
        sample = self.ds_train[0]
        return {
            "n_channels": sample["X"].shape[1],
            "sequence_len": sample["X"].shape[0],
            "sequence_len_time": self.context_len_time + self.pred_len_time,
            "context_len": self.context_len_time,
            "context_len_time": self.context_len_time,
            "pred_len": sample["X"].shape[0],
            "pred_len_time": self.pred_len_time,
        }
