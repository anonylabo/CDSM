#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Create a new samples_history batch that uses a single MC trajectory "
            "as the prediction (from samples_mc.pt), without re-sampling."
        )
    )
    p.add_argument(
        "--batch-dir",
        nargs="+",
        action="append",
        required=True,
        help="One or more samples_history/batch-... directories.",
    )
    p.add_argument(
        "--mc-index",
        type=int,
        default=0,
        help="Which MC trajectory to use (0-based).",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Optional output root. If set, output batch dirs are created under this path.",
    )
    p.add_argument(
        "--suffix",
        type=str,
        default=None,
        help="Suffix appended to each batch dir name (default: -mc{index}).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output batch directories.",
    )
    return p.parse_args()


def _iter_subdirs(batch_dir: Path) -> Iterable[Path]:
    return (p for p in sorted(batch_dir.iterdir()) if p.is_dir())


def _select_mc(samples_mc: torch.Tensor, mc_index: int) -> torch.Tensor:
    if samples_mc.ndim < 3:
        raise ValueError(f"samples_mc has unexpected shape {tuple(samples_mc.shape)}")
    if mc_index < 0 or mc_index >= samples_mc.size(0):
        raise IndexError(f"mc_index {mc_index} out of range for samples_mc with size {samples_mc.size(0)}")
    return samples_mc[mc_index]


def main() -> None:
    args = parse_args()
    mc_index = int(args.mc_index)
    suffix = args.suffix if args.suffix is not None else f"-mc{mc_index}"

    batch_dirs = [p for group in args.batch_dir for p in group]
    for batch_dir_str in batch_dirs:
        batch_dir = Path(batch_dir_str).expanduser().resolve()
        if not batch_dir.is_dir():
            raise FileNotFoundError(f"Batch dir not found: {batch_dir}")

        if args.out_root:
            out_batch = Path(args.out_root).expanduser().resolve() / f"{batch_dir.name}{suffix}"
        else:
            out_batch = batch_dir.with_name(f"{batch_dir.name}{suffix}")

        if out_batch.exists():
            if not args.overwrite:
                raise FileExistsError(f"Output batch already exists: {out_batch} (use --overwrite to replace)")
        out_batch.mkdir(parents=True, exist_ok=True)

        processed = 0
        skipped = 0
        for subdir in _iter_subdirs(batch_dir):
            samples_pt = subdir / "samples.pt"
            samples_mc_pt = subdir / "samples_mc.pt"
            if not samples_pt.exists() or not samples_mc_pt.exists():
                skipped += 1
                continue

            data = torch.load(samples_pt, map_location="cpu", weights_only=False)
            mc_data = torch.load(samples_mc_pt, map_location="cpu", weights_only=False)
            samples_mc = mc_data.get("samples_mc", None)
            if samples_mc is None:
                samples_mc = mc_data.get("samples", None)
            if samples_mc is None:
                skipped += 1
                continue

            selected = _select_mc(samples_mc, mc_index)
            if isinstance(data.get("samples"), torch.Tensor):
                selected = selected.to(data["samples"].dtype)
            data["samples"] = selected

            # If samples_fourier stores MC stacks, slice it too for consistency.
            sf = data.get("samples_fourier")
            if isinstance(sf, torch.Tensor) and sf.ndim >= 4 and sf.size(0) == samples_mc.size(0):
                data["samples_fourier"] = sf[mc_index]

            out_sub = out_batch / subdir.name
            out_sub.mkdir(parents=True, exist_ok=True)
            torch.save(data, out_sub / "samples.pt")

            processed += 1

        meta = {
            "source_batch": str(batch_dir),
            "mc_index": mc_index,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "processed_subdirs": processed,
            "skipped_subdirs": skipped,
        }
        with (out_batch / "mc_select_meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"[OK] {batch_dir} -> {out_batch} (processed={processed}, skipped={skipped})")


if __name__ == "__main__":
    main()
