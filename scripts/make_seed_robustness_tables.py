#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from plot_correlation_mean import _stage_indices, aggregate_model

ROOT = Path(__file__).resolve().parents[1]


def format_sig(value: float, sig: int) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "nan"
    if value == 0:
        return f"{0:.{sig}f}"
    v = abs(float(value))
    exp = math.floor(math.log10(v))
    decimals = sig - 1 - exp
    if decimals >= 0:
        return f"{value:.{decimals}f}"
    q = 10 ** (-decimals)
    return f"{round(value / q) * q:.0f}"


def format_pm(mean: float, std: float, sig: int) -> str:
    return f"{format_sig(mean, sig)} $\\pm$ {format_sig(std, sig)}"


def format_sci(value: float, sig: int) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "nan"
    if value == 0:
        return f"{0:.{sig}f}"
    fmt = f"{float(value):.{max(sig - 1, 0)}e}"
    mant, exp = fmt.split("e")
    return f"{mant}e{int(exp)}"


def format_pm_std_sci(mean: float, std: float, sig: int) -> str:
    return f"{format_sig(mean, sig)} $\\pm$ {format_sci(std, sig)}"


def load_payload(samples_pt: Path) -> Dict[str, np.ndarray]:
    data = torch.load(samples_pt, map_location="cpu", weights_only=False)
    payload = {
        "pred": data["samples"].cpu().numpy(),
        "truth": data["truth"].cpu().numpy(),
        "context": data.get("context").cpu().numpy() if isinstance(data.get("context"), torch.Tensor) else None,
    }
    stage_counts = data.get("window_stage_counts")
    if stage_counts is not None:
        payload["stage_counts"] = {k: int(v) for k, v in stage_counts.items()}
    return payload


def compute_metric_for_batch(
    run_dir: Path,
    batch_tag: str,
    metric_name: str,
    use_correlation: bool,
    asset_indices: List[int],
    stage: str,
    use_augmented_cov: bool,
) -> Tuple[float, float]:
    """
    Compute mean/std across repeat subdirectories under a batch.
    Returns (mean, std) of pred_minus_truth.
    """
    batch_dir = run_dir / "samples_history" / batch_tag
    if not batch_dir.is_dir():
        raise FileNotFoundError(f"Batch dir not found: {batch_dir}")

    subdirs = sorted([d for d in batch_dir.iterdir() if d.is_dir()])
    if not subdirs:
        raise FileNotFoundError(f"No repeat subdirs under {batch_dir}")

    first_payload = load_payload(subdirs[0] / "samples.pt")
    counts = first_payload.get("stage_counts", {})
    ref_indices = _stage_indices(counts, stage) if counts else list(range(first_payload["pred"].shape[0]))

    values: List[float] = []
    for sub in subdirs:
        samples_pt = sub / "samples.pt"
        if not samples_pt.exists():
            continue
        payload = load_payload(samples_pt)
        _, _, mets = aggregate_model(
            ("time", "conditional"),
            payload,
            ref_indices,
            asset_indices,
            metric_name,
            stage,
            use_correlation,
            use_augmented_cov=use_augmented_cov,
        )
        values.append(float(mets["pred_minus_truth"]))

    if not values:
        raise RuntimeError(f"No usable samples.pt under {batch_dir}")
    return float(np.mean(values)), float(np.std(values))


def collect_values_for_batch(
    run_dir: Path,
    batch_tag: str,
    metric_name: str,
    use_correlation: bool,
    asset_indices: List[int],
    stage: str,
    use_augmented_cov: bool,
) -> List[float]:
    """
    Collect repeat-level values under a batch (no aggregation).
    """
    batch_dir = run_dir / "samples_history" / batch_tag
    if not batch_dir.is_dir():
        raise FileNotFoundError(f"Batch dir not found: {batch_dir}")

    subdirs = sorted([d for d in batch_dir.iterdir() if d.is_dir()])
    if not subdirs:
        raise FileNotFoundError(f"No repeat subdirs under {batch_dir}")

    first_payload = load_payload(subdirs[0] / "samples.pt")
    counts = first_payload.get("stage_counts", {})
    ref_indices = _stage_indices(counts, stage) if counts else list(range(first_payload["pred"].shape[0]))

    values: List[float] = []
    for sub in subdirs:
        samples_pt = sub / "samples.pt"
        if not samples_pt.exists():
            continue
        payload = load_payload(samples_pt)
        _, _, mets = aggregate_model(
            ("time", "conditional"),
            payload,
            ref_indices,
            asset_indices,
            metric_name,
            stage,
            use_correlation,
            use_augmented_cov=use_augmented_cov,
        )
        values.append(float(mets["pred_minus_truth"]))
    if not values:
        raise RuntimeError(f"No usable samples.pt under {batch_dir}")
    return values


def mean_std(values: List[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr))


def write_table(path: Path, rows: List[List[str]], headers: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = "l" + "c" * (len(headers) - 1)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"\\begin{{tabular}}{{{cols}}}\n")
        f.write(" \\toprule\n")
        f.write(" \\ & ".join(headers) + " \\\\\n")
        f.write(" \\midrule\n")
        for row in rows:
            f.write(" \\ & ".join(row) + " \\\\\n")
        f.write(" \\bottomrule\n\\end{tabular}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute seed-robustness tables (cov/corr) with single-trajectory samples."
    )
    p.add_argument("--sig-digits", type=int, default=4, help="Significant digits for formatting.")
    p.add_argument("--stage", default="all", choices=["all", "val", "test"], help="Stage split.")
    p.add_argument("--out-cov", type=Path, default=Path("assets/table_seed_robustness_cov_lambda0.tex"))
    p.add_argument("--out-corr", type=Path, default=Path("assets/table_seed_robustness_corr_lambda0.tex"))
    p.add_argument(
        "--out-cov-by-seed",
        type=Path,
        default=None,
        help="Optional: per-seed cov table (no pooling across seeds).",
    )
    p.add_argument(
        "--out-corr-by-seed",
        type=Path,
        default=None,
        help="Optional: per-seed corr table (no pooling across seeds).",
    )
    p.add_argument(
        "--lambda-tag",
        type=str,
        default=None,
        help="If set (e.g. 5e-4, 5e-3), auto-scan outputs for that lambda.",
    )
    return p.parse_args()


def resolve_run(path_str: str | Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (ROOT / p).resolve()


def find_latest_run(
    roots: List[Path],
    subdir: str,
    patterns: List[str],
    lambda_tag: str,
) -> Path | None:
    token = f"lambda{lambda_tag}"
    candidates: List[Path] = []
    for root in roots:
        base = root / subdir
        if not base.is_dir():
            continue
        for d in base.iterdir():
            if not d.is_dir():
                continue
            name = d.name
            if token not in name:
                continue
            if any(pat in name for pat in patterns):
                candidates.append(d)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.name)


def find_batch_tag(run_dir: Path | None) -> str | None:
    if run_dir is None:
        return None
    history = run_dir / "samples_history"
    if not history.is_dir():
        return None
    batches = sorted([p for p in history.iterdir() if p.is_dir() and p.name.startswith("batch-")])
    if not batches:
        return None
    for b in batches:
        if b.name.endswith("-mc0"):
            return b.name
    return batches[0].name


def as_rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def build_datasets_for_lambda(lambda_tag: str) -> List[Dict[str, object]]:
    roots = [ROOT / "outputs", ROOT / "outputs_local"]
    specs = [
        {
            "label": "Exchange",
            "assets": 8,
            "patterns": ["FX8_30_15val0.3train0.8"],
        },
        {
            "label": "FX29",
            "assets": 29,
            "patterns": ["FX30_ecb_45_15val0.3train0.8", "FX30_45_15val0.3train0.8"],
        },
        {
            "label": "Industry49",
            "assets": 49,
            "patterns": ["ind49_30_15val0.3train0.8"],
        },
        {
            "label": "iShares14",
            "assets": 14,
            "patterns": ["ishares14_30_15_val0.3train0.8"],
        },
    ]

    datasets: List[Dict[str, object]] = []
    for spec in specs:
        patterns = list(spec["patterns"])
        time_run = find_latest_run(roots, "time/conditional", patterns, lambda_tag)
        fourier_run = find_latest_run(roots, "fourier/conditional", patterns, lambda_tag)
        time_batch = find_batch_tag(time_run)
        fourier_batch = find_batch_tag(fourier_run)
        datasets.append(
            {
                "label": spec["label"],
                "assets": spec["assets"],
                "time_seed42": (as_rel(time_run), time_batch) if time_run and time_batch else None,
                "time_seeds": [],
                "fourier_seed42": (as_rel(fourier_run), fourier_batch) if fourier_run and fourier_batch else None,
                "fourier_seeds": [],
            }
        )
    return datasets


def main() -> None:
    args = parse_args()
    stage = args.stage

    if args.lambda_tag:
        datasets = build_datasets_for_lambda(args.lambda_tag)
    else:
        datasets = [
            {
                "label": "Exchange",
                "assets": 8,
                "time_seed42": (
                    "outputs/time/conditional/20260129-223951_FX8_30_15val0.3train0.8_lambda0",
                    "batch-20260130-075051-mc0",
                ),
                "time_seeds": [
                    (
                        "outputs/time/conditional/20260201-064735_FX8_30_15val0.3train0.8_lambda0_train_random_seed12",
                        "batch-20260201-194302-mc0",
                    ),
                    (
                        "outputs/time/conditional/20260201-064843_FX8_30_15val0.3train0.8_lambda0_train_random_seed22",
                        "batch-20260201-194518-mc0",
                    ),
                    (
                        "outputs/time/conditional/20260201-064936_FX8_30_15val0.3train0.8_lambda0_train_random_seed32",
                        "batch-20260201-194755-mc0",
                    ),
                ],
                "fourier_seed42": (
                    "outputs/fourier/conditional/20260129-224014_FX8_30_15val0.3train0.8_lambda0",
                    "batch-20260130-075153-mc0",
                ),
                "fourier_seeds": [
                    (
                        "outputs/fourier/conditional/20260201-064803_FX8_30_15val0.3train0.8_lambda0_train_random_seed12",
                        "batch-20260201-194401-mc0",
                    ),
                    (
                        "outputs/fourier/conditional/20260201-064905_FX8_30_15val0.3train0.8_lambda0_train_random_seed22",
                        "batch-20260201-194640-mc0",
                    ),
                    (
                        "outputs/fourier/conditional/20260201-064951_FX8_30_15val0.3train0.8_lambda0_train_random_seed32",
                        "batch-20260201-194903-mc0",
                    ),
                ],
            },
            {
                "label": "FX29",
                "assets": 29,
                "time_seed42": (
                    "outputs/time/conditional/20260129-224047_FX30_ecb_45_15val0.3train0.8_lambda0",
                    "batch-20260130-075313-mc0",
                ),
                "time_seeds": [
                    (
                        "outputs/time/conditional/20260201-065046_FX30_ecb_45_15val0.3train0.8_lambda0_train_random_seed12",
                        "batch-20260201-195024-mc0",
                    ),
                    (
                        "outputs/time/conditional/20260201-065152_FX30_ecb_45_15val0.3train0.8_lambda0_train_random_seed22",
                        "batch-20260201-195310-mc0",
                    ),
                    (
                        "outputs/time/conditional/20260201-065234_FX30_ecb_45_15val0.3train0.8_lambda0_train_random_seed32",
                        "batch-20260201-195644-mc0",
                    ),
                ],
                "fourier_seed42": (
                    "outputs/fourier/conditional/20260129-224103_FX30_ecb_45_15val0.3train0.8_lambda0",
                    "batch-20260130-075442-mc0",
                ),
                "fourier_seeds": [
                    (
                        "outputs/fourier/conditional/20260201-065102_FX30_ecb_45_15val0.3train0.8_lambda0_train_random_seed12",
                        "batch-20260201-195140-mc0",
                    ),
                    (
                        "outputs/fourier/conditional/20260201-065207_FX30_ecb_45_15val0.3train0.8_lambda0_train_random_seed22",
                        "batch-20260201-195451-mc0",
                    ),
                    (
                        "outputs/fourier/conditional/20260201-065253_FX30_ecb_45_15val0.3train0.8_lambda0_train_random_seed32",
                        "batch-20260201-195740-mc0",
                    ),
                ],
            },
            {
                "label": "Industry49",
                "assets": 49,
                "time_seed42": (
                    "outputs/time/conditional/20260129-224207_ind49_30_15val0.3train0.8_lambda0",
                    "batch-20260130-075618-mc0",
                ),
                "time_seeds": [
                    (
                        "outputs_local/time/conditional/20260202-160436_ind49_30_15val0.3train0.8_lambda0_train_random_seed12",
                        "batch-20260203-064607-mc0",
                    ),
                    (
                        "outputs_local/time/conditional/20260202-160514_ind49_30_15val0.3train0.8_lambda0_train_random_seed22",
                        "batch-20260203-064836-mc0",
                    ),
                    (
                        "outputs_local/time/conditional/20260202-160558_ind49_30_15val0.3train0.8_lambda0_train_random_seed32",
                        "batch-20260203-065039-mc0",
                    ),
                ],
                "fourier_seed42": (
                    "outputs/fourier/conditional/20260129-224232_ind49_30_15val0.3train0.8_lambda0",
                    "batch-20260130-075727-mc0",
                ),
                "fourier_seeds": [
                    (
                        "outputs_local/fourier/conditional/20260202-160447_ind49_30_15val0.3train0.8_lambda0_train_random_seed12",
                        "batch-20260203-064719-mc0",
                    ),
                    (
                        "outputs_local/fourier/conditional/20260202-160537_ind49_30_15val0.3train0.8_lambda0_train_random_seed22",
                        "batch-20260203-064925-mc0",
                    ),
                    (
                        "outputs_local/fourier/conditional/20260202-160612_ind49_30_15val0.3train0.8_lambda0_train_random_seed32",
                        "batch-20260203-065132-mc0",
                    ),
                ],
            },
            {
                "label": "iShares14",
                "assets": 14,
                "time_seed42": (
                    "outputs/time/conditional/20260129-224313_ishares14_30_15_val0.3train0.8_lambda0",
                    "batch-20260130-075904-mc0",
                ),
                "time_seeds": [
                    (
                        "outputs_local/time/conditional/20260202-160654_ishares14_30_15_val0.3train0.8_lambda0_train_random_seed12",
                        "batch-20260203-065303-mc0",
                    ),
                    (
                        "outputs_local/time/conditional/20260202-160723_ishares14_30_15_val0.3train0.8_lambda0_train_random_seed22",
                        "batch-20260203-065538-mc0",
                    ),
                    (
                        "outputs_local/time/conditional/20260202-160758_ishares14_30_15_val0.3train0.8_lambda0_train_random_seed32",
                        "batch-20260203-065800-mc0",
                    ),
                ],
                "fourier_seed42": (
                    "outputs/fourier/conditional/20260129-224344_ishares14_30_15_val0.3train0.8_lambda0",
                    "batch-20260130-080017-mc0",
                ),
                "fourier_seeds": [
                    (
                        "outputs_local/fourier/conditional/20260202-160704_ishares14_30_15_val0.3train0.8_lambda0_train_random_seed12",
                        "batch-20260203-065414-mc0",
                    ),
                    (
                        "outputs_local/fourier/conditional/20260202-160734_ishares14_30_15_val0.3train0.8_lambda0_train_random_seed22",
                        "batch-20260203-065644-mc0",
                    ),
                    (
                        "outputs_local/fourier/conditional/20260202-160808_ishares14_30_15_val0.3train0.8_lambda0_train_random_seed32",
                        "batch-20260203-065855-mc0",
                    ),
                ],
            },
        ]

    cov_rows: List[List[str]] = []
    corr_rows: List[List[str]] = []
    headers = ["Dataset"] + [ds["label"] for ds in datasets]

    cov_rows = [
        ["CDSM (Temporal)"],
        ["CDSM (Spectral)"],
    ]
    corr_rows = [
        ["CDSM (Temporal)"],
        ["CDSM (Spectral)"],
    ]

    def iter_runs(ds: Dict[str, object], key_main: str, key_seeds: str) -> List[Tuple[str, str]]:
        runs: List[Tuple[str, str]] = []
        main = ds.get(key_main)
        if main:
            runs.append(main)
        runs.extend(ds.get(key_seeds, []))
        return runs

    for ds in datasets:
        asset_indices = list(range(ds["assets"]))

        t_pool_vals: List[float] = []
        for run, batch in iter_runs(ds, "time_seed42", "time_seeds"):
            t_pool_vals.extend(
                collect_values_for_batch(
                    resolve_run(run), batch, "matrix_cov_fro", False, asset_indices, stage, True
                )
            )
        if not t_pool_vals:
            print(f"[WARN] Missing time seeds for {ds['label']}; leaving blanks.", file=sys.stderr)

        f_pool_vals: List[float] = []
        for run, batch in iter_runs(ds, "fourier_seed42", "fourier_seeds"):
            f_pool_vals.extend(
                collect_values_for_batch(
                    resolve_run(run), batch, "matrix_cov_fro", False, asset_indices, stage, True
                )
            )
        if not f_pool_vals:
            print(f"[WARN] Missing fourier seeds for {ds['label']}; leaving blanks.", file=sys.stderr)

        cov_rows[0].append(format_pm(*mean_std(t_pool_vals), args.sig_digits) if t_pool_vals else "--")
        cov_rows[1].append(format_pm(*mean_std(f_pool_vals), args.sig_digits) if f_pool_vals else "--")

        t_pool_vals = []
        for run, batch in iter_runs(ds, "time_seed42", "time_seeds"):
            t_pool_vals.extend(
                collect_values_for_batch(
                    resolve_run(run), batch, "matrix_corr_fro", True, asset_indices, stage, True
                )
            )
        if not t_pool_vals:
            print(f"[WARN] Missing time seeds for {ds['label']}; leaving blanks.", file=sys.stderr)

        f_pool_vals = []
        for run, batch in iter_runs(ds, "fourier_seed42", "fourier_seeds"):
            f_pool_vals.extend(
                collect_values_for_batch(
                    resolve_run(run), batch, "matrix_corr_fro", True, asset_indices, stage, True
                )
            )
        if not f_pool_vals:
            print(f"[WARN] Missing fourier seeds for {ds['label']}; leaving blanks.", file=sys.stderr)

        corr_rows[0].append(format_pm(*mean_std(t_pool_vals), args.sig_digits) if t_pool_vals else "--")
        corr_rows[1].append(format_pm(*mean_std(f_pool_vals), args.sig_digits) if f_pool_vals else "--")

    write_table(args.out_cov, cov_rows, headers)
    write_table(args.out_corr, corr_rows, headers)
    if args.out_cov_by_seed or args.out_corr_by_seed:
        seed_labels = ["seed42", "seed12", "seed22", "seed32"]
        cov_seed_rows: List[List[str]] = []
        corr_seed_rows: List[List[str]] = []

        def safe_metric(run: str, batch: str, metric: str, use_corr: bool, assets: List[int]) -> str:
            try:
                mean, std = compute_metric_for_batch(
                    resolve_run(run), batch, metric, use_corr, assets, stage, True
                )
                return format_pm_std_sci(mean, std, args.sig_digits)
            except Exception as exc:  # pragma: no cover - best-effort robustness
                print(f"[WARN] {exc}", file=sys.stderr)
                return "--"

        for model_name, key_time, key_seeds in [
            ("Temporal", "time_seed42", "time_seeds"),
            ("Spectral", "fourier_seed42", "fourier_seeds"),
        ]:
            for i, seed_label in enumerate(seed_labels):
                cov_row = [f"CDSM ({model_name}, {seed_label})"]
                corr_row = [f"CDSM ({model_name}, {seed_label})"]
                for ds in datasets:
                    runs = []
                    main = ds.get(key_time)
                    if main:
                        runs.append(main)
                    runs.extend(ds.get(key_seeds, []))
                    if i >= len(runs):
                        cov_row.append("--")
                        corr_row.append("--")
                        continue
                    run, batch = runs[i]
                    assets = list(range(ds["assets"]))
                    cov_row.append(safe_metric(run, batch, "matrix_cov_fro", False, assets))
                    corr_row.append(safe_metric(run, batch, "matrix_corr_fro", True, assets))
                cov_seed_rows.append(cov_row)
                corr_seed_rows.append(corr_row)

        if args.out_cov_by_seed:
            write_table(args.out_cov_by_seed, cov_seed_rows, headers)
        if args.out_corr_by_seed:
            write_table(args.out_corr_by_seed, corr_seed_rows, headers)

    print(f"[OK] Wrote {args.out_cov} and {args.out_corr}")


if __name__ == "__main__":
    main()
