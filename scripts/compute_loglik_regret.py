#!/usr/bin/env python3
"""
Compute per-window Gaussian negative log-likelihood (NLL) for diffusion runs and baselines,
optionally summarised as mean±std across the selected stage (val/test/all).

For each window:
- Use predicted covariance (from model samples or baseline files).
- True returns are the target window.
- NLL per time step: 0.5 * (logdet(Sigma) + trace(Sigma^{-1} S_emp) + A*log(2π)),
  where S_emp is the sample covariance of the true returns in the window.

Outputs a simple table: Model / Baseline, NLL per step (mean±std).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

MODEL_ORDER = (
    ("time", "conditional"),
    ("time", "unconditional"),
    ("fourier", "conditional"),
    ("fourier", "unconditional"),
)


def _stage_indices(counts: Dict[str, int], stage: str) -> List[int]:
    val = int(counts.get("val", 0) or 0)
    test = int(counts.get("test", 0) or 0)
    if stage == "val":
        return list(range(val))
    if stage == "test":
        return list(range(val, val + test))
    if stage == "all":
        return list(range(val + test))
    raise ValueError(f"Unsupported stage '{stage}'")


def _find_samples_path(run_dir: Path, sample_tag: Optional[str]) -> Path:
    if sample_tag and sample_tag not in {"", "-", "none"}:
        hist_dir = run_dir / "samples_history" / sample_tag
        if hist_dir.is_dir():
            candidates = sorted(hist_dir.glob("**/samples.pt"))
            if candidates:
                return candidates[-1]
    direct = run_dir / "samples.pt"
    if direct.is_file():
        return direct
    hist_root = run_dir / "samples_history"
    if hist_root.is_dir():
        candidates = sorted(hist_root.glob("**/samples.pt"))
        if candidates:
            return candidates[-1]
    raise FileNotFoundError(f"Could not locate samples.pt under {run_dir}")


def _load_samples(run_dir: Path, sample_tag: Optional[str], stage: str, asset_indices: Sequence[int]):
    path = _find_samples_path(run_dir, sample_tag)
    data = torch.load(path, map_location="cpu", weights_only=False)
    pred = data["samples"].cpu().numpy()
    truth = data["truth"].cpu().numpy()
    context = data.get("context")
    counts = data.get("window_stage_counts", {})
    idxs = _stage_indices(counts, stage) if counts else list(range(pred.shape[0]))

    if pred.ndim == 4:
        w, s, t, a = pred.shape
        pred = pred.reshape(w, s * t, a)
    elif pred.ndim == 3:
        pass
    else:
        raise RuntimeError(f"Unexpected pred shape {pred.shape}")

    if truth.ndim == 4:
        truth = truth[:, 0, :, :]
    elif truth.ndim == 3:
        pass
    else:
        raise RuntimeError(f"Unexpected truth shape {truth.shape}")

    if context is not None:
        context = context.cpu().numpy()
        if context.ndim == 4:
            context = context[:, 0, :, :]
        elif context.ndim != 3:
            raise RuntimeError(f"Unexpected context shape {context.shape}")

    pred = pred[idxs][:, :, asset_indices]
    truth = truth[idxs][:, :, asset_indices]
    if context is not None:
        context = context[idxs][:, :, asset_indices]
    return pred, truth, context


def _load_baseline(
    run_dir: Path,
    prefix: str,
    asset_indices: Sequence[int],
    stage: str,
    first_step_only: bool,
):
    subdirs = [p for p in run_dir.iterdir() if p.is_dir()]
    run_paths = sorted(subdirs) if subdirs else [run_dir]

    stage_path = run_paths[0] / "stage_counts.json"
    if not stage_path.is_file():
        stage_path = run_dir / "stage_counts.json"
    if not stage_path.is_file():
        raise FileNotFoundError(f"stage_counts.json not found under {run_dir}")
    counts = json.loads(stage_path.read_text())
    idxs = _stage_indices(counts, stage)

    vols: List[float] = []
    nlls: List[float] = []
    for rp in run_paths:
        est_files = sorted(rp.glob(f"{prefix}_win*_est.pt"))
        if not est_files:
            raise FileNotFoundError(f"No {prefix}_win*_est.pt under {rp}")
        for i, f in enumerate(est_files):
            if i not in idxs:
                continue
            idx = f.stem.split("_win")[1].split("_")[0]
            pred_cov = torch.load(f, map_location="cpu", weights_only=False).cpu().numpy()
            pred_cov = pred_cov[np.ix_(asset_indices, asset_indices)]

            tgt_pt = rp / f"{prefix}_win{idx}_target_series.pt"
            truth_pt = rp / f"{prefix}_win{idx}_truth.pt"
            if tgt_pt.is_file():
                tgt = torch.load(tgt_pt, map_location="cpu", weights_only=False).cpu().numpy()
            elif truth_pt.is_file():
                tgt = torch.load(truth_pt, map_location="cpu", weights_only=False).cpu().numpy()
            else:
                tgt_csv = rp / f"{prefix}_win{idx}_target_series.csv"
                truth_csv = rp / f"{prefix}_win{idx}_truth.csv"
                if tgt_csv.is_file():
                    tgt = pd.read_csv(tgt_csv).iloc[:, 1:].to_numpy()
                elif truth_csv.is_file():
                    tgt = pd.read_csv(truth_csv).iloc[:, 1:].to_numpy()
                else:
                    raise FileNotFoundError(f"Missing target/truth series for {f}")
            tgt = tgt[:, asset_indices]
            if first_step_only:
                tgt = tgt[:1]

            nlls.append(_nll_per_step(pred_cov, tgt))
    return float(np.mean(nlls)), float(np.std(nlls) if len(nlls) > 1 else 0.0)


def _nll_per_step(pred_cov: np.ndarray, true_returns: np.ndarray) -> float:
    """Negative log-likelihood per time step under Gaussian with covariance pred_cov."""
    n = pred_cov.shape[0]
    T = true_returns.shape[0]
    eps = 1e-4
    cov = pred_cov + eps * np.eye(n)
    cov = (cov + cov.T) * 0.5  # symmetrize
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0 or not np.isfinite(logdet):
        # Project to PSD via eigen clipping
        w, v = np.linalg.eigh(cov)
        w_clipped = np.clip(w, eps, None)
        cov = v @ np.diag(w_clipped) @ v.T
        sign, logdet = 1.0, np.log(w_clipped + 1e-12).sum()
    inv = np.linalg.pinv(cov, rcond=1e-6)
    # Empirical covariance of true returns
    if T <= 1:
        s_emp = np.outer(true_returns[0], true_returns[0])
    else:
        s_emp = (true_returns.T @ true_returns) / max(T - 1, 1)
    trace_term = np.trace(inv @ s_emp)
    nll = 0.5 * (logdet + trace_term + n * math.log(2 * math.pi))
    return float(nll)


def _resolve_runs(overrides: Optional[Sequence[str]]) -> Dict[Tuple[str, str], Path]:
    resolved: Dict[Tuple[str, str], Path] = {}
    if overrides:
        if len(overrides) != 4:
            raise ValueError("Exactly four run overrides must be supplied.")
        skip = {"", "-", "none", "skip"}
        for key, override in zip(MODEL_ORDER, overrides):
            text = (override or "").strip()
            if text.lower() in skip:
                continue
            resolved[key] = Path(text).expanduser().resolve()
    return resolved


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs=4, help="Explicit run dirs for time_cond, time_uncond, fourier_cond, fourier_uncond.")
    ap.add_argument("--sample-tags", nargs=4, help="Optional sample tags (use '-' to select latest).")
    ap.add_argument("--baseline", action="append", help="Format: RUN|PREFIX|LABEL (label optional).")
    ap.add_argument("--stage", choices=["all", "val", "test"], default="test")
    ap.add_argument("--asset-offset", type=int, default=0)
    ap.add_argument("--assets", type=int, default=10)
    ap.add_argument("--out-csv", type=Path, help="Save table as CSV.")
    ap.add_argument("--out-md", type=Path, help="Save table as Markdown.")
    ap.add_argument(
        "--first-step-only",
        action="store_true",
        help="If set, only use the first time step of each window (rolling one-step style).",
    )
    ap.add_argument(
        "--use-augmented-cov",
        action="store_true",
        help="If set, compute predicted covariance on [context; pred] instead of pred-only for matched runs.",
    )
    ap.add_argument(
        "--augmented-run-substr",
        action="append",
        help="Apply augmented cov only if model run path contains any of these substrings (overrides global off).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    asset_idx = list(range(args.asset_offset, args.asset_offset + args.assets))

    runs = _resolve_runs(args.runs)
    sample_tags = args.sample_tags or [None] * len(MODEL_ORDER)

    rows: List[Dict[str, object]] = []

    # Models
    for idx, key in enumerate(MODEL_ORDER):
        run_dir = runs.get(key)
        if run_dir is None:
            continue
        use_aug = args.use_augmented_cov or (
            args.augmented_run_substr and any(sub in str(run_dir) for sub in args.augmented_run_substr)
        )
        tag = sample_tags[idx]
        tag_clean = None if tag in {None, "-", ""} else tag
        pred, truth, context = _load_samples(run_dir, tag_clean, args.stage, asset_idx)
        nlls = []
        for p, t, ctx in zip(pred, truth, context if context is not None else [None] * len(pred)):
            if args.first_step_only:
                p = p[:1]
                t = t[:1]
                if ctx is not None:
                    ctx = ctx[:1]
            if use_aug and ctx is not None:
                window = np.concatenate([ctx, p], axis=0)
                cov = np.cov(window, rowvar=False)
            else:
                cov = np.cov(p, rowvar=False)
            nlls.append(_nll_per_step(cov, t))
        name = f"{key[0].title()} ({key[1].title()})"
        rows.append({"Model / Baseline": name, "NLL per step": f"{np.mean(nlls):.4f}±{np.std(nlls) if len(nlls)>1 else 0.0:.4f}"})

    # Baselines
    if args.baseline:
        for spec in args.baseline:
            parts = spec.strip().split("|")
            if len(parts) < 2:
                raise ValueError(f"--baseline entries must be 'RUN|PREFIX|LABEL' (label optional); got: {spec}")
            run = parts[0].strip()
            prefix = parts[1]
            label = parts[2] if len(parts) > 2 else None
            mean_nll, std_nll = _load_baseline(
                Path(run).expanduser(),
                prefix,
                asset_idx,
                args.stage,
                args.first_step_only,
            )
            display = label or prefix.replace("_", " ").title()
            rows.append({"Model / Baseline": display, "NLL per step": f"{mean_nll:.4f}±{std_nll:.4f}"})

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        print(f"Saved CSV to {args.out_csv}")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        lines = ["| " + " | ".join(df.columns) + " |", "| " + " | ".join(["---"] * len(df.columns)) + " |"]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(v) for v in row.tolist()) + " |")
        args.out_md.write_text("\n".join(lines))
        print(f"Saved Markdown to {args.out_md}")


if __name__ == "__main__":
    main()
