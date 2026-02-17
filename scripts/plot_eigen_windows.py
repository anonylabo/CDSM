#!/usr/bin/env python3
"""
Plot eigenvalue diagnostics for sliding windows (train/val/test) using the same
windowing/standardisation as the main experiments.

Visuals:
- Eigenvalue evolution: top-2 eigenvalues per window with MP upper edge.
- Eigenvalue scatter/heatmap view (all eigenvalues vs. window index).
- Aggregated density with Marchenko–Pastur (MP) PDF overlay.

Usage example:
python scripts/plot_eigen_windows.py \
  --dataset exchange_rate_clean \
  --split test \
  --context-len 60 --pred-len 30 --val-ratio 0.3 \
  --output assets/eigen_exchange_test.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from cfdiff.dataloaders.conditional_gluonts import ConditionalGluonTSJsonDatamodule


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset",
        type=str,
        default="exchange_rate_clean",
        help="Dataset name or path (GluonTS format). If name is given, resolved under ~/.gluonts/datasets/<name>.",
    )
    ap.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="test",
        help="Which split to plot.",
    )
    ap.add_argument("--context-len", type=int, default=60, help="Context length (time domain).")
    ap.add_argument("--pred-len", type=int, default=30, help="Prediction length.")
    ap.add_argument("--val-ratio", type=float, default=0.3, help="Validation ratio (same semantics as main experiments).")
    ap.add_argument("--train-val-gap", type=int, default=-1, help="Gap between train and val when val_ratio<0.")
    ap.add_argument("--val-test-gap", type=int, default=-1, help="Gap between val and test when val_ratio>=0.")
    ap.add_argument(
        "--align-tail-windows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match datamodule.align_tail_windows (append final window to touch end if stride misaligns).",
    )
    ap.add_argument("--stride", type=int, default=1, help="Window stride.")
    ap.add_argument("--max-windows", type=int, default=0, help="Optional cap on number of windows processed per split (0 = all).")
    ap.add_argument("--regime-manifest", type=Path, help="Optional window regime manifest (JSON) from label_window_regimes.py.")
    ap.add_argument("--regime-stats", type=Path, help="Optional regime stats JSON for annotation.")
    ap.add_argument("--regime-alpha", type=float, default=0.08, help="Alpha for regime shading.")
    ap.add_argument(
        "--regime-colors",
        type=str,
        default="bull:#b5e3b5,bear:#f5b7ae,unknown:#dddddd",
        help="Comma-separated mapping for regime colors, e.g. bull:#b5e3b5,bear:#f5b7ae.",
    )
    ap.add_argument(
        "--regime-legend",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show regime legend on eigenvalue panels.",
    )
    ap.add_argument(
        "--noise-wall-summary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Report how many eigenvalues exceed MP lambda_max (optionally by regime).",
    )
    ap.add_argument(
        "--noise-wall-include-top1",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include lambda_1 when counting eigenvalues above the noise wall.",
    )
    ap.add_argument(
        "--noise-wall-out",
        type=Path,
        help="Optional JSON path to write noise-wall summary stats.",
    )
    ap.add_argument("--output", type=Path, default=Path("assets/eigen_windows.png"), help="Output path.")
    ap.add_argument(
        "--output-format",
        default="png",
        choices=["png", "pdf", "svg"],
        help="Figure format (default: png).",
    )
    ap.add_argument("--dpi", type=int, default=200, help="Figure DPI.")
    return ap.parse_args()


def _resolve_data_dir(dataset: str | Path) -> Path:
    p = Path(dataset).expanduser()
    if p.exists():
        return p
    return Path.home() / ".gluonts" / "datasets" / dataset


def _build_datamodule(
    data_dir: Path,
    ctx: int,
    pred: int,
    stride: int,
    val_ratio: float,
    train_val_gap: int,
    val_test_gap: int,
    align_tail_windows: bool,
) -> ConditionalGluonTSJsonDatamodule:
    dm = ConditionalGluonTSJsonDatamodule(
        data_dir=str(data_dir),
        batch_size=64,
        context_len=ctx,
        pred_len=pred,
        stride=stride,
        val_ratio=val_ratio,
        train_val_gap=train_val_gap,
        val_test_gap=val_test_gap,
        standardize=True,
        num_workers=0,
        pin_memory=False,
        fourier_transform=False,
        estimate_sliding_cov=False,
        align_tail_windows=align_tail_windows,
    )
    dm.prepare_data()
    dm.setup()
    return dm


def _iter_windows(ds: Iterable, max_windows: int) -> np.ndarray:
    acc: List[np.ndarray] = []
    for idx, sample in enumerate(ds):
        if max_windows > 0 and idx >= max_windows:
            break
        ctx = sample["context_time"].detach().cpu().numpy()
        tgt = sample["target_time"].detach().cpu().numpy()
        full = np.concatenate([ctx, tgt], axis=0)  # (L, A)
        acc.append(full.astype(np.float32))
    if not acc:
        raise ValueError("No windows collected; check split selection or max_windows.")
    return np.stack(acc, axis=0)  # (N, L, A)


def _cov_eigvals(window: np.ndarray) -> np.ndarray:
    # window: (L, A) with standardised returns
    cov = np.cov(window, rowvar=False)
    cov = np.nan_to_num(cov, nan=0.0)
    # Symmetric; use eigvalsh
    vals = np.linalg.eigvalsh(cov)
    vals = np.maximum(vals, 0.0)  # clip tiny negatives
    vals_sorted = np.sort(vals)[::-1]
    return vals_sorted


def _mp_edges(num_assets: int, window_len: int) -> Tuple[float, float]:
    q = num_assets / float(window_len)
    lam_min = (1.0 - math.sqrt(q)) ** 2
    lam_max = (1.0 + math.sqrt(q)) ** 2
    return lam_min, lam_max


def _mp_pdf(lam: np.ndarray, lam_min: float, lam_max: float, q: float) -> np.ndarray:
    out = np.zeros_like(lam, dtype=float)
    mask = (lam >= lam_min) & (lam <= lam_max)
    out[mask] = np.sqrt((lam_max - lam[mask]) * (lam[mask] - lam_min)) / (2 * math.pi * q * lam[mask])
    return out


def _parse_color_map(spec: str) -> dict:
    mapping: dict = {}
    for part in str(spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            key, val = part.split(":", 1)
            mapping[key.strip()] = val.strip()
    return mapping


def _load_regime_manifest(path: Path) -> dict[int, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[int, str] = {}
    for rec in data:
        try:
            idx = int(rec.get("window_idx"))
        except Exception:
            continue
        out[idx] = str(rec.get("regime", "unknown"))
    return out


def _resolve_regime_series(
    regime_map: dict[int, str],
    split: str,
    n_train: int,
    n_val: int,
    n_test: int,
) -> List[str] | None:
    if not regime_map:
        return None
    max_idx = max(regime_map)
    total = max_idx + 1
    if split == "train":
        if total == n_train:
            return [regime_map.get(i, "unknown") for i in range(n_train)]
        return None
    if split == "val":
        if total == n_val:
            return [regime_map.get(i, "unknown") for i in range(n_val)]
        if n_val > 0 and total >= n_val + n_test:
            return [regime_map.get(i, "unknown") for i in range(n_val)]
        return None
    if split == "test":
        if total == n_test:
            return [regime_map.get(i, "unknown") for i in range(n_test)]
        if n_val > 0 and total >= n_val + n_test:
            return [regime_map.get(i + n_val, "unknown") for i in range(n_test)]
    return None


def _summarize_regimes(regimes: List[str]) -> str:
    counts: dict = {}
    for reg in regimes:
        counts[reg] = counts.get(reg, 0) + 1
    total = sum(counts.values())
    parts = []
    for reg in sorted(counts):
        ratio = counts[reg] / total if total else 0.0
        parts.append(f"{reg}: {counts[reg]} ({ratio:.1%})")
    return " | ".join(parts)


def _shade_regimes(ax, regimes: List[str], colors: dict, alpha: float) -> None:
    if not regimes:
        return
    start = 0
    current = regimes[0]
    for i in range(1, len(regimes) + 1):
        end_run = i == len(regimes) or regimes[i] != current
        if end_run:
            if current is not None:
                color = colors.get(current, colors.get("unknown", "#cccccc"))
                ax.axvspan(start - 0.5, i - 0.5, color=color, alpha=alpha, lw=0, zorder=0)
            if i < len(regimes):
                start = i
                current = regimes[i]
    return


def _noise_wall_stats(
    eigvals: np.ndarray,
    lam_max: float,
    regimes: List[str] | None,
    include_top1: bool,
) -> tuple[str, dict | None]:
    if eigvals.ndim != 2:
        return "λ>λmax", None
    if include_top1 or eigvals.shape[1] <= 1:
        check = eigvals
        label = "λ>λmax"
    else:
        check = eigvals[:, 1:]
        label = "λ>λmax (excl λ1)"

    exceed_counts = (check > lam_max).sum(axis=1)
    exceed_any = exceed_counts > 0
    total = len(exceed_counts)
    summary = {
        "total": total,
        "pct_any": float(exceed_any.mean()) if total else 0.0,
        "mean_count": float(exceed_counts.mean()) if total else 0.0,
    }

    by_regime = None
    if regimes:
        by_regime = {}
        unique_regs = sorted(set(regimes))
        for reg in unique_regs:
            idx = [i for i, r in enumerate(regimes) if r == reg]
            if not idx:
                continue
            reg_any = exceed_any[idx]
            reg_counts = exceed_counts[idx]
            by_regime[reg] = {
                "n": len(idx),
                "pct_any": float(reg_any.mean()) if len(idx) else 0.0,
                "mean_count": float(reg_counts.mean()) if len(idx) else 0.0,
            }
    return label, {"global": summary, "by_regime": by_regime}


def _make_plots(
    windows: np.ndarray,
    output: Path,
    lam_min: float,
    lam_max: float,
    dpi: int,
    regimes: List[str] | None = None,
    regime_colors: dict | None = None,
    regime_alpha: float = 0.08,
    regime_legend: bool = True,
    regime_text: str | None = None,
    noise_wall_summary: bool = False,
    noise_wall_include_top1: bool = False,
) -> dict | None:
    N, L, A = windows.shape
    eigvals = np.array([_cov_eigvals(w) for w in windows])  # (N, A)
    top1 = eigvals[:, 0]
    top2 = eigvals[:, 1] if A > 1 else np.zeros_like(top1)
    summary_lines = []
    noise_stats = None
    if regime_text:
        summary_lines.append(regime_text)
    if noise_wall_summary:
        label, stats = _noise_wall_stats(eigvals, lam_max, regimes, noise_wall_include_top1)
        if stats:
            global_stats = stats.get("global", {})
            pct_any = global_stats.get("pct_any", 0.0)
            mean_cnt = global_stats.get("mean_count", 0.0)
            print(f"[INFO] Noise-wall exceedance {label}: any={pct_any:.1%}, mean_count={mean_cnt:.2f}")
            by_reg = stats.get("by_regime")
            if by_reg:
                for reg in sorted(by_reg):
                    reg_stats = by_reg[reg]
                    print(
                        f"  {reg}: any={reg_stats['pct_any']:.1%}, mean_count={reg_stats['mean_count']:.2f} (n={reg_stats['n']})"
                    )
                parts = [
                    f"{reg} {by_reg[reg]['pct_any']:.0%}"
                    for reg in sorted(by_reg)
                ]
                summary_lines.append(f"{label}: " + " | ".join(parts))
            noise_stats = {
                "label": label,
                "lam_max": float(lam_max),
                "include_top1": bool(noise_wall_include_top1),
                "num_windows": int(N),
                "num_assets": int(A),
                "window_len": int(L),
                "global": stats.get("global"),
                "by_regime": stats.get("by_regime"),
            }

    # Aggregated eigenvalues (all)
    eig_all = eigvals.ravel()
    q = A / float(L)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=dpi)

    # Panel 1: evolution of eigenvalues excluding lambda_1
    ax = axes[0]
    if regimes:
        _shade_regimes(ax, regimes, regime_colors or {}, regime_alpha)
    xs = np.arange(len(top1))
    if A > 1:
        ax.plot(xs, top2, label=r"$\lambda_{2}$")
    if A > 2:
        ax.plot(xs, eigvals[:, 2], label=r"$\lambda_{3}$")
    if A > 3:
        ax.plot(xs, eigvals[:, 3], label=r"$\lambda_{4}$")
    ax.axhline(lam_max, color="red", linestyle="--", label=rf"MP $\lambda_{{\max}}$={lam_max:.2f}")
    ax.set_xlabel("Window index")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(r"Eigenvalue evolution (excluding $\lambda_{1}$)")
    handles, labels = ax.get_legend_handles_labels()
    if regimes and regime_legend:
        import matplotlib.patches as mpatches

        unique_regs = sorted({r for r in regimes if r is not None})
        patches = [
            mpatches.Patch(color=(regime_colors or {}).get(r, "#cccccc"), label=r) for r in unique_regs
        ]
        handles = handles + patches
        labels = labels + [p.get_label() for p in patches]
    ax.legend(handles, labels)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.8)

    # Panel 2: scatter of eigenvalues (excluding lambda_1) vs window index
    ax = axes[1]
    if regimes:
        _shade_regimes(ax, regimes, regime_colors or {}, regime_alpha)
    start_idx = 1 if A > 1 else 0
    for i in range(start_idx, min(A, 9)):  # plot up to λ9 for clarity
        ax.plot(xs, eigvals[:, i], marker=".", linestyle="", alpha=0.6, label=f"λ{i+1}")
    ax.axhline(lam_max, color="red", linestyle="--", label=rf"MP $\lambda_{{\max}}$={lam_max:.2f}")
    ax.set_xlabel("Window index")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(r"Eigenvalues per window (excluding $\lambda_{1}$)")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.8)
    ax.legend(ncol=2, fontsize=8)

    # Panel 3: aggregated density vs MP PDF (excl. lambda_1)
    ax = axes[2]
    # Focus on excluding lambda_1 for clarity in density; plot lambda_1 separately as marker
    eig_excl_top1 = eigvals[:, 1:].ravel() if A > 1 else np.array([], dtype=float)
    bins = max(80, min(200, len(eig_excl_top1) // 20)) if eig_excl_top1.size else 50
    if eig_excl_top1.size:
        ax.hist(eig_excl_top1, bins=bins, density=True, alpha=0.6, label=r"Empirical eigvals (excl. $\lambda_{1}$)")
    lam_grid = np.linspace(max(lam_min, eig_excl_top1.min() if eig_excl_top1.size else lam_min), min(lam_max, eig_excl_top1.max() if eig_excl_top1.size else lam_max), 400)
    ax.plot(lam_grid, _mp_pdf(lam_grid, lam_min, lam_max, q), color="black", linestyle="-", label="MP PDF")
    ax.axvline(lam_max, color="red", linestyle="--", label=rf"MP $\lambda_{{\max}}$={lam_max:.2f}")
    ax.set_xlabel("Eigenvalue")
    ax.set_ylabel("Density")
    ax.set_title(r"Aggregated eigenvalue density (excl. $\lambda_{1}$)")
    ax.set_xlim(0, lam_max * 1.5)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.8)
    ax.legend()
    if summary_lines:
        ax.text(0.02, 0.98, "\n".join(summary_lines), transform=ax.transAxes, ha="left", va="top", fontsize=9)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved eigen plots to {output}")
    return noise_stats


def main() -> None:
    args = parse_args()
    fmt = args.output_format.lstrip(".").lower()
    out_path = args.output.expanduser().with_suffix(f".{fmt}")
    data_dir = _resolve_data_dir(args.dataset)
    dm = _build_datamodule(
        data_dir,
        args.context_len,
        args.pred_len,
        args.stride,
        args.val_ratio,
        args.train_val_gap,
        args.val_test_gap,
        args.align_tail_windows,
    )

    if args.split == "train":
        ds = dm.ds_train
    elif args.split == "val":
        ds = dm.ds_val
    else:
        ds = dm.ds_test
    if ds is None:
        raise SystemExit(f"Requested split '{args.split}' not available.")

    windows = _iter_windows(ds, args.max_windows)
    regimes = None
    regime_text = None
    if args.regime_manifest:
        manifest_path = args.regime_manifest.expanduser()
        if manifest_path.exists():
            regime_map = _load_regime_manifest(manifest_path)
            n_train = len(dm.ds_train) if dm.ds_train is not None else 0
            n_val = len(dm.ds_val) if dm.ds_val is not None else 0
            n_test = len(dm.ds_test) if dm.ds_test is not None else 0
            regimes = _resolve_regime_series(regime_map, args.split, n_train, n_val, n_test)
            if regimes is None:
                print(
                    f"[WARN] Regime manifest indices do not align with split '{args.split}'. "
                    "Skipping regime overlay."
                )
        else:
            print(f"[WARN] Regime manifest not found: {manifest_path}")
    if regimes:
        if len(regimes) != windows.shape[0]:
            if len(regimes) > windows.shape[0]:
                regimes = regimes[: windows.shape[0]]
            else:
                print(
                    "[WARN] Regime series shorter than plotted windows; skipping regime overlay."
                )
                regimes = None
        regime_text = _summarize_regimes(regimes) if regimes else None
        if args.regime_stats and args.regime_stats.expanduser().exists():
            try:
                stats = json.loads(args.regime_stats.expanduser().read_text(encoding="utf-8"))
                if "bull" in stats and "bear" in stats and "total_windows" in stats:
                    bull = stats.get("bull", 0)
                    bear = stats.get("bear", 0)
                    total = stats.get("total_windows", bull + bear)
                    if total:
                        regime_text = f"bull: {bull} ({bull/total:.1%}) | bear: {bear} ({bear/total:.1%})"
            except Exception:
                pass
    _, L, A = windows.shape
    lam_min, lam_max = _mp_edges(A, L)
    want_noise_stats = bool(args.noise_wall_summary or args.noise_wall_out)
    noise_stats = _make_plots(
        windows,
        out_path,
        lam_min,
        lam_max,
        args.dpi,
        regimes=regimes,
        regime_colors=_parse_color_map(args.regime_colors),
        regime_alpha=float(args.regime_alpha),
        regime_legend=bool(args.regime_legend),
        regime_text=regime_text,
        noise_wall_summary=want_noise_stats,
        noise_wall_include_top1=bool(args.noise_wall_include_top1),
    )
    if args.noise_wall_out:
        payload = noise_stats or {}
        payload.update(
            {
                "dataset": str(data_dir),
                "split": args.split,
                "context_len": int(args.context_len),
                "pred_len": int(args.pred_len),
                "stride": int(args.stride),
                "val_ratio": float(args.val_ratio),
                "train_val_gap": int(args.train_val_gap),
                "val_test_gap": int(args.val_test_gap),
                "align_tail_windows": bool(args.align_tail_windows),
                "regime_manifest": str(args.regime_manifest) if args.regime_manifest else None,
                "regime_stats": str(args.regime_stats) if args.regime_stats else None,
                "regime_overlay": bool(regimes),
            }
        )
        out_path = args.noise_wall_out.expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[INFO] Saved noise-wall summary to {out_path}")


if __name__ == "__main__":
    main()
