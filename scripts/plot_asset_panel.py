#!/usr/bin/env python3
"""Visualise per-asset predictions for multiple models and windows.

Originally this script compared the four diffusion variants. It now supports
arbitrary model listings and multiple windows. Columns are laid out as
`(window, model)` pairs so that you can, for example, compare two models across
windows 0--9 in a single figure.
"""

from __future__ import annotations

import argparse
import gzip
import json
import itertools
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from cfdiff.dataloaders.conditional_gluonts import _resolve_split_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs",
        help="Root directory containing model outputs (default: %(default)s)",
    )
    parser.add_argument(
        "--assets",
        type=int,
        default=5,
        help="Number of assets to display (rows).",
    )
    parser.add_argument(
        "--asset-offset",
        type=int,
        default=0,
        help="Asset index offset (start from this asset).",
    )
    parser.add_argument(
        "--asset-indices",
        type=int,
        nargs="+",
        help="Explicit asset indices to plot (overrides --assets/--asset-offset).",
    )
    parser.add_argument(
        "--asset-ids",
        nargs="+",
        help="Explicit item_id labels to plot (requires --dataset; overrides --assets/--asset-offset).",
    )
    parser.add_argument(
        "--window-index",
        type=int,
        default=0,
        help="Deprecated; use --windows. Kept for backward compatibility.",
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        help="List of forecast window indices to visualise (e.g., 0 1 2).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="LABEL|RUN_DIR|[SAMPLE_TAG]",
        help=(
            "Explicit models to plot. Each entry is 'LABEL|RUN_DIR|[SAMPLE_TAG]', "
            "where RUN_DIR contains samples.pt (or samples_history/*/samples.pt). "
            "SAMPLE_TAG is optional and may be used to pick a specific samples_history subdir."
        ),
    )
    parser.add_argument(
        "--runs",
        nargs=4,
        metavar=("TIME_COND", "TIME_UNCOND", "FOURIER_COND", "FOURIER_UNCOND"),
        help=(
            "Deprecated; use --models. If provided, will plot the four diffusion variants "
            "with the newest checkpoints under outputs/<domain>/<mode> unless overridden."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Optional dataset name or path to annotate the figure.",
    )
    parser.add_argument(
        "--dataset-label",
        type=str,
        help="Optional label to use instead of --dataset when annotating the figure.",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        choices=["train", "test"],
        default="train",
        help="Split to read item_id labels from when --dataset is set.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the resulting figure (PNG, PDF, ...).",
    )
    parser.add_argument(
        "--panel-size",
        type=float,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(4.0, 2.5),
        help="Figure size (width, height) in inches per subplot.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="Resolution (dots per inch) when saving the figure.",
    )
    parser.add_argument(
        "--overlay-models",
        action="store_true",
        help="If set, overlay all models on the same axes per window (columns = windows). "
        "Otherwise columns correspond to (window, model) pairs.",
    )
    parser.add_argument(
        "--batch-aggregate",
        action="store_true",
        help="If run_dir/sample_tag points to a batch directory containing multiple sub-runs, "
        "aggregate their predictions by mean/std and plot mean with a shaded band.",
    )
    return parser.parse_args()


def resolve_artifact_path(run_dir: Path, filename: str, sample_tag: Optional[str] = None) -> Path:
    history_root = run_dir / "samples_history"
    if sample_tag:
        candidates = (
            sorted(
                (p for p in history_root.iterdir() if p.is_dir() and sample_tag in p.name),
                key=lambda p: p.name,
            )
            if history_root.is_dir()
            else []
        )
        for candidate in reversed(candidates):
            candidate_path = candidate / filename
            if candidate_path.exists():
                return candidate_path
        raise FileNotFoundError(f"No samples matching tag '{sample_tag}' under {history_root}.")

    if history_root.is_dir():
        candidates = sorted((p for p in history_root.iterdir() if p.is_dir()), key=lambda p: p.name)
        for candidate in reversed(candidates):
            candidate_path = candidate / filename
            if candidate_path.exists():
                return candidate_path
    fallback = run_dir / filename
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Unable to locate {filename} in {run_dir} (looked under samples_history/ and root).")


def latest_run(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Directory does not exist: {path}")
    candidates = [p for p in path.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {path}")
    return max(candidates, key=lambda p: p.name)


def resolve_default_four_models(root: Path) -> Dict[str, Tuple[str, str, Optional[str]]]:
    model_order = (
        ("Time (Conditional)", ("time", "conditional")),
        ("Time (Unconditional)", ("time", "unconditional")),
        ("Fourier (Conditional)", ("fourier", "conditional")),
        ("Fourier (Unconditional)", ("fourier", "unconditional")),
    )
    resolved: Dict[str, Tuple[str, str, Optional[str]]] = {}
    for label, (domain, mode) in model_order:
        base = root / domain / mode
        resolved[label] = (str(latest_run(base)), None)  # type: ignore[assignment]
    return resolved


def parse_models_arg(outputs_root: Path, models: Optional[Sequence[str]], runs_compat: Optional[Sequence[str]]) -> Dict[str, Tuple[Path, Optional[str]]]:
    """
    Return mapping: label -> (run_dir, sample_tag).
    - If --models is provided: entries of the form LABEL|RUN_DIR|[SAMPLE_TAG].
    - Else if --runs (deprecated) is provided: use the four diffusion variants.
    - Else: auto-select latest four diffusion variants.
    """
    resolved: Dict[str, Tuple[Path, Optional[str]]] = {}
    if models:
        for entry in models:
            parts = entry.split("|")
            if len(parts) < 2:
                raise ValueError(f"--models entry must be LABEL|RUN_DIR|[SAMPLE_TAG]; got: {entry}")
            label, run_dir = parts[0], parts[1]
            sample_tag = parts[2] if len(parts) >= 3 and parts[2].strip() else None
            run_path = Path(run_dir).expanduser().resolve()
            if not run_path.exists():
                raise FileNotFoundError(f"Run directory does not exist: {run_path}")
            resolved[label] = (run_path, sample_tag)
        return resolved

    if runs_compat:
        if len(runs_compat) != 4:
            raise ValueError("Exactly four run overrides must be supplied when using --runs.")
        labels = ["Time (Conditional)", "Time (Unconditional)", "Fourier (Conditional)", "Fourier (Unconditional)"]
        for label, override in zip(labels, runs_compat):
            run_path = Path(override).expanduser().resolve()
            if not run_path.exists():
                raise FileNotFoundError(f"Override path does not exist: {run_path}")
            resolved[label] = (run_path, None)
        return resolved

    # Default to the latest four diffusion variants
    defaults = resolve_default_four_models(outputs_root)
    for label, (run_str, sample_tag) in defaults.items():
        resolved[label] = (Path(run_str), sample_tag)
    return resolved


def resolve_dataset_label(dataset: Optional[str], dataset_label: Optional[str]) -> Optional[str]:
    if dataset_label:
        return dataset_label
    if not dataset:
        return None
    p = Path(dataset).expanduser()
    if p.exists():
        return p.name
    return dataset


def resolve_dataset_path(dataset: str) -> Path:
    p = Path(dataset).expanduser()
    if p.exists():
        return p
    candidate = Path.home() / ".gluonts" / "datasets" / dataset
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Dataset path not found: {dataset}")


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


def load_item_ids(data_dir: Path, split: str) -> List[str]:
    split_file = _resolve_split_file(data_dir, split)
    item_ids: List[str] = []
    for idx, obj in enumerate(_iter_jsonl(split_file)):
        item_id = obj.get("item_id")
        if item_id is None:
            item_id = f"series_{idx}"
        item_ids.append(str(item_id))
    return item_ids


def load_samples(run_dir: Path, sample_tag: Optional[str]) -> Dict[str, np.ndarray]:
    path = resolve_artifact_path(run_dir, "samples.pt", sample_tag=sample_tag)
    data = torch.load(path, map_location="cpu", weights_only=False)

    def _to_np(x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return np.asarray(x)

    samples = _to_np(data["samples"])
    truth = _to_np(data["truth"])
    context_raw = data.get("context")
    context_np = _to_np(context_raw) if context_raw is not None else None
    return {
        "samples": samples,
        "truth": truth,
        "context": context_np,
    }


def load_model_payloads(run_dir: Path, sample_tag: Optional[str], batch_aggregate: bool) -> Sequence[Dict[str, np.ndarray]]:
    if not batch_aggregate:
        return [load_samples(run_dir, sample_tag)]

    history_root = run_dir / "samples_history"
    candidates = []
    if history_root.is_dir():
        for sub in sorted(history_root.iterdir()):
            if not sub.is_dir():
                continue
            if sample_tag and sample_tag not in sub.name:
                continue
            direct = sub / "samples.pt"
            if direct.exists():
                candidates.append(direct)
                continue
            # Handle batch folders containing sub-run directories
            for child in sorted(sub.iterdir()):
                if not child.is_dir():
                    continue
                nested = child / "samples.pt"
                if nested.exists():
                    candidates.append(nested)
    # Also handle batch directories that directly contain sub-runs with samples.pt (no samples_history/)
    if not candidates:
        subruns = [p for p in sorted(run_dir.iterdir()) if p.is_dir()]
        for sub in subruns:
            if sample_tag and sample_tag not in sub.name:
                continue
            nested = sub / "samples.pt"
            if nested.exists():
                candidates.append(nested)
    if not candidates:
        # Fallback to single run if no batch sub-runs found
        return [load_samples(run_dir, sample_tag)]
    payloads = []
    for p in candidates:
        data = torch.load(p, map_location="cpu", weights_only=False)
        def _to_np(x):
            if isinstance(x, torch.Tensor):
                return x.cpu().numpy()
            return np.asarray(x)
        payloads.append(
            {
                "samples": _to_np(data["samples"]),
                "truth": _to_np(data["truth"]),
                "context": _to_np(data.get("context")) if data.get("context") is not None else None,
            }
        )
    return payloads


def aggregate_payloads(payloads: Sequence[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    if len(payloads) == 1:
        payload = payloads[0]
        return {
            "samples_mean": payload["samples"],
            "samples_std": None,
            "truth": payload["truth"],
            "context": payload["context"],
        }
    samples_stack = np.stack([p["samples"] for p in payloads], axis=0)  # (R, W, P, A)
    mean = samples_stack.mean(axis=0)
    std = samples_stack.std(axis=0)
    # assume truth/context identical across runs; use the first
    base = payloads[0]
    return {
        "samples_mean": mean,
        "samples_std": std,
        "truth": base["truth"],
        "context": base["context"],
    }


def prepare_panel(
    datasets: Dict[str, Dict[str, np.ndarray]],
    window_indices: Sequence[int],
    asset_indices: Sequence[int],
) -> Tuple[int, int, Dict[Tuple[int, str], Dict[str, np.ndarray]]]:
    # Determine sizes and slice the requested windows / assets.
    first = next(iter(datasets.values()))
    total_windows, pred_len, total_assets = first["samples_mean"].shape
    context_len = first["context"].shape[1] if first.get("context") is not None else 0

    for w in window_indices:
        if w < 0 or w >= total_windows:
            raise IndexError(f"window_index {w} outside [0, {total_windows - 1}]")

    for asset in asset_indices:
        if asset < 0 or asset >= total_assets:
            raise IndexError(f"asset index {asset} outside [0, {total_assets - 1}]")

    sliced: Dict[Tuple[int, str], Dict[str, np.ndarray]] = {}
    for w in window_indices:
        for label, payload in datasets.items():
            preds = payload["samples_mean"][w, :, asset_indices]
            preds_std = None
            if payload.get("samples_std") is not None:
                preds_std = payload["samples_std"][w, :, asset_indices]
            truth = payload["truth"][w, :, asset_indices]
            context = payload.get("context")
            context_slice = None
            if context is not None:
                context_slice = context[w, :, asset_indices]
            sliced[(w, label)] = {
                "pred": preds,
                "pred_std": preds_std,
                "truth": truth,
                "context": context_slice,
            }
    return context_len, pred_len, sliced


def main() -> None:
    args = parse_args()
    model_specs = parse_models_arg(args.outputs_root, args.models, args.runs)

    datasets = {}
    for label, (path, sample_tag) in model_specs.items():
        payloads = load_model_payloads(path, sample_tag, batch_aggregate=args.batch_aggregate)
        datasets[label] = aggregate_payloads(payloads)

    # Fallback context for models without explicit context stored.
    reference_context = None
    for payload in datasets.values():
        if payload["context"] is not None:
            reference_context = payload["context"]
            break

    if reference_context is None:
        raise RuntimeError("No context information available in any samples.pt file.")

    for payload in datasets.values():
        if payload["context"] is None:
            payload["context"] = reference_context

    item_ids = None
    asset_labels = None
    if args.dataset:
        try:
            ds_path = resolve_dataset_path(args.dataset)
            item_ids = load_item_ids(ds_path, args.dataset_split)
            total_assets = reference_context.shape[2] if reference_context is not None else len(item_ids)
            if len(item_ids) != total_assets:
                print(
                    f"[WARN] item_id count ({len(item_ids)}) does not match assets ({total_assets}); "
                    "falling back to Asset indices."
                )
                if args.asset_ids:
                    raise SystemExit(
                        "Cannot resolve --asset-ids because item_id count does not match samples assets."
                    )
            else:
                asset_labels = item_ids
        except Exception as exc:
            print(f"[WARN] Unable to load item_id labels: {exc}")

    if args.asset_indices and args.asset_ids:
        raise SystemExit("Use either --asset-indices or --asset-ids, not both.")
    if args.asset_indices:
        asset_indices = list(args.asset_indices)
    elif args.asset_ids:
        if not item_ids:
            raise SystemExit("--asset-ids requires --dataset so item_id labels can be loaded.")
        index_map = {name: idx for idx, name in enumerate(item_ids)}
        missing = [name for name in args.asset_ids if name not in index_map]
        if missing:
            raise SystemExit(f"Unknown item_id labels: {', '.join(missing)}")
        asset_indices = [index_map[name] for name in args.asset_ids]
    else:
        asset_indices = list(range(args.asset_offset, args.asset_offset + args.assets))

    window_indices = args.windows if args.windows is not None else [args.window_index]
    context_len, pred_len, panel = prepare_panel(
        datasets=datasets,
        window_indices=window_indices,
        asset_indices=asset_indices,
    )

    n_cols = len(window_indices) if args.overlay_models else len(model_specs) * len(window_indices)
    fig_width = args.panel_size[0] * n_cols
    fig_height = args.panel_size[1] * len(asset_indices)
    fig, axes = plt.subplots(
        nrows=len(asset_indices),
        ncols=n_cols,
        figsize=(fig_width, fig_height),
        sharex="col",
        dpi=args.dpi,
    )

    if len(asset_indices) == 1:
        axes = np.array([axes])

    colors = ("C0", "C1", "C2", "C3", "C4", "C5")
    context_x = np.arange(-context_len, 0)
    target_x = np.arange(0, pred_len)

    if args.overlay_models:
        for col, (widx, ax_column) in enumerate(zip(window_indices, axes.T)):
            title = f"Window {widx}"
            for row, ax in enumerate(ax_column):
                asset_idx = asset_indices[row]
                n_assets = len(asset_indices)

                # use first model as reference for context/truth
                ref_label = next(iter(model_specs.keys()))
                ref_data = panel[(widx, ref_label)]

                def select(series: np.ndarray | None) -> Optional[np.ndarray]:
                    if series is None:
                        return None
                    if series.shape[0] == n_assets:
                        return series[row, :]
                    return series[:, row]

                context = select(ref_data["context"])
                truth = select(ref_data["truth"])
                if context is not None:
                    ax.plot(context_x, context, color="lightgray", linewidth=1.5, label="context" if col == 0 and row == 0 else None)
                ax.plot(target_x, truth, color="black", linewidth=1.5, label="truth" if col == 0 and row == 0 else None)

                for m_idx, label in enumerate(model_specs.keys()):
                    data = panel[(widx, label)]
                    pred = select(data["pred"])
                    pred_std = select(data.get("pred_std"))
                    color = colors[m_idx % len(colors)]
                    ax.plot(target_x, pred, color=color, linewidth=1.5, label=label if col == 0 and row == 0 else None)
                    if pred_std is not None:
                        ax.fill_between(
                            target_x,
                            pred - pred_std,
                            pred + pred_std,
                            color=color,
                            alpha=0.2,
                            linewidth=0.0,
                        )

                ax.axvline(x=-0.5, color="gray", linewidth=1.0, linestyle="--")
                ax.set_xlim(-context_len, pred_len - 1)

                if col == 0:
                    label = asset_labels[asset_idx] if asset_labels is not None else f"Asset {asset_idx}"
                    ax.set_ylabel(label, rotation=0, labelpad=40, fontsize=10)
                if row == 0:
                    ax.set_title(title, fontsize=12)
                if row == len(asset_indices) - 1:
                    ax.set_xlabel("Steps relative to forecast start")
    else:
        columns = []
        for w in window_indices:
            for label in model_specs.keys():
                columns.append((w, label))

        for col, ((widx, label), ax_column) in enumerate(zip(columns, axes.T)):
            title = f"{label} (win {widx})"
            color = colors[col % len(colors)]
            data = panel[(widx, label)]
            for row, ax in enumerate(ax_column):
                asset_idx = asset_indices[row]
                n_assets = len(asset_indices)

                def select(series: np.ndarray | None) -> Optional[np.ndarray]:
                    if series is None:
                        return None
                    if series.shape[0] == n_assets:
                        return series[row, :]
                    return series[:, row]

                context = select(data["context"])
                truth = select(data["truth"])
                pred = select(data["pred"])
                pred_std = select(data.get("pred_std"))

                if context is not None:
                    ax.plot(context_x, context, color="lightgray", linewidth=1.5, label="context")
                ax.plot(target_x, truth, color="black", linewidth=1.5, label="truth")
                ax.plot(target_x, pred, color=color, linewidth=1.5, label="prediction")
                if pred_std is not None:
                    ax.fill_between(
                        target_x,
                        pred - pred_std,
                        pred + pred_std,
                        color=color,
                        alpha=0.2,
                        linewidth=0.0,
                    )
                ax.axvline(x=-0.5, color="gray", linewidth=1.0, linestyle="--")
                ax.set_xlim(-context_len, pred_len - 1)

                if col == 0:
                    label = asset_labels[asset_idx] if asset_labels is not None else f"Asset {asset_idx}"
                    ax.set_ylabel(label, rotation=0, labelpad=40, fontsize=10)

                if row == 0:
                    ax.set_title(title, fontsize=12)

                if row == len(asset_indices) - 1:
                    ax.set_xlabel("Steps relative to forecast start")

    # Shared legend (use the first axis).
    handles, labels = axes[0, 0].get_legend_handles_labels()
    dataset_label = resolve_dataset_label(args.dataset, args.dataset_label)
    if dataset_label:
        fig.text(0.5, 0.995, f"Dataset: {dataset_label}", ha="center", va="top", fontsize=13)
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=3, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
    else:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.97))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=args.dpi)
        print(f"Saved figure to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
