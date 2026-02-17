#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from plot_correlation_mean import aggregate_model

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOTS = [ROOT / "outputs", ROOT / "outputs_local"]


DATASETS = [
    {"label": "Exchange", "assets": 8, "patterns": ["FX8_30_15val0.3train0.8"]},
    {
        "label": "FX29",
        "assets": 29,
        "patterns": ["FX30_45_15val0.3train0.8", "FX30_ecb_45_15val0.3train0.8"],
    },
    {"label": "Industry", "assets": 49, "patterns": ["ind49_30_15val0.3train0.8"]},
    {"label": "iShares14", "assets": 14, "patterns": ["ishares14_30_15_val0.3train0.8"]},
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate ablation context pooled table (trajectory=0).")
    ap.add_argument("--metric", choices=["cov", "corr"], default="cov")
    ap.add_argument("--lambda-tag", default="0", help="Lambda tag for conditional runs (e.g., 0, 5e-4).")
    ap.add_argument("--stage", choices=["all", "val", "test"], default="test")
    ap.add_argument("--trajectory-index", type=int, default=0)
    ap.add_argument("--sig-digits", type=int, default=4)
    ap.add_argument(
        "--use-ca",
        action="store_true",
        help="Use context augmentation (CA): concatenate context+prediction when computing cov/corr.",
    )
    ap.add_argument(
        "--out-tex",
        type=Path,
        default=ROOT / "assets" / "table_ablation_context_pooled.tex",
    )
    return ap.parse_args()


def _to_np(value):
    if isinstance(value, torch.Tensor):
        return value.cpu().numpy()
    return value


def _format_sig(value: float, sig: int = 4) -> str:
    if value == 0:
        return f"{0:.{sig}f}"
    v = abs(float(value))
    exp = math.floor(math.log10(v))
    decimals = sig - 1 - exp
    if decimals >= 0:
        return f"{value:.{decimals}f}"
    q = 10 ** (-decimals)
    return f"{round(value / q) * q:.0f}"


def _format_sci(value: float, sig: int = 4) -> str:
    if value == 0:
        return "0"
    text = f"{value:.{sig - 1}e}"
    return text.replace("e-0", "e-").replace("e+0", "e+")


def format_pm(mean: float, std: float, sig: int, bold: bool = False) -> str:
    text = f"{_format_sig(mean, sig)} $\\pm$ {_format_sci(std, sig)}"
    return f"\\textbf{{{text}}}" if bold else text


def find_run(subdir: str, include: Sequence[str], exclude: Sequence[str] | None = None) -> Path:
    exclude = exclude or []
    candidates: List[Path] = []
    for root in OUTPUT_ROOTS:
        base = root / subdir
        if not base.exists():
            continue
        for p in base.iterdir():
            if not p.is_dir():
                continue
            if not all(token in p.name for token in include):
                continue
            if any(token in p.name for token in exclude):
                continue
            candidates.append(p)
    if not candidates:
        raise FileNotFoundError(f"No run found under {subdir} with include={include} exclude={exclude}")
    return max(candidates, key=lambda p: p.name)


def find_run_multi(subdir: str, patterns: Sequence[str], extra_tokens: Sequence[str]) -> Path:
    last_err: Optional[Exception] = None
    for pattern in patterns:
        include = [pattern] + list(extra_tokens)
        try:
            return find_run(subdir, include)
        except FileNotFoundError as exc:
            last_err = exc
    if last_err is not None:
        raise last_err
    raise FileNotFoundError(f"No run found under {subdir} with patterns={patterns} tokens={extra_tokens}")


def find_seed_run(subdir: str, patterns: Sequence[str], lambda_tag: str, seed: int) -> Optional[Path]:
    token = f"lambda{lambda_tag}"
    seed_token = f"train_random_seed{seed}"
    candidates: List[Path] = []
    for root in OUTPUT_ROOTS:
        base = root / subdir
        if not base.is_dir():
            continue
        for d in base.iterdir():
            if not d.is_dir():
                continue
            name = d.name
            if token not in name:
                continue
            if not any(pat in name for pat in patterns):
                continue
            if seed == 42:
                if "train_random_seed" in name:
                    continue
            else:
                if seed_token not in name:
                    continue
            candidates.append(d)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.name)


def find_mc0_batch(run_dir: Path) -> Path:
    history = run_dir / "samples_history"
    if not history.is_dir():
        raise FileNotFoundError(f"samples_history not found: {history}")
    batches = sorted([p for p in history.iterdir() if p.is_dir() and p.name.startswith("batch-")])
    if not batches:
        raise FileNotFoundError(f"No batch directories under {history}")
    mc0 = [b for b in batches if b.name.endswith("-mc0")]
    return mc0[-1] if mc0 else batches[-1]


def iter_repeat_dirs(batch_dir: Path) -> List[Path]:
    repeats = []
    for d in sorted(batch_dir.iterdir()):
        if not d.is_dir():
            continue
        if re.search(r"r\d{2}", d.name):
            repeats.append(d)
    return repeats


def load_payload(samples_pt: Path, trajectory_index: int) -> Dict[str, np.ndarray]:
    data = torch.load(samples_pt, map_location="cpu", weights_only=False)
    samples = _to_np(data["samples"])
    truth = _to_np(data["truth"])
    context = _to_np(data.get("context")) if data.get("context") is not None else None
    if isinstance(samples, np.ndarray) and samples.ndim == 4:
        if samples.shape[0] == truth.shape[0]:
            samples = samples[:, trajectory_index % samples.shape[1], :, :]
        elif samples.shape[1] == truth.shape[0]:
            samples = samples[trajectory_index % samples.shape[0]]
        else:
            samples = samples[:, trajectory_index % samples.shape[1], :, :]
    payload = {"pred": samples, "truth": truth, "context": context}
    stage_counts = data.get("window_stage_counts")
    if stage_counts is not None:
        payload["stage_counts"] = {k: int(v) for k, v in stage_counts.items()}
    return payload


def metric_for_payload(
    payload: Dict[str, np.ndarray],
    assets: int,
    metric_name: str,
    stage: str,
    use_augmented_cov: bool,
    ref_contexts: Optional[Dict[int, np.ndarray]] = None,
) -> float:
    asset_indices = list(range(assets))
    use_correlation = metric_name.startswith("matrix_corr")
    _len, _mats, metrics = aggregate_model(
        ("time", "conditional"),
        payload,
        [],
        asset_indices,
        metric_name,
        stage,
        use_correlation,
        ref_contexts=ref_contexts,
        use_augmented_cov=use_augmented_cov,
    )
    return float(metrics["pred_minus_truth"])


def collect_values(
    run_dir: Path,
    assets: int,
    metric_name: str,
    stage: str,
    use_augmented_cov: bool,
    trajectory_index: int,
    ref_contexts: Optional[Dict[int, np.ndarray]] = None,
) -> List[float]:
    batch_dir = find_mc0_batch(run_dir)
    values: List[float] = []
    for sub in iter_repeat_dirs(batch_dir):
        samples_pt = sub / "samples.pt"
        if not samples_pt.exists():
            continue
        payload = load_payload(samples_pt, trajectory_index)
        if payload.get("context") is None and ref_contexts is not None:
            payload["context"] = None
        values.append(
            metric_for_payload(payload, assets, metric_name, stage, use_augmented_cov, ref_contexts=ref_contexts)
        )
    return values


def get_ref_contexts(
    run_dir: Path,
    trajectory_index: int,
) -> Optional[Dict[int, np.ndarray]]:
    batch_dir = find_mc0_batch(run_dir)
    repeats = iter_repeat_dirs(batch_dir)
    if not repeats:
        return None
    payload = load_payload(repeats[0] / "samples.pt", trajectory_index)
    context = payload.get("context")
    if context is None:
        return None
    return {idx: context[idx] for idx in range(context.shape[0])}


def main() -> None:
    args = parse_args()
    if str(args.lambda_tag) != "0":
        raise SystemExit("This table is defined for lambda=0 only. Use --lambda-tag 0.")
    metric_name = "matrix_cov_fro" if args.metric == "cov" else "matrix_corr_fro"
    use_augmented_cov = bool(args.use_ca)

    seeds = [42, 12, 22, 32]
    suffix = "+CA" if args.use_ca else ""
    results: Dict[str, Dict[str, Tuple[float, float]]] = {
        f"CDSM-T{suffix}": {},
        f"UCDSM-T{suffix}": {},
        f"CDSM-S{suffix}": {},
        f"UCDSM-S{suffix}": {},
    }

    for ds in DATASETS:
        patterns = ds["patterns"]
        assets = ds["assets"]

        # Reference contexts from conditional temporal run (seed 42) if needed.
        ref_contexts = None
        ref_run = find_seed_run("time/conditional", patterns, args.lambda_tag, 42)
        if ref_run is not None:
            ref_contexts = get_ref_contexts(ref_run, args.trajectory_index)

        # CDSM-T (conditional temporal)
        vals: List[float] = []
        for seed in seeds:
            run_dir = find_seed_run("time/conditional", patterns, args.lambda_tag, seed)
            if run_dir is None:
                continue
            vals.extend(
                collect_values(
                    run_dir,
                    assets,
                    metric_name,
                    args.stage,
                    use_augmented_cov,
                    args.trajectory_index,
                )
            )
        if vals:
            results[f"CDSM-T{suffix}"][ds["label"]] = (float(np.mean(vals)), float(np.std(vals)))

        # UCDSM-T (unconditional temporal)
        vals = []
        run_dir = find_run_multi("time/unconditional", patterns, [])
        vals.extend(
            collect_values(
                run_dir,
                assets,
                metric_name,
                args.stage,
                use_augmented_cov,
                args.trajectory_index,
                ref_contexts=ref_contexts,
            )
        )
        if vals:
            results[f"UCDSM-T{suffix}"][ds["label"]] = (float(np.mean(vals)), float(np.std(vals)))

        # CDSM-S (conditional spectral)
        vals = []
        for seed in seeds:
            run_dir = find_seed_run("fourier/conditional", patterns, args.lambda_tag, seed)
            if run_dir is None:
                continue
            vals.extend(
                collect_values(
                    run_dir,
                    assets,
                    metric_name,
                    args.stage,
                    use_augmented_cov,
                    args.trajectory_index,
                )
            )
        if vals:
            results[f"CDSM-S{suffix}"][ds["label"]] = (float(np.mean(vals)), float(np.std(vals)))

        # UCDSM-S (unconditional spectral)
        vals = []
        run_dir = find_run_multi("fourier/unconditional", patterns, [])
        vals.extend(
            collect_values(
                run_dir,
                assets,
                metric_name,
                args.stage,
                use_augmented_cov,
                args.trajectory_index,
                ref_contexts=ref_contexts,
            )
        )
        if vals:
            results[f"UCDSM-S{suffix}"][ds["label"]] = (float(np.mean(vals)), float(np.std(vals)))

    # Bold per pair (CDSM vs UCDSM) within T and S blocks.
    bold_flags: Dict[Tuple[str, str], bool] = {}
    for ds in DATASETS:
        label = ds["label"]
        t_key = f"CDSM-T{suffix}"
        ut_key = f"UCDSM-T{suffix}"
        s_key = f"CDSM-S{suffix}"
        us_key = f"UCDSM-S{suffix}"
        t_vals = results.get(t_key, {}).get(label)
        ut_vals = results.get(ut_key, {}).get(label)
        if t_vals and ut_vals:
            if t_vals[0] <= ut_vals[0]:
                bold_flags[(t_key, label)] = True
            else:
                bold_flags[(ut_key, label)] = True

        s_vals = results.get(s_key, {}).get(label)
        us_vals = results.get(us_key, {}).get(label)
        if s_vals and us_vals:
            if s_vals[0] <= us_vals[0]:
                bold_flags[(s_key, label)] = True
            else:
                bold_flags[(us_key, label)] = True

    args.out_tex.parent.mkdir(parents=True, exist_ok=True)
    with args.out_tex.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")
        if args.use_ca:
            f.write(
                "\\multicolumn{5}{l}{\\footnotesize CDSM uses $\\lambda=0$; UCDSM has no $\\lambda$. "
                "CA concatenates context+prediction. } \\\\\n"
            )
        else:
            f.write("\\multicolumn{5}{l}{\\footnotesize CDSM uses $\\lambda=0$; UCDSM has no $\\lambda$. } \\\\\n")
        f.write("Model & " + " & ".join(d["label"] for d in DATASETS) + " \\\\\n")
        f.write("\\midrule\n")

        for model in [f"CDSM-T{suffix}", f"UCDSM-T{suffix}"]:
            row = [model]
            for ds in DATASETS:
                vals = results.get(model, {}).get(ds["label"])
                if vals is None:
                    row.append("--")
                    continue
                mean, std = vals
                bold = bold_flags.get((model, ds["label"]), False)
                row.append(format_pm(mean, std, args.sig_digits, bold))
            f.write(" & ".join(row) + " \\\\ \n")

        f.write("\\midrule\n")

        for model in [f"CDSM-S{suffix}", f"UCDSM-S{suffix}"]:
            row = [model]
            for ds in DATASETS:
                vals = results.get(model, {}).get(ds["label"])
                if vals is None:
                    row.append("--")
                    continue
                mean, std = vals
                bold = bold_flags.get((model, ds["label"]), False)
                row.append(format_pm(mean, std, args.sig_digits, bold))
            f.write(" & ".join(row) + " \\\\ \n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

    print(f"Saved {args.out_tex}")


if __name__ == "__main__":
    main()
