#!/usr/bin/env python3
"""
Aggregate bull/bear counts from regime stats JSON files under assets/paper.

Example:
python scripts/make_regime_summary.py \
  --root assets/paper \
  --out-md assets/regime_test_summary.md \
  --out-tex assets/regime_test_summary.tex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "assets" / "paper",
        help="Root directory containing regime stats JSON files.",
    )
    ap.add_argument("--out-md", type=Path, default=Path("assets/regime_test_summary.md"))
    ap.add_argument("--out-tex", type=Path, default=Path("assets/regime_test_summary.tex"))
    return ap.parse_args()


def _latex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def _format_ratio(value: float) -> str:
    return f"{value:.2f}"


def _collect_rows(root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not root.exists():
        raise FileNotFoundError(f"Root does not exist: {root}")
    for path in sorted(root.rglob("window_regimes_*_test_stats_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        bull = int(data.get("bull", 0))
        bear = int(data.get("bear", 0))
        total = int(data.get("total_windows", bull + bear))
        if total <= 0:
            continue
        group = path.parents[1].name
        exp = path.parent.name
        rows.append(
            {
                "Dataset": group,
                "Experiment": exp,
                "Bull": bull,
                "Bear": bear,
                "Total": total,
                "BullRatio": bull / total if total else 0.0,
                "BearRatio": bear / total if total else 0.0,
            }
        )
    rows.sort(key=lambda r: (r["Dataset"], r["Experiment"]))
    return rows


def _write_md(rows: List[Dict[str, object]], out_path: Path) -> None:
    headers = ["Dataset", "Experiment", "Bull", "Bear", "Total", "Bull%", "Bear%"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["Dataset"]),
                    str(row["Experiment"]),
                    str(row["Bull"]),
                    str(row["Bear"]),
                    str(row["Total"]),
                    _format_ratio(float(row["BullRatio"])),
                    _format_ratio(float(row["BearRatio"])),
                ]
            )
            + " |"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _write_tex(rows: List[Dict[str, object]], out_path: Path) -> None:
    lines = []
    lines.append(r"\begin{tabular}{@{}llrrrrr@{}}")
    lines.append(r" \toprule")
    lines.append(r"Dataset \ & Experiment \ & Bull \ & Bear \ & Total \ & Bull\% \ & Bear\% \\")
    lines.append(r" \midrule")
    for row in rows:
        lines.append(
            " ".join(
                [
                    _latex_escape(str(row["Dataset"])),
                    r"\ & ",
                    _latex_escape(str(row["Experiment"])),
                    r"\ & ",
                    str(row["Bull"]),
                    r"\ & ",
                    str(row["Bear"]),
                    r"\ & ",
                    str(row["Total"]),
                    r"\ & ",
                    _format_ratio(float(row["BullRatio"])),
                    r"\ & ",
                    _format_ratio(float(row["BearRatio"])),
                    r" \\",
                ]
            )
        )
    lines.append(r" \bottomrule")
    lines.append(r"\end{tabular}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = _collect_rows(args.root)
    if not rows:
        raise SystemExit(f"No test stats found under {args.root}")
    _write_md(rows, args.out_md)
    _write_tex(rows, args.out_tex)
    print(f"[DONE] Wrote {args.out_md} and {args.out_tex}")


if __name__ == "__main__":
    main()
