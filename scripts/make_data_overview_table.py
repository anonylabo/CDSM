#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
import re


def clean_cell(cell: str) -> str:
    cell = cell.strip()
    if cell.endswith("\\"):
        cell = cell[:-1].strip()
    cell = cell.replace("\\_", "_")
    return cell


def strip_row_suffix(line: str) -> str:
    line = line.split("%", 1)[0].strip()
    line = re.sub(r"\\\\\s*$", "", line).strip()
    return line


def parse_tabular_rows(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("\\begin{tabular}") or line.startswith("\\end{tabular}"):
            continue
        if line.startswith("\\toprule") or line.startswith("\\midrule") or line.startswith("\\bottomrule"):
            continue
        if "&" not in line:
            continue
        line = strip_row_suffix(line)
        parts = [clean_cell(p) for p in line.split("&")]
        rows.append(parts)
    return rows


def build_header_map(header_row: list[str]) -> dict[str, int]:
    header = [clean_cell(h) for h in header_row]
    return {h: i for i, h in enumerate(header)}


def parse_window_tuple(window_str: str) -> tuple[int, int]:
    match = re.search(r"\((\d+)\s*,\s*(\d+)\)", window_str)
    if not match:
        raise ValueError(f"Unrecognized window format: {window_str}")
    return int(match.group(1)), int(match.group(2))


def format_period(train_period: str, test_period: str) -> str:
    train = train_period.replace(" to ", "--")
    test = test_period.replace(" to ", "--")
    return f"\\shortstack{{{train}\\\\{test}}}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a compact dataset overview table from dataset_summary and window_counts."
    )
    parser.add_argument(
        "--dataset-summary",
        default="assets/dataset_summary.tex",
        help="Path to dataset_summary.tex",
    )
    parser.add_argument(
        "--window-counts",
        default="assets/window_counts.tex",
        help="Path to window_counts.tex",
    )
    parser.add_argument(
        "--out",
        default="assets/data_overview.tex",
        help="Output path for the merged table (LaTeX tabular).",
    )
    args = parser.parse_args()

    dataset_summary_path = Path(args.dataset_summary)
    window_counts_path = Path(args.window_counts)
    out_path = Path(args.out)

    summary_rows = parse_tabular_rows(dataset_summary_path)
    if not summary_rows:
        raise SystemExit(f"No rows parsed from {dataset_summary_path}")
    summary_header = build_header_map(summary_rows[0])
    summary_rows = summary_rows[1:]

    counts_rows = parse_tabular_rows(window_counts_path)
    if not counts_rows:
        raise SystemExit(f"No rows parsed from {window_counts_path}")
    counts_header = build_header_map(counts_rows[0])
    counts_rows = counts_rows[1:]

    counts_map: dict[tuple[str, int, int], dict[str, str]] = {}
    for row in counts_rows:
        dataset = row[counts_header["Dataset"]]
        c_val = int(row[counts_header["C"]])
        p_val = int(row[counts_header["P"]])
        counts_map[(dataset, c_val, p_val)] = {
            "train": row[counts_header["Train"]],
            "val": row[counts_header["Val"]],
            "test": row[counts_header["Test"]],
            "val_ratio": row[counts_header["val_ratio"]],
            "train_ratio": row[counts_header["train_ratio"]],
        }

    merged_rows: list[dict[str, str]] = []
    for row in summary_rows:
        dataset = row[summary_header["Dataset"]]
        train_period = row[summary_header["Period (train)"]]
        test_period = row[summary_header["Period (test)"]]
        assets = row[summary_header["Assets (N)"]]
        c_val, p_val = parse_window_tuple(row[summary_header["Window (C, P)"]])
        counts = counts_map.get((dataset, c_val, p_val))
        if counts is None:
            raise SystemExit(f"Missing window counts for {dataset} ({c_val}, {p_val})")
        merged_rows.append(
            {
                "dataset": dataset,
                "period_key": f"{train_period}|{test_period}",
                "period_text": format_period(train_period, test_period),
                "assets": assets,
                "c": str(c_val),
                "p": str(p_val),
                "train": counts["train"],
                "val": counts["val"],
                "test": counts["test"],
                "val_ratio": counts["val_ratio"],
                "train_ratio": counts["train_ratio"],
            }
        )

    grouped: "OrderedDict[str, list[dict[str, str]]]" = OrderedDict()
    for row in merged_rows:
        grouped.setdefault(row["dataset"], []).append(row)

    lines: list[str] = []
    lines.append("\\begin{tabular}{@{}llccccc@{}}")
    lines.append(" \\toprule")
    lines.append(
        "Dataset \\ & \\shortstack{Period\\\\(train/test)} \\ & N \\ & C/P \\ & "
        "\\shortstack{Train/\\\\Val/Test} \\ & val\\_ratio \\ & train\\_ratio \\\\ "
    )
    lines.append(" \\midrule")

    dataset_items = list(grouped.items())
    for dataset_idx, (dataset, rows) in enumerate(dataset_items):
        dataset_span = len(rows)
        period_run_start: dict[int, int] = {}
        i = 0
        while i < len(rows):
            key = rows[i]["period_key"]
            j = i + 1
            while j < len(rows) and rows[j]["period_key"] == key:
                j += 1
            period_run_start[i] = j - i
            i = j

        for row_idx, row in enumerate(rows):
            dataset_cell = (
                f"\\multirow{{{dataset_span}}}{{*}}{{{dataset}}}" if row_idx == 0 else ""
            )
            if row_idx in period_run_start:
                period_cell = f"\\multirow{{{period_run_start[row_idx]}}}{{*}}{{{row['period_text']}}}"
            else:
                period_cell = ""

            cells = [
                dataset_cell,
                period_cell,
                row["assets"],
                f"{row['c']}/{row['p']}",
                f"{row['train']}/{row['val']}/{row['test']}",
                row["val_ratio"],
                row["train_ratio"],
            ]
            lines.append(" \\ & ".join(cells) + " \\\\ ")

        if dataset_idx < len(dataset_items) - 1:
            lines.append(" \\midrule")

    lines.append(" \\bottomrule")
    lines.append("\\end{tabular}")

    out_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
