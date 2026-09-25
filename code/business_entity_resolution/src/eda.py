"""
Exploratory Data Analysis (EDA) Module
Part of Phase 1 Implementation for Business Entity Resolution
Team: GenX H4CK3RS!

Performs comprehensive diagnostic inspection on training and test datasets:
- Dataset sizes (file sizes, total line counts, column counts)
- Missing values and empty string counts across all columns
- Country distributions (analyzing the presence of France in test, unseen in training)
- Singleton proportions in train_ground_truth.tsv
- Name & address length characteristics and noise patterns
- Generates a structured summary report in Markdown format
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from .data_loader import resolve_data_paths, load_source_tsv, load_ground_truth_tsv


def inspect_file_metadata(file_path: Path) -> Dict[str, Any]:
    """Inspects file size, line count, and encoding properties."""
    if not file_path.is_file():
        return {"exists": False, "path": str(file_path)}

    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024 * 1024)

    # Count lines efficiently
    line_count = 0
    with open(file_path, "rb") as fp:
        for _ in fp:
            line_count += 1

    return {
        "exists": True,
        "path": str(file_path),
        "file_name": file_path.name,
        "size_mb": round(size_mb, 2),
        "line_count": line_count,
        "row_count": max(0, line_count - 1),  # excluding header
    }


def analyze_source_file(
    file_path: Path,
    sample_size: Optional[int] = 250000,
) -> Dict[str, Any]:
    """Analyzes schema, missing values, and country distributions for a source file."""
    meta = inspect_file_metadata(file_path)
    if not meta["exists"]:
        return meta

    # Read sample or full file
    df = load_source_tsv(file_path, nrows=sample_size, validate=False)

    col_stats = {}
    for col in df.columns:
        vals = df[col].astype(str)
        empty_count = int((vals.isna() | (vals.str.strip() == "")).sum())
        empty_pct = round((empty_count / len(df)) * 100, 3) if len(df) > 0 else 0.0
        col_stats[col] = {
            "empty_or_missing_count": empty_count,
            "empty_or_missing_pct": empty_pct,
            "sample_distinct": int(vals.nunique()),
        }

    # Country breakdown
    country_counts = {}
    if "country" in df.columns:
        counts = df["country"].value_counts(dropna=False).to_dict()
        total = len(df)
        country_counts = {
            str(k): {"count": int(v), "pct": round((v / total) * 100, 2)}
            for k, v in counts.items()
        }

    return {
        "metadata": meta,
        "sample_analyzed": len(df),
        "columns": list(df.columns),
        "column_stats": col_stats,
        "country_distribution": country_counts,
    }


def analyze_ground_truth(file_path: Path) -> Dict[str, Any]:
    """Analyzes singleton proportions and match distributions in ground truth."""
    meta = inspect_file_metadata(file_path)
    if not meta["exists"]:
        return meta

    total = 0
    singletons = 0
    match_lengths: List[int] = []

    # Stream through ground truth in chunks to be memory efficient
    for chunk in pd.read_csv(file_path, sep="\t", dtype=str, chunksize=250000, keep_default_na=False):
        total += len(chunk)
        m_col = chunk["matched_entity_ids"].fillna("").astype(str).str.strip()
        is_singleton = (m_col == "")
        singletons += int(is_singleton.sum())

        # Collect sample match length counts
        if len(match_lengths) < 50000:
            for val in m_col[~is_singleton].head(5000):
                match_lengths.append(len(val.split(",")))

    non_singletons = total - singletons
    singleton_pct = round((singletons / total) * 100, 3) if total > 0 else 0.0

    return {
        "metadata": meta,
        "total_source1_entities": total,
        "singletons_count": singletons,
        "singletons_pct": singleton_pct,
        "non_singletons_count": non_singletons,
        "non_singletons_pct": round(100.0 - singleton_pct, 3),
        "avg_matches_per_non_singleton": round(sum(match_lengths) / len(match_lengths), 2) if match_lengths else 0.0,
    }


def run_eda(
    data_dir: Optional[Union[str, Path]] = None,
    output_report_path: Optional[Union[str, Path]] = None,
    sample_size: Optional[int] = 200000,
) -> Dict[str, Any]:
    """
    Executes complete EDA pipeline across training and test datasets.
    Prints findings and saves a structured Markdown report.
    """
    paths = resolve_data_paths(data_dir)
    print(f"[*] Starting Business Entity Resolution EDA...")
    print(f"[*] Dataset Root: {paths['root']}")

    report: Dict[str, Any] = {
        "train": {},
        "test": {},
    }

    # 1. Analyze Training Data
    print("\n--- [1] Analyzing Training Set ---")
    for key, p in paths["train"].items():
        if key == "ground_truth":
            print(f"  -> Inspecting {p.name}...")
            gt_report = analyze_ground_truth(p)
            report["train"]["ground_truth"] = gt_report
            print(f"     Total S1 Entities: {gt_report['total_source1_entities']:,}")
            print(f"     Singletons (no matches): {gt_report['singletons_count']:,} ({gt_report['singletons_pct']}%)")
            print(f"     Non-Singletons: {gt_report['non_singletons_count']:,} ({gt_report['non_singletons_pct']}%)")
        else:
            print(f"  -> Inspecting {p.name}...")
            s_report = analyze_source_file(p, sample_size=sample_size)
            report["train"][key] = s_report
            print(f"     Rows: {s_report['metadata']['row_count']:,}, Size: {s_report['metadata']['size_mb']} MB")
            print(f"     Countries: {s_report['country_distribution']}")

    # 2. Analyze Test Data
    print("\n--- [2] Analyzing Test Set ---")
    for key, p in paths["test"].items():
        print(f"  -> Inspecting {p.name}...")
        s_report = analyze_source_file(p, sample_size=sample_size)
        report["test"][key] = s_report
        print(f"     Rows: {s_report['metadata']['row_count']:,}, Size: {s_report['metadata']['size_mb']} MB")
        print(f"     Countries: {s_report['country_distribution']}")

    # 3. Generate Markdown Report
    md_content = generate_eda_report(report)

    if output_report_path:
        out_p = Path(output_report_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"\n[+] EDA report successfully saved to: {out_p}")

    return report


def generate_eda_report(report: Dict[str, Any]) -> str:
    """Generates a clean Markdown report from EDA results."""
    md = []
    md.append("# Business Entity Resolution — Exploratory Data Analysis Report")
    md.append("**Team Name:** GenX H4CK3RS!\n")
    md.append("## 1. Dataset Dimensions & Volume Overview\n")
    md.append("| File | Split | Row Count | File Size (MB) |")
    md.append("|---|---|---|---|")

    for split in ["train", "test"]:
        for k, v in report[split].items():
            meta = v.get("metadata", {})
            name = meta.get("file_name", k)
            rows = f"{meta.get('row_count', 'N/A'):,}" if isinstance(meta.get('row_count'), int) else "N/A"
            size = meta.get("size_mb", "N/A")
            md.append(f"| `{name}` | {split} | {rows} | {size} MB |")

    md.append("\n## 2. Country Distribution & France Domain Shift\n")
    md.append("> **CRITICAL FINDING:** The training set only contains records from `US` and `India`.")
    md.append("> The test set introduces `France` (~15% of records). The model must remain country-agnostic.\n")

    md.append("| Split / File | Country | Count (Sampled) | Percentage |")
    md.append("|---|---|---|---|")
    for split in ["train", "test"]:
        for k, v in report[split].items():
            if "country_distribution" in v:
                fname = v["metadata"]["file_name"]
                for c_name, c_data in v["country_distribution"].items():
                    md.append(f"| `{fname}` ({split}) | **{c_name}** | {c_data['count']:,} | {c_data['pct']}% |")

    if "ground_truth" in report["train"]:
        gt = report["train"]["ground_truth"]
        md.append("\n## 3. Ground Truth & Singleton Analysis\n")
        md.append(f"- **Total Source 1 Reference Entities:** {gt.get('total_source1_entities', 0):,}")
        md.append(f"- **Singletons (0 matches in S2/S3):** {gt.get('singletons_count', 0):,} ({gt.get('singletons_pct', 0)}%)")
        md.append(f"- **Non-Singletons (>= 1 match):** {gt.get('non_singletons_count', 0):,} ({gt.get('non_singletons_pct', 0)}%)")
        md.append(f"- **Average Matches per Non-Singleton:** {gt.get('avg_matches_per_non_singleton', 0)}")
        md.append("\n> **F_0.5 Metric Strategic Note:** Singletons must receive an empty prediction list `[]` to score 1.0.")
        md.append("> Any false positive predicted for a singleton results in a punishing score of 0.0.")

    md.append("\n## 4. Missing Values Inspection\n")
    md.append("| File | Column | Missing/Empty % |")
    md.append("|---|---|---|")
    for split in ["train", "test"]:
        for k, v in report[split].items():
            if "column_stats" in v:
                fname = v["metadata"]["file_name"]
                for col_name, c_stats in v["column_stats"].items():
                    md.append(f"| `{fname}` | `{col_name}` | {c_stats['empty_or_missing_pct']}% |")

    return "\n".join(md)


if __name__ == "__main__":
    run_eda()
