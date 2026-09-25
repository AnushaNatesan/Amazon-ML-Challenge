#!/usr/bin/env python3
"""
Pairwise Feature Extraction Runner
Part of Phase 4 Implementation for Business Entity Resolution
Team: GenX H4CK3RS!

Reads candidate_pairs.tsv and source TSV files, then computes the 33-dimensional
feature matrix for candidate pairs (with ground truth binary labels if in validation mode).
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set
import pandas as pd

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Add module to sys.path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from src.data_loader import load_source_tsv, load_ground_truth_tsv, resolve_data_paths
from src.features import extract_features_for_candidate_pairs, FEATURE_NAMES


def run_feature_extraction_pipeline(
    candidates_path: Optional[str] = None,
    mode: str = "val",
    sample_size: int = 1000,
    output_path: Optional[str] = None,
):
    """
    Executes end-to-end feature extraction across candidate pairs:
    - candidates_path: Path to candidate_pairs.tsv (defaults to output/candidate_pairs.tsv).
    - mode: 'val' (includes training labels from train_ground_truth.tsv) or 'test'.
    - sample_size: Maximum number of Source 1 entities to extract features for.
    - output_path: Where to save extracted feature matrix (.csv or .parquet).
    """
    start_time = time.time()
    paths = resolve_data_paths()
    workspace_root = current_dir.parent.parent

    if candidates_path is None:
        candidates_path = workspace_root / "output" / "candidate_pairs.tsv"
    cand_p = Path(candidates_path)

    if not cand_p.is_file():
        raise FileNotFoundError(f"Candidate pairs file not found at: {cand_p}. Run candidate blocking first.")

    print("=" * 70)
    print(f"   ML Challenge 2026 — Phase 4 Pairwise Feature Extraction (Mode: {mode.upper()})")
    print(f"   Team: GenX H4CK3RS!")
    print("=" * 70)

    # 1. Read candidates from candidate_pairs.tsv
    print(f"[*] Reading candidate pairs from: {cand_p}")
    candidate_dict: Dict[str, List[str]] = {}
    target_ids_needed: Set[str] = set()
    s1_ids_needed: List[str] = []

    with open(cand_p, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            s1_id = parts[0].strip()
            cands_str = parts[1].strip() if len(parts) > 1 else ""
            cands = [c.strip() for c in cands_str.split(",") if c.strip()]
            candidate_dict[s1_id] = cands
            s1_ids_needed.append(s1_id)
            target_ids_needed.update(cands)

            if sample_size and len(s1_ids_needed) >= sample_size:
                break

    print(f"[+] Loaded {len(s1_ids_needed):,} Source 1 entities with {len(target_ids_needed):,} unique target candidates.")

    # 2. Load Source 1 records
    s1_file = paths["train"]["source1"] if mode == "val" else paths["test"]["source1"]
    print(f"[*] Loading Source 1 records from: {s1_file.name}")
    s1_df = load_source_tsv(s1_file)
    s1_df = s1_df[s1_df["entity_id"].isin(set(s1_ids_needed))].reset_index(drop=True)

    # 3. Load Target pool records (Source 2 and Source 3)
    target_frames = []
    for s_name in ["source2", "source3"]:
        target_file = paths["train"][s_name] if mode == "val" else paths["test"][s_name]
        print(f"[*] Scanning {target_file.name} for candidate records...")
        for chunk in load_source_tsv(target_file, chunksize=250000):
            sub = chunk[chunk["entity_id"].isin(target_ids_needed)]
            if len(sub) > 0:
                target_frames.append(sub)

    target_pool_df = pd.concat(target_frames, ignore_index=True).drop_duplicates(subset=["entity_id"])
    print(f"[+] Loaded {len(target_pool_df):,} target candidate records.")

    # 4. Load ground truth for validation labels
    gt_dict = None
    if mode == "val":
        print("[*] Loading ground truth for binary match labeling...")
        gt_df = load_ground_truth_tsv(paths["train"]["ground_truth"])
        gt_dict = {
            r["source1_entity_id"]: r["matched_set"]
            for _, r in gt_df.iterrows()
            if r["source1_entity_id"] in set(s1_ids_needed)
        }

    # 5. Extract pairwise features
    features_df, labels, pair_ids = extract_features_for_candidate_pairs(
        candidate_dict,
        s1_df,
        target_pool_df,
        ground_truth_dict=gt_dict,
    )

    # Attach entity pair IDs and labels
    features_df["source1_entity_id"] = [p[0] for p in pair_ids]
    features_df["candidate_entity_id"] = [p[1] for p in pair_ids]
    if labels is not None:
        features_df["is_match"] = labels

    # 6. Save feature matrix
    if output_path is None:
        output_path = workspace_root / "output" / f"features_{mode}_sample.csv"
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(out_p, index=False)
    print(f"\n[+] Feature matrix saved to: {out_p} ({out_p.stat().st_size / 1024:.1f} KB)")

    elapsed = time.time() - start_time
    print(f"[+] Feature extraction completed in {elapsed:.1f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pairwise Feature Extraction Runner")
    parser.add_argument("--candidates", type=str, default=None, help="Path to candidate_pairs.tsv")
    parser.add_argument("--mode", choices=["val", "test"], default="val", help="Run mode (val or test)")
    parser.add_argument("--sample-size", type=int, default=500, help="Number of S1 entities to process")
    parser.add_argument("--output", type=str, default=None, help="Output destination for feature matrix")
    args = parser.parse_args()

    run_feature_extraction_pipeline(
        candidates_path=args.candidates,
        mode=args.mode,
        sample_size=args.sample_size,
        output_path=args.output,
    )
