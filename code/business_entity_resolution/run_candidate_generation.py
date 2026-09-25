#!/usr/bin/env python3
"""
Candidate Generation (Blocking) Pipeline Runner
Part of Phase 3 Implementation for Business Entity Resolution
Team: GenX H4CK3RS!

Executes country-partitioned multi-pass inverted index blocking:
1. Hard Country Partitioning (US, India, France).
2. Multi-Pass Keys (Core Name, Building No + Name/Address, Postal + Name, 4-gram Prefix).
3. Candidate Ranking and Pruning (capped at top K candidates).
4. Exports candidates to output/candidate_pairs.tsv.
5. Computes validation recall ceiling on ground truth matches.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional
import pandas as pd

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Add module to sys.path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from src.blocking import (
    MultiPassBlocker,
    evaluate_blocking_recall,
    export_candidate_pairs_tsv,
)
from src.data_loader import (
    load_source_tsv,
    load_ground_truth_tsv,
    resolve_data_paths,
)
from src.metrics import create_validation_split


def run_blocking_pipeline(
    mode: str = "val",
    sample_size: int = 50000,
    max_candidates: int = 40,
    output_path: Optional[str] = None,
):
    """
    Executes end-to-end blocking pass:
    - mode: 'val' (evaluate recall ceiling on 20% validation split) or 'test' (generate test candidates).
    - sample_size: Number of Source 1 entities to evaluate (None for full split).
    - max_candidates: Maximum candidates per S1 entity.
    - output_path: Destination path for candidate_pairs.tsv.
    """
    start_time = time.time()
    paths = resolve_data_paths()
    workspace_root = current_dir.parent.parent
    if output_path is None:
        output_path = workspace_root / "output" / "candidate_pairs.tsv"

    print("=" * 70)
    print(f"   ML Challenge 2026 — Phase 3 Candidate Generation (Mode: {mode.upper()})")
    print(f"   Team: GenX H4CK3RS!")
    print("=" * 70)

    if mode == "val":
        print(f"[*] Loading training datasets for validation blocking evaluation...")
        s1_df = load_source_tsv(paths["train"]["source1"], nrows=sample_size)
        gt_df = load_ground_truth_tsv(paths["train"]["ground_truth"], nrows=sample_size)

        # 20% hold-out validation split
        s1_train, s1_val, gt_train, gt_val = create_validation_split(
            s1_df, gt_df, val_ratio=0.20, random_state=42
        )
        print(f"[+] Local 20% Validation Split Created: {len(s1_val):,} S1 Validation Entities.")

        # Ground truth mapping for validation entities
        val_s1_ids = set(s1_val["entity_id"])
        gt_val_dict = {}
        target_ids_needed = set()
        for _, r in gt_val.iterrows():
            s1_id = r["source1_entity_id"]
            m_set = r["matched_set"]
            gt_val_dict[s1_id] = m_set
            target_ids_needed.update(m_set)

        print(f"[*] Target true matches needed in pool: {len(target_ids_needed):,}")

        # Load true target records and background distractors from Source 2 and Source 3
        matched_target_chunks = []
        distractor_chunks = []
        found_target_ids = set()

        for s_name in ["source2", "source3"]:
            print(f"[*] Scanning {s_name} for ground-truth targets and background distractors...")
            for chunk in load_source_tsv(paths["train"][s_name], chunksize=250000):
                # Extract true target records present in this chunk
                m_sub = chunk[chunk["entity_id"].isin(target_ids_needed)]
                if len(m_sub) > 0:
                    matched_target_chunks.append(m_sub)
                    found_target_ids.update(m_sub["entity_id"])
                # Collect background distractors
                if len(distractor_chunks) < 4:
                    distractor_chunks.append(chunk.head(10000))
                # Stop early if all targets are found
                if len(found_target_ids) >= len(target_ids_needed) and len(distractor_chunks) >= 4:
                    break

        all_target_frames = matched_target_chunks + distractor_chunks
        target_pool = pd.concat(all_target_frames, ignore_index=True).drop_duplicates(subset=["entity_id"])
        print(f"[+] Target pool size: {len(target_pool):,} records (Found {len(found_target_ids):,} / {len(target_ids_needed):,} true targets).")


        # Initialize blocker
        blocker = MultiPassBlocker(
            max_candidates_per_entity=max_candidates,
            max_bucket_size=5000,
        )
        blocker.fit_target_pool(target_pool)

        # Generate candidates for S1 validation entities
        candidates = blocker.block_dataframe(s1_val)

        # Evaluate recall ceiling
        eval_metrics = evaluate_blocking_recall(candidates, gt_val_dict)
        print("\n" + "=" * 70)
        print("   BLOCKING RECALL CEILING EVALUATION RESULTS")
        print("=" * 70)
        print(f"  Recall Ceiling:             {eval_metrics['recall_ceiling'] * 100:.2f}%")
        print(f"  Entity Coverage:            {eval_metrics['entity_coverage'] * 100:.2f}%")
        print(f"  Total True Matches:         {eval_metrics['total_true_matches']:,}")
        print(f"  Captured True Matches:      {eval_metrics['captured_true_matches']:,}")
        print(f"  Avg Candidates per Entity:  {eval_metrics['avg_candidates_per_entity']}")
        print(f"  Singleton Retention Rate:   {eval_metrics['singleton_retention_rate'] * 100:.2f}%")
        print("=" * 70)

        # Export candidates
        export_candidate_pairs_tsv(candidates, output_path)

    elif mode == "test":
        print(f"[*] Loading test datasets for candidate generation...")
        test_s1 = load_source_tsv(paths["test"]["source1"], nrows=sample_size)
        test_s2 = load_source_tsv(paths["test"]["source2"], nrows=sample_size * 3 if sample_size else None)
        test_s3 = load_source_tsv(paths["test"]["source3"], nrows=sample_size * 3 if sample_size else None)

        target_pool = pd.concat([test_s2, test_s3], ignore_index=True)
        print(f"[+] Test target pool: {len(target_pool):,} records.")

        blocker = MultiPassBlocker(
            max_candidates_per_entity=max_candidates,
            max_bucket_size=5000,
        )
        blocker.fit_target_pool(target_pool)
        candidates = blocker.block_dataframe(test_s1)
        export_candidate_pairs_tsv(candidates, output_path)

    elapsed = time.time() - start_time
    print(f"\n[+] Blocking stage completed in {elapsed:.1f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Candidate Generation (Blocking) Pipeline")
    parser.add_argument("--mode", choices=["val", "test"], default="val", help="Run mode (val or test)")
    parser.add_argument("--sample-size", type=int, default=20000, help="Number of S1 entities to process")
    parser.add_argument("--max-candidates", type=int, default=40, help="Max candidates per entity")
    parser.add_argument("--output", type=str, default=None, help="Custom output path for candidate_pairs.tsv")
    args = parser.parse_args()

    run_blocking_pipeline(
        mode=args.mode,
        sample_size=args.sample_size,
        max_candidates=args.max_candidates,
        output_path=args.output,
    )
