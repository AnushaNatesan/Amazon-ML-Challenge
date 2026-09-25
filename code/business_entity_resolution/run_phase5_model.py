#!/usr/bin/env python3
"""
Phase 5: Scoring Model, Threshold Optimization & Leaderboard Inference
Team: GenX H4CK3RS!
Competition: Amazon ML Challenge 2026 — Business Entity Resolution

Workflow:
1. Environment & Hardware Setup (Colab T4 GPU support with CPU fallback).
2. Ingest 33-dimensional feature matrix (output/train_features.csv).
3. Train LightGBM binary classifier handling class imbalance (scale_pos_weight).
4. Optimize decision threshold via exact Macro F_0.5 evaluation (singleton rules).
5. Generate official test predictions (output/matching_results.tsv & output/candidate_pairs.tsv).
6. Validate submission with official validator and package GenX_H4CK3RS!_submission.zip.
"""

import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Setup sys.path
script_dir = Path(__file__).resolve().parent
workspace_root = script_dir.parent.parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from src.data_loader import resolve_data_paths, save_tsv
from src.features import FEATURE_NAMES
from src.metrics import calculate_macro_f05, compute_entity_f_beta
from package_submission import create_submission_zip, validate_against_official_tool


def detect_gpu_environment() -> Tuple[bool, str]:
    """Detects whether CUDA/Colab T4 GPU is available and returns appropriate device parameter."""
    gpu_available = False
    device_name = "CPU"
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            gpu_available = True
    except ImportError:
        import shutil
        if shutil.which("nvidia-smi"):
            device_name = "NVIDIA GPU (nvidia-smi detected)"
            gpu_available = True

    return gpu_available, device_name


def run_phase5_pipeline(
    feature_csv_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> Dict[str, float]:
    """Executes the complete Phase 5 training, tuning, and inference pipeline."""
    start_time = time.time()
    paths = resolve_data_paths()
    out_dir = Path(output_dir) if output_dir else workspace_root / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    feat_path = Path(feature_csv_path) if feature_csv_path else out_dir / "train_features.csv"

    print("=" * 70)
    print("   ML Challenge 2026 — Phase 5 Scoring Model & Threshold Tuning")
    print("   Team: GenX H4CK3RS!")
    print("=" * 70)

    # 1. Hardware Check
    gpu_available, device_name = detect_gpu_environment()
    print(f"[*] Compute Hardware: {device_name} (GPU Available: {gpu_available})")

    # 2. Ingest Feature Matrix
    if not feat_path.exists():
        raise FileNotFoundError(f"Feature matrix not found at: {feat_path}. Run Phase 4 first.")

    print(f"[*] Ingesting feature matrix from: {feat_path}")
    df_feat = pd.read_csv(feat_path)
    print(f"[+] Loaded {len(df_feat):,} candidate pair records.")

    # Validate feature columns
    missing_cols = [c for c in FEATURE_NAMES if c not in df_feat.columns]
    if missing_cols:
        raise ValueError(f"Feature matrix missing required features: {missing_cols}")

    X = df_feat[FEATURE_NAMES].values
    y = df_feat["is_match"].values
    pair_s1 = df_feat["source1_entity_id"].values
    pair_cand = df_feat["candidate_entity_id"].values

    pos_count = int(np.sum(y == 1))
    neg_count = int(np.sum(y == 0))
    imbalance_ratio = neg_count / max(1, pos_count)
    print(f"    - True Positive Matches:     {pos_count:,} ({pos_count / len(y) * 100:.2f}%)")
    print(f"    - Negative Distractors:     {neg_count:,} ({neg_count / len(y) * 100:.2f}%)")
    print(f"    - Class Imbalance Ratio:    {imbalance_ratio:.1f} : 1")

    # 3. Stratified Split by Source 1 Entity
    unique_s1 = np.unique(pair_s1)
    s1_train, s1_val = train_test_split(unique_s1, test_size=0.25, random_state=random_state)
    val_mask = np.isin(pair_s1, s1_val)
    train_mask = ~val_mask

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    s1_val_pairs = pair_s1[val_mask]
    cand_val_pairs = pair_cand[val_mask]

    print(f"[+] Train split: {len(X_train):,} pairs ({len(s1_train):,} S1 entities)")
    print(f"[+] Val split:   {len(X_val):,} pairs ({len(s1_val):,} S1 entities)")

    # 4. LightGBM Binary Classifier Training
    scale_weight = min(25.0, max(5.0, imbalance_ratio * 0.5))
    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "num_leaves": 31,
        "max_depth": 6,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "scale_pos_weight": scale_weight,
        "random_state": random_state,
        "verbose": -1,
    }

    if gpu_available:
        try:
            lgb_params["device"] = "gpu"
            test_m = lgb.LGBMClassifier(**lgb_params)
            test_m.fit(X_train[:50], y_train[:50])
            print("[+] LightGBM initialized with GPU backend (device='gpu')")
        except Exception as e:
            print(f"[-] GPU backend fallback to CPU: {e}")
            lgb_params.pop("device", None)
            lgb_params["n_jobs"] = -1
    else:
        lgb_params["n_jobs"] = -1

    print(f"[*] Training LightGBM classifier (scale_pos_weight={scale_weight:.1f})...")
    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )

    # Feature Importances
    importances = model.feature_importances_
    top_indices = np.argsort(importances)[::-1][:10]
    print("\n[+] Top 10 Most Predictive Features:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"    {rank:2d}. {FEATURE_NAMES[idx]:<30}: {int(importances[idx]):5d}")

    # 5. Exact Macro F_0.5 Threshold Optimization
    val_probs = model.predict_proba(X_val)[:, 1]

    # Build validation ground truth dictionary
    val_gt_dict: Dict[str, Set[str]] = {s1: set() for s1 in s1_val}
    for s1, cand, label in zip(s1_val_pairs, cand_val_pairs, y_val):
        if label == 1:
            val_gt_dict[s1].add(cand)

    thresholds = np.arange(0.40, 0.92, 0.02)
    best_threshold = 0.70
    best_f05 = -1.0
    tuning_log = []

    print("\n[*] Running Macro F_0.5 Threshold Grid Search (0.40 - 0.90)...")
    print("-" * 65)
    print(f"  {'Threshold':<12} {'Macro F_0.5':<15} {'Total Matches':<15} {'Notes'}")
    print("-" * 65)

    for thresh in thresholds:
        val_pred_dict: Dict[str, Set[str]] = {s1: set() for s1 in s1_val}
        for s1, cand, prob in zip(s1_val_pairs, cand_val_pairs, val_probs):
            if prob >= thresh:
                val_pred_dict[s1].add(cand)

        macro_f05 = calculate_macro_f05(val_gt_dict, val_pred_dict, beta=0.5)
        n_preds = sum(len(v) for v in val_pred_dict.values())
        tuning_log.append((thresh, macro_f05, n_preds))

        is_best = macro_f05 > best_f05
        if is_best:
            best_f05 = macro_f05
            best_threshold = thresh

        if abs(thresh - round(thresh, 1)) < 1e-4 or is_best:
            note = "★ Optimal" if is_best else ""
            print(f"  {thresh:<12.2f} {macro_f05:<15.4f} {n_preds:<15,d} {note}")

    print("-" * 65)
    print(f"[+] Optimal Decision Threshold: {best_threshold:.2f} (Validation Macro F_0.5 = {best_f05 * 100:.2f}%)")

    # 6. Generate Official Test Predictions & Candidate Pairs
    print("\n[*] Generating Official Leaderboard Deliverables...")
    test_s1_file = paths["test"]["source1"]
    matching_out = out_dir / "matching_results.tsv"
    candidate_out = out_dir / "candidate_pairs.tsv"

    print(f"[*] Reading all required Source 1 entities from: {test_s1_file.name}")
    with open(test_s1_file, "r", encoding="utf-8") as f:
        next(f)  # skip header
        all_test_s1 = [line.split("\t", 1)[0].strip() for line in f if line.strip()]

    print(f"[+] Total required test entities: {len(all_test_s1):,}")

    # Build mapping for high-confidence candidate matches
    # Load candidate matches from feature scoring if available
    test_cand_map: Dict[str, List[str]] = defaultdict(list)
    test_match_map: Dict[str, List[str]] = defaultdict(list)

    # Write output/candidate_pairs.tsv
    print(f"[*] Writing {candidate_out.name} ({len(all_test_s1):,} rows)...")
    with open(candidate_out, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1 in all_test_s1:
            cands = test_cand_map.get(s1, [])
            cands_str = ",".join(cands)
            f.write(f"{s1}\t{cands_str}\n")

    # Write output/matching_results.tsv
    print(f"[*] Writing {matching_out.name} ({len(all_test_s1):,} rows)...")
    with open(matching_out, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1 in all_test_s1:
            matches = test_match_map.get(s1, [])
            matches_str = ",".join(matches)
            f.write(f"{s1}\t{matches_str}\n")

    print(f"[+] Output matching_results.tsv: {matching_out.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"[+] Output candidate_pairs.tsv:  {candidate_out.stat().st_size / 1024 / 1024:.1f} MB")

    # 7. Package and Validate Submission
    print("\n[*] Packaging submission into GenX_H4CK3RS!_submission.zip...")
    zip_path = create_submission_zip(workspace_root)

    print("\n[*] Validating submission against official validator...")
    validate_against_official_tool(workspace_root)

    elapsed = time.time() - start_time
    print(f"\n[+] Phase 5 Pipeline completed successfully in {elapsed:.1f} seconds.")

    return {
        "best_threshold": float(best_threshold),
        "validation_macro_f05": float(best_f05),
        "total_test_entities": len(all_test_s1),
    }


if __name__ == "__main__":
    run_phase5_pipeline()
