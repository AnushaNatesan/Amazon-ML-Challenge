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

import joblib
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
from src.features import (
    prepare_record_profile,
    compute_pairwise_features,
    FEATURE_NAMES,
)
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
    n_estimators: int = 250,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> Dict[str, float]:
    """Executes the complete Phase 5 training, tuning, and inference pipeline."""
    start_time = time.time()
    paths = resolve_data_paths()
    out_dir = Path(output_dir) if output_dir else workspace_root / "output"
    models_dir = workspace_root / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

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
    scale_weight = min(25.0, max(1.0, imbalance_ratio))
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

    # Save model weights to models/
    pkl_model_path = models_dir / "lgb_model.pkl"
    txt_model_path = models_dir / "lgb_model.txt"
    joblib.dump(model, pkl_model_path)
    model.booster_.save_model(str(txt_model_path))
    print(f"[+] Saved trained model to {pkl_model_path.name} and {txt_model_path.name}")

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
    print("\n[*] Generating Official Leaderboard Deliverables (Inference)...")
    test_s1_file = paths["test"]["source1"]
    matching_out = out_dir / "matching_results.tsv"
    candidate_out = out_dir / "candidate_pairs.tsv"

    # Step A: Index a targeted pool of S1 clean names (first 50,000 S1 records)
    print(f"[*] Reading Source 1 test entities from: {test_s1_file.name}")
    s1_dict = {}
    s1_name_map = {}
    with open(test_s1_file, "r", encoding="utf-8") as f:
        next(f)
        for i, line in enumerate(f):
            if i >= 50000:
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                p = prepare_record_profile({
                    "entity_id": parts[0],
                    "business_name": parts[1],
                    "business_address": parts[2],
                    "country": parts[3],
                })
                s1_dict[parts[0]] = p
                if p["clean_name"]:
                    s1_name_map.setdefault(p["clean_name"], []).append(parts[0])

    print(f"[+] Prepared profiles and name index for {len(s1_dict):,} test S1 entities.")

    # Step B: Scan candidate target records from test_source2 and test_source3
    test_cand_map: Dict[str, List[str]] = defaultdict(list)
    cand_pairs_to_score = []

    for s_name, path in [
        ("Source 2", paths["test"]["source2"]),
        ("Source 3", paths["test"]["source3"]),
    ]:
        print(f"[*] Scanning {s_name} ({path.name}) for candidate matches...")
        with open(path, "r", encoding="utf-8") as f:
            next(f)
            for i, line in enumerate(f):
                if i >= 100000:
                    break
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 4:
                    rec = {
                        "entity_id": parts[0],
                        "business_name": parts[1],
                        "business_address": parts[2],
                        "country": parts[3],
                    }
                    p = prepare_record_profile(rec)
                    cn = p["clean_name"]
                    if cn and cn in s1_name_map:
                        for s1_id in s1_name_map[cn]:
                            test_cand_map[s1_id].append(parts[0])
                            cand_pairs_to_score.append((s1_dict[s1_id], p))

    print(f"[+] Found {len(cand_pairs_to_score):,} candidate pairs across {len(test_cand_map):,} S1 entities.")

    # Step C: Extract 33 features and score with trained LightGBM model
    test_match_map: Dict[str, List[str]] = defaultdict(list)
    if cand_pairs_to_score:
        print(f"[*] Computing 33 pairwise features for {len(cand_pairs_to_score):,} test candidate pairs...")
        feats = [compute_pairwise_features(p1, p2) for p1, p2 in cand_pairs_to_score]
        X_test = pd.DataFrame(feats)[FEATURE_NAMES].values

        print(f"[*] Scoring candidate pairs with trained LightGBM model...")
        test_probs = model.predict_proba(X_test)[:, 1]

        # Apply optimal decision threshold best_threshold
        for (p1, p2), prob in zip(cand_pairs_to_score, test_probs):
            if prob >= best_threshold:
                s1_id = p1["entity_id"]
                cand_id = p2["entity_id"]
                if cand_id not in test_match_map[s1_id]:
                    test_match_map[s1_id].append(cand_id)

        n_matched_entities = len(test_match_map)
        n_matched_pairs = sum(len(v) for v in test_match_map.values())
        print(f"[+] Predicted {n_matched_pairs:,} true matches across {n_matched_entities:,} S1 entities (τ* = {best_threshold:.2f}).")

    # Step D: Stream all 1,732,544 rows to official candidate_pairs.tsv and matching_results.tsv
    print(f"[*] Streaming official outputs for all 1.73M entities...")
    with open(test_s1_file, "r", encoding="utf-8") as f_in, \
         open(candidate_out, "w", encoding="utf-8", newline="") as f_cand, \
         open(matching_out, "w", encoding="utf-8", newline="") as f_match:

        next(f_in)  # skip header
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        f_match.write("source1_entity_id\tmatched_entity_ids\n")

        total_written = 0
        for line in f_in:
            if not line.strip():
                continue
            s1_id = line.split("\t", 1)[0].strip()

            cands = test_cand_map.get(s1_id, [])
            matches = test_match_map.get(s1_id, [])

            cands_str = ",".join(cands)
            matches_str = ",".join(matches)

            f_cand.write(f"{s1_id}\t{cands_str}\n")
            f_match.write(f"{s1_id}\t{matches_str}\n")
            total_written += 1

    print(f"[+] Output {candidate_out.name}: {candidate_out.stat().st_size / 1024 / 1024:.1f} MB ({total_written:,} rows)")
    print(f"[+] Output {matching_out.name}:  {matching_out.stat().st_size / 1024 / 1024:.1f} MB ({total_written:,} rows)")

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
        "total_test_entities": total_written,
        "matched_entities": len(test_match_map),
        "total_matches": sum(len(v) for v in test_match_map.values()),
    }


if __name__ == "__main__":
    run_phase5_pipeline()
