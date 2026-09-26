import os
import sys
import time
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

script_dir = Path(__file__).resolve().parent
workspace_root = script_dir.parent.parent
sys.path.insert(0, str(script_dir))

from src.data_loader import resolve_data_paths
from src.features import prepare_record_profile, compute_pairwise_features, FEATURE_NAMES
from src.metrics import calculate_macro_f05
from package_submission import create_submission_zip, validate_against_official_tool

# --- Heuristics for Finetuning ---
def extract_numbers(text: str) -> Set[str]:
    return set(re.findall(r'\d+', text))

def passes_heuristics(p1: Dict, p2: Dict) -> bool:
    # 1. Hard Geographic Filtering (Country Match)
    c1, c2 = p1.get("country", "").strip().lower(), p2.get("country", "").strip().lower()
    if c1 and c2 and c1 != c2:
        return False
        
    # 2. The Number Conflict Rule (Building / Zip Codes)
    nums1 = extract_numbers(p1.get("business_address", ""))
    nums2 = extract_numbers(p2.get("business_address", ""))
    if nums1 and nums2 and not nums1.intersection(nums2):
        return False
        
    return True

def get_keys(p: Dict) -> List[str]:
    keys = []
    cn = p.get("clean_name", "")
    if cn: keys.append(f"exact:{cn}")
    
    tokens = p.get("name_tokens", [])
    if tokens:
        longest_token = max(tokens, key=len)
        if len(longest_token) >= 4:
            keys.append(f"core:{longest_token}")
            
    return keys

def main():
    start_time = time.time()
    paths = resolve_data_paths()
    out_dir = workspace_root / "output"
    models_dir = workspace_root / "models"
    
    print("=" * 70)
    print("   ML Challenge 2026 - Phase 5 Finetuned Inference")
    print("=" * 70)

    # 1. Load Cached Model
    model_path = models_dir / "lgb_model.pkl"
    print(f"[*] Loading cached LightGBM model from {model_path}")
    model = joblib.load(model_path)

    # 2. High-resolution Threshold Sweep (0.15 - 0.50)
    feat_path = out_dir / "train_features.csv"
    print(f"[*] Loading training features for threshold tuning from {feat_path}")
    df_feat = pd.read_csv(feat_path)
    X = df_feat[FEATURE_NAMES].values
    y = df_feat["is_match"].values
    pair_s1 = df_feat["source1_entity_id"].values
    pair_cand = df_feat["candidate_entity_id"].values

    unique_s1 = np.unique(pair_s1)
    _, s1_val = train_test_split(unique_s1, test_size=0.25, random_state=42)
    val_mask = np.isin(pair_s1, s1_val)
    X_val, y_val = X[val_mask], y[val_mask]
    s1_val_pairs = pair_s1[val_mask]
    cand_val_pairs = pair_cand[val_mask]

    val_probs = model.predict_proba(X_val)[:, 1]
    val_gt_dict = {s1: set() for s1 in s1_val}
    for s1, cand, label in zip(s1_val_pairs, cand_val_pairs, y_val):
        if label == 1:
            val_gt_dict[s1].add(cand)

    thresholds = np.arange(0.15, 0.51, 0.01)
    best_threshold = 0.40
    best_f05 = -1.0

    print("\n[*] Running High-Resolution Macro F0.5 Threshold Sweep...")
    for thresh in thresholds:
        val_pred_dict = {s1: set() for s1 in s1_val}
        for s1, cand, prob in zip(s1_val_pairs, cand_val_pairs, val_probs):
            if prob >= thresh:
                val_pred_dict[s1].add(cand)
        macro_f05 = calculate_macro_f05(val_gt_dict, val_pred_dict, beta=0.5)
        if macro_f05 > best_f05:
            best_f05 = macro_f05
            best_threshold = thresh

    print(f"[+] Optimal Decision Threshold: {best_threshold:.2f} (Macro F0.5 = {best_f05:.4f})")

    # 3. Exhaustive Multi-Key Index for S1
    test_s1_file = paths["test"]["source1"]
    print(f"\n[*] Building exhaustive multi-key index for {test_s1_file.name}...")
    
    s1_dict = {}
    s1_index = defaultdict(list)
    
    with open(test_s1_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                p = prepare_record_profile({
                    "entity_id": parts[0],
                    "business_name": parts[1],
                    "business_address": parts[2],
                    "country": parts[3],
                })
                s1_dict[parts[0]] = p
                for key in get_keys(p):
                    s1_index[key].append(parts[0])

    # To avoid O(N^2) explosion without arbitrary limits, cap ONLY hyper-generic keys (stop-words) 
    # but keep a high cap (1000) to ensure valid matches aren't arbitrarily dropped.
    for k in list(s1_index.keys()):
        if len(s1_index[k]) > 1000:
            del s1_index[k]
            
    print(f"[+] Indexed {len(s1_dict):,} S1 entities across {len(s1_index):,} unique keys.")

    # 4. Stream Source 2 and 3, Extract, Predict, and Export
    matching_out = out_dir / "matching_results.tsv"
    candidate_out = out_dir / "candidate_pairs.tsv"
    
    test_match_map = defaultdict(list)
    test_cand_map = defaultdict(list)

    batch_pairs = []
    batch_feats = []
    
    def process_batch():
        if not batch_pairs: return
        X_batch = pd.DataFrame(batch_feats)[FEATURE_NAMES].values
        probs = model.predict_proba(X_batch)[:, 1]
        for (s1_id, cand_id), prob in zip(batch_pairs, probs):
            if prob >= best_threshold:
                if cand_id not in test_match_map[s1_id]:
                    test_match_map[s1_id].append(cand_id)
        batch_pairs.clear()
        batch_feats.clear()

    for s_name, path in [("Source 2", paths["test"]["source2"]), ("Source 3", paths["test"]["source3"])]:
        print(f"\n[*] Streaming {s_name} ({path.name}) using memory-efficient inference...")
        with open(path, "r", encoding="utf-8") as f:
            next(f)
            for line_idx, line in enumerate(f):
                if line_idx > 0 and line_idx % 500000 == 0:
                    print(f"    - Processed {line_idx:,} rows...")
                
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    p2 = prepare_record_profile({
                        "entity_id": parts[0],
                        "business_name": parts[1],
                        "business_address": parts[2],
                        "country": parts[3],
                    })
                    
                    matched_s1s = set()
                    for key in get_keys(p2):
                        if key in s1_index:
                            matched_s1s.update(s1_index[key])
                            
                    for s1_id in matched_s1s:
                        p1 = s1_dict[s1_id]
                        
                        # Apply Heuristic Finetuning Filters
                        if not passes_heuristics(p1, p2):
                            continue
                            
                        test_cand_map[s1_id].append(parts[0])
                        batch_pairs.append((s1_id, parts[0]))
                        batch_feats.append(compute_pairwise_features(p1, p2))
                        
                        if len(batch_pairs) >= 50000:
                            process_batch()

    process_batch()
    
    n_matched_entities = len(test_match_map)
    n_matched_pairs = sum(len(v) for v in test_match_map.values())
    print(f"\n[+] Predicted {n_matched_pairs:,} true matches across {n_matched_entities:,} S1 entities.")
    
    print("[*] Exporting TSV outputs...")
    with open(test_s1_file, "r", encoding="utf-8") as f_in, \
         open(candidate_out, "w", encoding="utf-8", newline="") as f_cand, \
         open(matching_out, "w", encoding="utf-8", newline="") as f_match:
         
        next(f_in)
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        
        for line in f_in:
            if not line.strip(): continue
            s1_id = line.split("\t", 1)[0].strip()
            cands = ",".join(test_cand_map.get(s1_id, []))
            matches = ",".join(test_match_map.get(s1_id, []))
            f_cand.write(f"{s1_id}\t{cands}\n")
            f_match.write(f"{s1_id}\t{matches}\n")

    print(f"[+] Export complete: {matching_out.name}")

    # 5. Package Submission
    print("\n[*] Packaging submission...")
    zip_path = create_submission_zip(workspace_root)
    print("\n[*] Validating submission...")
    validate_against_official_tool(workspace_root)
    
    elapsed = time.time() - start_time
    print(f"\n[+] Complete Pipeline finished in {elapsed:.1f} seconds.")

if __name__ == "__main__":
    main()
