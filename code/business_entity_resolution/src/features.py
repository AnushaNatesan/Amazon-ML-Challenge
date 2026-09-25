"""
Pairwise Feature Engineering Module
Part of Phase 4 Implementation for Business Entity Resolution
Team: GenX H4CK3RS!

CRITICAL SPECIFICATIONS & CONSTRAINTS:
1. Country-Agnostic Processing:
   - Evaluates similarity metrics without hardcoding country rules.
   - Fully generalized across US, Indian, and French datasets.
2. Fast RapidFuzz Metric Extraction:
   - C-accelerated Levenshtein, Jaro-Winkler, Token Sort, Token Set, and Partial Ratio.
   - Character n-gram Jaccard and word token containment ratios.
3. Component-Wise Disagreement Detection:
   - Building numbers: +1 (match), 0 (missing/neutral), -1 (explicit conflict).
   - Postal / PIN codes: +1 (match), 0 (missing/neutral), -1 (explicit conflict).
   - Legal form compatibility: +1 (match), 0 (missing/neutral), -1 (conflict).
4. Safe Missing Address Handling:
   - Flags missing addresses with explicit indicators (addr_is_missing).
   - Never produces NaN values or silent errors.
"""

import collections
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler

from .text_preprocessing import (
    clean_business_name,
    clean_business_address,
    extract_numeric_features,
)


def char_ngram_jaccard(s1: str, s2: str, n: int = 3) -> float:
    """Computes character n-gram Jaccard similarity between two strings."""
    if not s1 or not s2:
        return 0.0
    if len(s1) < n or len(s2) < n:
        return 1.0 if s1 == s2 else 0.0
    q1 = {s1[i : i + n] for i in range(len(s1) - n + 1)}
    q2 = {s2[i : i + n] for i in range(len(s2) - n + 1)}
    union = q1 | q2
    return len(q1 & q2) / len(union) if union else 0.0


def token_jaccard(tokens1: List[str], tokens2: List[str]) -> float:
    """Computes Jaccard similarity over two token lists."""
    if not tokens1 or not tokens2:
        return 0.0
    s1, s2 = set(tokens1), set(tokens2)
    union = s1 | s2
    return len(s1 & s2) / len(union) if union else 0.0


def token_containment_ratio(tokens1: List[str], tokens2: List[str]) -> float:
    """Computes overlap ratio relative to the smaller token set."""
    if not tokens1 or not tokens2:
        return 0.0
    s1, s2 = set(tokens1), set(tokens2)
    min_len = min(len(s1), len(s2))
    return len(s1 & s2) / min_len if min_len > 0 else 0.0


def prepare_record_profile(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pre-computes and caches normalized text and extracted tokens for an entity.
    Significantly accelerates pairwise comparisons when entities are evaluated against multiple candidates.
    """
    bname = str(record.get("business_name", ""))
    baddr = str(record.get("business_address", ""))
    country = str(record.get("country", ""))

    name_info = clean_business_name(bname)
    addr_info = clean_business_address(baddr)
    num_info = extract_numeric_features(baddr, country)

    clean_n = name_info["name_clean"]
    clean_a = addr_info["address_clean"]

    return {
        "entity_id": record.get("entity_id", ""),
        "clean_name": clean_n,
        "name_raw_clean": name_info["name_raw_clean"],
        "legal_form": name_info["legal_form"],
        "name_tokens": clean_n.split() if clean_n else [],
        "clean_address": clean_a,
        "addr_tokens": addr_info["address_tokens"],
        "postal_code": num_info["postal_code"],
        "building_numbers": set(num_info["building_numbers"]),
        "all_numbers": set(num_info["all_numbers"]),
        "has_address": bool(clean_a.strip()),
    }


def compute_pairwise_features(
    prof1: Dict[str, Any],
    prof2: Dict[str, Any],
    candidate_rank: int = 0,
) -> Dict[str, float]:
    """
    Computes a comprehensive vector of similarity and compatibility features
    between Source 1 profile (prof1) and Candidate profile (prof2).
    """
    feat: Dict[str, float] = {}

    # -----------------------------------------------------------------------
    # 1. Name Similarity Features (Operating on clean_name)
    # -----------------------------------------------------------------------
    n1 = prof1["clean_name"]
    n2 = prof2["clean_name"]
    toks1 = prof1["name_tokens"]
    toks2 = prof2["name_tokens"]

    feat["name_exact_match"] = 1.0 if (n1 and n1 == n2) else 0.0
    feat["name_levenshtein"] = float(Levenshtein.normalized_similarity(n1, n2)) if (n1 or n2) else 0.0
    feat["name_jaro_winkler"] = float(JaroWinkler.similarity(n1, n2)) if (n1 or n2) else 0.0
    feat["name_token_sort_ratio"] = float(fuzz.token_sort_ratio(n1, n2)) / 100.0 if (n1 or n2) else 0.0
    feat["name_token_set_ratio"] = float(fuzz.token_set_ratio(n1, n2)) / 100.0 if (n1 or n2) else 0.0
    feat["name_partial_ratio"] = float(fuzz.partial_ratio(n1, n2)) / 100.0 if (n1 or n2) else 0.0
    feat["name_char_3gram_jaccard"] = float(char_ngram_jaccard(n1, n2, n=3))
    feat["name_token_jaccard"] = float(token_jaccard(toks1, toks2))
    feat["name_token_containment"] = float(token_containment_ratio(toks1, toks2))

    # First token match (brand name anchor)
    if toks1 and toks2:
        feat["name_first_token_match"] = 1.0 if toks1[0] == toks2[0] else 0.0
        feat["name_first_token_lev"] = float(Levenshtein.normalized_similarity(toks1[0], toks2[0]))
    else:
        feat["name_first_token_match"] = 0.0
        feat["name_first_token_lev"] = 0.0

    len_max_n = max(len(n1), len(n2))
    feat["name_length_ratio"] = (min(len(n1), len(n2)) / len_max_n) if len_max_n > 0 else 0.0
    feat["name_token_count_diff"] = float(abs(len(toks1) - len(toks2)))

    # Raw name similarity (capturing legal form variations before stripping)
    raw1 = prof1["name_raw_clean"]
    raw2 = prof2["name_raw_clean"]
    feat["name_raw_token_set_ratio"] = float(fuzz.token_set_ratio(raw1, raw2)) / 100.0 if (raw1 or raw2) else 0.0

    # -----------------------------------------------------------------------
    # 2. Address Similarity & Missing Address Handling
    # -----------------------------------------------------------------------
    has_addr1 = prof1["has_address"]
    has_addr2 = prof2["has_address"]
    a1 = prof1["clean_address"]
    a2 = prof2["clean_address"]
    atoks1 = prof1["addr_tokens"]
    atoks2 = prof2["addr_tokens"]

    # Explicit missing address indicator (critical for the ~3.3% missing address records)
    addr_missing = not (has_addr1 and has_addr2)
    feat["addr_is_missing"] = 1.0 if addr_missing else 0.0

    if not addr_missing:
        feat["addr_levenshtein"] = float(Levenshtein.normalized_similarity(a1, a2))
        feat["addr_jaro_winkler"] = float(JaroWinkler.similarity(a1, a2))
        feat["addr_token_set_ratio"] = float(fuzz.token_set_ratio(a1, a2)) / 100.0
        feat["addr_token_jaccard"] = float(token_jaccard(atoks1, atoks2))
        feat["addr_token_containment"] = float(token_containment_ratio(atoks1, atoks2))
        len_max_a = max(len(a1), len(a2))
        feat["addr_length_ratio"] = min(len(a1), len(a2)) / len_max_a if len_max_a > 0 else 0.0
    else:
        # Impute missing address similarity with neutral 0.0 paired with addr_is_missing flag
        feat["addr_levenshtein"] = 0.0
        feat["addr_jaro_winkler"] = 0.0
        feat["addr_token_set_ratio"] = 0.0
        feat["addr_token_jaccard"] = 0.0
        feat["addr_token_containment"] = 0.0
        feat["addr_length_ratio"] = 0.0

    # -----------------------------------------------------------------------
    # 3. Numeric & Locality Disagreement Detection (Ternary Matching)
    # -----------------------------------------------------------------------
    # Building Numbers: +1.0 = match, -1.0 = conflict, 0.0 = missing
    b1 = prof1["building_numbers"]
    b2 = prof2["building_numbers"]
    if b1 and b2:
        feat["building_number_match"] = 1.0 if bool(b1 & b2) else -1.0
    else:
        feat["building_number_match"] = 0.0

    # Postal / PIN codes: +1.0 = match, -1.0 = conflict, 0.0 = missing
    p1 = prof1["postal_code"]
    p2 = prof2["postal_code"]
    if p1 and p2:
        feat["postal_code_match"] = 1.0 if (p1 == p2) else -1.0
        feat["postal_code_prefix_match"] = 1.0 if (p1[:3] == p2[:3]) else 0.0
    else:
        feat["postal_code_match"] = 0.0
        feat["postal_code_prefix_match"] = 0.0

    # General numeric tokens overlap
    nums1 = prof1["all_numbers"]
    nums2 = prof2["all_numbers"]
    shared_nums = nums1 & nums2
    feat["numeric_overlap_count"] = float(len(shared_nums))
    feat["has_shared_number"] = 1.0 if bool(shared_nums) else 0.0

    # -----------------------------------------------------------------------
    # 4. Legal Entity Compatibility
    # -----------------------------------------------------------------------
    leg1 = prof1["legal_form"]
    leg2 = prof2["legal_form"]
    feat["legal_form_present_both"] = 1.0 if (leg1 and leg2) else 0.0
    if leg1 and leg2:
        feat["legal_form_match"] = 1.0 if leg1 == leg2 else -1.0
    else:
        feat["legal_form_match"] = 0.0

    # -----------------------------------------------------------------------
    # 5. Candidate Metadata & Structural Features
    # -----------------------------------------------------------------------
    feat["candidate_rank"] = float(candidate_rank)
    cid = prof2["entity_id"]
    feat["is_source_2"] = 1.0 if cid.startswith("S2-") else 0.0
    feat["is_source_3"] = 1.0 if cid.startswith("S3-") else 0.0

    # Interaction Features
    feat["name_and_addr_overlap"] = feat["name_token_jaccard"] * feat["addr_token_jaccard"]
    feat["high_confidence_anchor"] = 1.0 if (feat["name_exact_match"] == 1.0 and feat["building_number_match"] == 1.0) else 0.0

    return feat


FEATURE_NAMES = list(compute_pairwise_features(
    {"clean_name": "", "name_raw_clean": "", "legal_form": "", "name_tokens": [], "clean_address": "", "addr_tokens": [], "postal_code": "", "building_numbers": set(), "all_numbers": set(), "has_address": False, "entity_id": "S1-0"},
    {"clean_name": "", "name_raw_clean": "", "legal_form": "", "name_tokens": [], "clean_address": "", "addr_tokens": [], "postal_code": "", "building_numbers": set(), "all_numbers": set(), "has_address": False, "entity_id": "S2-0"},
    candidate_rank=0,
).keys())


def extract_features_for_candidate_pairs(
    candidate_dict: Dict[str, List[str]],
    source1_df: pd.DataFrame,
    target_pool_df: pd.DataFrame,
    ground_truth_dict: Optional[Dict[str, Set[str]]] = None,
) -> Tuple[pd.DataFrame, Optional[np.ndarray], List[Tuple[str, str]]]:
    """
    Extracts vectorized pairwise feature matrix across all (source1, candidate) pairs.

    Parameters:
    - candidate_dict: Mapping {source1_id -> list of candidate_ids} from Phase 3 blocking.
    - source1_df: DataFrame of Source 1 entities.
    - target_pool_df: DataFrame of candidate entities (Source 2 and/or Source 3).
    - ground_truth_dict: Optional ground truth mapping for generating binary training labels (1 = match, 0 = non-match).

    Returns:
    - features_df: DataFrame of computed feature values for each candidate pair.
    - labels: Optional 1D numpy array of binary labels (if ground_truth_dict is provided).
    - pair_ids: List of (source1_id, candidate_id) tuples aligned with features_df.
    """
    print(f"[*] Pre-computing profiles for {len(source1_df):,} Source 1 records...")
    s1_profiles: Dict[str, Dict[str, Any]] = {}
    for row in source1_df.to_dict("records"):
        s1_profiles[row["entity_id"]] = prepare_record_profile(row)

    print(f"[*] Pre-computing profiles for {len(target_pool_df):,} target records...")
    target_profiles: Dict[str, Dict[str, Any]] = {}
    for row in target_pool_df.to_dict("records"):
        target_profiles[row["entity_id"]] = prepare_record_profile(row)

    pair_ids: List[Tuple[str, str]] = []
    rows: List[Dict[str, float]] = []
    labels_list: List[int] = []

    total_pairs = sum(len(cands) for cands in candidate_dict.values())
    print(f"[*] Extracting features across {total_pairs:,} candidate pairs...")

    for s1_id, cands in candidate_dict.items():
        if s1_id not in s1_profiles:
            continue
        p1 = s1_profiles[s1_id]
        true_set = ground_truth_dict.get(s1_id, set()) if ground_truth_dict is not None else None

        for rank, cid in enumerate(cands):
            if cid not in target_profiles:
                continue
            p2 = target_profiles[cid]
            feat_vec = compute_pairwise_features(p1, p2, candidate_rank=rank)

            pair_ids.append((s1_id, cid))
            rows.append(feat_vec)

            if true_set is not None:
                labels_list.append(1 if cid in true_set else 0)

    features_df = pd.DataFrame(rows, columns=FEATURE_NAMES)
    labels = np.array(labels_list, dtype=np.int32) if ground_truth_dict is not None else None

    print(f"[+] Feature matrix built: {features_df.shape[0]:,} pairs x {features_df.shape[1]} features.")
    if labels is not None:
        pos_count = int(np.sum(labels))
        neg_count = len(labels) - pos_count
        pos_ratio = (pos_count / len(labels)) * 100 if len(labels) > 0 else 0.0
        print(f"    Labels: {pos_count:,} positives ({pos_ratio:.2f}%), {neg_count:,} negatives.")

    return features_df, labels, pair_ids
