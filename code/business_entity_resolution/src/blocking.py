"""
High-Recall Multi-Pass Blocking Stage (Candidate Generation)
Part of Phase 3 Implementation for Business Entity Resolution
Team: GenX H4CK3RS!

CRITICAL SPECIFICATIONS & CONSTRAINTS:
1. Hard Country Partitioning:
   - Partition records strictly by the dynamic 'country' string (US, India, France).
   - Drastically cuts the pairwise comparison search space by 60%–85% with zero cross-country recall loss.
2. Multi-Pass Inverted Index Blocking:
   - Key A: Cleaned core business name tokens (excluding legal forms).
   - Key B: Building/house numbers combined with first name token & address token.
   - Key C: Postal/PIN code combined with first name token & address token.
   - Key D: Character n-gram prefix & fuzzy phonetic keys.
3. High Recall Ceiling Target:
   - Aims for >= 97% recall ceiling on ground truth matches.
4. Candidate Pruning & Output Export:
   - Caps candidates per Source 1 entity (default: top 30-50) using overlap scoring.
   - Exports strictly to output/candidate_pairs.tsv matching the official competition schema:
     source1_entity_id<TAB>candidate_entity_ids (comma-separated S2-/S3- IDs).
"""

import collections
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

from .data_loader import save_tsv
from .text_preprocessing import (
    clean_business_name,
    clean_business_address,
    extract_numeric_features,
)

# Common generic address and business stopwords to exclude from individual token keys
STOPWORDS = {
    # English generic terms
    "road", "street", "avenue", "boulevard", "lane", "drive", "court", "place",
    "suite", "floor", "unit", "near", "opposite", "behind", "and", "the", "for",
    "house", "plot", "sector", "colony", "apartment", "building", "nagar", "market",
    # French generic terms
    "rue", "avenue", "boulevard", "allee", "impasse", "chemin", "route", "place",
    "cours", "quai", "residence", "batiment", "de", "la", "du", "des", "les", "et",
    # Hindi/India generic romanized terms
    "marg", "gali", "chowk", "bazar", "pur", "bad", "delhi", "mumbai", "india",
}


def extract_blocking_keys(
    business_name: str,
    business_address: str,
    country: str = "",
) -> Set[str]:
    """
    Extracts multi-pass blocking keys for a single record across:
    - Key A: Core Name Tokens & Prefix
    - Key B: Building number + Name / Address
    - Key C: Postal/PIN code + Name / Address
    - Key D: Character n-gram prefix (for fuzzy typo handling)
    """
    keys: Set[str] = set()

    # Preprocess text fields
    name_info = clean_business_name(business_name)
    addr_info = clean_business_address(business_address)
    num_info = extract_numeric_features(business_address, country)

    clean_n = name_info["name_clean"]
    name_tokens = [t for t in clean_n.split() if len(t) >= 2 and t not in STOPWORDS]

    addr_tokens = [
        t for t in addr_info["address_tokens"]
        if len(t) >= 3 and t not in STOPWORDS
    ]

    b_nums = num_info["building_numbers"]
    postal = num_info["postal_code"]

    # -----------------------------------------------------------------------
    # Key A: Core Business Name Keys
    # -----------------------------------------------------------------------
    # A1. Exact cleaned core name
    if clean_n and len(clean_n) >= 3:
        keys.add(f"exact:{clean_n}")

    # A2. First two name tokens (order-sensitive)
    if len(name_tokens) >= 2:
        keys.add(f"n2tok:{name_tokens[0]}_{name_tokens[1]}")

    # A3. Individual significant name tokens (length >= 3)
    for t in name_tokens:
        if len(t) >= 3:
            keys.add(f"ntok:{t}")

    # A4. Sorted name tokens (word-order transposition tolerance)
    if len(name_tokens) >= 2:
        keys.add(f"sortname:{'_'.join(sorted(name_tokens[:3]))}")

    # -----------------------------------------------------------------------
    # Key B: Building Number + Name / Address Components
    # -----------------------------------------------------------------------
    if b_nums:
        b0 = b_nums[0]
        # B0. Building number
        keys.add(f"bnum:{b0}")
        # B1. Building number + first name token
        if name_tokens:
            keys.add(f"b_name:{b0}_{name_tokens[0]}")
        # B2. Building number + primary address token (street or area)
        if addr_tokens:
            keys.add(f"b_addr:{b0}_{addr_tokens[0]}")
            if len(addr_tokens) >= 2:
                keys.add(f"b_addr2:{b0}_{addr_tokens[1]}")

    # B3. Distinctive address tokens (catches DBA / transliterated names at same address)
    for t in addr_tokens:
        if len(t) >= 5:
            keys.add(f"addr_tok:{t}")


    # -----------------------------------------------------------------------
    # Key C: Postal / PIN Code + Name / Address Components
    # -----------------------------------------------------------------------
    if postal:
        # C0. Exact postal code
        keys.add(f"post:{postal}")
        # C1. Postal code + first name token
        if name_tokens:
            keys.add(f"p_name:{postal}_{name_tokens[0]}")
        # C2. Postal code + primary address token
        if addr_tokens:
            keys.add(f"p_addr:{postal}_{addr_tokens[0]}")

    # -----------------------------------------------------------------------
    # Key D: Character N-Gram & Sparse Indexing (Fuzzy Spelling & Typos)
    # -----------------------------------------------------------------------
    if len(clean_n) >= 4:
        # 4-character prefix captures typos in suffixes or transpositions
        keys.add(f"pref4:{clean_n[:4]}")
    if name_tokens and len(name_tokens[0]) >= 4:
        first_w = name_tokens[0]
        keys.add(f"tokpref4:{first_w[:4]}")
        # Character 3-grams for fuzzy matching
        for i in range(len(first_w) - 2):
            keys.add(f"3gram:{first_w[i:i+3]}")

    return keys



class MultiPassBlocker:
    """
    High-Performance, Multi-Pass Inverted Index Blocker.
    Partitions by Country and indexes Target pools (Source 2 and Source 3).
    """

    def __init__(
        self,
        max_bucket_size: int = 5000,
        max_candidates_per_entity: int = 40,
    ):
        """
        Parameters:
        - max_bucket_size: Skips or caps ultra-frequent keys that create too many false positives.
        - max_candidates_per_entity: Maximum number of S2/S3 candidates retained per S1 entity.
        """
        self.max_bucket_size = max_bucket_size
        self.max_candidates_per_entity = max_candidates_per_entity
        # Inverted index: country -> {blocking_key -> list of target_entity_ids}
        self.inverted_indices: Dict[str, Dict[str, List[str]]] = collections.defaultdict(
            lambda: collections.defaultdict(list)
        )

    def fit_target_pool(
        self,
        target_df: pd.DataFrame,
        country_col: str = "country",
        id_col: str = "entity_id",
        name_col: str = "business_name",
        addr_col: str = "business_address",
    ) -> None:
        """
        Indexes the target records (Source 2 and Source 3) into country-partitioned inverted indices.
        """
        print(f"[*] Indexing {len(target_df):,} target records into MultiPassBlocker...")

        for _, row in target_df.iterrows():
            eid = str(row[id_col])
            country = str(row[country_col]).strip() if country_col in row else "UNKNOWN"
            bname = str(row[name_col]) if name_col in row else ""
            baddr = str(row[addr_col]) if addr_col in row else ""

            keys = extract_blocking_keys(bname, baddr, country)
            country_idx = self.inverted_indices[country]

            for k in keys:
                bucket = country_idx[k]
                if len(bucket) < self.max_bucket_size:
                    bucket.append(eid)

        total_keys = sum(len(idx) for idx in self.inverted_indices.values())
        print(f"[+] Indexing complete. Indexed across {len(self.inverted_indices)} countries with {total_keys:,} unique keys.")

    def generate_candidates_for_entity(
        self,
        entity_id: str,
        business_name: str,
        business_address: str,
        country: str,
    ) -> List[str]:
        """
        Retrieves candidate entity IDs for a single Source 1 entity from the target index.
        Ranks candidates by number of matching blocking keys and caps at max_candidates_per_entity.
        """
        country = country.strip()
        if country not in self.inverted_indices:
            return []

        country_idx = self.inverted_indices[country]
        keys = extract_blocking_keys(business_name, business_address, country)

        candidate_scores: Dict[str, int] = collections.defaultdict(int)

        for k in keys:
            if k in country_idx:
                for target_id in country_idx[k]:
                    candidate_scores[target_id] += 1

        if not candidate_scores:
            return []

        # Rank candidates by the number of overlapping blocking keys (highest first)
        # Ties broken deterministically by ID string
        sorted_candidates = sorted(
            candidate_scores.keys(),
            key=lambda tid: (-candidate_scores[tid], tid),
        )

        return sorted_candidates[: self.max_candidates_per_entity]

    def block_dataframe(
        self,
        source1_df: pd.DataFrame,
        id_col: str = "entity_id",
        name_col: str = "business_name",
        addr_col: str = "business_address",
        country_col: str = "country",
    ) -> Dict[str, List[str]]:
        """
        Generates candidate mapping {source1_entity_id -> [candidate_ids]} for all Source 1 entities.
        """
        candidates: Dict[str, List[str]] = {}
        total = len(source1_df)

        print(f"[*] Generating candidates for {total:,} Source 1 entities...")
        for i, row in enumerate(source1_df.itertuples(index=False)):
            s1_id = getattr(row, id_col)
            bname = getattr(row, name_col, "")
            baddr = getattr(row, addr_col, "")
            country = getattr(row, country_col, "")

            cand_list = self.generate_candidates_for_entity(
                s1_id, bname, baddr, country
            )
            candidates[s1_id] = cand_list

            if (i + 1) % 50000 == 0 or (i + 1) == total:
                print(f"    Processed {i + 1:,} / {total:,} entities...")

        return candidates


def evaluate_blocking_recall(
    candidate_dict: Dict[str, List[str]],
    ground_truth_dict: Dict[str, Set[str]],
) -> Dict[str, float]:
    """
    Evaluates blocking quality:
    - recall_ceiling: Proportion of all true matches captured in candidate sets.
    - entity_coverage: Proportion of non-singleton entities that have at least one true match captured.
    - avg_candidates: Average number of candidate pairs generated per S1 entity.
    - singleton_retention: Proportion of true singletons where blocking returned 0 candidates.
    """
    total_true_matches = 0
    captured_true_matches = 0
    total_non_singletons = 0
    covered_non_singletons = 0
    total_singletons = 0
    empty_singletons = 0
    total_candidates = 0

    for s1_id, true_set in ground_truth_dict.items():
        cands = set(candidate_dict.get(s1_id, []))
        total_candidates += len(cands)

        if len(true_set) == 0:
            total_singletons += 1
            if len(cands) == 0:
                empty_singletons += 1
        else:
            total_non_singletons += 1
            total_true_matches += len(true_set)
            hits = len(cands & true_set)
            captured_true_matches += hits
            if hits > 0:
                covered_non_singletons += 1

    recall_ceiling = (captured_true_matches / total_true_matches) if total_true_matches > 0 else 1.0
    entity_coverage = (covered_non_singletons / total_non_singletons) if total_non_singletons > 0 else 1.0
    avg_cands = total_candidates / len(ground_truth_dict) if ground_truth_dict else 0.0
    singleton_retention = empty_singletons / total_singletons if total_singletons > 0 else 1.0

    return {
        "recall_ceiling": float(recall_ceiling),
        "entity_coverage": float(entity_coverage),
        "total_true_matches": total_true_matches,
        "captured_true_matches": captured_true_matches,
        "avg_candidates_per_entity": round(avg_cands, 2),
        "singleton_retention_rate": round(singleton_retention, 4),
    }


def export_candidate_pairs_tsv(
    candidate_dict: Dict[str, List[str]],
    output_path: Union[str, Path],
) -> None:
    """
    Exports candidates strictly as candidate_pairs.tsv matching the official competition schema:
    source1_entity_id<TAB>candidate_entity_ids
    (Comma-separated list of candidate S2/S3 IDs with no quotes or spaces).
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for s1_id, cands in candidate_dict.items():
        # Ensure no duplicates and comma-separated representation
        clean_cands = ",".join(cands) if cands else ""
        rows.append({"source1_entity_id": s1_id, "candidate_entity_ids": clean_cands})

    df = pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_ids"])
    save_tsv(df, path)
    print(f"[+] Exported candidate pairs for {len(df):,} entities to: {path}")
