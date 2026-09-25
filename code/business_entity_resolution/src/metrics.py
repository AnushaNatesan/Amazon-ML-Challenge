"""
Evaluation Metrics & Local Validation Harness
Part of Phase 1 Implementation for Business Entity Resolution
Team: GenX H4CK3RS!

CRITICAL METRIC SPECIFICATION:
F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
The metric is macro-averaged across ALL Source 1 entities in the evaluation set.

Singleton Handling Rule:
A Source 1 entity with NO true matches (singleton):
- Scores 1.0 when an empty list of matches is predicted.
- Scores 0.0 when ANY match is predicted (penalizing false merges).

For non-singletons:
- Precision = |predicted & true| / |predicted|
- Recall = |predicted & true| / |true|
- If predicted is empty: score = 0.0
- If precision + recall == 0: score = 0.0
"""

from typing import Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def compute_entity_f_beta(
    predicted_ids: Union[Set[str], List[str], str],
    true_ids: Union[Set[str], List[str], str],
    beta: float = 0.5,
) -> Tuple[float, float, float]:
    """
    Computes (f_beta, precision, recall) for a single Source 1 entity.

    Parameters:
    - predicted_ids: Set, List, or comma-separated string of predicted S2/S3 IDs.
    - true_ids: Set, List, or comma-separated string of true matching S2/S3 IDs.
    - beta: Beta weight for F-score (default: 0.5 for 2x precision weighting).

    Returns:
    - (f_score, precision, recall) as floats.
    """
    # Normalize inputs to sets
    if isinstance(predicted_ids, str):
        pred_set = {x.strip() for x in predicted_ids.split(",") if x.strip()}
    else:
        pred_set = {x for x in predicted_ids if str(x).strip()}

    if isinstance(true_ids, str):
        true_set = {x.strip() for x in true_ids.split(",") if x.strip()}
    else:
        true_set = {x for x in true_ids if str(x).strip()}

    n_pred = len(pred_set)
    n_true = len(true_set)

    # Singleton Case: No true matches exist
    if n_true == 0:
        if n_pred == 0:
            # Correctly predicted no match
            return 1.0, 1.0, 1.0
        else:
            # False merge penalty: predicted matches for a singleton
            return 0.0, 0.0, 0.0

    # Non-Singleton Case: True matches exist
    if n_pred == 0:
        # Missed all matches
        return 0.0, 0.0, 0.0

    tp = len(pred_set & true_set)
    if tp == 0:
        return 0.0, 0.0, 0.0

    precision = tp / n_pred
    recall = tp / n_true

    beta_sq = beta ** 2
    weight = 1.0 + beta_sq  # 1.25 for beta=0.5
    denominator = (beta_sq * precision) + recall

    if denominator == 0.0:
        f_score = 0.0
    else:
        f_score = (weight * precision * recall) / denominator

    return float(f_score), float(precision), float(recall)


def evaluate_predictions(
    predictions: Union[pd.DataFrame, Dict[str, Union[Set[str], List[str], str]]],
    ground_truth: Union[pd.DataFrame, Dict[str, Union[Set[str], List[str], str]]],
    beta: float = 0.5,
    country_mapping: Optional[Dict[str, str]] = None,
) -> Dict[str, Union[float, int, Dict[str, float]]]:
    """
    Computes macro-averaged F_0.5 score across all Source 1 entities in the evaluation set.

    Parameters:
    - predictions: Either DataFrame with columns ['source1_entity_id', 'matched_entity_ids']
                   or Dict[source1_id -> collection of matched IDs]
    - ground_truth: Either DataFrame with columns ['source1_entity_id', 'matched_entity_ids']
                    or Dict[source1_id -> collection of matched IDs]
    - beta: F-score beta parameter (default: 0.5)
    - country_mapping: Optional dict of source1_id -> country string for per-country diagnostics.

    Returns:
    - Detailed evaluation dictionary containing:
        - macro_f_score (The official competition leaderboard metric)
        - macro_precision
        - macro_recall
        - singleton_accuracy
        - non_singleton_macro_f_score
        - total_entities
        - num_singletons
        - num_non_singletons
        - per_country_f_score (if country_mapping provided)
    """
    # Convert predictions to dictionary
    if isinstance(predictions, pd.DataFrame):
        pred_dict = {}
        for s1, m in zip(predictions["source1_entity_id"], predictions["matched_entity_ids"]):
            if isinstance(m, str):
                pred_dict[s1] = {x.strip() for x in m.split(",") if x.strip()}
            elif isinstance(m, (set, list, tuple)):
                pred_dict[s1] = set(m)
            elif pd.isna(m):
                pred_dict[s1] = set()
            else:
                pred_dict[s1] = set()
    else:
        pred_dict = {
            k: ({x.strip() for x in v.split(",") if x.strip()} if isinstance(v, str) else set(v))
            for k, v in predictions.items()
        }

    # Convert ground truth to dictionary
    if isinstance(ground_truth, pd.DataFrame):
        gt_dict = {}
        for s1, m in zip(ground_truth["source1_entity_id"], ground_truth["matched_entity_ids"]):
            if isinstance(m, str):
                gt_dict[s1] = {x.strip() for x in m.split(",") if x.strip()}
            elif isinstance(m, (set, list, tuple)):
                gt_dict[s1] = set(m)
            elif pd.isna(m):
                gt_dict[s1] = set()
            else:
                gt_dict[s1] = set()
    else:
        gt_dict = {
            k: ({x.strip() for x in v.split(",") if x.strip()} if isinstance(v, str) else set(v))
            for k, v in ground_truth.items()
        }

    all_s1_ids = list(gt_dict.keys())
    total_entities = len(all_s1_ids)

    if total_entities == 0:
        raise ValueError("Ground truth is empty. Cannot evaluate empty set.")

    f_scores = []
    precisions = []
    recalls = []

    singleton_scores = []
    non_singleton_scores = []

    country_scores: Dict[str, List[float]] = {}

    for s1_id in all_s1_ids:
        true_ids = gt_dict[s1_id]
        pred_ids = pred_dict.get(s1_id, set())

        f_val, p_val, r_val = compute_entity_f_beta(pred_ids, true_ids, beta=beta)

        f_scores.append(f_val)
        precisions.append(p_val)
        recalls.append(r_val)

        if len(true_ids) == 0:
            singleton_scores.append(f_val)
        else:
            non_singleton_scores.append(f_val)

        if country_mapping and s1_id in country_mapping:
            c = country_mapping[s1_id]
            if c not in country_scores:
                country_scores[c] = []
            country_scores[c].append(f_val)

    macro_f = float(np.mean(f_scores))
    macro_p = float(np.mean(precisions))
    macro_r = float(np.mean(recalls))

    singleton_acc = float(np.mean(singleton_scores)) if singleton_scores else 1.0
    non_singleton_f = float(np.mean(non_singleton_scores)) if non_singleton_scores else 0.0

    result = {
        "macro_f_score": macro_f,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "singleton_accuracy": singleton_acc,
        "non_singleton_macro_f_score": non_singleton_f,
        "total_entities": total_entities,
        "num_singletons": len(singleton_scores),
        "num_non_singletons": len(non_singleton_scores),
    }

    if country_scores:
        result["per_country_f_score"] = {
            c: float(np.mean(scores)) for c, scores in country_scores.items()
        }

    return result


def create_validation_split(
    train_source1_df: pd.DataFrame,
    train_ground_truth_df: pd.DataFrame,
    val_ratio: float = 0.20,
    random_state: int = 42,
    stratify_by_country: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Creates a clean, reproducible local validation split by holding out 20% of Source 1 entities
    and partitioning their corresponding ground-truth matches.

    Parameters:
    - train_source1_df: DataFrame of Source 1 training records.
    - train_ground_truth_df: DataFrame of ground-truth matches for Source 1 records.
    - val_ratio: Proportion of S1 entities to hold out (default: 0.20 = 20%).
    - random_state: Seed for reproducibility.
    - stratify_by_country: Whether to stratify by the 'country' column.

    Returns:
    - (s1_train_df, s1_val_df, gt_train_df, gt_val_df)
    """
    # Align S1 IDs
    s1_ids = train_source1_df["entity_id"].values
    stratify_target = train_source1_df["country"].values if (stratify_by_country and "country" in train_source1_df.columns) else None

    train_idx, val_idx = train_test_split(
        np.arange(len(train_source1_df)),
        test_size=val_ratio,
        random_state=random_state,
        stratify=stratify_target,
    )

    s1_train_df = train_source1_df.iloc[train_idx].copy().reset_index(drop=True)
    s1_val_df = train_source1_df.iloc[val_idx].copy().reset_index(drop=True)

    val_s1_set = set(s1_val_df["entity_id"])
    train_s1_set = set(s1_train_df["entity_id"])

    # Partition ground truth
    gt_val_mask = train_ground_truth_df["source1_entity_id"].isin(val_s1_set)
    gt_train_mask = train_ground_truth_df["source1_entity_id"].isin(train_s1_set)

    gt_val_df = train_ground_truth_df[gt_val_mask].copy().reset_index(drop=True)
    gt_train_df = train_ground_truth_df[gt_train_mask].copy().reset_index(drop=True)

    return s1_train_df, s1_val_df, gt_train_df, gt_val_df


def calculate_macro_f05(
    y_true_dict: Dict[str, Set[str]],
    y_pred_dict: Dict[str, Set[str]],
    beta: float = 0.5,
) -> float:
    """
    Computes exact macro-averaged F_0.5 score across all Source 1 entities,
    respecting singleton rules (empty prediction on true singleton = 1.0, false merge = 0.0).
    """
    scores = []
    for s1_id, true_set in y_true_dict.items():
        pred_set = y_pred_dict.get(s1_id, set())
        score, _, _ = compute_entity_f_beta(pred_set, true_set, beta=beta)
        scores.append(score)
    return float(np.mean(scores)) if scores else 0.0
