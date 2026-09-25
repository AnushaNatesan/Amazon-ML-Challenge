"""
Business Entity Resolution Pipeline Source Package
Team: GenX H4CK3RS!
"""

from .data_loader import (
    load_source_tsv,
    load_ground_truth_tsv,
    save_tsv,
    validate_source_dataframe,
    resolve_data_paths,
)
from .metrics import (
    compute_entity_f_beta,
    evaluate_predictions,
    create_validation_split,
)
from .text_preprocessing import (
    normalize_text,
    clean_business_name,
    clean_business_address,
    extract_legal_suffix,
    extract_numeric_features,
    preprocess_dataframe,
)
from .eda import (
    run_eda,
    generate_eda_report,
)
from .blocking import (
    extract_blocking_keys,
    MultiPassBlocker,
    evaluate_blocking_recall,
    export_candidate_pairs_tsv,
)
from .features import (
    char_ngram_jaccard,
    token_jaccard,
    token_containment_ratio,
    prepare_record_profile,
    compute_pairwise_features,
    extract_features_for_candidate_pairs,
    FEATURE_NAMES,
)

__all__ = [
    "load_source_tsv",
    "load_ground_truth_tsv",
    "save_tsv",
    "validate_source_dataframe",
    "resolve_data_paths",
    "compute_entity_f_beta",
    "evaluate_predictions",
    "create_validation_split",
    "normalize_text",
    "clean_business_name",
    "clean_business_address",
    "extract_legal_suffix",
    "extract_numeric_features",
    "preprocess_dataframe",
    "run_eda",
    "generate_eda_report",
    "extract_blocking_keys",
    "MultiPassBlocker",
    "evaluate_blocking_recall",
    "export_candidate_pairs_tsv",
    "char_ngram_jaccard",
    "token_jaccard",
    "token_containment_ratio",
    "prepare_record_profile",
    "compute_pairwise_features",
    "extract_features_for_candidate_pairs",
    "FEATURE_NAMES",
]
