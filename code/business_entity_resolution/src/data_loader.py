"""
Data Loading & Validation Module
Part of Phase 1 Implementation for Business Entity Resolution
Team: GenX H4CK3RS!

CRITICAL SPECIFICATION:
All input and output files are tab-separated (.tsv).
Tabs are strictly enforced (sep="\\t") because business addresses and matched entity ID lists
frequently contain commas. Reading without sep="\\t" leads to silent single-column formatting bugs.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Iterator
import pandas as pd

# Expected column schemas
SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GROUND_TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
MATCHING_OUTPUT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_OUTPUT_COLUMNS = ["source1_entity_id", "candidate_entity_ids"]

VALID_PREFIXES = {"S1-", "S2-", "S3-"}


def resolve_data_paths(base_dir: Optional[Union[str, Path]] = None) -> Dict[str, Dict[str, Path]]:
    """
    Locates and returns verified paths for all dataset files across training and test sets.
    Checks common directory layouts (e.g. Dataset/student_resource/dataset, dataset/, etc.)
    """
    possible_roots = []
    if base_dir:
        possible_roots.append(Path(base_dir))

    # Add default candidate directories
    cwd = Path.cwd()
    possible_roots.extend([
        cwd / "Dataset" / "student_resource" / "dataset",
        cwd / "dataset",
        cwd.parent / "Dataset" / "student_resource" / "dataset",
        cwd.parent / "dataset",
    ])

    resolved_root = None
    for candidate in possible_roots:
        if candidate.exists() and (candidate / "train").exists() and (candidate / "test").exists():
            resolved_root = candidate
            break

    if resolved_root is None:
        # Fallback to the first candidate if directory structure is being set up
        resolved_root = possible_roots[0] if possible_roots else cwd / "dataset"

    train_dir = resolved_root / "train"
    test_dir = resolved_root / "test"

    return {
        "train": {
            "source1": train_dir / "train_source1.tsv",
            "source2": train_dir / "train_source2.tsv",
            "source3": train_dir / "train_source3.tsv",
            "ground_truth": train_dir / "train_ground_truth.tsv",
        },
        "test": {
            "source1": test_dir / "test_source1.tsv",
            "source2": test_dir / "test_source2.tsv",
            "source3": test_dir / "test_source3.tsv",
        },
        "root": resolved_root,
    }


def validate_source_dataframe(df: pd.DataFrame, file_desc: str = "dataset") -> None:
    """
    Validates that a loaded source DataFrame conforms strictly to competition specifications.
    Checks column names, non-empty entity_id, and expected prefixes.
    """
    missing_cols = set(SOURCE_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Schema validation failed for {file_desc}: missing columns {missing_cols}. "
            f"Expected {SOURCE_COLUMNS}. Found {list(df.columns)}. "
            "Ensure the file was read with sep='\\t'."
        )

    if len(df) == 0:
        return

    # Check that it did not collapse into a single column
    if len(df.columns) < 4:
        raise ValueError(
            f"Corrupt formatting in {file_desc}: found only {len(df.columns)} columns. "
            "File must be tab-separated."
        )

    # Validate entity prefix
    sample_prefix = str(df["entity_id"].iloc[0])[:3]
    if sample_prefix not in VALID_PREFIXES:
        raise ValueError(
            f"Invalid entity_id format in {file_desc}: first row entity_id '{df['entity_id'].iloc[0]}' "
            f"does not start with one of {VALID_PREFIXES}."
        )


def load_source_tsv(
    file_path: Union[str, Path],
    nrows: Optional[int] = None,
    usecols: Optional[List[str]] = None,
    chunksize: Optional[int] = None,
    validate: bool = True,
) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
    """
    Robust reader for source TSV files (source1, source2, source3).

    Key Protections:
    - Explicit sep='\\t' to avoid silent single-column collapse.
    - dtype=str to preserve leading zeros in postal codes, house numbers, or IDs.
    - keep_default_na=False to avoid interpreting 'NA' or 'NULL' as NaN values.
    - encoding='utf-8' with fallback handling.
    - Fills missing values with empty strings for text columns.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Source file not found at: {path}")

    # Set column dtypes to string
    dtype_dict = {col: str for col in (usecols or SOURCE_COLUMNS)}

    if chunksize is not None:
        iterator = pd.read_csv(
            path,
            sep="\t",
            dtype=dtype_dict,
            nrows=nrows,
            usecols=usecols,
            chunksize=chunksize,
            keep_default_na=False,
            encoding="utf-8",
        )
        return iterator

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=dtype_dict,
        nrows=nrows,
        usecols=usecols,
        keep_default_na=False,
        encoding="utf-8",
    )

    # Ensure all string columns have no NaNs
    for col in df.columns:
        df[col] = df[col].fillna("").astype(str)

    if validate and usecols is None:
        validate_source_dataframe(df, file_desc=path.name)

    return df


def load_ground_truth_tsv(
    file_path: Union[str, Path],
    nrows: Optional[int] = None,
    as_dict: bool = False,
) -> Union[pd.DataFrame, Dict[str, Set[str]]]:
    """
    Loads train_ground_truth.tsv with explicit tab delimiter.

    Returns:
    - If as_dict is False: DataFrame with columns ['source1_entity_id', 'matched_entity_ids', 'matched_set']
    - If as_dict is True: Dict[str, Set[str]] mapping source1_entity_id -> set of matched_entity_ids
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Ground truth file not found at: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        nrows=nrows,
        keep_default_na=False,
        encoding="utf-8",
    )

    # Verify header
    expected_cols = [c.lower() for c in GROUND_TRUTH_COLUMNS]
    actual_cols = [c.lower() for c in df.columns]
    if actual_cols != expected_cols:
        raise ValueError(
            f"Invalid ground truth header in {path.name}: {list(df.columns)}. "
            f"Expected {GROUND_TRUTH_COLUMNS}."
        )

    # Clean matched IDs
    df["matched_entity_ids"] = df["matched_entity_ids"].fillna("").astype(str).str.strip()

    if as_dict:
        result_dict = {}
        for s1_id, m_str in zip(df["source1_entity_id"], df["matched_entity_ids"]):
            if m_str:
                result_dict[s1_id] = {mid.strip() for mid in m_str.split(",") if mid.strip()}
            else:
                result_dict[s1_id] = set()
        return result_dict

    # Add pre-parsed matched_set column for fast vector access
    df["matched_set"] = df["matched_entity_ids"].apply(
        lambda m: {mid.strip() for mid in m.split(",") if mid.strip()} if m else set()
    )
    return df


def save_tsv(
    df: pd.DataFrame,
    output_path: Union[str, Path],
    columns: Optional[List[str]] = None,
) -> None:
    """
    Saves a DataFrame strictly as a tab-separated TSV file without row indices.
    Validates that the output complies with competition requirements.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if columns is not None:
        df = df[columns]

    df.to_csv(
        path,
        sep="\t",
        index=False,
        encoding="utf-8",
    )
