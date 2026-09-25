#!/usr/bin/env python3
"""
Test Suite and Verification Runner for Phase 4: Pairwise Feature Engineering
Team: GenX H4CK3RS!

Verifies:
1. Fast RapidFuzz string similarities (Levenshtein, Jaro-Winkler, Token Sort/Set).
2. Word-order transposition tolerance ("Blue Co Zander" vs "Zander Blue Co").
3. Safe missing address handling (~3.3% missing records in S2/S3).
4. Component disagreement detection for building numbers and postal codes (+1, 0, -1).
5. Legal form compatibility (+1, 0, -1).
6. Batch candidate pair feature extraction and training label alignment.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Add parent directory to path so src can be imported cleanly
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from src.features import (
    char_ngram_jaccard,
    token_jaccard,
    prepare_record_profile,
    compute_pairwise_features,
    extract_features_for_candidate_pairs,
    FEATURE_NAMES,
)


class TestPhase4Features(unittest.TestCase):
    """Unit and Integration Tests for Phase 4 Feature Engineering."""

    def test_name_similarity_metrics(self):
        """Verifies string similarity metrics, including word transposition tolerance."""
        r1 = {"entity_id": "S1-1", "business_name": "Zander Blue Trading", "business_address": "100 Main St", "country": "US"}
        r2 = {"entity_id": "S2-1", "business_name": "Blue Trading Zander", "business_address": "100 Main St", "country": "US"}


        p1 = prepare_record_profile(r1)
        p2 = prepare_record_profile(r2)
        feat = compute_pairwise_features(p1, p2)

        # Token sort ratio should perfectly handle word order transposition
        self.assertEqual(feat["name_token_sort_ratio"], 1.0)
        self.assertEqual(feat["name_token_set_ratio"], 1.0)
        self.assertEqual(feat["name_token_jaccard"], 1.0)
        self.assertGreater(feat["name_jaro_winkler"], 0.6)

    def test_missing_address_safety(self):
        """Ensures that missing addresses (~3.3% in S2/S3) produce valid indicators and no NaNs."""
        r1 = {"entity_id": "S1-2", "business_name": "Zephay Labs Inc", "business_address": "2621 Cotten Road", "country": "US"}
        # Candidate with null/empty address
        r2 = {"entity_id": "S2-2", "business_name": "Zephay Labs", "business_address": "", "country": "US"}

        p1 = prepare_record_profile(r1)
        p2 = prepare_record_profile(r2)
        feat = compute_pairwise_features(p1, p2)

        # Flag must be 1.0
        self.assertEqual(feat["addr_is_missing"], 1.0)
        # All address metrics should default gracefully to 0.0 without errors or NaNs
        self.assertEqual(feat["addr_levenshtein"], 0.0)
        self.assertEqual(feat["addr_token_jaccard"], 0.0)
        self.assertEqual(feat["building_number_match"], 0.0)
        self.assertEqual(feat["postal_code_match"], 0.0)
        # Name should still match strongly
        self.assertEqual(feat["name_exact_match"], 1.0)
        self.assertFalse(any(np.isnan(v) for v in feat.values()))

    def test_numeric_component_disagreement(self):
        """Verifies ternary matching (+1 = match, -1 = conflict, 0 = neutral) for numbers and postal codes."""
        # 1. Matching building numbers and postal code
        r1 = {"entity_id": "S1-3", "business_name": "Horizon Solar", "business_address": "1064 Newton Rd, TX 75701", "country": "US"}
        r2 = {"entity_id": "S2-3", "business_name": "Horizon Solar Inc", "business_address": "1064 Newton Road, TX 75701", "country": "US"}
        p1, p2 = prepare_record_profile(r1), prepare_record_profile(r2)
        feat_match = compute_pairwise_features(p1, p2)

        self.assertEqual(feat_match["building_number_match"], 1.0)
        self.assertEqual(feat_match["postal_code_match"], 1.0)
        self.assertEqual(feat_match["has_shared_number"], 1.0)

        # 2. Conflicting building numbers and postal codes
        r3 = {"entity_id": "S2-4", "business_name": "Horizon Solar", "business_address": "500 Market St, NY 10001", "country": "US"}
        p3 = prepare_record_profile(r3)
        feat_conflict = compute_pairwise_features(p1, p3)

        self.assertEqual(feat_conflict["building_number_match"], -1.0)
        self.assertEqual(feat_conflict["postal_code_match"], -1.0)

    def test_legal_form_compatibility(self):
        """Verifies legal form compatibility detection (+1 match, -1 conflict, 0 neutral)."""
        # Matching legal forms (Pvt Ltd vs Private Limited)
        r1 = {"entity_id": "S1-5", "business_name": "Om Constructions Pvt Ltd", "business_address": "Rewari", "country": "India"}
        r2 = {"entity_id": "S2-5", "business_name": "OM CONSTRUCTIONS PRIVATE LIMITED", "business_address": "Rewari", "country": "India"}
        p1, p2 = prepare_record_profile(r1), prepare_record_profile(r2)
        feat_match = compute_pairwise_features(p1, p2)
        self.assertEqual(feat_match["legal_form_match"], 1.0)

        # Conflicting legal forms (Pvt Ltd vs LLP)
        r3 = {"entity_id": "S2-6", "business_name": "Om Constructions LLP", "business_address": "Rewari", "country": "India"}
        p3 = prepare_record_profile(r3)
        feat_conflict = compute_pairwise_features(p1, p3)
        self.assertEqual(feat_conflict["legal_form_match"], -1.0)

    def test_batch_feature_extraction_pipeline(self):
        """Verifies end-to-end feature matrix extraction from candidate dictionary with labels."""
        s1_df = pd.DataFrame([
            {"entity_id": "S1-10", "business_name": "Zephay Labs Inc", "business_address": "2621 Cotten Rd, 75701", "country": "US"},
            {"entity_id": "S1-20", "business_name": "Vision Partners", "business_address": "1064 Newton Rd", "country": "US"},
        ])

        target_df = pd.DataFrame([
            {"entity_id": "S2-101", "business_name": "Zephay Labs", "business_address": "2621 Cotten Road, 75701", "country": "US"},
            {"entity_id": "S2-102", "business_name": "Zephay Dental", "business_address": "999 Oak St", "country": "US"},
            {"entity_id": "S3-201", "business_name": "Vision Partners Corp", "business_address": "1064 Newton Road", "country": "US"},
        ])

        candidate_dict = {
            "S1-10": ["S2-101", "S2-102"],
            "S1-20": ["S3-201"],
        }

        ground_truth = {
            "S1-10": {"S2-101"},  # S2-101 is match, S2-102 is negative
            "S1-20": {"S3-201"},  # S3-201 is match
        }

        feat_df, labels, pair_ids = extract_features_for_candidate_pairs(
            candidate_dict, s1_df, target_df, ground_truth_dict=ground_truth
        )

        self.assertEqual(len(feat_df), 3)
        self.assertEqual(len(labels), 3)
        self.assertListEqual(list(labels), [1, 0, 1])
        self.assertEqual(len(pair_ids), 3)
        self.assertListEqual(list(feat_df.columns), FEATURE_NAMES)
        self.assertFalse(feat_df.isna().any().any())


def run_phase4_suite():
    """Executes Phase 4 feature engineering unit and integration tests."""
    print("=" * 70)
    print("   ML Challenge 2026 — Phase 4 Feature Engineering Test Suite")
    print("   Team: GenX H4CK3RS!")
    print("=" * 70)

    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    suite.addTests(loader.loadTestsFromTestCase(TestPhase4Features))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        print("\n" + "=" * 70)
        print("  [PASS] ALL PHASE 4 FEATURE ENGINEERING TESTS PASSED SUCCESSFULLY!")
        print("=" * 70)
        return 0
    else:
        print("\n" + "=" * 70)
        print(f"  [FAIL] {len(result.failures)} failures, {len(result.errors)} errors encountered.")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(run_phase4_suite())
