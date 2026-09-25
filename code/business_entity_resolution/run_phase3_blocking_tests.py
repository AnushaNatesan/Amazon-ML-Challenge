#!/usr/bin/env python3
"""
Test Suite and Verification Runner for Phase 3: High-Recall Multi-Pass Blocking
Team: GenX H4CK3RS!

Verifies:
1. Hard country partitioning in inverted index creation.
2. Multi-pass blocking key extraction (Key A, B, C, D) across US, India, and France.
3. Candidate generation ranking and maximum candidate pruning.
4. Validation recall ceiling measurement (ensuring >= 97% recall on target pairs).
5. Output format conformity for candidate_pairs.tsv.
"""

import sys
import unittest
from pathlib import Path

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Add parent directory to path so src can be imported cleanly
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from src.blocking import (
    extract_blocking_keys,
    MultiPassBlocker,
    evaluate_blocking_recall,
    export_candidate_pairs_tsv,
)
from src.data_loader import load_source_tsv, load_ground_truth_tsv, resolve_data_paths


class TestPhase3Blocking(unittest.TestCase):
    """Unit and Integration Tests for Phase 3 Candidate Generation."""

    def test_blocking_key_extraction_multilingual(self):
        """Verifies multi-pass blocking key generation across US, India, and France."""
        # 1. US Record
        us_keys = extract_blocking_keys(
            business_name="Zephay Labs Inc.",
            business_address="2621 Cotten Road, Tyler, TX 75701",
            country="US",
        )
        self.assertIn("exact:zephay labs", us_keys)
        self.assertIn("ntok:zephay", us_keys)
        self.assertIn("b_name:2621_zephay", us_keys)
        self.assertIn("p_name:75701_zephay", us_keys)
        self.assertIn("pref4:zeph", us_keys)

        # 2. Indian Record with zero-padded house number
        in_keys = extract_blocking_keys(
            business_name="Om Constructions Pvt Ltd",
            business_address="House No.-0037, Sector-3, Rewari, Haryana 123401",
            country="India",
        )
        self.assertIn("exact:om constructions", in_keys)
        self.assertIn("b_name:37_om", in_keys)
        self.assertIn("p_name:123401_om", in_keys)

        # 3. French Record with prefix legal entity
        fr_keys = extract_blocking_keys(
            business_name="SCI Ptit Àmicale",
            business_address="18 RUE JEN ZAY, Dunkerque 59140",
            country="France",
        )
        # Prefix legal entity SCI should be stripped from exact name
        self.assertIn("exact:ptit amicale", fr_keys)
        self.assertIn("ntok:amicale", fr_keys)
        self.assertIn("b_name:18_ptit", fr_keys)
        self.assertIn("p_name:59140_ptit", fr_keys)

    def test_hard_country_partitioning(self):
        """Ensures that records from different countries are isolated into distinct indices."""
        blocker = MultiPassBlocker(max_candidates_per_entity=10)

        import pandas as pd
        targets = pd.DataFrame([
            {"entity_id": "S2-100", "business_name": "Apex Tech Inc", "business_address": "10 Main St", "country": "US"},
            {"entity_id": "S2-200", "business_name": "Apex Tech Pvt Ltd", "business_address": "10 MG Road", "country": "India"},
            {"entity_id": "S2-300", "business_name": "Apex Tech SARL", "business_address": "10 Rue de Paris", "country": "France"},
        ])

        blocker.fit_target_pool(targets)

        # Query US entity
        cands_us = blocker.generate_candidates_for_entity("S1-1", "Apex Tech", "10 Main St", "US")
        self.assertIn("S2-100", cands_us)
        self.assertNotIn("S2-200", cands_us)
        self.assertNotIn("S2-300", cands_us)

        # Query France entity
        cands_fr = blocker.generate_candidates_for_entity("S1-2", "Apex Tech", "10 Rue de Paris", "France")
        self.assertIn("S2-300", cands_fr)
        self.assertNotIn("S2-100", cands_fr)

    def test_candidate_ranking_and_capping(self):
        """Verifies that candidates with higher key overlap rank higher and respect maximum cap."""
        blocker = MultiPassBlocker(max_candidates_per_entity=2)

        import pandas as pd
        targets = pd.DataFrame([
            # 1 overlapping key (first token 'horizon')
            {"entity_id": "S2-1", "business_name": "Horizon Logistics LLC", "business_address": "999 Oak St", "country": "US"},
            # 4 overlapping keys (exact name, building number, street, 4-char prefix)
            {"entity_id": "S2-2", "business_name": "Horizon Solar Inc", "business_address": "500 Market St", "country": "US"},
            # 2 overlapping keys
            {"entity_id": "S2-3", "business_name": "Horizon Solar Farm", "business_address": "111 Pine St", "country": "US"},
        ])

        blocker.fit_target_pool(targets)

        cands = blocker.generate_candidates_for_entity(
            "S1-10", "Horizon Solar Inc", "500 Market St", "US"
        )
        # Cap is 2: S2-2 must be #1, followed by S2-3 or S2-1, total len <= 2
        self.assertEqual(len(cands), 2)
        self.assertEqual(cands[0], "S2-2")

    def test_recall_ceiling_evaluation(self):
        """Tests the evaluation metric calculation for blocking recall."""
        candidate_dict = {
            "S1-1": ["S2-10", "S2-20", "S3-30"],  # captured both true
            "S1-2": ["S2-40"],                     # missed S3-50
            "S1-3": [],                            # correct empty singleton
        }
        ground_truth_dict = {
            "S1-1": {"S2-10", "S2-20"},            # 2 true matches
            "S1-2": {"S2-40", "S3-50"},            # 2 true matches
            "S1-3": set(),                         # singleton
        }

        res = evaluate_blocking_recall(candidate_dict, ground_truth_dict)
        # Total true matches = 4, captured = 3 (75% recall ceiling)
        self.assertAlmostEqual(res["recall_ceiling"], 3 / 4, places=4)
        self.assertEqual(res["entity_coverage"], 1.0)
        self.assertEqual(res["singleton_retention_rate"], 1.0)

    def test_export_schema_candidate_pairs(self):
        """Ensures that exported candidate_pairs.tsv matches the competition schema."""
        import shutil
        tmp_dir = current_dir / "tmp_test_export"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        out_file = tmp_dir / "candidate_pairs.tsv"

        try:
            sample_candidates = {
                "S1-00001": ["S2-00047", "S2-00193", "S3-00812"],
                "S1-00002": ["S3-00004"],
                "S1-00003": [],  # Singleton
            }

            export_candidate_pairs_tsv(sample_candidates, out_file)

            # Read raw lines to verify strict TAB separator and lack of quoting
            with open(out_file, "r", encoding="utf-8") as f:
                lines = [line.rstrip("\r\n") for line in f]

            self.assertEqual(lines[0], "source1_entity_id\tcandidate_entity_ids")
            self.assertEqual(lines[1], "S1-00001\tS2-00047,S2-00193,S3-00812")
            self.assertEqual(lines[2], "S1-00002\tS3-00004")
            self.assertEqual(lines[3], "S1-00003\t")

        finally:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)


def run_phase3_suite():
    """Executes Phase 3 unit and integration tests."""
    print("=" * 70)
    print("   ML Challenge 2026 — Phase 3 Blocking Test Verification Suite")
    print("   Team: GenX H4CK3RS!")
    print("=" * 70)

    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    suite.addTests(loader.loadTestsFromTestCase(TestPhase3Blocking))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        print("\n" + "=" * 70)
        print("  [PASS] ALL PHASE 3 BLOCKING TESTS PASSED SUCCESSFULLY!")
        print("=" * 70)
        return 0
    else:
        print("\n" + "=" * 70)
        print(f"  [FAIL] {len(result.failures)} failures, {len(result.errors)} errors encountered.")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(run_phase3_suite())
