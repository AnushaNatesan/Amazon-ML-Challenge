#!/usr/bin/env python3
"""
Test Suite and Verification Runner for Phase 1 and Phase 2 Modules
Team: GenX H4CK3RS!

Verifies:
1. Data Loader & Schema Validation (Strict TSV enforcement, schema protection)
2. Local Evaluation Harness & Exact Macro F_0.5 Metric (Precision weighting, singletons)
3. 20% Stratified Validation Split
4. Text Normalization, Accent Handling, Legal Suffix Extraction, Numeric Extraction
5. Integration verification on actual dataset records
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

from src.data_loader import (
    load_source_tsv,
    load_ground_truth_tsv,
    save_tsv,
    validate_source_dataframe,
    resolve_data_paths,
)
from src.metrics import (
    compute_entity_f_beta,
    evaluate_predictions,
    create_validation_split,
)
from src.text_preprocessing import (
    strip_accents,
    clean_business_name,
    clean_business_address,
    extract_legal_form,
    extract_numeric_features,
    preprocess_dataframe,
)
from src.eda import inspect_file_metadata


class TestPhase1DataLoader(unittest.TestCase):
    """Verifies Data Loading, TSV formatting integrity, and schema checks."""

    def setUp(self):
        self.paths = resolve_data_paths()
        self.tmp_dir = current_dir / "tmp_test_data"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        import shutil
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir)

    def test_tsv_loading_preserves_columns(self):
        """Ensures that tabs are strictly respected and columns do not collapse into one."""
        s1_path = self.paths["train"]["source1"]
        self.assertTrue(s1_path.exists(), f"Source 1 not found at {s1_path}")

        df = load_source_tsv(s1_path, nrows=50)
        self.assertEqual(len(df.columns), 4)
        self.assertListEqual(
            list(df.columns),
            ["entity_id", "business_name", "business_address", "country"]
        )
        # Prefix check
        self.assertTrue(df["entity_id"].iloc[0].startswith("S1-"))

    def test_ground_truth_parsing(self):
        """Verifies ground truth loading both as DataFrame and dict of sets."""
        gt_path = self.paths["train"]["ground_truth"]
        self.assertTrue(gt_path.exists(), f"Ground truth not found at {gt_path}")

        df_gt = load_ground_truth_tsv(gt_path, nrows=100)
        self.assertIn("matched_set", df_gt.columns)
        self.assertIsInstance(df_gt["matched_set"].iloc[0], set)

        dict_gt = load_ground_truth_tsv(gt_path, nrows=100, as_dict=True)
        self.assertIsInstance(dict_gt, dict)
        first_key = list(dict_gt.keys())[0]
        self.assertTrue(first_key.startswith("S1-"))
        self.assertIsInstance(dict_gt[first_key], set)

    def test_save_and_reload_tsv(self):
        """Ensures save_tsv writes strict tab-delimited files that pass schema validation."""
        s1_path = self.paths["train"]["source1"]
        df_orig = load_source_tsv(s1_path, nrows=20)

        out_path = self.tmp_dir / "sample_out.tsv"
        save_tsv(df_orig, out_path)

        df_reloaded = load_source_tsv(out_path)
        self.assertEqual(len(df_orig), len(df_reloaded))
        self.assertListEqual(list(df_orig.columns), list(df_reloaded.columns))


class TestPhase1Metrics(unittest.TestCase):
    """Verifies the exact competition F_0.5 metric, singleton rules, and validation split."""

    def test_competition_readme_exact_example(self):
        """
        Tests the exact example provided in the competition README:
        - Predicts: [S2-00047, S2-00193, S3-00812]
        - Ground truth: [S2-00047, S3-00812]
        - Precision: 2/3, Recall: 2/2 = 1.0
        - F_0.5: 0.714
        """
        preds = ["S2-00047", "S2-00193", "S3-00812"]
        truth = ["S2-00047", "S3-00812"]

        f_val, p_val, r_val = compute_entity_f_beta(preds, truth, beta=0.5)

        self.assertAlmostEqual(p_val, 2 / 3, places=4)
        self.assertAlmostEqual(r_val, 1.0, places=4)
        # Expected: (1.25 * (2/3) * 1.0) / (0.25 * (2/3) + 1.0) = 0.833333 / 1.166667 = 0.7142857
        self.assertAlmostEqual(f_val, 0.7142857, places=4)

    def test_singleton_handling_exact(self):
        """
        Tests singleton rules:
        - True = [] and Pred = [] -> F_0.5 = 1.0
        - True = [] and Pred = ['S2-00001'] -> F_0.5 = 0.0 (punish false merge)
        - True = ['S2-00001'] and Pred = [] -> F_0.5 = 0.0 (missed match)
        """
        # Correct singleton prediction
        f_val, p_val, r_val = compute_entity_f_beta([], [], beta=0.5)
        self.assertEqual(f_val, 1.0)
        self.assertEqual(p_val, 1.0)
        self.assertEqual(r_val, 1.0)

        # False merge on singleton
        f_val, p_val, r_val = compute_entity_f_beta(["S2-99999"], [], beta=0.5)
        self.assertEqual(f_val, 0.0)
        self.assertEqual(p_val, 0.0)

        # Missed match on non-singleton
        f_val, p_val, r_val = compute_entity_f_beta([], ["S2-99999"], beta=0.5)
        self.assertEqual(f_val, 0.0)

    def test_macro_evaluation(self):
        """Tests evaluate_predictions over a batch of mixed singleton and non-singleton entities."""
        predictions = {
            "S1-1": ["S2-10", "S2-20"],  # Perfect match
            "S1-2": [],                   # Correct singleton
            "S1-3": ["S2-30"],            # False merge on singleton
            "S1-4": ["S2-40"],            # Half precision, full recall
        }
        ground_truth = {
            "S1-1": ["S2-10", "S2-20"],  # F = 1.0
            "S1-2": [],                   # F = 1.0
            "S1-3": [],                   # F = 0.0
            "S1-4": ["S2-40", "S3-40"],  # P=1.0, R=0.5 -> F_0.5 = (1.25*1*0.5)/(0.25*1+0.5) = 0.625/0.75 = 0.8333
        }

        results = evaluate_predictions(predictions, ground_truth, beta=0.5)

        expected_scores = [1.0, 1.0, 0.0, 0.8333333]
        expected_macro_f = sum(expected_scores) / len(expected_scores)

        self.assertAlmostEqual(results["macro_f_score"], expected_macro_f, places=4)
        self.assertEqual(results["total_entities"], 4)
        self.assertEqual(results["num_singletons"], 2)
        self.assertEqual(results["singleton_accuracy"], 0.5)  # 1 out of 2 singletons correct

    def test_validation_split_stratification(self):
        """Verifies the 20% hold-out validation split preserves country ratios and is disjoint."""
        paths = resolve_data_paths()
        s1_df = load_source_tsv(paths["train"]["source1"], nrows=5000)
        gt_df = load_ground_truth_tsv(paths["train"]["ground_truth"], nrows=5000)

        s1_train, s1_val, gt_train, gt_val = create_validation_split(
            s1_df, gt_df, val_ratio=0.20, random_state=42
        )

        self.assertEqual(len(s1_train) + len(s1_val), len(s1_df))
        self.assertAlmostEqual(len(s1_val) / len(s1_df), 0.20, delta=0.01)

        # Check disjointness
        train_ids = set(s1_train["entity_id"])
        val_ids = set(s1_val["entity_id"])
        self.assertEqual(len(train_ids & val_ids), 0)

        # Check stratification
        orig_us_ratio = (s1_df["country"] == "US").mean()
        val_us_ratio = (s1_val["country"] == "US").mean()
        self.assertAlmostEqual(orig_us_ratio, val_us_ratio, delta=0.02)


class TestPhase2TextPreprocessing(unittest.TestCase):
    """Verifies multilingual text normalization, legal suffix handling, and numeric extraction."""

    def test_unicode_accents_and_ligatures(self):
        """Tests Latin accent removal, European ligatures, and script preservation."""
        # French accents & ligatures
        french_sample = "175 Boulevard du Président Franklin Roosevelt œuvres naïves"
        cleaned_fr = strip_accents(french_sample)
        self.assertEqual(cleaned_fr, "175 Boulevard du President Franklin Roosevelt oeuvres naives")

        # Noise accents injected in English words
        noise_sample = "Gautam Nagar Recording Prívate Limited CÓNSULTANTS"
        cleaned_noise = strip_accents(noise_sample)
        self.assertEqual(cleaned_noise, "Gautam Nagar Recording Private Limited CONSULTANTS")

        # Devanagari script preservation
        hindi_sample = "Rewari, Haryana, Gokalgarh, हरियाणा"
        cleaned_hi = strip_accents(hindi_sample)
        self.assertIn("हरियाणा", cleaned_hi)

    def test_country_agnostic_legal_suffixes(self):
        """Tests legal suffix stripping across US, India, and France."""
        # US / UK
        r_us = clean_business_name("Zephay Labs Inc.")
        self.assertEqual(r_us["name_clean"], "zephay labs")
        self.assertEqual(r_us["legal_form"], "inc")

        # India
        r_in = clean_business_name("Om Constructions Pvt Ltd")
        self.assertEqual(r_in["name_clean"], "om constructions")
        self.assertEqual(r_in["legal_form"], "pvt_ltd")

        r_llp = clean_business_name("Nandlal Kisan LLP")
        self.assertEqual(r_llp["name_clean"], "nandlal kisan")
        self.assertEqual(r_llp["legal_form"], "llp")

        # France Suffix
        r_fr = clean_business_name("Elephant Centre EURL")
        self.assertEqual(r_fr["name_clean"], "elephant centre")
        self.assertEqual(r_fr["legal_form"], "eurl")

        # France Prefix (Common in Source 2 / Source 3)
        r_sci = clean_business_name("SCI Ptit Àmicale")
        self.assertEqual(r_sci["name_clean"], "ptit amicale")
        self.assertEqual(r_sci["legal_form"], "sci")

        # France Brackets
        r_bracket = clean_business_name("QHC Culture [EURL]")
        self.assertEqual(r_bracket["name_clean"], "qhc culture")
        self.assertEqual(r_bracket["legal_form"], "eurl")

    def test_address_abbreviations_expansion(self):
        """Tests address abbreviation normalization for US, Indian, and French patterns."""
        # US Address
        us_addr = "6207 OCEAN FRONT AVE, VIRGINIA BEACH CITY, VA"
        clean_us = clean_business_address(us_addr)
        self.assertIn("avenue", clean_us["address_tokens"])

        # French Address
        fr_addr = "63 R. DE DIEPPE, LILLE"
        clean_fr = clean_business_address(fr_addr)
        self.assertIn("rue", clean_fr["address_tokens"])

        # Boulevard abbreviation
        bd_addr = "154 BD du President Wilson"
        clean_bd = clean_business_address(bd_addr)
        self.assertIn("boulevard", clean_bd["address_tokens"])

    def test_numeric_feature_extraction(self):
        """Tests postal code detection and building number isolation with zero-stripping."""
        # US Address with 5-digit ZIP and building number
        us_res = extract_numeric_features("2621 Cotten Road, Tyler, TX 75701", country="US")
        self.assertEqual(us_res["postal_code"], "75701")
        self.assertIn("2621", us_res["building_numbers"])

        # Indian Address with 6-digit PIN and house number
        in_res = extract_numeric_features("House No.-37, Sector-3, Rewari, Haryana 123401", country="India")
        self.assertEqual(in_res["postal_code"], "123401")
        self.assertIn("37", in_res["all_numbers"])

        # French Address with 5-digit postal code
        fr_res = extract_numeric_features("175 Boulevard Roosevelt, 33000 Bordeaux", country="France")
        self.assertEqual(fr_res["postal_code"], "33000")
        self.assertIn("175", fr_res["building_numbers"])

        # Leading zero normalization in alphanumeric designators (K-00303 -> K-303)
        zero_res = extract_numeric_features("K-00303 Gautam Nagar, Delhi")
        self.assertTrue(any("303" in num for num in zero_res["building_numbers"]))


def run_full_suite():
    """Executes all unit tests and prints verification results."""
    print("=" * 70)
    print("   ML Challenge 2026 — Phase 1 & Phase 2 Test Verification Suite")
    print("   Team: GenX H4CK3RS!")
    print("=" * 70)

    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    suite.addTests(loader.loadTestsFromTestCase(TestPhase1DataLoader))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase1Metrics))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase2TextPreprocessing))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        print("\n" + "=" * 70)
        print("  [PASS] ALL PHASE 1 AND PHASE 2 TESTS PASSED SUCCESSFULLY!")
        print("=" * 70)
        return 0
    else:
        print("\n" + "=" * 70)
        print(f"  [FAIL] {len(result.failures)} failures, {len(result.errors)} errors encountered.")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    exit_code = run_full_suite()
    sys.exit(exit_code)
