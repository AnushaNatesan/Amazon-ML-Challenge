#!/usr/bin/env python3
"""
Unified End-to-End Pipeline Verification Suite (Phases 1, 2, and 3)
Team: GenX H4CK3RS!
"""

import sys
import unittest
from pathlib import Path

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Add parent directory to path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from run_phase1_phase2_tests import (
    TestPhase1DataLoader,
    TestPhase1Metrics,
    TestPhase2TextPreprocessing,
)
from run_phase3_blocking_tests import TestPhase3Blocking


def run_all_tests():
    print("=" * 70)
    print("   ML Challenge 2026 — End-to-End Test Suite (Phases 1, 2, 3)")
    print("   Team: GenX H4CK3RS!")
    print("=" * 70)

    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    suite.addTests(loader.loadTestsFromTestCase(TestPhase1DataLoader))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase1Metrics))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase2TextPreprocessing))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase3Blocking))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        print("\n" + "=" * 70)
        print("  [PASS] ALL PIPELINE TESTS (PHASES 1, 2, 3) PASSED SUCCESSFULLY!")
        print("=" * 70)
        return 0
    else:
        print("\n" + "=" * 70)
        print(f"  [FAIL] {len(result.failures)} failures, {len(result.errors)} errors encountered.")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
