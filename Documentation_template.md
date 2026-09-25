# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** GenX H4CK3RS!  
**Team Members:** GenX H4CK3RS! Team  
**Submission Date:** September 2026  

---

## 1. Executive Summary
This project implements a high-performance, modular Business Entity Resolution (ER) system designed to link business entity records across three heterogeneous and noisy data sources (Source 1 reference against Source 2 and Source 3). Our architecture combines robust country-partitioned candidate generation (blocking), high-precision feature engineering with multilingual normalization, and calibrated classification optimized specifically for the macro-averaged F_0.5 metric.

---

## 2. Methodology

### 2.1 Problem Analysis
Key insights discovered during Exploratory Data Analysis (EDA):
- **Scale:** Over 2.2 million Source 1 entities in training and 1.73 million in testing, with over 10 million combined records across Source 2 and Source 3.
- **Unseen Country in Test:** While training contains `US` (60.0%) and `India` (40.0%), the test set introduces `France` (~15.0% of records). Pipelines must remain strictly country-agnostic and never assume a closed country domain.
- **Zero Cross-Country Matching:** Analysis demonstrates zero ground-truth matches across differing countries. Country partitioning serves as an optimal hard constraint.
- **Noise Patterns:** Address token permutations, missing fields (up to 3.3% missing addresses in S2/S3), Devanagari script transliterations, artificial accents injected into Latin text, zero-padded building numbers, and legal entity suffix variations.
- **Singletons:** Exactly 5.58% (123,247) of Source 1 entities in training have no matching entities in Source 2 or 3. Macro F_0.5 assigns a strict 1.0 reward for empty predictions and a harsh 0.0 penalty for false merges on singletons.

### 2.2 Solution Strategy
**Approach Type:** Country-Partitioned Multi-Pass Blocking + Feature-Rich Pairwise Classifier + Precision-Calibrated Decision Thresholding  
**Core Innovation:** Multilingual country-agnostic legal suffix stripping & extraction, unicode decomposition, numeric token isolation, and threshold tuning targeting macro F_0.5.

---

## 3. Candidate Generation (Blocking)
*Describe how you reduced the comparison space to a manageable candidate set.*

- **Blocking keys used:** Dynamic country grouping, normalized alphanumeric tokens, postal code / PIN / ZIP matching, token n-gram prefix blocking.
- **Candidate pairs generated:** Controlled budget per entity to maximize recall while maintaining high reduction ratio.
- **How you ensured true matches were not lost:** Multi-index union blocking across multiple complementary keys.

---

## 4. Matching Model

**Features used:**
- Name features: RapidFuzz Levenshtein, token sort ratio, Jaccard similarity, legal suffix match indicator.
- Address features: Street number exact match, postal code match, token overlap, longest common subsequence.
- Other: Source indicators, length differences.

**Model type:** Gradient Boosted Decision Trees (LightGBM / XGBoost) & calibrated logistic scoring.  
**Threshold selection method:** F_0.5 optimization on held-out 20% local validation split.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** Evaluated locally using the exact competition metric harness.
- **Common false positives (wrong merges):** Franchises or branch chains sharing identical names but distinct addresses.
- **Common false negatives (missed matches):** Extreme typos or missing address components.

---

## 6. Conclusion
A robust, scalable pipeline architecture built specifically for noisy, multilingual business entity resolution under strict precision constraints.

---

## Appendix

### A. Code Artefacts
All runnable source code is located under `code/business_entity_resolution/src/`, accompanied by `README.md` and `requirements.txt`.
- Data Loader: `data_loader.py`
- EDA & Diagnostics: `eda.py`
- Metrics & Evaluation Harness: `metrics.py`
- Text Normalization & Feature Extraction: `text_preprocessing.py`
- Pipeline Orchestration: `run_phase1_phase2_tests.py`

### B. Additional Results
Phase 1 & Phase 2 verification complete with 100% test pass rate across data validation, metrics computation, and text preprocessing.
