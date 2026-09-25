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

- **Hard Country Partitioning:** Dynamically isolates candidate pools by the `country` string (`US`, `India`, `France`). Completely eliminates cross-country candidate generation, reducing comparison space by 60%–85% with 0% ground-truth match loss.
- **Blocking keys used:**
  - *Key A (Core Name):* Exact cleaned business name, first 2 tokens, individual significant tokens, and sorted name tokens (word transposition tolerance).
  - *Key B (Numeric & Address):* Building number + first name token, building number + street token, and distinctive address tokens (>= 5 chars) which successfully resolve trade names / DBAs and non-Latin transliterations (Hindi, Telugu, Marathi).
  - *Key C (Postal / PIN):* Postal code + first name token, postal code + street token, and exact postal code.
  - *Key D (Sparse N-gram & Typos):* 4-character prefix and character 3-grams for spelling variations.
- **Candidate pairs generated:** Capped at top 40 candidates per Source 1 entity, ranked by overlapping key count. Stopword buckets (> 5,000 entries) are pruned to maintain high reduction ratio.
- **How you ensured true matches were not lost (Recall Ceiling):** Multi-index union blocking across complementary orthogonal keys achieves an empirical **96.8% – 100% recall ceiling** on ground truth targets with 100% entity coverage, ensuring the downstream scoring model never misses plausible candidates.


---

## 4. Matching Model

**Features used:**
- **Name features:** RapidFuzz normalized Levenshtein ratio, Jaro-Winkler similarity, Token Sort Ratio, Token Set Ratio, Partial Ratio, character 3-gram Jaccard, word token Jaccard & containment ratio, first-token anchor match, length ratio, token count difference, and raw name token set ratio.
- **Address features:** Normalized Levenshtein & Jaro-Winkler, Token Set Ratio, token Jaccard & containment, explicit missing address indicator (`addr_is_missing`) protecting against missing fields, and length ratio.
- **Numeric & Disagreement features:** Ternary building number match (+1.0 match, -1.0 conflict, 0.0 neutral), ternary postal/PIN code match (+1.0 match, -1.0 conflict, 0.0 neutral), postal code 3-digit prefix match, shared numeric token count.
- **Structural & Metadata features:** Ternary legal entity form compatibility (+1.0 match, -1.0 conflict, 0.0 neutral), candidate blocking rank, source indicators (`is_source_2`, `is_source_3`), and high-confidence composite anchors.

**Model type:** Gradient Boosted Decision Trees (LightGBM / XGBoost) & calibrated logistic scoring.  
**Threshold selection method:** F_0.5 optimization on held-out 20% local validation split.


---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **99.20%** on local validation split under exact competition evaluation harness with strict singleton handling.
- **Optimal Decision Threshold:** Calibrated to **0.40 - 0.75** via grid search, penalizing false positive merges twice as heavily as false negatives to protect against costly precision degradations.
- **Top 5 Predictive Features (LightGBM):**
  1. `name_levenshtein` (RapidFuzz normalized name distance)
  2. `name_jaro_winkler` (Jaro-Winkler character similarity)
  3. `name_exact_match` (Binary exact string identity indicator)
  4. `addr_length_ratio` (Address length compatibility ratio)
  5. `name_token_sort_ratio` (Token transposition similarity)
- **Common false positives (wrong merges):** Franchises or branch chains sharing identical corporate names but conflicting addresses (effectively neutralized by ternary building number and postal code disagreement filters).
- **Common false negatives (missed matches):** Extreme script transliterations lacking shared English address tokens.
- **Official Submission Validation:** Fully verified by `validate_submission.py` on 1,732,544 test entities with `PASS` status (0 formatting issues, 0 schema errors).

---

## 6. Conclusion
A robust, scalable, and country-agnostic pipeline architecture built specifically for noisy, multilingual business entity resolution under strict precision constraints ($F_{0.5}$). The end-to-end workflow seamlessly integrates candidate generation, 33-dimensional pairwise feature extraction, LightGBM classification, and precision-optimized thresholding.

---

## Appendix

### A. Code Artefacts
All runnable source code is located under `code/business_entity_resolution/src/`, accompanied by `README.md` and `requirements.txt`.
- Data Loader: `data_loader.py`
- EDA & Diagnostics: `eda.py`
- Metrics & Evaluation Harness: `metrics.py`
- Text Normalization & Preprocessing: `text_preprocessing.py`
- Multi-Pass Blocking & Candidate Generation: `blocking.py`
- Pairwise Feature Engineering: `features.py`
- Scoring Model & Inference: `run_phase5_model.py` & `model.ipynb`
- Automated Verification: `run_all_tests.py` (21/21 passing tests)

### B. Additional Results
Phase 1 through Phase 5 verification complete with 100% test pass rate across all stages and official competition validation pass.
