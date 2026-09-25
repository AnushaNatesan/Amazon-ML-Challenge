# Business Entity Resolution Pipeline

**Team Name:** GenX H4CK3RS!  
**Competition:** Amazon ML Challenge 2026 — Business Entity Resolution  
**Evaluation Metric:** Macro-averaged $F_{0.5}$ with Strict Singleton Enforcement  

---

## 1. Overview & Directory Architecture

This codebase provides an end-to-end Machine Learning pipeline for large-scale Business Entity Resolution. It resolves records across three independent, noisy data sources (`Source 1`, `Source 2`, and `Source 3`) to reference `Source 1` entities under strict precision-weighted constraints ($F_{0.5}$).

The submission package follows the exact required layout:

```text
GenX_H4CK3RS!_submission.zip/
├── output/
│   ├── matching_results.tsv        # Scored leaderboard predictions
│   └── candidate_pairs.tsv         # Final candidate set fed to matcher
├── code/
│   └── business_entity_resolution/
│       ├── src/                    # Source code modules
│       │   ├── __init__.py
│       │   ├── data_loader.py      # TSV loading with tab safety & schema checks
│       │   ├── eda.py              # Exploratory data analysis & reporting
│       │   ├── metrics.py          # Exact macro F_0.5 & local validation split
│       │   ├── text_preprocessing.py # Multilingual & country-agnostic normalization
│       │   └── blocking.py         # Country-partitioned multi-pass candidate blocker
│       ├── run_all_tests.py        # Unified test runner (Phases 1, 2, 3)
│       ├── run_candidate_generation.py # End-to-end blocking runner
│       ├── package_submission.py   # Packager & official validator check utility
│       ├── README.md               # End-to-end reproduction instructions
│       └── requirements.txt        # Pinned Python environment dependencies
└── Documentation_template.md       # Methodology documentation
```

---

## 2. Environment Setup

Python 3.10+ is required. Install all dependencies from `requirements.txt`:

```bash
cd code/business_entity_resolution
pip install -r requirements.txt
```

---

## 3. Implemented Modules (Phases 1, 2, and 3)

### Phase 1: Data Loader & Local Evaluation Harness
1. **`src/data_loader.py`**:
   - Explicitly enforces `sep="\t"` to avoid silent single-column collapse.
   - Enforces `dtype=str` and `keep_default_na=False` to preserve leading zeros in postal codes and building numbers.
   - Validates entity prefixes (`S1-`, `S2-`, `S3-`) and column headers.
   - Provides streaming/chunked readers for multi-million row datasets.
2. **`src/metrics.py`**:
   - Implements the exact macro-averaged $F_{0.5}$ metric:
     $$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
   - **Singleton Handling:** A Source 1 entity with zero true matches scores `1.0` if an empty list is predicted, and `0.0` if any false match is predicted.
   - **Local Validation Split:** Partitions 20% of Source 1 entities and their ground truth, stratified by `country`.
3. **`src/eda.py`**:
   - Inspects file dimensions, line counts, missing fields, country distributions (flagging `France` in test), and singleton rates.

### Phase 2: Multilingual Text Preprocessing & Normalization
1. **`src/text_preprocessing.py`**:
   - **Latin Accent Decomposition:** Uses Unicode NFKD to convert French/European accents (e.g., `é` $\to$ `e`, `ç` $\to$ `c`, `ô` $\to$ `o`) and synthetic noise (e.g., `Prívate` $\to$ `Private`, `CÓNSULTANTS` $\to$ `CONSULTANTS`) without corrupting non-Latin Indic scripts (`हिंदी`).
   - **Country-Agnostic Legal Entity Extraction:** Detects prefixes (`SCI`, `SARL`, `SAS`), suffixes (`Inc`, `Corp`, `Pvt Ltd`, `LLP`, `EURL`), and bracketed forms (`[EURL]`). Normalizes them and produces a clean root name.
   - **Address Expansion:** Expands street/address abbreviations across US (`St` $\to$ `Street`, `Ave` $\to$ `Avenue`), France (`R.` $\to$ `Rue`, `Bd` $\to$ `Boulevard`), and India (`Opp` $\to$ `Opposite`, `H.No` $\to$ `House Number`).
   - **Numeric Extraction:** Isolates postal codes (US 5-digit ZIP, India 6-digit PIN, France 5-digit postal code) and normalizes zero-padded building numbers (`K-00303` $\to$ `K-303`).

### Phase 3: High-Recall Multi-Pass Blocking (Candidate Generation)
1. **`src/blocking.py`**:
   - **Hard Country Partitioning:** Dynamically isolates candidate generation by the `country` string (`US`, `India`, `France`). Cuts comparison space by 60%–85% with zero cross-country recall loss.
   - **Multi-Pass Keys:**
     * *Key A (Core Name Tokens):* Exact clean name, first 2 tokens, individual tokens, sorted tokens.
     * *Key B (Numeric + Name/Address):* Building number + first name token, building number + street token, distinctive address tokens ($\ge 5$ characters, robust to transliterated script / DBA names).
     * *Key C (Postal / PIN + Name/Address):* Postal code + first name token, postal code + address token, exact postal code.
     * *Key D (Sparse N-gram & Typos):* 4-character prefix and character 3-grams for fuzzy spelling variations.
   - **Bucket Size Protection & Ranking:** Discards ultra-frequent stopword buckets ($> 5,000$ records) to maintain high reduction ratio. Ranks candidates by overlapping key count.
   - **Candidate Pruning & Export:** Caps candidates per entity (top 40-50). Strictly exports tab-separated `output/candidate_pairs.tsv`.
   - **Recall Ceiling Performance:** Achieves $\ge 97\%$ empirical recall ceiling on validation ground truth.

---

## 4. End-to-End Pipeline Execution

### Step 1: Run All Verification Tests
Execute the complete test suite across Phases 1, 2, and 3:
```bash
python code/business_entity_resolution/run_all_tests.py
```

### Step 2: Run Exploratory Data Analysis
Generate empirical profiling report:
```bash
python -m src.eda
```

### Step 3: Run Candidate Generation (Blocking)
Run blocking and evaluate validation recall ceiling:
```bash
python code/business_entity_resolution/run_candidate_generation.py --mode val --sample-size 50000 --max-candidates 40
```
Or generate candidates for test records:
```bash
python code/business_entity_resolution/run_candidate_generation.py --mode test --max-candidates 40
```

### Step 4: Submission Validation & Packaging
Package the submission into `GenX_H4CK3RS!_submission.zip` and run the official validator:
```bash
python code/business_entity_resolution/package_submission.py
```
Or run the official validator directly:
```bash
python Dataset/student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir Dataset/student_resource/dataset/test
```
