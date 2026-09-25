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
│       │   └── text_preprocessing.py # Multilingual & country-agnostic normalization
│       ├── run_phase1_phase2_tests.py # Automated test verification runner
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

## 3. Phase 1 & Phase 2 Modules

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

---

## 4. End-to-End Pipeline Execution

### Step 1: Run Verification Tests
Verify all Phase 1 and Phase 2 modules:
```bash
python code/business_entity_resolution/run_phase1_phase2_tests.py
```

### Step 2: Run Exploratory Data Analysis
Run the automated dataset inspection:
```bash
python -m src.eda
```

### Step 3: Run Text Preprocessing & Local Validation
Use `src/text_preprocessing.py` and `src/metrics.py` within your training/blocking scripts to generate normalized features, evaluate candidate pairs on the 20% validation split, and compute macro $F_{0.5}$.

### Step 4: Submission Validation
Before submitting, validate the output files against official competition rules:
```bash
python Dataset/student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir Dataset/student_resource/dataset/test
```
