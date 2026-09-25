# Business Entity Resolution — Exploratory Data Analysis Report
**Team Name:** GenX H4CK3RS!

## 1. Dataset Dimensions & Volume Overview

| File | Split | Row Count | File Size (MB) |
|---|---|---|---|
| `train_source1.tsv` | train | 2,206,821 | 200.34 MB |
| `train_source2.tsv` | train | 5,034,616 | 466.63 MB |
| `train_source3.tsv` | train | 5,285,603 | 480.37 MB |
| `train_ground_truth.tsv` | train | 2,206,821 | 121.13 MB |
| `test_source1.tsv` | test | 1,732,544 | 166.91 MB |
| `test_source2.tsv` | test | 4,887,273 | 485.86 MB |
| `test_source3.tsv` | test | 5,082,316 | 482.56 MB |

## 2. Country Distribution & France Domain Shift

> **CRITICAL FINDING:** The training set only contains records from `US` and `India`.
> The test set introduces `France` (~15% of records). The model must remain country-agnostic.

| Split / File | Country | Count (Sampled) | Percentage |
|---|---|---|---|
| `train_source1.tsv` (train) | **US** | 59,890 | 59.89% |
| `train_source1.tsv` (train) | **India** | 40,110 | 40.11% |
| `train_source2.tsv` (train) | **US** | 59,936 | 59.94% |
| `train_source2.tsv` (train) | **India** | 40,064 | 40.06% |
| `train_source3.tsv` (train) | **US** | 59,529 | 59.53% |
| `train_source3.tsv` (train) | **India** | 40,471 | 40.47% |
| `test_source1.tsv` (test) | **India** | 46,598 | 46.6% |
| `test_source1.tsv` (test) | **US** | 38,419 | 38.42% |
| `test_source1.tsv` (test) | **France** | 14,983 | 14.98% |
| `test_source2.tsv` (test) | **India** | 47,328 | 47.33% |
| `test_source2.tsv` (test) | **US** | 38,169 | 38.17% |
| `test_source2.tsv` (test) | **France** | 14,503 | 14.5% |
| `test_source3.tsv` (test) | **India** | 47,307 | 47.31% |
| `test_source3.tsv` (test) | **US** | 38,466 | 38.47% |
| `test_source3.tsv` (test) | **France** | 14,227 | 14.23% |

## 3. Ground Truth & Singleton Analysis

- **Total Source 1 Reference Entities:** 2,206,821
- **Singletons (0 matches in S2/S3):** 123,247 (5.585%)
- **Non-Singletons (>= 1 match):** 2,083,574 (94.415%)
- **Average Matches per Non-Singleton:** 3.67

> **F_0.5 Metric Strategic Note:** Singletons must receive an empty prediction list `[]` to score 1.0.
> Any false positive predicted for a singleton results in a punishing score of 0.0.

## 4. Missing Values Inspection

| File | Column | Missing/Empty % |
|---|---|---|
| `train_source1.tsv` | `entity_id` | 0.0% |
| `train_source1.tsv` | `business_name` | 0.0% |
| `train_source1.tsv` | `business_address` | 0.0% |
| `train_source1.tsv` | `country` | 0.0% |
| `train_source2.tsv` | `entity_id` | 0.0% |
| `train_source2.tsv` | `business_name` | 0.0% |
| `train_source2.tsv` | `business_address` | 3.33% |
| `train_source2.tsv` | `country` | 0.0% |
| `train_source3.tsv` | `entity_id` | 0.0% |
| `train_source3.tsv` | `business_name` | 0.0% |
| `train_source3.tsv` | `business_address` | 3.352% |
| `train_source3.tsv` | `country` | 0.0% |
| `test_source1.tsv` | `entity_id` | 0.0% |
| `test_source1.tsv` | `business_name` | 0.0% |
| `test_source1.tsv` | `business_address` | 0.0% |
| `test_source1.tsv` | `country` | 0.0% |
| `test_source2.tsv` | `entity_id` | 0.0% |
| `test_source2.tsv` | `business_name` | 0.0% |
| `test_source2.tsv` | `business_address` | 2.718% |
| `test_source2.tsv` | `country` | 0.0% |
| `test_source3.tsv` | `entity_id` | 0.0% |
| `test_source3.tsv` | `business_name` | 0.0% |
| `test_source3.tsv` | `business_address` | 2.708% |
| `test_source3.tsv` | `country` | 0.0% |