# Phase 01: Research Foundation, Schema Detection & Preprocessing Pipeline

## 1. Objective
Establish the foundational data ingestion, schema validation, and zero-leakage preprocessing pipeline for TabSynth AI across three standardized tabular benchmarks:
1. **UCI Adult Income** (32,561 records, 14 continuous/categorical features; binary target: `income`).
2. **UCI Bank Marketing** (45,211 records, 16 features; binary target: `y`).
3. **UCI Default of Credit Card Clients** (30,000 records, 23 features; binary target: `default.payment.next.month`).

## 2. Methodology & Mathematics
- **Automatic Column Classification:** Identifies continuous versus categorical columns using heuristic uniqueness thresholds ($\text{cardinality} \le 20$ or non-numeric dtype $\to$ categorical).
- **Zero-Leakage Policy:** Feature statistics (means, standard deviations, quantile mappings, category vocabularies, imputation medians) are computed strictly on the training partition ($D_{\text{train}}$) and frozen.
- **Continuous Preprocessing:** Standard scaling $z = \frac{x - \mu}{\sigma}$ with fallback quantile transformation mapping empirical continuous marginals to uniform $[0, 1]$ or normal intervals.
- **Categorical Preprocessing:** Ordinal tokenization mapping category levels to discrete indices $\{0, \dots, K-1\}$. Missing values are explicitly represented as reserved tokens (`<missing>`), and unseen evaluation categories are handled via `<unk>`.

## 3. Architecture & Implementation
- **`src/tabsynth/data/schema.py`:** `ColumnMetadata` and `TableSchema` Pydantic models. `infer_table_schema` performs automated type discovery and cardinality extraction.
- **`src/tabsynth/data/preprocessor.py`:** `TabularPreprocessor` providing bidirectional `fit`, `transform`, and `inverse_transform` methods.
- **`src/tabsynth/data/dataset.py`:** `create_stratified_split` (70% Train, 15% Val, 15% Test) and PyTorch `TabularDataset` and DataLoader builders.

## 4. Key Design Decisions
- **Isolation of Target Column $Y$:** The target class label is removed from feature matrix $X$ and preserved exclusively as conditioning vector $c$, preventing self-conditioning and label leakage.
- **Missing Token Invariance:** Reserving `<missing>` token guarantees deterministic discrete state space dimensions for downstream D3PM transition matrices.

## 5. Relevant Source Files
- `src/tabsynth/data/schema.py`
- `src/tabsynth/data/preprocessor.py`
- `src/tabsynth/data/dataset.py`
- `scripts/download_datasets.py`

## 6. Tests Performed
- `tests/unit/test_preprocessor.py`: Tests bidirectional transform and inverse transform, ensuring zero information loss and schema preservation.
- Preprocessing and schema tests pass (100% success rate).

## 7. Actual Results
- Preprocessor successfully fits on 22,792 Adult Income training records.
- Inversion reconstructs original column names, continuous ranges, and categorical string labels without null values.

## 8. Known Limitations
- High-cardinality categorical features (> 100 levels) increase D3PM transition matrix memory footprint.

## 9. Reproduction Instructions
```bash
python scripts/download_datasets.py
pytest tests/unit/test_preprocessor.py -v
```
