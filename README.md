# SMS Spam Classification

This project studies how text preprocessing affects SMS spam classification. We compare a majority-class baseline (B0), TF-IDF with logistic regression (B1), and three preprocessing variants (P1-P3). Stronger representations and an LLM comparison are planned for later milestones.

The current milestone 2 results use validation data. Spam F1 is the primary metric; precision, recall, macro-F1, accuracy, PR-AUC and average precision are also recorded.

## Setup

Use Python 3.12 and run commands from the repository root. The recorded model experiments used Python 3.12.3. Exact library versions and hardware information are saved in each run's `metadata.json`.

Create and activate a virtual environment:

**Linux / macOS**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies and register the notebook kernel:

```bash
python -m pip install -r requirements.txt
python -m ipykernel install --user --name ppd-proj --display-name "PPD_Proj (.venv)"
python -m jupyter lab
```

Select the **PPD_Proj (.venv)** kernel when opening a notebook.

## Notebooks

| Notebook | Contents |
|---|---|
| [eda.ipynb](eda.ipynb) | Milestone 1: integrity checks, class distributions, duplicate audits, split analysis, EDA and hypotheses. |
| [milestone2_experiments.ipynb](milestone2_experiments.ipynb) | Milestone 2: configurations, B0/B1, the H4 diagnostic, P1-P3, validation tables, confusion matrices and a concluding milestone summary. |

Both notebooks include saved outputs for presentation. The milestone 2 notebook can be run from top to bottom without running EDA first. Its training cells create new experiment artifacts; simply opening it does not rerun experiments.

## Repository layout

| Path | Purpose |
|---|---|
| `data/SMSSpamCollection` | Original labeled SMS data. |
| `data/readme` | Dataset documentation supplied by its authors. |
| `data_splits/` | Canonical frozen train/validation/test CSV files and their manifest. |
| `data/data_splits/` | Historical milestone 1 copies; the experiment loader does not use them. |
| `configs/baselines.json` | Shared TF-IDF, logistic regression and threshold settings. |
| `sms_spam/data.py` | Data loading, validation and manifest verification. |
| `sms_spam/models.py`, `sms_spam/config.py` | Model construction and configuration loading. |
| `sms_spam/preprocessing.py` | Deterministic P1-P3 transformations and their recorded rules. |
| `sms_spam/evaluation.py`, `sms_spam/experiments.py` | Prediction, metrics, experiment execution and artifact saving. |
| `sms_spam/audit.py` | Exact, normalized and near-duplicate audits. |
| `create_splits.py` | Reproduction of the milestone 1 splits with stable record IDs. |
| `reports/` | Saved audits, experiment summaries and individual run artifacts. |
| `tests/` | Data, evaluation, pipeline and preprocessing checks. |

## Data and experiment rules

The dataset is **SMS Spam Collection v.1**, by Tiago A. Almeida and José María Gómez Hidalgo. The local source contains 5,574 messages: 4,827 ham and 747 spam. The [UCI dataset page](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) lists the data license as **CC BY 4.0**; the dataset DOI is [10.24432/C5CC84](https://doi.org/10.24432/C5CC84). Original documentation is retained in `data/readme`.

The source checksum, split seed **42**, class counts and CSV checksums are recorded in `data_splits/manifest.json`. Each `record_id` identifies a source line within this frozen dataset version.

| Dataset variant | Train | Validation | Test | Role |
|---|---:|---:|---:|---|
| `no_duplicates` | 3,619 | 776 | 776 | Main B1 versus preprocessing comparisons; exact text duplicates removed before splitting. |
| `with_duplicates` | 3,901 | 836 | 837 | Separate H4 diagnostic of the naive split with known cross-split duplicate leakage. |

The files and manifest are tracked in Git, so a clone already contains the required splits. There is no need to download data or regenerate splits before running experiments.

- Fit TF-IDF and the classifier only on training data. Each preprocessing variant fits its own vocabulary and IDF with the same vectorizer settings.
- Use validation data for development and comparisons. The experiment runner does not load test CSV files. EDA and integrity audits can inspect all partitions to check the split.
- Keep the test set frozen for a later evaluation after selecting methods and settings on validation data.
- For B1 and P1-P3, keep the same model, hyperparameters and decision threshold of **0.5**. The configured `lbfgs` solver is deterministic; varying its seed does not provide independent runs.
- B0 learns the majority label from training data. B1 uses original text with word unigrams, lowercase normalization, tokens of at least two word characters, no stop-word removal and L2-normalized TF-IDF. All settings are explicit in `configs/baselines.json`.
- P1 masks supported URLs, P2 masks phone/shortcode candidates, and P3 shortens repeated letters. Their exact rules are recorded in each run's `config.json`.
- H4 compares different validation records across the two dataset variants. Its score difference does not isolate the causal effect of duplicates. The leaking split is a diagnostic, not the main generalization estimate.
- Exact deduplication does not remove every normalized or near-duplicate candidate; the existing audit reports document this limitation.

## Reproduce the experiments

Run all cells in `milestone2_experiments.ipynb`, or use the same experiment runner from a terminal:

**B0 and B1 on both dataset variants**

```bash
python -m sms_spam.experiments --datasets no_duplicates with_duplicates --models B0 B1 --output-dir reports/baselines
```

**B1 and P1-P3 on the main split**

```bash
python -m sms_spam.experiments --datasets no_duplicates --models B1 P1 P2 P3 --output-dir reports/preprocessing
```

The default shared configuration is `configs/baselines.json`; a different file can be supplied with `--config`. Changing settings creates a separate experiment and should be documented. The preprocessing notebook additionally writes `delta_f1`, the difference from B1, to its summary.

Each run creates a new directory:

```text
reports/<experiment>/<dataset_variant>/<model>/<run_id>/
    config.json
    metrics.json
    predictions.csv
    metadata.json
```

Predictions contain `record_id`, `y_true`, `y_pred` and `p_spam`. Metadata records dataset hashes, code identity, package versions, hardware, preprocessing change counts and training/inference times. Timings depend on the machine and execution conditions. Runs use no external API.

`summary.csv` is updated for the latest batch; earlier individual run directories remain available. The notebook's final markdown summarizes the recorded milestone snapshot and should be reviewed when results are updated. Dependency constraints allow newer library versions, so consult the recorded package versions when reproducing the saved numerical results.

## Optional data maintenance

To reproduce the frozen splits from the source file:

```bash
python create_splits.py
```

The script checks existing records and their order before writing, and refuses to replace different assignments or a different source dataset version. It writes only to the canonical `data_splits/` directory.

To regenerate the duplicate audit reports:

```bash
python -m sms_spam.audit --variant no_duplicates --output reports/data_audit/no_duplicates.json
python -m sms_spam.audit --variant with_duplicates --output reports/data_audit/with_duplicates.json
```

Similarity matches are candidates for inspection, not automatically confirmed duplicates. The audits do not remove records or modify the splits.

## Verification

```bash
python -m unittest discover -s tests -v
```

The checks cover split integrity, training-only vocabulary/IDF fitting, metrics, preprocessing boundaries, unchanged reference records and explicit test-set isolation in the experiment runner.
