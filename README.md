# BHE Underperformance Screening

Code accompanying the manuscript:

**Uncertainty-Aware Machine Learning for Persistent Underperformance Screening of Borehole Heat Exchanger Circuits**

Authors: **Taha Sezer** and **Ahmet Haşim Yurttakal**.

## Overview

The workflow implements the analyses reported in the revised manuscript: preprocessing, Ridge and LightGBM expected-performance modelling, model-consensus screening, leave-one-BHE-out (LOBO) validation, empirical prediction-bound analysis, month-level block bootstrap, SHAP and conditioned robustness analyses, threshold sensitivity, and the exploratory heat-extraction analysis.

The study performs **underperformance screening**; it does not assign confirmed mechanical fault labels.

## Repository structure

```text
code/                       Analysis scripts
data/metadata/              BHE metadata
data/README.md              Public-data information
model_parameters.json       Fixed analysis settings
requirements.txt            Python dependencies
CITATION.cff                 Citation metadata
LICENSE                      MIT License
```

Generated outputs (`data/processed/`, `data/intermediate/`, `models/`, and `figures/`) are created only when the workflow is run and are not stored in the repository.

## Data

The original monitoring dataset is available at:

https://doi.org/10.5281/zenodo.12724484

The raw data are not redistributed here.

## Installation

Python 3.12 is recommended.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

From the repository root:

```bash
python code/run_all.py --raw-source PATH_TO_PREPARED_DATA
```

The input may be a prepared CSV file, a directory of monthly CSV files, or a ZIP archive containing the prepared 5-min monitoring files.

For a reduced computational check:

```bash
python code/run_all.py --raw-source PATH_TO_PREPARED_DATA --quick
```

`--quick` is for testing only and is not intended to reproduce manuscript results.

## Main analysis scripts

```text
01_preprocessing.py
02_model_development.py
03_consensus_screening.py
04_lobo_conformal_bootstrap.py
05_shap_conditioned_analysis.py
06_threshold_sensitivity.py
07_empirical_coverage.py
08_extraction_historical_bootstrap.py
09_generate_figures.py
```

Shared utilities and paths are defined in `analysis_utils.py` and `config.py`.

## Key revised-manuscript settings

The fixed settings are stored in `model_parameters.json`. Important values include:

- training: July 2018-December 2021;
- development/empirical uncertainty calibration: January-December 2022;
- independent test: January 2023-June 2024;
- LOBO sampling: 100,000 training and 30,000 calibration observations per fold after excluding the held-out BHE;
- 90% empirical prediction bounds;
- 5,000 bootstrap iterations;
- minimum six eligible test months, with at least 24 valid hours per eligible month;
- peer-comparable flow ratio: 0.90-1.10;
- approximately +/-20% threshold-sensitivity analysis;
- exploratory extraction calibration from pre-2022 rolling-origin residuals.

## Citation

Please cite the associated manuscript when using this code. Citation metadata are provided in `CITATION.cff`.

## License

MIT License.
