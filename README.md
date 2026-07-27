# BHE Underperformance Screening

Code for the manuscript:

**Uncertainty-Aware Machine Learning for Persistent Underperformance Screening of Borehole Heat Exchanger Circuits**

The workflow performs preprocessing, Ridge and LightGBM modelling, consensus screening, leave-one-BHE-out validation, empirical prediction-bound analysis, block bootstrap, SHAP analysis, conditioned robustness checks, and figure generation.

## Repository structure

```text
code/                   Analysis scripts
data/metadata/          BHE metadata
data/README.md           Raw-data instructions
model_parameters.json   Fixed analysis settings
requirements.txt        Python dependencies
```

Generated tables, models, figures, and logs are created automatically and are not stored in the repository.

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

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Data

The original monitoring dataset is publicly available at:

```text
https://doi.org/10.5281/zenodo.12724484
```

The raw data are not redistributed in this repository.

## Run

Run the complete workflow from the repository root:

```bash
python code/run_all.py --raw-source PATH_TO_PREPARED_DATA
```

For a reduced computational check:

```bash
python code/run_all.py --raw-source PATH_TO_PREPARED_DATA --quick
```

Outputs are written to:

```text
data/processed/
models/
figures/
logs/
```

## Main scripts

```text
01_preprocessing.py
02_model_development.py
03_consensus_screening.py
04_lobo_conformal_bootstrap.py
05_shap_conditioned_analysis.py
06_generate_figures.py
```

## Citation

Please cite the associated manuscript. Citation metadata are provided in `CITATION.cff`.

## License

MIT License.
