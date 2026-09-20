"""Run the complete analysis used in the revised manuscript."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
from config import PACKAGE_ROOT

def run(command: list[str]) -> None:
    print("\n$", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=PACKAGE_ROOT)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-source", required=True, type=Path,
        help="CSV file, directory of monthly CSV files, or ZIP archive containing the prepared 5-min monitoring data.",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Reduced computational check only; not intended for manuscript results.",
    )
    args = parser.parse_args()
    py = sys.executable
    quick = ["--quick"] if args.quick else []

    run([py, "code/01_preprocessing.py", "--raw-source", str(args.raw_source)])
    run([py, "code/02_model_development.py", *quick])
    run([py, "code/03_consensus_screening.py"])
    run([py, "code/04_lobo_conformal_bootstrap.py", *quick])
    run([py, "code/05_shap_conditioned_analysis.py", *quick])
    run([py, "code/06_threshold_sensitivity.py"])
    run([py, "code/07_empirical_coverage.py"])
    run([py, "code/08_extraction_historical_bootstrap.py"])
    run([py, "code/09_generate_figures.py"])

    print("\nAnalysis completed. Generated outputs are in data/processed/, models/, and figures/.")

if __name__ == "__main__":
    main()
