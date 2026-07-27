"""Run the complete BHE analysis workflow."""

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
        "--raw-source",
        required=True,
        type=Path,
        help="Directory, CSV file, or ZIP archive containing the prepared 5-min data.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use smaller samples and fewer bootstrap iterations for a quick check.",
    )
    args = parser.parse_args()

    python = sys.executable
    quick = ["--quick"] if args.quick else []

    run([python, "code/01_preprocessing.py", "--raw-source", str(args.raw_source)])
    run([python, "code/02_model_development.py", *quick])
    run([python, "code/03_consensus_screening.py"])
    run([python, "code/04_lobo_conformal_bootstrap.py", *quick])
    run([python, "code/05_shap_conditioned_analysis.py", *quick])
    run([python, "code/06_generate_figures.py"])

    print("\nAnalysis completed. Tables are in data/processed and figures are in figures/.")


if __name__ == "__main__":
    main()
