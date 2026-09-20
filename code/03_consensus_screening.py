"""Generate the Ridge-LightGBM model-consensus screening tables.

The analysis is restricted to ground heat rejection, which is the dominant
operating mode in the independent test period. Monthly normalized residuals are
calculated separately for Ridge and LightGBM. A BHE is labelled a consensus
candidate only when both models satisfy the pre-specified deficit and
persistence criteria and the raw relative performance is below 0.90.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_utils import longest_true_run, validate_columns
from config import INTERMEDIATE_DIR, PROCESSED_DIR, ensure_directories, load_parameters


def summarize_model_bhe(frame: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Summarize monthly residual persistence for one expected-performance model."""
    model_data = frame.loc[frame["model"] == model_name].copy()
    model_data["residual_pct"] = (
        100.0 * (model_data["observed"] - model_data["predicted"])
        / model_data["predicted"].replace(0, np.nan)
    )
    model_data["beyond_uncertainty"] = (
        model_data["residual_pct"] < -model_data["measurement_uncertainty_pct"]
    )

    monthly = (
        model_data.groupby(["bhe", "vault", "month"], as_index=False)
        .agg(
            monthly_residual_pct=("residual_pct", "median"),
            hourly_fraction_beyond_uncertainty=("beyond_uncertainty", "mean"),
            hours=("hour", "size"),
            median_raw_rpi=("raw_relative_performance", "median"),
        )
        .sort_values(["bhe", "month"])
    )

    rows = []
    for (bhe, vault), group in monthly.groupby(["bhe", "vault"], sort=True):
        ordered = group.sort_values("month")
        below_10 = ordered["monthly_residual_pct"] < -10.0
        below_20 = ordered["monthly_residual_pct"] < -20.0
        rows.append(
            {
                "bhe": int(bhe),
                "vault": int(vault),
                "eligible_months": int(len(ordered)),
                "hours": int(ordered["hours"].sum()),
                "median_raw_rpi": float(ordered["median_raw_rpi"].median()),
                f"median_monthly_resid_pct_{model_name}": float(
                    ordered["monthly_residual_pct"].median()
                ),
                f"persistence_lt10_{model_name}": float(below_10.mean()),
                f"persistence_lt20_{model_name}": float(below_20.mean()),
                f"median_hourly_fraction_beyond_uncertainty_{model_name}": float(
                    ordered["hourly_fraction_beyond_uncertainty"].median()
                ),
                f"longest_run_lt10_{model_name}": int(longest_true_run(below_10)),
            }
        )
    return pd.DataFrame(rows)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction-file",
        type=Path,
        default=INTERMEDIATE_DIR / "model_predictions.pkl.gz",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    ensure_directories()
    parameters = load_parameters()

    if not args.prediction_file.exists():
        raise FileNotFoundError(
            f"{args.prediction_file} was not found. Run 02_model_development.py first."
        )
    predictions = pd.read_pickle(args.prediction_file, compression="gzip")
    validate_columns(
        predictions,
        [
            "hour",
            "month",
            "bhe",
            "vault",
            "mode",
            "model",
            "target",
            "split",
            "observed",
            "predicted",
            "raw_relative_performance",
            "delta_t_k",
        ],
        "Model prediction table",
    )

    test = predictions.loc[
        (predictions["mode"] == "ground_heat_rejection")
        & (predictions["target"] == "thermal_power")
        & (predictions["split"] == "test_2023_2024H1")
    ].copy()
    if test.empty:
        raise RuntimeError("No ground-heat-rejection test predictions were found.")

    uncertainty = parameters["measurement_uncertainty"]
    test["measurement_uncertainty_pct"] = 100.0 * np.sqrt(
        uncertainty["flow_relative_fraction"] ** 2
        + (
            uncertainty["delta_t_absolute_k"]
            / test["delta_t_k"].abs().clip(lower=1e-6)
        )
        ** 2
    )

    ridge = summarize_model_bhe(test, "Ridge")
    lightgbm = summarize_model_bhe(test, "LightGBM")
    ranking = ridge.merge(
        lightgbm,
        on=["bhe", "vault", "eligible_months", "hours", "median_raw_rpi"],
        how="outer",
    )

    uncertainty_by_bhe = (
        test.groupby("bhe", as_index=False)
        .agg(median_measurement_uncertainty_pct=("measurement_uncertainty_pct", "median"))
    )
    ranking = ranking.merge(uncertainty_by_bhe, on="bhe", how="left")
    ranking["consensus_median_resid_pct"] = 0.5 * (
        ranking["median_monthly_resid_pct_Ridge"]
        + ranking["median_monthly_resid_pct_LightGBM"]
    )
    ranking["model_disagreement_pct"] = (
        ranking["median_monthly_resid_pct_Ridge"]
        - ranking["median_monthly_resid_pct_LightGBM"]
    ).abs()

    screening = parameters["screening"]
    ranking["consensus_candidate"] = (
        (ranking["eligible_months"] >= screening["minimum_eligible_months"])
        & (ranking["median_raw_rpi"] < screening["raw_relative_performance_threshold"])
        & (
            ranking["median_monthly_resid_pct_Ridge"]
            <= screening["residual_threshold_pct"]
        )
        & (
            ranking["median_monthly_resid_pct_LightGBM"]
            <= screening["residual_threshold_pct"]
        )
        & (
            ranking["persistence_lt10_Ridge"]
            >= screening["persistence_fraction"]
        )
        & (
            ranking["persistence_lt10_LightGBM"]
            >= screening["persistence_fraction"]
        )
    )
    ranking["high_confidence_candidate"] = (
        ranking["consensus_candidate"]
        & (
            ranking["median_monthly_resid_pct_Ridge"]
            <= screening["strong_residual_threshold_pct"]
        )
        & (
            ranking["median_monthly_resid_pct_LightGBM"]
            <= screening["strong_residual_threshold_pct"]
        )
        & (ranking["median_hourly_fraction_beyond_uncertainty_Ridge"] >= 0.5)
        & (ranking["median_hourly_fraction_beyond_uncertainty_LightGBM"] >= 0.5)
    )

    ordered_columns = [
        "bhe",
        "vault",
        "eligible_months",
        "hours",
        "median_raw_rpi",
        "median_measurement_uncertainty_pct",
        "median_monthly_resid_pct_Ridge",
        "persistence_lt10_Ridge",
        "persistence_lt20_Ridge",
        "median_hourly_fraction_beyond_uncertainty_Ridge",
        "longest_run_lt10_Ridge",
        "median_monthly_resid_pct_LightGBM",
        "persistence_lt10_LightGBM",
        "persistence_lt20_LightGBM",
        "median_hourly_fraction_beyond_uncertainty_LightGBM",
        "longest_run_lt10_LightGBM",
        "consensus_median_resid_pct",
        "model_disagreement_pct",
        "consensus_candidate",
        "high_confidence_candidate",
    ]
    ranking = ranking[ordered_columns].sort_values(
        "consensus_median_resid_pct"
    ).reset_index(drop=True)
    candidates = ranking.loc[ranking["consensus_candidate"]].copy()

    ranking.to_csv(PROCESSED_DIR / "04_consensus_bhe_ranking_regenerated.csv", index=False)
    candidates.to_csv(PROCESSED_DIR / "05_consensus_candidates_regenerated.csv", index=False)
    print(f"Consensus candidates: {candidates['bhe'].astype(int).tolist()}")


if __name__ == "__main__":
    main()
