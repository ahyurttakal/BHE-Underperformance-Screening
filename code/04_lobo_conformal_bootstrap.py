"""Leave-one-BHE-out validation, conformal intervals and month bootstrap.

For every eligible BHE, the held-out unit is removed from model fitting and
conformal calibration before it is evaluated in the independent test period.
Separate Ridge models are fitted for absolute thermal power and relative thermal
performance. The script also performs a pipe-length ablation for prior or LOBO
screening candidates.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from analysis_utils import (
    bootstrap_median_ci,
    deterministic_sample,
    finite_sample_conformal_quantile,
    regression_metrics,
    validate_columns,
)
from config import (
    INTERMEDIATE_DIR,
    METADATA_DIR,
    PROCESSED_DIR,
    SEEDS,
    ensure_directories,
    load_parameters,
)


def make_ridge_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    alpha: float,
) -> Pipeline:
    """Create the imputed, standardized Ridge pipeline used in LOBO folds."""
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def assign_split(timestamp: pd.Series, parameters: dict) -> pd.Series:
    temporal = parameters["temporal_split"]
    output = pd.Series(index=timestamp.index, dtype="object")
    output.loc[
        timestamp.between(
            pd.Timestamp(temporal["training_start"]),
            pd.Timestamp(temporal["training_end"]),
        )
    ] = "train"
    output.loc[
        timestamp.between(
            pd.Timestamp(temporal["calibration_start"]),
            pd.Timestamp(temporal["calibration_end"]),
        )
    ] = "calibration"
    output.loc[
        timestamp.between(
            pd.Timestamp(temporal["testing_start"]),
            pd.Timestamp(temporal["testing_end"]),
        )
    ] = "test"
    return output


def evaluate_fold(
    data: pd.DataFrame,
    held_out_bhe: int,
    parameters: dict,
    remove_pipe_length: bool,
    quick: bool,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Fit both LOBO targets and return hourly held-out predictions and metrics."""
    categorical = parameters["features"]["categorical"]
    q_numeric = parameters["features"]["thermal_power_numeric"].copy()
    rpi_numeric = parameters["features"]["relative_performance_numeric"].copy()
    if remove_pipe_length:
        q_numeric = [feature for feature in q_numeric if feature != "horizontal_pipe_length_m"]
        rpi_numeric = [feature for feature in rpi_numeric if feature != "horizontal_pipe_length_m"]

    train = data.loc[(data["split"] == "train") & (data["bhe"] != held_out_bhe)].copy()
    calibration = data.loc[
        (data["split"] == "calibration") & (data["bhe"] != held_out_bhe)
    ].copy()
    test = data.loc[(data["split"] == "test") & (data["bhe"] == held_out_bhe)].copy()
    if train.empty or calibration.empty or test.empty:
        raise RuntimeError(f"BHE {held_out_bhe}: one or more LOBO splits are empty.")

    # The revised manuscript uses fixed LOBO sample sizes of 100,000 training
    # and 30,000 calibration observations in every fold. The held-out BHE is
    # removed before deterministic sampling, and the same sampling seeds are
    # reused across folds for direct comparability.
    max_train = parameters["sampling"]["lobo_training_rows"]
    max_calibration = parameters["sampling"]["lobo_calibration_rows"]
    if quick:
        max_train = min(max_train, 6000)
        max_calibration = min(max_calibration, 2500)
    train = deterministic_sample(train, max_train, SEEDS["sampling"])
    calibration = deterministic_sample(
        calibration, max_calibration, SEEDS["sampling"] + 1000
    )

    alpha = float(parameters["ridge"]["lobo_alpha"])
    coverage = float(parameters["conformal"]["coverage"])

    q_features = q_numeric + categorical
    rpi_features = rpi_numeric + categorical
    q_model = make_ridge_pipeline(q_numeric, categorical, alpha)
    rpi_model = make_ridge_pipeline(rpi_numeric, categorical, alpha)
    q_model.fit(train[q_features], train["thermal_power_kw"])
    rpi_model.fit(train[rpi_features], train["raw_relative_performance"])

    q_cal_pred = q_model.predict(calibration[q_features])
    rpi_cal_pred = rpi_model.predict(calibration[rpi_features])
    q_half_width = finite_sample_conformal_quantile(
        calibration["thermal_power_kw"] - q_cal_pred, coverage
    )
    rpi_half_width = finite_sample_conformal_quantile(
        calibration["raw_relative_performance"] - rpi_cal_pred, coverage
    )

    q_pred = q_model.predict(test[q_features])
    rpi_pred = rpi_model.predict(test[rpi_features])
    output_columns = [
        "hour",
        "month",
        "bhe",
        "vault",
        "thermal_power_kw",
        "raw_relative_performance",
        "flow_l_min",
        "delta_t_k",
        "flow_ratio_to_peers",
        "tin_difference_to_peers",
        "peer_q_median_kw",
        "peer_count",
    ]
    output = test[[column for column in output_columns if column in test.columns]].copy()
    output["predicted_q_kw"] = q_pred
    output["predicted_rpi"] = rpi_pred
    output["q_resid_pct"] = 100.0 * (
        output["thermal_power_kw"] - output["predicted_q_kw"]
    ) / output["predicted_q_kw"].replace(0, np.nan)
    output["rpi_resid_pct"] = 100.0 * (
        output["raw_relative_performance"] - output["predicted_rpi"]
    ) / output["predicted_rpi"].replace(0, np.nan)
    output["q_lower"] = output["predicted_q_kw"] - q_half_width
    output["q_upper"] = output["predicted_q_kw"] + q_half_width
    output["rpi_lower"] = output["predicted_rpi"] - rpi_half_width
    output["rpi_upper"] = output["predicted_rpi"] + rpi_half_width
    output["q_below"] = output["thermal_power_kw"] < output["q_lower"]
    output["rpi_below"] = output["raw_relative_performance"] < output["rpi_lower"]
    output["q_covered"] = output["thermal_power_kw"].between(
        output["q_lower"], output["q_upper"]
    )
    output["rpi_covered"] = output["raw_relative_performance"].between(
        output["rpi_lower"], output["rpi_upper"]
    )

    q_metrics = regression_metrics(output["thermal_power_kw"], output["predicted_q_kw"])
    rpi_metrics = regression_metrics(
        output["raw_relative_performance"], output["predicted_rpi"]
    )
    metrics = {
        "q_model_MAE_kW": q_metrics["MAE"],
        "q_model_RMSE_kW": q_metrics["RMSE"],
        "q_model_R2": q_metrics["R2"],
        "rpi_model_MAE": rpi_metrics["MAE"],
        "rpi_model_RMSE": rpi_metrics["RMSE"],
        "rpi_model_R2": rpi_metrics["R2"],
        "q_conformal_half_width_kw": q_half_width,
        "rpi_conformal_absolute_half_width": rpi_half_width,
    }
    return output, metrics


def monthly_summary(hourly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate held-out hourly results to month-level analysis units."""
    hourly = hourly.copy()
    hourly["month"] = pd.to_datetime(hourly["hour"]).dt.to_period("M").astype(str)
    return (
        hourly.groupby(["month", "bhe", "vault"], as_index=False)
        .agg(
            hours=("hour", "size"),
            observed_q_kw=("thermal_power_kw", "median"),
            predicted_q_kw=("predicted_q_kw", "median"),
            q_resid=("q_resid_pct", "median"),
            q_below=("q_below", "mean"),
            q_coverage=("q_covered", "mean"),
            observed_rpi=("raw_relative_performance", "median"),
            predicted_rpi=("predicted_rpi", "median"),
            rpi_resid=("rpi_resid_pct", "median"),
            rpi_below=("rpi_below", "mean"),
            rpi_coverage=("rpi_covered", "mean"),
        )
        .sort_values(["bhe", "month"])
        .reset_index(drop=True)
    )


def summarize_bhe(
    held_out_bhe: int,
    hourly: pd.DataFrame,
    monthly: pd.DataFrame,
    metrics: dict[str, float],
    metadata: pd.DataFrame,
    parameters: dict,
    seed_offset: int,
) -> dict[str, object]:
    """Create one row of the LOBO BHE summary table."""
    screening = parameters["screening"]
    bootstrap = parameters["bootstrap"]
    eligible_monthly = monthly.loc[monthly["hours"] >= screening["minimum_hours_per_eligible_month"]].copy()
    q_low, q_high = bootstrap_median_ci(
        eligible_monthly["q_resid"],
        bootstrap["month_level_iterations"],
        bootstrap["confidence_level"],
        SEEDS["bootstrap"] + seed_offset,
    )
    rpi_low, rpi_high = bootstrap_median_ci(
        eligible_monthly["rpi_resid"],
        bootstrap["month_level_iterations"],
        bootstrap["confidence_level"],
        SEEDS["bootstrap"] + 1000 + seed_offset,
    )
    meta = metadata.set_index("bhe").loc[held_out_bhe]
    delta_t_uncertainty = parameters["measurement_uncertainty"]["delta_t_absolute_k"]
    flow_uncertainty = parameters["measurement_uncertainty"]["flow_relative_fraction"]
    measurement_uncertainty = 100.0 * np.sqrt(
        flow_uncertainty**2
        + (delta_t_uncertainty / hourly["delta_t_k"].abs().clip(lower=1e-6)) ** 2
    )
    q_relative_half_width = 100.0 * metrics["q_conformal_half_width_kw"] / hourly[
        "predicted_q_kw"
    ].abs().replace(0, np.nan)

    return {
        "bhe": held_out_bhe,
        "vault": int(meta["vault"]),
        "pipe_length_m": meta["horizontal_pipe_length_m"],
        "test_hours": int(len(hourly)),
        "eligible_months": int(len(eligible_monthly)),
        "median_raw_rpi": float(hourly["raw_relative_performance"].median()),
        "q_model_MAE_kW": metrics["q_model_MAE_kW"],
        "q_model_RMSE_kW": metrics["q_model_RMSE_kW"],
        "q_model_R2": metrics["q_model_R2"],
        "q_conformal_relative_half_width_pct": float(q_relative_half_width.median()),
        "median_monthly_q_resid_pct": float(eligible_monthly["q_resid"].median()),
        "q_bootstrap_CI_low_pct": q_low,
        "q_bootstrap_CI_high_pct": q_high,
        "q_persistence_below_minus10": float(
            (eligible_monthly["q_resid"] < screening["residual_threshold_pct"]).mean()
        ),
        "q_hourly_below_90PI_lower_pct": float(100.0 * hourly["q_below"].mean()),
        "q_empirical_90PI_coverage_pct": float(100.0 * hourly["q_covered"].mean()),
        "rpi_conformal_absolute_half_width": metrics[
            "rpi_conformal_absolute_half_width"
        ],
        "median_monthly_rpi_resid_pct": float(eligible_monthly["rpi_resid"].median()),
        "rpi_bootstrap_CI_low_pct": rpi_low,
        "rpi_bootstrap_CI_high_pct": rpi_high,
        "rpi_persistence_below_minus10": float(
            (eligible_monthly["rpi_resid"] < screening["residual_threshold_pct"]).mean()
        ),
        "rpi_hourly_below_90PI_lower_pct": float(100.0 * hourly["rpi_below"].mean()),
        "rpi_empirical_90PI_coverage_pct": float(100.0 * hourly["rpi_covered"].mean()),
        "median_measurement_uncertainty_pct": float(measurement_uncertainty.median()),
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hourly-file",
        type=Path,
        default=INTERMEDIATE_DIR / "hourly_features.pkl.gz",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Evaluate only the four main candidate BHEs with fewer bootstrap iterations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    ensure_directories()
    parameters = load_parameters()
    if args.quick:
        parameters["bootstrap"]["month_level_iterations"] = 100

    if not args.hourly_file.exists():
        raise FileNotFoundError(
            f"{args.hourly_file} was not found. Run 01_preprocessing.py first."
        )
    data = pd.read_pickle(args.hourly_file, compression="gzip")
    validate_columns(
        data,
        [
            "hour",
            "bhe",
            "vault",
            "mode",
            "thermal_power_kw",
            "raw_relative_performance",
        ],
        "Hourly feature table",
    )
    data["hour"] = pd.to_datetime(data["hour"])
    data["split"] = assign_split(data["hour"], parameters)
    data = data.loc[data["mode"] == "ground_heat_rejection"].dropna(
        subset=["split"]
    )
    metadata = pd.read_csv(METADATA_DIR / "bhe_metadata.csv")

    # Require the same temporal evidence used in the manuscript: at least six
    # independent-test months with >=24 valid hourly observations per month.
    screening = parameters["screening"]
    min_hours = int(screening["minimum_hours_per_eligible_month"])
    min_months = int(screening["minimum_eligible_months"])
    test_for_eligibility = data.loc[data["split"] == "test", ["hour", "bhe"]].copy()
    test_for_eligibility["month"] = test_for_eligibility["hour"].dt.to_period("M").astype(str)
    month_counts = test_for_eligibility.groupby(["bhe", "month"]).size()
    eligible_month_counts = (month_counts >= min_hours).groupby(level="bhe").sum()
    evaluated_bhes = sorted(
        eligible_month_counts.loc[eligible_month_counts >= min_months].index.astype(int).tolist()
    )
    if args.quick:
        preferred = [2, 24, 36, 38]
        evaluated_bhes = [bhe for bhe in preferred if bhe in evaluated_bhes]

    all_hourly: list[pd.DataFrame] = []
    all_monthly: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for fold_number, bhe in enumerate(evaluated_bhes, start=1):
        print(f"LOBO fold {fold_number}/{len(evaluated_bhes)}: BHE {bhe}")
        hourly, metrics = evaluate_fold(
            data, bhe, parameters, remove_pipe_length=False, quick=args.quick
        )
        monthly = monthly_summary(hourly)
        summary_rows.append(
            summarize_bhe(
                bhe,
                hourly,
                monthly,
                metrics,
                metadata,
                parameters,
                seed_offset=bhe,
            )
        )
        all_hourly.append(hourly)
        all_monthly.append(monthly)

    hourly_table = pd.concat(all_hourly, ignore_index=True)
    monthly_table = pd.concat(all_monthly, ignore_index=True)
    summary = pd.DataFrame(summary_rows)

    prior_path = PROCESSED_DIR / "05_consensus_candidates_regenerated.csv"
    if not prior_path.exists():
        prior_path = PROCESSED_DIR / "05_consensus_candidates.csv"
    prior_candidates: set[int] = set()
    if prior_path.exists():
        prior_candidates = set(pd.read_csv(prior_path)["bhe"].astype(int))

    screening = parameters["screening"]
    summary["prior_model_consensus_candidate"] = summary["bhe"].isin(prior_candidates)
    # Reconstruct the complete Table 4 LOBO-side evidence requirements.
    summary["lobo_robust_candidate"] = (
        (summary["eligible_months"] >= screening["minimum_eligible_months"])
        & (summary["median_monthly_q_resid_pct"] <= screening["residual_threshold_pct"])
        & (summary["median_monthly_rpi_resid_pct"] <= screening["residual_threshold_pct"])
        & (summary["q_persistence_below_minus10"] >= screening["persistence_fraction"])
        & (summary["rpi_persistence_below_minus10"] >= screening["persistence_fraction"])
        & (summary["q_bootstrap_CI_high_pct"] <= screening["conservative_bootstrap_upper_limit_pct"])
        & (summary["rpi_bootstrap_CI_high_pct"] <= screening["conservative_bootstrap_upper_limit_pct"])
        & (summary["q_hourly_below_90PI_lower_pct"] >= screening["empirical_lower_bound_breach_pct"])
    )
    summary["strong_lobo_signal"] = (
        summary["lobo_robust_candidate"]
        & (
            summary["q_bootstrap_CI_high_pct"]
            < screening["strong_bootstrap_upper_limit_pct"]
        )
        & (
            summary["rpi_bootstrap_CI_high_pct"]
            < screening["strong_bootstrap_upper_limit_pct"]
        )
    )
    summary["q_deficit_percentile"] = 100.0 * summary[
        "median_monthly_q_resid_pct"
    ].rank(pct=True, ascending=False)
    summary["rpi_deficit_percentile"] = 100.0 * summary[
        "median_monthly_rpi_resid_pct"
    ].rank(pct=True, ascending=False)

    # Pipe-length ablation for any prior or LOBO candidate.
    ablation_bhes = sorted(
        set(summary.loc[summary["lobo_robust_candidate"], "bhe"].astype(int))
        | prior_candidates
    )
    if args.quick:
        ablation_bhes = [bhe for bhe in ablation_bhes if bhe in evaluated_bhes]
    ablation_rows = []
    for bhe in ablation_bhes:
        print(f"Pipe-length ablation: BHE {bhe}")
        hourly_ablation, _ = evaluate_fold(
            data, bhe, parameters, remove_pipe_length=True, quick=args.quick
        )
        monthly_ablation = monthly_summary(hourly_ablation)
        monthly_ablation = monthly_ablation.loc[monthly_ablation["hours"] >= screening["minimum_hours_per_eligible_month"]]
        ablation_rows.append(
            {
                "bhe": bhe,
                "eligible_months": int(len(monthly_ablation)),
                "no_pipe_median_q_resid_pct": float(
                    monthly_ablation["q_resid"].median()
                ),
                "no_pipe_q_persistence_lt10": float(
                    (monthly_ablation["q_resid"] < -10.0).mean()
                ),
                "no_pipe_median_rpi_resid_pct": float(
                    monthly_ablation["rpi_resid"].median()
                ),
                "no_pipe_rpi_persistence_lt10": float(
                    (monthly_ablation["rpi_resid"] < -10.0).mean()
                ),
            }
        )
    ablation = pd.DataFrame(ablation_rows)
    summary = summary.merge(ablation, on="bhe", how="left")
    summary["pipe_ablation_persistent"] = (
        (summary["no_pipe_median_q_resid_pct"] <= -10.0)
        & (summary["no_pipe_q_persistence_lt10"] >= 0.5)
        & (summary["no_pipe_median_rpi_resid_pct"] <= -10.0)
        & (summary["no_pipe_rpi_persistence_lt10"] >= 0.5)
    ).fillna(False)
    summary["cross_framework_confirmed"] = (
        summary["prior_model_consensus_candidate"]
        & summary["lobo_robust_candidate"]
        & summary["pipe_ablation_persistent"]
    )
    summary["lobo_only_signal"] = (
        summary["lobo_robust_candidate"]
        & ~summary["prior_model_consensus_candidate"]
    )
    summary["prior_candidate_not_lobo_confirmed"] = (
        summary["prior_model_consensus_candidate"]
        & ~summary["cross_framework_confirmed"]
    )
    summary["consensus_median_resid_pct"] = np.nan
    summary["consensus_candidate"] = summary["prior_model_consensus_candidate"]

    summary = summary.sort_values("median_monthly_q_resid_pct").reset_index(drop=True)
    monthly_table.to_csv(PROCESSED_DIR / "07_lobo_monthly_results_regenerated.csv", index=False)
    summary.to_csv(PROCESSED_DIR / "06_lobo_bhe_summary_regenerated.csv", index=False)
    ablation.to_csv(PROCESSED_DIR / "09_pipe_length_ablation_regenerated.csv", index=False)
    summary.loc[summary["lobo_robust_candidate"]].to_csv(
        PROCESSED_DIR / "08_lobo_robust_candidates_regenerated.csv", index=False
    )
    summary.loc[summary["cross_framework_confirmed"]].to_csv(
        PROCESSED_DIR / "10_cross_framework_confirmed_candidates_regenerated.csv",
        index=False,
    )
    hourly_table.to_pickle(
        INTERMEDIATE_DIR / "lobo_hourly_predictions.pkl.gz",
        compression="gzip",
    )
    print(
        "Cross-framework confirmed BHEs:",
        summary.loc[summary["cross_framework_confirmed"], "bhe"].astype(int).tolist(),
    )


if __name__ == "__main__":
    main()
