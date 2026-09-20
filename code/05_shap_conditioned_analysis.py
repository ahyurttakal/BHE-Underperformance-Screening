"""Create SHAP, seasonal, load-conditioned and paired-control outputs.

This script explains the independent-test LightGBM models and then uses the
LOBO hourly predictions to test whether candidate deficits persist across
seasons, hydraulic conditions and contemporaneous field-load regimes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from analysis_utils import bootstrap_median_ci, month_to_season, validate_columns
from config import (
    INTERMEDIATE_DIR,
    METADATA_DIR,
    MODEL_DIR,
    PROCESSED_DIR,
    SEEDS,
    ensure_directories,
    load_parameters,
)


def clean_feature_name(name: str) -> str:
    """Convert ColumnTransformer names to manuscript-facing feature names."""
    clean = name.split("__", 1)[-1]
    if clean.startswith("vault_"):
        return "vault"
    return clean


def aggregate_shap_importance(
    transformed_feature_names: list[str],
    shap_values: np.ndarray,
    value_name: str,
) -> pd.DataFrame:
    """Aggregate one-hot features and calculate mean absolute SHAP importance."""
    long = pd.DataFrame(
        {
            "feature": [clean_feature_name(name) for name in transformed_feature_names],
            "mean_abs": np.mean(np.abs(shap_values), axis=0),
        }
    )
    output = long.groupby("feature", as_index=False)["mean_abs"].sum()
    output = output.sort_values("mean_abs", ascending=False).reset_index(drop=True)
    output["importance_share_pct"] = 100.0 * output["mean_abs"] / output[
        "mean_abs"
    ].sum()
    return output.rename(columns={"mean_abs": value_name})


def calculate_shap_tables(
    hourly: pd.DataFrame,
    parameters: dict,
    quick: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Explain saved LightGBM thermal-power and relative-performance models."""
    candidate_control_bhes = [2, 4, 15, 24, 34, 36, 38]
    importance_tables: dict[str, pd.DataFrame] = {}
    signed_profiles: list[pd.DataFrame] = []

    for target_name, value_name in (
        ("thermal_power", "mean_abs_SHAP_kW"),
        ("relative_performance", "mean_abs_SHAP_RPI"),
    ):
        model_path = (
            MODEL_DIR
            / f"ground_heat_rejection__{target_name}__LightGBM.joblib"
        )
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} was not found. Run 02_model_development.py first."
            )
        bundle = joblib.load(model_path)
        pipeline = bundle["pipeline"]
        features = bundle["features"]

        test = hourly.loc[
            (hourly["hour"] >= pd.Timestamp(parameters["temporal_split"]["testing_start"]))
            & (hourly["hour"] <= pd.Timestamp(parameters["temporal_split"]["testing_end"]))
            & (hourly["mode"] == "ground_heat_rejection")
        ].copy()
        maximum = parameters["sampling"]["maximum_shap_rows"]
        if quick:
            maximum = min(maximum, 1200)
        if len(test) > maximum:
            sample = test.sample(maximum, random_state=SEEDS["shap"])
        else:
            sample = test

        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]
        transformed = preprocessor.transform(sample[features])
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        transformed_names = list(preprocessor.get_feature_names_out())
        explainer = shap.TreeExplainer(model)
        shap_values = np.asarray(explainer.shap_values(transformed))
        if shap_values.ndim == 3:
            shap_values = shap_values[0]
        importance_tables[target_name] = aggregate_shap_importance(
            transformed_names, shap_values, value_name
        )

        if target_name == "relative_performance":
            signed = pd.DataFrame(shap_values, columns=transformed_names)
            signed["bhe"] = sample["bhe"].to_numpy()
            melted = signed.melt(
                id_vars="bhe",
                var_name="transformed_feature",
                value_name="signed_shap",
            )
            melted["feature"] = melted["transformed_feature"].map(
                clean_feature_name
            )
            signed_profiles.append(
                melted.loc[melted["bhe"].isin(candidate_control_bhes)]
                .groupby(["bhe", "feature"], as_index=False)
                .agg(mean_signed_SHAP_RPI=("signed_shap", "mean"))
            )

    return (
        importance_tables["thermal_power"],
        importance_tables["relative_performance"],
        pd.concat(signed_profiles, ignore_index=True),
    )


def day_block_interval(values: pd.Series, days: pd.Series, parameters: dict, seed: int) -> tuple[float, float]:
    """Resample complete calendar-day blocks and return a median confidence interval."""
    daily = pd.DataFrame({"value": values, "day": days}).groupby("day")[
        "value"
    ].median()
    return bootstrap_median_ci(
        daily.to_numpy(),
        parameters["bootstrap"]["day_block_iterations"],
        parameters["bootstrap"]["confidence_level"],
        seed,
    )


def conditioned_outputs(
    lobo: pd.DataFrame,
    metadata: pd.DataFrame,
    parameters: dict,
) -> dict[str, pd.DataFrame]:
    """Generate candidate/control operating, persistence and paired tables."""
    roles = metadata.set_index("bhe")["screening_role"].to_dict()
    selected_bhes = [2, 4, 15, 24, 34, 36, 38]
    selected = lobo.loc[lobo["bhe"].isin(selected_bhes)].copy()
    selected["role"] = selected["bhe"].map(roles).str.replace("_", " ")
    selected["season"] = selected["hour"].dt.month.map(month_to_season)

    operating = (
        selected.groupby(["bhe", "role", "vault"], as_index=False)
        .agg(
            hours=("hour", "size"),
            median_observed_q_kw=("thermal_power_kw", "median"),
            median_predicted_q_kw=("predicted_q_kw", "median"),
            median_normalized_residual_pct=("q_resid_pct", "median"),
            median_raw_relative_performance=("raw_relative_performance", "median"),
            median_flow_ratio_to_peers=("flow_ratio_to_peers", "median"),
        )
    )
    pipe = metadata[["bhe", "horizontal_pipe_length_m"]].rename(
        columns={"horizontal_pipe_length_m": "pipe_length_m"}
    )
    operating = operating.merge(pipe, on="bhe", how="left")
    operating["median_tin_difference_to_peers_K"] = (
        selected.groupby("bhe")["tin_difference_to_peers"].median().reindex(
            operating["bhe"]
        ).to_numpy()
        if "tin_difference_to_peers" in selected.columns
        else np.nan
    )
    operating = operating[
        [
            "bhe",
            "role",
            "vault",
            "pipe_length_m",
            "hours",
            "median_observed_q_kw",
            "median_predicted_q_kw",
            "median_normalized_residual_pct",
            "median_raw_relative_performance",
            "median_flow_ratio_to_peers",
            "median_tin_difference_to_peers_K",
        ]
    ]

    seasonal = (
        selected.groupby(["bhe", "season"], as_index=False)
        .agg(
            hours=("hour", "size"),
            median_LOBO_residual_pct=("q_resid_pct", "median"),
            median_raw_relative_performance=("raw_relative_performance", "median"),
            median_flow_ratio=("flow_ratio_to_peers", "median"),
        )
        .sort_values(["bhe", "season"])
    )

    selected["flow_regime"] = pd.cut(
        selected["flow_ratio_to_peers"],
        bins=[-np.inf, parameters["conditioned_analysis"]["flow_ratio_peer_comparable_lower"], parameters["conditioned_analysis"]["flow_ratio_peer_comparable_upper"], np.inf],
        labels=["Below-peer flow", "Peer-comparable flow", "Above-peer flow"],
    )
    flow_conditioned = (
        selected.groupby(["bhe", "flow_regime"], observed=True, as_index=False)
        .agg(
            hours=("hour", "size"),
            median_LOBO_residual_pct=("q_resid_pct", "median"),
            median_raw_relative_performance=("raw_relative_performance", "median"),
        )
        .sort_values(["bhe", "flow_regime"])
    )

    load_source = selected["peer_q_median_kw"].dropna()
    low_q, high_q = parameters["conditioned_analysis"]["load_quantiles"]
    low_threshold, high_threshold = load_source.quantile([low_q, high_q]).tolist()
    selected["peer_load_regime"] = pd.cut(
        selected["peer_q_median_kw"],
        bins=[-np.inf, low_threshold, high_threshold, np.inf],
        labels=["Low field load", "Medium field load", "High field load"],
    )
    field_load = (
        selected.groupby(["bhe", "peer_load_regime"], observed=True, as_index=False)
        .agg(
            hours=("hour", "size"),
            median_LOBO_residual_pct=("q_resid_pct", "median"),
            median_raw_relative_performance=("raw_relative_performance", "median"),
        )
        .sort_values(["bhe", "peer_load_regime"])
    )

    paired_rows = []
    matched_controls = {
        int(candidate): int(control)
        for candidate, control in parameters["matched_controls"].items()
    }
    for candidate, control in matched_controls.items():
        candidate_frame = selected.loc[selected["bhe"] == candidate].copy()
        control_frame = selected.loc[selected["bhe"] == control].copy()
        paired = candidate_frame.merge(
            control_frame,
            on="hour",
            suffixes=("_candidate", "_control"),
            how="inner",
        )
        if paired.empty:
            continue
        paired["residual_difference"] = (
            paired["q_resid_pct_candidate"] - paired["q_resid_pct_control"]
        )
        paired["rpi_difference"] = (
            paired["raw_relative_performance_candidate"]
            - paired["raw_relative_performance_control"]
        )
        paired["day"] = paired["hour"].dt.floor("D")
        q_low, q_high = day_block_interval(
            paired["residual_difference"],
            paired["day"],
            parameters,
            SEEDS["bootstrap"] + candidate,
        )
        rpi_low, rpi_high = day_block_interval(
            paired["rpi_difference"],
            paired["day"],
            parameters,
            SEEDS["bootstrap"] + 1000 + candidate,
        )
        paired_rows.append(
            {
                "candidate_bhe": candidate,
                "matched_control_bhe": control,
                "paired_hours": int(len(paired)),
                "paired_days": int(paired["day"].nunique()),
                "median_candidate_minus_control_residual_pct": float(
                    paired["residual_difference"].median()
                ),
                "day_block_bootstrap_CI_low_pct": q_low,
                "day_block_bootstrap_CI_high_pct": q_high,
                "median_candidate_minus_control_RPI": float(
                    paired["rpi_difference"].median()
                ),
                "RPI_day_block_CI_low": rpi_low,
                "RPI_day_block_CI_high": rpi_high,
                "median_candidate_flow_ratio": float(
                    paired["flow_ratio_to_peers_candidate"].median()
                ),
                "median_control_flow_ratio": float(
                    paired["flow_ratio_to_peers_control"].median()
                ),
            }
        )
    paired_results = pd.DataFrame(paired_rows)

    expected_rpi = (
        selected.groupby(["bhe", "role"], as_index=False)
        .agg(
            observed_median_RPI=("raw_relative_performance", "median"),
            model_expected_median_RPI=("predicted_rpi", "median"),
        )
    )
    expected_rpi["observed_minus_expected_RPI"] = (
        expected_rpi["observed_median_RPI"]
        - expected_rpi["model_expected_median_RPI"]
    )

    return {
        "operating": operating,
        "seasonal": seasonal,
        "flow": flow_conditioned,
        "field_load": field_load,
        "paired": paired_results,
        "expected_rpi": expected_rpi,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hourly-file",
        type=Path,
        default=INTERMEDIATE_DIR / "hourly_features.pkl.gz",
    )
    parser.add_argument(
        "--lobo-file",
        type=Path,
        default=INTERMEDIATE_DIR / "lobo_hourly_predictions.pkl.gz",
    )
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    ensure_directories()
    parameters = load_parameters()
    if args.quick:
        parameters["bootstrap"]["day_block_iterations"] = 100

    if not args.hourly_file.exists() or not args.lobo_file.exists():
        raise FileNotFoundError(
            "Hourly feature or LOBO prediction data are missing. Run scripts 01-04 first."
        )
    hourly = pd.read_pickle(args.hourly_file, compression="gzip")
    hourly["hour"] = pd.to_datetime(hourly["hour"])
    lobo = pd.read_pickle(args.lobo_file, compression="gzip")
    lobo["hour"] = pd.to_datetime(lobo["hour"])
    metadata = pd.read_csv(METADATA_DIR / "bhe_metadata.csv")

    q_importance, rpi_importance, signed_profiles = calculate_shap_tables(
        hourly, parameters, args.quick
    )
    outputs = conditioned_outputs(lobo, metadata, parameters)

    q_importance.to_csv(PROCESSED_DIR / "18_global_SHAP_importance_regenerated.csv", index=False)
    rpi_importance.to_csv(
        PROCESSED_DIR / "19_relative_performance_SHAP_importance_regenerated.csv",
        index=False,
    )
    signed_profiles.to_csv(
        PROCESSED_DIR / "21_candidate_control_signed_SHAP_profiles_regenerated.csv",
        index=False,
    )
    outputs["operating"].to_csv(
        PROCESSED_DIR / "13_candidate_control_operating_summary_regenerated.csv",
        index=False,
    )
    outputs["seasonal"].to_csv(
        PROCESSED_DIR / "14_seasonal_conditioned_results_regenerated.csv", index=False
    )
    outputs["flow"].to_csv(
        PROCESSED_DIR / "15_flow_regime_conditioned_results_regenerated.csv", index=False
    )
    outputs["field_load"].to_csv(
        PROCESSED_DIR / "16_field_load_conditioned_results_regenerated.csv", index=False
    )
    outputs["paired"].to_csv(
        PROCESSED_DIR / "17_paired_candidate_control_results_regenerated.csv", index=False
    )
    outputs["expected_rpi"].to_csv(
        PROCESSED_DIR / "20_candidate_control_expected_RPI_profiles_regenerated.csv",
        index=False,
    )
    print("SHAP and conditioned-analysis tables were saved successfully.")


if __name__ == "__main__":
    main()
