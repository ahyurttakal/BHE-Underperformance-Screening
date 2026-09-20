"""
Reviewer 2 - Comment 4
Exploratory heat-extraction analysis using historical rolling-origin residuals
and month-block bootstrap calibration.

Purpose
-------
This analysis is intentionally exploratory and does NOT modify the principal
ground-heat-rejection classification. It addresses the limited 2022 extraction
calibration sample by deriving empirical uncertainty widths from out-of-sample
residuals generated within the 2018-2021 training period.

Workflow
--------
1) Filter to ground_heat_extraction.
2) Generate historical out-of-sample residuals using rolling-origin fits:
   - fit 2018 -> predict 2019
   - fit 2018-2019 -> predict 2020
   - fit 2018-2020 -> predict 2021
3) For each target (thermal power and RPI), calculate a 90th-percentile
   absolute-residual width within each historical calendar month.
4) Use month-level bootstrap resampling (5,000 iterations) of those monthly
   widths; the bootstrap median is used as the exploratory empirical half-width.
5) Fit the final Ridge model on all extraction observations from 2018-2021
   and evaluate on the independent 2023-2024H1 extraction test period.
6) Report global test coverage and BHE-level residual / breach summaries.

Important
---------
- Ridge alpha is fixed at the prespecified LOBO value (default 10.0) so that
  the sparse 2022 extraction period is not used for tuning this exploratory
  analysis.
- No final cross-framework candidate labels are assigned from extraction mode.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import INTERMEDIATE_DIR, PROCESSED_DIR, SEEDS, load_parameters


THERMAL_NUMERIC = [
    "horizontal_pipe_length_m",
    "tin_c",
    "flow_l_min",
    "peer_q_median_kw",
    "peer_flow_median_l_min",
    "peer_tin_median_c",
    "flow_ratio_to_peers",
    "tin_difference_to_peers",
    "peer_count",
    "month_sin",
    "month_cos",
    "hour_sin",
    "hour_cos",
    "year_index",
]

RPI_NUMERIC = [
    "horizontal_pipe_length_m",
    "tin_c",
    "flow_l_min",
    "peer_flow_median_l_min",
    "peer_tin_median_c",
    "flow_ratio_to_peers",
    "tin_difference_to_peers",
    "peer_count",
    "month_sin",
    "month_cos",
    "hour_sin",
    "hour_cos",
    "year_index",
]

CATEGORICAL = ["vault"]


def make_model(numeric_features: list[str], alpha: float) -> Pipeline:
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
            ("categorical", categorical_pipeline, CATEGORICAL),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", Ridge(alpha=float(alpha))),
        ]
    )


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(y, pred)
    rmse = mean_squared_error(y, pred) ** 0.5
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2_score(y, pred)),
    }


def historical_rolling_residuals(
    data: pd.DataFrame,
    target: str,
    numeric_features: list[str],
    alpha: float,
) -> pd.DataFrame:
    """Create strictly historical out-of-sample residuals inside 2018-2021."""
    features = numeric_features + CATEGORICAL
    pieces = []

    for validation_year in (2019, 2020, 2021):
        train = data.loc[data["hour"].dt.year < validation_year].dropna(subset=[target]).copy()
        valid = data.loc[data["hour"].dt.year == validation_year].dropna(subset=[target]).copy()

        if train.empty or valid.empty:
            continue

        model = make_model(numeric_features, alpha)
        model.fit(train[features], train[target])
        pred = model.predict(valid[features])

        out = valid[["hour", "bhe", "vault", target]].copy()
        out["predicted"] = pred
        out["absolute_residual"] = np.abs(out[target].to_numpy(dtype=float) - pred)
        out["validation_year"] = validation_year
        out["calibration_month"] = out["hour"].dt.to_period("M").astype(str)
        pieces.append(out)

    if not pieces:
        raise RuntimeError("No historical rolling-origin residuals could be generated.")

    return pd.concat(pieces, ignore_index=True)


def bootstrap_monthly_q90(
    historical: pd.DataFrame,
    iterations: int,
    seed: int,
) -> tuple[float, float, float, pd.DataFrame]:
    """
    Treat each calendar month as one temporal block.

    First calculate the within-month 90th percentile of absolute residuals.
    Then bootstrap those monthly q90 values. The bootstrap median is the
    exploratory empirical half-width; the 2.5th and 97.5th percentiles
    describe calibration uncertainty.
    """
    monthly = (
        historical.groupby("calibration_month", as_index=False)
        .agg(
            n=("absolute_residual", "size"),
            q90_absolute_residual=("absolute_residual", lambda x: float(np.quantile(x, 0.90))),
        )
        .sort_values("calibration_month")
        .reset_index(drop=True)
    )

    values = monthly["q90_absolute_residual"].to_numpy(dtype=float)
    if len(values) < 2:
        raise RuntimeError("At least two historical calibration months are required.")

    rng = np.random.default_rng(seed)
    bootstrap_stats = np.empty(iterations, dtype=float)

    for i in range(iterations):
        sample = rng.choice(values, size=len(values), replace=True)
        bootstrap_stats[i] = np.median(sample)

    return (
        float(np.median(bootstrap_stats)),
        float(np.quantile(bootstrap_stats, 0.025)),
        float(np.quantile(bootstrap_stats, 0.975)),
        monthly,
    )


def evaluate_target(
    extraction: pd.DataFrame,
    target_name: str,
    target_column: str,
    numeric_features: list[str],
    alpha: float,
    iterations: int,
    seed: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    features = numeric_features + CATEGORICAL

    historical = historical_rolling_residuals(
        extraction.loc[extraction["hour"] < pd.Timestamp("2022-01-01")].copy(),
        target_column,
        numeric_features,
        alpha,
    )

    half_width, half_low, half_high, monthly_calibration = bootstrap_monthly_q90(
        historical,
        iterations=iterations,
        seed=seed,
    )

    train = extraction.loc[
        extraction["hour"].between(
            pd.Timestamp("2018-07-01"),
            pd.Timestamp("2021-12-31 23:59:59"),
        )
    ].dropna(subset=[target_column]).copy()

    test = extraction.loc[
        extraction["hour"].between(
            pd.Timestamp("2023-01-01"),
            pd.Timestamp("2024-06-30 23:59:59"),
        )
    ].dropna(subset=[target_column]).copy()

    if train.empty or test.empty:
        raise RuntimeError(f"{target_name}: training or independent test split is empty.")

    final_model = make_model(numeric_features, alpha)
    final_model.fit(train[features], train[target_column])
    pred = final_model.predict(test[features])

    hourly = test[
        ["hour", "bhe", "vault", target_column]
    ].copy()
    hourly["predicted"] = pred
    hourly["residual_pct"] = 100.0 * (
        hourly[target_column] - hourly["predicted"]
    ) / hourly["predicted"].replace(0, np.nan)

    hourly["lower"] = hourly["predicted"] - half_width
    hourly["upper"] = hourly["predicted"] + half_width
    hourly["below_lower"] = hourly[target_column] < hourly["lower"]
    hourly["above_upper"] = hourly[target_column] > hourly["upper"]
    hourly["covered"] = hourly[target_column].between(hourly["lower"], hourly["upper"])
    hourly["month"] = hourly["hour"].dt.to_period("M").astype(str)

    m = metrics(hourly[target_column].to_numpy(), pred)

    global_row = {
        "target": target_name,
        "ridge_alpha_fixed": alpha,
        "historical_calibration_rows": len(historical),
        "historical_calibration_months": historical["calibration_month"].nunique(),
        "training_rows_2018_2021": len(train),
        "test_rows_2023_2024H1": len(test),
        "bootstrap_iterations": iterations,
        "empirical_half_width": half_width,
        "half_width_bootstrap_95CI_low": half_low,
        "half_width_bootstrap_95CI_high": half_high,
        "test_coverage_pct": 100.0 * hourly["covered"].mean(),
        "test_below_lower_pct": 100.0 * hourly["below_lower"].mean(),
        "test_above_upper_pct": 100.0 * hourly["above_upper"].mean(),
        **m,
    }

    # BHE-level descriptive screening summary.
    rows = []
    for bhe, group in hourly.groupby("bhe"):
        monthly = (
            group.groupby("month", as_index=False)
            .agg(
                hours=("hour", "size"),
                median_residual_pct=("residual_pct", "median"),
                lower_breach_fraction=("below_lower", "mean"),
            )
        )
        eligible = monthly.loc[monthly["hours"] >= 24].copy()

        rows.append(
            {
                "target": target_name,
                "bhe": int(bhe),
                "test_hours": len(group),
                "eligible_test_months": len(eligible),
                "median_hourly_residual_pct": float(group["residual_pct"].median()),
                "lower_bound_breach_pct": float(100.0 * group["below_lower"].mean()),
                "empirical_coverage_pct": float(100.0 * group["covered"].mean()),
                "monthly_persistence_below_minus10_pct": (
                    float(100.0 * (eligible["median_residual_pct"] <= -10.0).mean())
                    if len(eligible)
                    else np.nan
                ),
            }
        )

    bhe_summary = pd.DataFrame(rows).sort_values(
        ["median_hourly_residual_pct", "bhe"]
    ).reset_index(drop=True)

    monthly_calibration.insert(0, "target", target_name)
    return global_row, bhe_summary, monthly_calibration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hourly-file",
        type=Path,
        default=INTERMEDIATE_DIR / "hourly_features.pkl.gz",
        help="Hourly feature table produced by 01_preprocessing.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DIR,
    )
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--bootstrap-iterations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parameters = load_parameters()
    if args.alpha is None:
        args.alpha = float(parameters["ridge"]["lobo_alpha"])
    if args.bootstrap_iterations is None:
        args.bootstrap_iterations = int(parameters["bootstrap"]["month_level_iterations"])
    if args.seed is None:
        args.seed = int(SEEDS["bootstrap"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.hourly_file.exists():
        raise FileNotFoundError(
            f"{args.hourly_file} not found. Provide --hourly-file explicitly."
        )

    data = pd.read_pickle(args.hourly_file, compression="gzip")
    data["hour"] = pd.to_datetime(data["hour"])

    extraction = data.loc[data["mode"] == "ground_heat_extraction"].copy()
    if extraction.empty:
        raise RuntimeError("No ground_heat_extraction observations were found.")

    specs = [
        ("thermal_power", "thermal_power_kw", THERMAL_NUMERIC),
        ("relative_performance", "raw_relative_performance", RPI_NUMERIC),
    ]

    global_rows = []
    bhe_frames = []
    calibration_frames = []

    for index, (name, column, numeric) in enumerate(specs):
        global_row, bhe_summary, calibration = evaluate_target(
            extraction=extraction,
            target_name=name,
            target_column=column,
            numeric_features=numeric,
            alpha=args.alpha,
            iterations=args.bootstrap_iterations,
            seed=args.seed + index,
        )
        global_rows.append(global_row)
        bhe_frames.append(bhe_summary)
        calibration_frames.append(calibration)

    global_table = pd.DataFrame(global_rows)
    by_bhe = pd.concat(bhe_frames, ignore_index=True)
    calibration_table = pd.concat(calibration_frames, ignore_index=True)

    # Wide table for the manuscript's principal rejection-mode BHEs plus BHE 2.
    key = by_bhe.loc[by_bhe["bhe"].isin([2, 24, 36, 38])].copy()
    key = key.sort_values(["bhe", "target"]).reset_index(drop=True)

    global_table.to_csv(
        args.output_dir / "R2_C4_extraction_global_regenerated.csv", index=False
    )
    by_bhe.to_csv(
        args.output_dir / "R2_C4_extraction_by_bhe_regenerated.csv", index=False
    )
    key.to_csv(
        args.output_dir / "R2_C4_extraction_key_bhes_regenerated.csv", index=False
    )
    calibration_table.to_csv(
        args.output_dir / "R2_C4_historical_monthly_calibration_regenerated.csv", index=False
    )

    print("\n=== Exploratory extraction-mode global results ===")
    print(global_table.to_string(index=False))

    print("\n=== BHEs 2, 24, 36 and 38 ===")
    print(key.to_string(index=False))

    print("\nSaved:")
    for filename in (
        "R2_C4_extraction_global_regenerated.csv",
        "R2_C4_extraction_by_bhe_regenerated.csv",
        "R2_C4_extraction_key_bhes_regenerated.csv",
        "R2_C4_historical_monthly_calibration_regenerated.csv",
    ):
        print(" -", args.output_dir / filename)


if __name__ == "__main__":
    main()
