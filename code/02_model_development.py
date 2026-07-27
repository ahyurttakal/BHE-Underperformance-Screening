"""Develop Ridge and LightGBM expected-performance models.

The script reads the hourly feature table produced by 01_preprocessing.py,
performs the fixed chronological split, selects the Ridge penalty using the
2022 calibration period, fits Ridge and LightGBM models, and stores calibration
and independent-test predictions for both thermal power and relative thermal
performance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from analysis_utils import deterministic_sample, regression_metrics, validate_columns
from config import (
    INTERMEDIATE_DIR,
    MODEL_DIR,
    PROCESSED_DIR,
    SEEDS,
    ensure_directories,
    load_parameters,
)


def assign_split(timestamp: pd.Series, parameters: dict) -> pd.Series:
    """Assign each timestamp to the manuscript's chronological split."""
    split = pd.Series(index=timestamp.index, dtype="object")
    temporal = parameters["temporal_split"]
    split.loc[
        timestamp.between(
            pd.Timestamp(temporal["training_start"]),
            pd.Timestamp(temporal["training_end"]),
        )
    ] = "train_2018_2021"
    split.loc[
        timestamp.between(
            pd.Timestamp(temporal["calibration_start"]),
            pd.Timestamp(temporal["calibration_end"]),
        )
    ] = "calibration_2022"
    split.loc[
        timestamp.between(
            pd.Timestamp(temporal["testing_start"]),
            pd.Timestamp(temporal["testing_end"]),
        )
    ] = "test_2023_2024H1"
    return split


def make_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """Construct the shared imputation, scaling and one-hot preprocessing."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def fit_ridge_with_validation(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    features: list[str],
    categorical_features: list[str],
    target: str,
    alphas: list[float],
) -> tuple[Pipeline, float, dict[str, float]]:
    """Select Ridge alpha on the 2022 calibration period and refit on training data."""
    numeric_features = [feature for feature in features if feature not in categorical_features]
    best_alpha: float | None = None
    best_metrics: dict[str, float] | None = None
    best_model: Pipeline | None = None

    for alpha in alphas:
        model = Pipeline(
            steps=[
                ("preprocessor", make_preprocessor(numeric_features, categorical_features)),
                ("model", Ridge(alpha=float(alpha))),
            ]
        )
        model.fit(train[features], train[target])
        prediction = model.predict(calibration[features])
        metrics = regression_metrics(calibration[target], prediction)
        if best_metrics is None or metrics["MAE"] < best_metrics["MAE"]:
            best_alpha = float(alpha)
            best_metrics = metrics
            best_model = model

    if best_model is None or best_alpha is None or best_metrics is None:
        raise RuntimeError("Ridge model selection failed.")
    return best_model, best_alpha, best_metrics


def make_lightgbm_pipeline(
    features: list[str],
    categorical_features: list[str],
    parameters: dict,
) -> Pipeline:
    """Construct a deterministic LightGBM regression pipeline."""
    numeric_features = [feature for feature in features if feature not in categorical_features]
    model_parameters = parameters["lightgbm"].copy()
    model_parameters["random_state"] = SEEDS["lightgbm"]
    model_parameters["verbosity"] = -1
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(numeric_features, categorical_features)),
            ("model", LGBMRegressor(**model_parameters)),
        ]
    )


def prediction_frame(
    source: pd.DataFrame,
    prediction: np.ndarray,
    model_name: str,
    target_name: str,
    split_name: str,
) -> pd.DataFrame:
    """Create a compact prediction table while preserving analysis covariates."""
    columns = [
        "hour",
        "month",
        "bhe",
        "vault",
        "mode",
        "thermal_power_kw",
        "raw_relative_performance",
        "flow_l_min",
        "delta_t_k",
        "flow_ratio_to_peers",
        "peer_q_median_kw",
        "peer_count",
    ]
    available = [column for column in columns if column in source.columns]
    output = source[available].copy()
    observed_column = (
        "thermal_power_kw" if target_name == "thermal_power" else "raw_relative_performance"
    )
    output["observed"] = source[observed_column].to_numpy(dtype=float)
    output["predicted"] = prediction
    output["model"] = model_name
    output["target"] = target_name
    output["split"] = split_name
    return output


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
        help="Use smaller deterministic samples for a fast smoke test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    ensure_directories()
    parameters = load_parameters()

    if not args.hourly_file.exists():
        raise FileNotFoundError(
            f"{args.hourly_file} was not found. Run 01_preprocessing.py first."
        )
    data = pd.read_pickle(args.hourly_file, compression="gzip")
    validate_columns(
        data,
        [
            "hour",
            "mode",
            "thermal_power_kw",
            "raw_relative_performance",
            "bhe",
            "vault",
        ],
        "Hourly feature table",
    )
    data["hour"] = pd.to_datetime(data["hour"])
    data["split"] = assign_split(data["hour"], parameters)
    data = data.dropna(subset=["split"]).copy()

    maximum_training_rows = parameters["sampling"]["maximum_training_rows"]
    maximum_calibration_rows = parameters["sampling"]["maximum_calibration_rows"]
    if args.quick:
        maximum_training_rows = min(maximum_training_rows, 8000)
        maximum_calibration_rows = min(maximum_calibration_rows, 3000)
        parameters["lightgbm"]["n_estimators"] = 50
        parameters["lightgbm"]["n_jobs"] = 1

    categorical_features = parameters["features"]["categorical"]
    target_specs = {
        "thermal_power": {
            "column": "thermal_power_kw",
            "features": parameters["features"]["thermal_power_numeric"]
            + categorical_features,
        },
        "relative_performance": {
            "column": "raw_relative_performance",
            "features": parameters["features"]["relative_performance_numeric"]
            + categorical_features,
        },
    }

    metrics_rows: list[dict] = []
    predictions: list[pd.DataFrame] = []

    for mode in sorted(data["mode"].dropna().unique()):
        mode_data = data.loc[data["mode"] == mode].copy()
        train_full = mode_data.loc[mode_data["split"] == "train_2018_2021"]
        calibration_full = mode_data.loc[mode_data["split"] == "calibration_2022"]
        test = mode_data.loc[mode_data["split"] == "test_2023_2024H1"]
        if train_full.empty or calibration_full.empty or test.empty:
            print(f"Skipping {mode}: one or more chronological splits are empty.")
            continue

        train = deterministic_sample(
            train_full, maximum_training_rows, SEEDS["sampling"]
        )
        calibration = deterministic_sample(
            calibration_full, maximum_calibration_rows, SEEDS["sampling"] + 1
        )

        for target_name, spec in target_specs.items():
            target_column = spec["column"]
            features = spec["features"]
            validate_columns(mode_data, features + [target_column], f"{mode} modelling data")

            clean_train = train.dropna(subset=[target_column]).copy()
            clean_calibration = calibration.dropna(subset=[target_column]).copy()
            clean_test = test.dropna(subset=[target_column]).copy()

            ridge, selected_alpha, validation_metrics = fit_ridge_with_validation(
                clean_train,
                clean_calibration,
                features,
                categorical_features,
                target_column,
                parameters["ridge"]["candidate_alphas"],
            )
            lightgbm = make_lightgbm_pipeline(features, categorical_features, parameters)
            lightgbm.fit(clean_train[features], clean_train[target_column])

            models = {
                "Ridge": ridge,
                "LightGBM": lightgbm,
            }
            for model_name, model in models.items():
                model_path = MODEL_DIR / f"{mode}__{target_name}__{model_name}.joblib"
                joblib.dump(
                    {
                        "pipeline": model,
                        "features": features,
                        "target_column": target_column,
                        "mode": mode,
                        "target_name": target_name,
                        "selected_ridge_alpha": selected_alpha if model_name == "Ridge" else None,
                    },
                    model_path,
                )

                for split_name, frame in (
                    ("calibration_2022", clean_calibration),
                    ("test_2023_2024H1", clean_test),
                ):
                    predicted = model.predict(frame[features])
                    metrics = regression_metrics(frame[target_column], predicted)
                    metrics_rows.append(
                        {
                            "mode": mode,
                            "target": target_name,
                            "model": model_name,
                            "split": split_name,
                            "selected_ridge_alpha": selected_alpha
                            if model_name == "Ridge"
                            else np.nan,
                            **metrics,
                        }
                    )
                    predictions.append(
                        prediction_frame(
                            frame,
                            predicted,
                            model_name,
                            target_name,
                            split_name,
                        )
                    )

            print(
                f"Completed {mode} / {target_name}; selected Ridge alpha={selected_alpha}; "
                f"validation MAE={validation_metrics['MAE']:.5f}"
            )

    if not predictions:
        raise RuntimeError("No models were fitted. Check the chronological split coverage.")

    metrics_frame = pd.DataFrame(metrics_rows)
    prediction_table = pd.concat(predictions, ignore_index=True)
    metrics_frame.to_csv(PROCESSED_DIR / "12_model_performance_metrics_regenerated.csv", index=False)
    prediction_table.to_pickle(
        INTERMEDIATE_DIR / "model_predictions.pkl.gz",
        compression="gzip",
    )
    print(f"Saved {INTERMEDIATE_DIR / 'model_predictions.pkl.gz'}")


if __name__ == "__main__":
    main()
