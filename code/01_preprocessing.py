"""Convert prepared 5-minute BHE monitoring files to analysis-ready hourly data.

The script accepts either a directory of monthly CSV files or a ZIP archive that
contains those files. It applies the manuscript validity rules, aggregates the
5-minute observations to hourly medians, constructs contemporaneous peer
features, and writes the monthly coverage and raw-performance tables.
"""

from __future__ import annotations

import argparse
import io
import math
import re
import zipfile
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import pandas as pd

from analysis_utils import leave_one_out_group_median, validate_columns
from config import (
    INTERMEDIATE_DIR,
    METADATA_DIR,
    PROCESSED_DIR,
    ensure_directories,
    load_parameters,
)


def iter_monthly_sources(raw_source: Path) -> Iterator[Tuple[str, object]]:
    """Yield monthly CSV names and file-like objects from a directory or ZIP."""
    if raw_source.is_dir():
        files = sorted(raw_source.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files were found in {raw_source}")
        for path in files:
            yield path.name, path
        return

    if raw_source.is_file() and raw_source.suffix.lower() == ".zip":
        archive = zipfile.ZipFile(raw_source)
        members = sorted(
            member for member in archive.namelist() if member.lower().endswith(".csv")
        )
        if not members:
            raise FileNotFoundError(f"No CSV files were found in {raw_source}")
        for member in members:
            yield Path(member).name, (archive, member)
        return

    if raw_source.is_file() and raw_source.suffix.lower() == ".csv":
        yield raw_source.name, raw_source
        return

    raise FileNotFoundError(
        "--raw-source must be a CSV file, a directory of CSV files, or a ZIP archive."
    )


def read_monthly_source(source: object) -> pd.DataFrame:
    """Read a monthly source yielded by iter_monthly_sources."""
    if isinstance(source, tuple):
        archive, member = source
        with archive.open(member) as handle:
            return pd.read_csv(io.BytesIO(handle.read()))
    return pd.read_csv(source)


def process_month(
    raw: pd.DataFrame,
    metadata: pd.DataFrame,
    parameters: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply validity rules and aggregate one monthly file to hourly rows."""
    validate_columns(raw, ["Time"], "Raw monthly monitoring file")
    settings = parameters["preprocessing"]
    fluid = parameters["fluid_properties"]

    time = pd.to_datetime(raw["Time"], errors="coerce")
    monthly_frames: list[pd.DataFrame] = []

    for bhe in range(1, 41):
        prefix = f"Probe_{bhe:02d}"
        required = [f"{prefix}_T_in", f"{prefix}_T_out", f"{prefix}_V_dot"]
        if not set(required).issubset(raw.columns):
            continue

        frame = pd.DataFrame(
            {
                "timestamp": time,
                "bhe": bhe,
                "tin_c": pd.to_numeric(raw[required[0]], errors="coerce"),
                "tout_c": pd.to_numeric(raw[required[1]], errors="coerce"),
                "flow_l_min": pd.to_numeric(raw[required[2]], errors="coerce"),
            }
        ).dropna(subset=["timestamp"])

        frame["delta_t_k"] = frame["tin_c"] - frame["tout_c"]
        frame["valid_measurement"] = (
            frame["tin_c"].notna()
            & frame["tout_c"].notna()
            & frame["flow_l_min"].notna()
            & (frame["flow_l_min"] >= settings["minimum_flow_l_min"])
            & (frame["delta_t_k"].abs() >= settings["minimum_absolute_delta_t_k"])
        )
        frame = frame.loc[frame["valid_measurement"]].copy()
        if frame.empty:
            continue

        frame["mode"] = np.where(
            frame["delta_t_k"] > 0,
            "ground_heat_rejection",
            "ground_heat_extraction",
        )
        frame["thermal_power_kw"] = (
            fluid["density_kg_m3"]
            * fluid["specific_heat_j_kg_k"]
            * (frame["flow_l_min"] / 60000.0)
            * frame["delta_t_k"].abs()
            / 1000.0
        )
        frame["hour"] = frame["timestamp"].dt.floor("h")

        grouped = frame.groupby(["bhe", "hour"], sort=False)
        hourly = grouped.agg(
            subobservations=("timestamp", "size"),
            tin_c=("tin_c", "median"),
            tout_c=("tout_c", "median"),
            flow_l_min=("flow_l_min", "median"),
            delta_t_k=("delta_t_k", "median"),
            thermal_power_kw=("thermal_power_kw", "median"),
            rejection_fraction=("mode", lambda series: float((series == "ground_heat_rejection").mean())),
        ).reset_index()
        hourly["mode_sign_consistency"] = np.maximum(
            hourly["rejection_fraction"],
            1.0 - hourly["rejection_fraction"],
        )
        hourly["mode"] = np.where(
            hourly["rejection_fraction"] >= 0.5,
            "ground_heat_rejection",
            "ground_heat_extraction",
        )
        hourly = hourly.loc[
            (hourly["subobservations"] >= settings["minimum_valid_subobservations_per_hour"])
            & (
                hourly["mode_sign_consistency"]
                >= settings["minimum_mode_sign_consistency"]
            )
        ].copy()
        if not hourly.empty:
            monthly_frames.append(hourly)

    if not monthly_frames:
        return pd.DataFrame(), pd.DataFrame()

    all_valid = pd.concat(monthly_frames, ignore_index=True)
    all_valid = all_valid.merge(metadata, on="bhe", how="left", validate="many_to_one")
    all_valid["month"] = all_valid["hour"].dt.to_period("M").astype(str)

    group_columns = ["hour", "vault", "mode"]
    all_valid["peer_count"] = (
        all_valid.groupby(group_columns)["bhe"].transform("size") - 1
    )
    all_valid["peer_q_median_kw"] = leave_one_out_group_median(
        all_valid, group_columns, "thermal_power_kw"
    )
    all_valid["peer_flow_median_l_min"] = leave_one_out_group_median(
        all_valid, group_columns, "flow_l_min"
    )
    all_valid["peer_tin_median_c"] = leave_one_out_group_median(
        all_valid, group_columns, "tin_c"
    )

    eligible = all_valid.loc[
        all_valid["peer_count"] >= settings["minimum_simultaneous_peers"]
    ].copy()
    eligible["raw_relative_performance"] = (
        eligible["thermal_power_kw"] / eligible["peer_q_median_kw"]
    )
    eligible["flow_ratio_to_peers"] = (
        eligible["flow_l_min"] / eligible["peer_flow_median_l_min"]
    )
    eligible["tin_difference_to_peers"] = (
        eligible["tin_c"] - eligible["peer_tin_median_c"]
    )

    month_number = eligible["hour"].dt.month
    hour_number = eligible["hour"].dt.hour
    eligible["month_sin"] = np.sin(2.0 * math.pi * month_number / 12.0)
    eligible["month_cos"] = np.cos(2.0 * math.pi * month_number / 12.0)
    eligible["hour_sin"] = np.sin(2.0 * math.pi * hour_number / 24.0)
    eligible["hour_cos"] = np.cos(2.0 * math.pi * hour_number / 24.0)
    eligible["year_index"] = (
        eligible["hour"] - pd.Timestamp("2018-07-01")
    ).dt.total_seconds() / (365.25 * 24.0 * 3600.0)

    return all_valid, eligible


def build_monthly_tables(
    all_valid: pd.DataFrame,
    eligible: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create the monthly source tables used by the manuscript figures."""
    all_valid_counts = (
        all_valid.groupby(["month", "bhe", "vault"], as_index=False)
        .agg(all_valid_hour_count=("hour", "size"))
    )
    eligible_counts = (
        eligible.groupby(["month", "bhe", "vault"], as_index=False)
        .agg(
            eligible_hour_count=("hour", "size"),
            extraction_hour_count=(
                "mode",
                lambda values: int((values == "ground_heat_extraction").sum()),
            ),
            rejection_hour_count=(
                "mode",
                lambda values: int((values == "ground_heat_rejection").sum()),
            ),
        )
    )
    coverage = all_valid_counts.merge(
        eligible_counts,
        on=["month", "bhe", "vault"],
        how="outer",
    ).fillna(0)
    integer_columns = [
        "eligible_hour_count",
        "all_valid_hour_count",
        "extraction_hour_count",
        "rejection_hour_count",
    ]
    coverage[integer_columns] = coverage[integer_columns].astype(int)
    coverage = coverage.sort_values(["month", "bhe"]).reset_index(drop=True)

    raw_performance = (
        eligible.groupby(["month", "bhe", "vault", "mode"], as_index=False)
        .agg(
            operating_hours=("hour", "size"),
            median_thermal_power_kw=("thermal_power_kw", "median"),
            mean_thermal_power_kw=("thermal_power_kw", "mean"),
            median_flow_l_min=("flow_l_min", "median"),
            median_tin_c=("tin_c", "median"),
            median_raw_relative_performance=("raw_relative_performance", "median"),
            fraction_raw_rpi_below_0_8=(
                "raw_relative_performance",
                lambda values: float((values < 0.8).mean()),
            ),
        )
        .sort_values(["month", "bhe", "mode"])
        .reset_index(drop=True)
    )

    split_labels = pd.Series(index=eligible.index, dtype="object")
    split_labels.loc[eligible["hour"] <= pd.Timestamp("2021-12-31 23:59:59")] = "train_2018_2021"
    split_labels.loc[
        (eligible["hour"] >= pd.Timestamp("2022-01-01"))
        & (eligible["hour"] <= pd.Timestamp("2022-12-31 23:59:59"))
    ] = "calibration_2022"
    split_labels.loc[eligible["hour"] >= pd.Timestamp("2023-01-01")] = "test_2023_2024H1"
    split_summary = (
        eligible.assign(split=split_labels)
        .dropna(subset=["split"])
        .groupby(["split", "mode"], as_index=False)
        .agg(rows=("hour", "size"))
        .sort_values(["split", "mode"])
        .reset_index(drop=True)
    )
    return coverage, raw_performance, split_summary


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-source",
        type=Path,
        required=True,
        help="Directory, CSV file or ZIP archive containing prepared monthly monitoring CSVs.",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=0,
        help="Optional number of monthly files to process; intended only for testing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    ensure_directories()
    parameters = load_parameters()
    metadata = pd.read_csv(METADATA_DIR / "bhe_metadata.csv")

    all_valid_parts: list[pd.DataFrame] = []
    eligible_parts: list[pd.DataFrame] = []

    for index, (name, source) in enumerate(iter_monthly_sources(args.raw_source), start=1):
        if args.limit_files and index > args.limit_files:
            break
        print(f"[{index}] Processing {name}")
        raw = read_monthly_source(source)
        all_valid, eligible = process_month(raw, metadata, parameters)
        if not all_valid.empty:
            all_valid_parts.append(all_valid)
        if not eligible.empty:
            eligible_parts.append(eligible)

    if not eligible_parts:
        raise RuntimeError("No eligible hourly observations were generated.")

    all_valid = pd.concat(all_valid_parts, ignore_index=True)
    eligible = pd.concat(eligible_parts, ignore_index=True)
    eligible = eligible.sort_values(["hour", "bhe"]).reset_index(drop=True)

    coverage, raw_performance, split_summary = build_monthly_tables(all_valid, eligible)

    eligible.to_pickle(
        INTERMEDIATE_DIR / "hourly_features.pkl.gz",
        compression="gzip",
    )
    coverage.to_csv(PROCESSED_DIR / "01_monthly_data_coverage.csv", index=False)
    raw_performance.to_csv(
        PROCESSED_DIR / "02_monthly_raw_thermal_performance.csv", index=False
    )
    split_summary.to_csv(PROCESSED_DIR / "03_temporal_split_summary.csv", index=False)

    print(f"Hourly eligible rows: {len(eligible):,}")
    print(f"Saved {INTERMEDIATE_DIR / 'hourly_features.pkl.gz'}")


if __name__ == "__main__":
    main()
