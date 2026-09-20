"""Utility functions used across preprocessing, modelling and uncertainty scripts."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from numba import njit
except ImportError:  # pragma: no cover - a pure-Python fallback is supplied below
    njit = None


def validate_columns(frame: pd.DataFrame, required: Iterable[str], context: str) -> None:
    """Raise a clear error when a required input column is missing."""
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")


def deterministic_sample(frame: pd.DataFrame, maximum_rows: int, seed: int) -> pd.DataFrame:
    """Return all rows or a deterministic random sample when a frame is large."""
    if maximum_rows <= 0 or len(frame) <= maximum_rows:
        return frame.copy()
    return frame.sample(n=maximum_rows, random_state=seed).sort_index()


def regression_metrics(observed: Sequence[float], predicted: Sequence[float]) -> dict[str, float]:
    """Calculate the model metrics reported by the analysis scripts."""
    observed_array = np.asarray(observed, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    mae = mean_absolute_error(observed_array, predicted_array)
    rmse = math.sqrt(mean_squared_error(observed_array, predicted_array))
    scale = np.nanmean(np.abs(observed_array))
    nmae = np.nan if scale == 0 else 100.0 * mae / scale
    return {
        "n": int(observed_array.size),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2_score(observed_array, predicted_array)),
        "NMAE_pct": float(nmae),
    }


def finite_sample_conformal_quantile(residuals: Sequence[float], coverage: float) -> float:
    """Calculate the finite-sample split-conformal absolute-residual quantile."""
    values = np.sort(np.abs(np.asarray(residuals, dtype=float)))
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    rank = int(math.ceil((values.size + 1) * coverage))
    rank = min(max(rank, 1), values.size)
    return float(values[rank - 1])


def bootstrap_median_ci(
    values: Sequence[float],
    iterations: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap the median of independent blocks such as months or days."""
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, clean.size, size=(iterations, clean.size))
    medians = np.median(clean[indices], axis=1)
    alpha = 1.0 - confidence_level
    return (
        float(np.quantile(medians, alpha / 2.0)),
        float(np.quantile(medians, 1.0 - alpha / 2.0)),
    )


def longest_true_run(values: Sequence[bool]) -> int:
    """Return the longest consecutive run of True values."""
    longest = 0
    current = 0
    for value in values:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def month_to_season(month: int) -> str:
    """Map a calendar month to a meteorological season."""
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def sha256sum(path: Path) -> str:
    """Return the SHA-256 checksum of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if njit is not None:

    @njit(cache=True)
    def _loo_median_numba(values: np.ndarray, group_codes: np.ndarray) -> np.ndarray:
        """Exact leave-one-out median for small groups, accelerated with Numba."""
        n_rows = values.shape[0]
        output = np.empty(n_rows, dtype=np.float64)
        output[:] = np.nan
        order = np.argsort(group_codes)
        start = 0
        while start < n_rows:
            group_code = group_codes[order[start]]
            end = start + 1
            while end < n_rows and group_codes[order[end]] == group_code:
                end += 1
            group_size = end - start
            if group_size > 1:
                for local_position in range(group_size):
                    original_index = order[start + local_position]
                    others = np.empty(group_size - 1, dtype=np.float64)
                    write_position = 0
                    for j in range(group_size):
                        if j == local_position:
                            continue
                        value = values[order[start + j]]
                        others[write_position] = value
                        write_position += 1
                    others.sort()
                    m = others.size
                    if m % 2 == 1:
                        output[original_index] = others[m // 2]
                    else:
                        output[original_index] = 0.5 * (
                            others[m // 2 - 1] + others[m // 2]
                        )
            start = end
        return output


def leave_one_out_group_median(
    frame: pd.DataFrame,
    group_columns: list[str],
    value_column: str,
) -> np.ndarray:
    """Calculate an exact peer median excluding the current BHE observation."""
    group_codes, _ = pd.factorize(
        pd.MultiIndex.from_frame(frame[group_columns]),
        sort=False,
    )
    values = frame[value_column].to_numpy(dtype=float)
    if njit is not None:
        return _loo_median_numba(values, group_codes.astype(np.int64))

    output = np.full(len(frame), np.nan, dtype=float)
    for _, indices in frame.groupby(group_columns, sort=False).indices.items():
        index_array = np.asarray(indices, dtype=int)
        group_values = values[index_array]
        for local_position, original_index in enumerate(index_array):
            output[original_index] = np.median(
                np.delete(group_values, local_position)
            )
    return output
