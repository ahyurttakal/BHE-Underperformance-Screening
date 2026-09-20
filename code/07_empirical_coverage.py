"""Observed empirical coverage of the post-selection 90% screening bounds."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from config import PROCESSED_DIR

def choose() -> Path:
    regenerated = PROCESSED_DIR / "06_lobo_bhe_summary_regenerated.csv"
    base = PROCESSED_DIR / "06_lobo_bhe_summary.csv"
    return regenerated if regenerated.exists() else base

def main() -> None:
    df=pd.read_csv(choose())
    n=int(df["test_hours"].sum())
    rows=[]
    for target,prefix in [("thermal_power","q"),("RPI","rpi")]:
        cov=float((df[f"{prefix}_empirical_90PI_coverage_pct"]*df["test_hours"]).sum()/n)
        low=float((df[f"{prefix}_hourly_below_90PI_lower_pct"]*df["test_hours"]).sum()/n)
        high=100.0-cov-low
        rows.append({"target":target,"independent_test_observations":n,"nominal_coverage_pct":90.0,
                     "observed_coverage_pct":cov,"below_lower_pct":low,"above_upper_pct":high})
    out=pd.DataFrame(rows)
    suffix=""
    out.to_csv(PROCESSED_DIR/f"R2_C2_empirical_coverage_global{suffix}.csv",index=False)
    by=df[["bhe","test_hours","q_empirical_90PI_coverage_pct","q_hourly_below_90PI_lower_pct",
           "rpi_empirical_90PI_coverage_pct","rpi_hourly_below_90PI_lower_pct"]].copy()
    by["q_above_upper_pct"]=100-by["q_empirical_90PI_coverage_pct"]-by["q_hourly_below_90PI_lower_pct"]
    by["rpi_above_upper_pct"]=100-by["rpi_empirical_90PI_coverage_pct"]-by["rpi_hourly_below_90PI_lower_pct"]
    by.to_csv(PROCESSED_DIR/f"R2_C2_empirical_coverage_by_bhe{suffix}.csv",index=False)
    print(out.to_string(index=False))
if __name__=="__main__": main()
