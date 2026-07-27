"""Regenerate the ten manuscript figures from the distributed source tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle

from config import FIGURE_DIR, METADATA_DIR, PROCESSED_DIR, ensure_directories

COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "magenta": "#CC79A7",
    "redorange": "#D55E00",
    "grey": "#B8B8B8",
    "dark": "#303030",
    "threshold": "#7F7F7F",
    "grid": "#E5E5E5",
    "insufficient": "#D9D9D9",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 12.5,
        "axes.labelsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)



def read_processed(filename: str) -> pd.DataFrame:
    """Read a regenerated table when available, otherwise use the manuscript table."""
    path = PROCESSED_DIR / filename
    regenerated = path.with_name(f"{path.stem}_regenerated{path.suffix}")
    selected = regenerated if regenerated.exists() else path
    return pd.read_csv(selected)

def save_figure(fig: plt.Figure, stem: str) -> None:
    """Save publication-quality PNG and vector PDF versions."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def figure_01_field_layout(metadata: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    vault_colors = {1: COLORS["blue"], 2: COLORS["orange"], 3: COLORS["green"]}
    for vault, group in metadata.groupby("vault"):
        ax.scatter(
            group["field_x_m"], group["field_y_m"], s=70,
            color=vault_colors[int(vault)], edgecolor="black", linewidth=0.6, zorder=2,
        )
    controls = metadata.loc[metadata["screening_role"] == "matched_control"]
    ax.scatter(
        controls["field_x_m"], controls["field_y_m"], s=155, marker="s",
        facecolor="none", edgecolor=COLORS["dark"], linewidth=1.7, zorder=4,
    )
    borderline = metadata.loc[metadata["screening_role"] == "borderline_candidate"]
    ax.scatter(
        borderline["field_x_m"], borderline["field_y_m"], s=160, marker="D",
        color=COLORS["magenta"], edgecolor="black", linewidth=0.8, zorder=5,
    )
    confirmed = metadata.loc[metadata["screening_role"] == "confirmed_candidate"]
    ax.scatter(
        confirmed["field_x_m"], confirmed["field_y_m"], s=285, marker="*",
        color=COLORS["redorange"], edgecolor="black", linewidth=0.9, zorder=6,
    )
    for row in metadata.itertuples(index=False):
        ax.text(
            row.field_x_m + 0.9, row.field_y_m + 0.8, str(int(row.bhe)),
            fontsize=11.5, color="#202020", ha="left", va="bottom", zorder=7,
        )
    ax.set_xlabel("Field x-coordinate (m)")
    ax.set_ylabel("Field y-coordinate (m)")
    ax.set_xlim(0, 122)
    ax.set_ylim(3, 55)
    ax.set_aspect("equal", adjustable="box")
    ax.spines[["top", "right"]].set_visible(False)
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=COLORS["blue"], markeredgecolor="black", markersize=8, label="Vault 1"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=COLORS["orange"], markeredgecolor="black", markersize=8, label="Vault 2"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=COLORS["green"], markeredgecolor="black", markersize=8, label="Vault 3"),
        Line2D([0], [0], marker="s", linestyle="none", markerfacecolor="none", markeredgecolor=COLORS["dark"], markersize=10, label="Matched control"),
        Line2D([0], [0], marker="D", linestyle="none", markerfacecolor=COLORS["magenta"], markeredgecolor="black", markersize=10, label="Borderline candidate"),
        Line2D([0], [0], marker="*", linestyle="none", markerfacecolor=COLORS["redorange"], markeredgecolor="black", markersize=14, label="Confirmed candidate"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False, fontsize=11.5)
    fig.tight_layout()
    save_figure(fig, "Fig01_field_layout")


def _workflow_box(ax, x, y, w, h, text, facecolor, fontsize=11.5) -> None:
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.08",
        linewidth=1.2, edgecolor="#4A4A4A", facecolor=facecolor, zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, linespacing=1.02, zorder=3)


def _workflow_arrow(ax, start, end) -> None:
    ax.add_patch(
        FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.4, color="#555555", shrinkA=2, shrinkB=2, zorder=1)
    )


def figure_02_workflow() -> None:
    fig, ax = plt.subplots(figsize=(15.5, 7.8))
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 7.8)
    ax.axis("off")
    top_y, top_h, top_w = 6.1, 0.95, 2.45
    top = [
        (0.2, "Six-year monitoring data\n40 BHEs\n$T_{in}$, $T_{out}$ and flow rate", "#DCEAF7"),
        (3.25, "Physical preprocessing\n5-min to hourly aggregation\nOperating-mode separation", "#E4F2DE"),
        (6.30, "Feature engineering\nPeer load and flow ratio\nGeometry and time variables", "#E4F2DE"),
        (9.35, "Chronological data split\n2018–2021: training\n2022: calibration\n2023–2024H1: testing", "#DCEAF7"),
        (12.40, "Expected-performance models\nRidge regression\nLightGBM", "#F7EBC6"),
    ]
    for x, text, color in top:
        _workflow_box(ax, x, top_y, top_w, top_h, text, color, 11.7)
    for left, right in zip(top[:-1], top[1:]):
        _workflow_arrow(ax, (left[0] + top_w, top_y + top_h / 2), (right[0], top_y + top_h / 2))
    mid_y, mid_h, mid_w = 3.6, 1.08, 3.05
    mid = [
        (0.35, "Model-consensus screening\nAgreement between Ridge\nand LightGBM residuals", "#F7EBC6"),
        (4.10, "Leave-one-BHE-out validation\nEach BHE tested as a\npreviously unseen unit", "#F6E2D4"),
        (7.85, "Uncertainty quantification\n90% conformal intervals\nMonthly block bootstrap", "#F6E2D4"),
        (11.60, "Explainability and robustness\nSHAP, seasonal and load analyses\nPipe-length ablation", "#E8E2F0"),
    ]
    for x, text, color in mid:
        _workflow_box(ax, x, mid_y, mid_w, mid_h, text, color, 12.0)
    for left, right in zip(mid[:-1], mid[1:]):
        _workflow_arrow(ax, (left[0] + mid_w, mid_y + mid_h / 2), (right[0], mid_y + mid_h / 2))
    _workflow_arrow(ax, (14.65, top_y), (14.65, 5.35))
    _workflow_arrow(ax, (14.65, 5.35), (1.85, 5.35))
    _workflow_arrow(ax, (1.85, 5.35), (1.85, mid_y + mid_h))
    result_x, result_y, result_w, result_h = 3.0, 1.45, 9.5, 1.02
    _workflow_box(
        ax, result_x, result_y, result_w, result_h,
        "Cross-framework screening of persistent relative thermal underperformance\nConfirmed candidates: BHEs 24, 36 and 38\nBorderline candidate: BHE 2",
        "#DDEBD7", 12.2,
    )
    _workflow_arrow(ax, (13.125, mid_y), (13.125, 2.9))
    _workflow_arrow(ax, (13.125, 2.9), (7.75, 2.9))
    _workflow_arrow(ax, (7.75, 2.9), (7.75, result_y + result_h))
    number_positions = [(0.38,7.16),(3.43,7.16),(6.48,7.16),(9.53,7.16),(12.58,7.16),(0.53,4.79),(4.28,4.79),(8.03,4.79),(11.78,4.79)]
    for number, (x, y) in enumerate(number_positions, start=1):
        ax.text(x, y, str(number), ha="center", va="center", fontsize=9, fontweight="bold", color="white", bbox=dict(boxstyle="circle,pad=0.24", facecolor="#4F81BD", edgecolor="white", linewidth=0.7), zorder=6)
    ax.text(0.2, 0.28, "BHE: borehole heat exchanger; LOBO: leave-one-BHE-out; SHAP: Shapley additive explanations.", fontsize=9.2, color="#404040")
    fig.tight_layout()
    save_figure(fig, "Fig02_methodology_workflow")


def _add_bhe_ticks(ax) -> None:
    major_bhes = [1, 10, 20, 30, 40]
    minor_bhes = [bhe for bhe in range(1, 41) if bhe not in major_bhes]
    ax.set_yticks([bhe - 1 for bhe in major_bhes])
    ax.set_yticklabels([str(bhe) for bhe in major_bhes], fontsize=14)
    ax.set_yticks([bhe - 1 for bhe in minor_bhes], minor=True)
    ax.tick_params(axis="y", which="major", length=8, width=1.2, direction="out")
    ax.tick_params(axis="y", which="minor", length=4, width=0.8, direction="out", labelleft=False)


def figure_03_availability() -> None:
    df = read_processed("01_monthly_data_coverage.csv")
    df["month"] = pd.to_datetime(df["month"])
    df["availability"] = 100.0 * df["eligible_hour_count"] / (df["month"].dt.days_in_month * 24)
    months = pd.date_range(df["month"].min(), df["month"].max(), freq="MS")
    matrix = df.pivot_table(index="bhe", columns="month", values="availability").reindex(index=range(1,41), columns=months).fillna(0)
    fig, ax = plt.subplots(figsize=(15, 8.8))
    image = ax.imshow(matrix.values, cmap="viridis", aspect="auto", interpolation="nearest", vmin=0, vmax=100)
    _add_bhe_ticks(ax)
    ax.set_ylabel("BHE ID")
    ticks = [i for i, date in enumerate(months) if date.month in (1,7)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([months[i].strftime("%b\n%Y") for i in ticks], fontsize=12.5)
    ax.set_xlabel("Monitoring month")
    for boundary in np.arange(0.5, 40, 1):
        ax.axhline(boundary, color="white", linewidth=0.30, alpha=0.40)
    ax.axhline(11.5, color="white", linewidth=3.0)
    ax.axhline(24.5, color="white", linewidth=3.0)
    ax.text(1.015, 0.845, "Vault 1", transform=ax.transAxes, color=COLORS["blue"], fontweight="bold", fontsize=14)
    ax.text(1.015, 0.535, "Vault 2", transform=ax.transAxes, color=COLORS["orange"], fontweight="bold", fontsize=14)
    ax.text(1.015, 0.190, "Vault 3", transform=ax.transAxes, color=COLORS["green"], fontweight="bold", fontsize=14)
    cbar = fig.colorbar(image, ax=ax, fraction=0.036, pad=0.115)
    cbar.set_label("Eligible hourly data availability (%)", fontsize=15, labelpad=14)
    cbar.ax.tick_params(labelsize=13)
    fig.subplots_adjust(left=0.08, right=0.82, bottom=0.15, top=0.97)
    save_figure(fig, "Fig03_data_availability")


def figure_04_monthly_rpi() -> None:
    df = read_processed("02_monthly_raw_thermal_performance.csv")
    df = df.loc[df["mode"] == "ground_heat_rejection"].copy()
    df["month"] = pd.to_datetime(df["month"])
    months = pd.date_range(df["month"].min(), df["month"].max(), freq="MS")
    perf = df.pivot_table(index="bhe", columns="month", values="median_raw_relative_performance").reindex(index=range(1,41), columns=months)
    hours = df.pivot_table(index="bhe", columns="month", values="operating_hours").reindex(index=range(1,41), columns=months)
    perf = perf.mask(perf.isna() | hours.isna() | (hours < 24))
    cmap = plt.colormaps["RdBu_r"].copy()
    cmap.set_bad(COLORS["insufficient"])
    fig, ax = plt.subplots(figsize=(15.2, 8.8))
    image = ax.imshow(perf.values, cmap=cmap, aspect="auto", interpolation="nearest", vmin=0.6, vmax=1.4)
    _add_bhe_ticks(ax)
    ax.set_ylabel("BHE ID")
    ticks = [i for i, date in enumerate(months) if date.month in (1,7)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([months[i].strftime("%b\n%Y") for i in ticks], fontsize=12.5)
    ax.set_xlabel("Monitoring month")
    for boundary in np.arange(0.5, 40, 1):
        ax.axhline(boundary, color="white", linewidth=0.30, alpha=0.35)
    ax.axhline(11.5, color="#303030", linewidth=1.6)
    ax.axhline(24.5, color="#303030", linewidth=1.6)
    for bhe in [24,36,38]:
        ax.add_patch(Rectangle((-0.5, bhe - 1.5), len(months), 1, fill=False, edgecolor=COLORS["orange"], linewidth=1.8, zorder=6))
    ax.add_patch(Rectangle((-0.5, 0.5), len(months), 1, fill=False, edgecolor=COLORS["magenta"], linestyle="--", linewidth=1.7, zorder=6))
    ax.text(1.015,0.845,"Vault 1",transform=ax.transAxes,color=COLORS["blue"],fontweight="bold",fontsize=14)
    ax.text(1.015,0.535,"Vault 2",transform=ax.transAxes,color=COLORS["orange"],fontweight="bold",fontsize=14)
    ax.text(1.015,0.190,"Vault 3",transform=ax.transAxes,color=COLORS["green"],fontweight="bold",fontsize=14)
    cbar = fig.colorbar(image, ax=ax, fraction=0.036, pad=0.115)
    cbar.set_label("Median relative thermal performance", fontsize=15, labelpad=14)
    cbar.ax.tick_params(labelsize=13)
    handles = [
        Rectangle((0,0),1,1,fill=False,edgecolor=COLORS["orange"],linewidth=1.8,label="Confirmed candidate"),
        Rectangle((0,0),1,1,fill=False,edgecolor=COLORS["magenta"],linestyle="--",linewidth=1.7,label="Borderline candidate"),
        Patch(facecolor=COLORS["insufficient"],edgecolor="#808080",label="Insufficient data"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52,-0.11), ncol=3, frameon=False, fontsize=12.5)
    fig.subplots_adjust(left=0.08,right=0.82,bottom=0.20,top=0.97)
    save_figure(fig, "Fig04_monthly_relative_performance")


def figure_05_cross_framework() -> None:
    df = read_processed("06_lobo_bhe_summary.csv")
    confirmed, borderline = [24,36,38], [2]
    other = df.loc[~df["bhe"].isin(confirmed + borderline)]
    fig, axes = plt.subplots(1,2,figsize=(15,6.8))
    def format_panel(ax,xlim,ylim,label):
        ax.axhline(0,color="#404040",linewidth=1.1)
        ax.axvline(0,color="#404040",linewidth=1.1)
        ax.axhline(-10,color=COLORS["threshold"],linestyle="--",linewidth=1.1)
        ax.axvline(-10,color=COLORS["threshold"],linestyle="--",linewidth=1.1)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.grid(True,color="#E7E7E7",linewidth=0.6,linestyle=":")
        ax.spines[["top","right"]].set_visible(False)
        ax.set_xlabel("Median monthly thermal-power residual (%)")
        ax.set_ylabel("Median monthly relative-performance residual (%)")
        ax.text(0.02,0.95,label,transform=ax.transAxes,fontweight="bold",fontsize=13,bbox=dict(facecolor="white",edgecolor="none",pad=1.5),zorder=10)
    for ax, limits, label in zip(axes,[((-35,53),(-38,64)),((-35,2),(-40,2))],["(a)","(b)"]):
        format_panel(ax,*limits,label)
        ax.scatter(other["median_monthly_q_resid_pct"],other["median_monthly_rpi_resid_pct"],s=48,facecolor="#BFBFBF",edgecolor="#303030",linewidth=0.8,zorder=2)
        for bhe in confirmed:
            row=df.loc[df["bhe"]==bhe].iloc[0]
            ax.scatter(row["median_monthly_q_resid_pct"],row["median_monthly_rpi_resid_pct"],s=210,marker="*",facecolor=COLORS["redorange"],edgecolor="black",linewidth=0.9,zorder=6)
        row=df.loc[df["bhe"]==2].iloc[0]
        ax.scatter(row["median_monthly_q_resid_pct"],row["median_monthly_rpi_resid_pct"],s=115,marker="D",facecolor=COLORS["magenta"],edgecolor="black",linewidth=0.8,zorder=6)
    offsets={24:(0.8,1.5),36:(0.8,-1.6),38:(0.8,-0.2),2:(0.8,1.4)}
    for bhe in confirmed+borderline:
        row=df.loc[df["bhe"]==bhe].iloc[0]; dx,dy=offsets[bhe]
        axes[1].text(row["median_monthly_q_resid_pct"]+dx,row["median_monthly_rpi_resid_pct"]+dy,f"BHE {bhe}",fontsize=11,fontweight="bold")
    handles=[
        Line2D([0],[0],marker="o",linestyle="none",markerfacecolor="#BFBFBF",markeredgecolor="#303030",markersize=7,label="Other BHE"),
        Line2D([0],[0],marker="D",linestyle="none",markerfacecolor=COLORS["magenta"],markeredgecolor="black",markersize=9,label="Borderline candidate"),
        Line2D([0],[0],marker="*",linestyle="none",markerfacecolor=COLORS["redorange"],markeredgecolor="black",markersize=14,label="Confirmed candidate"),
    ]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(0.5,0.015),ncol=3,frameon=False,fontsize=13)
    fig.subplots_adjust(left=0.075,right=0.985,top=0.96,bottom=0.19,wspace=0.28)
    save_figure(fig, "Fig05_cross_framework_classification")


def figure_06_conformal_breaches() -> None:
    df=read_processed("06_lobo_bhe_summary.csv")
    all_bhes=pd.DataFrame({"bhe":range(1,41)}).merge(df[["bhe","q_hourly_below_90PI_lower_pct"]],on="bhe",how="left").fillna(0)
    confirmed=[24,36,38]; borderline=[2]
    colors=[COLORS["redorange"] if b in confirmed else COLORS["magenta"] if b in borderline else "#BDBDBD" for b in all_bhes["bhe"]]
    fig,ax=plt.subplots(figsize=(15.2,7.0))
    ax.bar(all_bhes["bhe"],all_bhes["q_hourly_below_90PI_lower_pct"],color=colors,edgecolor="#4A4A4A",linewidth=0.7,width=0.9,zorder=3)
    ax.axhline(5,color="#6F6F6F",linestyle=":",linewidth=1.2)
    ax.axhline(10,color="#8F8F8F",linestyle="--",linewidth=1.2)
    ax.set_xlabel("Held-out BHE ID"); ax.set_ylabel("Observations below the lower 90% prediction bound (%)")
    major=[1,5,10,15,20,25,30,35,40]; minor=[b for b in range(1,41) if b not in major]
    ax.set_xticks(major); ax.set_xticks(minor,minor=True)
    ax.tick_params(axis="x",which="major",length=7); ax.tick_params(axis="x",which="minor",length=3.5,labelbottom=False)
    ax.yaxis.grid(True,linestyle="--",linewidth=0.6,color="#E8E8E8"); ax.spines[["top","right"]].set_visible(False)
    ax.set_xlim(0.2,40.8); ax.set_ylim(0,max(65,all_bhes["q_hourly_below_90PI_lower_pct"].max()+6))
    for bhe in confirmed+borderline:
        value=float(all_bhes.loc[all_bhes["bhe"]==bhe,"q_hourly_below_90PI_lower_pct"].iloc[0])
        ax.text(bhe,value+0.9,f"{value:.1f}%",ha="center",fontsize=10.5,fontweight="bold")
    handles=[Patch(facecolor=COLORS["redorange"],edgecolor="#4A4A4A",label="Confirmed candidate"),Patch(facecolor=COLORS["magenta"],edgecolor="#4A4A4A",label="Borderline candidate"),Patch(facecolor="#BDBDBD",edgecolor="#4A4A4A",label="Other BHE"),Line2D([0],[0],color="#6F6F6F",linestyle=":",label="Nominal lower-tail rate (5%)"),Line2D([0],[0],color="#8F8F8F",linestyle="--",label="Screening threshold (10%)")]
    ax.legend(handles=handles,loc="upper center",bbox_to_anchor=(0.5,-0.085),ncol=5,frameon=False,fontsize=12.5)
    fig.subplots_adjust(left=0.11,right=0.98,top=0.97,bottom=0.20)
    save_figure(fig,"Fig06_conformal_lower_bound_breaches")


def figure_07_shap() -> None:
    df=read_processed("19_relative_performance_SHAP_importance.csv").sort_values("mean_abs_SHAP_RPI",ascending=False).reset_index(drop=True)
    labels={"pipe_length_m":"Horizontal pipe length","tin_difference_to_peers":"Inlet-temperature difference\nto peers","flow_ratio_to_peers":"Flow ratio to peers","vault":"Vault","month_sin":"Seasonal term","month_cos":"Seasonal term (cosine)","peer_count":"Number of active peers","year_index":"Long-term time index","flow_l_min":"BHE flow rate","peer_tin_median":"Median peer inlet temperature","peer_tin_median_c":"Median peer inlet temperature","tin_c":"BHE inlet temperature","peer_flow_median":"Median peer flow rate","peer_flow_median_l_min":"Median peer flow rate","hour_cos":"Diurnal term (cosine)","hour_sin":"Diurnal term"}
    df["label"]=df["feature"].map(labels).fillna(df["feature"].str.replace("_"," ").str.title())
    bar_colors=[COLORS["redorange"] if i==0 else COLORS["blue"] if i in (1,2) else COLORS["grey"] for i in range(len(df))]
    fig,ax=plt.subplots(figsize=(11.8,8.2))
    bars=ax.barh(df["label"],df["mean_abs_SHAP_RPI"],color=bar_colors,edgecolor="#4A4A4A",linewidth=0.8,zorder=3); ax.invert_yaxis()
    xmax=df["mean_abs_SHAP_RPI"].max()*1.23
    for bar,pct in zip(bars,df["importance_share_pct"]):
        ax.text(bar.get_width()+xmax*0.015,bar.get_y()+bar.get_height()/2,f"{pct:.1f}%",va="center",fontsize=11.5)
    ax.set_xlabel("Mean absolute SHAP value\n(relative-performance units)"); ax.xaxis.grid(True,linestyle=":",linewidth=0.7,color="#E6E6E6"); ax.spines[["top","right"]].set_visible(False); ax.set_xlim(0,xmax)
    handles=[Patch(facecolor=COLORS["redorange"],edgecolor="#4A4A4A",label="Dominant structural driver"),Patch(facecolor=COLORS["blue"],edgecolor="#4A4A4A",label="Secondary peer-relative drivers"),Patch(facecolor=COLORS["grey"],edgecolor="#4A4A4A",label="Other predictors")]
    ax.legend(handles=handles,loc="upper center",bbox_to_anchor=(0.5,-0.12),ncol=3,frameon=False,fontsize=13.5)
    fig.subplots_adjust(left=0.33,right=0.97,top=0.97,bottom=0.20)
    save_figure(fig,"Fig07_relative_performance_SHAP")


def figure_08_seasonal() -> None:
    df=read_processed("14_seasonal_conditioned_results.csv")
    settings={24:(COLORS["blue"],"o","BHE 24"),36:(COLORS["orange"],"s","BHE 36"),38:(COLORS["green"],"^","BHE 38"),2:(COLORS["magenta"],"D","BHE 2 (borderline)")}
    seasons=["DJF","MAM","JJA","SON"]; x=np.arange(4)
    fig,ax=plt.subplots(figsize=(12.5,7.4))
    for bhe,(color,marker,label) in settings.items():
        values=df.loc[df["bhe"]==bhe].set_index("season").reindex(seasons)["median_LOBO_residual_pct"]
        ax.plot(x,values,color=color,marker=marker,linewidth=2.5,markersize=8.5,label=label,zorder=4)
    ax.axhline(0,color="#404040",linewidth=1.2); ax.axhline(-10,color=COLORS["threshold"],linestyle="--",linewidth=1.5)
    ax.set_xticks(x); ax.set_xticklabels(seasons,fontsize=14); ax.set_xlabel("Season"); ax.set_ylabel("Median LOBO residual (%)")
    ax.yaxis.grid(True,linestyle=":",linewidth=0.7,color=COLORS["grid"]); ax.spines[["top","right"]].set_visible(False); ax.set_ylim(-55,5)
    threshold=Line2D([0],[0],color=COLORS["threshold"],linestyle="--",linewidth=1.7,label="Screening threshold (−10%)")
    handles=ax.get_legend_handles_labels()[0]+[threshold]
    ax.legend(handles=handles,loc="upper center",bbox_to_anchor=(0.5,-0.13),ncol=3,frameon=False,fontsize=13.5)
    fig.subplots_adjust(left=0.12,right=0.98,top=0.97,bottom=0.24)
    save_figure(fig,"Fig08_seasonal_persistence")


def figure_09_field_load() -> None:
    df=read_processed("16_field_load_conditioned_results.csv")
    candidates=[24,36,38,2]; source_order=["Low field load","Medium field load","High field load"]; labels=["Low load","Medium load","High load"]
    settings={24:(COLORS["blue"],None,"BHE 24"),36:(COLORS["orange"],"//","BHE 36"),38:(COLORS["green"],"\\\\","BHE 38"),2:(COLORS["magenta"],"..","BHE 2 (borderline)")}
    fig,ax=plt.subplots(figsize=(12.8,7.5)); positions=np.arange(3); width=0.19; offsets={24:-1.5*width,36:-0.5*width,38:0.5*width,2:1.5*width}; handles=[]
    for bhe in candidates:
        values=df.loc[df["bhe"]==bhe].set_index("peer_load_regime").reindex(source_order)["median_LOBO_residual_pct"].to_numpy(float)
        valid=~np.isnan(values); color,hatch,label=settings[bhe]
        bars=ax.bar(positions[valid]+offsets[bhe],values[valid],width=width,color=color,edgecolor="#202020",linewidth=1.0,hatch=hatch,zorder=3)
        handles.append(Patch(facecolor=color,edgecolor="#202020",hatch=hatch,label=label))
        for bar,value in zip(bars,values[valid]):
            ax.text(bar.get_x()+bar.get_width()/2,value-1.0,f"{value:.1f}%",ha="center",va="top",fontsize=10.5,fontweight="bold")
        for xpos in positions[~valid]+offsets[bhe]:
            ax.text(xpos,-3.0,"N/A",ha="center",va="center",fontsize=13,fontstyle="italic",fontweight="bold")
    ax.axhline(0,color="#404040",linewidth=1.2); ax.axhline(-10,color=COLORS["threshold"],linestyle="--",linewidth=1.5)
    ax.set_xticks(positions); ax.set_xticklabels(labels,fontsize=14); ax.set_xlabel("Simultaneous vault-load regime"); ax.set_ylabel("Median LOBO residual (%)")
    ax.yaxis.grid(True,linestyle=":",linewidth=0.7,color=COLORS["grid"]); ax.spines[["top","right"]].set_visible(False); ax.set_ylim(-58,5)
    handles.append(Line2D([0],[0],color=COLORS["threshold"],linestyle="--",linewidth=1.7,label="Screening threshold (−10%)"))
    ax.legend(handles=handles,loc="upper center",bbox_to_anchor=(0.5,-0.14),ncol=3,frameon=False,fontsize=13.5)
    fig.subplots_adjust(left=0.12,right=0.98,top=0.97,bottom=0.25)
    save_figure(fig,"Fig09_field_load_conditioned_persistence")


def figure_10_paired_controls() -> None:
    df=read_processed("17_paired_candidate_control_results.csv")
    order=[24,36,38,2]; df["order"]=pd.Categorical(df["candidate_bhe"],categories=order,ordered=True); df=df.sort_values("order")
    settings={24:(COLORS["blue"],"o","BHE 24"),36:(COLORS["orange"],"s","BHE 36"),38:(COLORS["green"],"^","BHE 38"),2:(COLORS["magenta"],"D","BHE 2 (borderline)")}
    fig,ax=plt.subplots(figsize=(13,7.8)); x=np.arange(len(df))
    for i,row in enumerate(df.itertuples(index=False)):
        color,marker,_=settings[int(row.candidate_bhe)]; median=row.median_candidate_minus_control_residual_pct; low=row.day_block_bootstrap_CI_low_pct; high=row.day_block_bootstrap_CI_high_pct
        ax.errorbar(i,median,yerr=np.array([[median-low],[high-median]]),fmt=marker,markersize=10.5,markerfacecolor=color,markeredgecolor="#202020",markeredgewidth=1.1,ecolor=color,elinewidth=2.0,capsize=7,capthick=1.8,zorder=5)
        ax.text(i,high+0.9,f"{median:.1f}%",ha="center",fontsize=12.5,fontweight="bold")
    ax.axhline(0,color="#303030",linewidth=1.3); ax.axhline(-10,color=COLORS["threshold"],linestyle="--",linewidth=1.6)
    labels=[]
    for row in df.itertuples(index=False):
        label=f"BHE {int(row.candidate_bhe)} vs {int(row.matched_control_bhe)}"
        if int(row.candidate_bhe)==2: label += "\n(borderline)"
        labels.append(label)
    ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=13.5); ax.set_xlabel("Candidate–control BHE pair"); ax.set_ylabel("Median LOBO residual difference (%)")
    ax.yaxis.grid(True,linestyle=":",linewidth=0.7,color=COLORS["grid"]); ax.spines[["top","right"]].set_visible(False); ax.set_ylim(min(-36,df["day_block_bootstrap_CI_low_pct"].min()-4),max(3,df["day_block_bootstrap_CI_high_pct"].max()+4))
    handles=[Line2D([0],[0],marker=settings[b][1],linestyle="none",markerfacecolor=settings[b][0],markeredgecolor="#202020",markersize=10,label=settings[b][2]) for b in order]
    handles += [Line2D([0],[0],color=COLORS["threshold"],linestyle="--",linewidth=1.8,label="Screening threshold (−10%)"),Line2D([0],[0],color="#303030",marker="|",markersize=13,markeredgewidth=2,linestyle="none",label="Day-block bootstrap interval")]
    ax.legend(handles=handles,loc="upper center",bbox_to_anchor=(0.5,-0.15),ncol=3,frameon=False,fontsize=13.5)
    fig.subplots_adjust(left=0.12,right=0.98,top=0.97,bottom=0.27)
    save_figure(fig,"Fig10_paired_candidate_control")


def main() -> None:
    ensure_directories()
    metadata=pd.read_csv(METADATA_DIR/"bhe_metadata.csv")
    figure_01_field_layout(metadata)
    figure_02_workflow()
    figure_03_availability()
    figure_04_monthly_rpi()
    figure_05_cross_framework()
    figure_06_conformal_breaches()
    figure_07_shap()
    figure_08_seasonal()
    figure_09_field_load()
    figure_10_paired_controls()
    print(f"Generated figures in {FIGURE_DIR}")


if __name__ == "__main__":
    main()
