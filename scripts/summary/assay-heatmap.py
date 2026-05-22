"""
Build a heatmap summary for MIC, potentiation, and NPN assays.
Rows are assays and columns are OMPP labels. Colors are normalized per row.
"""
from pathlib import Path
from typing import Tuple
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

REFERENCE_MIC = 64.0


def _sort_ompp_label(label: str):
    numeric_label = pd.to_numeric(label, errors="coerce")
    if pd.notna(numeric_label):
        return (0, float(numeric_label))
    return (1, str(label))


def _parse_mic_value(value) -> Tuple[float, bool, float]:
    value_str = str(value).strip()
    if value_str == "" or value_str.lower() == "nan":
        return np.nan, False, np.nan

    if value_str.startswith(">"):
        threshold_value = pd.to_numeric(value_str[1:].strip(), errors="coerce")
        if pd.isna(threshold_value):
            threshold_value = 256.0
        return float(threshold_value), True, float(threshold_value)

    numeric_value = pd.to_numeric(value_str, errors="coerce")
    if pd.isna(numeric_value):
        return np.nan, False, np.nan
    return float(numeric_value), False, np.nan


def _parse_pot_value(value) -> Tuple[float, bool, float]:
    value_str = str(value).strip()
    if value_str == "" or value_str.lower() == "nan":
        return np.nan, False, np.nan

    if value_str.startswith("<"):
        threshold_value = pd.to_numeric(value_str[1:].strip(), errors="coerce")
        if pd.isna(threshold_value):
            threshold_value = 0.25
        return float(threshold_value), True, float(threshold_value)

    numeric_value = pd.to_numeric(value_str, errors="coerce")
    if pd.isna(numeric_value):
        return np.nan, False, np.nan
    return float(numeric_value), False, np.nan


def _calculate_log2fc(values: pd.Series, reference_mic: float) -> pd.Series:
    return np.log2(reference_mic / values)


def load_mic_means(file_path: str) -> pd.Series:
    df = pd.read_csv(file_path)
    data = df[["sample", "mic"]].copy()
    data.columns = ["Sample", "MicRaw"]
    data["Sample"] = data["Sample"].astype(str).str.strip()

    parsed = data["MicRaw"].apply(_parse_mic_value)
    data[["MIC", "IsCensored", "CensorThreshold"]] = pd.DataFrame(parsed.tolist(), index=data.index)
    data = data.dropna(subset=["MIC"]).reset_index(drop=True)

    means = data.groupby("Sample", as_index=True)["MIC"].mean()
    means.index = means.index.astype(str).str.strip()
    return means


def load_pot_means(file_path: str, reference_mic: float) -> pd.Series:
    df = pd.read_csv(file_path)
    data = df[["sample", "mic"]].copy()
    data.columns = ["Sample", "MicRaw"]
    data["Sample"] = data["Sample"].astype(str).str.strip()

    parsed = data["MicRaw"].apply(_parse_pot_value)
    data[["MIC", "IsCensored", "MicThreshold"]] = pd.DataFrame(parsed.tolist(), index=data.index)
    data = data.dropna(subset=["MIC"]).reset_index(drop=True)
    data = data[data["MIC"] > 0].copy()
    data["Log2FC"] = _calculate_log2fc(data["MIC"].astype(float), reference_mic)

    means = data.groupby("Sample", as_index=True)["Log2FC"].mean()
    means.index = means.index.astype(str).str.strip()
    return means


def load_npn_means(file_path: str) -> pd.Series:
    df = pd.read_excel(file_path)
    df.columns = [str(col).strip() for col in df.columns]

    if "ompp" not in df.columns:
        raise ValueError("NPN input file must contain an 'ompp' column.")

    rep_cols = [col for col in df.columns if col != "ompp"]
    if not rep_cols:
        raise ValueError("NPN input file must contain at least one biological replicate column.")

    tech_df = df.melt(
        id_vars=["ompp"],
        value_vars=rep_cols,
        var_name="BioRep",
        value_name="Value",
    )
    tech_df["ompp"] = tech_df["ompp"].astype(str).str.strip()
    tech_df["Value"] = pd.to_numeric(tech_df["Value"], errors="coerce")
    tech_df = tech_df.dropna(subset=["ompp", "Value"]).reset_index(drop=True)

    tech_df["Value"] = tech_df["Value"] * 100.0

    bio_df = (
        tech_df.groupby(["ompp", "BioRep"], as_index=False)
        .agg(BioMean=("Value", "mean"))
    )

    means = bio_df.groupby("ompp", as_index=True)["BioMean"].mean()
    means.index = means.index.astype(str).str.strip()
    return means


def build_summary(mic_path: str, pot_path: str, npn_path: str, reference_mic: float) -> pd.DataFrame:
    mic_means = load_mic_means(mic_path)
    pot_means = load_pot_means(pot_path, reference_mic)
    npn_means = load_npn_means(npn_path)

    all_ompp = sorted(
        set(mic_means.index) | set(pot_means.index) | set(npn_means.index),
        key=_sort_ompp_label,
    )

    summary_df = pd.DataFrame(index=["MIC", "Potentiation", "NPN"], columns=all_ompp, dtype=float)
    summary_df.loc["MIC", mic_means.index] = mic_means.values
    summary_df.loc["Potentiation", pot_means.index] = pot_means.values
    summary_df.loc["NPN", npn_means.index] = npn_means.values
    summary_df.index.name = "Assay"
    return summary_df


def rowwise_normalize(summary_df: pd.DataFrame) -> pd.DataFrame:
    normalized = summary_df.copy()
    for row_name in summary_df.index:
        row_values = summary_df.loc[row_name]
        valid = row_values.dropna()
        if valid.empty:
            continue
        min_val = float(valid.min())
        max_val = float(valid.max())
        if np.isclose(min_val, max_val):
            normalized.loc[row_name, valid.index] = 0.5
        else:
            normalized.loc[row_name, valid.index] = (valid - min_val) / (max_val - min_val)
    return normalized


def _format_value(row_name: str, value: float) -> str:
    if row_name == "NPN":
        return f"{value:.1f}"
    if row_name == "Potentiation":
        return f"{value:.2f}"
    return f"{value:g}"


def make_heatmap(summary_df: pd.DataFrame, output_file: Path, annotate: bool) -> None:
    norm_df = rowwise_normalize(summary_df)
    mask = norm_df.isna()

    n_cols = len(norm_df.columns)
    fig_width = max(6.0, 0.7 * n_cols + 2.0)
    fig_height = max(2.4, 0.55 * len(norm_df.index) + 1.2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    cmap = sns.color_palette("Blues", as_cmap=True)

    heatmap_kwargs = dict(
        data=norm_df,
        mask=mask,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "Low → High (row-wise)"},
    )

    if annotate:
        annot_df = pd.DataFrame(index=summary_df.index, columns=summary_df.columns, dtype=object)
        for row_name in summary_df.index:
            for ompp in summary_df.columns:
                value = summary_df.loc[row_name, ompp]
                annot_df.loc[row_name, ompp] = "" if pd.isna(value) else _format_value(row_name, float(value))
        heatmap_kwargs["annot"] = annot_df
        heatmap_kwargs["fmt"] = ""
        heatmap_kwargs["annot_kws"] = {"fontsize": 8}

    sns.heatmap(ax=ax, **heatmap_kwargs)

    ax.set_xlabel("OMPP", fontsize=12)
    ax.set_ylabel("Assay", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_title("", fontsize=13, pad=12) # Assay summary heatmap

    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a row-normalized heatmap summarizing MIC, potentiation, and NPN")
    parser.add_argument("--mic_path", type=str, default="data/MIC/ompp_mic.csv", help="Path to MIC summary CSV")
    parser.add_argument("--pot_path", type=str, default="data/Potentiation/ompp_pot.csv", help="Path to potentiation summary CSV")
    parser.add_argument("--npn_path", type=str, default="data/NPN/npn_results.xlsx", help="Path to NPN summary Excel")
    parser.add_argument("--reference_mic", type=float, default=REFERENCE_MIC, help="Reference MIC for log2 fold change")
    parser.add_argument("--output_dir", type=str, default="outputs/summary", help="Output directory for summary and figure")
    parser.add_argument("--annotate", action="store_true", help="Annotate heatmap with raw mean values")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_file = output_dir / "assay_heatmap.pdf"

    summary_df = build_summary(args.mic_path, args.pot_path, args.npn_path, args.reference_mic)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_dir / "assay_summary.csv")

    make_heatmap(summary_df, output_file, annotate=args.annotate)
    print(f"Saved summary table to {output_dir / 'assay_summary.csv'}")
    print(f"Saved heatmap to {output_file}")


if __name__ == "__main__":
    main()
