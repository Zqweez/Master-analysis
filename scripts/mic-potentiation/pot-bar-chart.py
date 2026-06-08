"""
Reads potentiation MIC data and generates a log2 fold-change bar chart per OMPP
with mean +/- std and overlaid replicate datapoints.
"""
from pathlib import Path
from typing import Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np

REFERENCE_MIC = 64.0


def _sort_sample_label(label: str):
    numeric_label = pd.to_numeric(label, errors="coerce")
    if pd.notna(numeric_label):
        return (0, float(numeric_label))
    return (1, str(label))


def _parse_pot_value(value) -> Tuple[float, bool, float]:
    """
    Parse MIC values while tracking censored values written as '<X'.
    Censored values are stored as X for plotting and statistics.
    """
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


def load_data(file_path: str, reference_mic: float):
    df = pd.read_csv(file_path)
    data = df[["sample", "mic"]].copy()
    data.columns = ["Sample", "MicRaw"]
    data["Sample"] = data["Sample"].astype(str).str.strip()

    parsed = data["MicRaw"].apply(lambda value: _parse_pot_value(value))
    data[["MIC", "IsCensored", "MicThreshold"]] = pd.DataFrame(parsed.tolist(), index=data.index)
    data = data.dropna(subset=["MIC"]).reset_index(drop=True)
    data = data[data["MIC"] > 0].copy()
    data["Log2FC"] = _calculate_log2fc(data["MIC"].astype(float), reference_mic)
    data["CensorThreshold"] = np.where(
        data["IsCensored"],
        _calculate_log2fc(data["MicThreshold"].astype(float), reference_mic),
        np.nan,
    )
    return data


def summarize_data(data: pd.DataFrame, output_dir: Path):
    stats_df = (
        data.groupby("Sample", as_index=False)
        .agg(
            MeanLog2FC=("Log2FC", "mean"),
            StdLog2FC=("Log2FC", "std"),
            N=("Log2FC", "count"),
            AnyCensored=("IsCensored", "any"),
            AllCensored=("IsCensored", "all"),
            MaxCensorThreshold=("CensorThreshold", "max"),
        )
        .sort_values("Sample", key=lambda sample_col: sample_col.map(_sort_sample_label))
        .reset_index(drop=True)
    )
    stats_df["StdLog2FC"] = stats_df["StdLog2FC"].fillna(0.0)

    output_dir.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(output_dir / "summary_statistics.csv", index=False)
    return stats_df

def _log2fc_ticks(min_value: float, max_value: float):
    min_tick = int(np.floor(min_value))
    max_tick = int(np.ceil(max_value))
    tick_positions = np.arange(min_tick, max_tick + 1).astype(float)
    tick_labels = [f"{tick:g}" for tick in tick_positions]
    return tick_positions, tick_labels


def make_plot(data: pd.DataFrame, output_file: Path, reference_mic: float):
    stats_df = summarize_data(data, output_file.parent)

    n_samples = len(stats_df)
    fig_width = max(5.0, 1 * n_samples)
    fig, ax = plt.subplots(figsize=(fig_width, 5))

    x_values = np.arange(n_samples)
    bar_color = sns.color_palette("Spectral", n_colors=8)
    edge_color = "#2F2F2F"

    mean_values = stats_df["MeanLog2FC"].to_numpy(dtype=float)
    std_values = stats_df["StdLog2FC"].to_numpy(dtype=float)

    bars = ax.bar(
        x_values,
        mean_values,
        yerr=std_values,
        capsize=4,
        width=0.72,
        color=bar_color,
        edgecolor=edge_color,
        linewidth=0.9,
        error_kw={"elinewidth": 1.0, "ecolor": edge_color},
        zorder=2,
    )

    unique_censor_levels = sorted(data.loc[data["IsCensored"], "CensorThreshold"].dropna().unique().tolist())
    for level in unique_censor_levels:
        ax.axhline(level, color="#888888", linestyle="--", linewidth=0.9, alpha=0.9, zorder=1)

    for idx, sample_name in enumerate(stats_df["Sample"]):
        sample_rows = data[data["Sample"] == sample_name]
        offsets = np.linspace(-0.14, 0.14, len(sample_rows))

        for offset, row in zip(offsets, sample_rows.itertuples(index=False)):
            log2fc_value = float(row.Log2FC)
            is_censored = bool(row.IsCensored)
            marker_style = "^" if is_censored else "o"
            point_y = log2fc_value + 0.12 if is_censored else log2fc_value

            ax.scatter(
                idx + offset,
                point_y,
                marker=marker_style,
                s=38,
                color="#8C8C8C",
                edgecolors="#666666",
                linewidths=0.5,
                zorder=3,
            )

        # Set hash pattern for bars with censored values
        if bool(stats_df.loc[idx, "AnyCensored"]):
            bars[idx].set_hatch("///")

        # Add text annotation for fully censored samples clearly showing that the mean is above the theshold
        if bool(stats_df.loc[idx, "AllCensored"]):
            censor_level = stats_df.loc[idx, "MaxCensorThreshold"]
            ax.text(
                idx,
                stats_df.loc[idx, "MeanLog2FC"] + 0.12,
                f">{censor_level:g}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#3A3A3A",
                fontweight="bold",
            )

    lower_values = mean_values - std_values
    upper_values = mean_values + std_values
    min_value = float(np.min(lower_values))
    max_value = float(np.max(upper_values))
    if unique_censor_levels:
        min_value = min(min_value, min(unique_censor_levels))
        max_value = max(max_value, max(unique_censor_levels))

    tick_positions, tick_labels = _log2fc_ticks(min_value, max_value)
    y_min = float(tick_positions.min() - 0.35)
    y_max = float(tick_positions.max() + 0.45)

    ax.set_ylim(y_min, y_max)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    ax.set_xticks(x_values)
    ax.set_xticklabels(stats_df["Sample"])
    ax.set_xlabel("OMPP", fontsize=14)
    ax.set_ylabel(f"log2 fold change vs {reference_mic:g} μg/ml", fontsize=14)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    fig.suptitle("Potentiation log2 fold change for each OMPP", fontsize=14, y=1)
    # Note that all OMPPs had constant concentration of 10 μg/ml, but OMPP 53 had 2 μg/ml, so we can add this as a note in the figure
    fig.text(0.5, 0, f"Note: OMPP 53 had a constant concentration of 2 μg/ml", ha="center", va="bottom", fontsize=11, color="#3A3A3A")

    fig.tight_layout()
    fig.savefig(output_file, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Saved figure to {output_file}")


def main(data_path: str = "data/Potentiation/ompp_pot.csv", reference_mic: float = REFERENCE_MIC):
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file {data_path} does not exist")

    output_file = Path("outputs/potentiation") / f"{data_path.stem}_log2fc_bars.pdf"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    data_df = load_data(str(data_path), reference_mic)
    make_plot(data_df, output_file, reference_mic)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create potentiation log2 fold-change bar chart with mean +/- std and replicate points")
    parser.add_argument("--data_path", type=str, default="data/Potentiation/ompp_pot.csv", help="Path to input CSV containing Sample and mic")
    args = parser.parse_args()

    # sns.set_theme(style="whitegrid", context="paper")
    main(
        data_path=args.data_path,
    )