"""
Reads MIC summary CSV data and generates a clean bar chart per OMPP
with mean +/- std and overlaid replicate datapoints.
"""
from pathlib import Path
from typing import Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np


def _sort_sample_label(label: str):
    numeric_label = pd.to_numeric(label, errors="coerce")
    if pd.notna(numeric_label):
        return (0, float(numeric_label))
    return (1, str(label))


def _parse_mic_value(value, censor_floor: float = 256.0) -> Tuple[float, bool, float]:
    """
    Parse MIC values while tracking censored values written as '>X'.
    Censored values are stored as X for plotting and statistics.
    """
    value_str = str(value).strip()
    if value_str == "" or value_str.lower() == "nan":
        return np.nan, False, np.nan

    if value_str.startswith(">"):
        threshold_value = pd.to_numeric(value_str[1:].strip(), errors="coerce")
        if pd.isna(threshold_value):
            threshold_value = float(censor_floor)
        return float(threshold_value), True, float(threshold_value)

    numeric_value = pd.to_numeric(value_str, errors="coerce")
    if pd.isna(numeric_value):
        return np.nan, False, np.nan
    return float(numeric_value), False, np.nan


def load_data(file_path: str, censor_floor: float = 256.0):
    df = pd.read_csv(file_path)
    columns_lower = {col.lower().strip(): col for col in df.columns}
    if "sample" not in columns_lower or "mic" not in columns_lower:
        raise ValueError("CSV must contain columns named 'Sample' and 'mic'.")

    sample_col = columns_lower["sample"]
    mic_col = columns_lower["mic"]

    data = df[[sample_col, mic_col]].copy()
    data.columns = ["Sample", "MicRaw"]
    data["Sample"] = data["Sample"].astype(str).str.strip()

    parsed = data["MicRaw"].apply(lambda value: _parse_mic_value(value, censor_floor=censor_floor))
    data[["MIC", "IsCensored", "CensorThreshold"]] = pd.DataFrame(parsed.tolist(), index=data.index)
    data = data.dropna(subset=["MIC"]).reset_index(drop=True)
    return data


def summarize_data(data: pd.DataFrame):
    stats_df = (
        data.groupby("Sample", as_index=False)
        .agg(
            MeanMIC=("MIC", "mean"),
            StdMIC=("MIC", "std"),
            N=("MIC", "count"),
            AnyCensored=("IsCensored", "any"),
            AllCensored=("IsCensored", "all"),
            MaxCensorThreshold=("CensorThreshold", "max"),
        )
        .sort_values("Sample", key=lambda sample_col: sample_col.map(_sort_sample_label))
        .reset_index(drop=True)
    )
    stats_df["StdMIC"] = stats_df["StdMIC"].fillna(0.0)
    return stats_df


def _point_offsets(n_points: int, spread: float = 0.14):
    if n_points <= 1:
        return np.array([0.0])
    return np.linspace(-spread, spread, n_points)


def _powers_of_two_ticks(min_value: float, max_value: float):
    min_exp = int(np.floor(np.log2(min_value)))
    max_exp = int(np.ceil(np.log2(max_value)))
    tick_exponents = np.arange(min_exp, max_exp + 1)
    tick_positions = tick_exponents.astype(float)
    tick_labels = [f"{(2 ** exp):g}" for exp in tick_exponents]
    return tick_positions, tick_labels


def make_plot(data: pd.DataFrame, title: str):
    stats_df = summarize_data(data)
    if stats_df.empty:
        raise ValueError("No valid MIC values found to plot.")

    if (data["MIC"] <= 0).any():
        raise ValueError("MIC values must be positive to use a log2 y-axis.")

    n_samples = len(stats_df)
    fig_width = max(7.0, 0.95 * n_samples)
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))

    x_values = np.arange(n_samples)
    bar_color = sns.color_palette("Spectral", n_colors=8)[2]
    edge_color = "#2F2F2F"

    min_mic = float(data.loc[data["MIC"] > 0, "MIC"].min())
    mean_values = stats_df["MeanMIC"].to_numpy(dtype=float)
    std_values = stats_df["StdMIC"].to_numpy(dtype=float)
    lower_linear = np.clip(mean_values - std_values, min_mic * 0.5, None)
    upper_linear = np.clip(mean_values + std_values, min_mic * 0.5, None)
    mean_log2 = np.log2(mean_values)
    yerr_log2 = np.vstack([
        mean_log2 - np.log2(lower_linear),
        np.log2(upper_linear) - mean_log2,
    ])

    bars = ax.bar(
        x_values,
        mean_log2,
        yerr=yerr_log2,
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
        ax.axhline(np.log2(level), color="#888888", linestyle="--", linewidth=0.9, alpha=0.9, zorder=1)

    for idx, sample_name in enumerate(stats_df["Sample"]):
        sample_rows = data[data["Sample"] == sample_name]
        offsets = _point_offsets(len(sample_rows))

        for offset, row in zip(offsets, sample_rows.itertuples(index=False)):
            mic_value = float(row.MIC)
            is_censored = bool(row.IsCensored)
            marker_style = "^" if is_censored else "o"
            point_y = mic_value * 1.03 if is_censored else mic_value

            ax.scatter(
                idx + offset,
                np.log2(point_y),
                marker=marker_style,
                s=38,
                color="#8C8C8C",
                edgecolors="#666666",
                linewidths=0.5,
                zorder=3,
            )

        if bool(stats_df.loc[idx, "AnyCensored"]):
            bars[idx].set_hatch("///")

        if bool(stats_df.loc[idx, "AllCensored"]):
            censor_level = stats_df.loc[idx, "MaxCensorThreshold"]
            ax.text(
                idx,
                np.log2(stats_df.loc[idx, "MeanMIC"]) + 0.12,
                f">{censor_level:g}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#3A3A3A",
                fontweight="bold",
            )

    max_value = float(np.max(upper_linear))
    if unique_censor_levels:
        max_value = max(max_value, max(unique_censor_levels) * 1.1)

    tick_positions, tick_labels = _powers_of_two_ticks(min_mic, max_value)
    y_min = float(tick_positions.min() - 0.25)
    y_max = float(tick_positions.max() + 0.35)

    ax.set_ylim(y_min, y_max)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    ax.set_xticks(x_values)
    ax.set_xticklabels(stats_df["Sample"])
    ax.set_xlabel("OMPP")
    ax.set_ylabel("MIC (log2 scale)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)

    fig.tight_layout()
    return fig


def main(
    data_path: str = "data/MIC/ompp_mic.csv",
    output_path: str = None,
    title: str = "MIC per OMPP",
    censor_floor: float = 256.0,
):
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file {data_path} does not exist")

    if output_path is None:
        output_file = Path("outputs/growth-aggregated/MIC") / f"{data_path.stem}_bar_chart.pdf"
    else:
        output_file = Path(output_path)

    data_df = load_data(str(data_path), censor_floor=censor_floor)
    fig = make_plot(data_df, title=title)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create MIC bar chart with mean +/- std and replicate points")
    parser.add_argument("--data_path", type=str, default="data/MIC/ompp_mic.csv", help="Path to input CSV containing Sample and mic")
    parser.add_argument("--output_path", type=str, default=None, help="Optional explicit output file path")
    parser.add_argument("--title", type=str, default="MIC per OMPP", help="Plot title")
    parser.add_argument("--censor_floor", type=float, default=256.0, help="Fallback threshold for malformed >X entries")
    args = parser.parse_args()

    sns.set_theme(style="whitegrid", context="paper")
    main(
        data_path=args.data_path,
        output_path=args.output_path,
        title=args.title,
        censor_floor=args.censor_floor,
    )