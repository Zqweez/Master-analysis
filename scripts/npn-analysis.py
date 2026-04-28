from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np

def _sort_ompp_label(label: str):
    numeric_label = pd.to_numeric(label, errors="coerce")
    if pd.notna(numeric_label):
        return (0, float(numeric_label))
    return (1, str(label))


def _point_offsets(n_points: int, spread: float = 0.14) -> np.ndarray:
    if n_points <= 1:
        return np.array([0.0])
    return np.linspace(-spread, spread, n_points)


def load_data(data_path: Path | str):
    df = pd.read_excel(data_path)
    df.columns = [str(col).strip() for col in df.columns]

    rep_cols = [col for col in df.columns if col != "ompp"]
    if not rep_cols:
        raise ValueError("Input file must contain at least one biological replicate column.")

    tech_df = df.melt(
        id_vars=["ompp"],
        value_vars=rep_cols,
        var_name="BioRep",
        value_name="Value",
    )
    tech_df["ompp"] = tech_df["ompp"].astype(str).str.strip()
    tech_df["Value"] = pd.to_numeric(tech_df["Value"], errors="coerce")
    tech_df = tech_df.dropna(subset=["ompp", "Value"]).reset_index(drop=True)

    # Convert to percent and label technical replicate order within each biological replicate.
    tech_df["Value"] = tech_df["Value"] * 100.0
    tech_df["TechnicalRep"] = tech_df.groupby(["ompp", "BioRep"]).cumcount() + 1

    tech_df.to_csv("outputs/npn/technical_replicates.csv", index=False)

    bio_df = (
        tech_df.groupby(["ompp", "BioRep"], as_index=False)
        .agg(
            BioMean=("Value", "mean"),
            NTechnical=("Value", "count"),
        )
    )

    bio_df.to_csv("outputs/npn/biological_replicates.csv", index=False)

    stats_df = (
        bio_df.groupby("ompp", as_index=False)
        .agg(
            mean=("BioMean", "mean"),
            std=("BioMean", "std"),
            n_bio=("BioMean", "count"),
        )
        .sort_values("ompp", key=lambda ompp_col_values: ompp_col_values.map(_sort_ompp_label))
        .reset_index(drop=True)
    )
    stats_df["std"] = stats_df["std"].fillna(0.0)

    stats_df.to_csv("outputs/npn/summary_statistics.csv", index=False)

    return tech_df, bio_df, stats_df


def make_barchart(
    tech_df: pd.DataFrame,
    bio_df: pd.DataFrame,
    stats_df: pd.DataFrame,
):
    if stats_df.empty:
        raise ValueError("No valid NPN values found to plot.")

    n_samples = len(stats_df)
    fig_width = max(7.0, 0.95 * n_samples)
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))

    x_values = np.arange(n_samples)
    mean_vals = stats_df["mean"].to_numpy(dtype=float)
    std_vals = stats_df["std"].to_numpy(dtype=float)

    bar_color = sns.color_palette("Spectral", n_colors=n_samples)
    edge_color = "#2F2F2F"

    ax.bar(
        x_values,
        mean_vals,
        yerr=std_vals,
        capsize=4,
        width=0.72,
        color=bar_color,
        edgecolor=edge_color,
        linewidth=0.9,
        error_kw={"elinewidth": 1.0, "ecolor": edge_color},
        zorder=2,
    )

    for idx, ompp in enumerate(stats_df["ompp"]):
        ompp_bio = bio_df[bio_df["ompp"] == ompp].reset_index(drop=True)
        bio_offsets = _point_offsets(len(ompp_bio), spread=0.16)

        for bio_offset, bio_row in zip(bio_offsets, ompp_bio.itertuples(index=False)):
            x_bio = idx + bio_offset

            ompp_tech = tech_df[(tech_df["ompp"] == ompp) & (tech_df["BioRep"] == bio_row.BioRep)]
            tech_offsets = _point_offsets(len(ompp_tech), spread=0.035)

            """
            # For plotting each technical replicate as a separate point as well
            for tech_offset, tech_row in zip(tech_offsets, ompp_tech.itertuples(index=False)):
                ax.scatter(
                    x_bio + tech_offset,
                    float(tech_row.Value),
                    s=10,
                    color="#8A8A8A",
                    alpha=0.9,
                    linewidths=0,
                    zorder=3,
                )
            """

            ax.scatter(
                x_bio,
                float(bio_row.BioMean),
                s=52,
                color="#3F3F3F",
                edgecolors="white",
                linewidths=0.7,
                zorder=4,
            )

    ax.set_ylim(-10, 100)
    ax.set_xticks(x_values)
    ax.set_xticklabels(stats_df["ompp"])
    ax.set_xlabel("OMPP")
    ax.set_ylabel("% NPN")
    ax.set_title("NPN for each OMPP", fontdict={"fontsize": 12})
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    fig.tight_layout()
    output_path = Path("outputs/npn/bar-chart.pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Create NPN bar chart with biological and technical replicate points")
    parser.add_argument("--data_path", type=str, default="data/NPN/npn_results.xlsx", help="Path to input Excel file")
    args = parser.parse_args()

    tech_df, bio_df, stats_df = load_data(args.data_path)
    make_barchart(
        tech_df=tech_df,
        bio_df=bio_df,
        stats_df=stats_df,
    )

if __name__ == "__main__":
    main()