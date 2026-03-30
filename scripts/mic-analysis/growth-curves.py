"""
Reads the xlsx file from the CLARIOstar and generates overlaying growthcurves for each row corersponding to one chemical
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def _timestamps_to_hours(timestamps):
    timestamps_numeric = []
    for ts in timestamps:
        parts = ts.split()
        hours = int(parts[0])
        minutes = int(parts[2]) if len(parts) > 2 else 0
        timestamps_numeric.append(hours + minutes / 60)

    hour_tick_positions = []
    hour_tick_labels = []
    for idx, t in enumerate(timestamps_numeric):
        if float(t).is_integer():
            hour_tick_positions.append(idx)
            hour_tick_labels.append(str(int(t)))

    return hour_tick_positions, hour_tick_labels

def load_data(file_path: str):
    df = pd.read_excel(file_path, "Table All Cycles", skiprows=12)
    # Rename column 0 to Well and column 1 to Content
    df.rename(columns={df.columns[0]: "Well", df.columns[1]: "Content"}, inplace=True)
    # print(df.head())

    return df

def make_plot(ax, rows: pd.DataFrame, sample_id: str, mapping_dict: dict, conc_dict: dict):

    # Only make plots when there are 10 rows
    if len(rows) < 8:
        return False

    rows = rows.copy()
    
    # Use the conc_dict to get the concentration for each row and use it as label
    max_conc = conc_dict.get(sample_id, None)
    conc_range = None
    if max_conc is not None:
        conc_range = [max_conc * (2 ** -i) for i in range(10)]
        conc_range.reverse()
    
    timestamps = list(rows.columns[2:])

    if conc_range is not None:
        palette = sns.color_palette("viridis", n_colors=len(rows))
        for idx, (_, row) in enumerate(rows.iterrows()):
            y_values = pd.to_numeric(row[timestamps], errors="coerce")
            ax.plot(
                timestamps,
                y_values,
                color=palette[idx],
                label=f"{conc_range[idx]:g}",
            )
    else:
        palette_name = "Greys"
        palette = sns.color_palette(palette_name, n_colors=len(rows))
        for idx, (_, row) in enumerate(rows.iterrows()):
            y_values = pd.to_numeric(row[timestamps], errors="coerce")
            ax.plot(
                timestamps,
                y_values,
                color=palette[idx],
                label=row["Content"],
            )

    hour_tick_positions, hour_tick_labels = _timestamps_to_hours(timestamps)
    ax.set_xticks(hour_tick_positions)
    ax.set_xticklabels(hour_tick_labels, fontsize=8)
    ax.set_title(f"{mapping_dict.get(sample_id, sample_id)}")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("OD")
    ax.legend(fontsize=8)

    return True

def make_growth_curves(df: pd.DataFrame, save_path: str = "outputs/growth_curves", sample_mapping: str = None):
    """
    We want to make one figure for each row, corresponding to one compound tested
    But we want to also make plots for the controls
    """
    save_path = Path(save_path)

    # Sort out the SC and GC
    sc_df = df[df["Content"].str.startswith("Control")]
    gc_df = df[df["Content"].str.startswith("Negative")]

    # Make df with only the samples
    sample_df = df.drop(sc_df.index).drop(gc_df.index)
    sample_ids = sample_df["Well"].str[0].unique()

    # Load sample mapping mapping well id to actual sample name
    mapping_df = pd.read_csv(sample_mapping)
    mapping_dict = dict(zip(mapping_df["Well"], mapping_df["Sample"]))
    conc_dict = dict(zip(mapping_df["Well"], mapping_df["Max_conc"]))
    mapping_dict.update({"Control": "Sterile Control", "Negative": "Growth Control"})

    plot_data = [
        ("Control", sc_df),
        ("Negative", gc_df),
    ]
    
    # Make one plot for each sample, overlaying all rows where the first letter of the well is the same
    # Timestamps are the columns starting from the 3rd column, so we get them by df.columns[2:]
    for i in range(len(sample_ids)):
        sample_id = sample_ids[i]
        sample_rows = sample_df[sample_df["Well"].str.startswith(sample_id)]
        plot_data.append((sample_id, sample_rows))

    # Individual plots
    save_path.mkdir(parents=True, exist_ok=True)
    valid_plot_data = []
    for sample_id, rows in plot_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        was_plotted = make_plot(ax, rows, sample_id, mapping_dict, conc_dict)
        if was_plotted:
            fig.tight_layout()
            save_file = save_path / f"{mapping_dict.get(sample_id, sample_id)}_growth_curve.pdf"
            fig.savefig(save_file, bbox_inches="tight")
            valid_plot_data.append((sample_id, rows))
        plt.close(fig)

    # Combined figure with 3 columns and as many rows as needed
    if valid_plot_data:
        n_cols = 3
        n_rows = (len(valid_plot_data) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
        axes_flat = axes.flatten()

        for idx, (sample_id, rows) in enumerate(valid_plot_data):
            make_plot(axes_flat[idx], rows, sample_id, mapping_dict, conc_dict)

        for idx in range(len(valid_plot_data), len(axes_flat)):
            axes_flat[idx].set_visible(False)

        fig.tight_layout()
        combined_save_file = save_path / "all_growth_curves_subfigures.pdf"
        fig.savefig(combined_save_file, bbox_inches="tight")
        plt.close(fig)


def main(data_path: str):
    # Load data
    data_path = Path(data_path)
    data_df = load_data(data_path)

    save_path = Path("outputs/growth_curves") / data_path.stem
    sample_mapping = data_path.with_suffix(".csv")
    make_growth_curves(data_df, save_path, sample_mapping)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate growth curves from CLARIOstar data")
    parser.add_argument("--data_path", type=str, default=None, help="Path to the input xlsx file")
    args = parser.parse_args()
    main(args.data_path)
