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

def extract_sample_controls_df(df: pd.DataFrame, sample_ids: list, mapping_dict: dict):
    if not sample_ids:
        return pd.DataFrame()

    max_sample_row = max(sample_ids)
    control_row_ord = ord(max_sample_row) + 1

    # If samples already use row H, there is no next row for per-sample controls.
    if control_row_ord > ord("H"):
        return pd.DataFrame()

    control_row = chr(control_row_ord)
    control_well_to_sample = {
        f"{control_row}{idx + 2:02d}": sample_id
        for idx, sample_id in enumerate(sample_ids)
    }

    controls_df = df[df["Well"].isin(control_well_to_sample.keys())].copy()
    if controls_df.empty:
        return controls_df

    controls_df["_sample_id"] = controls_df["Well"].map(control_well_to_sample)
    controls_df["_sample_id"] = pd.Categorical(controls_df["_sample_id"], categories=sample_ids, ordered=True)
    controls_df = controls_df.sort_values("_sample_id")
    controls_df["Content"] = controls_df["_sample_id"].map(lambda sid: f"{mapping_dict.get(sid, sid)} control")
    controls_df = controls_df.drop(columns=["_sample_id"])

    return controls_df

def make_plot(
    ax,
    rows: pd.DataFrame,
    sample_id: str,
    mapping_dict: dict,
    conc_dict: dict,
    shared_y_max: float = None,
    sterile_blank: pd.Series = None,
):
    # Only make plots when there are 10 rows
    if len(rows) < 8 and sample_id != "SampleControls":
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
        palette = sns.color_palette("Spectral", n_colors=len(rows))
        for idx, (_, row) in enumerate(rows.iterrows()):
            y_values = pd.to_numeric(row[timestamps], errors="coerce")
            if sterile_blank is not None:
                y_values = y_values - sterile_blank
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
            if sterile_blank is not None:
                y_values = y_values - sterile_blank
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
    if shared_y_max is not None:
        y_min = 0 if sterile_blank is None else -0.25
        ax.set_ylim(y_min, shared_y_max)

    return True

def make_growth_curves(
    df: pd.DataFrame,
    save_path: str = "outputs/growth_curves",
    sample_mapping: str = None,
    subtract_sterile_blank: bool = False,
):
    """
    We want to make one figure for each row, corresponding to one compound tested
    But we want to also make plots for the controls
    """
    save_path = Path(save_path)

    # Sort out the SC and GC
    # SC is in A1, B1, etc and GC is in A12, B12, etc
    sc_df = df[df["Well"].str.match(r"^[A-Z]01$")].copy()
    gc_df = df[df["Well"].str.match(r"^[A-Z]12$")].copy()

    sc_df["Content"] = [f"SC{i+1}" for i in range(len(sc_df))]
    gc_df["Content"] = [f"GC{i+1}" for i in range(len(gc_df))]

    timestamps = list(df.columns[2:])
    sterile_blank = None
    if subtract_sterile_blank:
        if sc_df.empty:
            print("Warning: no sterile control wells found; skipping blank subtraction.")
        else:
            sterile_blank = sc_df[timestamps].apply(pd.to_numeric, errors="coerce").mean(axis=0)
            sterile_blank = sterile_blank.iloc[0]
            
    # Load sample mapping mapping well id to actual sample name
    mapping_df = pd.read_csv(sample_mapping)
    mapping_df["Well"] = mapping_df["Well"].astype(str).str.strip().str.upper()
    sample_ids = mapping_df["Well"].tolist()
    mapping_dict = dict(zip(mapping_df["Well"], mapping_df["Sample"]))
    conc_dict = dict(zip(mapping_df["Well"], mapping_df["Max_conc"]))
    mapping_dict.update({
        "Control": "Sterile Control",
        "Negative": "Growth Control",
        "SampleControls": "Sample negative controls",
    })

    # Make df with only the samples
    sample_df = df.drop(sc_df.index).drop(gc_df.index)

    sample_controls_df = extract_sample_controls_df(df, sample_ids, mapping_dict)

    plot_data = [
        ("Control", sc_df),
        ("Negative", gc_df),
    ]

    if not sample_controls_df.empty:
        plot_data.append(("SampleControls", sample_controls_df))
    
    # Make one plot for each sample, overlaying all rows where the first letter of the well is the same
    # Timestamps are the columns starting from the 3rd column, so we get them by df.columns[2:]
    for i in range(len(sample_ids)):
        sample_id = sample_ids[i]
        sample_rows = sample_df[sample_df["Well"].str.startswith(sample_id)]
        plot_data.append((sample_id, sample_rows))

    # Max value across all rows and columns in df
    df_max = df.max(numeric_only=True).max()
    shared_y_max = df_max * 1.05

    # Individual plots
    save_path.mkdir(parents=True, exist_ok=True)
    valid_plot_data = []
    for sample_id, rows in plot_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        was_plotted = make_plot(
            ax,
            rows,
            sample_id,
            mapping_dict,
            conc_dict,
            shared_y_max=shared_y_max,
            sterile_blank=sterile_blank,
        )
        if was_plotted:
            fig.tight_layout()
            save_file = save_path / f"{mapping_dict.get(sample_id, sample_id)}_growth_curve.pdf"
            fig.savefig(save_file, bbox_inches="tight")
            valid_plot_data.append((sample_id, rows))
        plt.close(fig)

    # Combined figure with 3 columns and as many rows as needed
    if valid_plot_data:
        n_cols = 4 if len(valid_plot_data) % 4 == 0 or len(valid_plot_data) > 9 else 3
        n_rows = (len(valid_plot_data) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
        axes_flat = axes.flatten()

        for idx, (sample_id, rows) in enumerate(valid_plot_data):
            make_plot(
                axes_flat[idx],
                rows,
                sample_id,
                mapping_dict,
                conc_dict,
                shared_y_max=shared_y_max,
                sterile_blank=sterile_blank,
            )

        for idx in range(len(valid_plot_data), len(axes_flat)):
            axes_flat[idx].set_visible(False)

        fig.tight_layout()
        combined_save_file = save_path / "all_growth_curves_subfigures.pdf"
        fig.savefig(combined_save_file, bbox_inches="tight")
        plt.close(fig)

def plot_all_data(subtract_sterile_blank: bool = False):
    # Get all data paths
    xlsx_paths = Path("data/").rglob("*.xlsx")
    for path in xlsx_paths:
        print(f"Running {path.stem}...")
        main(path, subtract_sterile_blank=subtract_sterile_blank)


def main(data_path: str, subtract_sterile_blank: bool = False):
    # Load data
    data_path = Path(data_path)
    if not data_path.exists():
        raise(f"Data file {data_path} does not exist. Please provide a valid path.")

    data_df = load_data(data_path)

    save_path = Path("outputs/growth_curves") / data_path.stem
    sample_mapping = data_path.with_suffix(".csv")
    make_growth_curves(
        data_df,
        save_path,
        sample_mapping,
        subtract_sterile_blank=subtract_sterile_blank,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate growth curves from CLARIOstar data")
    parser.add_argument("--data_path", type=str, default="data/MIC/2026_03_24_MIC_LMC139_Test.xlsx", help="Path to the input xlsx file")
    parser.add_argument("--run_all", action="store_true")
    parser.add_argument("--blank", action="store_true", help="Blank with averaged sterile control",)
    args = parser.parse_args()
    if args.run_all:
        plot_all_data(subtract_sterile_blank=args.blank)
    else:
        main(args.data_path, subtract_sterile_blank=args.blank)
