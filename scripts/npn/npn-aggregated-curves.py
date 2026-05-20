"""
Aggregate NPN fluorescence data per sample and plot mean +/- std over time.

Reads each .xlsx in the NPN folder that has a matching .csv mapping file.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


REPS_PER_SAMPLE = 3
OMPP_ORDER = list(range(48, 56))
OMPP_PALETTE = sns.color_palette("Spectral", n_colors=len(OMPP_ORDER))
OMPP_COLOR_BY_NUM = {num: color for num, color in zip(OMPP_ORDER, OMPP_PALETTE)}

FIXED_SAMPLE_COLORS = {
	"PolB": "#303030",
	"Media + NPN": "#7EA8AF",
	"Cells + NPN": "#C6BB83",
	"Cells + Media": "#A1A1A1",
	"Media only": "#A1A1A1",
}


def _sanitize_filename(name: str) -> str:
	return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def _ompp_sort_key(sample_name: str):
	match = re.findall(r"\d+", sample_name)
	if match:
		return (0, int(match[0]))
	return (1, sample_name)


def _get_ompp_number(sample_name: str) -> int | None:
	match = re.findall(r"\d+", sample_name)
	if match:
		return int(match[0])
	return None


def _parse_time_minutes(label) -> float:
	if pd.isna(label):
		return float("nan")
	if isinstance(label, (int, float)):
		return float(label)

	parts = re.findall(r"\d+", str(label))
	if len(parts) >= 2:
		minutes = int(parts[0])
		seconds = int(parts[1])
		return minutes + seconds / 60.0
	if len(parts) == 1:
		return float(parts[0])
	return float("nan")


def _load_mapping(mapping_path: Path) -> list[str]:
	mapping_df = pd.read_csv(mapping_path)
	required_columns = {"Spot", "Sample"}
	missing = required_columns.difference(mapping_df.columns)
	if missing:
		raise ValueError(
			f"Mapping file {mapping_path.name} missing required columns: {sorted(missing)}"
		)

	mapping_df["Spot"] = pd.to_numeric(mapping_df["Spot"], errors="coerce")
	mapping_df["Sample"] = mapping_df["Sample"].astype(str).str.strip()
	mapping_df = mapping_df.dropna(subset=["Spot", "Sample"]).sort_values("Spot")
	return mapping_df["Sample"].tolist()


def _load_plate_data(xlsx_path: Path) -> tuple[pd.DataFrame, list[str], list[float]]:
	df = pd.read_excel(xlsx_path, sheet_name="Table All Cycles", skiprows=12)
	if df.shape[1] < 3:
		raise ValueError(f"Expected at least 3 columns in {xlsx_path.name}.")

	df.columns = [str(col).strip() for col in df.columns]
	df.rename(columns={df.columns[0]: "Well", df.columns[1]: "SampleLabel"}, inplace=True)

	time_cols = list(df.columns[2:])
	for col in time_cols:
		df[col] = pd.to_numeric(df[col], errors="coerce")

	valid_time_cols = [col for col in time_cols if df[col].notna().any()]
	df = df[["Well", "SampleLabel"] + valid_time_cols]
	df = df.dropna(subset=valid_time_cols, how="all").reset_index(drop=True)

	time_minutes = [_parse_time_minutes(col) for col in valid_time_cols]
	if any(pd.isna(val) for val in time_minutes):
		print(
			"Warning: could not parse one or more time labels; falling back to 30-second steps."
		)
		time_minutes = [idx * 0.5 for idx in range(len(valid_time_cols))]

	return df, valid_time_cols, time_minutes


def _assign_samples_by_order(
	df: pd.DataFrame,
	sample_names: list[str],
	reps_per_sample: int = REPS_PER_SAMPLE,
) -> pd.DataFrame:
	expected_rows = len(sample_names) * reps_per_sample
	if len(df) < expected_rows:
		print(
			f"Warning: {df.shape[0]} rows found but {expected_rows} expected; using available rows."
		)
	elif len(df) > expected_rows:
		print(
			f"Warning: {df.shape[0]} rows found but {expected_rows} expected; truncating to the first {expected_rows}."
		)

	trimmed = df.iloc[:expected_rows].copy()
	sample_assignments = np.repeat(sample_names, reps_per_sample)[: len(trimmed)]
	trimmed["Sample"] = sample_assignments
	trimmed["TechRep"] = (
		np.tile(np.arange(1, reps_per_sample + 1), len(sample_names))[: len(trimmed)]
	)
	return trimmed


def _aggregate_samples(
	assigned_df: pd.DataFrame, time_cols: list[str], time_minutes: list[float]
) -> pd.DataFrame:
	long_df = assigned_df.melt(
		id_vars=["Sample", "TechRep"],
		value_vars=time_cols,
		var_name="TimeColumn",
		value_name="Fluorescence",
	)
	long_df = long_df.dropna(subset=["Fluorescence"]).reset_index(drop=True)

	time_index_lookup = {col: idx for idx, col in enumerate(time_cols)}
	time_minutes_lookup = {col: minutes for col, minutes in zip(time_cols, time_minutes)}
	long_df["TimeIndex"] = long_df["TimeColumn"].map(time_index_lookup)
	long_df["TimeMinutes"] = long_df["TimeColumn"].map(time_minutes_lookup)

	stats_df = (
		long_df.groupby(["Sample", "TimeIndex"], as_index=False)
		.agg(
			MeanFluorescence=("Fluorescence", "mean"),
			StdFluorescence=("Fluorescence", "std"),
			N=("Fluorescence", "count"),
			TimeMinutes=("TimeMinutes", "first"),
		)
		.sort_values(["Sample", "TimeIndex"])
	)
	stats_df["StdFluorescence"] = stats_df["StdFluorescence"].fillna(0.0)
	return stats_df


def _build_color_map(sample_names: list[str]) -> dict[str, tuple[float, float, float]]:
	ompp_samples = [name for name in sample_names if "OMPP" in name.upper()]
	ompp_samples_sorted = sorted(ompp_samples, key=_ompp_sort_key)
	palette = sns.color_palette("Spectral", n_colors=max(1, len(ompp_samples_sorted)))

	color_map: dict[str, tuple[float, float, float]] = {}
	for sample_name in sample_names:
		if sample_name in FIXED_SAMPLE_COLORS:
			color_map[sample_name] = FIXED_SAMPLE_COLORS[sample_name]
		elif sample_name in ompp_samples_sorted:
			ompp_num = _get_ompp_number(sample_name)
			if ompp_num in OMPP_COLOR_BY_NUM:
				color_map[sample_name] = OMPP_COLOR_BY_NUM[ompp_num]
			else:
				color_map[sample_name] = palette[ompp_samples_sorted.index(sample_name)]
		else:
			color_map[sample_name] = "#4C4C4C"
	return color_map


def _plot_file(stats_df: pd.DataFrame, output_path: Path, title: str):
	if stats_df.empty:
		print(f"No valid data to plot for {title}.")
		return

	sample_names = stats_df["Sample"].unique().tolist()
	color_map = _build_color_map(sample_names)

	fig, ax = plt.subplots(figsize=(9.2, 5.5))
	for sample_name in sample_names:
		sample_stats = stats_df[stats_df["Sample"] == sample_name]
		x_values = sample_stats["TimeMinutes"].to_numpy(dtype=float)
		y_values = sample_stats["MeanFluorescence"].to_numpy(dtype=float)
		y_err = sample_stats["StdFluorescence"].to_numpy(dtype=float)

		color = color_map.get(sample_name, "#4C4C4C")
		ax.plot(
			x_values,
			y_values,
			marker="o",
			markersize=3.6,
			linewidth=0.8,
			color=color,
			label=sample_name,
		)
		ax.errorbar(
			x_values,
			y_values,
			yerr=y_err,
			fmt="none",
			ecolor=color,
			elinewidth=0.6,
			capsize=0,
			alpha=0.8,
		)

	ax.set_xlabel("Time (min)", fontsize=12)
	ax.set_ylabel("Fluorescence", fontsize=12)
	ax.set_title(title, fontsize=14)
	ax.grid(alpha=0.2)
	ax.legend(fontsize=8, loc="upper left")
	fig.tight_layout()

	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, bbox_inches="tight")
	plt.close(fig)


def process_npndata_file(xlsx_path: Path, output_dir: Path, reps_per_sample: int = REPS_PER_SAMPLE):
	mapping_path = xlsx_path.with_suffix(".csv")
	if not mapping_path.exists():
		print(f"Skipping {xlsx_path.name}: no mapping csv found.")
		return

	print(f"Processing {xlsx_path.name}...")
	sample_names = _load_mapping(mapping_path)
	raw_df, time_cols, time_minutes = _load_plate_data(xlsx_path)
	assigned_df = _assign_samples_by_order(raw_df, sample_names, reps_per_sample=reps_per_sample)
	stats_df = _aggregate_samples(assigned_df, time_cols, time_minutes)

	stem = _sanitize_filename(xlsx_path.stem)
	stats_output = output_dir / f"{stem}_summary.csv"
	stats_df.to_csv(stats_output, index=False)

	plot_output = output_dir / f"{stem}_aggregated_npndata.pdf"
	_plot_file(stats_df, plot_output, title=f"{' '.join(xlsx_path.stem.split(' ')[-2:])}")


def main():
	parser = argparse.ArgumentParser(
		description="Aggregate NPN fluorescence curves per sample and plot mean +/- std."
	)
	parser.add_argument("--data_dir", type=str, default="data/NPN", help="NPN data folder")
	parser.add_argument(
		"--output_dir",
		type=str,
		default="outputs/npn/aggregated-curves",
		help="Output folder for plots and summary stats",
	)
	parser.add_argument(
		"--reps",
		type=int,
		default=REPS_PER_SAMPLE,
		help="Technical replicates per sample",
	)
	args = parser.parse_args()

	data_dir = Path(args.data_dir)
	if not data_dir.exists():
		raise FileNotFoundError(f"Data directory {data_dir} does not exist")

	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	xlsx_paths = sorted(p for p in data_dir.glob("*.xlsx") if not p.name.startswith("~$"))
	if not xlsx_paths:
		print(f"No .xlsx files found in {data_dir}")
		return

	for xlsx_path in xlsx_paths:
		process_npndata_file(xlsx_path, output_dir, reps_per_sample=args.reps)


if __name__ == "__main__":
	main()
