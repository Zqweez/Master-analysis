"""
Reads all xlsx files from MIC and Potentiation separately and generates
aggregated growth curves (mean +/- std) for each sample.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import re


def _timestamps_to_hours(timestamps):
	timestamps_numeric = []
	for ts in timestamps:
		parts = str(ts).split()
		try:
			hours = int(parts[0])
			minutes = int(parts[2]) if len(parts) > 2 else 0
			timestamps_numeric.append(hours + minutes / 60)
		except (ValueError, IndexError):
			nums = re.findall(r"\d+", str(ts))
			if len(nums) >= 2:
				timestamps_numeric.append(int(nums[0]) + int(nums[1]) / 60)
			elif len(nums) == 1:
				timestamps_numeric.append(float(nums[0]))
			else:
				timestamps_numeric.append(float("nan"))

	hour_tick_positions = []
	hour_tick_labels = []
	for idx, t in enumerate(timestamps_numeric):
		if pd.notna(t) and float(t).is_integer():
			hour_tick_positions.append(idx)
			hour_tick_labels.append(str(int(t)))

	return timestamps_numeric, hour_tick_positions, hour_tick_labels


def load_data(file_path: str):
	df = pd.read_excel(file_path, "Table All Cycles", skiprows=12)
	# Rename column 0 to Well and column 1 to Content
	df.rename(columns={df.columns[0]: "Well", df.columns[1]: "Content"}, inplace=True)
	df["Well"] = df["Well"].astype(str).str.strip().str.upper()
	return df


def load_mapping(sample_mapping: str):
	mapping_df = pd.read_csv(sample_mapping)
	required_columns = {"Well", "Sample", "Max_conc"}
	missing_columns = required_columns.difference(mapping_df.columns)
	if missing_columns:
		raise ValueError(
			f"Mapping file {sample_mapping} missing required columns: {sorted(missing_columns)}"
		)

	mapping_df["Well"] = mapping_df["Well"].astype(str).str.strip().str.upper()
	return mapping_df


def _get_concentration_lookup(sample_id: str, max_conc: float):
	well_ids = [f"{sample_id}{idx:02d}" for idx in range(2, 12)]
	conc_range = [max_conc * (2 ** -i) for i in range(10)]
	conc_range.reverse()
	return dict(zip(well_ids, conc_range))


def extract_plate_long_df(df: pd.DataFrame, mapping_df: pd.DataFrame, group_name: str, source_file: str):
	timestamps = list(df.columns[2:])
	time_hours, _, _ = _timestamps_to_hours(timestamps)
	records = []

	for _, map_row in mapping_df.iterrows():
		sample_id = str(map_row["Well"]).strip().upper()
		sample_name = str(map_row["Sample"]).strip()
		max_conc = pd.to_numeric(map_row["Max_conc"], errors="coerce")
		if pd.isna(max_conc):
			print(f"Warning: invalid Max_conc for sample {sample_name} in {source_file}; skipping.")
			continue

		well_conc_lookup = _get_concentration_lookup(sample_id, float(max_conc))
		sample_rows = df[df["Well"].isin(well_conc_lookup.keys())].copy()

		for _, row in sample_rows.iterrows():
			well = row["Well"]
			concentration = well_conc_lookup[well]
			for time_index, timestamp in enumerate(timestamps):
				od_value = pd.to_numeric(row[timestamp], errors="coerce")
				if pd.isna(od_value):
					continue

				records.append(
					{
						"Group": group_name,
						"SourceFile": source_file,
						"SampleID": sample_id,
						"Sample": sample_name,
						"Well": well,
						"Concentration": concentration,
						"TimeIndex": time_index,
						"TimeLabel": str(timestamp),
						"TimeHours": time_hours[time_index],
						"OD": float(od_value),
					}
				)

	return pd.DataFrame(records)


def aggregate_long_df(long_df: pd.DataFrame):
	stats_df = (
		long_df.groupby(["Sample", "Concentration", "TimeIndex"], as_index=False)
		.agg(
			MeanOD=("OD", "mean"),
			StdOD=("OD", "std"),
			N=("OD", "count"),
			TimeLabel=("TimeLabel", "first"),
			TimeHours=("TimeHours", "mean"),
		)
		.sort_values(["Sample", "Concentration", "TimeIndex"])
	)
	stats_df["StdOD"] = stats_df["StdOD"].fillna(0.0)
	return stats_df


def _sanitize_filename(name: str):
	return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def make_plot(ax, sample_stats: pd.DataFrame, sample_name: str, shared_y_max: float = None):
	concentrations = sorted(sample_stats["Concentration"].dropna().unique())
	if len(concentrations) == 0:
		return False

	palette = sns.color_palette("Spectral", n_colors=len(concentrations))

	time_info = (
		sample_stats[["TimeIndex", "TimeLabel", "TimeHours"]]
		.drop_duplicates("TimeIndex")
		.sort_values("TimeIndex")
	)
	tick_positions = time_info["TimeIndex"].tolist()
	hour_tick_positions = []
	hour_tick_labels = []
	for i, (_, row) in enumerate(time_info.iterrows()):
		time_hour = row["TimeHours"]
		if pd.notna(time_hour) and float(time_hour).is_integer():
			hour_tick_positions.append(tick_positions[i])
			hour_tick_labels.append(str(int(time_hour)))

	for idx, conc in enumerate(concentrations):
		conc_stats = sample_stats[sample_stats["Concentration"] == conc].sort_values("TimeIndex")
		x_values = conc_stats["TimeIndex"].to_numpy(dtype=float)
		y_values = conc_stats["MeanOD"].to_numpy(dtype=float)
		std_values = conc_stats["StdOD"].to_numpy(dtype=float)

		ax.plot(
			x_values,
			y_values,
			color=palette[idx],
			label=f"{conc:g}",
		)
		ax.fill_between(
			x_values,
			y_values - std_values,
			y_values + std_values,
			color=palette[idx],
			alpha=0.25,
		)

	ax.set_xticks(hour_tick_positions if hour_tick_positions else tick_positions)
	ax.set_xticklabels(hour_tick_labels if hour_tick_labels else time_info["TimeLabel"].tolist(), fontsize=8)
	ax.set_title(sample_name, fontsize=14)
	ax.set_xlabel("Time (hours)", fontsize=12)
	ax.set_ylabel("OD", fontsize=12)
	ax.legend(title="Concentration", fontsize=8, loc="upper left")
	ax.grid(alpha=0.2)

	if shared_y_max is not None:
		ax.set_ylim(0, shared_y_max)

	return True


def collect_group_long_df(group_path: Path, group_name: str):
	xlsx_paths = sorted(group_path.rglob("*.xlsx"))
	if not xlsx_paths:
		return pd.DataFrame()

	long_dfs = []
	for xlsx_path in xlsx_paths:
		sample_mapping = xlsx_path.with_suffix(".csv")
		if not sample_mapping.exists():
			print(f"Warning: no mapping file found for {xlsx_path.name}; skipping.")
			continue

		print(f"Processing {group_name}: {xlsx_path.name}...")
		df = load_data(str(xlsx_path))
		mapping_df = load_mapping(str(sample_mapping))
		plate_long_df = extract_plate_long_df(df, mapping_df, group_name, xlsx_path.name)
		if not plate_long_df.empty:
			long_dfs.append(plate_long_df)

	if not long_dfs:
		return pd.DataFrame()

	return pd.concat(long_dfs, ignore_index=True)


def make_group_aggregated_growth_curves(group_long_df: pd.DataFrame, save_path: Path):
	if group_long_df.empty:
		return

	save_path.mkdir(parents=True, exist_ok=True)

	stats_df = aggregate_long_df(group_long_df)

	shared_y_max = (stats_df["MeanOD"] + stats_df["StdOD"]).max() * 1.05

	for sample_name, sample_stats in stats_df.groupby("Sample"):
		fig, ax = plt.subplots(figsize=(8, 4.6))
		was_plotted = make_plot(ax, sample_stats, sample_name, shared_y_max=shared_y_max)
		if was_plotted:
			fig.tight_layout()
			save_file = save_path / f"{_sanitize_filename(sample_name)}_aggregated_growth_curve.pdf"
			fig.savefig(save_file, bbox_inches="tight")
		plt.close(fig)


def main(data_dir: str = "data", output_dir: str = "outputs/growth-aggregated"):
	data_path = Path(data_dir)
	if not data_path.exists():
		raise FileNotFoundError(f"Data directory {data_path} does not exist")

	output_path = Path(output_dir)
	output_path.mkdir(parents=True, exist_ok=True)

	group_names = ["MIC", "Potentiation"]
	for group_name in group_names:
		group_path = data_path / group_name
		if not group_path.exists():
			print(f"Skipping {group_name}: folder not found at {group_path}")
			continue

		group_long_df = collect_group_long_df(group_path, group_name)
		if group_long_df.empty:
			print(f"No valid data found for {group_name}; nothing to aggregate.")
			continue

		group_save_path = output_path / group_name
		make_group_aggregated_growth_curves(group_long_df, group_save_path)
		print(f"Saved aggregated outputs for {group_name} to {group_save_path}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Aggregate growth curves from all MIC and Potentiation data files")
	parser.add_argument("--data_dir", type=str, default="data", help="Path to the data root folder")
	parser.add_argument("--output_dir", type=str, default="outputs/growth-aggregated", help="Path where grouped aggregated outputs are saved")
	args = parser.parse_args()
	main(data_dir=args.data_dir, output_dir=args.output_dir)
