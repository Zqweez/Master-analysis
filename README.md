### Master's Thesis Analysis

This repository contains the code used for the in vitro analysis and visualizations of the results. The in silico part of the project is available in a separate repository: 
<a href="https://github.com/Zqweez/Masters-thesis">https://github.com/Zqweez/Masters-thesis</a>

## How to set up the environment
Conda has been used to manage the Python environment for this project. The enviroment is combined with the in silico part of the project, such that the same environment can be used for both, resulting in a larger environment than necessary for this in vitro analysis.

To set up the environment, first install Conda if you haven't already. Then, navigate to the root directory of this repository and run the following command, naming the environment using the `-n` flag, for example `masters`:

```bash
conda env create -f environment.yml -n masters

# Then activate the environment
conda activate masters
```

## How to run the code
Scripts are grouped by experiment type. Inputs live under `data/` and outputs are written to `outputs/`.

### MIC and Potentiation
- `scripts/mic-potentiation/growth-curves.py`: creates growth curves per xlsx file (one output folder per file, with one curve per sample). Each xlsx needs a matching CSV mapping file with the same name, see test file in `data/`.
- `scripts/mic-potentiation/aggregated-curves.py`: aggregates all MIC or potentiation xlsx files in `data/MIC` and `data/Potentiation` into mean +/- std curves per peptide.
- `scripts/mic-potentiation/mic-bar-chart.py`: MIC bar charts from a MIC summary spreadsheet.
- `scripts/mic-potentiation/pot-bar-chart.py`: potentiation bar charts from a potentiation summary spreadsheet.

Example:
```bash
python scripts/mic-potentiation/growth-curves.py --data_path data/MIC/2026_03_24_MIC_LMC139_Test.xlsx
```

### NPN
- `scripts/npn/npn-analysis.py`: bar charts from the NPN summary spreadsheet.
- `scripts/npn/npn-aggregated-curves.py`: fluorescence curves (mean +/- std) aggregated per sample for technical replicates from each xlsx file.