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
Currently, only code for making growth curves from MIC data is available.

### MIC growth curves
Make sure the xlsx files with the MIC data are in the `data` folder. And that there is a corresponding CSV file with the sample mapping. See the test files included in the data folder. Then run the following command from the root of the repository, replacing the path to the xlsx file with the correct path to your data file:

```bash
python scripts/mic-analysis/growth-curves.py --data_path data/Kasper_Test_MIC_20260324.xlsx
```
Then the growth curves will be saved in the `outputs/growth_curves` folder, in a subfolder named after the input xlsx file.