# Well Log Analyzer

<p align="center">
  <img src="img/logvip_screenshot.png" alt="LOGVIP Screenshot" width="800"/>
</p>


A Python-based tool for petrophysical analysis and lithology estimation from well log data. This project provides a suite of functions to read, clean, and interpret well log data, culminating in a robust petrophysical pipeline and advanced multi-mineral analysis.

## Features

*   **Data Handling**: Reads well log data from `.las`, `.csv`, and `.xlsx` files.
*   **Data Cleaning**: Handles missing values and allows for the selection of relevant log curves.
*   **Petrophysical Calculations**: A complete pipeline to compute:
    *   Gamma Ray Index (`Igr`).
    *   Volume of Shale (`Vsh`) using various models (Linear, Larionov, Steiber, Clavier).
    *   Shale-corrected logs (e.g., `RHOB_ns`, `NPHI_ns`).
    *   Porosity (Density, Neutron, and effective porosity).
    *   Water Saturation (`Sw`) using Archie's and Simandoux's equations.
    *   Permeability using Timur and Coates models.
    *   Net Reservoir and Net Pay thickness.
*   **Lithology Estimation**:
    *   Simple two-mineral (Sand-Shale) model.
    *   Advanced multi-mineral model (e.g., Sand, Shale, Calcite, Heavy Minerals) using `scipy.optimize`.
*   **Visualization**: Generates crossplots like the Density-Neutron plot to aid in interpretation.

## Project Structure

```
src/
├── data/
│   ├── cleaning_log.py
│   └── read.py
├── petrophysics/
│   ├── __init__.py
│   ├── net_pay.py
│   ├── permeability.py
│   ├── porosity.py
│   ├── shale_volume.py
│   └── sw.py
├── visualization/
│   ├── __init__.py
│   └── crossplots.py
├── lithology.py
└── petrophysics.py
```

*   `data/`: Modules for reading and cleaning well log data.
*   `petrophysics/`: Contains individual modules for calculating shale volume, porosity, water saturation, permeability, and net pay.
*   `visualization/`: Modules for creating plots and crossplots.
*   `lithology.py`: Implements two-mineral and multi-mineral lithology models.
*   `petrophysics.py`: The core of the project, providing a refactored and robust petrophysical analysis pipeline that integrates functionalities from other modules.

## Getting Started

### Prerequisites

Make sure you have the following Python libraries installed:

*   pandas
*   numpy
*   matplotlib
*   lasio
*   scipy

You can install them using pip:
```bash
pip install pandas numpy matplotlib lasio scipy
```

### Example Usage

Here is a basic workflow for analyzing a well log file.

#### 1. Load and Clean Data

First, read your well log file and perform basic cleaning.

```python
import pandas as pd
from src.data.read import read_log
from src.data.cleaning_log import cleaning_log

df = read_log(file_path='path/to/your/data', file_name='well.las')

# Define the logs you need for the analysis
valid_logs = ['DEPTH', 'GR', 'RHOB', 'NPHI', 'RDEEP']
df_cleaned = cleaning_log(df, valid_logs=valid_logs)
```

#### 2. Run the Petrophysics Pipeline

The `run_petrophysics_pipeline` function from `src.petrophysics` provides a comprehensive, one-stop solution for petrophysical analysis. It calculates shale volume, porosity, water saturation, and more, in the correct dependency order.

```python
from src.petrophysics import run_petrophysics_pipeline

# Run the full pipeline with default parameters
df_petro = run_petrophysics_pipeline(df_cleaned)

# You can also customize parameters
custom_params = {
    'vsh_method': 'steiber',
    'rho_matrix': 2.71, # Limestone
    'phi_cut': 0.10
}
df_petro_custom = run_petrophysics_pipeline(df_cleaned, params=custom_params)

print(df_petro.columns)
```

#### 3. Perform Multi-mineral Analysis

For a more detailed lithological breakdown, use the `multimineral_split` function from `src.lithology`.

```python
from src.lithology import multimineral_split

df_litho = multimineral_split(df_petro, rhob_col='RHOB', nphi_col='NPHI')

print(df_litho[['mm_sand', 'mm_shale', 'mm_calcite', 'mm_heavy']].head())
```