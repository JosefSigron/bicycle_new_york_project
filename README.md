# Bicycle New York Project

This repository contains a data analysis project that examines the relationship between weather conditions and Citibike ridership in New York City. The project processes and analyzes Citibike ride data along with weather data to understand patterns and correlations.

## Project Overview

This project aims to:
- Analyze Citibike ridership patterns across different years
- Process and analyze weather data for New York City
- Explore the impact of weather conditions on bicycle ridership
- Create visualizations of ride distributions and weather impacts

## Dependencies

The project requires the following Python packages:
```
numpy>=1.21.0
pyarrow>=7.0.0
pandas>=1.5.0
dask>=2023.3.0
distributed>=2023.3.0
bokeh>=2.4.0  # For Dask dashboard
dask-cuda>=23.12.0  # For GPU acceleration with Dask 
```

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Directory Structure

- `src/`: Python scripts for data processing and analysis
- `data/`: Contains raw and processed data files
- `results/`: Output directory for analysis results and visualizations
- `word_docs/`: Documentation files

## Source Files

### `src/ride_distribution.py`
Analyzes the distribution of Citibike rides across different time periods. Creates visualizations showing daily ride counts, hourly patterns, and yearly trends. Outputs statistics on ride patterns to the results directory.

### `src/preprocess_weather.py`
Processes raw weather data, including parsing sky cover codes, calculating mean radiant temperature (MRT), and computing the Universal Thermal Climate Index (UTCI). Implements weather categorization based on UTCI values and precipitation data.

### `src/weather_distribution.py`
Analyzes the distribution of weather conditions in New York City. Creates visualizations of temperature, humidity, wind speed, and other weather metrics over time.

### `src/weather_impact_analysis.py`
Examines the relationship between weather conditions and Citibike ridership. Analyzes how factors like temperature, precipitation, and thermal comfort affect ride counts.

### `src/ride_weather_distribution.py`
Combines ride data with weather data to create joint distributions and correlation analyses. Visualizes how ridership varies with different weather conditions.

### `src/combine_weather_citibike.py`
Merges processed weather data with Citibike ride data to create a unified dataset for analysis.

### `src/combine_years_to_parquet.py`
Combines Citibike data from multiple years into consolidated Parquet files for efficient storage and analysis.

### `src/view_parquet.py`
Utility script to quickly view and explore the contents of Parquet files.

### `src/proccess_detailed_csv.py`
Processes detailed CSV data files with additional metrics and converts them to a standardized format.

### `src/psv_to_csv.py`
Converts pipe-separated value (PSV) files to CSV format for easier processing.

### `src/dly_to_csv.py`
Converts weather data in DLY format to CSV format for analysis.

### `src/read_dly.py`
Utility functions for reading and parsing weather data in DLY format.

## UTCI Calculation

The project uses the Universal Thermal Climate Index (UTCI) to assess thermal comfort. The calculation methodology is documented in `UTCI_calculation.md`, which explains the simplified approach for calculating Mean Radiant Temperature (MRT) used in the weather analysis.

## Results

Analysis results, including visualizations and data summaries, are stored in the `results/` directory, organized by analysis type. 