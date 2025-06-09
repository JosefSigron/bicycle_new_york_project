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
- `info/`: Detailed documentation for each analysis script and module
- `word_docs/`: Project documentation files

## Source Files

This directory contains Python scripts for data processing and analysis. For detailed documentation of each file, see the `info/` directory.

### Data Processing Pipeline

#### `src/preprocess_weather.py`
Processes raw weather data from multiple NYC boroughs. Converts meteorological codes to standardized formats, calculates Mean Radiant Temperature (MRT), computes Universal Thermal Climate Index (UTCI), and categorizes weather conditions. Essential preprocessing step that creates weather datasets with thermal comfort metrics.

#### `src/combine_weather_citibike.py`
Merges processed weather data with Citibike ride data based on geographic location and temporal alignment. Performs spatial matching of rides to borough-specific weather conditions and creates unified datasets for analysis.

#### `src/combine_years_to_parquet.py`
Consolidates Citibike data from multiple years (2019-2024) into optimized Parquet files. Handles schema variations across years, standardizes column names and data types, and significantly improves storage efficiency and query performance.

### Analysis Scripts

#### `src/weekly_ridetime_analysis.py`
Analyzes median ride times by day of week across different weather and thermal comfort categories. Uses chunked processing to handle large datasets (146M+ rides) and creates comprehensive visualizations with statistical significance testing for different time periods.

#### `src/weather_impact_analysis.py`
Examines relationships between weather conditions and bicycle ridership patterns. Quantifies correlations, identifies optimal weather conditions for cycling, and analyzes how temperature, precipitation, and thermal comfort affect usage patterns.

#### `src/manhattan_analysis.py`
Provides specialized analysis of Manhattan's Citibike system. Divides Manhattan into regions (Downtown, Midtown, Central Park Area, Upper Manhattan) and examines trip patterns, station performance, user behavior, and temporal dynamics specific to the borough.

#### `src/ride_decrease_probability_analysis.py`
Analyzes the probability of ridership decreases under various weather conditions. Uses statistical modeling to understand and predict how different weather factors affect the likelihood of reduced cycling activity.

#### `src/yearly_comparison_analysis.py`
Conducts comprehensive multi-year comparisons of ridership patterns. Examines trends across 2019-2024, accounting for seasonal variations, growth patterns, and the impact of external factors like the COVID-19 pandemic.

### Distribution Analysis

#### `src/ride_distribution.py`
Analyzes temporal and spatial distributions of Citibike rides. Creates visualizations of daily, hourly, and seasonal patterns, examines station usage distributions, and identifies peak usage times and locations.

#### `src/weather_distribution.py`
Analyzes distributions of weather conditions across NYC. Examines temperature, humidity, wind, and precipitation patterns over time, creates seasonal weather profiles, and provides context for understanding weather-ridership relationships.

#### `src/ride_weather_distribution.py`
Creates joint distributions combining ride and weather data. Visualizes how ridership varies across different weather conditions, generates correlation matrices, and provides integrated views of weather-transportation relationships.

### Data Utilities

#### `src/view_parquet.py`
Lightweight utility for inspecting Parquet files without loading them entirely into memory. Displays schema information, basic statistics, and sample data for quick exploration of large datasets.

#### `src/closest_station.py`
Utility for finding the closest Citibike stations to given coordinates. Supports geographic analysis and helps in understanding service coverage and accessibility patterns.

### Data Format Conversion

#### `src/dly_to_csv.py`
Converts weather data from DLY (daily) format to CSV format. Handles meteorological data format conversion for integration with the analysis pipeline.

#### `src/read_dly.py`
Core functions for reading and parsing weather data in DLY format. Provides the foundation for weather data extraction and preprocessing.

#### `src/psv_to_csv.py`
Converts pipe-separated value (PSV) files to CSV format. Supports data integration from various sources with different delimiters.

#### `src/proccess_detailed_csv.py`
Processes detailed CSV files with additional metrics and converts them to standardized formats. Handles data enrichment and normalization tasks.

## UTCI Calculation

The project uses the Universal Thermal Climate Index (UTCI) to assess thermal comfort. The calculation methodology is documented in `UTCI_calculation.md`, which explains the simplified approach for calculating Mean Radiant Temperature (MRT) used in the weather analysis.

## Documentation

### Detailed Module Documentation
The `info/` directory contains comprehensive documentation for each script in the `src/` folder:

- **Processing Pipeline**: Detailed documentation for data preprocessing and integration modules
- **Analysis Scripts**: In-depth explanations of analysis methodologies and outputs  
- **Utility Functions**: Documentation for helper scripts and data conversion tools
- **Technical Details**: Implementation specifics, dependencies, and usage examples

Each documentation file follows a consistent format including overview, key functions, usage examples, dependencies, and applications.

### UTCI Calculation Methodology
The `UTCI_calculation.md` file documents the simplified approach for calculating Mean Radiant Temperature (MRT) used in thermal comfort analysis.

## Results

Analysis results, including visualizations and data summaries, are stored in the `results/` directory, organized by analysis type:

- `results/weekly_ridetime_analysis/`: Day-of-week ridership analysis by weather conditions
- `results/weather_impact_analysis/`: Weather correlation and impact studies  
- `results/manhattan_analysis/`: Manhattan-specific regional analysis
- `results/ride_distribution/`: Temporal and spatial ridership patterns
- Additional analysis outputs organized by script name 