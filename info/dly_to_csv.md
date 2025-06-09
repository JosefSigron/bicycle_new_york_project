# DLY to CSV Converter (`dly_to_csv.py`)

## Overview

This utility script converts GHCN (Global Historical Climatology Network) DLY format files to CSV format with optional year filtering. It's specifically designed to process daily weather station data from the NOAA climate database and convert it to a more accessible CSV format for analysis.

## Key Features

### Format Conversion
- **DLY to CSV**: Converts GHCN daily format files to standard CSV
- **Year Filtering**: Extracts data for specific year ranges (default: 2019-2024)
- **Data Preservation**: Maintains all data integrity during conversion
- **Structured Output**: Creates organized CSV files with proper headers

### Command Line Interface
- **Flexible Arguments**: Supports input/output file specification
- **Optional Parameters**: Configurable start and end years
- **Error Handling**: Comprehensive validation and error reporting
- **Progress Feedback**: Status updates during conversion process

## Core Functionality

### Main Conversion Function (`convert_dly_to_csv`)

**Purpose**: Performs the conversion from DLY to CSV format with year filtering.

**Parameters**:
- `input_file` (str): Path to input DLY file
- `output_file` (str): Path for output CSV file
- `start_year` (int): Start year for filtering (default: 2019)
- `end_year` (int): End year for filtering (default: 2024)

**Processing Steps**:
```python
1. Initialize DLY reader with input file
2. Read and parse DLY file into DataFrame
3. Reset index to make all columns accessible
4. Filter data for specified year range
5. Create output directory if needed
6. Write filtered data to CSV format
7. Return processed DataFrame for validation
```

### Year Filtering Logic
```python
# Filter for specific years (inclusive)
filtered_df = df[(df['year'] >= start_year) & (df['year'] <= end_year)]
```

## GHCN DLY Format Background

### File Format Description
- **GHCN Daily**: Global Historical Climatology Network daily weather data
- **Fixed-Width Format**: Structured text format with specific column positions
- **Weather Elements**: Temperature, precipitation, wind, pressure, etc.
- **Quality Flags**: Data quality indicators for each measurement

### Example DLY Data Elements
- **TMAX**: Maximum temperature
- **TMIN**: Minimum temperature  
- **PRCP**: Precipitation
- **SNOW**: Snowfall
- **SNWD**: Snow depth
- **AWND**: Average wind speed

## Integration with DLY Reader

### Dependency on `read_dly.py`
This script relies on the `DLYReader` class from `read_dly.py` for parsing DLY files:

```python
from read_dly import DLYReader

# Initialize reader and parse file
reader = DLYReader(input_file)
df = reader.read()
```

### Data Structure After Reading
- **Multi-Index DataFrame**: Indexed by station_id, element, year, month, day
- **Value Column**: Numerical weather measurements
- **Quality Flags**: Data quality indicators
- **Missing Data**: Properly handled with NaN values

## Usage Examples

### Command Line Usage
```bash
# Basic conversion with default years (2019-2024)
python src/dly_to_csv.py input_file.dly output_file.csv

# Custom year range
python src/dly_to_csv.py input_file.dly output_file.csv --start_year 2020 --end_year 2023

# Using only start year (goes to default end year)
python src/dly_to_csv.py input_file.dly output_file.csv --start_year 2021
```

### Programmatic Usage
```python
from dly_to_csv import convert_dly_to_csv

# Convert with default years
df = convert_dly_to_csv('USW00094728.dly', 'nyc_weather.csv')

# Convert with custom year range
df = convert_dly_to_csv(
    'station_data.dly', 
    'weather_subset.csv',
    start_year=2020,
    end_year=2022
)

print(f"Converted {len(df)} records")
```

## Command Line Arguments

### Required Arguments
- **input_file**: Path to the input DLY file (e.g., `USW00094728.dly`)
- **output_file**: Path for the output CSV file (e.g., `weather_data.csv`)

### Optional Arguments
- **--start_year**: Start year for filtering (default: 2019)
- **--end_year**: End year for filtering (default: 2024)

### Argument Parsing
```python
parser = argparse.ArgumentParser(description='Convert DLY file to CSV with year filtering.')
parser.add_argument('input_file', type=str, help='Path to the input DLY file')
parser.add_argument('output_file', type=str, help='Path to the output CSV file')
parser.add_argument('--start_year', type=int, default=2019, help='Start year for filtering (inclusive)')
parser.add_argument('--end_year', type=int, default=2024, help='End year for filtering (inclusive)')
```

## Output Structure

### CSV Format
The output CSV contains the following columns:
- **station_id**: Weather station identifier
- **element**: Weather measurement type (TMAX, TMIN, PRCP, etc.)
- **year**: Year of measurement
- **month**: Month of measurement (1-12)
- **day**: Day of measurement (1-31)
- **value**: Numerical measurement value
- **quality_flag**: Data quality indicator

### Example Output
```csv
station_id,element,year,month,day,value,quality_flag
USW00094728,TMAX,2019,1,1,56, 
USW00094728,TMAX,2019,1,2,67, 
USW00094728,TMIN,2019,1,1,39, 
USW00094728,PRCP,2019,1,1,0, 
```

## Error Handling

### File Validation
- **Input File Check**: Verifies DLY file exists before processing
- **Directory Creation**: Creates output directory structure as needed
- **Permission Validation**: Ensures read/write permissions for files

### Processing Errors
- **DLY Format Errors**: Handles malformed DLY files gracefully
- **Date Range Validation**: Validates year parameters are reasonable
- **Memory Management**: Handles large files without memory overflow

### Return Codes
- **Success (0)**: Conversion completed successfully
- **Error (1)**: Conversion failed due to errors
- **Progress Reporting**: Detailed status messages during processing

## Performance Characteristics

### Memory Usage
- **Efficient Processing**: Uses DLYReader's optimized parsing
- **Year Filtering**: Reduces memory by filtering early in process
- **DataFrame Operations**: Leverages pandas' efficient data handling

### Processing Speed
- **File Size Dependent**: Speed varies with input file size
- **Year Range Impact**: Smaller year ranges process faster
- **I/O Bound**: Performance limited by disk read/write speed

## Common Use Cases

### Weather Data Analysis Preparation
- **NYC Central Park Data**: Convert `USW00094728.dly` for NYC analysis
- **Multi-Station Processing**: Batch convert multiple station files
- **Time Series Analysis**: Prepare data for temporal analysis

### Data Pipeline Integration
- **Preprocessing Step**: Convert before combining with other datasets
- **Format Standardization**: Create CSV files for broader tool compatibility
- **Quality Control**: Filter to relevant time periods for analysis

## NYC Central Park Example

### Specific Use Case
The file `USW00094728.dly` represents NYC Central Park weather station:

```bash
# Convert NYC Central Park data for 2019-2024 analysis
python src/dly_to_csv.py data/weather/USW00094728.dly data/weather/csv/nyc_central_park.csv

# Result: CSV with daily weather measurements for NYC Central Park
```

### Data Content
- **Location**: Central Park, Manhattan, NYC
- **Elements**: Temperature, precipitation, wind, snow measurements
- **Time Range**: Filtered to analysis period (2019-2024)
- **Quality**: High-quality urban weather station data

## Dependencies

- **argparse**: Command line argument parsing
- **pandas**: DataFrame operations and CSV output
- **os**: Directory operations and file management
- **read_dly**: Custom DLY file reader (must be in same project)

## Integration with Analysis Pipeline

### Workflow Position
```
1. Download DLY files from NOAA → 
2. Convert to CSV using dly_to_csv.py →
3. Process with preprocess_weather.py →
4. Combine with Citibike data →
5. Analyze with various analysis scripts
```

### Related Scripts
- **read_dly.py**: Provides DLY parsing functionality
- **preprocess_weather.py**: Further processes the CSV output
- **combine_weather_citibike.py**: Integrates with ridership data

## Best Practices

1. **Backup Original Files**: Keep DLY files as authoritative source
2. **Validate Output**: Check CSV files for completeness and accuracy
3. **Year Range Selection**: Choose appropriate years for analysis scope
4. **File Organization**: Use consistent naming for processed files
5. **Documentation**: Record conversion parameters for reproducibility

## Limitations

- **DLY Format Dependency**: Requires properly formatted GHCN DLY files
- **Year-Based Filtering**: Only supports year-level filtering, not finer granularity
- **Single File Processing**: Processes one file at a time (no batch mode)
- **No Data Validation**: Doesn't validate weather data values for reasonableness 