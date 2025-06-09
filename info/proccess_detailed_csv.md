# Process Detailed CSV (`proccess_detailed_csv.py`)

## Overview

This utility script combines individual weather CSV files from NYC boroughs (Manhattan, Bronx, Brooklyn, Queens) into consolidated CSV files. It's designed to merge detailed weather data that has been organized by borough into manageable combined datasets for analysis.

## Key Features

### Multi-Borough Processing
- **Four NYC Boroughs**: Manhattan, Bronx, Brooklyn, Queens
- **Batch Processing**: Processes all CSV files in each borough directory
- **Consolidated Output**: Creates one combined file per borough

### File Management
- **Automatic Directory Creation**: Creates output directories as needed
- **Progress Tracking**: Reports processing status for each borough and file
- **Error Handling**: Gracefully handles missing directories or files

## Core Functionality

### Main Function (`combine_csv_files`)

**Purpose**: Combines all CSV files within each borough subdirectory into a single CSV file per borough.

**Directory Structure Expected**:
```
data/weather/csv/detailed_weather_nyc/
├── manhatten/           # Note: Original spelling variation
│   ├── file1.csv
│   ├── file2.csv
│   └── ...
├── bronx/
│   ├── file1.csv
│   └── ...
├── brooklyn/
│   ├── file1.csv
│   └── ...
└── queens/
    ├── file1.csv
    └── ...
```

### Processing Workflow

```python
# For each borough:
1. Scan directory for CSV files
2. Load each CSV file with low_memory=False
3. Concatenate all DataFrames for that borough
4. Save combined result as single CSV file
5. Report statistics (file count, row count)
```

### Spelling Correction
- **Manhattan Handling**: Automatically corrects "manhatten" to "manhattan" in output filename
- **Consistent Naming**: Ensures standardized borough names in output files

## Input Requirements

### Directory Structure
- **Base Directory**: `data/weather/csv/detailed_weather_nyc/`
- **Borough Subdirectories**: `manhatten/`, `bronx/`, `brooklyn/`, `queens/`
- **File Format**: CSV files with any naming convention

### Data Format
- **Standard CSV**: Any valid CSV format accepted
- **Memory Optimization**: Uses `low_memory=False` to avoid dtype warnings
- **Flexible Schema**: No specific column requirements

## Output Structure

### Generated Files
```
data/weather/csv/
├── combined_weather_manhattan.csv
├── combined_weather_bronx.csv
├── combined_weather_brooklyn.csv
└── combined_weather_queens.csv
```

### Processing Summary
```
==================================================
SUMMARY:
Total files processed: 124
Boroughs processed: 4
Output files created:
  - combined_weather_manhattan.csv
  - combined_weather_bronx.csv
  - combined_weather_brooklyn.csv
  - combined_weather_queens.csv
```

## Usage Example

```python
from proccess_detailed_csv import combine_csv_files

# Run the combination process
combine_csv_files()
```

**Command Line Usage**:
```bash
python src/proccess_detailed_csv.py
```

## Error Handling

### Missing Directories
- **Graceful Skip**: Continues processing if a borough directory is missing
- **Warning Messages**: Reports missing directories without stopping execution
- **Partial Success**: Processes available boroughs even if some are missing

### File Processing Errors
- **Exception Handling**: Catches and reports file-specific errors
- **Continue Processing**: Doesn't stop on individual file failures
- **Error Reporting**: Provides specific error messages for debugging

## Performance Characteristics

### Memory Usage
- **Moderate Memory**: Loads all files for one borough into memory simultaneously
- **Sequential Processing**: Processes one borough at a time to manage memory
- **Cleanup**: Clears data between borough processing

### Processing Speed
- **I/O Bound**: Speed depends primarily on disk read/write performance
- **Scalable**: Handles varying numbers of files per borough
- **Progress Feedback**: Real-time status updates during processing

## Key Statistics Reported

1. **File Count**: Number of CSV files found per borough
2. **Row Count**: Total rows in combined dataset per borough  
3. **Processing Status**: Success/failure for each borough
4. **Date Range**: Automatic detection of data date ranges
5. **Output Locations**: Full paths to generated combined files

## Dependencies

- **pandas**: DataFrame operations and CSV I/O
- **glob**: File pattern matching and directory scanning
- **os**: Operating system interface and directory operations

## Common Use Cases

### Weather Data Consolidation
- **Multi-Station Data**: Combining readings from multiple weather stations per borough
- **Temporal Continuity**: Merging time series data from different sources
- **Quality Control**: Consolidating validated weather measurements

### Data Pipeline Integration
- **Preprocessing Step**: Prepares data for further analysis scripts
- **Format Standardization**: Creates consistent file structure across boroughs
- **Batch Processing**: Enables automated data preparation workflows

## File Naming Convention

### Input Files
- **Flexible Naming**: Accepts any CSV filename within borough directories
- **No Format Requirements**: Works with various naming schemes
- **Extension Based**: Identifies files by .csv extension

### Output Files
- **Standardized Format**: `combined_weather_{borough}.csv`
- **Lowercase Borough Names**: Consistent casing for all outputs
- **Manhattan Correction**: Handles spelling variation in source directory

## Best Practices

1. **Data Backup**: Ensure source files are backed up before processing
2. **Disk Space**: Verify sufficient space for combined output files
3. **Data Validation**: Review combined files for completeness
4. **Regular Processing**: Run after new source data becomes available

## Limitations

- **Memory Constraints**: Large borough datasets may require more memory
- **Same Schema Assumption**: Assumes all files within a borough have compatible schemas
- **No Deduplication**: Does not remove duplicate records across files
- **Simple Concatenation**: Performs basic row concatenation without data validation 