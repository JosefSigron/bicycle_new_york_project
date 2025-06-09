# Parquet File Viewer Utility

This utility provides quick inspection and exploration capabilities for Parquet files used throughout the NYC bicycle analysis project.

## Overview

The `view_parquet.py` script is a lightweight tool for examining the structure, content, and basic statistics of Parquet files without loading them entirely into memory. It's particularly useful for exploring large datasets and understanding data schemas before running more intensive analyses.

## Features

### Data Inspection
- **Schema Exploration**: Display column names, data types, and basic metadata
- **Sample Data**: View first/last rows to understand data structure
- **Statistical Summary**: Basic descriptive statistics for numerical columns
- **Data Quality**: Identify missing values, data ranges, and potential issues

### Memory Efficient
- **Chunked Reading**: Processes large files without loading everything into memory
- **Selective Loading**: Can examine specific columns or row ranges
- **Quick Preview**: Fast overview without full data processing

### File Information
- **File Size**: Display file size and compression information
- **Row Count**: Total number of records in the dataset
- **Metadata**: Parquet-specific metadata and schema information

## Usage Examples

### Basic File Exploration
```python
from view_parquet import explore_parquet

# Quick overview of a parquet file
explore_parquet('data/combined/2023_combined_citibike_weather.parquet')
```

### Column Analysis
```python
# Examine specific columns
explore_parquet('data/file.parquet', columns=['temperature', 'trip_duration'])
```

### Sample Data
```python
# View first and last rows
explore_parquet('data/file.parquet', head=10, tail=5)
```

## Key Functions

### File Inspection
**Schema Display**: Shows column names, data types, and null counts for quick data understanding.

**Data Preview**: Displays sample rows from the beginning and end of the dataset.

**Statistics Generation**: Computes basic descriptive statistics for numerical columns.

### Performance Monitoring
**File Size Reporting**: Shows compressed and uncompressed file sizes.

**Read Performance**: Measures and reports file reading performance.

**Memory Usage**: Monitors memory consumption during data exploration.

## Applications

### Data Validation
- Verify data integrity after processing steps
- Check column names and data types before analysis
- Identify potential data quality issues

### Exploratory Analysis
- Quick data overview before detailed analysis
- Understanding data distributions and ranges
- Identifying interesting patterns or anomalies

### Development Support
- Debug data processing pipelines
- Validate output from data transformation scripts
- Support development of analysis workflows

## Dependencies

- **pandas**: Data reading and basic analysis
- **pyarrow**: Parquet file handling and metadata extraction
- **numpy**: Numerical computations for statistics

## Typical Output

### File Information
```
File: 2023_combined_citibike_weather.parquet
Size: 1.4 GB
Rows: 35,107,029
Columns: 37
```

### Schema Overview
```
Column                    Type        Non-Null Count
trip_duration            int64       35,107,029
start_time               datetime64  35,107,029
temperature              float64     35,106,892
utci                     float64     35,105,234
```

### Data Preview
```
   trip_duration              start_time  temperature    utci
0           420  2023-01-01 00:01:23      -2.8        -8.3
1           360  2023-01-01 00:03:45      -2.8        -8.3
...
```

## Use Cases

### Quality Assurance
- Verify data processing results
- Check for expected column presence and types
- Identify data anomalies or corruption

### Performance Optimization
- Understand data characteristics for query optimization
- Identify large columns that might affect performance
- Plan memory allocation for analysis scripts

### Data Documentation
- Generate quick data summaries for documentation
- Understand data schema evolution over time
- Support data dictionary creation 