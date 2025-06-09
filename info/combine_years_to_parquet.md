# Multi-Year Data Consolidation

This module consolidates Citibike data from multiple years into unified Parquet files for efficient storage and analysis.

## Overview

The `combine_years_to_parquet.py` script combines individual CSV files from multiple years of Citibike data into consolidated Parquet files. This process significantly improves data access performance, reduces storage requirements, and standardizes data formats across different years.

## Key Features

### Data Consolidation
- **Multi-Year Processing**: Combines data from 2019-2024 into single files
- **Format Standardization**: Ensures consistent column names and data types across years
- **Memory Efficient**: Processes large datasets without memory overflow
- **Error Handling**: Robust processing of files with different schemas

### Performance Optimization
- **Parquet Format**: Uses columnar storage for faster analytical queries
- **Compression**: Significantly reduces file sizes compared to CSV
- **Schema Validation**: Ensures data consistency across years
- **Chunked Processing**: Handles large files efficiently

### Data Quality
- **Column Mapping**: Maps varying column names to standard schema
- **Data Type Conversion**: Ensures consistent data types across years
- **Missing Data Handling**: Manages columns that don't exist in all years
- **Validation**: Verifies data integrity during consolidation

## Processing Steps

1. **Schema Analysis**: Examines all input files to understand column variations
2. **Column Standardization**: Maps different column names to consistent schema
3. **Data Type Harmonization**: Ensures consistent data types across years
4. **Consolidation**: Combines data while preserving temporal information
5. **Optimization**: Applies compression and indexing for performance
6. **Validation**: Verifies output data integrity and completeness

## Input Requirements

### Expected File Structure
```
data/citibike/
├── 2019/
│   ├── 201901-citibike-tripdata.csv
│   ├── 201902-citibike-tripdata.csv
│   └── ...
├── 2020/
│   ├── 202001-citibike-tripdata.csv
│   └── ...
```

### Column Mapping
- Handles variations in column names across years
- Maps legacy column names to current standards
- Manages addition/removal of columns over time

## Output Format

### Consolidated Parquet Files
```
data/citibike/
├── 2019_citibike_data.parquet
├── 2020_citibike_data.parquet
├── 2021_citibike_data.parquet
└── ...
```

### Performance Benefits
- **File Size**: 60-80% reduction compared to CSV
- **Read Speed**: 10-20x faster query performance
- **Memory Usage**: Reduced memory footprint for analysis
- **Schema**: Embedded schema information

## Usage Example

```python
from combine_years_to_parquet import consolidate_citibike_data

# Consolidate all years
consolidate_citibike_data(input_dir='data/citibike/csv/', 
                         output_dir='data/citibike/')

# Process specific years
consolidate_citibike_data(years=[2022, 2023, 2024])
```

## Dependencies

- **pandas**: Data manipulation and CSV reading
- **pyarrow**: Parquet file creation and optimization
- **numpy**: Data type handling and numerical operations
- **pathlib**: File system navigation
- **tqdm**: Progress tracking for long operations

## Applications

### Analysis Enablement
- Provides foundation for all subsequent analyses
- Enables cross-year comparisons and trend analysis
- Supports large-scale statistical analysis

### Performance Improvement
- Dramatically reduces data loading times
- Enables interactive analysis of large datasets
- Supports efficient query operations

### Storage Optimization
- Reduces storage requirements significantly
- Improves backup and transfer efficiency
- Maintains data fidelity while compressing 