# PSV to CSV Converter (`psv_to_csv.py`)

## Overview

A simple utility script that converts pipe-separated values (PSV) files to comma-separated values (CSV) format. This tool is particularly useful for processing weather station data and other datasets that use pipe (|) delimiters instead of the more common comma delimiter.

## Key Features

### Format Conversion
- **PSV to CSV**: Converts pipe-delimited files to comma-delimited format
- **Automatic Extension Handling**: Intelligently handles file extensions
- **Preserve Data Integrity**: Maintains all data while changing only the delimiter

### Command Line Interface
- **Simple Usage**: Single command with input and optional output file specification
- **Flexible Output**: Auto-generates output filename or accepts custom path
- **Error Handling**: Comprehensive error checking and user feedback

## Core Functionality

### Main Conversion Function (`psv_to_csv`)

**Purpose**: Performs the actual file conversion from PSV to CSV format.

**Parameters**:
- `input_file` (str): Path to the input PSV file
- `output_file` (str, optional): Path for output CSV file

**Conversion Logic**:
```python
# Read PSV with pipe delimiter
psv_reader = csv.reader(psv_file, delimiter='|')

# Write CSV with comma delimiter  
csv_writer = csv.writer(csv_file)

# Copy all rows maintaining data integrity
for row in psv_reader:
    csv_writer.writerow(row)
```

### Automatic File Naming
- **Extension Detection**: Checks for `.psv` extension in input file
- **Smart Replacement**: Replaces `.psv` with `.csv` for output filename
- **Fallback Strategy**: Appends `.csv` if input doesn't have `.psv` extension

## Usage Examples

### Command Line Usage
```bash
# Basic conversion (auto-generates output filename)
python src/psv_to_csv.py input_file.psv

# Custom output filename
python src/psv_to_csv.py input_file.psv output_file.csv

# Convert file without .psv extension
python src/psv_to_csv.py weather_data.txt weather_data.csv
```

### Programmatic Usage
```python
from psv_to_csv import psv_to_csv

# Auto-generate output filename
success = psv_to_csv('data.psv')

# Specify custom output
success = psv_to_csv('data.psv', 'converted_data.csv')

# Check conversion success
if success:
    print("Conversion completed successfully")
else:
    print("Conversion failed")
```

## File Extension Handling

### Input File Extensions
- **`.psv` files**: Automatically detected and handled
- **Other extensions**: Accepted but output naming differs
- **No extension**: Handled gracefully with appropriate output naming

### Output File Naming
```python
# Examples of automatic naming:
'weather.psv' → 'weather.csv'
'data.txt' → 'data.txt.csv'
'stations' → 'stations.csv'
```

## Error Handling

### File Validation
- **Input File Existence**: Checks if input file exists before processing
- **Path Validation**: Validates file paths and accessibility
- **Permission Checks**: Ensures read access to input and write access to output location

### Processing Errors
- **Exception Handling**: Catches and reports I/O errors during conversion
- **Graceful Failure**: Returns failure status without crashing
- **User Feedback**: Provides clear error messages for troubleshooting

### Return Values
- **Success**: Returns `True` when conversion completes successfully
- **Failure**: Returns `False` when errors occur during processing
- **Status Messages**: Prints informative messages about conversion progress

## Common Use Cases

### Weather Data Processing
- **Station Lists**: Converting pipe-delimited weather station inventories
- **Historical Data**: Processing legacy weather datasets in PSV format
- **Data Integration**: Preparing PSV files for analysis tools that expect CSV

### Data Pipeline Integration
- **Format Standardization**: Converting various delimited formats to CSV standard
- **Preprocessing Step**: Preparing data for pandas or other analysis libraries
- **Batch Processing**: Part of larger data transformation workflows

## Technical Implementation

### CSV Module Usage
- **Standard Library**: Uses Python's built-in `csv` module for reliability
- **Delimiter Specification**: Explicitly sets delimiters for both input and output
- **Row-by-Row Processing**: Processes files line by line for memory efficiency

### Memory Efficiency
- **Streaming Processing**: Doesn't load entire file into memory
- **Line-by-Line**: Processes one row at a time
- **Scalable**: Handles files of any size within system constraints

## Performance Characteristics

### Speed
- **Fast Processing**: Direct conversion without data transformation
- **I/O Bound**: Performance limited primarily by disk read/write speed
- **Linear Scaling**: Processing time scales linearly with file size

### Memory Usage
- **Low Memory**: Minimal memory footprint regardless of file size
- **Constant Usage**: Memory usage remains constant for files of any size
- **No Buffering**: Direct pipe from input to output

## Integration Examples

### Weather Data Workflow
```bash
# Convert weather station list
python src/psv_to_csv.py ghcnh-station-list.psv

# Use converted file in analysis
python src/closest_station.py  # Now can read CSV format
```

### Batch Processing
```python
import glob
from psv_to_csv import psv_to_csv

# Convert all PSV files in directory
psv_files = glob.glob('data/*.psv')
for psv_file in psv_files:
    success = psv_to_csv(psv_file)
    if success:
        print(f"Converted {psv_file}")
    else:
        print(f"Failed to convert {psv_file}")
```

## Dependencies

- **csv**: Python standard library module for CSV operations
- **sys**: Command line argument processing
- **os**: File system operations and path validation

## Best Practices

1. **File Backup**: Keep original PSV files as backup before conversion
2. **Output Verification**: Check converted files to ensure data integrity
3. **Path Handling**: Use absolute paths for important data files
4. **Error Checking**: Always check return value in programmatic usage

## Limitations

- **Simple Conversion**: Only changes delimiter, doesn't validate or transform data
- **No Data Cleaning**: Doesn't handle data quality issues or formatting problems
- **Single Delimiter**: Designed specifically for pipe-to-comma conversion
- **No Compression**: Doesn't handle compressed input or output files

## Related Tools

- **Standard CSV Tools**: Works alongside pandas, Excel, and other CSV-compatible tools
- **Data Pipeline**: Often used before `closest_station.py` and other analysis scripts
- **Format Converters**: Complementary to other file format conversion utilities 