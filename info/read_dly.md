# GHCN DLY File Reader (`read_dly.py`)

## Overview

This module provides the `DLYReader` class for parsing GHCN (Global Historical Climatology Network) daily weather data files. It converts the fixed-width DLY format used by NOAA into structured pandas DataFrames for analysis, handling the complex formatting and quality flags inherent in climate data.

## Key Features

### GHCN Format Support
- **Fixed-Width Parsing**: Handles GHCN's complex fixed-width format specification
- **Quality Flag Processing**: Preserves data quality indicators for each measurement
- **Missing Data Handling**: Properly converts missing value indicators (-9999) to NaN
- **Multi-Index Structure**: Creates hierarchical index for efficient data access

### Data Processing Capabilities
- **Weather Element Parsing**: Supports all GHCN weather elements (TMAX, TMIN, PRCP, etc.)
- **Temporal Organization**: Organizes data by station, element, year, month, and day
- **Value Validation**: Handles malformed lines and invalid data gracefully
- **Memory Efficient**: Processes large files without excessive memory usage

## DLYReader Class

### Class Initialization
```python
class DLYReader:
    def __init__(self, file_path: str):
        """
        Initialize the DLY reader with a file path.
        
        Args:
            file_path (str): Path to the .dly file
        """
```

### Core Methods

#### `read()` - Main Processing Method
**Purpose**: Reads and parses the entire DLY file into a pandas DataFrame.

**Returns**: Multi-indexed DataFrame with weather data

**Processing Steps**:
1. Validate file existence
2. Process each line of the DLY file
3. Extract station, element, year, and month information
4. Parse daily values for the month (up to 31 days)
5. Handle quality flags and missing data
6. Create structured DataFrame with multi-index

#### `_parse_element()` - Line Parsing Method
**Purpose**: Parses individual lines from the DLY file.

**Line Format Structure**:
```
Positions 1-11:   Station ID
Positions 12-15:  Year
Positions 16-17:  Month
Positions 18-21:  Element (TMAX, TMIN, PRCP, etc.)
Positions 22-end: Daily values (8 characters per day)
```

**Daily Value Structure** (8 characters per day):
- Positions 1-5: Value (integer, tenths of units)
- Position 6: Measurement flag
- Position 7: Quality flag
- Position 8: Source flag

## GHCN DLY Format Specification

### Weather Elements
Common weather elements found in DLY files:

- **TMAX**: Maximum temperature (tenths of degrees C)
- **TMIN**: Minimum temperature (tenths of degrees C)
- **PRCP**: Precipitation (tenths of mm)
- **SNOW**: Snowfall (mm)
- **SNWD**: Snow depth (mm)
- **AWND**: Average wind speed (tenths of m/s)
- **WSF2**: Fastest 2-minute wind speed (tenths of m/s)
- **WT01-WT22**: Weather type indicators (fog, rain, snow, etc.)

### Quality Flags
- **Blank**: No quality control applied
- **D**: Failed duplicate check
- **G**: Failed gap check
- **I**: Failed internal consistency check
- **K**: Failed streak/frequent-value check
- **L**: Failed length check
- **M**: Failed mega-consistency check
- **N**: Failed naught check
- **O**: Failed climatological outlier check
- **R**: Failed lagged range check
- **S**: Failed spatial consistency check
- **T**: Failed temporal consistency check
- **W**: Temperature too warm for snow
- **X**: Failed bounds check
- **Z**: Flagged as a result of an official Datzilla investigation

### Missing Data Handling
```python
# Convert GHCN missing value indicator to NaN
if value == -9999:
    value = np.nan
```

## Output Data Structure

### Multi-Index DataFrame
The parsed data is organized with a hierarchical index:

```python
Index Levels:
0. station_id: Weather station identifier
1. element: Weather measurement type  
2. year: Year of measurement
3. month: Month of measurement
4. day: Day of measurement

Columns:
- value: Numerical measurement value
- quality_flag: Data quality indicator
```

### Example DataFrame Structure
```
                                          value quality_flag
station_id   element year month day                        
USW00094728  TMAX    2019 1     1       56                
                           1     2       67                
                           1     3       45                
             TMIN    2019 1     1       39                
                           1     2       41                
             PRCP    2019 1     1        0                
                           1     2        0                
```

## Usage Examples

### Basic Usage
```python
from read_dly import DLYReader

# Initialize reader with DLY file
reader = DLYReader("USW00094728.dly")

# Read and parse the file
df = reader.read()

# Access specific data
print(df.head())
print(f"Data shape: {df.shape}")
```

### Command Line Usage
```python
# Example command line usage included in script
python src/read_dly.py path/to/file.dly
```

### Data Access Examples
```python
# Access temperature data for specific station and year
station_data = df.loc['USW00094728', 'TMAX', 2019]

# Get all precipitation data
precip_data = df.xs('PRCP', level='element')

# Filter by quality (only high-quality data)
quality_data = df[df['quality_flag'] == ' ']
```

## Error Handling

### File Validation
- **File Existence**: Checks if DLY file exists before processing
- **Path Validation**: Validates file path accessibility
- **Format Verification**: Basic validation of DLY file structure

### Line Processing
- **Malformed Lines**: Handles lines that don't match expected format
- **Invalid Values**: Processes non-numeric values gracefully
- **Index Errors**: Manages lines with unexpected length or structure

### Warning System
```python
# Example warning for problematic lines
print(f"Warning: Skipping malformed line: {line.strip()}")
```

## NYC Central Park Example

### Specific Use Case
The file `USW00094728.dly` represents NYC Central Park weather station:

```python
# Process NYC Central Park data
reader = DLYReader("data/weather/USW00094728.dly")
nyc_weather = reader.read()

# Extract temperature data
temp_data = nyc_weather.xs('TMAX', level='element')
print(f"NYC temperature records: {len(temp_data)}")
```

### Data Content
- **Station**: USW00094728 (NYC Central Park)
- **Location**: Central Park, Manhattan, NYC
- **Elements**: Temperature, precipitation, wind, snow measurements
- **Coverage**: Typically decades of daily weather measurements

## Performance Characteristics

### Memory Usage
- **Line-by-Line Processing**: Reads file incrementally to manage memory
- **Efficient Data Types**: Uses appropriate pandas data types
- **Index Optimization**: Multi-index structure for fast data access

### Processing Speed
- **File Size Dependent**: Larger DLY files take proportionally longer
- **I/O Bound**: Performance limited by disk read speed
- **Parsing Overhead**: Fixed-width parsing adds computational cost

### Scalability
- **Large Files**: Can handle multi-decade weather station files
- **Memory Efficient**: Doesn't load entire file into memory at once
- **DataFrame Size**: Output size scales with temporal coverage

## Dependencies

- **pandas**: DataFrame operations and multi-indexing
- **numpy**: Numerical operations and NaN handling
- **typing**: Type hints for better code documentation
- **os**: File system operations

## Integration with Analysis Pipeline

### Workflow Position
```
1. Download DLY files from NOAA →
2. Parse with read_dly.py →
3. Convert to CSV with dly_to_csv.py →
4. Process with preprocess_weather.py →
5. Integrate with Citibike data
```

### Common Use Cases
- **Weather Data Preparation**: First step in weather data processing
- **Quality Control**: Identify and handle data quality issues
- **Format Conversion**: Convert to more accessible formats
- **Data Exploration**: Understand available weather measurements

## Best Practices

### Data Validation
1. **Check Quality Flags**: Always consider data quality in analysis
2. **Handle Missing Data**: Properly account for NaN values
3. **Validate Date Ranges**: Ensure data covers expected time periods
4. **Cross-Reference Stations**: Verify station metadata accuracy

### Performance Optimization
1. **Filter Early**: Use multi-index slicing for subset analysis
2. **Batch Processing**: Process multiple files systematically
3. **Memory Monitoring**: Monitor memory usage with large files
4. **Index Usage**: Leverage multi-index for efficient queries

## Limitations

- **Format Specific**: Only works with GHCN DLY format files
- **Fixed Structure**: Assumes standard DLY file organization
- **Quality Flag Complexity**: Quality flags require domain knowledge to interpret
- **Missing Metadata**: Doesn't include station location or measurement unit information

## Related Resources

- **NOAA GHCN Documentation**: Official format specification
- **Weather Station Metadata**: Station location and description files
- **Quality Control Documentation**: Detailed explanation of quality flags
- **Data Access Tools**: NOAA's online data access interfaces 