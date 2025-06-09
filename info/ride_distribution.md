# Ride Distribution Analysis (`ride_distribution.py`)

## Overview

This script performs comprehensive analysis of NYC Citibike ridership patterns, generating statistical summaries and visualizations of ride distributions across time periods. It processes large parquet files to create insights into daily ridership patterns, hourly usage trends, and year-over-year ridership changes.

## Key Features

### Temporal Analysis
- **Daily Ridership Patterns**: Analysis of day-to-day ride volume variations
- **Hourly Distribution**: 24-hour usage patterns showing peak and off-peak hours
- **Yearly Comparisons**: Multi-year trend analysis and growth patterns
- **Statistical Summaries**: Comprehensive metrics for each year's ridership data

### Visualization Components
- **Time Series Plots**: Daily ridership trends across multiple years
- **Hourly Bar Charts**: Average ridership by hour of day with value labels
- **Statistical Overlays**: Mean and median lines on time series plots
- **Comparative Analysis**: Side-by-side yearly comparisons

## Core Functionality

### Data Loading and Processing (`load_and_process_data`)

**Purpose**: Efficiently loads and processes multiple years of Citibike data.

**Processing Pipeline**:
```python
# For each parquet file:
1. Extract year from filename
2. Load only necessary columns (start_time) for memory efficiency
3. Extract date and hour from timestamp
4. Calculate daily ride counts
5. Filter data to correct year
6. Generate hourly aggregations
7. Store results for analysis
```

**Memory Optimization**:
- Loads only required columns to reduce memory usage
- Processes files individually to prevent memory overflow
- Uses efficient datetime operations for temporal feature extraction

### Statistical Analysis (`calculate_stats`)

**Purpose**: Generates comprehensive statistical summaries for each year.

**Metrics Calculated**:
- **Median Daily Rides**: Robust measure of central tendency
- **Mean Daily Rides**: Average daily ridership
- **Maximum Daily Rides**: Peak single-day ridership
- **Minimum Daily Rides**: Lowest single-day ridership  
- **Total Annual Rides**: Sum of all rides for the year

**Output Format**:
```python
stats_data = {
    'year': '2023',
    'median_daily_rides': 45832,
    'mean_daily_rides': 46127,
    'max_daily_rides': 89234,
    'min_daily_rides': 8943,
    'total_rides': 16836405
}
```

## Visualization Functions

### 1. Hourly Distribution Analysis (`plot_hourly_distribution`)

**Purpose**: Shows average ridership patterns throughout the day.

**Features**:
- Bar chart with 24-hour timeline (00:00-23:00)
- Value labels on top of each bar showing exact ride counts
- Clear peak identification for rush hours
- Professional styling with grid lines and proper formatting

**Insights Revealed**:
- Morning rush hour peaks (typically 7-9 AM)
- Evening rush hour peaks (typically 5-7 PM)
- Low-usage overnight hours
- Weekend vs weekday patterns (when data is segmented)

### 2. Daily Ridership Analysis (`plot_daily_rides`)

**Purpose**: Creates comprehensive daily ridership visualizations.

**Two-Part Analysis**:

**Combined Multi-Year Plot**:
- All years overlaid on single plot with different colors
- Shows seasonal patterns and year-over-year trends
- Enables easy identification of growth/decline patterns
- Includes legend for year identification

**Individual Year Analysis**:
- Separate detailed plot for each year
- Mean and median reference lines
- Statistical annotations showing key metrics
- Clear identification of outliers and unusual patterns

### 3. Comparative Statistics (`plot_rides_per_bike`)

**Purpose**: Creates summary statistics comparison across years.

**Analysis Components**:
- Bar charts comparing key metrics across years
- Growth rate calculations and trend identification
- Statistical tables with formatted numbers
- Professional presentation suitable for reporting

## Output Structure

### Directory Organization
```
results/ride_distribution/
├── plots/
│   ├── hourly_ride_distribution.png
│   ├── daily_rides_by_year.png
│   ├── daily_rides_2019.png
│   ├── daily_rides_2020.png
│   ├── daily_rides_2021.png
│   └── ...
└── statistics/
    ├── yearly_ride_statistics.csv
    └── summary_report.txt
```

### Statistical Output Files
- **Yearly Statistics CSV**: Comprehensive metrics for all years
- **Summary Reports**: Text-based summaries with key findings
- **Data Tables**: Formatted statistical tables for presentation

## Key Analytical Insights

### Usage Patterns
- **Peak Hours**: Identification of highest-usage time periods
- **Seasonal Variations**: How ridership changes throughout the year
- **Weather Impacts**: Visible effects of weather on ridership (when combined with weather data)
- **Growth Trends**: Year-over-year ridership evolution

### Statistical Characteristics
- **Distribution Shape**: Understanding of ridership variability
- **Outlier Analysis**: Identification of unusual ridership days
- **Trend Analysis**: Long-term growth or decline patterns
- **Seasonal Amplitude**: Magnitude of seasonal ridership changes

### System Performance Metrics
- **Daily Capacity**: Understanding of system utilization levels
- **Peak Demand**: Maximum system stress periods
- **Off-Peak Efficiency**: Low-usage period characteristics
- **Growth Sustainability**: Analysis of ridership growth patterns

## Technical Implementation

### Memory Management
- **Column Selection**: Loads only necessary columns to reduce memory usage
- **File-by-File Processing**: Processes one year at a time to manage memory
- **Efficient Data Types**: Uses appropriate data types for temporal data
- **Garbage Collection**: Clears intermediate data structures

### Performance Optimization
- **Vectorized Operations**: Uses pandas vectorized operations for speed
- **Efficient Grouping**: Optimized groupby operations for aggregations
- **Smart Indexing**: Uses datetime indexing for temporal operations
- **Batch Processing**: Processes multiple files in systematic batches

### Error Handling
- **File Validation**: Checks for file existence and accessibility
- **Data Validation**: Validates date ranges and data integrity
- **Graceful Degradation**: Continues processing if individual files fail
- **Progress Reporting**: Provides status updates during long processing

## Input Requirements

### File Structure
- **Data Directory**: `data/citibike/combined/`
- **File Format**: Parquet files with year prefix (e.g., `2023_combined.parquet`)
- **Required Columns**: `start_time` (datetime of trip start)

### Data Quality Expectations
- **Complete Timestamps**: Valid datetime values for all records
- **Consistent Format**: Standardized datetime format across files
- **Reasonable Date Ranges**: Dates within expected years

## Usage Examples

### Basic Analysis
```python
from ride_distribution import main

# Run complete analysis
main()
```

### Custom Analysis
```python
# Load and process specific data directory
yearly_data, hourly_data = load_and_process_data('custom/data/path')

# Calculate statistics
stats, stats_df = calculate_stats(yearly_data)

# Generate specific visualizations
plot_hourly_distribution(hourly_data, output_dir)
plot_daily_rides(stats, output_dir)
```

## Dependencies

- **pandas**: Data manipulation and temporal analysis
- **matplotlib**: Core plotting and visualization
- **seaborn**: Statistical visualization enhancements
- **numpy**: Numerical operations and statistics
- **pathlib**: File system operations and path management

## Applications

### System Planning
- **Capacity Planning**: Understanding peak demand for infrastructure planning
- **Service Optimization**: Identifying optimal service hours and coverage
- **Resource Allocation**: Data-driven bike and dock distribution

### Research and Analysis
- **Urban Mobility Studies**: Understanding cycling patterns in urban environments
- **Transportation Planning**: Integration with broader transportation analysis
- **Policy Evaluation**: Assessing impact of policy changes on ridership

### Business Intelligence
- **Performance Monitoring**: Tracking system usage and growth
- **Trend Analysis**: Identifying long-term patterns and projections
- **Operational Insights**: Understanding system efficiency and utilization

## Customization Options

### Analysis Parameters
- **Time Periods**: Adjustable analysis windows and date ranges
- **Aggregation Levels**: Custom temporal aggregation (daily, weekly, monthly)
- **Statistical Metrics**: Additional or alternative statistical measures
- **Filtering Options**: Custom filters for specific analysis needs

### Visualization Options
- **Chart Types**: Alternative visualization styles and formats
- **Color Schemes**: Customizable color palettes for different purposes
- **Resolution Settings**: Adjustable output quality and size
- **Annotation Options**: Custom labels and statistical annotations 