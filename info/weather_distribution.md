# Weather Distribution Analysis (`weather_distribution.py`)

## Overview

This script performs comprehensive statistical analysis and visualization of NYC weather patterns, focusing on temperature distributions, seasonal variations, and thermal comfort (UTCI) categories. It generates detailed plots showing how weather conditions vary by time, season, and thermal stress levels.

## Key Features

### Temperature Analysis
- **Distribution Analysis**: Histograms and statistical summaries of temperature data
- **Seasonal Patterns**: Box plots and violin plots showing seasonal temperature variations
- **Yearly Trends**: Multi-year temperature trend analysis with regression lines
- **Monthly Patterns**: Detailed monthly temperature ranges and averages

### UTCI Thermal Comfort Analysis
- **Category Distribution**: Analysis of thermal stress levels throughout the day
- **Seasonal Breakdown**: How thermal comfort varies by season
- **Hourly Patterns**: 24-hour thermal comfort distribution patterns
- **Color-Coded Visualization**: Intuitive blue-to-red color scheme for cold-to-hot categories

## Data Processing Pipeline

### Input Data Requirements
- **File Location**: `data/weather/csv/nyc_weather_with_utci.csv`
- **Required Columns**: datetime, temperature, utci_cat
- **Time Period**: Typically covers 2019-2024 weather data

### Data Cleaning and Enhancement
```python
# Data preprocessing steps:
1. Load NYC weather data with UTCI categories
2. Drop rows with missing temperature values
3. Convert datetime to proper datetime type
4. Extract temporal components (year, month, day, hour)
5. Create season classifications
6. Generate derived analytical features
```

### Temporal Feature Engineering
```python
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'
```

## Visualization Components

### 1. Basic Temperature Analysis (Figure 1)
**Four-Panel Analysis**:
- **Temperature Distribution**: Histogram with KDE overlay showing overall temperature patterns
- **Monthly Temperature Box Plot**: Box plots for each month showing median, quartiles, and outliers
- **Seasonal Temperature Violin Plot**: Distribution shapes for each season
- **Yearly Temperature Trends**: Line plot showing average temperature trends over years

### 2. Detailed Temperature Analysis (Figure 2)
**Advanced Four-Panel Analysis**:
- **Monthly-Yearly Heatmap**: Temperature patterns across months and years
- **Temperature Range Analysis**: Monthly min/max ranges with average trend lines
- **Seasonal-Yearly Box Plots**: Temperature distributions by season across years
- **Daily Temperature Variation**: Scatter plot with polynomial trend line for sample year

### 3. UTCI Thermal Comfort Analysis (Figure 3)
**Seasonal UTCI Distribution**:
- **Four Seasonal Subplots**: One subplot per season (Winter, Spring, Summer, Fall)
- **Hourly Distribution**: 24-hour thermal comfort category patterns
- **Color-Coded Categories**: Blue (cold stress) to green (neutral) to red (heat stress)
- **Professional Styling**: Large format (30×24) with comprehensive legends

## UTCI Category Analysis

### Thermal Stress Categories
```python
temperature_order = [
    'Extreme cold stress',      # < -40°C UTCI
    'Very strong cold stress',  # -40 to -27°C UTCI  
    'Strong cold stress',       # -27 to -13°C UTCI
    'Moderate cold stress',     # -13 to 0°C UTCI
    'Slight cold stress',       # 0 to 9°C UTCI
    'No thermal stress',        # 9 to 26°C UTCI
    'Moderate heat stress',     # 26 to 32°C UTCI
    'Strong heat stress',       # 32 to 38°C UTCI
    'Very strong heat stress',  # 38 to 46°C UTCI
    'Extreme heat stress'       # > 46°C UTCI
]
```

### Color Mapping Strategy
```python
custom_colors = {
    'Extreme cold stress': '#0000FF',         # Dark blue
    'Very strong cold stress': '#4444FF',     # Blue
    'Strong cold stress': '#8888FF',          # Light blue
    'Moderate cold stress': '#AAAAFF',        # Very light blue
    'Slight cold stress': '#CCCCFF',          # Pale blue
    'No thermal stress': '#00CC00',           # Green
    'Moderate heat stress': '#FFDD55',        # Yellow
    'Strong heat stress': '#FF8800',          # Orange
    'Very strong heat stress': '#FF4400',     # Orange-red
    'Extreme heat stress': '#FF0000'          # Red
}
```

## Output Structure

### Generated Visualizations
```
results/weather_distribution/
├── nyc_temp_distribution_analysis.png    # Basic temperature analysis
├── nyc_temp_detailed_analysis.png        # Advanced temperature analysis  
└── nyc_utci_seasonal_hourly_distribution.png  # UTCI thermal comfort analysis
```

### Plot Specifications
- **High Resolution**: 300 DPI for publication quality
- **Large Format**: 20×16 and 30×24 inch figures for detail visibility
- **Professional Styling**: Seaborn-v0.8 style with colorblind-friendly palettes
- **Comprehensive Legends**: Full category names and clear color coding

## Key Analytical Insights

### Temperature Patterns
- **Seasonal Amplitude**: Quantifies NYC's seasonal temperature variation
- **Urban Heat Effects**: Identifies potential urban heat island signatures
- **Yearly Variations**: Tracks climate variability and potential trends
- **Extreme Events**: Highlights unusual temperature events and patterns

### Thermal Comfort Analysis
- **Peak Stress Hours**: Identifies times of day with highest thermal stress
- **Seasonal Comfort**: Shows how livable conditions vary throughout the year
- **Extreme Conditions**: Quantifies frequency of dangerous thermal conditions
- **Optimal Periods**: Identifies periods of ideal thermal comfort

### Monthly and Seasonal Insights
- **Winter Severity**: Analysis of cold stress patterns during winter months
- **Summer Heat**: Documentation of heat stress frequency and intensity
- **Transition Seasons**: Spring and fall thermal comfort characteristics
- **Diurnal Patterns**: How thermal comfort changes throughout the day

## Statistical Analysis Features

### Distribution Analysis
- **Descriptive Statistics**: Mean, median, mode, standard deviation, skewness
- **Outlier Detection**: Identification of extreme temperature events
- **Trend Analysis**: Linear and polynomial trend fitting for long-term patterns
- **Correlation Analysis**: Relationships between different temporal scales

### Quality Control
- **Missing Data Handling**: Systematic removal of incomplete records
- **Data Validation**: Range checks and consistency verification
- **Temporal Continuity**: Assessment of data gaps and coverage

## Technical Implementation

### Memory Management
- **Efficient Processing**: Optimized pandas operations for large datasets
- **Figure Management**: Proper figure closing to prevent memory leaks
- **Data Filtering**: Strategic filtering to reduce memory footprint

### Performance Characteristics
- **Processing Speed**: Optimized for datasets with millions of records
- **Visualization Quality**: High-resolution output without excessive processing time
- **Scalability**: Handles multi-year datasets efficiently

## Dependencies

- **pandas**: Data manipulation and temporal analysis
- **matplotlib**: Core plotting and figure generation
- **seaborn**: Statistical visualization and styling
- **numpy**: Numerical operations and statistical calculations

## Usage Example

```python
# Script runs automatically when executed
python src/weather_distribution.py

# Key steps performed:
1. Load NYC weather data with UTCI categories
2. Perform data cleaning and validation
3. Generate basic temperature distribution analysis
4. Create detailed temporal temperature analysis  
5. Produce UTCI thermal comfort seasonal analysis
6. Save all visualizations to results directory
```

## Customization Options

### Analysis Parameters
- **Time Period**: Adjust year range for analysis
- **Seasonal Definitions**: Modify month groupings for seasons
- **Temperature Units**: Support for Celsius/Fahrenheit conversion
- **UTCI Categories**: Customizable thermal stress thresholds

### Visualization Options
- **Color Schemes**: Alternative palettes for accessibility
- **Figure Sizes**: Adjustable dimensions for different display needs
- **Plot Types**: Additional chart types for specific analysis needs
- **Resolution Settings**: Configurable DPI for output quality

## Applications

### Climate Research
- **Urban Climate Analysis**: Understanding NYC's thermal environment
- **Climate Change Studies**: Long-term temperature trend analysis
- **Extreme Event Analysis**: Documentation of heat waves and cold snaps

### Public Health
- **Heat Stress Assessment**: Identifying dangerous thermal conditions
- **Seasonal Health Planning**: Understanding thermal stress seasonality
- **Vulnerability Analysis**: Identifying high-risk periods for temperature-related health issues 