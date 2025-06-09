# Ride Weather Distribution Analysis (`ride_weather_distribution.py`)

## Overview

This script creates comprehensive line graph visualizations showing how Citibike ridership patterns vary by hour of day across different weather conditions and thermal comfort (UTCI) categories. It processes large parquet files to generate intuitive visualizations that reveal the relationship between weather and cycling behavior.

**⚠️ IMPORTANT WARNING**: This script is computationally intensive and can freeze systems with limited resources. Use with caution and ensure adequate system memory before running.

## Key Features

### Visualization Types
1. **UTCI Category Analysis**: Ridership by thermal comfort levels
2. **Weather Category Analysis**: Ridership by precipitation and sky conditions  
3. **User Type Breakdown**: Separate analysis for subscribers vs casual riders

### Color-Coded Insights
- **UTCI Categories**: Blue (cold) to green (neutral) to red (hot) spectrum
- **Weather Categories**: Intuitive colors (gold for clear, blues for rain, white for snow)
- **User Types**: Distinct colors for easy comparison

## Visualization Functions

### 1. UTCI Thermal Comfort Analysis (`create_utci_graphs`)

**Purpose**: Shows how thermal comfort affects ridership patterns throughout the day.

**UTCI Categories Analyzed**:
- Extreme cold stress → Very strong cold stress → Strong cold stress
- Moderate cold stress → Slight cold stress → No thermal stress
- Moderate heat stress → Strong heat stress → Very strong heat stress → Extreme heat stress

**Color Mapping**:
```python
utci_colors = {
    'extreme_cold_stress': '#0000FF',      # Deep blue
    'very_strong_cold_stress': '#1E90FF',  # Dodger blue
    'strong_cold_stress': '#4169E1',       # Royal blue
    'moderate_cold_stress': '#6495ED',     # Cornflower blue
    'slight_cold_stress': '#87CEEB',       # Sky blue
    'no_thermal_stress': '#32CD32',        # Lime green
    'moderate_heat_stress': '#FFA07A',     # Light salmon
    'strong_heat_stress': '#FF6347',       # Tomato
    'very_strong_heat_stress': '#FF4500',  # Orange red
    'extreme_heat_stress': '#FF0000'       # Red
}
```

### 2. Weather Category Analysis (`create_weather_graphs`)

**Purpose**: Demonstrates how precipitation and sky conditions impact ridership.

**Weather Categories**:
- Clear, Partly Cloudy, Mostly Cloudy, Cloudy
- Fog, Light Rain, Rain, Heavy Rain
- Thunderstorm, Snow, Sleet, Freezing Rain

**Color Strategy**:
```python
weather_colors = {
    'clear': '#FFD700',           # Gold/yellow for clear sky
    'partly_cloudy': '#87CEEB',   # Sky blue 
    'mostly_cloudy': '#708090',   # Slate gray
    'cloudy': '#A9A9A9',          # Dark gray
    'fog': '#D3D3D3',             # Light gray
    'light_rain': '#ADD8E6',      # Light blue
    'rain': '#4682B4',            # Steel blue
    'heavy_rain': '#000080',      # Navy blue
    'thunderstorm': '#800080',    # Purple
    'snow': '#FFFFFF',            # White
    'sleet': '#E0FFFF',           # Light cyan
    'freezing_rain': '#B0E0E6'    # Powder blue
}
```

### 3. User Type Segmentation (`create_weather_by_user_type_graphs`)

**Purpose**: Compares how different user types (subscribers vs customers) respond to weather conditions.

**Analysis Dimensions**:
- Separate plots for each user type
- Weather condition overlays
- Peak hour identification by user behavior

## Technical Implementation

### Data Processing Workflow
```python
# Load parquet file for specific year
df = pd.read_parquet(file_path)

# Create hourly datetime if needed
if 'datetime_hour' not in df.columns:
    df['datetime_hour'] = pd.to_datetime(df['start_time']).dt.floor('h')

# Extract hour of day for analysis
df['hour_of_day'] = pd.to_datetime(df['datetime_hour']).dt.hour

# Aggregate by hour and weather condition
rides_by_hour = category_data.groupby('hour_of_day').size()
```

### Scaling for Readability
- **Million-Scale Conversion**: Divides ride counts by 1,000,000 for readable y-axis labels
- **Comprehensive Legends**: Includes all weather/UTCI categories with proper labels
- **Grid System**: Adds gridlines for easier value reading

## Output Structure

### File Naming Convention
```
results/ride_weather_distribution/
├── ride_distribution_by_utci_{year}.png
├── ride_distribution_by_weather_{year}.png
└── ride_distribution_by_weather_usertype_{year}.png
```

### Plot Features
- **24-Hour Timeline**: X-axis shows all hours (00:00 to 23:00)
- **Multiple Series**: One line per weather/UTCI category
- **Reference Line**: Bold black line showing total ridership pattern
- **Professional Styling**: Large figure size (14×10) with clear legends

## Key Insights Revealed

### Thermal Comfort Patterns
- **Cold Sensitivity**: How ridership drops with decreasing thermal comfort
- **Heat Tolerance**: Upper limits of comfortable cycling temperatures
- **Optimal Conditions**: Peak ridership under "no thermal stress" conditions
- **Extreme Weather**: Near-zero ridership during extreme thermal stress

### Weather Impact Analysis
- **Precipitation Effects**: Progressive ridership decrease from light rain to heavy rain
- **Snow Impact**: Dramatic ridership reduction during snow events
- **Clear Weather Premium**: Higher ridership on clear vs cloudy days
- **Fog and Visibility**: Impact of reduced visibility conditions

### User Behavior Differences
- **Commuter Resilience**: Subscribers showing more weather tolerance
- **Recreational Sensitivity**: Casual riders more affected by poor weather
- **Peak Hour Persistence**: How weather affects rush hour vs leisure ridership
- **Weekend vs Weekday**: Different weather sensitivities by trip purpose

## Usage Example

```python
# Create all three types of visualizations
create_utci_graphs()          # Thermal comfort analysis
create_weather_graphs()       # Weather condition analysis  
create_weather_by_user_type_graphs()  # User type segmentation
```

## Input Requirements

- **Parquet Files**: `{year}_combined_citibike_weather.parquet` in `data/combined/`
- **Required Columns**:
  - `start_time`: Trip start timestamp
  - `utci_cat`: UTCI thermal comfort category
  - `weather_cat`: Weather condition category
  - `user_type`: Subscriber or customer designation

## Performance Considerations

### System Requirements
- **Memory**: Minimum 8GB RAM recommended for large files
- **Processing Time**: 5-15 minutes per year depending on file size
- **Storage**: ~2MB per visualization file generated

### Memory Management
- **File-by-File Processing**: Processes one year at a time
- **Garbage Collection**: Clears memory between file processing
- **Large Figure Handling**: Uses high DPI for quality but manages memory

## Dependencies

- pandas: Data manipulation and grouping operations
- matplotlib: Core plotting functionality and customization
- numpy: Numerical operations and data transformations
- pathlib: File system operations and directory management
- glob: File pattern matching for batch processing

## Customization Options

### Color Schemes
- Modify color dictionaries for different visual themes
- Colorblind-friendly alternatives available
- High-contrast options for accessibility

### Plot Styling
- Adjustable figure sizes for different display needs
- Customizable line weights and transparency
- Flexible legend positioning

## Best Practices

1. **System Resources**: Monitor memory usage during execution
2. **File Management**: Ensure sufficient disk space for outputs
3. **Progressive Analysis**: Start with single year before batch processing
4. **Visual Review**: Check color combinations for clarity and accessibility 