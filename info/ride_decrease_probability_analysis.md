# Ride Decrease Probability Analysis (`ride_decrease_probability_analysis.py`)

## Overview

This script analyzes the probability of ride decreases compared to normal weather conditions, providing statistical insights into how different weather scenarios impact Citibike ridership patterns throughout the day. It calculates the likelihood of reduced ridership for each hour under various weather and thermal comfort conditions.

## Key Features

### Probability Analysis
- **Baseline Comparison**: Uses normal weather conditions as baseline for comparison
- **Hourly Granularity**: Analyzes probability of decrease for each hour of the day (0-23)
- **User Type Segmentation**: Separate analysis for subscribers vs casual riders
- **Statistical Significance**: Includes t-tests to verify significance of differences

### Weather Classifications
- **Weather Categories**: Clear, cloudy, rainy, snowy, foggy conditions
- **UTCI Categories**: Thermal comfort levels from extreme cold to extreme heat stress
- **Normal Conditions**: "Neutral" weather and "No thermal stress" as baselines

## Core Methodology

### Probability Calculation
```python
# For each weather condition and hour:
normal_mean = hour_normal_data.mean()
condition_values = hour_condition_data.values

# Count days with fewer rides than normal average
decreases = np.sum(condition_values < normal_mean)
total_days = len(condition_values)
prob_decrease = decreases / total_days
```

### Statistical Testing
- **T-tests**: Independent sample t-tests between normal and adverse conditions
- **Significance Level**: p < 0.05 for statistical significance
- **Effect Direction**: Verifies that adverse conditions actually reduce ridership

## Analysis Functions

### 1. Data Loading (`load_parquet_files`)
- Loads all parquet files from combined data directory
- Extracts temporal features (hour, date) from timestamps
- Organizes data by year for analysis

### 2. Daily Aggregation (`calculate_daily_ride_counts_by_hour`)
- Groups rides by date, hour, and weather conditions
- Creates daily ride count summaries
- Enables day-to-day variability analysis

### 3. Probability Calculation (`calculate_decrease_probability_vs_normal`)
- Compares each weather condition against normal baseline
- Calculates probability metrics for each hour and user type
- Performs statistical significance testing

### 4. Visualization (`plot_decrease_probabilities`)
- Creates intuitive color-coded plots
- Shows probability trends by hour
- Includes significance indicators
- Separate plots for different user types

## Output Visualizations

### Color Mapping Strategy
- **Weather Categories**: Intuitive colors (blue for cold, red for heat, gray for precipitation)
- **UTCI Categories**: Cold-to-hot spectrum (navy blue to crimson red)
- **Significance Markers**: Visual indicators for statistically significant differences

### Plot Features
- **Hourly Trends**: Line plots showing probability changes throughout the day
- **User Type Comparison**: Side-by-side analysis for subscribers vs customers
- **Statistical Annotations**: Markers for significant differences
- **Peak Hour Identification**: Highlights hours with highest decrease probability

## Key Metrics Calculated

1. **Probability of Decrease**: Fraction of days with lower ridership than normal
2. **Normal Baseline**: Average ridership under normal weather conditions
3. **Condition Mean**: Average ridership under specific weather conditions
4. **Sample Sizes**: Number of days for each condition and hour
5. **Statistical Significance**: P-values from t-tests
6. **Effect Direction**: Whether conditions actually reduce ridership

## Usage Example

```python
# Load data for all available years
data_dict = load_parquet_files("data/combined")

# Analyze weather impact for each year
for year, df in data_dict.items():
    # Calculate daily ride counts by weather conditions
    ride_counts = calculate_daily_ride_counts_by_hour(
        df, ['hour', 'weather_cat', 'user_type']
    )
    
    # Calculate decrease probabilities
    prob_data = calculate_decrease_probability_vs_normal(
        ride_counts, 'weather_cat', 'user_type'
    )
    
    # Generate visualizations
    plot_decrease_probabilities(prob_data, 'weather_cat', year)
```

## Input Requirements

- **Parquet Files**: Combined Citibike-weather data files
- **Required Columns**: 
  - `start_time`: Trip start timestamp
  - `weather_cat`: Weather category classification
  - `utci_cat`: UTCI thermal comfort category
  - `user_type`: Subscriber or customer designation

## Output Structure

```
results/ride_decrease_probability_analysis/
├── weather_probability_2019.png
├── weather_probability_2020.png
├── utci_probability_2019.png
├── utci_probability_2020.png
└── summary_statistics.csv
```

## Key Insights Generated

### Weather Impact Patterns
- **Morning Commute**: How weather affects 7-9 AM ridership
- **Evening Rush**: Impact on 5-7 PM travel patterns
- **Recreational Hours**: Weather sensitivity during leisure times
- **User Type Differences**: How subscribers vs customers respond differently

### Seasonal Considerations
- **Winter Effects**: Cold and snow impact analysis
- **Summer Heat**: High temperature ridership patterns
- **Precipitation Impact**: Rain and storm effects across all hours
- **Thermal Comfort**: UTCI category influence on usage

## Statistical Rigor

### Sample Size Requirements
- Minimum 3 days of data for statistical testing
- Larger samples provide more reliable probability estimates
- Reports sample sizes for transparency

### Significance Testing
- Two-tailed t-tests for mean differences
- Multiple comparison awareness (though not corrected)
- Effect size consideration beyond just significance

## Dependencies

- pandas: Data manipulation and temporal analysis
- numpy: Statistical calculations and array operations
- matplotlib: Visualization and plotting
- seaborn: Enhanced statistical plotting
- scipy.stats: Statistical testing functions
- pathlib: File system operations

## Performance Notes

- **Memory Efficient**: Processes data year-by-year to manage memory
- **Scalable**: Handles multiple years of data systematically
- **Fast Processing**: Vectorized operations for probability calculations
- **Progress Tracking**: Year-by-year processing with status updates

## Limitations

- **Baseline Definition**: Results depend on definition of "normal" weather
- **Temporal Independence**: Assumes daily ridership values are independent
- **Weather Granularity**: Limited by weather categorization scheme
- **Seasonal Variations**: May conflate weather effects with seasonal patterns 