# Weather Impact on Bicycle Ridership Analysis

This module analyzes the relationship between weather conditions and Citibike ridership patterns, providing insights into how various meteorological factors influence cycling behavior in New York City.

## Overview

The `weather_impact_analysis.py` script examines how weather conditions affect bicycle usage patterns by analyzing combined weather and ridership datasets. It identifies correlations, quantifies impacts, and creates visualizations that help understand the complex relationship between environmental conditions and transportation choices.

## Analysis Components

### 1. **Temperature Impact Analysis**
- **Ridership vs Temperature**: Correlation analysis across temperature ranges
- **Optimal Temperature Range**: Identification of peak cycling temperatures
- **Seasonal Variations**: How temperature effects vary by season
- **Extreme Weather**: Impact of heat waves and cold snaps

### 2. **Thermal Comfort Analysis**
- **UTCI Categories**: Ridership patterns across thermal stress categories
- **Comfort Zone Analysis**: Identifying optimal thermal conditions for cycling
- **Heat Stress Impact**: How extreme heat affects usage patterns
- **Cold Stress Impact**: Analysis of cycling behavior in cold conditions

### 3. **Precipitation Analysis**
- **Rain Impact**: Immediate and delayed effects of rainfall on ridership
- **Snow Conditions**: How snow affects cycling patterns and safety
- **Intensity Analysis**: Different impacts of light vs heavy precipitation
- **Recovery Patterns**: How quickly ridership returns after weather events

### 4. **Wind and Visibility Analysis**
- **Wind Speed Effects**: Impact of wind conditions on cycling comfort
- **Visibility Conditions**: How fog, mist, and haze affect ridership
- **Combined Weather Effects**: Interaction between multiple weather factors
- **Safety Considerations**: Weather conditions that may affect cycling safety

## Key Metrics and Calculations

### Weather Impact Quantification
- **Correlation Coefficients**: Pearson and Spearman correlations between weather variables and ridership
- **Regression Analysis**: Quantifying the relationship between weather and cycling
- **Elasticity Measures**: Percentage change in ridership per unit change in weather conditions
- **Threshold Analysis**: Critical weather values that significantly impact ridership

### Statistical Methods
- **Time Series Analysis**: Examining weather effects over time
- **Seasonal Decomposition**: Separating weather effects from seasonal trends
- **Multi-variable Analysis**: Understanding combined effects of multiple weather factors
- **Significance Testing**: Determining statistical significance of weather relationships

## Visualization Outputs

### Correlation Plots
- Scatter plots showing ridership vs individual weather variables
- Correlation matrices showing relationships between all variables
- Trend lines with confidence intervals
- Seasonal overlay plots

### Distribution Analysis
- Ridership distributions under different weather conditions
- Box plots comparing ridership across weather categories
- Density plots showing frequency distributions
- Cumulative distribution comparisons

### Temporal Analysis
- Time series plots showing weather and ridership together
- Seasonal pattern analysis with weather overlay
- Event impact analysis (storms, heat waves, cold snaps)
- Recovery time analysis after weather events

## Analysis Functions

### Core Processing
**Weather Categorization**: Groups continuous weather variables into discrete categories for analysis.

**Impact Quantification**: Calculates numerical measures of weather impact on ridership patterns.

**Statistical Testing**: Performs hypothesis tests to determine significance of weather relationships.

**Visualization Generation**: Creates comprehensive plots and charts showing weather-ridership relationships.

### Data Processing
**Outlier Handling**: Identifies and appropriately handles extreme weather events and data anomalies.

**Missing Data Treatment**: Manages missing weather or ridership data to maintain analysis integrity.

**Temporal Alignment**: Ensures proper synchronization between weather conditions and ridership measurements.

## Key Findings Categories

### Temperature Relationships
- **Optimal Range**: Typical optimal cycling temperatures (15-25°C)
- **Linear Relationships**: Where ridership increases/decreases linearly with temperature
- **Threshold Effects**: Critical temperatures where behavior changes dramatically
- **Seasonal Variations**: How temperature sensitivity varies by season

### Precipitation Effects
- **Immediate Impact**: Ridership changes during precipitation events
- **Anticipatory Behavior**: Ridership changes before predicted precipitation
- **Recovery Patterns**: How long it takes for normal ridership to resume
- **Intensity Relationships**: Different impacts of light vs heavy precipitation

### Thermal Comfort Insights
- **UTCI Sensitivity**: How thermal comfort index relates to cycling behavior
- **Heat Stress Thresholds**: Critical thermal conditions that deter cycling
- **Cold Tolerance**: Understanding cycling behavior in cold conditions
- **Comfort Zone Definition**: Optimal thermal conditions for cycling

## Statistical Outputs

### Correlation Analysis
- Pearson correlation coefficients for linear relationships
- Spearman rank correlations for non-linear relationships
- Partial correlations controlling for other variables
- Time-lagged correlations for delayed effects

### Regression Models
- Linear regression models predicting ridership from weather
- Multiple regression including several weather variables
- Non-linear models for complex relationships
- Seasonal regression models

### Significance Testing
- Hypothesis tests for weather impact significance
- Confidence intervals for effect size estimates
- Multiple comparison corrections for simultaneous tests
- Power analysis for sample size adequacy

## Usage Example

```python
from weather_impact_analysis import analyze_weather_impact

# Run comprehensive weather impact analysis
results = analyze_weather_impact('data/combined/')

# Access specific analyses
temperature_analysis = results['temperature_impact']
precipitation_analysis = results['precipitation_impact']
thermal_comfort_analysis = results['thermal_comfort_impact']
```

## Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations and statistical functions
- **scipy**: Advanced statistical analysis and hypothesis testing
- **matplotlib/seaborn**: Statistical visualization and plotting
- **sklearn**: Machine learning models for regression analysis
- **statsmodels**: Advanced statistical modeling

## Applications

### Transportation Planning
- **Infrastructure Design**: Understanding weather-resistant transportation needs
- **Service Planning**: Anticipating ridership variations due to weather
- **Capacity Management**: Planning for weather-related demand changes

### Public Health
- **Active Transportation**: Understanding barriers to cycling for health
- **Climate Adaptation**: Preparing for changing weather patterns
- **Safety Planning**: Identifying weather conditions requiring safety measures

### Business Operations
- **Demand Forecasting**: Predicting ridership based on weather forecasts
- **Resource Allocation**: Optimizing bike distribution based on weather patterns
- **Marketing Strategy**: Weather-based promotion and communication strategies

## Data Quality Considerations

### Temporal Alignment
- Ensures weather conditions are properly matched to ridership timing
- Accounts for potential delays between weather and behavioral response
- Handles time zone and daylight saving time changes

### Statistical Robustness
- Uses appropriate statistical methods for the data characteristics
- Accounts for multiple testing when examining many relationships
- Provides confidence intervals and significance levels for all findings

## Future Enhancements

### Advanced Analysis
- **Machine Learning Models**: More sophisticated prediction models
- **Spatial Analysis**: How weather effects vary across different areas
- **Extreme Event Analysis**: Detailed analysis of rare weather events
- **Climate Change Projections**: Analysis of future weather impacts 