# Weekly Ride Time Analysis Summary

## Overview
This analysis examined median ride times by day of the week across different weather and thermal comfort categories using 6 years of Citi Bike data (2019-2024), totaling over 146 million rides.

## Data Processing
- **Total Records Processed**: 146,589,571 rides
- **Years Analyzed**: 2019-2024
- **Processing Method**: Chunked processing (50,000 rows per chunk) to handle large datasets
- **Time Ranges Analyzed**: 
  - All Day (complete dataset)
  - 6-9 AM (morning rush hour)
  - 10-14 PM (midday period)
  - 15-18 PM (afternoon rush hour)

## Key Findings

### Statistical Significance
- **All analyses showed statistically significant differences** (p < 0.05) across all days of the week
- This indicates that both weather conditions and thermal comfort categories have measurable impacts on ride duration patterns

### UTCI (Universal Thermal Climate Index) Categories
The analysis identified 9 thermal comfort categories:
- Extreme cold stress
- Very strong cold stress  
- Strong cold stress
- Moderate cold stress
- Slight cold stress
- No thermal stress
- Moderate heat stress
- Strong heat stress
- Very strong heat stress

**Key Observations:**
- **All Day**: Median ride times range from 6.4 to 12.8 minutes
- **Morning Rush (6-9 AM)**: Most consistent times (6.8-10.5 min), likely due to commuter efficiency
- **Midday (10-14 PM)**: Widest range (6.0-13.0 min), suggesting more leisurely rides
- **Afternoon Rush (15-18 PM)**: Slightly longer times (6.2-13.6 min) than morning

### Weather Categories
The analysis identified 6 weather categories:
- Cold
- Heat
- Mist/Fog
- Neutral
- Rain
- Snow

**Key Observations:**
- **All Day**: Median ride times range from 8.1 to 12.7 minutes
- **Morning Rush (6-9 AM)**: Shortest and most consistent (7.4-10.4 min)
- **Midday (10-14 PM)**: Range of 8.0-12.9 minutes
- **Afternoon Rush (15-18 PM)**: Longest times (8.5-13.6 min)

## Time Period Patterns

### Morning Rush Hour (6-9 AM)
- **Most efficient ride times** across all categories
- Narrowest ranges for both UTCI and weather categories
- Suggests commuter-focused, direct routing

### Midday Period (10-14 PM)
- **Greatest variability** in ride times
- Likely includes more recreational and tourist rides
- Weather and thermal comfort have larger impacts

### Afternoon Rush Hour (15-18 PM)
- **Longest median ride times** in most categories
- May reflect different commuting patterns or increased traffic
- Thermal stress categories show maximum impact

## Statistical Methodology
- **Significance Testing**: Kruskal-Wallis test (non-parametric ANOVA)
- **Sample Management**: Limited to 5,000 samples per category/day for memory efficiency
- **Aggregation**: Weighted median calculations across multiple data chunks
- **Visualization**: Significant differences marked with stars (*)

## Generated Visualizations
8 comprehensive plots were created, each containing:
1. **Line plot** showing median ride times by day of week for each category
2. **Statistical significance markers** (red-outlined stars)
3. **Data table** with median times, sample sizes, and p-values
4. **Professional formatting** with clear legends and labels

## Files Generated
1. `utci_cat_All_Day_analysis.png` - UTCI categories for all rides
2. `utci_cat_6_9_AM_analysis.png` - UTCI categories during morning rush
3. `utci_cat_10_14_PM_analysis.png` - UTCI categories during midday
4. `utci_cat_15_18_PM_analysis.png` - UTCI categories during afternoon rush
5. `weather_cat_All_Day_analysis.png` - Weather categories for all rides
6. `weather_cat_6_9_AM_analysis.png` - Weather categories during morning rush
7. `weather_cat_10_14_PM_analysis.png` - Weather categories during midday
8. `weather_cat_15_18_PM_analysis.png` - Weather categories during afternoon rush

## Technical Implementation
- **Memory Efficient**: Chunked processing handled 5.9GB of parquet data
- **Robust Statistics**: Proper handling of large sample sizes and missing data
- **Professional Visualization**: High-quality plots with embedded data tables
- **Reproducible**: All analysis parameters documented and version controlled

## Implications
This analysis provides valuable insights for:
- **Urban Transportation Planning**: Understanding how weather affects ride patterns
- **Bike Share Operations**: Optimizing bike distribution based on weather forecasts
- **Public Health**: Assessing thermal comfort impacts on active transportation
- **Climate Adaptation**: Planning for changing weather patterns and extreme conditions 