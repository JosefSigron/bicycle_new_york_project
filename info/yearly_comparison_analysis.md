# Yearly Comparison Analysis (`yearly_comparison_analysis.py`)

## Overview

This script performs comprehensive year-over-year comparison analysis of NYC Citibike ridership patterns combined with weather data. It's designed to handle extremely large parquet files (1.7GB+ each) using memory-efficient streaming processing techniques to analyze patterns across multiple years (2019-2024).

## Key Features

### Memory-Efficient Processing
- **Chunked Analysis**: Uses PyArrow to read parquet files in 50,000-row batches to prevent memory overload
- **Streaming Statistics**: Calculates aggregate statistics without loading full datasets into memory
- **Garbage Collection**: Implements aggressive memory management with garbage collection
- **Safety Limits**: Includes batch limits to prevent infinite loops and system freezes

### Analysis Components

1. **Yearly Summary Statistics**
   - Trip duration analysis (mean, median, min, max)
   - Weather pattern distributions
   - Seasonal ridership breakdowns
   - User type analysis (subscriber vs customer)

2. **Ride Pattern Analysis**
   - Daily ridership trends by year
   - Peak usage hours comparison
   - Seasonal variation patterns
   - Weekend vs weekday usage

3. **Weather Trend Analysis**
   - Temperature, precipitation, humidity, and wind patterns
   - Weather category distributions by year
   - UTCI (thermal comfort) trends
   - Weather-ridership correlations

4. **Comparative Reporting**
   - Year-over-year growth rates
   - Statistical significance testing
   - Trend identification
   - Summary reports with key findings

## Technical Implementation

### Data Processing Strategy
```python
# Memory-safe batch processing
parquet_file = pq.ParquetFile(file_path)
batch_size = 50000  # Small batch size for memory safety

for batch in parquet_file.iter_batches(batch_size=batch_size):
    chunk = batch.to_pandas()
    self._process_mini_chunk(chunk, stats, year)
    del chunk
    gc.collect()
```

### Statistical Aggregation
- **Incremental Statistics**: Builds comprehensive statistics incrementally across batches
- **Weighted Calculations**: Properly weights averages across chunks
- **Category Counting**: Tracks distributions of categorical variables
- **Temporal Grouping**: Aggregates by seasons, months, and user types

### Error Handling
- **Multiple Fallback Strategies**: Progressively smaller batch sizes if memory issues occur
- **Graceful Degradation**: Uses file metadata if direct processing fails
- **Comprehensive Logging**: Detailed progress reporting and error tracking

## Output Generation

### Visualizations
- **Comparative Charts**: Side-by-side yearly comparisons
- **Trend Analysis**: Multi-year trend lines with statistical indicators
- **Heatmaps**: Monthly/seasonal pattern visualizations
- **Distribution Plots**: Weather and ride pattern distributions

### Reports
- **Summary Statistics**: Comprehensive yearly comparison tables
- **Trend Analysis**: Growth rates and pattern changes
- **Weather Correlations**: Impact of weather on ridership by year
- **Executive Summary**: Key findings and recommendations

## Usage

```python
from yearly_comparison_analysis import YearlyAnalyzer

# Initialize analyzer
analyzer = YearlyAnalyzer(data_dir="data/combined")

# Run complete analysis for specific years
analyzer.run_complete_analysis(years=[2019, 2020, 2021, 2022, 2023, 2024])

# Or run individual analysis components
analyzer.load_yearly_data_efficiently()
analyzer.calculate_yearly_summary_stats()
analyzer.analyze_ride_patterns()
analyzer.analyze_weather_trends()
analyzer.create_comparative_summary_report()
```

## Input Requirements

- **Parquet Files**: `{year}_combined_citibike_weather.parquet` files in `data/combined/`
- **Required Columns**: trip_duration, start_time, user_type, weather_cat, utci_cat, temperature, precipitation, humidity, wind_speed

## Output Structure

```
results/yearly_analysis/
├── summary_statistics/
│   ├── yearly_comparison_summary.csv
│   ├── weather_trends_by_year.csv
│   └── ridership_growth_analysis.csv
├── visualizations/
│   ├── yearly_ridership_trends.png
│   ├── weather_pattern_comparison.png
│   ├── seasonal_analysis_heatmap.png
│   └── user_type_trends.png
└── reports/
    ├── executive_summary.txt
    └── detailed_findings.md
```

## Performance Characteristics

- **Memory Usage**: Maintains ~1GB peak memory usage regardless of input file size
- **Processing Speed**: ~50,000 rows per second with full statistical analysis
- **Scalability**: Can handle files up to 100M+ records with safety limits
- **Reliability**: Includes fallback strategies for various failure scenarios

## Key Insights Generated

1. **Year-over-Year Growth**: Quantifies ridership changes with statistical significance
2. **Weather Impact Evolution**: How weather sensitivity has changed over time
3. **Usage Pattern Shifts**: Changes in peak hours, seasonal patterns, user demographics
4. **Recovery Analysis**: Post-pandemic ridership recovery patterns
5. **Long-term Trends**: Multi-year patterns in urban mobility

## Dependencies

- pandas: Data manipulation and analysis
- numpy: Numerical computations
- matplotlib/seaborn: Visualization
- pyarrow: Memory-efficient parquet processing
- dask: Large dataset handling
- scipy: Statistical analysis
- pathlib: File system operations

## Notes

- **Memory Critical**: Designed specifically for memory-constrained environments
- **Long Running**: Complete analysis can take 30+ minutes for full dataset
- **Cache Enabled**: Uses caching to avoid reprocessing unchanged data
- **Progress Tracking**: Detailed progress reporting for long-running operations 