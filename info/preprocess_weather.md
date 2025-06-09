# Weather Data Preprocessing

This module preprocesses raw weather data for the NYC bicycle analysis project, converting weather station data into standardized formats and calculating derived metrics including thermal comfort indices.

## Overview

The `preprocess_weather.py` script processes weather data from multiple NYC boroughs to prepare it for analysis with Citibike ridership data. It handles sky cover parsing, calculates Mean Radiant Temperature (MRT), computes the Universal Thermal Climate Index (UTCI), and categorizes weather conditions.

## Key Functions

### Sky Cover Parsing

**Function**: `parse_sky_cover(df)`

Converts meteorological sky cover codes (oktas system) to cloud cover fractions:

| Code | Description | Cloud Cover Fraction |
|------|-------------|---------------------|
| CLR:00 | Clear sky | 0.0 |
| FEW:01 | Few clouds (1/10) | 0.1 |
| FEW:02 | Few clouds (2/10-3/10) | 0.25 |
| SCT:03-04 | Scattered (4/10-5/10) | 0.4-0.5 |
| BKN:05-07 | Broken (6/10-9/10) | 0.6-0.9 |
| OVC:08 | Overcast (10/10) | 1.0 |
| VV:09 | Sky obscured | 1.0 |

**Features**:
- Handles multiple cloud layers (sky_cover_1, sky_cover_2, sky_cover_3)
- Uses the last available layer as the overall cloud state
- Provides fallback values for missing data

### Mean Radiant Temperature Calculation

**Function**: `calculate_mean_radiant_temperature(df)`

Implements a simplified MRT calculation based on air temperature and cloud cover:

```
MRT = Tair + Solar_Adjustment

where:
Solar_Adjustment = (1 - 0.75 × Cloud_Cover) × 4°C (daytime only)
                 = 0°C (nighttime)
```

**Parameters**:
- Daytime: 6:00 AM to 6:00 PM
- Maximum adjustment: 4°C on clear days
- No adjustment during nighttime hours

### UTCI Calculation

**Function**: `calculate_utci(temperature, wind_speed, humidity, mrt)`

Uses the `pythermalcomfort` library to calculate the Universal Thermal Climate Index:

**Valid Input Ranges**:
- Air temperature: -50°C to 50°C
- Wind speed: 0.5 to 17 m/s (adjusted if outside range)
- Relative humidity: 5% to 100% (adjusted if outside range)
- Mean radiant temperature: No specific limits

**Error Handling**:
- Returns NaN for invalid inputs
- Automatically adjusts wind speed and humidity to valid ranges
- Handles missing data gracefully

### Weather Categorization

**Function**: `categorize_utci(utci_value)`

Classifies UTCI values into thermal stress categories:

| UTCI Range (°C) | Category |
|-----------------|----------|
| > 46 | Extreme heat stress |
| 38 to 46 | Very strong heat stress |
| 32 to 38 | Strong heat stress |
| 26 to 32 | Moderate heat stress |
| 9 to 26 | No thermal stress |
| 0 to 9 | Slight cold stress |
| -13 to 0 | Moderate cold stress |
| -27 to -13 | Strong cold stress |
| -40 to -27 | Very strong cold stress |
| < -40 | Extreme cold stress |

**Function**: `categorize_weather(utci, rain, snow, mist_fog)`

Creates comprehensive weather categories considering thermal comfort and precipitation:

- **Heat**: UTCI > 26°C
- **Cold**: UTCI < 9°C  
- **Rain**: Any precipitation > 0
- **Snow**: Snow present
- **Mist/Fog**: Visibility reduced conditions
- **Neutral**: UTCI 9-26°C, no precipitation

### Weather Condition Detection

**Function**: `detect_weather_conditions(df)`

Parses present weather codes to identify specific conditions:

**Rain Indicators**: 
- Drizzle, light rain, moderate rain, heavy rain
- Freezing rain, rain showers

**Snow Indicators**:
- Light snow, moderate snow, heavy snow
- Snow pellets, snow grains, blowing snow

**Mist/Fog Indicators**:
- Mist, fog, freezing fog
- Haze, smoke, volcanic ash

### Data Processing Pipeline

**Function**: `main()`

Processes weather data for all NYC boroughs:

1. **Data Loading**: Reads CSV files for Manhattan, Bronx, Brooklyn, Queens
2. **Sky Cover Parsing**: Converts oktas codes to cloud fractions
3. **MRT Calculation**: Computes mean radiant temperature
4. **UTCI Calculation**: Calculates thermal comfort index
5. **Weather Categorization**: Assigns weather and thermal categories
6. **Condition Detection**: Identifies specific weather phenomena
7. **Data Export**: Saves processed data with UTCI metrics

## Input Data Requirements

### Required Columns
- `datetime`: Timestamp in datetime format
- `temperature`: Air temperature in °C
- `wind_speed`: Wind speed in m/s
- `relative_humidity`: Relative humidity in %

### Optional Columns
- `sky_cover_1`, `sky_cover_2`, `sky_cover_3`: Cloud layer data
- `pres_wx_*`: Present weather condition codes
- Precipitation and visibility data

## Output Data

### New Columns Added
- `cloud_cover`: Cloud fraction (0-1)
- `mean_radiant_temp`: Mean radiant temperature (°C)
- `utci`: Universal Thermal Climate Index (°C)
- `utci_cat`: Thermal comfort category
- `weather_cat`: Comprehensive weather category
- `rain`, `snow`, `mist_fog`: Weather condition flags

## Usage Example

```python
from preprocess_weather import main

# Process all borough weather data
main()

# Results saved to: ./data/weather/csv/{borough}_weather_with_utci.csv
```

## Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **pythermalcomfort**: UTCI calculation
- **math**: Mathematical functions

## Performance Considerations

- **Memory Efficient**: Processes each borough separately
- **Vectorized Operations**: Uses pandas operations for speed
- **Error Handling**: Graceful handling of missing/invalid data
- **Progress Reporting**: Provides detailed processing feedback

## Applications

This preprocessed weather data enables:
- **Thermal Comfort Analysis**: Understanding how weather affects outdoor activities
- **Transportation Planning**: Linking weather conditions to bicycle usage
- **Climate Research**: Long-term weather pattern analysis
- **Public Health**: Assessing heat stress and cold exposure risks 