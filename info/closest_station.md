# Closest Weather Station Finder (`closest_station.py`)

## Overview

This script identifies the closest weather stations to each NYC borough using the Haversine formula to calculate great circle distances. It's designed to help select appropriate weather stations for analysis by finding the stations with the most accurate representation of local weather conditions for each borough.

## Key Features

### Geographic Analysis
- **Haversine Distance Calculation**: Accurate great circle distance computation
- **Multi-Borough Analysis**: Finds closest stations for Bronx, Brooklyn, Manhattan, and Queens
- **Regional Filtering**: Focuses on NY, NJ, and CT stations for relevance
- **Distance Ranking**: Returns multiple closest options per borough

### Weather Station Processing
- **GHCN Station Data**: Processes Global Historical Climatology Network station lists
- **CSV Format Support**: Reads comma-separated weather station inventory files
- **Comprehensive Metadata**: Extracts station ID, coordinates, elevation, and descriptive information

## Core Functionality

### Distance Calculation (`haversine`)

**Purpose**: Calculates the great circle distance between two geographic points.

**Formula Implementation**:
```python
def haversine(lon1, lat1, lon2, lat2):
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r
```

### Station Data Parsing (`parse_station_line`)

**Purpose**: Parses individual lines from GHCN station list files.

**Data Fields Extracted**:
- Station ID (unique identifier)
- Latitude and longitude coordinates
- Elevation above sea level
- State/region code
- Station name and description
- Special flags (GSN, HCN/CRN, WMO)

### Station Analysis (`find_closest_stations`)

**Purpose**: Identifies the nearest weather stations to target locations.

**Analysis Process**:
1. Calculate distances from each target location to all available stations
2. Sort stations by distance for each location
3. Return specified number of closest stations with full metadata
4. Generate comprehensive distance ranking reports

## NYC Borough Target Coordinates

### Reference Points (Borough Centers)
```python
nyc_boroughs = {
    'Bronx': (40.8448, -73.8648),
    'Brooklyn': (40.6782, -73.9442),
    'Manhattan': (40.7831, -73.9712),
    'Queens': (40.7282, -73.7949)
}
```

These coordinates represent approximate geographic centers of each borough for optimal station selection.

## Input Requirements

### Station Data File
- **Expected Location**: `data/weather/csv/ghcnh-station-list.csv`
- **Format**: Comma-separated values with station metadata
- **Required Fields**: ID, latitude, longitude, elevation, state, name

### File Format Example
```csv
ACW00011604,17.1167,-61.7833,10.1,AG,ST JOHNS COOLIDGE FLD,,,
ACW00011647,17.1333,-61.7833,19.2,AG,ST JOHNS,,,
USW00014739,40.7789,-73.9692,39.6,NY,NEW YORK LAGUARDIA AIRPORT,,,
```

## Output Generation

### Console Output
Detailed station analysis with distance rankings:
```
Manhattan (Lat: 40.7831, Lon: -73.9712):
------------------------------------------------------------
1. Station ID: USW00014739
   Name: NEW YORK LAGUARDIA AIRPORT
   Location: 40.7789, -73.9692
   State: NY
   Distance: 2.34 km

2. Station ID: USW00094728
   Name: NEW YORK CENTRAL PARK
   Location: 40.7794, -73.9692
   State: NY
   Distance: 2.89 km
```

### CSV Results File
**Output Location**: `results/closest_weather_stations.csv`

**Columns**:
- Borough: NYC borough name
- Rank: Distance ranking (1 = closest)
- Station_ID: Unique weather station identifier
- Station_Name: Descriptive station name
- Latitude/Longitude: Station coordinates
- State: State/region code
- Distance_km: Distance in kilometers

### Summary Report
**Key Recommendations**:
```
RECOMMENDED STATIONS (Closest to each borough):
=============================================================
Bronx: USW00014739 - NEW YORK LAGUARDIA AIRPORT
  Distance: 8.12 km

Brooklyn: USW00094728 - NEW YORK CENTRAL PARK
  Distance: 15.67 km

Manhattan: USW00094728 - NEW YORK CENTRAL PARK
  Distance: 0.89 km

Queens: USW00014739 - NEW YORK LAGUARDIA AIRPORT
  Distance: 4.56 km
```

## Usage Examples

### Command Line Execution
```bash
python src/closest_station.py
```

### Programmatic Usage
```python
from closest_station import find_closest_stations, load_station_data

# Load station data
stations_df = load_station_data('data/weather/csv/ghcnh-station-list.csv')

# Define target locations
locations = {
    'Manhattan': (40.7831, -73.9712),
    'Brooklyn': (40.6782, -73.9442)
}

# Find closest stations
results = find_closest_stations(stations_df, locations, num_closest=3)
```

## Regional Filtering Strategy

### Multi-State Coverage
- **Primary**: New York (NY) stations for direct coverage
- **Secondary**: New Jersey (NJ) stations for western areas
- **Tertiary**: Connecticut (CT) stations for northern areas
- **Fallback**: All available stations if regional data insufficient

### Benefits of Regional Approach
- **Climate Similarity**: Nearby states share similar weather patterns
- **Data Redundancy**: Multiple station options for reliability
- **Coverage Gaps**: Fills areas with sparse NY station coverage

## Key Insights from Analysis

### Station Distribution Patterns
- **Manhattan**: Often closest to Central Park weather station
- **Bronx**: Frequently closest to LaGuardia Airport station
- **Queens**: Usually best served by LaGuardia or JFK airport stations
- **Brooklyn**: May use Manhattan stations due to proximity

### Distance Considerations
- **Urban Effects**: City stations may have urban heat island effects
- **Airport Stations**: Often provide high-quality, consistent measurements
- **Elevation Differences**: Consider elevation impact on temperature/pressure
- **Water Proximity**: Coastal effects on Brooklyn and Manhattan stations

## Technical Implementation

### Error Handling
- **File Validation**: Checks for station file existence
- **Data Parsing**: Handles malformed lines gracefully
- **Coordinate Validation**: Validates latitude/longitude ranges
- **State Filtering**: Manages missing or invalid state codes

### Performance Characteristics
- **Fast Computation**: Vectorized distance calculations
- **Memory Efficient**: Processes station data without excessive memory usage
- **Scalable**: Handles large station inventories efficiently

## Dependencies

- **pandas**: DataFrame operations for station data management
- **numpy**: Mathematical operations for distance calculations
- **math**: Trigonometric functions for Haversine formula
- **os**: File system operations and path management

## Applications

### Weather Data Selection
- **Historical Analysis**: Choose stations for multi-year weather studies
- **Forecast Validation**: Select reference stations for model verification
- **Climate Research**: Identify representative stations for regional analysis

### Data Integration
- **Preprocessing Step**: Station selection before weather data download
- **Quality Control**: Choose high-quality stations with complete records
- **Spatial Analysis**: Understand geographic representation in datasets

## Best Practices

1. **Multiple Options**: Consider several closest stations for redundancy
2. **Data Quality**: Verify selected stations have complete historical records
3. **Urban Effects**: Be aware of urban heat island effects in city stations
4. **Temporal Consistency**: Ensure selected stations cover your analysis period
5. **Metadata Review**: Check station descriptions for relocation history

## Limitations

- **Point Representation**: Treats boroughs as single points rather than areas
- **Elevation Ignored**: Distance calculation doesn't account for elevation differences
- **Station Quality**: Doesn't assess data quality or completeness
- **Temporal Changes**: Doesn't account for station relocations over time 