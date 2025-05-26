import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
import os

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    Returns distance in kilometers
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

def parse_station_line(line):
    """
    Parse a line from the ghcnh-station-list.csv file according to the comma-separated format
    """
    if len(line.strip()) == 0:
        return None
    
    try:
        # Split by comma and strip whitespace
        parts = [part.strip() for part in line.split(',')]
        
        if len(parts) < 6:  # Need at least ID, lat, lon, elevation, state, name
            return None
            
        station_id = parts[0]
        latitude = float(parts[1])
        longitude = float(parts[2])
        elevation = float(parts[3]) if parts[3] else None
        state = parts[4] if len(parts) > 4 else ''
        name = parts[5] if len(parts) > 5 else ''
        gsn_flag = parts[6] if len(parts) > 6 else ''
        hcn_crn_flag = parts[7] if len(parts) > 7 else ''
        wmo_id = parts[8] if len(parts) > 8 else ''
        
        return {
            'ID': station_id,
            'LATITUDE': latitude,
            'LONGITUDE': longitude,
            'ELEVATION': elevation,
            'STATE': state,
            'NAME': name,
            'GSN_FLAG': gsn_flag,
            'HCN_CRN_FLAG': hcn_crn_flag,
            'WMO_ID': wmo_id
        }
    except (ValueError, IndexError) as e:
        print(f"Error parsing line: {line[:50]}... Error: {e}")
        return None

def load_station_data(file_path):
    """
    Load weather station data from the ghcnh-station-list.csv file
    """
    stations = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line_num == 1 and line.strip() == '0':  # Skip header if present
                continue
            
            station = parse_station_line(line)
            if station:
                stations.append(station)
    
    return pd.DataFrame(stations)

def find_closest_stations(stations_df, target_locations, num_closest=3):
    """
    Find the closest weather stations for each target location
    """
    results = {}
    
    for location_name, (lat, lon) in target_locations.items():
        # Calculate distances to all stations
        distances = []
        
        for idx, station in stations_df.iterrows():
            distance = haversine(lon, lat, station['LONGITUDE'], station['LATITUDE'])
            distances.append({
                'station': station,
                'distance_km': distance
            })
        
        # Sort by distance and get the closest ones
        distances.sort(key=lambda x: x['distance_km'])
        closest = distances[:num_closest]
        
        results[location_name] = closest
        
        print(f"\n{location_name} (Lat: {lat}, Lon: {lon}):")
        print("-" * 60)
        for i, result in enumerate(closest, 1):
            station = result['station']
            dist = result['distance_km']
            print(f"{i}. Station ID: {station['ID']}")
            print(f"   Name: {station['NAME']}")
            print(f"   Location: {station['LATITUDE']:.4f}, {station['LONGITUDE']:.4f}")
            print(f"   State: {station['STATE']}")
            print(f"   Distance: {dist:.2f} km")
            print()
    
    return results

def main():
    # NYC Borough coordinates (approximate centers)
    nyc_boroughs = {
        'Bronx': (40.8448, -73.8648),
        'Brooklyn': (40.6782, -73.9442),
        'Manhattan': (40.7831, -73.9712),
        'Queens': (40.7282, -73.7949)
    }
    
    # Path to the weather station data file
    station_file = os.path.join('data', 'weather', 'csv', 'ghcnh-station-list.csv')
    
    if not os.path.exists(station_file):
        print(f"Error: Station file not found at {station_file}")
        return
    
    print("Loading weather station data...")
    stations_df = load_station_data(station_file)
    print(f"Loaded {len(stations_df)} weather stations")
    
    # Filter for stations in New York state and nearby areas
    # Include NY, NJ, CT for better coverage
    nearby_states = ['NY', 'NJ', 'CT']
    ny_stations = stations_df[stations_df['STATE'].isin(nearby_states)]
    
    if len(ny_stations) == 0:
        print("No stations found in NY, NJ, or CT. Using all stations...")
        ny_stations = stations_df
    else:
        print(f"Found {len(ny_stations)} stations in NY, NJ, and CT")
    
    print("\nFinding closest weather stations for NYC boroughs...")
    results = find_closest_stations(ny_stations, nyc_boroughs, num_closest=3)
    
    # Create a summary DataFrame
    summary_data = []
    for borough, closest_stations in results.items():
        for i, result in enumerate(closest_stations):
            station = result['station']
            summary_data.append({
                'Borough': borough,
                'Rank': i + 1,
                'Station_ID': station['ID'],
                'Station_Name': station['NAME'],
                'Latitude': station['LATITUDE'],
                'Longitude': station['LONGITUDE'],
                'State': station['STATE'],
                'Distance_km': result['distance_km']
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save results to CSV
    output_file = os.path.join('results', 'closest_weather_stations.csv')
    os.makedirs('results', exist_ok=True)
    summary_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Print the best station for each borough
    print("\n" + "="*60)
    print("RECOMMENDED STATIONS (Closest to each borough):")
    print("="*60)
    
    for borough in nyc_boroughs.keys():
        best_station = summary_df[summary_df['Borough'] == borough].iloc[0]
        print(f"{borough}: {best_station['Station_ID']} - {best_station['Station_Name']}")
        print(f"  Distance: {best_station['Distance_km']:.2f} km")
        print()

if __name__ == "__main__":
    main() 