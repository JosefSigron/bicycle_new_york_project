import pandas as pd
import os
import glob
from pathlib import Path
import numpy as np
from tqdm import tqdm

def load_weather_data(weather_file):
    """Load weather data and convert datetime to pandas datetime."""
    print(f"Loading weather data from {weather_file}...")
    weather_df = pd.read_csv(weather_file)
    
    # Assuming the datetime column in weather data is named 'datetime'
    if 'datetime' in weather_df.columns:
        weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])
    else:
        # Try to identify the datetime column
        for col in weather_df.columns:
            if 'time' in col.lower() or 'date' in col.lower():
                print(f"Using {col} as datetime column")
                weather_df['datetime'] = pd.to_datetime(weather_df[col])
                break
    
    # Sort by datetime to ensure proper matching later
    weather_df = weather_df.sort_values('datetime')
    
    # Create a datetime_rounded column rounded to the nearest hour for easier matching
    weather_df['datetime_hour'] = weather_df['datetime'].dt.floor('H')
    
    print(f"Weather data loaded. Shape: {weather_df.shape}")
    return weather_df

def process_year_data(citibike_file, weather_df, output_dir):
    """Process a single year of Citibike data and combine with weather data."""
    year = os.path.basename(citibike_file).split('_')[0]
    print(f"Processing Citibike data for year {year}...")
    
    # Load Citibike data
    citibike_df = pd.read_parquet(citibike_file)
    print(f"Citibike data loaded. Shape: {citibike_df.shape}")
    
    # Ensure datetime columns are in datetime format
    citibike_df['start_time'] = pd.to_datetime(citibike_df['start_time'])
    citibike_df['stop_time'] = pd.to_datetime(citibike_df['stop_time'])
    
    # Add rounded hour columns for merging
    citibike_df['start_hour'] = citibike_df['start_time'].dt.floor('H')
    
    # Filter weather data for the year to reduce memory usage
    year_start = pd.Timestamp(f"{year}-01-01")
    year_end = pd.Timestamp(f"{year}-12-31 23:59:59")
    year_weather = weather_df[(weather_df['datetime'] >= year_start) & 
                             (weather_df['datetime'] <= year_end)].copy()
    
    print(f"Weather data for {year}: {year_weather.shape[0]} records")
    
    # Process in chunks to manage memory
    chunk_size = 500000
    total_chunks = (len(citibike_df) + chunk_size - 1) // chunk_size
    all_combined_chunks = []
    
    for chunk_idx in range(total_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(citibike_df))
        
        print(f"Processing chunk {chunk_idx+1}/{total_chunks} ({start_idx} to {end_idx})...")
        chunk = citibike_df.iloc[start_idx:end_idx].copy()
        
        # Merge with weather data
        combined_chunk = merge_with_weather(chunk, year_weather)
        all_combined_chunks.append(combined_chunk)
    
    # Combine all chunks
    if all_combined_chunks:
        combined_df = pd.concat(all_combined_chunks, ignore_index=True)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the combined data
        output_file = os.path.join(output_dir, f"{year}_combined_citibike_weather.parquet")
        combined_df.to_parquet(output_file)
        print(f"Combined data saved to {output_file}. Shape: {combined_df.shape}")
        
        # Calculate percentage of rows with non-null weather data
        weather_cols = [col for col in combined_df.columns if col in year_weather.columns]
        non_null_count = combined_df[weather_cols[0]].notnull().sum() if weather_cols else 0
        total_count = len(combined_df)
        print(f"Percentage of rides with weather data: {non_null_count/total_count*100:.2f}%")
        
        return combined_df
    else:
        print("No data to combine")
        return None

def merge_with_weather(rides_df, weather_df):
    """Merge rides with weather data based on a smarter algorithm."""
    print(f"Merging {len(rides_df)} rides with weather data...")
    
    # Create a copy of the rides dataframe to avoid modifying the original
    result_df = rides_df.copy()
    
    # Method 1: Merge based on the hour of the start time
    # This creates a direct hour-to-hour match between ride start time and weather
    print("Performing hour-based merge...")
    hour_merged = pd.merge_asof(
        rides_df.sort_values('start_time'),
        weather_df.sort_values('datetime'),
        left_on='start_time',
        right_on='datetime',
        direction='nearest',
        tolerance=pd.Timedelta(hours=1)  # Allow up to 1 hour difference
    )
    
    # If that worked well, return it
    if len(hour_merged) >= 0.9 * len(rides_df):  # If we preserved at least 90% of rows
        print(f"Hour-based merge successful, keeping {len(hour_merged)} of {len(rides_df)} rows")
        return hour_merged
    
    # Method 2: Use a midpoint approach - find the weather at the midpoint of each ride
    print("Performing midpoint-based merge...")
    # Calculate midpoint of each ride
    rides_df['midpoint_time'] = rides_df['start_time'] + (rides_df['stop_time'] - rides_df['start_time']) / 2
    
    # Merge asof using midpoint
    midpoint_merged = pd.merge_asof(
        rides_df.sort_values('midpoint_time'),
        weather_df.sort_values('datetime'),
        left_on='midpoint_time',
        right_on='datetime',
        direction='nearest',
        tolerance=pd.Timedelta(hours=2)  # Allow up to 2 hours difference
    )
    
    # If that worked well, return it
    if len(midpoint_merged) >= 0.9 * len(rides_df):
        print(f"Midpoint-based merge successful, keeping {len(midpoint_merged)} of {len(rides_df)} rows")
        return midpoint_merged
    
    # Method 3: Most flexible approach - merge with the nearest weather record
    # without any time constraints
    print("Performing nearest-time merge with no constraints...")
    nearest_merged = pd.merge_asof(
        rides_df.sort_values('start_time'),
        weather_df.sort_values('datetime'),
        left_on='start_time',
        right_on='datetime',
        direction='nearest'  # Just take the nearest no matter how far
    )
    
    print(f"Nearest-time merge completed, keeping {len(nearest_merged)} of {len(rides_df)} rows")
    return nearest_merged

def main():
    # Set up file paths
    weather_file = "./data/weather/csv/nyc_weather_with_utci.csv"
    citibike_dir = "./data/citibike/combined"
    output_dir = "./data/combined"
    
    # Load weather data once
    weather_df = load_weather_data(weather_file)
    
    # Process each Citibike parquet file
    citibike_files = glob.glob(os.path.join(citibike_dir, "*_citibike_data.parquet"))
    citibike_files.sort()  # Process in order by year
    
    for citibike_file in citibike_files:
        process_year_data(citibike_file, weather_df, output_dir)

if __name__ == "__main__":
    main() 