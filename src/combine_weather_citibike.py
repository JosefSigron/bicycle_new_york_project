import pandas as pd
import os
import glob
from pathlib import Path
import numpy as np
from tqdm import tqdm
import shutil

def load_all_weather_data():
    """Load weather data from all 4 borough files."""
    print("Loading weather data from all boroughs...")
    
    boroughs = ['manhattan', 'bronx', 'brooklyn', 'queens']
    weather_data = {}
    
    for borough in boroughs:
        weather_file = f"./data/weather/csv/{borough}_weather_with_utci.csv"
        
        if os.path.exists(weather_file):
            print(f"Loading weather data for {borough}...")
            # Fix DtypeWarning by setting low_memory=False
            df = pd.read_csv(weather_file, low_memory=False)
            
            # Convert datetime column
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Sort by datetime for efficient merging
            df = df.sort_values('datetime')
            
            # Create hourly rounded datetime for matching (fix FutureWarning)
            df['datetime_hour'] = df['datetime'].dt.floor('h')  # Changed 'H' to 'h'
            
            weather_data[borough] = df
            print(f"  - {borough}: {len(df)} weather records loaded")
            
            # Debug: Print column names for the first borough
            if borough == 'manhattan':
                print(f"  - Weather columns: {list(df.columns)}")
        else:
            print(f"Warning: Weather file for {borough} not found at {weather_file}")
    
    return weather_data

def determine_borough(lat, lon):
    """
    Determine which NYC borough a coordinate belongs to based on approximate boundaries.
    
    These are rough boundaries - for more precise mapping, you'd want to use actual
    borough boundary shapefiles, but this should work for most cases.
    
    Returns: 'manhattan', 'bronx', 'brooklyn', 'queens', or 'unknown'
    """
    if pd.isna(lat) or pd.isna(lon):
        return 'unknown'
    
    # Rough borough boundaries (these are approximations)
    # Manhattan: Generally between -74.02 to -73.93 longitude, 40.70 to 40.88 latitude
    if (-74.02 <= lon <= -73.93) and (40.70 <= lat <= 40.88):
        return 'manhattan'
    
    # Brooklyn: Generally between -74.05 to -73.83 longitude, 40.57 to 40.74 latitude
    elif (-74.05 <= lon <= -73.83) and (40.57 <= lat <= 40.74):
        return 'brooklyn'
    
    # Queens: Generally between -73.96 to -73.70 longitude, 40.54 to 40.80 latitude
    elif (-73.96 <= lon <= -73.70) and (40.54 <= lat <= 40.80):
        return 'queens'
    
    # Bronx: Generally between -73.93 to -73.76 longitude, 40.78 to 40.92 latitude
    elif (-73.93 <= lon <= -73.76) and (40.78 <= lat <= 40.92):
        return 'bronx'
    
    # If coordinates don't fall in any borough, try to make a best guess
    # based on proximity to known areas
    else:
        # Calculate distances to rough borough centers
        borough_centers = {
            'manhattan': (40.7831, -73.9712),
            'brooklyn': (40.6782, -73.9442),
            'queens': (40.7282, -73.7949),
            'bronx': (40.8448, -73.8648)
        }
        
        min_distance = float('inf')
        closest_borough = 'unknown'
        
        for borough, (center_lat, center_lon) in borough_centers.items():
            # Simple Euclidean distance (not perfect for lat/lon but good enough for rough estimates)
            distance = ((lat - center_lat) ** 2 + (lon - center_lon) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                closest_borough = borough
        
        return closest_borough

def merge_with_borough_weather(rides_df, weather_data):
    """Merge rides with appropriate borough weather data based on start location."""
    print(f"Merging {len(rides_df)} rides with borough-specific weather data...")
    
    # Debug: Print sample of ride times
    if len(rides_df) > 0:
        print(f"  - Ride time range: {rides_df['start_time'].min()} to {rides_df['start_time'].max()}")
    
    # Determine borough for each ride based on start coordinates
    print("Determining borough for each ride...")
    rides_df['start_borough'] = rides_df.apply(
        lambda row: determine_borough(row['start_station_latitude'], row['start_station_longitude']), 
        axis=1
    )
    
    # Count rides per borough
    borough_counts = rides_df['start_borough'].value_counts()
    print("Rides per borough:")
    for borough, count in borough_counts.items():
        print(f"  - {borough}: {count} rides ({count/len(rides_df)*100:.1f}%)")
    
    # Initialize result dataframe
    result_df = rides_df.copy()
    
    # Get weather columns from any available weather dataset
    weather_columns = []
    sample_weather_data = None
    for borough_data in weather_data.values():
        if not borough_data.empty:
            weather_columns = [col for col in borough_data.columns 
                             if col not in ['datetime', 'datetime_hour']]
            sample_weather_data = borough_data
            break
    
    print(f"Weather columns to add: {len(weather_columns)} columns")
    
    # Initialize weather columns with NaN/None values
    if sample_weather_data is not None:
        for col in weather_columns:
            if col in sample_weather_data.columns:
                sample_dtype = sample_weather_data[col].dtype
                
                if sample_dtype == 'object' or pd.api.types.is_string_dtype(sample_dtype):
                    result_df[col] = None
                    result_df[col] = result_df[col].astype('object')
                elif pd.api.types.is_integer_dtype(sample_dtype):
                    result_df[col] = pd.NA
                    if sample_dtype == 'int64':
                        result_df[col] = result_df[col].astype('Int64')
                    elif sample_dtype == 'int32':
                        result_df[col] = result_df[col].astype('Int32')
                    else:
                        result_df[col] = result_df[col].astype('Int64')
                elif pd.api.types.is_float_dtype(sample_dtype):
                    result_df[col] = np.nan
                    result_df[col] = result_df[col].astype(sample_dtype)
                elif pd.api.types.is_bool_dtype(sample_dtype):
                    result_df[col] = pd.NA
                    result_df[col] = result_df[col].astype('boolean')
                else:
                    result_df[col] = None
                    result_df[col] = result_df[col].astype('object')
    
    # Process each borough separately using efficient merging
    total_merged = 0
    for borough in ['manhattan', 'bronx', 'brooklyn', 'queens']:
        if borough not in weather_data or weather_data[borough].empty:
            print(f"No weather data available for {borough}")
            continue
            
        # Get rides that started in this borough
        borough_mask = result_df['start_borough'] == borough
        borough_rides_count = borough_mask.sum()
        
        if borough_rides_count == 0:
            print(f"No rides found starting in {borough}")
            continue
            
        print(f"Merging weather data for {borough_rides_count} rides in {borough}...")
        
        # Get weather data for this borough
        borough_weather = weather_data[borough].copy()
        
        if len(borough_weather) == 0:
            print(f"  - No weather data available for {borough}")
            continue
        
        # Sort weather data by datetime for efficient searching
        borough_weather = borough_weather.sort_values('datetime').reset_index(drop=True)
        
        # Create a temporary dataframe for borough rides
        borough_rides_df = result_df[borough_mask].copy()
        
        # Use merge_asof for efficient time-based merging
        # This finds the nearest weather record for each ride
        merged_borough = pd.merge_asof(
            borough_rides_df.sort_values('start_time'),
            borough_weather,
            left_on='start_time',
            right_on='datetime',
            direction='nearest',
            tolerance=pd.Timedelta(hours=6),  # Only match if within 6 hours
            suffixes=('', '_weather')  # Add suffix to weather columns to avoid conflicts
        )
        
        # Count successful merges (where weather data was found)
        # Look for weather columns with the '_weather' suffix
        available_weather_cols = []
        weather_col_mapping = {}
        
        for col in weather_columns:
            # Check for the column with _weather suffix
            weather_col_name = f"{col}_weather"
            if weather_col_name in merged_borough.columns:
                available_weather_cols.append(col)
                weather_col_mapping[col] = weather_col_name
            elif col in merged_borough.columns:
                # If the original column name exists (no conflict)
                available_weather_cols.append(col)
                weather_col_mapping[col] = col
        
        if available_weather_cols:
            # Count non-null values in the first available weather column
            first_weather_col = weather_col_mapping[available_weather_cols[0]]
            successful_merges = merged_borough[first_weather_col].notna().sum()
        else:
            successful_merges = 0
            print(f"  - Warning: No weather columns found in merged data for {borough}")
        
        # Update the result dataframe with merged weather data
        if successful_merges > 0:
            # Get the original indices
            original_indices = borough_rides_df.index
            
            # Update weather columns for this borough's rides
            for col in available_weather_cols:
                source_col = weather_col_mapping[col]
                if source_col in merged_borough.columns:
                    # Get the source values
                    source_values = merged_borough[source_col].values
                    
                    # Get the target column dtype from the result dataframe
                    target_dtype = result_df[col].dtype
                    
                    try:
                        # Cast the values to match the target dtype to avoid warnings
                        if pd.api.types.is_integer_dtype(target_dtype):
                            # For integer columns, handle NaN properly
                            source_values = pd.array(source_values, dtype=target_dtype)
                        elif pd.api.types.is_float_dtype(target_dtype):
                            # For float columns, check if we can convert
                            # Handle mixed types by converting to numeric, errors='coerce' will turn invalid values to NaN
                            source_values = pd.to_numeric(source_values, errors='coerce').astype(target_dtype)
                        elif pd.api.types.is_bool_dtype(target_dtype):
                            # For boolean columns, handle NaN properly
                            source_values = pd.array(source_values, dtype=target_dtype)
                        # For object/string columns, no casting needed
                        
                        result_df.loc[original_indices, col] = source_values
                        
                    except (ValueError, TypeError) as e:
                        print(f"  - Warning: Could not convert column '{col}' for {borough}: {str(e)}")
                        print(f"    Sample values: {source_values[:5]}")
                        print(f"    Target dtype: {target_dtype}, Source dtype: {type(source_values[0]) if len(source_values) > 0 else 'empty'}")
                        # Skip this column and continue with others
                        continue
        
        print(f"  - Successfully merged {successful_merges}/{borough_rides_count} rides in {borough}")
        total_merged += successful_merges
    
    # Report on overall merge success
    if len(weather_columns) > 0:
        total_count = len(result_df)
        print(f"Successfully merged weather data for {total_merged}/{total_count} rides ({total_merged/total_count*100:.1f}%)")
    
    return result_df

def process_year_data(citibike_file, weather_data, output_dir):
    """Process a single year of Citibike data and combine with borough-specific weather data."""
    year = os.path.basename(citibike_file).split('_')[0]
    print(f"\nProcessing Citibike data for year {year}...")
    
    # Load Citibike data
    citibike_df = pd.read_parquet(citibike_file)
    print(f"Citibike data loaded. Shape: {citibike_df.shape}")
    
    # Ensure datetime columns are in datetime format
    citibike_df['start_time'] = pd.to_datetime(citibike_df['start_time'])
    citibike_df['stop_time'] = pd.to_datetime(citibike_df['stop_time'])
    
    # Check if we have the required coordinate columns
    required_cols = ['start_station_latitude', 'start_station_longitude']
    missing_cols = [col for col in required_cols if col not in citibike_df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        return None
    
    # Get the actual time range of the data
    data_start = citibike_df['start_time'].min()
    data_end = citibike_df['start_time'].max()
    print(f"Data time range: {data_start} to {data_end}")
    
    # Filter weather data for the actual data time range (with buffer)
    buffer_days = 1  # Add 1 day buffer on each side
    weather_start = data_start - pd.Timedelta(days=buffer_days)
    weather_end = data_end + pd.Timedelta(days=buffer_days)
    
    filtered_weather_data = {}
    for borough, weather_df in weather_data.items():
        if not weather_df.empty:
            filtered_weather = weather_df[
                (weather_df['datetime'] >= weather_start) & 
                (weather_df['datetime'] <= weather_end)
            ].copy()
            filtered_weather_data[borough] = filtered_weather
            print(f"Weather data for {borough}: {len(filtered_weather)} records in time range")
            
            # Debug: Print date range of weather data
            if len(filtered_weather) > 0:
                print(f"  - {borough} weather date range: {filtered_weather['datetime'].min()} to {filtered_weather['datetime'].max()}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a temporary directory for chunks
    temp_dir = os.path.join(output_dir, f"temp_{year}")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Split data by time periods (e.g., weekly chunks)
    chunk_duration = pd.Timedelta(days=7)  # 1 week chunks
    
    # Generate time ranges for chunks
    current_start = data_start
    chunk_ranges = []
    
    while current_start < data_end:
        current_end = min(current_start + chunk_duration, data_end)
        chunk_ranges.append((current_start, current_end))
        current_start = current_end
    
    total_chunks = len(chunk_ranges)
    print(f"Processing {total_chunks} time-based chunks of {chunk_duration.days} days each...")
    
    # Process each time-based chunk
    for chunk_idx, (chunk_start, chunk_end) in enumerate(chunk_ranges):
        print(f"\nProcessing chunk {chunk_idx+1}/{total_chunks}: {chunk_start} to {chunk_end}")
        
        # Filter citibike data for this time range
        chunk_mask = (
            (citibike_df['start_time'] >= chunk_start) & 
            (citibike_df['start_time'] < chunk_end)
        )
        chunk = citibike_df[chunk_mask].copy()
        
        if len(chunk) == 0:
            print(f"  - No rides in this time period, skipping...")
            continue
            
        print(f"  - Found {len(chunk)} rides in this time period")
        
        # Get the actual time range of rides in this chunk
        actual_chunk_start = chunk['start_time'].min()
        actual_chunk_end = chunk['stop_time'].max()
        
        print(f"  - Actual rides time range: {actual_chunk_start} to {actual_chunk_end}")
        
        # Use the exact same time range for weather data (no buffer)
        chunk_weather_start = actual_chunk_start
        chunk_weather_end = actual_chunk_end
        
        print(f"  - Weather search range: {chunk_weather_start} to {chunk_weather_end}")
        
        # Filter weather data for this chunk's exact time range
        chunk_weather_data = {}
        for borough, weather_df in filtered_weather_data.items():
            if not weather_df.empty:
                chunk_weather = weather_df[
                    (weather_df['datetime'] >= chunk_weather_start) & 
                    (weather_df['datetime'] <= chunk_weather_end)
                ].copy()
                chunk_weather_data[borough] = chunk_weather
                
                if len(chunk_weather) > 0:
                    print(f"  - {borough}: {len(chunk_weather)} weather records for this chunk")
                    print(f"    Weather range: {chunk_weather['datetime'].min()} to {chunk_weather['datetime'].max()}")
                else:
                    print(f"  - {borough}: No weather records for this chunk time range")
        
        # Merge with borough-specific weather data for this chunk
        combined_chunk = merge_with_borough_weather(chunk, chunk_weather_data)
        
        # Save chunk to disk
        chunk_file = os.path.join(temp_dir, f"chunk_{chunk_idx:04d}.parquet")
        combined_chunk.to_parquet(chunk_file)
        print(f"  - Saved chunk to {chunk_file}")
        
        # Clear memory
        del combined_chunk
        del chunk
        del chunk_weather_data
    
    # Combine all chunks using a simpler approach
    print("\nCombining chunks...")
    output_file = os.path.join(output_dir, f"{year}_combined_citibike_weather.parquet")
    
    # Read and combine chunks in batches to avoid memory issues
    batch_size = 5  # Process 5 chunks at a time (since they're larger now)
    all_chunks = []
    
    for batch_start in range(0, total_chunks, batch_size):
        batch_end = min(batch_start + batch_size, total_chunks)
        print(f"Combining chunks {batch_start} to {batch_end-1}...")
        
        batch_chunks = []
        for chunk_idx in range(batch_start, batch_end):
            chunk_file = os.path.join(temp_dir, f"chunk_{chunk_idx:04d}.parquet")
            if os.path.exists(chunk_file):
                chunk = pd.read_parquet(chunk_file)
                batch_chunks.append(chunk)
        
        if batch_chunks:
            batch_combined = pd.concat(batch_chunks, ignore_index=True)
            all_chunks.append(batch_combined)
            del batch_chunks
            del batch_combined
    
    # Final combination
    if all_chunks:
        print("Final combination...")
        final_df = pd.concat(all_chunks, ignore_index=True)
        final_df.to_parquet(output_file)
        print(f"Combined data saved to {output_file}. Shape: {final_df.shape}")
        
        # Clean up temporary files
        print("Cleaning up temporary files...")
        try:
            for chunk_idx in range(total_chunks):
                chunk_file = os.path.join(temp_dir, f"chunk_{chunk_idx:04d}.parquet")
                if os.path.exists(chunk_file):
                    try:
                        os.remove(chunk_file)
                    except OSError as e:
                        print(f"Warning: Could not remove {chunk_file}: {e}")
            
            # Remove the temporary directory
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    print(f"Successfully removed temporary directory: {temp_dir}")
                except OSError as e:
                    print(f"Warning: Could not remove temporary directory {temp_dir}: {e}")
                    print("You may need to manually delete this directory later.")
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")
            print("Some temporary files may remain and need manual cleanup.")
        
        return final_df
    else:
        print("No data to combine")
        return None

def main():
    # Set up file paths
    citibike_dir = "./data/citibike/combined"
    output_dir = "./data/combined"
    
    # Load weather data from all boroughs
    weather_data = load_all_weather_data()
    
    if not weather_data:
        print("Error: No weather data loaded. Please check that the weather files exist.")
        return
    
    # Process each Citibike parquet file
    citibike_files = glob.glob(os.path.join(citibike_dir, "*_citibike_data.parquet"))
    citibike_files.sort()  # Process in order by year
    
    if not citibike_files:
        print(f"No Citibike files found in {citibike_dir}")
        return
    
    print(f"Found {len(citibike_files)} Citibike files to process")
    
    for citibike_file in citibike_files:
        process_year_data(citibike_file, weather_data, output_dir)
    
    print("\nAll files processed successfully!")

if __name__ == "__main__":
    main() 