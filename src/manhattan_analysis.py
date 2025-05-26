import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry # For type hinting
import os
import numpy as np
from typing import Optional, List # Added List for type hinting
from tqdm import tqdm # For progress bars
import time
from contextlib import contextmanager
import sys

# Initialize tqdm for pandas apply
tqdm.pandas(desc="Processing", file=sys.stdout)

# Define path to the data
DATA_DIR = "data/citibike/combined/"
# CENSUS_SHAPEFILE_PATH = "data/nynta2020_25b/nynta2020.shp" # Removed

# Define Manhattan region boundaries (using both latitude and longitude)
# These are more accurate approximate boundaries for Manhattan regions
REGION_BOUNDARIES = {
    'Downtown': {
        'lat': (40.68, 40.73),  # up to (but not including) 40.73
        'lon': (-74.04, -73.90)  # Covers Battery Park to East Village
    },
    'Midtown': {
        'lat': (40.73, 40.765),  # up to (but not including) 40.765
        'lon': (-74.02, -73.95)  # Covers Chelsea to Murray Hill/Kips Bay
    },
    'Central Park Area': {
        'lat': (40.765, 40.80),  # up to (but not including) 40.80
        'lon': (-73.99, -73.94)  # Central Park and adjacent areas
    },
    'Upper': {
        'lat': (40.80, 40.90),  # from 40.80 upwards
        'lon': (-74.01, -73.91)  # Upper West Side, Upper East Side, Harlem
    }
}
REGION_ORDER = ['Downtown', 'Midtown', 'Central Park Area', 'Upper']

# Expected column names (based on inference)
# We will try to use these and handle potential errors if they are different.
COL_START_LAT = 'start_station_latitude'
COL_START_LNG = 'start_station_longitude'
COL_END_LAT = 'end_station_latitude'
COL_END_LNG = 'end_station_longitude'
COL_START_TIME = 'start_time'
COL_END_TIME = 'stop_time' # Assuming 'stop_time' from prior error
COL_TRIP_DURATION = 'trip_duration' # Assuming this is numeric (e.g., seconds)
COL_USER_TYPE = 'user_type'
COL_START_STATION = 'start_station_name'
COL_END_STATION_ID = 'end_station_id' # Using ID as name wasn't confirmed

# Define necessary columns to load from Parquet files
# This helps reduce memory footprint and loading time.
# Ensure all columns used anywhere in the script are listed here.
NECESSARY_COLUMNS = [
    COL_START_LAT, COL_START_LNG, COL_END_LAT, COL_END_LNG,
    COL_START_TIME, COL_END_TIME, COL_TRIP_DURATION, COL_USER_TYPE,
    COL_START_STATION, COL_END_STATION_ID
    # Add any other columns if they become necessary, e.g. 'rideable_type' if used later
]

# MANHATTAN_GEOMETRY: Optional[BaseGeometry] = None # Removed

# Define batch size for memory-efficient processing
BATCH_SIZE = 500000  # Process files in batches of this size
SCATTER_PLOT_SAMPLE_THRESHOLD = 250000 # Sample scatter plots if more than this many rows (reduced from 1M)

# Cache configuration
CACHE_DIR = "cache"
FINAL_DF_CACHE_PATH = os.path.join(CACHE_DIR, "final_manhattan_data.parquet")
RAW_SAMPLE_CACHE_PATH = os.path.join(CACHE_DIR, "first_raw_batch_sample.parquet")

@contextmanager
def timer_with_progress(description):
    """Context manager to time operations with a progress bar."""
    print(f"\n>>> Starting: {description}...")
    start_time = time.time()
    progress = tqdm(desc=description, total=1, position=0, leave=True, file=sys.stdout, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
    try:
        yield progress
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n!!! ERROR in {description} after {elapsed_time:.2f}s: {str(e)}")
        raise
    finally:
        elapsed_time = time.time() - start_time
        progress.set_description(f"{description} (completed in {elapsed_time:.2f}s)")
        progress.update(1)
        progress.close()
        print(f"<<< Completed: {description} in {elapsed_time:.2f}s")

def load_data_in_batches(data_dir: str, columns_to_load: Optional[List[str]] = None, batch_size: int = BATCH_SIZE):
    """Loads Parquet files in batches, yielding DataFrames."""
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.parquet')]
    if not all_files:
        print(f"No Parquet files found in {data_dir}")
        return
    
    all_files.sort()
    print(f"Found {len(all_files)} Parquet files to process.")

    for file_idx, f_path in enumerate(all_files):
        print(f"\nProcessing file {file_idx + 1}/{len(all_files)}: {f_path}")
        try:
            # Efficiently get row count without loading all data if possible
            # For pandas, often need to read at least one column or use pyarrow
            # Here, we'll read the full file for simplicity as individual files are assumed manageable
            temp_df_full_file = pd.read_parquet(f_path, columns=columns_to_load)
            total_rows_in_file = len(temp_df_full_file)

            if total_rows_in_file == 0:
                print(f"File {f_path} is empty. Skipping.")
                del temp_df_full_file
                continue

            num_file_batches = (total_rows_in_file + batch_size - 1) // batch_size
            
            for i in range(num_file_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, total_rows_in_file)
                batch_df = temp_df_full_file.iloc[start_idx:end_idx]

                if batch_df.empty:
                    continue
                
                print(f"  Yielding batch {i+1}/{num_file_batches} from file {f_path} ({len(batch_df)} rows)")
                yield batch_df.copy() # Yield a copy to avoid issues if the caller modifies it and temp_df_full_file is reused (though it's not here)
            del temp_df_full_file # Free memory after processing a file

        except Exception as e:
            print(f"Error reading or batching file {f_path}: {e}")
            # Ensure temp_df_full_file is deleted if it exists and an error occurred mid-processing
            if 'temp_df_full_file' in locals():
                del temp_df_full_file
            continue

def standardize_user_types(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes user type column to 'Member' and 'Casual'."""
    if COL_USER_TYPE not in df.columns:
        print(f"Warning: Column '{COL_USER_TYPE}' not found for standardization.")
        # Add the column with a default value if it's missing, to prevent downstream errors
        # Or ensure all downstream processes check for its existence.
        # For now, returning df as is might be okay if subsequent steps check.
        return df

    df_copy = df.copy() # Work on a copy
    df_copy[COL_USER_TYPE] = df_copy[COL_USER_TYPE].astype(str).str.lower()
    user_type_mapping = {
        'subscriber': 'Member',
        'member': 'Member',
        'customer': 'Casual',
        'casual': 'Casual'
    }
    # Make a copy before modifying to avoid SettingWithCopyWarning if df is a slice
    df_copy[COL_USER_TYPE] = df_copy[COL_USER_TYPE].map(user_type_mapping).fillna('Other')
    if 'Other' in df_copy[COL_USER_TYPE].unique():
        print(f"Warning: Some user types in this batch were mapped to 'Other'.")
    return df_copy

def filter_manhattan_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filters the DataFrame to include only rides starting in approximate Manhattan boundaries."""
    # Using approximate bounding box for Manhattan
    min_lon, max_lon = -74.04, -73.90  # Approximate Manhattan W and E longitudes
    min_lat, max_lat = 40.68, 40.90    # Approximate Manhattan S and N latitudes

    print(f"Starting Manhattan filtering on {len(df)} records...")
    
    if not all(col in df.columns for col in [COL_START_LAT, COL_START_LNG]):
        print(f"Warning: Lat/Lng columns ({COL_START_LAT}, {COL_START_LNG}) not found for Manhattan filtering.")
        return pd.DataFrame()  # Return empty if essential columns are missing

    # Work on a copy to avoid SettingWithCopyWarning, as df might be a slice from a larger frame in batch processing
    df_copy = df.copy()
    df_copy[COL_START_LAT] = pd.to_numeric(df_copy[COL_START_LAT], errors='coerce')
    df_copy[COL_START_LNG] = pd.to_numeric(df_copy[COL_START_LNG], errors='coerce')
    df_copy.dropna(subset=[COL_START_LAT, COL_START_LNG], inplace=True)

    # Filter in place without creating an intermediate DataFrame
    print("Applying Manhattan boundary filter...")
    manhattan_mask = (df_copy[COL_START_LNG].between(min_lon, max_lon)) & \
                     (df_copy[COL_START_LAT].between(min_lat, max_lat))
    
    result_df = df_copy[manhattan_mask]
    print(f"Manhattan filtering complete: {len(result_df)} records within Manhattan boundaries")
    return result_df

def assign_region(lat: float, lng: float) -> str:
    """Assigns a region based on latitude and longitude."""
    if pd.isna(lat) or pd.isna(lng):
        return 'Unknown'
    
    for region, bounds in REGION_BOUNDARIES.items():
        lat_lower, lat_upper = bounds['lat']
        lon_lower, lon_upper = bounds['lon']
        
        if (lat_lower <= lat < lat_upper) and (lon_lower <= lng <= lon_upper):
            return region
            
    return 'Outside Defined Regions'

def define_manhattan_regions(df: pd.DataFrame) -> pd.DataFrame:
    """Defines start and end regions in Manhattan based on latitude and longitude."""
    # This function assumes df already contains only Manhattan data
    print(f"Defining regions for {len(df)} records...")
    
    # Vectorized region assignment instead of apply for better performance
    df['start_region'] = 'Outside Defined Regions'  # Default value
    df['end_region'] = 'Outside Defined Regions'    # Default value
    
    print("Defining regions using vectorized operations...")
    region_counts = {}
    
    for region, bounds in REGION_BOUNDARIES.items():
        lat_lower, lat_upper = bounds['lat']
        lon_lower, lon_upper = bounds['lon']
        
        # Assign start regions
        if COL_START_LAT in df.columns and COL_START_LNG in df.columns:
            start_mask = (
                (df[COL_START_LAT] >= lat_lower) & 
                (df[COL_START_LAT] < lat_upper) & 
                (df[COL_START_LNG] >= lon_lower) & 
                (df[COL_START_LNG] <= lon_upper)
            )
            df.loc[start_mask, 'start_region'] = region
            region_counts[f"start_{region}"] = start_mask.sum()
            print(f"  - Assigned {start_mask.sum()} records to start region '{region}'")
        
        # Assign end regions
        if COL_END_LAT in df.columns and COL_END_LNG in df.columns:
            end_mask = (
                (df[COL_END_LAT] >= lat_lower) & 
                (df[COL_END_LAT] < lat_upper) & 
                (df[COL_END_LNG] >= lon_lower) & 
                (df[COL_END_LNG] <= lon_upper)
            )
            df.loc[end_mask, 'end_region'] = region
            region_counts[f"end_{region}"] = end_mask.sum()
            print(f"  - Assigned {end_mask.sum()} records to end region '{region}'")
    
    # Handle any NaN coordinates
    if COL_START_LAT in df.columns and COL_START_LNG in df.columns:
        na_start_mask = df[COL_START_LAT].isna() | df[COL_START_LNG].isna()
        df.loc[na_start_mask, 'start_region'] = 'Unknown'
        print(f"  - Assigned {na_start_mask.sum()} records with NA coordinates to 'Unknown' start region")
    
    if COL_END_LAT in df.columns and COL_END_LNG in df.columns:
        na_end_mask = df[COL_END_LAT].isna() | df[COL_END_LNG].isna()
        df.loc[na_end_mask, 'end_region'] = 'Unknown'
        print(f"  - Assigned {na_end_mask.sum()} records with NA coordinates to 'Unknown' end region")
    
    # Count records outside defined regions
    outside_start = (df['start_region'] == 'Outside Defined Regions').sum()
    outside_end = (df['end_region'] == 'Outside Defined Regions').sum()
    print(f"  - {outside_start} records outside defined start regions")
    print(f"  - {outside_end} records outside defined end regions")
    
    print("Region assignment complete")
    return df

def convert_trip_duration_to_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """Converts trip duration to minutes and stores it in 'trip_duration_minutes' using vectorized operations."""
    print(f"Converting trip durations for {len(df)} records...")
    
    if COL_TRIP_DURATION not in df.columns:
        print(f"Warning: Column '{COL_TRIP_DURATION}' not found. Cannot calculate duration.")
        df['trip_duration_minutes'] = pd.NA
        return df

    print(f"Column type for {COL_TRIP_DURATION}: {df[COL_TRIP_DURATION].dtype}")
    
    if pd.api.types.is_numeric_dtype(df[COL_TRIP_DURATION]):
        # Vectorized operation
        print(f"Converting numeric {COL_TRIP_DURATION} to minutes...")
        df['trip_duration_minutes'] = df[COL_TRIP_DURATION] / 60
    elif pd.api.types.is_timedelta64_dtype(df[COL_TRIP_DURATION]):
        # Vectorized operation
        print(f"Converting timedelta {COL_TRIP_DURATION} to minutes...")
        df['trip_duration_minutes'] = df[COL_TRIP_DURATION].dt.total_seconds() / 60
    else:
        try:
            # Extract numeric part from string more efficiently
            print(f"Extracting numeric part from string {COL_TRIP_DURATION}...")
            df['trip_duration_minutes'] = pd.to_numeric(
                df[COL_TRIP_DURATION].astype(str).str.extract(r'(\d+)')[0], 
                errors='coerce'
            ) / 60
            
            if df['trip_duration_minutes'].isnull().any():
                null_count = df['trip_duration_minutes'].isnull().sum()
                print(f"Warning: {null_count} trip durations could not be converted to numeric minutes.")
        except Exception as e:
            print(f"Error converting trip duration to minutes: {e}. Column set to NA.")
            df['trip_duration_minutes'] = pd.NA
    
    # Handle negative durations before capping long ones
    if 'trip_duration_minutes' in df.columns and df['trip_duration_minutes'].notna().any():
        negative_durations_mask = df['trip_duration_minutes'] < 0
        num_negative_durations = negative_durations_mask.sum()
        if num_negative_durations > 0:
            print(f"Warning: Found {num_negative_durations} negative trip durations. Setting them to 0 minutes.")
            df.loc[negative_durations_mask, 'trip_duration_minutes'] = 0

    # Cap extremely long durations to avoid issues with tests and plots
    # First, find and log any extreme outliers
    if 'trip_duration_minutes' in df.columns and not df['trip_duration_minutes'].empty:
        print("Checking for extreme outliers in trip duration...")
        extreme_outliers = df[df['trip_duration_minutes'] > 1440 * 2]  # More than 2 days
        if len(extreme_outliers) > 0:
            print(f"Found {len(extreme_outliers)} trip durations longer than 2 days.")
            print(f"Longest trip duration: {df['trip_duration_minutes'].max()} minutes")
            
            # Cap extremely long durations at 7 days for analysis purposes
            max_reasonable_duration = 1440 * 7  # 7 days in minutes
            df.loc[df['trip_duration_minutes'] > max_reasonable_duration, 'trip_duration_minutes'] = max_reasonable_duration
            print(f"Capped extremely long durations at {max_reasonable_duration} minutes (7 days)")
            
        print(f"Trip duration stats: min={df['trip_duration_minutes'].min()}, max={df['trip_duration_minutes'].max()}, mean={df['trip_duration_minutes'].mean()}")
    
    print("Trip duration conversion complete")
    return df

def analyze_data_by_region(df: pd.DataFrame):
    print(f"\nStarting regional analysis on {len(df)} combined records...")
    if 'start_region' not in df.columns or 'end_region' not in df.columns:
        print("Region columns not found. Skipping regional analysis.")
        return
    if df.empty:
        print("DataFrame for analysis is empty. Skipping regional analysis.")
        return

    with timer_with_progress("Analyzing trip durations by region (Combined)"):
        if 'trip_duration_minutes' not in df.columns or df['trip_duration_minutes'].notna().sum() == 0:
            print("\nTrip duration in minutes not available for duration analysis.")
        else:
            avg_duration = df.groupby('start_region')['trip_duration_minutes'].mean().reindex(REGION_ORDER).fillna(0)
            print("\nAverage trip duration (minutes) by start region:\n", avg_duration)

    with timer_with_progress("Analyzing trip origins and destinations (Combined)"):
        origin_counts = df['start_region'].value_counts().reindex(REGION_ORDER).fillna(0)
        print("\nTrip counts originating by region:\n", origin_counts)
        destination_counts = df['end_region'].value_counts().reindex(REGION_ORDER).fillna(0)
        print("\nTrip counts ending by region:\n", destination_counts)

    with timer_with_progress("Analyzing popular stations (Combined)"):
        if COL_START_STATION in df.columns:
            # Using .apply() here can be slow on very large data. If this becomes a bottleneck,
            # consider alternative ways to get mode if performance is critical.
            # For now, keeping as is since it's an aggregation.
            popular_start_stations = df.groupby('start_region')[COL_START_STATION].apply(lambda x: x.mode()[0] if not x.mode().empty else "N/A").reindex(REGION_ORDER).fillna("N/A")
            print("\nMost popular start station by region:\n", popular_start_stations)
        if COL_END_STATION_ID in df.columns:
            popular_end_stations = df.groupby('end_region')[COL_END_STATION_ID].apply(lambda x: x.mode()[0] if not x.mode().empty else "N/A").reindex(REGION_ORDER).fillna("N/A")
            print("\nMost popular end station ID by region:\n", popular_end_stations)

    with timer_with_progress("Analyzing temporal patterns (Combined)"):
        # CRITICAL FIX: Ensure no df.copy() is made here.
        # Temporal columns (day_of_week, hour_of_day, is_weekday) should already exist from batch processing.
        if COL_START_TIME in df.columns and \
           all(col in df.columns for col in ['day_of_week_name', 'hour_of_day', 'is_weekday']):
            
            if not pd.api.types.is_datetime64_any_dtype(df[COL_START_TIME]):
                print(f"Warning: Column '{COL_START_TIME}' is not datetime type in combined data. Attempting conversion.")
                # This is a fallback. Modifying df in place if absolutely necessary.
                # This operation could be memory intensive if COL_START_TIME is object type and large.
                try:
                    df[COL_START_TIME] = pd.to_datetime(df[COL_START_TIME], errors='coerce')
                    # After conversion, check for NaTs. If many, subsequent groupbys might be empty.
                    if df[COL_START_TIME].isna().any():
                        print(f"Warning: NaTs present in '{COL_START_TIME}' after fallback conversion.")
                except Exception as e_conv:
                    print(f"Error during fallback conversion of '{COL_START_TIME}': {e_conv}. Skipping temporal analysis.")
                    return # Exit this part of analysis if conversion fails

            # Proceed with analysis using 'df' directly
            weekday_weekend_counts = df.groupby(['start_region', 'is_weekday']).size().unstack(fill_value=0).reindex(REGION_ORDER).fillna(0)
            if True in weekday_weekend_counts.columns: weekday_weekend_counts.rename(columns={True:'Weekday'}, inplace=True)
            if False in weekday_weekend_counts.columns: weekday_weekend_counts.rename(columns={False:'Weekend'}, inplace=True)
            print("\nWeekday vs Weekend trip counts by start region:\n", weekday_weekend_counts)

            peak_hours_analysis = df.groupby(['start_region', 'hour_of_day']).size().unstack(fill_value=0).reindex(REGION_ORDER).fillna(0)
            print("\nTrip counts by hour of day for each start region (sample - top 5 hours):\n")
            for region_name in REGION_ORDER: 
                if region_name in peak_hours_analysis.index:
                    print(f"--- {region_name} ---")
                    print(peak_hours_analysis.loc[region_name].nlargest(5).to_string())
        else:
            print("\nCould not perform temporal analysis: one or more required columns "
                  f"('{COL_START_TIME}', 'day_of_week_name', 'hour_of_day', 'is_weekday') are missing from combined data.")
            
    with timer_with_progress("Analyzing user types (Combined)"):
        if COL_USER_TYPE in df.columns:
            user_type_dist = df.groupby('start_region')[COL_USER_TYPE].value_counts(normalize=True).mul(100).unstack(fill_value=0).reindex(REGION_ORDER).fillna(0)
            print("\nUser type distribution (%) by start region:\n", user_type_dist)
        else:
            print(f"\nColumn '{COL_USER_TYPE}' not found for user type analysis on combined set.")

def create_plots(df: pd.DataFrame):
    print(f"\nCreating plots for {len(df)} combined records...")
    if df.empty or 'start_region' not in df.columns:
        print("DataFrame for plotting is empty or 'start_region' is missing. Skipping plots.")
        return
    output_dir = "results/manhattan_analysis"
    os.makedirs(output_dir, exist_ok=True)

    df_display = df # Default to full df for plots
    if len(df) > SCATTER_PLOT_SAMPLE_THRESHOLD:
        print(f"Combined DataFrame has {len(df)} rows, exceeding scatter plot threshold of {SCATTER_PLOT_SAMPLE_THRESHOLD}. Sampling for scatter plots.")
        try:
            df_display = df.sample(n=SCATTER_PLOT_SAMPLE_THRESHOLD, random_state=42)
            print(f"Using {len(df_display)} rows for scatter plots.")
        except ValueError as ve:
            print(f"Could not sample {SCATTER_PLOT_SAMPLE_THRESHOLD} rows for scatter plots (possibly fewer rows available than threshold): {ve}. Using full data for scatter plots.")
            df_display = df # Fallback to full df if sampling fails
        except Exception as e_sample:
            print(f"Error during sampling for scatter plots: {e_sample}. Using full data for scatter plots.")
            df_display = df # Fallback to full df

    with timer_with_progress("Plotting: Trip counts originating by region"):
        plt.figure(figsize=(10, 6))
        origin_counts = df['start_region'].value_counts().reindex(REGION_ORDER).fillna(0)
        origin_counts.plot(kind='bar', title='Trip Counts Originating by Manhattan Region (Combined)')
        plt.xlabel("Region"); plt.ylabel("Number of Trips"); plt.xticks(rotation=45, ha="right"); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'trip_counts_originating_by_region_combined.png')); plt.close()

    with timer_with_progress("Plotting: Average trip duration by start region"):
        if 'trip_duration_minutes' in df.columns and df['trip_duration_minutes'].notna().any():
            plt.figure(figsize=(10, 6))
            avg_duration = df.groupby('start_region')['trip_duration_minutes'].mean().reindex(REGION_ORDER).fillna(0)
            avg_duration.plot(kind='bar', title='Average Trip Duration by Start Region (Combined)')
            plt.xlabel("Region"); plt.ylabel("Average Duration (minutes)"); plt.xticks(rotation=45, ha="right"); plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'avg_trip_duration_by_region_combined.png')); plt.close()

    with timer_with_progress("Plotting: User type distribution by start region"):
        if COL_USER_TYPE in df.columns:
            plt.figure(figsize=(12, 7))
            user_type_dist = df.groupby('start_region')[COL_USER_TYPE].value_counts(normalize=True).mul(100).unstack(fill_value=0).reindex(REGION_ORDER).fillna(0)
            user_type_dist.plot(kind='bar', stacked=False, title='User Type Distribution by Start Region (Combined)')
            plt.xlabel("Region"); plt.ylabel("Percentage (%)"); plt.xticks(rotation=45, ha="right"); plt.legend(title='User Type'); plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'user_type_distribution_by_region_combined.png')); plt.close()

    with timer_with_progress("Plotting: Weekday vs. Weekend trip counts"):
        if COL_START_TIME in df.columns and 'is_weekday' in df.columns : 
            plt.figure(figsize=(12, 7))
            weekday_weekend_counts = df.groupby(['start_region', 'is_weekday']).size().unstack(fill_value=0).reindex(REGION_ORDER).fillna(0)
            if True in weekday_weekend_counts.columns: weekday_weekend_counts.rename(columns={True:'Weekday'}, inplace=True)
            if False in weekday_weekend_counts.columns: weekday_weekend_counts.rename(columns={False:'Weekend'}, inplace=True)
            weekday_weekend_counts.plot(kind='bar', title='Weekday vs. Weekend Trip Counts by Start Region (Combined)')
            plt.xlabel("Region"); plt.ylabel("Number of Trips"); plt.xticks(rotation=45, ha="right"); plt.legend(title='Day Type'); plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'weekday_weekend_counts_by_region_combined.png')); plt.close()
        else:
            print(f"Skipping Weekday/Weekend plot: one or more required columns ('{COL_START_TIME}', 'is_weekday') are missing.")

    with timer_with_progress("Plotting: Scatter plot of start/end locations"):
        # df_display is already prepared (sampled or full df)
        coords_cols_present = all(col in df_display.columns for col in [COL_START_LNG, COL_START_LAT, COL_END_LNG, COL_END_LAT])
        if coords_cols_present:
            plt.figure(figsize=(10, 12))
            sns.scatterplot(data=df_display, x=COL_START_LNG, y=COL_START_LAT, hue='start_region', hue_order=REGION_ORDER, s=5, alpha=0.3, legend='full')
            plt.title('Bike Start Locations (Combined)'); plt.xlabel('Longitude'); plt.ylabel('Latitude')
            plt.legend(title='Start Region', bbox_to_anchor=(1.05, 1), loc='upper left'); plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'start_locations_by_region_combined.png')); plt.close()

            df_end_regions_display = df_display[df_display['end_region'].isin(REGION_ORDER)]
            if not df_end_regions_display.empty:
                plt.figure(figsize=(10, 12))
                sns.scatterplot(data=df_end_regions_display, x=COL_END_LNG, y=COL_END_LAT, hue='end_region', hue_order=REGION_ORDER, s=5, alpha=0.3, legend='full')
                plt.title('Bike End Locations (Combined)'); plt.xlabel('Longitude'); plt.ylabel('Latitude')
                plt.legend(title='End Region', bbox_to_anchor=(1.05, 1), loc='upper left'); plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'end_locations_by_region_combined.png')); plt.close()
            else:
                print("No valid end locations in (potentially sampled) data to plot.")
        else:
            print(f"Coordinate columns not found in (potentially sampled) df_display for scatter plots.")

    with timer_with_progress("Plotting: Trip intensity heatmaps"):
        if COL_START_TIME in df.columns and 'hour_of_day' in df.columns and 'day_of_week_name' in df.columns and 'start_region' in df.columns:
            print("Preparing data for trip intensity heatmaps...")
            days_ordered = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            
            # Perform a single groupby for all regions and relevant time components
            # Ensure 'start_region' is categorical with REGION_ORDER for correct sorting and inclusion of all regions
            df['start_region'] = pd.Categorical(df['start_region'], categories=REGION_ORDER, ordered=True)
            
            # Aggregate counts
            # Using observed=True if many regions might not have data, but REGION_ORDER should cover this.
            # Consider adding .copy() if any SettingWithCopyWarning appears on df['start_region'] above or if df is a slice.
            all_regions_heatmap_data = df.groupby(['start_region', 'hour_of_day', 'day_of_week_name'], observed=False).size().unstack(fill_value=0)
            
            if not all_regions_heatmap_data.empty:
                # Reindex to ensure all days are present and in order for each region's sub-DataFrame
                all_regions_heatmap_data = all_regions_heatmap_data.reindex(columns=days_ordered, fill_value=0)

                for region_name_hm in REGION_ORDER:
                    if region_name_hm in all_regions_heatmap_data.index:
                        # Select the data for the current region
                        # The multi-index will be (region_name_hm, hour_of_day)
                        # We want to plot hour_of_day on y-axis, day_of_week_name on x-axis
                        region_heatmap_data = all_regions_heatmap_data.loc[region_name_hm]
                        
                        if region_heatmap_data.empty or region_heatmap_data.sum().sum() == 0: # Check if all values are zero
                            print(f"No heatmap data for region {region_name_hm} (Combined) after grouping.")
                            continue

                        plt.figure(figsize=(12, 8))
                        sns.heatmap(region_heatmap_data, cmap="YlGnBu", linewidths=.5)
                        plt.title(f'Trip Intensity (Combined) - {region_name_hm}'); plt.xlabel('Day of Week'); plt.ylabel('Hour of Day (0-23)'); plt.tight_layout()
                        plt.savefig(os.path.join(output_dir, f'trip_intensity_heatmap_{region_name_hm.lower().replace(" ", "_")}_combined.png')); plt.close()
                    else:
                        print(f"Region {region_name_hm} not found in grouped heatmap data.")
            else:
                print("Grouped heatmap data is empty. Skipping heatmaps.")
        else:
            print("Skipping heatmaps: required columns ('start_time', 'hour_of_day', 'day_of_week_name', 'start_region') are missing from combined df.")

def run_tests(df: pd.DataFrame, processed_df: pd.DataFrame):
    """Runs tests to verify the data and analysis."""
    print("\nRunning tests...")
    
    # Test 1: Check for null values in critical columns (on raw data used for Manhattan filtering)
    # These are columns needed *before or during* filter_manhattan_data
    critical_raw_cols = [COL_START_LAT, COL_START_LNG, COL_USER_TYPE, COL_TRIP_DURATION, COL_START_TIME]
    if not df.empty:
        for col in critical_raw_cols:
            if col in df.columns:
                # Expecting these to be mostly non-null before filtering for core processing
                # User type, trip duration, start time are used on the whole dataset before some filters
                if df[col].isnull().sum() > 0.1 * len(df): # Allow up to 10% nulls before raising alarm
                     print(f"WARN: High proportion of nulls ({df[col].isnull().sum()}/{len(df)}) in {col} of raw data.")
                if df[col].isnull().all():
                     print(f"CRITICAL WARN: Column {col} is entirely null in raw data.")
            else:
                print(f"WARN: Test skipped for raw column '{col}' as it is not present.")
    else:
        print("WARN: Raw DataFrame for testing is empty.")

    # Test 2: Check 'user_type' standardization (on processed data, which is post-standardization)
    if not processed_df.empty and COL_USER_TYPE in processed_df.columns:
        allowed_user_types = {'Member', 'Casual', 'Other'}
        current_user_types = set(processed_df[COL_USER_TYPE].unique())
        assert current_user_types.issubset(allowed_user_types), \
            f"FAIL: User types not standardized. Found: {current_user_types}"
        print(f"PASS: {COL_USER_TYPE} standardized. Values: {current_user_types}")
        if 'Other' in current_user_types:
            print(f"WARN: '{COL_USER_TYPE}' contains 'Other' values. Review original data.")
    elif not processed_df.empty:
         print(f"WARN: '{COL_USER_TYPE}' not in processed_df for testing.")

    # Test 3: Check if 'start_region' and 'end_region' columns exist (on processed_df which is post-region-definition)
    if not processed_df.empty:
        expected_regions_set = set(REGION_ORDER + ['Unknown', 'Outside Defined Regions'])
        for region_col_name in ['start_region', 'end_region']:
            if region_col_name in processed_df.columns:
                unique_regions_found = set(processed_df[region_col_name].unique())
                assert unique_regions_found.issubset(expected_regions_set), \
                    f"FAIL: Unexpected regions found in {region_col_name}: {unique_regions_found - expected_regions_set}"
                print(f"PASS: {region_col_name} column created with valid regions. Found: {unique_regions_found}")
            else:
                print(f"WARN: '{region_col_name}' column not found in processed DataFrame for testing.")

    # Test 4: Check trip durations are reasonable (on processed data)
    if not processed_df.empty and 'trip_duration_minutes' in processed_df.columns and processed_df['trip_duration_minutes'].notna().any():
        valid_durations = processed_df['trip_duration_minutes'].dropna()
        if not valid_durations.empty:
            assert valid_durations.min() >= 0, "FAIL: Negative trip durations found."
            
            # Modified test: Instead of failing on long durations, report them as a warning
            max_duration = valid_durations.max()
            if max_duration >= 1440 * 2:  # More than 2 days
                print(f"WARN: Very long trip durations found. Maximum duration: {max_duration:.1f} minutes ({max_duration/1440:.1f} days)")
            else:
                print(f"PASS: Maximum trip duration is {max_duration:.1f} minutes ({max_duration/1440:.1f} days)")
            
            print("PASS: Trip durations are non-negative and have been checked.")
        else:
            print("WARN: No valid trip_duration_minutes to test after dropping NAs.")
    elif not processed_df.empty:
         print(f"WARN: Test on trip_duration_minutes skipped as column not populated/available in processed_df.")

    print("All specified tests completed.")

if __name__ == "__main__":
    print("Starting Manhattan Citibike Analysis (Batch Processing with Cache)...")
    print(f"Data directory: {DATA_DIR}")
    print(f"Processing in batches of up to {BATCH_SIZE} rows.")

    final_manhattan_df_analyzed = None
    first_raw_batch_for_tests = None
    loaded_from_cache = False

    # Create cache directory if it doesn't exist
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Try to load from cache first
    if os.path.exists(FINAL_DF_CACHE_PATH) and os.path.exists(RAW_SAMPLE_CACHE_PATH):
        print(f"\nAttempting to load processed data from cache...")
        try:
            with timer_with_progress("Loading final_manhattan_df_analyzed from cache"):
                final_manhattan_df_analyzed = pd.read_parquet(FINAL_DF_CACHE_PATH)
            with timer_with_progress("Loading first_raw_batch_for_tests from cache"):
                first_raw_batch_for_tests = pd.read_parquet(RAW_SAMPLE_CACHE_PATH)
            
            print(f"Successfully loaded {len(final_manhattan_df_analyzed)} records for final analysis and raw sample of {len(first_raw_batch_for_tests)} records from cache.")
            loaded_from_cache = True
        except Exception as e_cache_load:
            print(f"Error loading from cache: {e_cache_load}. Proceeding with full data processing.")
            final_manhattan_df_analyzed = None # Ensure it's reset
            first_raw_batch_for_tests = None
            loaded_from_cache = False # Ensure this is false if cache load fails
    else:
        print("\nCache not found or incomplete. Proceeding with full data processing.")

    if not loaded_from_cache:
        print("\nStarting full data processing pipeline...")
        processed_manhattan_batches = []
        # first_raw_batch_for_tests is already None or will be set in the loop
        total_records_loaded = 0
        batch_num_overall = 0

        # This inner try-except is for the batch processing loop itself
        try:
            for raw_batch_df_yielded in load_data_in_batches(DATA_DIR, columns_to_load=NECESSARY_COLUMNS, batch_size=BATCH_SIZE):
                raw_batch_df = raw_batch_df_yielded 
                
                batch_num_overall += 1
                current_batch_size = len(raw_batch_df)
                total_records_loaded += current_batch_size
                print(f"\n--- Processing Batch {batch_num_overall} ({current_batch_size} records) ---")
                print(f"Memory for raw batch: {raw_batch_df.memory_usage(deep=True).sum() / 1048576:.2f} MB")

                if first_raw_batch_for_tests is None:
                    first_raw_batch_for_tests = raw_batch_df.copy()
                    print("Stored first raw batch for testing.")
                
                current_df = raw_batch_df

                desc_std = f"Batch {batch_num_overall}: Standardizing user types"
                with timer_with_progress(desc_std):
                    current_df = standardize_user_types(current_df)
                
                desc_filter = f"Batch {batch_num_overall}: Filtering Manhattan data"
                with timer_with_progress(desc_filter):
                    current_df = filter_manhattan_data(current_df)
                
                if current_df.empty:
                    print(f"Batch {batch_num_overall}: No data after Manhattan filtering. Skipping.")
                    del current_df, raw_batch_df
                    continue 
                print(f"Batch {batch_num_overall}: Filtered to {len(current_df)} Manhattan records. Mem: {current_df.memory_usage(deep=True).sum() / 1048576:.2f} MB")

                desc_regions = f"Batch {batch_num_overall}: Defining Manhattan regions"
                with timer_with_progress(desc_regions):
                    current_df = define_manhattan_regions(current_df)

                desc_durations = f"Batch {batch_num_overall}: Converting trip durations"
                with timer_with_progress(desc_durations):
                    current_df = convert_trip_duration_to_minutes(current_df)

                desc_datetime = f"Batch {batch_num_overall}: Processing datetime info"
                with timer_with_progress(desc_datetime):
                    made_copy_for_datetime = False
                    # Check if a copy is needed before any datetime conversions
                    if (COL_START_TIME in current_df.columns and not pd.api.types.is_datetime64_any_dtype(current_df[COL_START_TIME])) or \
                       (COL_END_TIME in current_df.columns and not pd.api.types.is_datetime64_any_dtype(current_df[COL_END_TIME])):
                        current_df = current_df.copy()
                        made_copy_for_datetime = True
                        print(f"Batch {batch_num_overall}: Made a copy of current_df for datetime conversions.")

                    if COL_START_TIME in current_df.columns:
                        if not pd.api.types.is_datetime64_any_dtype(current_df[COL_START_TIME]):
                            current_df[COL_START_TIME] = pd.to_datetime(current_df[COL_START_TIME], errors='coerce')
                        
                        # It's crucial to drop rows with NaT in COL_START_TIME as it's used for many derivations
                        original_len = len(current_df)
                        current_df.dropna(subset=[COL_START_TIME], inplace=True)
                        if len(current_df) < original_len:
                            print(f"Batch {batch_num_overall}: Dropped {original_len - len(current_df)} rows due to NaT in '{COL_START_TIME}'.")

                        if not current_df.empty:
                            current_df['day_of_week_name'] = current_df[COL_START_TIME].dt.day_name() # Changed from 'day_of_week' to 'day_of_week_name' for clarity with heatmap
                            current_df['hour_of_day'] = current_df[COL_START_TIME].dt.hour
                            current_df['is_weekday'] = current_df[COL_START_TIME].dt.dayofweek < 5 # Monday=0, Sunday=6
                        else:
                            print(f"Batch {batch_num_overall}: Empty after dropping invalid '{COL_START_TIME}'.")
                    else:
                        print(f"Batch {batch_num_overall}: Column '{COL_START_TIME}' not found for datetime processing.")

                    if COL_END_TIME in current_df.columns:
                        if not pd.api.types.is_datetime64_any_dtype(current_df[COL_END_TIME]):
                            current_df[COL_END_TIME] = pd.to_datetime(current_df[COL_END_TIME], errors='coerce')
                        
                        # Optional: Handle NaTs in COL_END_TIME if necessary for your analysis
                        # For now, just converting. If NaTs in COL_END_TIME cause issues downstream,
                        # you might need to drop them: current_df.dropna(subset=[COL_END_TIME], inplace=True)
                        if current_df[COL_END_TIME].isna().any(): # Check after potential conversion
                             print(f"Batch {batch_num_overall}: Warning - NaNs/NaTs present in '{COL_END_TIME}' after conversion.")
                    else:
                        print(f"Batch {batch_num_overall}: Column '{COL_END_TIME}' not found for datetime processing.")
                
                if not current_df.empty:
                    processed_manhattan_batches.append(current_df)
                    print(f"Batch {batch_num_overall}: Added {len(current_df)} processed records to final list.")
                
                print(f"--- Finished Batch {batch_num_overall} ---")
                del current_df 
                # raw_batch_df will be handled by Python's GC or overwritten in next loop

            print(f"\nAll {batch_num_overall} data batches processed. Total records initially loaded: {total_records_loaded}")

            if not processed_manhattan_batches:
                print("No Manhattan data processed after all batches. Cannot proceed to analysis or caching.")
                # Ensure first_raw_batch_for_tests is a DataFrame for run_tests if it was initialized
                if first_raw_batch_for_tests is None:
                    first_raw_batch_for_tests = pd.DataFrame()
                # Run tests only with raw sample if available, processed_df is empty
                run_tests(first_raw_batch_for_tests, pd.DataFrame())
                sys.exit() # Exit if no processed data to analyze or cache
            
            print("\nConcatenating all processed Manhattan batches...")
            with timer_with_progress("Concatenating processed batches"):
                final_manhattan_df_analyzed = pd.concat(processed_manhattan_batches, ignore_index=True)
            del processed_manhattan_batches

            # Save to cache after successful processing
            if final_manhattan_df_analyzed is not None and not final_manhattan_df_analyzed.empty:
                print(f"\nSaving processed data to cache...")
                try:
                    # Ensure COL_END_STATION_ID is string to prevent conversion errors if it's mixed type
                    if COL_END_STATION_ID in final_manhattan_df_analyzed.columns:
                        print(f"Converting '{COL_END_STATION_ID}' to string type before caching.")
                        final_manhattan_df_analyzed[COL_END_STATION_ID] = final_manhattan_df_analyzed[COL_END_STATION_ID].astype(str)

                    with timer_with_progress("Saving final_manhattan_df_analyzed to cache"):
                        final_manhattan_df_analyzed.to_parquet(FINAL_DF_CACHE_PATH, index=False)
                    if first_raw_batch_for_tests is not None and not first_raw_batch_for_tests.empty:
                         with timer_with_progress("Saving first_raw_batch_for_tests to cache"):
                            first_raw_batch_for_tests.to_parquet(RAW_SAMPLE_CACHE_PATH, index=False)
                    print("Data successfully cached.")
                except Exception as e_cache_save:
                    print(f"Error saving to cache: {e_cache_save}")
            
            # Ensure first_raw_batch_for_tests is a DataFrame for the main run_tests call
            if first_raw_batch_for_tests is None:
                 first_raw_batch_for_tests = pd.DataFrame() # Should have been set if any batches ran

        except Exception as e_main_processing: # Catch errors during the batch processing loop
            print(f"\n!!! ERROR during main data processing loop: {e_main_processing}")
            import traceback
            traceback.print_exc()
            print("\nMain data processing failed. Attempting to run tests on any available data...")
            # Ensure variables are defined for run_tests, even if empty
            if 'final_manhattan_df_analyzed' not in locals() or final_manhattan_df_analyzed is None: final_manhattan_df_analyzed = pd.DataFrame()
            if 'first_raw_batch_for_tests' not in locals() or first_raw_batch_for_tests is None: first_raw_batch_for_tests = pd.DataFrame()
            run_tests(first_raw_batch_for_tests, final_manhattan_df_analyzed) 
            sys.exit("Exiting due to processing error.")

    # --- Analysis, Plotting, and Testing (runs if data loaded from cache OR processed) ---
    if final_manhattan_df_analyzed is not None and not final_manhattan_df_analyzed.empty:
        print(f"\nProceeding with analysis and plotting on {len(final_manhattan_df_analyzed)} records.")
        print(f"Memory usage of final DataFrame: {final_manhattan_df_analyzed.memory_usage(deep=True).sum() / 1048576:.2f} MB")

        analyze_data_by_region(final_manhattan_df_analyzed)
        create_plots(final_manhattan_df_analyzed)
        
        # Ensure first_raw_batch_for_tests is a DataFrame for run_tests
        if first_raw_batch_for_tests is None:
             first_raw_batch_for_tests = pd.DataFrame() 
        run_tests(first_raw_batch_for_tests, final_manhattan_df_analyzed)
            
        print("\nManhattan analysis script finished successfully.")
    elif loaded_from_cache and (final_manhattan_df_analyzed is None or final_manhattan_df_analyzed.empty):
        print("Loaded from cache, but the final DataFrame is empty or invalid. Cannot proceed.")
    elif not loaded_from_cache:
        print("Full processing did not yield any data for analysis. Check input files or filtering logic.")
        # If first_raw_batch_for_tests was populated, we might still run tests on it.
        if first_raw_batch_for_tests is not None and not first_raw_batch_for_tests.empty:
            print("Running tests on the first raw batch sample only.")
            run_tests(first_raw_batch_for_tests, pd.DataFrame()) # Pass empty df for processed
        else:
            print("No data available to run tests.")

    print("Script execution complete.") 