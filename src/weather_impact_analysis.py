import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from tqdm import tqdm
import pyarrow.parquet as pq
warnings.filterwarnings('ignore')

# Create results directory if it doesn't exist
results_dir = os.path.join('results', 'weather_impact_analysis')
os.makedirs(results_dir, exist_ok=True)

# Set global matplotlib parameters for larger fonts
plt.rcParams.update({
    'font.size': 18,         # Increase base font size
    'axes.titlesize': 24,    # Larger title
    'axes.labelsize': 22,    # Larger axis labels
    'xtick.labelsize': 20,   # Larger x-tick labels
    'ytick.labelsize': 20,   # Larger y-tick labels
    'legend.fontsize': 20,   # Larger legend font
    'figure.titlesize': 26   # Larger figure title
})

# Process data year by year to avoid memory issues
def process_years_separately():
    """Process each year file separately to avoid memory overload"""
    combined_dir = 'data/combined'
    available_files = os.listdir(combined_dir)
    years = [int(filename.split('_')[0]) for filename in available_files if filename.endswith('.parquet')]
    years.sort()
    
    # Initialize DataFrames to store aggregated results
    user_type_results = pd.DataFrame()
    region_results = pd.DataFrame()
    
    # Process each year separately
    for year in tqdm(years, desc="Processing years"):
        print(f"\nProcessing data for {year}...")
        file_path = f'data/combined/{year}_combined_citibike_weather.parquet'
        
        # Read the parquet file metadata to get number of row groups
        parquet_file = pq.ParquetFile(file_path)
        num_row_groups = parquet_file.num_row_groups
        
        # Process each row group separately
        with tqdm(total=num_row_groups, desc=f"Processing {year} data", unit="chunks") as pbar:
            for i in range(num_row_groups):
                # Read one row group at a time
                chunk = parquet_file.read_row_group(i).to_pandas()
                pbar.update(1)
                
                # Standardize user types
                chunk = standardize_user_types(chunk)
                
                # Get hourly ride counts by weather and user type
                user_type_counts = get_hourly_counts(chunk, ['datetime_hour', 'weather_cat', 'user_type'])
                
                # Add region information and get hourly counts by weather and region
                chunk_with_regions = identify_regions(chunk)
                region_counts = get_hourly_counts(chunk_with_regions, ['datetime_hour', 'weather_cat', 'region'])
                
                # Aggregate the results
                user_type_results = pd.concat([user_type_results, user_type_counts], ignore_index=True)
                region_results = pd.concat([region_results, region_counts], ignore_index=True)
                
                # Clear memory
                del chunk, chunk_with_regions
    
    print("\nCalculating ride changes compared to Neutral weather and generating visualizations...")
    
    # Analyze aggregated results
    with tqdm(total=2, desc="Calculating ride changes", unit="analysis") as pbar:
        user_type_changes = calculate_ride_changes_from_neutral(user_type_results, 'user_type')
        pbar.update(1)
        region_changes = calculate_ride_changes_from_neutral(region_results, 'region')
        pbar.update(1)
    
    # Create visualizations and save results
    with tqdm(total=2, desc="Creating visualizations", unit="plot") as pbar:
        create_user_type_visualization(user_type_changes)
        pbar.update(1)
        create_region_visualization(region_changes)
        pbar.update(1)
    
    return user_type_changes, region_changes

# Function to standardize user types
def standardize_user_types(df):
    """Standardize user types: Subscriber/member -> 'Member', Customer/casual -> 'Casual'"""
    df_copy = df.copy()
    
    # Create a mapping dictionary for standardizing user types
    user_type_mapping = {
        'Subscriber': 'Member',
        'member': 'Member',
        'Customer': 'Casual',
        'casual': 'Casual'
    }
    
    # Standardize the user_type column
    if 'user_type' in df_copy.columns:
        df_copy['user_type'] = df_copy['user_type'].map(lambda x: user_type_mapping.get(x, x))
    
    return df_copy

# Function to get hourly ride counts
def get_hourly_counts(df, group_columns):
    """Get hourly ride counts grouped by specified columns"""
    return df.groupby(group_columns).size().reset_index(name='ride_count')

# Function to identify regions based on predefined geographic boundaries
def identify_regions(df):
    """Create regions based on predefined NYC borough boundaries"""
    # Create a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    # Define NYC borough boundaries (approximate)
    # These are approximate boundaries for NYC boroughs
    region_boundaries = {
        'Manhattan': {
            'lat_min': 40.70, 'lat_max': 40.88,
            'long_min': -74.02, 'long_max': -73.91
        },
        'Brooklyn': {
            'lat_min': 40.57, 'lat_max': 40.74,
            'long_min': -74.05, 'long_max': -73.83
        },
        'Queens': {
            'lat_min': 40.54, 'lat_max': 40.80,
            'long_min': -73.96, 'long_max': -73.70
        },
        'Bronx': {
            'lat_min': 40.79, 'lat_max': 40.92,
            'long_min': -73.93, 'long_max': -73.77
        },
        'Staten_Island': {
            'lat_min': 40.49, 'lat_max': 40.65,
            'long_min': -74.26, 'long_max': -74.05
        }
    }
    
    # Initialize region column with Manhattan (default) rather than 'Other'
    # Since Citibike is primarily in Manhattan, this is a reasonable default
    df_copy['region'] = 'Manhattan'
    
    # Assign each station to a borough based on its coordinates
    for region, bounds in region_boundaries.items():
        mask = (
            (df_copy['start_station_latitude'] >= bounds['lat_min']) & 
            (df_copy['start_station_latitude'] <= bounds['lat_max']) & 
            (df_copy['start_station_longitude'] >= bounds['long_min']) & 
            (df_copy['start_station_longitude'] <= bounds['long_max'])
        )
        df_copy.loc[mask, 'region'] = region
    
    return df_copy

# Function to calculate ride changes compared to Neutral weather
def calculate_ride_changes_from_neutral(hourly_counts, group_column):
    """Calculate ride changes by weather category compared to Neutral weather category"""
    # Group by weather category and the specified column (user_type or region)
    avg_rides = hourly_counts.groupby(['weather_cat', group_column])['ride_count'].mean().reset_index()
    
    # Create a separate DataFrame for the Neutral weather data
    neutral_rides = avg_rides[avg_rides['weather_cat'] == 'Neutral'].copy()
    neutral_rides = neutral_rides.rename(columns={'ride_count': 'neutral_ride_count'})
    neutral_rides = neutral_rides.drop('weather_cat', axis=1)
    
    # Merge to calculate the difference relative to Neutral weather
    result = pd.merge(avg_rides, neutral_rides, on=group_column)
    
    # Calculate the percentage change compared to Neutral weather (multiply by 100 to get percentage)
    # Positive values = increase, Negative values = decrease
    result['change_from_neutral'] = ((result['ride_count'] - result['neutral_ride_count']) / result['neutral_ride_count']) * 100
    
    # Save results to CSV
    result.to_csv(os.path.join(results_dir, f'weather_impact_by_{group_column}.csv'), index=False)
    
    return result

# Function to create visualization for user type analysis
def create_user_type_visualization(result):
    """Create and save visualization for user type analysis"""
    plt.figure(figsize=(18, 14))
    
    # Filter out the Neutral weather category (it will always be 0)
    plot_data = result[result['weather_cat'] != 'Neutral'].copy()
    
    # Use a more muted color palette
    sns.set_palette("muted")
    
    chart = sns.barplot(x='weather_cat', y='change_from_neutral', hue='user_type', data=plot_data)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('Percentage Change in Rides Compared to Neutral Weather by User Type', fontsize=26)
    plt.xlabel('Weather Category', fontsize=24)
    plt.ylabel('Percentage Change (+ = Increase, - = Decrease)', fontsize=24)
    plt.xticks(rotation=45, fontsize=22)
    plt.yticks(fontsize=22)
    
    # Make the legend larger
    leg = plt.legend(title='User Type', fontsize=22, title_fontsize=24)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'weather_impact_by_user_type.png'), dpi=300)
    plt.close()

# Function to create visualization for region analysis
def create_region_visualization(result):
    """Create and save visualization for region analysis"""
    plt.figure(figsize=(20, 16))
    
    # Filter out the Neutral weather category (it will always be 0)
    plot_data = result[result['weather_cat'] != 'Neutral'].copy()
    
    # Use a colorblind color palette - with enough distinct colors for boroughs
    palette = sns.color_palette("colorblind", n_colors=5)  # 5 colors for 5 boroughs
    
    chart = sns.barplot(x='weather_cat', y='change_from_neutral', hue='region', data=plot_data, palette=palette)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('Percentage Change in Rides Compared to Neutral Weather by Region', fontsize=26)
    plt.xlabel('Weather Category', fontsize=24)
    plt.ylabel('Percentage Change (+ = Increase, - = Decrease)', fontsize=24)
    plt.xticks(rotation=45, fontsize=22)
    plt.yticks(fontsize=22)
    
    # Make the legend larger and move it outside the plot
    leg = plt.legend(title='Region', fontsize=22, title_fontsize=24, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'weather_impact_by_region.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    print("Starting weather impact analysis with memory-efficient processing...")
    
    # Process all years with memory-efficient approach
    user_type_results, region_results = process_years_separately()
    
    print(f"Analysis complete. Results saved in {results_dir}") 