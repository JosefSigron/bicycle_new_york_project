import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
import os
import glob

def get_borough_vectorized(lats: np.ndarray, lons: np.ndarray, nyc_boroughs: Dict[str, Tuple[float, float]]) -> np.ndarray:
    """
    Vectorized function to determine which NYC borough coordinates belong to.
    Much faster than applying a function row-by-row.
    
    Args:
        lats: Array of latitude coordinates
        lons: Array of longitude coordinates  
        nyc_boroughs: Dictionary mapping borough names to their center coordinates
        
    Returns:
        Array of borough names
    """
    # Create arrays for borough coordinates
    borough_names = list(nyc_boroughs.keys())
    borough_lats = np.array([nyc_boroughs[name][0] for name in borough_names])
    borough_lons = np.array([nyc_boroughs[name][1] for name in borough_names])
    
    # Initialize result array
    result = np.full(len(lats), -1, dtype=int)  # -1 for unknown
    
    # Handle NaN values
    valid_mask = ~(np.isnan(lats) | np.isnan(lons))
    
    if np.any(valid_mask):
        valid_lats = lats[valid_mask]
        valid_lons = lons[valid_mask]
        
        # Calculate distances to all boroughs for all valid points at once
        # Broadcasting: (n_points, 1) - (1, n_boroughs) = (n_points, n_boroughs)
        lat_diff = valid_lats[:, np.newaxis] - borough_lats[np.newaxis, :]
        lon_diff = valid_lons[:, np.newaxis] - borough_lons[np.newaxis, :]
        distances = np.sqrt(lat_diff**2 + lon_diff**2)
        
        # Find closest borough for each point
        closest_borough_idx = np.argmin(distances, axis=1)
        min_distances = np.min(distances, axis=1)
        
        # Only assign if distance is reasonable (< 0.3 degrees ≈ 20 miles)
        reasonable_mask = min_distances < 0.3
        result[valid_mask] = np.where(reasonable_mask, closest_borough_idx, -1)
    
    # Convert to borough names
    borough_result = np.full(len(lats), 'Unknown', dtype=object)
    for i, name in enumerate(borough_names):
        borough_result[result == i] = name
    
    return borough_result

def process_file_in_chunks(file_path: str, chunk_size: int = 100000) -> pd.DataFrame:
    """
    Process a large parquet file in chunks to manage memory efficiently.
    
    Args:
        file_path: Path to the parquet file
        chunk_size: Number of rows to process at once
        
    Returns:
        DataFrame with start_borough and end_borough columns
    """
    print(f"Processing {os.path.basename(file_path)} in chunks...")
    
    # Define NYC borough centers
    nyc_boroughs = {
        'Bronx': (40.8448, -73.8648),
        'Brooklyn': (40.6782, -73.9442),
        'Manhattan': (40.7831, -73.9712),
        'Queens': (40.7282, -73.7949)
    }
    
    # Read the file info first
    df_info = pd.read_parquet(file_path, columns=['start_station_latitude'])
    total_rows = len(df_info)
    del df_info  # Free memory
    
    print(f"  Total rows: {total_rows:,}")
    
    all_borough_data = []
    
    # Process in chunks
    for start_idx in range(0, total_rows, chunk_size):
        end_idx = min(start_idx + chunk_size, total_rows)
        
        print(f"  Processing rows {start_idx:,} to {end_idx:,}")
        
        # Read only necessary columns for this chunk
        chunk = pd.read_parquet(
            file_path,
            columns=['start_station_latitude', 'start_station_longitude', 
                    'end_station_latitude', 'end_station_longitude', 'user_type']
        ).iloc[start_idx:end_idx]
        
        # Filter out rows with missing coordinates
        valid_coords = chunk.dropna()
        
        if len(valid_coords) == 0:
            continue
        
        # Vectorized borough assignment
        start_boroughs = get_borough_vectorized(
            valid_coords['start_station_latitude'].values,
            valid_coords['start_station_longitude'].values,
            nyc_boroughs
        )
        
        end_boroughs = get_borough_vectorized(
            valid_coords['end_station_latitude'].values,
            valid_coords['end_station_longitude'].values,
            nyc_boroughs
        )
        
        # Normalize user types
        user_types = valid_coords['user_type'].values
        normalized_user_types = np.where(
            np.isin(user_types, ['Subscriber', 'member']), 
            'Subscriber/Member',
            np.where(
                np.isin(user_types, ['Customer', 'casual']),
                'Customer/Casual',
                'Other'
            )
        )
        
        # Create DataFrame with borough assignments and user types
        borough_chunk = pd.DataFrame({
            'start_borough': start_boroughs,
            'end_borough': end_boroughs,
            'user_type_normalized': normalized_user_types
        })
        
        # Keep only rides within known boroughs
        valid_borough_chunk = borough_chunk[
            (borough_chunk['start_borough'] != 'Unknown') & 
            (borough_chunk['end_borough'] != 'Unknown')
        ]
        
        if len(valid_borough_chunk) > 0:
            all_borough_data.append(valid_borough_chunk)
        
        print(f"    Found {len(valid_borough_chunk):,} valid borough rides in this chunk")
    
    if not all_borough_data:
        return pd.DataFrame(columns=['start_borough', 'end_borough', 'user_type_normalized'])
    
    result = pd.concat(all_borough_data, ignore_index=True)
    print(f"  Total valid rides: {len(result):,}")
    
    return result

def analyze_borough_rides_fast(data_folder: str, year: str = None, sample_fraction: float = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fast analysis of rides between NYC boroughs using chunked processing.
    
    Args:
        data_folder: Path to folder containing parquet files
        year: Specific year to analyze (e.g., '2024'), or None for all years
        sample_fraction: If provided, randomly sample this fraction of data (0.1 = 10%)
        
    Returns:
        Tuple of (full_percentages, percentage_matrix, ride_counts)
    """
    # Get list of parquet files
    if year:
        parquet_files = [os.path.join(data_folder, f"{year}_citibike_data.parquet")]
    else:
        parquet_files = glob.glob(os.path.join(data_folder, "*_citibike_data.parquet"))
    
    print(f"Processing {len(parquet_files)} file(s)...")
    
    all_data = []
    
    for file_path in parquet_files:
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found")
            continue
        
        # Process file in chunks
        borough_data = process_file_in_chunks(file_path)
        
        # Optional sampling for even faster processing
        if sample_fraction and len(borough_data) > 0:
            n_sample = int(len(borough_data) * sample_fraction)
            borough_data = borough_data.sample(n=n_sample, random_state=42)
            print(f"  Sampled {len(borough_data):,} rides ({sample_fraction*100:.1f}% of total)")
        
        if len(borough_data) > 0:
            all_data.append(borough_data)
    
    # Combine all data
    if not all_data:
        raise ValueError("No valid data found in the specified files")
    
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal rides between boroughs: {len(combined_data):,}")
    
    # Create overall crosstab to count rides between boroughs
    print("Creating overall ride count matrix...")
    ride_counts = pd.crosstab(
        combined_data['start_borough'], 
        combined_data['end_borough'], 
        margins=True
    )
    
    # Calculate overall percentages
    total_rides = ride_counts.loc['All', 'All']
    ride_percentages = (ride_counts / total_rides * 100).round(2)
    
    # Create the final borough-to-borough matrix
    borough_names = ['Bronx', 'Brooklyn', 'Manhattan', 'Queens']
    percentage_matrix = ride_percentages.loc[borough_names, borough_names]
    
    # Create separate matrices for each user type
    user_type_results = {}
    user_types = combined_data['user_type_normalized'].unique()
    
    for user_type in user_types:
        if user_type == 'Other':
            continue  # Skip 'Other' category
            
        print(f"Creating matrix for {user_type}...")
        user_data = combined_data[combined_data['user_type_normalized'] == user_type]
        
        user_ride_counts = pd.crosstab(
            user_data['start_borough'], 
            user_data['end_borough'], 
            margins=True
        )
        
        user_total_rides = user_ride_counts.loc['All', 'All']
        user_ride_percentages = (user_ride_counts / user_total_rides * 100).round(2)
        user_percentage_matrix = user_ride_percentages.loc[borough_names, borough_names]
        
        user_type_results[user_type] = {
            'percentages': user_ride_percentages,
            'matrix': user_percentage_matrix,
            'counts': user_ride_counts
        }
        
        print(f"  {user_type}: {user_total_rides:,} rides")
    
    return ride_percentages, percentage_matrix, ride_counts, user_type_results

def plot_borough_matrix(percentage_matrix: pd.DataFrame, title_suffix: str = ""):
    """
    Create a heatmap visualization of the borough ride percentage matrix.
    """
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(
        percentage_matrix, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues',
        cbar_kws={'label': 'Percentage of Total Rides (%)'},
        square=True,
        linewidths=0.5
    )
    
    plt.title(f'NYC Citibike Rides: Start vs End Borough Matrix{title_suffix}')
    plt.xlabel('End Borough')
    plt.ylabel('Start Borough')
    plt.tight_layout()
    
    # Save the plot
    output_path = f"results/borough_ride_matrix/borough_ride_matrix{title_suffix.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    plt.show()

def main():
    """Main function with options for different analysis speeds."""
    data_folder = "data/citibike/combined"
    
    print("NYC Borough Ride Analysis (Optimized Version)")
    print("=" * 60)
    
    print("\nAnalyzing all years of data...")
    full_percentages, percentage_matrix, ride_counts, user_type_results = analyze_borough_rides_fast(
        data_folder
    )
    title_suffix = " (All Years)"
    
    print("\nResults:")
    print("=" * 40)
    print("\nBorough-to-Borough Percentage Matrix:")
    print(percentage_matrix)
    
    print("\nRide Counts Matrix:")
    print(ride_counts)
    
    print("\nFull Results (including totals):")
    print(full_percentages)
    
    # Create visualization for overall data
    plot_borough_matrix(percentage_matrix, title_suffix)
    
    # Save overall results to CSV
    output_file = f"results/borough_ride_matrix/borough_ride_analysis{title_suffix.replace(' ', '_').replace('(', '').replace(')', '')}.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    percentage_matrix.to_csv(output_file)
    print(f"\nOverall results saved to: {output_file}")
    
    # Create visualizations and save results for each user type
    print("\n" + "="*60)
    for user_type, results in user_type_results.items():
        print(f"\nAnalyzing {user_type}:")
        print("-" * 40)
        
        user_matrix = results['matrix']
        user_counts = results['counts']
        
        print(f"\n{user_type} Borough-to-Borough Matrix (percentages):")
        print(user_matrix)
        
        # Create visualization for this user type
        user_title_suffix = f"{title_suffix} - {user_type}"
        plot_borough_matrix(user_matrix, user_title_suffix)
        
        # Save user type results to CSV
        user_output_file = f"results/borough_ride_matrix/borough_ride_analysis_{user_type.replace('/', '_')}{title_suffix.replace(' ', '_').replace('(', '').replace(')', '')}.csv"
        user_matrix.to_csv(user_output_file)
        print(f"Results saved to: {user_output_file}")
        
        # Show top patterns for this user type
        print(f"\nTop ride patterns for {user_type}:")
        user_flat_matrix = user_matrix.stack().sort_values(ascending=False)
        for i, (pattern, percentage) in enumerate(user_flat_matrix.head(5).items()):
            start, end = pattern
            print(f"{i+1}. {start} → {end}: {percentage:.2f}%")
    
    # Show overall summary statistics
    print("\n" + "="*60)
    print("OVERALL SUMMARY:")
    print(f"Total rides analyzed: {ride_counts.loc['All', 'All']:,}")
    
    print("\nRides by user type:")
    for user_type, results in user_type_results.items():
        user_total = results['counts'].loc['All', 'All']
        percentage = (user_total / ride_counts.loc['All', 'All']) * 100
        print(f"  {user_type}: {user_total:,} rides ({percentage:.1f}%)")
    
    print("\nOverall most common ride patterns:")
    flat_matrix = percentage_matrix.stack().sort_values(ascending=False)
    for i, (pattern, percentage) in enumerate(flat_matrix.head(5).items()):
        start, end = pattern
        print(f"{i+1}. {start} → {end}: {percentage:.2f}%")

if __name__ == "__main__":
    main() 