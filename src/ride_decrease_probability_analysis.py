import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
from datetime import datetime
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

def load_parquet_files(data_path="data/combined"):
    """
    Load all parquet files from the combined data directory.
    
    Args:
        data_path (str): Path to the directory containing parquet files
        
    Returns:
        dict: Dictionary with years as keys and DataFrames as values
    """
    parquet_files = glob.glob(f"{data_path}/*.parquet")
    data_dict = {}
    
    print("Loading parquet files...")
    for file_path in parquet_files:
        # Extract year from filename
        year = int(Path(file_path).stem.split('_')[0])
        print(f"Loading {year} data...")
        
        df = pd.read_parquet(file_path)
        # Convert start_time to datetime if it's not already
        df['start_time'] = pd.to_datetime(df['start_time'])
        # Extract hour from start_time
        df['hour'] = df['start_time'].dt.hour
        # Extract date for daily grouping
        df['date'] = df['start_time'].dt.date
        
        data_dict[year] = df
        print(f"Loaded {len(df):,} records for {year}")
    
    return data_dict

def calculate_daily_ride_counts_by_hour(df, group_cols):
    """
    Calculate daily ride counts by hour for different grouping variables.
    
    Args:
        df (DataFrame): Input dataframe
        group_cols (list): List of columns to group by (including 'hour', 'date')
        
    Returns:
        DataFrame: Aggregated counts by hour, date and grouping variables
    """
    # Group by date, hour and specified columns to get daily counts
    ride_counts = df.groupby(group_cols + ['date']).size().reset_index(name='ride_count')
    return ride_counts

def calculate_decrease_probability_vs_normal(ride_counts, weather_col, user_type_col='user_type'):
    """
    Calculate the probability of decrease in rides for each hour compared to normal weather conditions.
    
    Args:
        ride_counts (DataFrame): DataFrame with daily ride counts by hour and conditions
        weather_col (str): Name of the weather condition column
        user_type_col (str): Name of the user type column
        
    Returns:
        DataFrame: DataFrame with decrease probabilities and significance tests
    """
    results = []
    
    # Define normal weather conditions
    if weather_col == 'weather_cat':
        normal_condition = 'Neutral'
    elif weather_col == 'utci_cat':
        normal_condition = 'No thermal stress'
    else:
        # If we don't know the normal condition, use the most frequent one
        normal_condition = ride_counts[weather_col].mode().iloc[0]
    
    print(f"Using '{normal_condition}' as normal weather condition for {weather_col}")
    
    # For each user type
    for user_type in ride_counts[user_type_col].unique():
        user_data = ride_counts[ride_counts[user_type_col] == user_type].copy()
        
        # Get normal weather data for this user type
        normal_data = user_data[user_data[weather_col] == normal_condition]
        
        if len(normal_data) == 0:
            print(f"No normal weather data found for {user_type}")
            continue
        
        # Calculate average rides per hour under normal conditions
        normal_avg_by_hour = normal_data.groupby('hour')['ride_count'].agg(['mean', 'std', 'count']).reset_index()
        normal_avg_by_hour.columns = ['hour', 'normal_mean', 'normal_std', 'normal_count']
        
        # For each weather condition (excluding normal)
        for weather_condition in user_data[weather_col].unique():
            if weather_condition == normal_condition:
                continue
                
            condition_data = user_data[user_data[weather_col] == weather_condition]
            
            if len(condition_data) == 0:
                continue
            
            # For each hour
            for hour in range(24):
                hour_normal = normal_data[normal_data['hour'] == hour]['ride_count']
                hour_condition = condition_data[condition_data['hour'] == hour]['ride_count']
                
                if len(hour_normal) == 0 or len(hour_condition) == 0:
                    continue
                
                # Calculate probability of decrease
                normal_mean = hour_normal.mean()
                condition_values = hour_condition.values
                
                # Count how many days had fewer rides than normal average
                decreases = np.sum(condition_values < normal_mean)
                total_days = len(condition_values)
                prob_decrease = decreases / total_days if total_days > 0 else 0
                
                # Perform t-test to check statistical significance
                if len(hour_normal) >= 3 and len(hour_condition) >= 3:
                    try:
                        t_stat, p_value = stats.ttest_ind(hour_condition, hour_normal, equal_var=False)
                        is_significant = p_value < 0.05
                        # Check if the condition actually has lower mean
                        condition_lower = hour_condition.mean() < hour_normal.mean()
                    except:
                        p_value = 1.0
                        is_significant = False
                        condition_lower = False
                else:
                    p_value = 1.0
                    is_significant = False
                    condition_lower = hour_condition.mean() < hour_normal.mean() if len(hour_condition) > 0 else False
                
                results.append({
                    'hour': hour,
                    weather_col: weather_condition,
                    user_type_col: user_type,
                    'probability_decrease': prob_decrease,
                    'normal_mean': normal_mean,
                    'condition_mean': hour_condition.mean(),
                    'normal_count': len(hour_normal),
                    'condition_count': len(hour_condition),
                    'p_value': p_value,
                    'is_significant': is_significant,
                    'condition_lower': condition_lower
                })
    
    return pd.DataFrame(results)

def plot_decrease_probabilities(prob_data, weather_col, year, output_dir="results/ride_decrease_probability_analysis"):
    """
    Create plots showing probability of ride decrease by hour for different weather conditions.
    
    Args:
        prob_data (DataFrame): DataFrame with decrease probabilities
        weather_col (str): Name of the weather condition column
        year (int): Year of the data
        output_dir (str): Directory to save plots
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get unique user types
    user_types = prob_data['user_type'].unique()
    
    # Create figure with subplots for each user type
    fig, axes = plt.subplots(1, len(user_types), figsize=(15, 6))
    if len(user_types) == 1:
        axes = [axes]
    
    fig.suptitle(f'Probability of Ride Decrease vs Normal Weather by Hour - {year}\n(Weather Classification: {weather_col})', 
                 fontsize=16, fontweight='bold')
    
    # Define intuitive color mapping for weather conditions
    def get_weather_color_map(weather_conditions, weather_col):
        color_map = {}
        
        if weather_col == 'weather_cat':
            # Intuitive colors for weather categories
            weather_colors = {
                'Neutral': '#2E8B57',      # Sea green (neutral)
                'Cold': '#4169E1',         # Royal blue (cold)
                'Heat': '#DC143C',         # Crimson (hot)
                'Rain': '#191970',         # Midnight blue (rain)
                'Snow': '#B0C4DE',         # Light steel blue (snow)
                'Mist/Fog': '#708090'      # Slate gray (mist/fog)
            }
        elif weather_col == 'utci_cat':
            # Colors for UTCI categories (cold to hot spectrum)
            weather_colors = {
                'Extreme cold stress': '#000080',      # Navy (very cold)
                'Very strong cold stress': '#0000CD',  # Medium blue
                'Strong cold stress': '#4169E1',       # Royal blue
                'Moderate cold stress': '#6495ED',     # Cornflower blue
                'Slight cold stress': '#87CEEB',       # Sky blue
                'No thermal stress': '#2E8B57',        # Sea green (neutral)
                'Moderate heat stress': '#FF6347',     # Tomato
                'Strong heat stress': '#DC143C',       # Crimson
                'Very strong heat stress': '#8B0000',  # Dark red
                'Extreme heat stress': '#B22222'       # Fire brick
            }
        else:
            # Fallback to default colors
            colors = plt.cm.Set3(np.linspace(0, 1, len(weather_conditions)))
            weather_colors = dict(zip(weather_conditions, colors))
        
        # Only include colors for conditions that exist in the data
        for condition in weather_conditions:
            if condition in weather_colors:
                color_map[condition] = weather_colors[condition]
            else:
                # Fallback color for unknown conditions
                color_map[condition] = '#808080'  # Gray
        
        return color_map
    
    # Get weather conditions and color mapping with logical ordering
    def get_logical_order(weather_conditions, weather_col):
        """Define logical ordering for weather conditions"""
        if weather_col == 'utci_cat':
            # Order from coldest to hottest
            utci_order = [
                'Extreme cold stress',
                'Very strong cold stress', 
                'Strong cold stress',
                'Moderate cold stress',
                'Slight cold stress',
                'No thermal stress',
                'Moderate heat stress',
                'Strong heat stress',
                'Very strong heat stress',
                'Extreme heat stress'
            ]
            # Only include categories that exist in the data, in the defined order
            ordered_conditions = [cat for cat in utci_order if cat in weather_conditions]
            # Add any categories not in our predefined order at the end
            remaining = [cat for cat in weather_conditions if cat not in utci_order]
            return ordered_conditions + sorted(remaining)
        
        elif weather_col == 'weather_cat':
            # Order from neutral to more extreme conditions
            weather_order = [
                'Neutral',
                'Cold',
                'Heat',
                'Mist/Fog',
                'Rain',
                'Snow'
            ]
            # Only include categories that exist in the data, in the defined order
            ordered_conditions = [cat for cat in weather_order if cat in weather_conditions]
            # Add any categories not in our predefined order at the end
            remaining = [cat for cat in weather_conditions if cat not in weather_order]
            return ordered_conditions + sorted(remaining)
        
        else:
            # Default alphabetical sorting for unknown weather columns
            return sorted(weather_conditions)
    
    unique_conditions = [wc for wc in prob_data[weather_col].unique() if pd.notna(wc)]
    weather_conditions = get_logical_order(unique_conditions, weather_col)
    color_map = get_weather_color_map(weather_conditions, weather_col)
    
    # Track legend elements (only need one set since they're the same for all subplots)
    legend_elements = []
    legend_labels = []
    
    for i, user_type in enumerate(user_types):
        ax = axes[i]
        
        # Filter data for this user type
        user_data = prob_data[prob_data['user_type'] == user_type]
        
        # Plot lines for each weather condition
        for weather_condition in weather_conditions:
            condition_data = user_data[user_data[weather_col] == weather_condition]
            if len(condition_data) > 0:
                # Sort by hour for proper line plotting
                condition_data = condition_data.sort_values('hour')
                
                # Plot the main line
                line = ax.plot(condition_data['hour'], condition_data['probability_decrease'],
                       marker='o', linewidth=2, markersize=6,
                       color=color_map[weather_condition],
                       label=f'{weather_condition}')[0]
                
                # Add to legend elements only once (from first subplot)
                if i == 0:
                    legend_elements.append(line)
                    legend_labels.append(weather_condition)
                
                # Add stars for significant points
                significant_data = condition_data[
                    condition_data['is_significant'] & condition_data['condition_lower']
                ]
                if len(significant_data) > 0:
                    ax.scatter(significant_data['hour'], significant_data['probability_decrease'],
                             marker='*', s=200, color=color_map[weather_condition],
                             edgecolors='black', linewidth=1, zorder=5)
        
        ax.set_xlabel('Hour of Day', fontsize=12)
        ax.set_ylabel('Probability of Decrease vs Normal Weather', fontsize=12)
        ax.set_title(f'{user_type}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 23)
        ax.set_ylim(0, 1)
        
        # Set x-axis ticks to show all hours
        ax.set_xticks(range(0, 24, 2))
    
    # Add single legend for the entire figure
    from matplotlib.lines import Line2D
    star_legend = Line2D([0], [0], marker='*', color='black', linestyle='None',
                       markersize=10, label='Statistically significant (p<0.05)')
    legend_elements.append(star_legend)
    legend_labels.append('Statistically significant (p<0.05)')
    
    # Place legend to the right of the figure
    fig.legend(legend_elements, legend_labels, bbox_to_anchor=(1.02, 0.5), loc='center left')
    
    plt.tight_layout()
    
    # Save plot
    filename = f"{output_dir}/ride_decrease_probability_{weather_col}_{year}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {filename}")
    plt.close()  # Close the figure to free memory

def analyze_weather_impact(data_dict):
    """
    Main function to analyze weather impact on ride decreases.
    
    Args:
        data_dict (dict): Dictionary with years as keys and DataFrames as values
    """
    print("\nStarting weather impact analysis...")
    
    for year, df in data_dict.items():
        print(f"\nAnalyzing year {year}...")
        print(f"Data shape: {df.shape}")
        
        # Check for required columns
        if 'weather_cat' not in df.columns or 'utci_cat' not in df.columns:
            print(f"Missing required columns in {year} data. Skipping...")
            continue
        
        # Remove rows with missing values in key columns - do this more efficiently
        required_cols = ['hour', 'user_type', 'weather_cat', 'utci_cat', 'date']
        print("Filtering data for required columns...")
        
        # Check which columns actually exist and have data
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Missing columns {missing_cols} in {year} data. Skipping...")
            continue
        
        # More memory-efficient filtering
        mask = df[required_cols].notna().all(axis=1)
        df_clean = df[mask].copy()
        print(f"Clean data shape after removing NAs: {df_clean.shape}")
        
        if len(df_clean) == 0:
            print(f"No clean data for {year}. Skipping...")
            continue
        
        # Analysis 1: Using weather_cat column
        print(f"Analyzing with weather_cat column...")
        try:
            ride_counts_weather = calculate_daily_ride_counts_by_hour(
                df_clean, ['hour', 'user_type', 'weather_cat']
            )
            
            if len(ride_counts_weather) > 0:
                prob_data_weather = calculate_decrease_probability_vs_normal(
                    ride_counts_weather, 'weather_cat'
                )
                
                if len(prob_data_weather) > 0:
                    print(f"Weather categories found: {sorted(df_clean['weather_cat'].unique())}")
                    print(f"Generated {len(prob_data_weather)} probability data points for weather_cat")
                    plot_decrease_probabilities(prob_data_weather, 'weather_cat', year)
                else:
                    print(f"No probability data generated for weather_cat in {year}")
        except Exception as e:
            print(f"Error processing weather_cat for {year}: {str(e)}")
        
        # Analysis 2: Using utci_cat column
        print(f"Analyzing with utci_cat column...")
        try:
            ride_counts_utci = calculate_daily_ride_counts_by_hour(
                df_clean, ['hour', 'user_type', 'utci_cat']
            )
            
            if len(ride_counts_utci) > 0:
                prob_data_utci = calculate_decrease_probability_vs_normal(
                    ride_counts_utci, 'utci_cat'
                )
                
                if len(prob_data_utci) > 0:
                    print(f"UTCI categories found: {sorted(df_clean['utci_cat'].unique())}")
                    print(f"Generated {len(prob_data_utci)} probability data points for utci_cat")
                    plot_decrease_probabilities(prob_data_utci, 'utci_cat', year)
                else:
                    print(f"No probability data generated for utci_cat in {year}")
        except Exception as e:
            print(f"Error processing utci_cat for {year}: {str(e)}")
        
        # Clear variables to free memory
        del df_clean
        if 'ride_counts_weather' in locals():
            del ride_counts_weather
        if 'ride_counts_utci' in locals():
            del ride_counts_utci
        if 'prob_data_weather' in locals():
            del prob_data_weather
        if 'prob_data_utci' in locals():
            del prob_data_utci
        
        import gc
        gc.collect()
        print(f"Completed analysis for {year}")

def generate_summary_statistics(data_dict):
    """
    Generate summary statistics for the analysis.
    
    Args:
        data_dict (dict): Dictionary with years as keys and DataFrames as values
    """
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    for year, df in data_dict.items():
        print(f"\nYear {year}:")
        print(f"  Total rides: {len(df):,}")
        print(f"  User types: {df['user_type'].value_counts().to_dict()}")
        
        if 'weather_cat' in df.columns:
            print(f"  Weather categories: {df['weather_cat'].value_counts().to_dict()}")
        
        if 'utci_cat' in df.columns:
            print(f"  UTCI categories: {df['utci_cat'].value_counts().to_dict()}")

def main():
    """
    Main function to run the complete analysis.
    """
    print("="*60)
    print("BIKE RIDE DECREASE PROBABILITY ANALYSIS")
    print("="*60)
    
    # Load and process data one year at a time to manage memory
    data_path = "data/combined"
    parquet_files = glob.glob(f"{data_path}/*.parquet")
    
    if not parquet_files:
        print("No data files found!")
        return
    
    # Sort files to process in order
    parquet_files.sort()
    
    print("="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # First pass: show summary statistics
    for file_path in parquet_files:
        year = int(Path(file_path).stem.split('_')[0])
        print(f"\nLoading {year} data for summary...")
        
        try:
            df = pd.read_parquet(file_path)
            print(f"Year {year}:")
            print(f"  Total rides: {len(df):,}")
            print(f"  User types: {df['user_type'].value_counts().to_dict()}")
            
            if 'weather_cat' in df.columns:
                print(f"  Weather categories: {df['weather_cat'].value_counts().to_dict()}")
            
            if 'utci_cat' in df.columns:
                print(f"  UTCI categories: {df['utci_cat'].value_counts().to_dict()}")
            
            del df
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"Error loading {year}: {str(e)}")
    
    print("\nStarting weather impact analysis...")
    
    # Second pass: process each year individually
    for file_path in parquet_files:
        year = int(Path(file_path).stem.split('_')[0])
        
        try:
            print(f"\nLoading and analyzing {year} data...")
            df = pd.read_parquet(file_path)
            
            # Convert start_time to datetime if it's not already
            df['start_time'] = pd.to_datetime(df['start_time'])
            # Extract hour from start_time
            df['hour'] = df['start_time'].dt.hour
            # Extract date for daily grouping
            df['date'] = df['start_time'].dt.date
            
            # Create a temporary dictionary for this year
            temp_data_dict = {year: df}
            
            # Analyze this year
            analyze_weather_impact(temp_data_dict)
            
            # Clean up
            del df
            del temp_data_dict
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"Error processing {year}: {str(e)}")
            continue
    
    print(f"\nAnalysis complete! Check the 'results/ride_decrease_probability_analysis' folder for generated plots.")
    print(f"Plots are saved as PNG files with naming convention:")
    print(f"  - ride_decrease_probability_weather_cat_YEAR.png")
    print(f"  - ride_decrease_probability_utci_cat_YEAR.png")

if __name__ == "__main__":
    main() 