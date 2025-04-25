import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Set up better visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

def create_output_dirs():
    """
    Create the necessary output directories.
    """
    # Create results directory if it doesn't exist
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Create ride_distribution directory
    ride_dist_dir = results_dir / 'ride_distribution'
    ride_dist_dir.mkdir(exist_ok=True)
    
    # Create plots directory
    plots_dir = ride_dist_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    return ride_dist_dir, plots_dir

def load_and_process_data(data_dir='data/citibike/combined'):
    """
    Load all parquet files from the specified directory and process them.
    Returns a dictionary with yearly DataFrames containing daily ride counts and hourly data.
    """
    yearly_ride_counts = {}
    all_hourly_data = []
    
    # Get all parquet files in the directory
    parquet_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.parquet')])
    
    for file in parquet_files:
        year = file.split('_')[0]
        print(f"Processing {file}...")
        
        # Read the parquet file
        file_path = os.path.join(data_dir, file)
        try:
            # Read only necessary columns for memory efficiency
            df = pd.read_parquet(file_path, columns=['start_time'])
            
            # Extract date and hour from timestamp
            df['date'] = pd.to_datetime(df['start_time']).dt.date
            df['hour'] = pd.to_datetime(df['start_time']).dt.hour
            
            # Calculate daily rides
            daily_rides = df.groupby('date').size().reset_index(name='ride_count')
            
            # Convert date to datetime for filtering
            daily_rides['date'] = pd.to_datetime(daily_rides['date'])
            
            # Filter to only include dates from the corresponding year
            daily_rides = daily_rides[daily_rides['date'].dt.year == int(year)]
            
            # Convert back to date for plotting
            daily_rides['date'] = daily_rides['date'].dt.date
            
            # Store the result
            yearly_ride_counts[year] = daily_rides
            
            # Calculate hourly rides and add to all_hourly_data
            hourly_rides = df.groupby('hour').size().reset_index(name='ride_count')
            hourly_rides['year'] = year
            all_hourly_data.append(hourly_rides)
            
            print(f"  Found {len(df):,} rides in {year}")
            print(f"  Date range: {daily_rides['date'].min()} to {daily_rides['date'].max()}")
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
    
    # Combine all hourly data
    hourly_df = pd.concat(all_hourly_data, ignore_index=True)
    
    return yearly_ride_counts, hourly_df

def calculate_stats(yearly_ride_counts):
    """
    Calculate statistics for each year's ride data.
    Returns both the stats dictionary and a DataFrame with the statistics.
    """
    stats = {}
    stats_data = []
    
    for year, data in yearly_ride_counts.items():
        median_rides = data['ride_count'].median()
        mean_rides = data['ride_count'].mean()
        max_rides = data['ride_count'].max()
        min_rides = data['ride_count'].min()
        total_rides = data['ride_count'].sum()
        
        stats[year] = {
            'median': median_rides,
            'mean': mean_rides,
            'max': max_rides,
            'min': min_rides,
            'total': total_rides,
            'data': data
        }
        
        stats_data.append({
            'year': year,
            'median_daily_rides': median_rides,
            'mean_daily_rides': mean_rides,
            'max_daily_rides': max_rides,
            'min_daily_rides': min_rides,
            'total_rides': total_rides
        })
        
        print(f"\nStatistics for {year}:")
        print(f"  Median daily rides: {median_rides:,.0f}")
        print(f"  Mean daily rides: {mean_rides:,.0f}")
        print(f"  Maximum daily rides: {max_rides:,.0f}")
        print(f"  Minimum daily rides: {min_rides:,.0f}")
        print(f"  Total rides: {total_rides:,.0f}")
    
    # Convert stats to DataFrame
    stats_df = pd.DataFrame(stats_data)
    stats_df = stats_df.sort_values('year')
    
    return stats, stats_df

def plot_hourly_distribution(hourly_df, plots_dir):
    """
    Create a plot showing the average hourly distribution of rides.
    """
    # Calculate average rides per hour
    hourly_avg = hourly_df.groupby('hour')['ride_count'].mean().reset_index()
    
    plt.figure(figsize=(15, 8))
    
    # Create the bar plot
    bars = plt.bar(hourly_avg['hour'], hourly_avg['ride_count'], 
                  color='#1f77b4', alpha=0.7)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1000,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=10)
    
    # Customize the plot
    plt.title('Average Hourly Distribution of Citibike Rides', fontsize=16)
    plt.xlabel('Hour of the Day', fontsize=14)
    plt.ylabel('Average Number of Rides', fontsize=14)
    
    # Set x-axis ticks to show every hour
    plt.xticks(range(24), [f'{h:02d}:00' for h in range(24)], rotation=45)
    
    # Add grid
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'hourly_ride_distribution.png', dpi=300)
    plt.close()

def plot_daily_rides(yearly_stats, plots_dir):
    """
    Create plots for the daily ride data.
    """
    # Plot 1: Daily rides for each year (combined)
    plt.figure(figsize=(15, 10))
    
    for year, stats in yearly_stats.items():
        data = stats['data']
        plt.plot(data['date'], data['ride_count'], label=f'{year}', alpha=0.7)
    
    plt.title('Daily Citibike Rides by Year', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Number of Rides', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(plots_dir / 'daily_rides_by_year.png', dpi=300)
    plt.close()
    
    # Plot 2: Individual yearly plots
    for year, stats in yearly_stats.items():
        plt.figure(figsize=(15, 8))
        data = stats['data']
        
        plt.plot(data['date'], data['ride_count'], color='#1f77b4', alpha=0.7)
        plt.title(f'Daily Citibike Rides in {year}', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Number of Rides', fontsize=14)
        
        # Add mean line
        mean_rides = data['ride_count'].mean()
        plt.axhline(y=mean_rides, color='r', linestyle='--', alpha=0.7, 
                   label=f'Mean: {mean_rides:,.0f} rides')
        
        # Add median line
        median_rides = data['ride_count'].median()
        plt.axhline(y=median_rides, color='g', linestyle='--', alpha=0.7,
                   label=f'Median: {median_rides:,.0f} rides')
        
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(plots_dir / f'daily_rides_{year}.png', dpi=300)
        plt.close()
    
    # Plot 3: Median daily rides by year (bar chart)
    plt.figure(figsize=(12, 8))
    
    years = list(yearly_stats.keys())
    medians = [stats['median'] for stats in yearly_stats.values()]
    
    bars = plt.bar(years, medians)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1000,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=12)
    
    plt.title('Median Daily Citibike Rides by Year', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Median Number of Rides per Day', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(plots_dir / 'median_rides_by_year.png', dpi=300)
    plt.close()
    
    # Plot 4: Box plots of daily rides by year
    plt.figure(figsize=(14, 8))
    
    data_to_plot = []
    labels = []
    
    for year, stats in yearly_stats.items():
        data_to_plot.append(stats['data']['ride_count'])
        labels.append(year)
    
    plt.boxplot(data_to_plot, tick_labels=labels, showfliers=False)
    
    plt.title('Distribution of Daily Citibike Rides by Year', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Number of Rides per Day', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(plots_dir / 'ride_distribution_by_year.png', dpi=300)
    plt.close()

def main():
    print("Starting Citibike ride analysis...")
    
    # Create output directories
    ride_dist_dir, plots_dir = create_output_dirs()
    
    # Load and process data
    yearly_ride_counts, hourly_df = load_and_process_data()
    
    # Calculate statistics
    yearly_stats, stats_df = calculate_stats(yearly_ride_counts)
    
    # Save statistics to CSV
    stats_df.to_csv(ride_dist_dir / 'ride_distribution.csv', index=False)
    print("\nStatistics saved to 'results/ride_distribution/ride_distribution.csv'")
    
    # Create plots
    plot_daily_rides(yearly_stats, plots_dir)
    plot_hourly_distribution(hourly_df, plots_dir)
    
    print("\nAnalysis complete! Plots have been saved to 'results/ride_distribution/plots' directory.")

if __name__ == "__main__":
    main()
