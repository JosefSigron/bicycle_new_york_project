import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path
import argparse
import matplotlib.colors as mcolors

#IMPORTANT: Don't run this unless completely necessary. It takes a long time to run and freezes the computer.


# Increase font sizes for better readability
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})

# Create results directory if it doesn't exist
results_dir = Path('results/ride_weather_distribution')
results_dir.mkdir(parents=True, exist_ok=True)

# Define UTCI category names for better labels
utci_categories = {
    'extreme_cold_stress': 'Extreme Cold Stress',
    'very_strong_cold_stress': 'Very Strong Cold Stress',
    'strong_cold_stress': 'Strong Cold Stress',
    'moderate_cold_stress': 'Moderate Cold Stress',
    'slight_cold_stress': 'Slight Cold Stress',
    'no_thermal_stress': 'No Thermal Stress',
    'moderate_heat_stress': 'Moderate Heat Stress',
    'strong_heat_stress': 'Strong Heat Stress',
    'very_strong_heat_stress': 'Very Strong Heat Stress',
    'extreme_heat_stress': 'Extreme Heat Stress'
}

# Define color mapping for UTCI categories - blue for cold, red for hot
utci_colors = {
    'extreme_cold_stress': '#0000FF',      # Deep blue
    'very_strong_cold_stress': '#1E90FF',  # Dodger blue
    'strong_cold_stress': '#4169E1',       # Royal blue
    'moderate_cold_stress': '#6495ED',     # Cornflower blue
    'slight_cold_stress': '#87CEEB',       # Sky blue
    'no_thermal_stress': '#32CD32',        # Lime green
    'moderate_heat_stress': '#FFA07A',     # Light salmon
    'strong_heat_stress': '#FF6347',       # Tomato
    'very_strong_heat_stress': '#FF4500',  # Orange red
    'extreme_heat_stress': '#FF0000'       # Red
}

# Define weather category names for better labels
weather_categories = {
    'clear': 'Clear',
    'partly_cloudy': 'Partly Cloudy',
    'mostly_cloudy': 'Mostly Cloudy',
    'cloudy': 'Cloudy',
    'fog': 'Fog',
    'light_rain': 'Light Rain',
    'rain': 'Rain',
    'heavy_rain': 'Heavy Rain',
    'thunderstorm': 'Thunderstorm',
    'snow': 'Snow',
    'sleet': 'Sleet',
    'freezing_rain': 'Freezing Rain'
}

# Define color mapping for weather categories
weather_colors = {
    'clear': '#FFD700',           # Gold/yellow for clear sky
    'partly_cloudy': '#87CEEB',   # Sky blue 
    'mostly_cloudy': '#708090',   # Slate gray
    'cloudy': '#A9A9A9',          # Dark gray
    'fog': '#D3D3D3',             # Light gray
    'light_rain': '#ADD8E6',      # Light blue
    'rain': '#4682B4',            # Steel blue
    'heavy_rain': '#000080',      # Navy blue
    'thunderstorm': '#800080',    # Purple
    'snow': '#FFFFFF',            # White
    'sleet': '#E0FFFF',           # Light cyan
    'freezing_rain': '#B0E0E6'    # Powder blue
}

# Define user type colors
user_type_colors = {
    'Subscriber': '#228B22',      # Forest green
    'Customer': '#FF8C00',        # Dark orange
    'member': '#228B22',          # Forest green (alternate name)
    'casual': '#FF8C00'           # Dark orange (alternate name)
}

def create_utci_graphs():
    """Create line graphs showing ride distribution by UTCI categories."""
    print("Creating graphs by UTCI categories...")
    
    # Get list of parquet files
    parquet_files = glob.glob('data/combined/*_combined_citibike_weather.parquet')
    
    # Create one graph per year
    for file_path in sorted(parquet_files):
        year = os.path.basename(file_path).split('_')[0]
        print(f"Processing year {year} for UTCI categories...")
        
        # Load the parquet file
        df = pd.read_parquet(file_path)
        
        # Create a datetime column from the datetime_hour to get hour of day
        df['hour_of_day'] = pd.to_datetime(df['datetime_hour']).dt.hour
        
        # Get total rides by hour of day (without filtering)
        total_by_hour = df.groupby('hour_of_day').size()
        
        # Convert to millions for readability
        total_by_hour = total_by_hour / 1_000_000
        
        # Get unique utci_cat values and prepare for plotting
        utci_cats = df['utci_cat'].dropna().unique()
        
        # Create the plot
        plt.figure(figsize=(14, 10))
        
        # Plot line for all rides
        plt.plot(total_by_hour.index, total_by_hour.values, label='All rides', linewidth=3, color='black')
        
        # Plot a line for each UTCI category
        for utci_cat in sorted(utci_cats):
            # Filter data for this utci_cat
            category_data = df[df['utci_cat'] == utci_cat]
            
            # Group by hour of day and convert to millions
            rides_by_hour = category_data.groupby('hour_of_day').size() / 1_000_000
            
            # Get the label from our mapping, or use the original value if not found
            label = utci_categories.get(utci_cat, utci_cat)
            
            # Get color for this category
            color = utci_colors.get(utci_cat, None)
            
            # Plot the line
            plt.plot(rides_by_hour.index, rides_by_hour.values, label=label, color=color)
        
        # Add labels and title
        plt.xlabel('Hour of Day')
        plt.ylabel('Number of Rides (in millions)')
        plt.title(f'Distribution of Rides by Hour and Thermal Comfort - {year}')
        plt.xticks(range(0, 24))
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(f'results/ride_weather_distribution/ride_distribution_by_utci_{year}.png')
        plt.close()
    
    print("UTCI graphs created successfully.")

def create_weather_graphs():
    """Create line graphs showing ride distribution by weather categories."""
    print("Creating graphs by weather categories...")
    
    # Get list of parquet files
    parquet_files = glob.glob('data/combined/*_combined_citibike_weather.parquet')
    
    # Create one graph per year
    for file_path in sorted(parquet_files):
        year = os.path.basename(file_path).split('_')[0]
        print(f"Processing year {year} for weather categories...")
        
        # Load the parquet file
        df = pd.read_parquet(file_path)
        
        # Create a datetime column from the datetime_hour to get hour of day
        df['hour_of_day'] = pd.to_datetime(df['datetime_hour']).dt.hour
        
        # Get total rides by hour of day (without filtering)
        total_by_hour = df.groupby('hour_of_day').size()
        
        # Convert to millions for readability
        total_by_hour = total_by_hour / 1_000_000
        
        # Get unique weather_cat values and prepare for plotting
        weather_cats = df['weather_cat'].dropna().unique()
        
        # Create the plot
        plt.figure(figsize=(14, 10))
        
        # Plot line for all rides
        plt.plot(total_by_hour.index, total_by_hour.values, label='All rides', linewidth=3, color='black')
        
        # Plot a line for each weather category
        for weather_cat in sorted(weather_cats):
            # Filter data for this weather_cat
            category_data = df[df['weather_cat'] == weather_cat]
            
            # Group by hour of day and convert to millions
            rides_by_hour = category_data.groupby('hour_of_day').size() / 1_000_000
            
            # Get the label from our mapping, or use the original value if not found
            label = weather_categories.get(weather_cat, weather_cat)
            
            # Get color for this category
            color = weather_colors.get(weather_cat, None)
            
            # Plot the line
            plt.plot(rides_by_hour.index, rides_by_hour.values, label=label, color=color)
        
        # Add labels and title
        plt.xlabel('Hour of Day')
        plt.ylabel('Number of Rides (in millions)')
        plt.title(f'Distribution of Rides by Hour and Weather Conditions - {year}')
        plt.xticks(range(0, 24))
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(f'results/ride_weather_distribution/ride_distribution_by_weather_{year}.png')
        plt.close()
    
    print("Weather category graphs created successfully.")

def create_weather_by_user_type_graphs():
    """Create line graphs showing ride distribution by weather categories for each user type."""
    print("Creating graphs by weather categories and user types...")
    
    # Get list of parquet files
    parquet_files = glob.glob('data/combined/*_combined_citibike_weather.parquet')
    
    # Create one graph per year and user type
    for file_path in sorted(parquet_files):
        year = os.path.basename(file_path).split('_')[0]
        print(f"Processing year {year} for weather categories by user type...")
        
        # Load the parquet file
        df = pd.read_parquet(file_path)
        
        # Create a datetime column from the datetime_hour to get hour of day
        df['hour_of_day'] = pd.to_datetime(df['datetime_hour']).dt.hour
        
        # Get unique user types
        user_types = df['user_type'].dropna().unique()
        
        # Process each user type separately
        for user_type in sorted(user_types):
            print(f"  Processing user type: {user_type}")
            
            # Filter data for this user type
            user_df = df[df['user_type'] == user_type]
            
            # Get total rides by hour of day for this user type
            total_by_hour = user_df.groupby('hour_of_day').size()
            
            # Convert to millions for readability
            total_by_hour = total_by_hour / 1_000_000
            
            # Get unique weather_cat values and prepare for plotting
            weather_cats = user_df['weather_cat'].dropna().unique()
            
            # Create the plot
            plt.figure(figsize=(14, 10))
            
            # Plot line for all rides of this user type
            user_color = user_type_colors.get(user_type, 'black')
            plt.plot(total_by_hour.index, total_by_hour.values, 
                     label=f'All {user_type} rides', linewidth=3, color=user_color)
            
            # Plot a line for each weather category
            for weather_cat in sorted(weather_cats):
                # Filter data for this weather_cat
                category_data = user_df[user_df['weather_cat'] == weather_cat]
                
                # Group by hour of day and convert to millions
                rides_by_hour = category_data.groupby('hour_of_day').size() / 1_000_000
                
                # Get the label from our mapping, or use the original value if not found
                label = weather_categories.get(weather_cat, weather_cat)
                
                # Get color for this category
                color = weather_colors.get(weather_cat, None)
                
                # Plot the line
                plt.plot(rides_by_hour.index, rides_by_hour.values, label=label, color=color)
            
            # Add labels and title
            plt.xlabel('Hour of Day')
            plt.ylabel('Number of Rides (in millions)')
            plt.title(f'Distribution of {user_type} Rides by Hour and Weather Conditions - {year}')
            plt.xticks(range(0, 24))
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(f'results/ride_weather_distribution/ride_distribution_by_weather_{user_type}_{year}.png')
            plt.close()
    
    print("Weather by user type graphs created successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate ride distribution graphs by weather metrics')
    parser.add_argument('--type', type=str, 
                      choices=['utci', 'weather', 'weather_user', 'all'], 
                      default='all',
                      help='Type of graphs to create: utci, weather, weather_user, or all (default)')
    
    args = parser.parse_args()
    
    if args.type == 'utci' or args.type == 'all':
        create_utci_graphs()
    
    if args.type == 'weather' or args.type == 'all':
        create_weather_graphs()
        
    if args.type == 'weather_user' or args.type == 'all':
        create_weather_by_user_type_graphs()
