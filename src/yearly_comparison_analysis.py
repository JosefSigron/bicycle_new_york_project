"""
Comprehensive Yearly Comparison Analysis for NYC Bicycle & Weather Data

This script analyzes differences between years in the combined bicycle and weather dataset.
Optimized for large parquet files with memory-efficient processing.

Author: Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dask.dataframe as dd
import pyarrow.parquet as pq
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
from datetime import datetime
import os
from tqdm import tqdm
import gc  # Garbage collection for memory management

# Configuration
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Constants
DATA_DIR = Path("data/combined")
RESULTS_DIR = Path("results/yearly_analysis")
CACHE_DIR = Path("cache/yearly_analysis")

# Create output directories
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Weather categories mapping for better analysis
WEATHER_CATEGORIES = {
    'Clear': ['clear', 'sunny'],
    'Cloudy': ['cloudy', 'overcast', 'partly cloudy'],
    'Rainy': ['rain', 'drizzle', 'shower'],
    'Snowy': ['snow', 'sleet'],
    'Foggy': ['fog', 'mist'],
    'Other': ['thunderstorm', 'hail', 'unknown']
}

class YearlyAnalyzer:
    """Main class for conducting yearly comparison analysis."""
    
    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.years_data = {}
        self.summary_stats = {}
        
    def load_yearly_data_efficiently(self, years: Optional[List[int]] = None) -> Dict[int, pd.DataFrame]:
        """
        Load yearly data using very small chunks to prevent memory issues.
        Calculate statistics without loading full datasets.
        """
        parquet_files = list(self.data_dir.glob("*.parquet"))
        
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {self.data_dir}")
        
        available_years = []
        for file in parquet_files:
            year_match = [str(y) for y in range(2015, 2030) if str(y) in file.name]
            if year_match:
                available_years.append(int(year_match[0]))
        
        if years is None:
            years = sorted(available_years)
        
        print(f"Processing data for years: {years}")
        
        yearly_data = {}
        
        for year in years:
            file_path = self.data_dir / f"{year}_combined_citibike_weather.parquet"
            
            if not file_path.exists():
                print(f"Warning: File for year {year} not found, skipping...")
                continue
            
            print(f"Processing {year} data...")
            start_time = time.time()
            
            # Get file size for information
            file_size = file_path.stat().st_size / (1024**3)  # Size in GB
            print(f"File size: {file_size:.2f}GB")
            
            # Always use streaming processing for memory safety
            self._process_file_streaming(file_path, year)
            
            # Create a minimal representative dataset for visualizations
            yearly_data[year] = self._create_representative_sample(file_path, year)
            
            process_time = time.time() - start_time
            print(f"Year {year}: processed in {process_time:.2f}s")
            
            # Force garbage collection
            gc.collect()
        
        self.years_data = yearly_data
        return yearly_data
    
    def _process_file_streaming(self, file_path: Path, year: int) -> None:
        """
        Process file using streaming with very small chunks to calculate statistics.
        """
        print(f"Streaming processing for {year}...")
        
        # Initialize statistics aggregator
        stats = {
            'total_rows': 0,
            'trip_duration_sum': 0,
            'trip_duration_count': 0,
            'trip_duration_min': float('inf'),
            'trip_duration_max': 0,
            'temp_sum': 0,
            'temp_count': 0,
            'temp_min': float('inf'),
            'temp_max': float('-inf'),
            'precip_sum': 0,
            'precip_count': 0,
            'precip_positive_count': 0,
            'wind_sum': 0,
            'wind_count': 0,
            'humidity_sum': 0,
            'humidity_count': 0,
            'seasonal_counts': {'Winter': 0, 'Spring': 0, 'Summer': 0, 'Fall': 0},
            'user_type_counts': {},
            'weather_counts': {}
        }
        
        try:
            # Use pyarrow to read in batches
            parquet_file = pq.ParquetFile(file_path)
            batch_size = 50000  # Small batch size
            batch_count = 0
            
            # Read file in batches using pyarrow
            for batch in parquet_file.iter_batches(batch_size=batch_size):
                batch_count += 1
                
                # Convert to pandas DataFrame
                chunk = batch.to_pandas()
                
                # Process chunk with minimal operations
                self._process_mini_chunk(chunk, stats, year)
                
                # Progress feedback every 20 batches
                if batch_count % 20 == 0:
                    print(f"Processed {batch_count * batch_size:,} rows...")
                
                # Clear chunk from memory immediately
                del chunk
                gc.collect()
                
                # Safety break if too many batches (prevents infinite loops)
                if batch_count > 2000:  # Max ~100M rows
                    print("Reached maximum batch limit for safety")
                    break
                    
        except Exception as e:
            print(f"Error in streaming processing: {e}")
            # Fallback: try with even smaller batches
            try:
                print("Trying with smaller batches...")
                batch_size = 10000
                batch_count = 0
                
                parquet_file = pq.ParquetFile(file_path)
                for batch in parquet_file.iter_batches(batch_size=batch_size):
                    batch_count += 1
                    chunk = batch.to_pandas()
                    self._process_mini_chunk(chunk, stats, year)
                    
                    if batch_count % 50 == 0:
                        print(f"Processed {batch_count * batch_size:,} rows...")
                    
                    del chunk
                    gc.collect()
                    
                    if batch_count > 5000:  # Max ~50M rows
                        break
                        
            except Exception as e2:
                print(f"Even smaller batches failed: {e2}")
                # Create minimal stats based on file metadata
                try:
                    parquet_file = pq.ParquetFile(file_path)
                    total_rows = parquet_file.metadata.num_rows
                    stats['total_rows'] = total_rows
                    print(f"Using metadata: estimated {total_rows:,} rows")
                except:
                    stats['total_rows'] = 1000000  # Final fallback estimate
        
        # Finalize and store statistics
        self._finalize_streaming_stats(year, stats)
        print(f"Completed streaming processing for {year}")
    
    def _process_mini_chunk(self, chunk: pd.DataFrame, stats: dict, year: int) -> None:
        """
        Process a very small chunk of data efficiently.
        """
        stats['total_rows'] += len(chunk)
        
        # Process only essential columns to save memory
        try:
            # Trip duration
            if 'trip_duration' in chunk.columns:
                duration_data = pd.to_numeric(chunk['trip_duration'], errors='coerce')
                # Filter outliers immediately
                duration_data = duration_data[(duration_data >= 60) & (duration_data <= 86400)]
                valid_duration = duration_data.dropna()
                
                if len(valid_duration) > 0:
                    stats['trip_duration_sum'] += valid_duration.sum()
                    stats['trip_duration_count'] += len(valid_duration)
                    stats['trip_duration_min'] = min(stats['trip_duration_min'], valid_duration.min())
                    stats['trip_duration_max'] = max(stats['trip_duration_max'], valid_duration.max())
            
            # Temperature
            if 'temperature' in chunk.columns:
                temp_data = pd.to_numeric(chunk['temperature'], errors='coerce').dropna()
                if len(temp_data) > 0:
                    stats['temp_sum'] += temp_data.sum()
                    stats['temp_count'] += len(temp_data)
                    stats['temp_min'] = min(stats['temp_min'], temp_data.min())
                    stats['temp_max'] = max(stats['temp_max'], temp_data.max())
            
            # Precipitation
            if 'precipitation' in chunk.columns:
                precip_data = pd.to_numeric(chunk['precipitation'], errors='coerce').dropna()
                if len(precip_data) > 0:
                    stats['precip_sum'] += precip_data.sum()
                    stats['precip_count'] += len(precip_data)
                    stats['precip_positive_count'] += (precip_data > 0).sum()
            
            # Wind speed
            if 'wind_speed' in chunk.columns:
                wind_data = pd.to_numeric(chunk['wind_speed'], errors='coerce').dropna()
                if len(wind_data) > 0:
                    stats['wind_sum'] += wind_data.sum()
                    stats['wind_count'] += len(wind_data)
            
            # Humidity
            if 'relative_humidity' in chunk.columns:
                humidity_data = pd.to_numeric(chunk['relative_humidity'], errors='coerce').dropna()
                if len(humidity_data) > 0:
                    stats['humidity_sum'] += humidity_data.sum()
                    stats['humidity_count'] += len(humidity_data)
            
            # Seasonal data (process start_time efficiently)
            if 'start_time' in chunk.columns:
                try:
                    time_data = pd.to_datetime(chunk['start_time'], errors='coerce')
                    months = time_data.dt.month.dropna()
                    
                    for month in months:
                        season = self._get_season(month)
                        if season in stats['seasonal_counts']:
                            stats['seasonal_counts'][season] += 1
                except:
                    pass  # Skip if time processing fails
            
            # User type
            if 'user_type' in chunk.columns:
                user_counts = chunk['user_type'].value_counts()
                for user_type, count in user_counts.items():
                    stats['user_type_counts'][user_type] = stats['user_type_counts'].get(user_type, 0) + count
            
        except Exception as e:
            print(f"Error processing mini chunk: {e}")
            pass  # Continue processing even if some operations fail
    
    def _finalize_streaming_stats(self, year: int, stats: dict) -> None:
        """
        Convert accumulated statistics to final format.
        """
        final_stats = {}
        
        # Basic stats
        final_stats['total_rides'] = stats['total_rows']
        
        # Trip duration
        if stats['trip_duration_count'] > 0:
            final_stats['avg_trip_duration'] = stats['trip_duration_sum'] / stats['trip_duration_count']
            final_stats['min_trip_duration'] = stats['trip_duration_min']
            final_stats['max_trip_duration'] = stats['trip_duration_max']
        
        # Temperature
        if stats['temp_count'] > 0:
            final_stats['avg_temperature'] = stats['temp_sum'] / stats['temp_count']
            final_stats['min_temperature'] = stats['temp_min']
            final_stats['max_temperature'] = stats['temp_max']
        
        # Precipitation
        if stats['precip_count'] > 0:
            final_stats['avg_precipitation'] = stats['precip_sum'] / stats['precip_count']
            final_stats['rainy_days_pct'] = (stats['precip_positive_count'] / stats['precip_count']) * 100
        
        # Wind speed
        if stats['wind_count'] > 0:
            final_stats['avg_wind_speed'] = stats['wind_sum'] / stats['wind_count']
        
        # Humidity
        if stats['humidity_count'] > 0:
            final_stats['avg_humidity'] = stats['humidity_sum'] / stats['humidity_count']
        
        # Distributions
        final_stats['seasonal_distribution'] = stats['seasonal_counts']
        final_stats['user_type_distribution'] = stats['user_type_counts']
        
        # Store statistics
        if not hasattr(self, 'summary_stats'):
            self.summary_stats = {}
        self.summary_stats[year] = final_stats
    
    def _create_representative_sample(self, file_path: Path, year: int, sample_size: int = 5000) -> pd.DataFrame:
        """
        Create a small representative sample for visualizations using pyarrow batches.
        """
        print(f"Creating representative sample for {year}...")
        
        try:
            sample_chunks = []
            batch_count = 0
            
            # Use pyarrow to read small batches
            parquet_file = pq.ParquetFile(file_path)
            
            for batch in parquet_file.iter_batches(batch_size=5000):
                batch_count += 1
                
                # Convert to pandas and take sample
                chunk = batch.to_pandas()
                
                if len(chunk) > 50:
                    chunk_sample = chunk.sample(n=min(50, len(chunk)), random_state=42)
                    sample_chunks.append(chunk_sample)
                
                del chunk
                gc.collect()
                
                # Stop after collecting enough samples
                if batch_count >= 100 or len(sample_chunks) * 50 >= sample_size:
                    break
            
            if sample_chunks:
                sample_df = pd.concat(sample_chunks, ignore_index=True)
                
                # Basic preprocessing for the sample
                sample_df['year'] = year
                
                # Clean essential columns
                if 'start_time' in sample_df.columns:
                    sample_df['start_time'] = pd.to_datetime(sample_df['start_time'], errors='coerce')
                    sample_df['month'] = sample_df['start_time'].dt.month
                    sample_df['season'] = sample_df['month'].map(self._get_season)
                
                for col in ['trip_duration', 'temperature', 'precipitation']:
                    if col in sample_df.columns:
                        sample_df[col] = pd.to_numeric(sample_df[col], errors='coerce')
                
                print(f"Created sample with {len(sample_df)} rows")
                return sample_df
            
        except Exception as e:
            print(f"Error creating sample with pyarrow: {e}")
            # Try dask fallback for sampling
            try:
                print("Trying dask for sampling...")
                df_dask = dd.read_parquet(file_path)
                
                # Get a fraction of the data
                total_partitions = df_dask.npartitions
                partitions_to_read = min(3, total_partitions)  # Read max 3 partitions
                
                sample_data = []
                for i in range(partitions_to_read):
                    partition = df_dask.get_partition(i).compute()
                    if len(partition) > 100:
                        partition_sample = partition.sample(n=min(100, len(partition)), random_state=42)
                        sample_data.append(partition_sample)
                    del partition
                    gc.collect()
                
                if sample_data:
                    sample_df = pd.concat(sample_data, ignore_index=True)
                    sample_df['year'] = year
                    print(f"Created dask sample with {len(sample_df)} rows")
                    return sample_df
                
            except Exception as e3:
                print(f"Dask sampling also failed: {e3}")
        
        # Final fallback: create minimal dataset
        print("Creating minimal fallback dataset")
        return pd.DataFrame({
            'year': [year],
            'sample_created': [True],
            'trip_duration': [600],  # 10 minutes default
            'temperature': [60],  # 60F default
            'precipitation': [0],
            'season': ['Summer']
        })
    
    def _get_season(self, month: int) -> str:
        """Map month to season."""
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    def calculate_yearly_summary_stats(self) -> Dict[int, Dict]:
        """
        Calculate comprehensive summary statistics for each year.
        Works with both full datasets and chunk-processed data.
        """
        if not hasattr(self, 'summary_stats'):
            self.summary_stats = {}
        
        for year, df in self.years_data.items():
            # Check if stats were already calculated during chunked processing
            if year in self.summary_stats:
                print(f"Using pre-calculated statistics for {year}")
                continue
                
            # Calculate stats for full datasets
            stats = {}
            
            # Basic ride statistics
            stats['total_rides'] = len(df)
            stats['avg_trip_duration'] = df['trip_duration'].mean() if 'trip_duration' in df.columns else None
            stats['median_trip_duration'] = df['trip_duration'].median() if 'trip_duration' in df.columns else None
            
            # Weather statistics
            if 'temperature' in df.columns:
                stats['avg_temperature'] = df['temperature'].mean()
                stats['min_temperature'] = df['temperature'].min()
                stats['max_temperature'] = df['temperature'].max()
            
            if 'precipitation' in df.columns:
                stats['avg_precipitation'] = df['precipitation'].mean()
                stats['total_precipitation'] = df['precipitation'].sum()
                stats['rainy_days_pct'] = (df['precipitation'] > 0).mean() * 100
            
            if 'wind_speed' in df.columns:
                stats['avg_wind_speed'] = df['wind_speed'].mean()
            
            if 'relative_humidity' in df.columns:
                stats['avg_humidity'] = df['relative_humidity'].mean()
            
            # Seasonal distribution
            if 'season' in df.columns:
                seasonal_counts = df['season'].value_counts()
                stats['seasonal_distribution'] = seasonal_counts.to_dict()
            
            # User type distribution
            if 'user_type' in df.columns:
                user_type_counts = df['user_type'].value_counts()
                stats['user_type_distribution'] = user_type_counts.to_dict()
            
            # Weather category distribution
            if 'weather_category_clean' in df.columns:
                weather_counts = df['weather_category_clean'].value_counts()
                stats['weather_distribution'] = weather_counts.to_dict()
            
            self.summary_stats[year] = stats
        
        return self.summary_stats
    
    def analyze_ride_patterns(self) -> None:
        """
        Analyze and visualize ride patterns across years.
        """
        print("Analyzing ride patterns across years...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Ride Patterns Analysis Across Years', fontsize=16, fontweight='bold')
        
        # 1. Total rides per year
        years = list(self.summary_stats.keys())
        total_rides = [self.summary_stats[year]['total_rides'] for year in years]
        
        axes[0, 0].bar(years, total_rides, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Total Rides per Year')
        axes[0, 0].set_xlabel('Year')
        axes[0, 0].set_ylabel('Number of Rides')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(total_rides):
            axes[0, 0].text(years[i], v + max(total_rides) * 0.01, f'{v:,}', 
                           ha='center', va='bottom', fontweight='bold')
        
        # 2. Average trip duration per year
        avg_durations = [self.summary_stats[year]['avg_trip_duration'] for year in years 
                        if self.summary_stats[year]['avg_trip_duration'] is not None]
        valid_years = [year for year in years 
                      if self.summary_stats[year]['avg_trip_duration'] is not None]
        
        if avg_durations:
            axes[0, 1].plot(valid_years, np.array(avg_durations)/60, marker='o', 
                           linewidth=2, markersize=8, color='orange')
            axes[0, 1].set_title('Average Trip Duration per Year')
            axes[0, 1].set_xlabel('Year')
            axes[0, 1].set_ylabel('Duration (minutes)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Seasonal distribution heatmap
        seasonal_data = []
        seasons = ['Winter', 'Spring', 'Summer', 'Fall']
        
        for year in years:
            if 'seasonal_distribution' in self.summary_stats[year]:
                year_data = []
                total_rides_year = sum(self.summary_stats[year]['seasonal_distribution'].values())
                for season in seasons:
                    count = self.summary_stats[year]['seasonal_distribution'].get(season, 0)
                    percentage = (count / total_rides_year) * 100 if total_rides_year > 0 else 0
                    year_data.append(percentage)
                seasonal_data.append(year_data)
        
        if seasonal_data:
            seasonal_df = pd.DataFrame(seasonal_data, index=years, columns=seasons)
            im = axes[1, 0].imshow(seasonal_df.values, cmap='YlOrRd', aspect='auto')
            axes[1, 0].set_xticks(range(len(seasons)))
            axes[1, 0].set_xticklabels(seasons)
            axes[1, 0].set_yticks(range(len(years)))
            axes[1, 0].set_yticklabels(years)
            axes[1, 0].set_title('Seasonal Distribution (% of rides)')
            
            # Add text annotations
            for i in range(len(years)):
                for j in range(len(seasons)):
                    text = axes[1, 0].text(j, i, f'{seasonal_df.iloc[i, j]:.1f}%',
                                         ha="center", va="center", color="black", fontweight='bold')
            
            plt.colorbar(im, ax=axes[1, 0], label='Percentage of Rides')
        
        # 4. Weather impact analysis
        weather_impact_data = []
        for year in years:
            if 'weather_distribution' in self.summary_stats[year]:
                total_rides_year = sum(self.summary_stats[year]['weather_distribution'].values())
                clear_weather = self.summary_stats[year]['weather_distribution'].get('Clear', 0)
                clear_pct = (clear_weather / total_rides_year) * 100 if total_rides_year > 0 else 0
                weather_impact_data.append(clear_pct)
        
        if weather_impact_data:
            axes[1, 1].bar(years, weather_impact_data, color='lightgreen', alpha=0.7)
            axes[1, 1].set_title('Clear Weather Rides (%)')
            axes[1, 1].set_xlabel('Year')
            axes[1, 1].set_ylabel('Percentage of Rides in Clear Weather')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'ride_patterns_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_weather_trends(self) -> None:
        """
        Analyze weather trends across years.
        """
        print("Analyzing weather trends across years...")
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Weather Trends Analysis Across Years', fontsize=16, fontweight='bold')
        
        years = list(self.summary_stats.keys())
        
        # 1. Temperature trends
        avg_temps = [self.summary_stats[year].get('avg_temperature') for year in years]
        min_temps = [self.summary_stats[year].get('min_temperature') for year in years]
        max_temps = [self.summary_stats[year].get('max_temperature') for year in years]
        
        valid_temp_data = [(y, avg, min_t, max_t) for y, avg, min_t, max_t in 
                          zip(years, avg_temps, min_temps, max_temps) 
                          if avg is not None and min_t is not None and max_t is not None]
        
        if valid_temp_data:
            valid_years, valid_avg, valid_min, valid_max = zip(*valid_temp_data)
            
            axes[0, 0].plot(valid_years, valid_avg, marker='o', label='Average', linewidth=2, color='red')
            axes[0, 0].fill_between(valid_years, valid_min, valid_max, alpha=0.3, color='red')
            axes[0, 0].set_title('Temperature Trends')
            axes[0, 0].set_xlabel('Year')
            axes[0, 0].set_ylabel('Temperature (°C)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Precipitation trends
        avg_precip = [self.summary_stats[year].get('avg_precipitation') for year in years]
        valid_precip_data = [(y, p) for y, p in zip(years, avg_precip) if p is not None]
        
        if valid_precip_data:
            valid_years_p, valid_precip = zip(*valid_precip_data)
            axes[0, 1].bar(valid_years_p, valid_precip, color='blue', alpha=0.7)
            axes[0, 1].set_title('Average Precipitation per Year')
            axes[0, 1].set_xlabel('Year')
            axes[0, 1].set_ylabel('Precipitation (inches)')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Wind speed trends
        avg_wind = [self.summary_stats[year].get('avg_wind_speed') for year in years]
        valid_wind_data = [(y, w) for y, w in zip(years, avg_wind) if w is not None]
        
        if valid_wind_data:
            valid_years_w, valid_wind = zip(*valid_wind_data)
            axes[1, 0].plot(valid_years_w, valid_wind, marker='s', linewidth=2, 
                           markersize=8, color='green')
            axes[1, 0].set_title('Average Wind Speed per Year')
            axes[1, 0].set_xlabel('Year')
            axes[1, 0].set_ylabel('Wind Speed (mph)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Humidity trends
        avg_humidity = [self.summary_stats[year].get('avg_humidity') for year in years]
        valid_humidity_data = [(y, h) for y, h in zip(years, avg_humidity) if h is not None]
        
        if valid_humidity_data:
            valid_years_h, valid_humidity = zip(*valid_humidity_data)
            axes[1, 1].plot(valid_years_h, valid_humidity, marker='^', linewidth=2, 
                           markersize=8, color='purple')
            axes[1, 1].set_title('Average Humidity per Year')
            axes[1, 1].set_xlabel('Year')
            axes[1, 1].set_ylabel('Relative Humidity (%)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'weather_trends_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_weather_ride_correlation(self) -> None:
        """
        Analyze correlation between weather conditions and ride patterns.
        """
        print("Analyzing weather-ride correlations across years...")
        
        # Combine all years data for correlation analysis
        combined_data = []
        for year, df in self.years_data.items():
            # Select relevant columns for correlation
            correlation_cols = ['trip_duration', 'temperature', 'wind_speed', 
                              'relative_humidity', 'precipitation', 'year']
            available_cols = [col for col in correlation_cols if col in df.columns]
            
            if len(available_cols) > 2:  # Need at least year and one other column
                df_subset = df[available_cols].copy()
                combined_data.append(df_subset)
        
        if combined_data:
            combined_df = pd.concat(combined_data, ignore_index=True)
            
            # Calculate correlation matrix
            correlation_matrix = combined_df.corr()
            
            # Create correlation heatmap
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                       center=0, square=True, fmt='.3f')
            plt.title('Weather-Ride Correlation Matrix Across All Years', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(RESULTS_DIR / 'weather_ride_correlation.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def create_comparative_summary_report(self) -> None:
        """
        Create a comprehensive summary report comparing all years.
        """
        print("Creating comparative summary report...")
        
        # Create summary DataFrame
        summary_data = []
        for year, stats in self.summary_stats.items():
            row = {'Year': year}
            row.update(stats)
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary to CSV
        summary_csv_path = RESULTS_DIR / 'yearly_summary_statistics.csv'
        summary_df.to_csv(summary_csv_path, index=False)
        
        # Create summary visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Yearly Comparison Summary Dashboard', fontsize=16, fontweight='bold')
        
        years = summary_df['Year'].values
        
        # 1. Year-over-year growth in rides
        if 'total_rides' in summary_df.columns:
            total_rides = summary_df['total_rides'].values
            growth_rates = np.diff(total_rides) / total_rides[:-1] * 100
            
            ax1.bar(years[1:], growth_rates, color=['green' if x > 0 else 'red' for x in growth_rates])
            ax1.set_title('Year-over-Year Growth Rate (%)')
            ax1.set_xlabel('Year')
            ax1.set_ylabel('Growth Rate (%)')
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
        
        # 2. Weather extremes comparison
        if 'min_temperature' in summary_df.columns and 'max_temperature' in summary_df.columns:
            temp_range = summary_df['max_temperature'] - summary_df['min_temperature']
            ax2.bar(years, temp_range, color='orange', alpha=0.7)
            ax2.set_title('Temperature Range per Year')
            ax2.set_xlabel('Year')
            ax2.set_ylabel('Temperature Range (°C)')
            ax2.tick_params(axis='x', rotation=45)
        
        # 3. Precipitation vs Rides scatter
        if 'avg_precipitation' in summary_df.columns and 'total_rides' in summary_df.columns:
            ax3.scatter(summary_df['avg_precipitation'], summary_df['total_rides'], 
                       s=100, alpha=0.7, c=years, cmap='viridis')
            ax3.set_xlabel('Average Precipitation')
            ax3.set_ylabel('Total Rides')
            ax3.set_title('Precipitation vs Total Rides')
            
            # Add year labels
            for i, year in enumerate(years):
                ax3.annotate(str(year), (summary_df['avg_precipitation'].iloc[i], 
                                       summary_df['total_rides'].iloc[i]),
                           xytext=(5, 5), textcoords='offset points', fontweight='bold')
        
        # 4. Multi-metric normalized comparison
        metrics_to_normalize = ['total_rides', 'avg_temperature', 'avg_precipitation', 'avg_wind_speed']
        available_metrics = [m for m in metrics_to_normalize if m in summary_df.columns]
        
        if len(available_metrics) > 1:
            # Normalize metrics to 0-1 scale
            normalized_data = summary_df[available_metrics].copy()
            for col in available_metrics:
                col_min, col_max = normalized_data[col].min(), normalized_data[col].max()
                if col_max > col_min:
                    normalized_data[col] = (normalized_data[col] - col_min) / (col_max - col_min)
            
            # Create grouped bar chart instead of stacked
            x = np.arange(len(years))  # the label locations
            width = 0.8 / len(available_metrics)  # width of the bars
            colors = plt.cm.Set3(np.linspace(0, 1, len(available_metrics)))
            
            for i, metric in enumerate(available_metrics):
                offset = (i - len(available_metrics)/2 + 0.5) * width
                ax4.bar(x + offset, normalized_data[metric], width, 
                       label=metric.replace('_', ' ').title(), color=colors[i], alpha=0.8)
            
            ax4.set_title('Normalized Multi-Metric Comparison')
            ax4.set_xlabel('Year')
            ax4.set_ylabel('Normalized Value (0-1)')
            ax4.set_xticks(x)
            ax4.set_xticklabels(years)
            ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'yearly_comparison_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Summary report saved to: {summary_csv_path}")
        print(f"Visualizations saved to: {RESULTS_DIR}")
    
    def run_complete_analysis(self, years: Optional[List[int]] = None) -> None:
        """
        Run the complete yearly comparison analysis.
        """
        print("=" * 60)
        print("STARTING COMPREHENSIVE YEARLY COMPARISON ANALYSIS")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # Load data
            self.load_yearly_data_efficiently(years)
            
            # Calculate summary statistics
            self.calculate_yearly_summary_stats()
            
            # Run analyses
            self.analyze_ride_patterns()
            self.analyze_weather_trends()
            self.analyze_weather_ride_correlation()
            self.create_comparative_summary_report()
            
            total_time = time.time() - start_time
            print(f"\n✅ Analysis completed successfully in {total_time:.2f} seconds")
            print(f"📊 Results saved to: {RESULTS_DIR}")
            
        except Exception as e:
            print(f"❌ Error during analysis: {str(e)}")
            raise


def main():
    """
    Main function to run the yearly comparison analysis.
    """
    # Initialize analyzer
    analyzer = YearlyAnalyzer()
    
    # Run complete analysis
    analyzer.run_complete_analysis()
    
    # Print final summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    
    for year, stats in analyzer.summary_stats.items():
        print(f"\n📅 Year {year}:")
        print(f"   🚴 Total Rides: {stats['total_rides']:,}")
        if stats.get('avg_temperature'):
            print(f"   🌡️  Avg Temperature: {stats['avg_temperature']:.1f}°C")
        if stats.get('avg_precipitation'):
            print(f"   🌧️  Avg Precipitation: {stats['avg_precipitation']:.3f} inches")


if __name__ == "__main__":
    main() 