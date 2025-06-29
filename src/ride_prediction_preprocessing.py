import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import gc
import os
warnings.filterwarnings('ignore')

class RideDataPreprocessor:
    """
    Memory-efficient preprocessor for citibike ride data to create features for prediction
    """
    
    def __init__(self, data_path="data/combined", chunk_size=100000):
        self.data_path = data_path
        self.chunk_size = chunk_size
        self.combined_data = None
        self.hourly_aggregated = None
        self.daily_aggregated = None
        
    def get_file_info(self, years=None):
        """
        Get information about available files without loading them
        """
        if years is None:
            years = [2019, 2020, 2021, 2022, 2023, 2024]
        
        file_info = []
        for year in years:
            file_path = f"{self.data_path}/{year}_combined_citibike_weather.parquet"
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path) / (1024**3)  # Size in GB
                file_info.append({
                    'year': year,
                    'path': file_path,
                    'size_gb': file_size
                })
        
        return file_info
    
    def load_data_chunked(self, years=None, sample_fraction=None, max_rows_per_file=None):
        """
        Load parquet files in chunks to manage memory usage
        Now processes each year individually and combines only aggregated results
        
        Parameters:
        - years: List of years to load
        - sample_fraction: Fraction of data to sample (0.1 = 10%)
        - max_rows_per_file: Maximum rows to load per file
        """
        if years is None:
            years = [2019, 2020, 2021, 2022, 2023, 2024]
        
        print("=== MEMORY-EFFICIENT CHUNKED DATA LOADING ===")
        file_info = self.get_file_info(years)
        
        if not file_info:
            raise ValueError("No data files found!")
        
        # Display file information
        total_size = sum(f['size_gb'] for f in file_info)
        print(f"Found {len(file_info)} files, total size: {total_size:.1f} GB")
        print("Using year-by-year processing to avoid memory issues...")
        
        if sample_fraction:
            print(f"Will sample {sample_fraction*100:.1f}% of data")
        if max_rows_per_file:
            print(f"Will load max {max_rows_per_file:,} rows per file")
        
        # Initialize aggregated data collectors
        hourly_aggregated_list = []
        daily_aggregated_list = []
        total_rows_processed = 0
        
        for file_info_item in file_info:
            year = file_info_item['year']
            file_path = file_info_item['path']
            
            print(f"\nProcessing {year} ({file_info_item['size_gb']:.1f} GB)...")
            
            try:
                # Read file metadata first
                parquet_file = pd.read_parquet(file_path, columns=['start_time'])
                total_rows = len(parquet_file)
                del parquet_file
                gc.collect()
                
                # Determine how many rows to process
                if max_rows_per_file:
                    rows_to_process = min(total_rows, max_rows_per_file)
                else:
                    rows_to_process = total_rows
                
                if sample_fraction:
                    rows_to_process = int(rows_to_process * sample_fraction)
                
                print(f"  Total rows: {total_rows:,}, processing: {rows_to_process:,}")
                
                # Load year data
                year_data = None
                if sample_fraction or max_rows_per_file:
                    if sample_fraction:
                        # Random sampling
                        indices = np.random.choice(total_rows, size=rows_to_process, replace=False)
                        indices = np.sort(indices)
                    else:
                        # Take first N rows
                        indices = np.arange(rows_to_process)
                    
                    # Load and sample
                    df_full = pd.read_parquet(file_path)
                    year_data = df_full.iloc[indices].copy()
                    del df_full
                    gc.collect()
                else:
                    # Load entire file
                    year_data = pd.read_parquet(file_path)
                
                print(f"  Loaded {len(year_data):,} rows for {year}")
                total_rows_processed += len(year_data)
                
                # Process this year's data immediately to create aggregations
                print(f"  Creating time features for {year}...")
                year_data = self.create_time_features(year_data)
                
                # Create hourly aggregation for this year
                print(f"  Creating hourly aggregation for {year}...")
                hourly_agg = self._create_hourly_aggregation_from_data(year_data)
                hourly_aggregated_list.append(hourly_agg)
                
                # Create daily aggregation for this year
                print(f"  Creating daily aggregation for {year}...")
                daily_agg = self._create_daily_aggregation_from_data(year_data)
                daily_aggregated_list.append(daily_agg)
                
                print(f"  Year {year} processed successfully")
                print(f"    Hourly records: {len(hourly_agg):,}")
                print(f"    Daily records: {len(daily_agg):,}")
                
                # Clean up year data immediately
                del year_data
                gc.collect()
                
            except Exception as e:
                print(f"Error processing {year}: {e}")
                continue
        
        if hourly_aggregated_list and daily_aggregated_list:
            print(f"\nCombining aggregated data from all years...")
            print(f"Total raw rows processed: {total_rows_processed:,}")
            
            # Combine aggregated data (much smaller than raw data)
            self.hourly_aggregated = pd.concat(hourly_aggregated_list, ignore_index=True)
            self.daily_aggregated = pd.concat(daily_aggregated_list, ignore_index=True)
            
            del hourly_aggregated_list, daily_aggregated_list
            gc.collect()
            
            print(f"Final aggregated data:")
            print(f"  Hourly records: {len(self.hourly_aggregated):,}")
            print(f"  Daily records: {len(self.daily_aggregated):,}")
            print(f"  Memory usage: ~{(self.hourly_aggregated.memory_usage(deep=True).sum() + self.daily_aggregated.memory_usage(deep=True).sum()) / 1024**2:.1f} MB")
            
            # Create a small sample of combined data for exploration
            sample_size = min(100000, total_rows_processed // 10)
            print(f"Creating sample dataset ({sample_size:,} rows) for exploration...")
            
            # We'll reload just a small sample for the combined_data attribute
            first_file = file_info[0]
            sample_data = pd.read_parquet(first_file['path'])
            if len(sample_data) > sample_size:
                sample_indices = np.random.choice(len(sample_data), size=sample_size, replace=False)
                sample_data = sample_data.iloc[sample_indices].copy()
            
            self.combined_data = self.create_time_features(sample_data)
            
            return self.combined_data
        else:
            raise ValueError("No data successfully processed!")
    
    def load_data_single_year(self, year=2024, sample_fraction=0.1):
        """
        Load data for a single year for quick testing
        """
        print(f"Loading single year ({year}) with {sample_fraction*100:.1f}% sample...")
        return self.load_data_chunked(years=[year], sample_fraction=sample_fraction)
    
    def load_data_lightweight(self, years=None, max_total_rows=500000):
        """
        Load a lightweight version of the data for quick testing
        """
        if years is None:
            years = [2024, 2023]  # Recent years only
        
        file_info = self.get_file_info(years)
        total_files = len(file_info)
        max_rows_per_file = max_total_rows // total_files if total_files > 0 else max_total_rows
        
        print(f"Loading lightweight data: max {max_total_rows:,} total rows...")
        return self.load_data_chunked(years=years, max_rows_per_file=max_rows_per_file)
    
    def explore_data(self):
        """
        Basic data exploration with memory efficiency
        """
        if self.combined_data is None:
            raise ValueError("Data not loaded. Call load_data_* first.")
        
        print("=== DATA EXPLORATION ===")
        print(f"Shape: {self.combined_data.shape}")
        print(f"Memory usage: ~{self.combined_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Convert datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(self.combined_data['start_time']):
            print("Converting start_time to datetime...")
            self.combined_data['start_time'] = pd.to_datetime(self.combined_data['start_time'])
        
        print(f"Date range: {self.combined_data['start_time'].min()} to {self.combined_data['start_time'].max()}")
        print(f"\nColumns: {list(self.combined_data.columns)}")
        
        # Sample data types and missing values (to avoid memory issues)
        sample_size = min(10000, len(self.combined_data))
        sample_data = self.combined_data.sample(n=sample_size)
        
        print(f"\nData types (from {sample_size:,} sample):")
        print(sample_data.dtypes)
        print(f"\nMissing values (from sample):")
        print(sample_data.isnull().sum())
        
        # Basic statistics
        print(f"\nTrip duration statistics:")
        print(self.combined_data['trip_duration'].describe())
        
        # Rides per day
        daily_rides = self.combined_data.groupby(self.combined_data['start_time'].dt.date).size()
        print(f"\nDaily rides statistics:")
        print(daily_rides.describe())
        
        return self.combined_data.head()
    
    def create_time_features(self, df):
        """
        Create time-based features from datetime columns
        """
        df = df.copy()
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df['start_time']):
            df['start_time'] = pd.to_datetime(df['start_time'])
        if not pd.api.types.is_datetime64_any_dtype(df['stop_time']):
            df['stop_time'] = pd.to_datetime(df['stop_time'])
        
        # Extract time features
        df['hour'] = df['start_time'].dt.hour
        df['day_of_week'] = df['start_time'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['month'] = df['start_time'].dt.month
        df['year'] = df['start_time'].dt.year
        df['day_of_year'] = df['start_time'].dt.dayofyear
        df['quarter'] = df['start_time'].dt.quarter
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Create time periods
        df['time_period'] = pd.cut(df['hour'], 
                                 bins=[0, 6, 12, 18, 24], 
                                 labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                                 include_lowest=True)
        
        # Season (rough approximation)
        df['season'] = df['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        return df
    
    def _create_hourly_aggregation_from_data(self, data):
        """
        Create hourly aggregated data from a given dataset (used for year-by-year processing)
        """
        # Create time features
        df = self.create_time_features(data)
        
        # Create hourly timestamp
        df['hour_timestamp'] = df['start_time'].dt.floor('H')
        
        # Helper function for safe mode calculation
        def safe_mode(x, default=0):
            """Safely calculate mode, returning default if no mode exists"""
            try:
                clean_x = x.dropna()
                if len(clean_x) == 0:
                    return default
                mode_vals = clean_x.mode()
                if len(mode_vals) > 0:
                    return mode_vals.iloc[0]
                else:
                    return default
            except:
                return default
        
        # Aggregate by hour with memory efficiency
        aggregations = {
            # Target variables
            'trip_duration': ['count', 'mean', 'median', 'std', 'min', 'max'],
            
            # Weather features (take mean since they should be similar within an hour)
            'temperature': 'mean',
            'wind_speed': 'mean',
            'relative_humidity': 'mean',
            'cloud_cover': 'mean',
            'precipitation': 'mean',
            'utci': 'mean',
            
            # Categorical weather (use most common value or default)
            'rain': lambda x: safe_mode(x, 0),
            'snow': lambda x: safe_mode(x, 0),
            'mist_fog': lambda x: safe_mode(x, 0),
            'weather_cat': lambda x: safe_mode(x, 'Clear'),
            'utci_cat': lambda x: safe_mode(x, 'No thermal stress'),
            
            # User type distribution
            'user_type': lambda x: (x == 'Subscriber').sum() / len(x) if len(x) > 0 else 0,
            
            # Time features (take first since they're the same within an hour)
            'hour': 'first',
            'day_of_week': 'first',
            'month': 'first',
            'year': 'first',
            'day_of_year': 'first',
            'quarter': 'first',
            'is_weekend': 'first',
            'season': 'first',
            'time_period': 'first'
        }
        
        hourly_agg = df.groupby('hour_timestamp').agg(aggregations).reset_index()
        
        # Flatten column names
        hourly_agg.columns = [f"{col[0]}_{col[1]}" if col[1] != '' else col[0] 
                             for col in hourly_agg.columns]
        
        # Rename target columns for clarity
        hourly_agg = hourly_agg.rename(columns={
            'trip_duration_count': 'ride_count',
            'trip_duration_mean': 'avg_duration',
            'trip_duration_median': 'median_duration',
            'trip_duration_std': 'std_duration',
            'trip_duration_min': 'min_duration',
            'trip_duration_max': 'max_duration'
        })
        
        # Sort by timestamp
        hourly_agg = hourly_agg.sort_values('hour_timestamp')
        
        return hourly_agg
    
    def _create_daily_aggregation_from_data(self, data):
        """
        Create daily aggregated data from a given dataset (used for year-by-year processing)
        """
        # Create time features
        df = self.create_time_features(data)
        
        # Create daily timestamp
        df['date'] = df['start_time'].dt.date
        
        # Aggregate by day
        daily_agg = df.groupby('date').agg({
            # Target variables
            'trip_duration': ['count', 'mean', 'median', 'std', 'min', 'max'],
            
            # Weather features (daily averages)
            'temperature': ['mean', 'min', 'max'],
            'wind_speed': 'mean',
            'relative_humidity': 'mean',
            'cloud_cover': 'mean',
            'precipitation': 'sum',
            'utci': 'mean',
            
            # Weather events (any occurrence during the day)
            'rain': 'max',
            'snow': 'max',
            'mist_fog': 'max',
            
            # User type distribution
            'user_type': lambda x: (x == 'Subscriber').sum() / len(x) if len(x) > 0 else 0,
            
            # Time features
            'day_of_week': 'first',
            'month': 'first',
            'year': 'first',
            'day_of_year': 'first',
            'quarter': 'first',
            'is_weekend': 'first',
            'season': 'first'
        }).reset_index()
        
        # Flatten column names
        daily_agg.columns = [f"{col[0]}_{col[1]}" if col[1] != '' else col[0] 
                            for col in daily_agg.columns]
        
        # Rename target columns for clarity
        daily_agg = daily_agg.rename(columns={
            'trip_duration_count': 'daily_ride_count',
            'trip_duration_mean': 'daily_avg_duration',
            'trip_duration_median': 'daily_median_duration',
            'trip_duration_std': 'daily_std_duration',
            'trip_duration_min': 'daily_min_duration',
            'trip_duration_max': 'daily_max_duration'
        })
        
        # Sort by date
        daily_agg = daily_agg.sort_values('date')
        
        return daily_agg
    
    def create_hourly_aggregated_data(self):
        """
        Aggregate data by hour for prediction - memory efficient version
        """
        if self.combined_data is None:
            raise ValueError("Data not loaded. Call load_data_* first.")
        
        print("Creating hourly aggregated data...")
        
        # Create time features in chunks to manage memory
        print("  Adding time features...")
        df = self.create_time_features(self.combined_data)
        
        # Create hourly timestamp
        df['hour_timestamp'] = df['start_time'].dt.floor('H')
        
        print("  Aggregating by hour...")
        
        # Helper function for safe mode calculation
        def safe_mode(x, default=0):
            """Safely calculate mode, returning default if no mode exists"""
            try:
                clean_x = x.dropna()
                if len(clean_x) == 0:
                    return default
                mode_vals = clean_x.mode()
                if len(mode_vals) > 0:
                    return mode_vals.iloc[0]
                else:
                    return default
            except:
                return default
        
        # Aggregate by hour with memory efficiency
        aggregations = {
            # Target variables
            'trip_duration': ['count', 'mean', 'median', 'std', 'min', 'max'],
            
            # Weather features (take mean since they should be similar within an hour)
            'temperature': 'mean',
            'wind_speed': 'mean',
            'relative_humidity': 'mean',
            'cloud_cover': 'mean',
            'precipitation': 'mean',
            'utci': 'mean',
            
            # Categorical weather (use most common value or default)
            'rain': lambda x: safe_mode(x, 0),
            'snow': lambda x: safe_mode(x, 0),
            'mist_fog': lambda x: safe_mode(x, 0),
            'weather_cat': lambda x: safe_mode(x, 'Clear'),
            'utci_cat': lambda x: safe_mode(x, 'No thermal stress'),
            
            # User type distribution
            'user_type': lambda x: (x == 'Subscriber').sum() / len(x) if len(x) > 0 else 0,
            
            # Time features (take first since they're the same within an hour)
            'hour': 'first',
            'day_of_week': 'first',
            'month': 'first',
            'year': 'first',
            'day_of_year': 'first',
            'quarter': 'first',
            'is_weekend': 'first',
            'season': 'first',
            'time_period': 'first'
        }
        
        hourly_agg = df.groupby('hour_timestamp').agg(aggregations).reset_index()
        
        # Clean up original data to free memory
        del df
        gc.collect()
        
        # Flatten column names
        hourly_agg.columns = [f"{col[0]}_{col[1]}" if col[1] != '' else col[0] 
                             for col in hourly_agg.columns]
        
        # Rename target columns for clarity
        hourly_agg = hourly_agg.rename(columns={
            'trip_duration_count': 'ride_count',
            'trip_duration_mean': 'avg_duration',
            'trip_duration_median': 'median_duration',
            'trip_duration_std': 'std_duration',
            'trip_duration_min': 'min_duration',
            'trip_duration_max': 'max_duration'
        })
        
        print("  Adding lag and rolling features...")
        
        # Add lag features for previous hours
        hourly_agg = hourly_agg.sort_values('hour_timestamp')
        for lag in [1, 2, 3, 6, 12, 24]:
            hourly_agg[f'ride_count_lag_{lag}h'] = hourly_agg['ride_count'].shift(lag)
            hourly_agg[f'avg_duration_lag_{lag}h'] = hourly_agg['avg_duration'].shift(lag)
        
        # Add rolling averages
        for window in [3, 6, 12, 24]:
            hourly_agg[f'ride_count_rolling_{window}h'] = hourly_agg['ride_count'].rolling(window=window).mean()
            hourly_agg[f'avg_duration_rolling_{window}h'] = hourly_agg['avg_duration'].rolling(window=window).mean()
        
        self.hourly_aggregated = hourly_agg
        print(f"  Created hourly data: {hourly_agg.shape}")
        print(f"  Memory usage: ~{hourly_agg.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        return hourly_agg
    
    def create_daily_aggregated_data(self):
        """
        Aggregate data by day for prediction - memory efficient version
        """
        if self.combined_data is None:
            raise ValueError("Data not loaded. Call load_data_* first.")
        
        print("Creating daily aggregated data...")
        
        # Create time features
        print("  Adding time features...")
        df = self.create_time_features(self.combined_data)
        
        # Create daily timestamp
        df['date'] = df['start_time'].dt.date
        
        print("  Aggregating by day...")
        
        # Aggregate by day
        daily_agg = df.groupby('date').agg({
            # Target variables
            'trip_duration': ['count', 'mean', 'median', 'std', 'min', 'max'],
            
            # Weather features (daily averages)
            'temperature': ['mean', 'min', 'max'],
            'wind_speed': 'mean',
            'relative_humidity': 'mean',
            'cloud_cover': 'mean',
            'precipitation': 'sum',
            'utci': 'mean',
            
            # Weather events (any occurrence during the day)
            'rain': 'max',
            'snow': 'max',
            'mist_fog': 'max',
            
            # User type distribution
            'user_type': lambda x: (x == 'Subscriber').sum() / len(x) if len(x) > 0 else 0,
            
            # Time features
            'day_of_week': 'first',
            'month': 'first',
            'year': 'first',
            'day_of_year': 'first',
            'quarter': 'first',
            'is_weekend': 'first',
            'season': 'first'
        }).reset_index()
        
        # Clean up
        del df
        gc.collect()
        
        # Flatten column names
        daily_agg.columns = [f"{col[0]}_{col[1]}" if col[1] != '' else col[0] 
                            for col in daily_agg.columns]
        
        # Rename target columns for clarity
        daily_agg = daily_agg.rename(columns={
            'trip_duration_count': 'daily_ride_count',
            'trip_duration_mean': 'daily_avg_duration',
            'trip_duration_median': 'daily_median_duration',
            'trip_duration_std': 'daily_std_duration',
            'trip_duration_min': 'daily_min_duration',
            'trip_duration_max': 'daily_max_duration'
        })
        
        print("  Adding lag and rolling features...")
        
        # Add lag features for previous days
        daily_agg = daily_agg.sort_values('date')
        for lag in [1, 2, 3, 7, 14, 30]:
            daily_agg[f'daily_ride_count_lag_{lag}d'] = daily_agg['daily_ride_count'].shift(lag)
            daily_agg[f'daily_avg_duration_lag_{lag}d'] = daily_agg['daily_avg_duration'].shift(lag)
        
        # Add rolling averages
        for window in [3, 7, 14, 30]:
            daily_agg[f'daily_ride_count_rolling_{window}d'] = daily_agg['daily_ride_count'].rolling(window=window).mean()
            daily_agg[f'daily_avg_duration_rolling_{window}d'] = daily_agg['daily_avg_duration'].rolling(window=window).mean()
        
        self.daily_aggregated = daily_agg
        print(f"  Created daily data: {daily_agg.shape}")
        print(f"  Memory usage: ~{daily_agg.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        return daily_agg
    
    def save_processed_data(self, output_dir="data/processed"):
        """
        Save processed data to parquet files
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        if self.hourly_aggregated is not None:
            self.hourly_aggregated.to_parquet(f"{output_dir}/hourly_aggregated.parquet", index=False)
            print(f"Hourly aggregated data saved: {len(self.hourly_aggregated)} rows")
        
        if self.daily_aggregated is not None:
            self.daily_aggregated.to_parquet(f"{output_dir}/daily_aggregated.parquet", index=False)
            print(f"Daily aggregated data saved: {len(self.daily_aggregated)} rows")
    
    def get_feature_target_split(self, data_type='hourly', target='ride_count'):
        """
        Split data into features and targets for ML
        """
        if data_type == 'hourly':
            data = self.hourly_aggregated
        elif data_type == 'daily':
            data = self.daily_aggregated
        else:
            raise ValueError("data_type must be 'hourly' or 'daily'")
        
        if data is None:
            raise ValueError(f"{data_type} data not created. Call create_{data_type}_aggregated_data() first.")
        
        # Remove rows with NaN in target
        data_clean = data.dropna(subset=[target])
        
        # Define feature columns (exclude timestamp and target columns)
        exclude_cols = ['hour_timestamp', 'date'] + [col for col in data_clean.columns 
                                                    if any(t in col for t in ['ride_count', 'duration', 'daily_ride_count', 'daily_avg_duration'])]
        
        feature_cols = [col for col in data_clean.columns if col not in exclude_cols]
        
        # Identify categorical columns more robustly
        categorical_cols = []
        for col in feature_cols:
            if col in data_clean.columns:
                # Check if column is object/string type or has limited unique values
                if (data_clean[col].dtype == 'object' or 
                    data_clean[col].dtype.name == 'category' or
                    col in ['weather_cat', 'utci_cat', 'season', 'time_period']):
                    categorical_cols.append(col)
        
        print(f"Found categorical columns: {categorical_cols}")
        
        # One-hot encode categorical variables
        data_encoded = data_clean.copy()
        if categorical_cols:
            try:
                data_encoded = pd.get_dummies(data_clean, columns=categorical_cols, prefix=categorical_cols, dummy_na=True)
                print(f"Encoded {len(categorical_cols)} categorical columns")
            except Exception as e:
                print(f"Warning: Error in encoding categorical columns: {e}")
                # Fall back to simple encoding
                for col in categorical_cols:
                    if col in data_clean.columns:
                        # Convert to category codes
                        data_encoded[col] = pd.Categorical(data_clean[col]).codes
                        print(f"Fallback encoding for {col}")
        
        # Update feature columns after encoding
        exclude_cols_updated = ['hour_timestamp', 'date'] + [col for col in data_encoded.columns 
                                                          if any(t in col for t in ['ride_count', 'duration', 'daily_ride_count', 'daily_avg_duration'])]
        
        feature_cols_final = [col for col in data_encoded.columns if col not in exclude_cols_updated]
        
        # Ensure all features are numeric
        X = data_encoded[feature_cols_final].copy()
        
        # Convert any remaining non-numeric columns
        for col in X.columns:
            if X[col].dtype == 'object':
                print(f"Converting remaining object column {col} to numeric")
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                except:
                    # If conversion fails, use category codes
                    X[col] = pd.Categorical(X[col]).codes
        
        # Fill any NaN values that might have been created
        X = X.fillna(0)
        
        y = data_encoded[target]
        
        print(f"Final feature matrix shape: {X.shape}")
        print(f"Feature types: {X.dtypes.value_counts()}")
        
        return X, y, data_encoded


if __name__ == "__main__":
    # Example usage with different loading options
    preprocessor = RideDataPreprocessor()
    
    print("=== MEMORY-EFFICIENT PREPROCESSING OPTIONS ===")
    print("1. Lightweight (quick test): load_data_lightweight()")
    print("2. Single year sample: load_data_single_year()")
    print("3. Full chunked loading: load_data_chunked()")
    
    # Option 1: Lightweight for quick testing
    print("\nOption 1: Lightweight loading for quick testing...")
    try:
        data = preprocessor.load_data_lightweight()
        sample = preprocessor.explore_data()
        
        # Create aggregated datasets
        print("\nCreating hourly aggregated data...")
        hourly_data = preprocessor.create_hourly_aggregated_data()
        
        print("\nCreating daily aggregated data...")
        daily_data = preprocessor.create_daily_aggregated_data()
        
        # Save processed data
        print("\nSaving processed data...")
        preprocessor.save_processed_data()
        
        print("\nLightweight preprocessing complete!")
        print(f"Hourly data shape: {hourly_data.shape}")
        print(f"Daily data shape: {daily_data.shape}")
        
    except Exception as e:
        print(f"Error in lightweight processing: {e}")
        print("Please ensure parquet files exist in data/combined/") 