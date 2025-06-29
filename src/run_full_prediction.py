#!/usr/bin/env python3
"""
Full dataset ride prediction with memory management
This script provides options for processing larger amounts of data while managing memory usage.

Usage:
    python src/run_full_prediction.py
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from ride_prediction_preprocessing import RideDataPreprocessor
from ride_prediction_models import RidePredictionModels, main_prediction_pipeline

def get_user_choice():
    """
    Get user's choice for data loading strategy
    """
    print("\n=== DATA LOADING OPTIONS ===")
    print("1. Lightweight Test (500K rows, ~5 min)")
    print("2. Single Year Sample (10% of 2024, ~10 min)")
    print("3. Single Year Full (All of 2024, ~30 min)")
    print("4. Recent Years Sample (10% of 2022-2024, ~20 min)")
    print("5. Recent Years Full (All 2022-2024, ~60 min)")
    print("6. All Years Sample (10% of all years, ~30 min)")
    print("7. All Years Full (Complete dataset, 2+ hours)")
    
    while True:
        try:
            choice = input("\nChoose option (1-7): ").strip()
            if choice in ['1', '2', '3', '4', '5', '6', '7']:
                return int(choice)
            else:
                print("Please enter a number between 1 and 7")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            sys.exit(0)
        except:
            print("Please enter a valid number")

def load_data_by_choice(preprocessor, choice):
    """
    Load data based on user choice
    """
    if choice == 1:
        print("Loading lightweight test data...")
        return preprocessor.load_data_lightweight(max_total_rows=500000)
    
    elif choice == 2:
        print("Loading single year sample...")
        return preprocessor.load_data_single_year(year=2024, sample_fraction=0.1)
    
    elif choice == 3:
        print("Loading full year 2024...")
        return preprocessor.load_data_chunked(years=[2024])
    
    elif choice == 4:
        print("Loading recent years sample...")
        return preprocessor.load_data_chunked(years=[2022, 2023, 2024], sample_fraction=0.1)
    
    elif choice == 5:
        print("Loading recent years full...")
        return preprocessor.load_data_chunked(years=[2022, 2023, 2024])
    
    elif choice == 6:
        print("Loading all years sample...")
        return preprocessor.load_data_chunked(years=[2019, 2020, 2021, 2022, 2023, 2024], sample_fraction=0.1)
    
    elif choice == 7:
        print("Loading complete dataset...")
        print("WARNING: This will take significant time and memory!")
        confirm = input("Are you sure? (yes/no): ").strip().lower()
        if confirm in ['yes', 'y']:
            return preprocessor.load_data_chunked(years=[2019, 2020, 2021, 2022, 2023, 2024])
        else:
            print("Cancelled. Defaulting to recent years sample...")
            return preprocessor.load_data_chunked(years=[2022, 2023, 2024], sample_fraction=0.1)

def main():
    """
    Main execution function with user choices
    """
    print("="*60)
    print("CitiBike Full Dataset Prediction Pipeline")
    print("="*60)
    
    # Check system resources
    import psutil
    memory_gb = psutil.virtual_memory().total / (1024**3)
    print(f"System RAM: {memory_gb:.1f} GB")
    
    if memory_gb < 8:
        print("⚠️  Warning: Less than 8GB RAM detected. Consider using smaller data options.")
    
    # Check if processed data exists
    processed_hourly = "data/processed/hourly_aggregated.parquet"
    processed_daily = "data/processed/daily_aggregated.parquet"
    
    use_existing = False
    if os.path.exists(processed_hourly) and os.path.exists(processed_daily):
        print("\nProcessed data found.")
        while True:
            choice = input("Use existing processed data? (y/n): ").strip().lower()
            if choice in ['y', 'yes']:
                use_existing = True
                break
            elif choice in ['n', 'no']:
                break
            else:
                print("Please enter 'y' or 'n'")
    
    if use_existing:
        print("Loading existing processed data...")
        preprocessor = RideDataPreprocessor()
        preprocessor.hourly_aggregated = pd.read_parquet(processed_hourly)
        preprocessor.daily_aggregated = pd.read_parquet(processed_daily)
        
        print(f"Loaded hourly data: {preprocessor.hourly_aggregated.shape}")
        print(f"Loaded daily data: {preprocessor.daily_aggregated.shape}")
        
    else:
        print("Starting fresh preprocessing...")
        
        # Get user choice for data loading
        choice = get_user_choice()
        
        # Initialize preprocessor with appropriate chunk size
        chunk_size = 50000 if choice >= 5 else 100000  # Smaller chunks for larger datasets
        preprocessor = RideDataPreprocessor(data_path="data/combined", chunk_size=chunk_size)
        
        # Show file information
        print("\n=== AVAILABLE DATA FILES ===")
        file_info = preprocessor.get_file_info()
        for info in file_info:
            print(f"{info['year']}: {info['size_gb']:.1f} GB")
        
        total_size = sum(f['size_gb'] for f in file_info)
        print(f"Total available: {total_size:.1f} GB")
        
        # Load data based on choice
        print(f"\n=== LOADING DATA (Option {choice}) ===")
        start_time = datetime.now()
        
        try:
            data = load_data_by_choice(preprocessor, choice)
        except MemoryError:
            print("❌ Memory error! Try a smaller dataset option.")
            return
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return
        
        load_time = datetime.now() - start_time
        print(f"Data loading completed in {load_time}")
        
        # Basic data exploration
        print("\n=== DATA EXPLORATION ===")
        sample_data = preprocessor.explore_data()
        
        # Add lag and rolling features to aggregated data
        print("\n=== ADDING LAG AND ROLLING FEATURES ===")
        start_time = datetime.now()
        
        print("Adding features to hourly data...")
        # Add lag features for previous hours
        hourly_data = preprocessor.hourly_aggregated.sort_values('hour_timestamp')
        for lag in [1, 2, 3, 6, 12, 24]:
            hourly_data[f'ride_count_lag_{lag}h'] = hourly_data['ride_count'].shift(lag)
            hourly_data[f'avg_duration_lag_{lag}h'] = hourly_data['avg_duration'].shift(lag)
        
        # Add rolling averages
        for window in [3, 6, 12, 24]:
            hourly_data[f'ride_count_rolling_{window}h'] = hourly_data['ride_count'].rolling(window=window).mean()
            hourly_data[f'avg_duration_rolling_{window}h'] = hourly_data['avg_duration'].rolling(window=window).mean()
        
        preprocessor.hourly_aggregated = hourly_data
        
        print("Adding features to daily data...")
        # Add lag features for previous days
        daily_data = preprocessor.daily_aggregated.sort_values('date')
        for lag in [1, 2, 3, 7, 14, 30]:
            daily_data[f'daily_ride_count_lag_{lag}d'] = daily_data['daily_ride_count'].shift(lag)
            daily_data[f'daily_avg_duration_lag_{lag}d'] = daily_data['daily_avg_duration'].shift(lag)
        
        # Add rolling averages
        for window in [3, 7, 14, 30]:
            daily_data[f'daily_ride_count_rolling_{window}d'] = daily_data['daily_ride_count'].rolling(window=window).mean()
            daily_data[f'daily_avg_duration_rolling_{window}d'] = daily_data['daily_avg_duration'].rolling(window=window).mean()
        
        preprocessor.daily_aggregated = daily_data
        
        feature_time = datetime.now() - start_time
        print(f"Feature engineering completed in {feature_time}")
        
        # Save processed data
        print("\n=== SAVING PROCESSED DATA ===")
        preprocessor.save_processed_data()
    
    # Run machine learning predictions
    print("\n" + "="*60)
    print("Starting Machine Learning Predictions")
    print("="*60)
    
    # Ask user which predictions to run
    print("\nSelect predictions to run:")
    print("1. Hourly ride count only")
    print("2. Daily ride count only") 
    print("3. Both ride counts")
    print("4. All predictions (counts + durations)")
    
    while True:
        try:
            pred_choice = input("Choose prediction option (1-4): ").strip()
            if pred_choice in ['1', '2', '3', '4']:
                pred_choice = int(pred_choice)
                break
            else:
                print("Please enter a number between 1 and 4")
        except:
            print("Please enter a valid number")
    
    results_summary = {}
    
    try:
        if pred_choice in [1, 3, 4]:
            # Hourly ride count prediction
            print("\n" + "-"*50)
            print("HOURLY RIDE COUNT PREDICTION")
            print("-"*50)
            
            ride_count_trainer, ride_count_results = main_prediction_pipeline(
                preprocessor, target='ride_count', data_type='hourly'
            )
            
            best_hourly_count = ride_count_results[ride_count_results['model'].str.contains('_test')].sort_values('rmse').iloc[0]
            results_summary['hourly_ride_count'] = {
                'best_model': best_hourly_count['model'].replace('_test', ''),
                'rmse': best_hourly_count['rmse'],
                'r2': best_hourly_count['r2']
            }
        
        if pred_choice in [2, 3, 4]:
            # Daily ride count prediction
            print("\n" + "-"*50)
            print("DAILY RIDE COUNT PREDICTION")
            print("-"*50)
            
            daily_count_trainer, daily_count_results = main_prediction_pipeline(
                preprocessor, target='daily_ride_count', data_type='daily'
            )
            
            best_daily_count = daily_count_results[daily_count_results['model'].str.contains('_test')].sort_values('rmse').iloc[0]
            results_summary['daily_ride_count'] = {
                'best_model': best_daily_count['model'].replace('_test', ''),
                'rmse': best_daily_count['rmse'],
                'r2': best_daily_count['r2']
            }
        
        if pred_choice == 4:
            # Duration predictions
            print("\n" + "-"*50)
            print("HOURLY AVERAGE DURATION PREDICTION")
            print("-"*50)
            
            duration_trainer, duration_results = main_prediction_pipeline(
                preprocessor, target='avg_duration', data_type='hourly'
            )
            
            best_hourly_duration = duration_results[duration_results['model'].str.contains('_test')].sort_values('rmse').iloc[0]
            results_summary['hourly_duration'] = {
                'best_model': best_hourly_duration['model'].replace('_test', ''),
                'rmse': best_hourly_duration['rmse'],
                'r2': best_hourly_duration['r2']
            }
            
            print("\n" + "-"*50)
            print("DAILY AVERAGE DURATION PREDICTION")
            print("-"*50)
            
            daily_duration_trainer, daily_duration_results = main_prediction_pipeline(
                preprocessor, target='daily_avg_duration', data_type='daily'
            )
            
            best_daily_duration = daily_duration_results[daily_duration_results['model'].str.contains('_test')].sort_values('rmse').iloc[0]
            results_summary['daily_duration'] = {
                'best_model': best_daily_duration['model'].replace('_test', ''),
                'rmse': best_daily_duration['rmse'],
                'r2': best_daily_duration['r2']
            }
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        print("This might be due to insufficient data or memory constraints.")
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    
    for task, results in results_summary.items():
        print(f"\n{task.upper().replace('_', ' ')}:")
        if 'error' in results:
            print(f"  ❌ Error: {results['error']}")
        else:
            print(f"  ✅ Best Model: {results['best_model']}")
            print(f"  📊 RMSE: {results['rmse']:.2f}")
            print(f"  📈 R²: {results['r2']:.3f}")
    
    print("\n" + "="*60)
    print("Pipeline Complete!")
    print("="*60)
    
    # Save summary to file
    if results_summary:
        summary_df = pd.DataFrame(results_summary).T
        summary_df.to_csv("results/full_prediction_summary.csv")
        print(f"Results summary saved to: results/full_prediction_summary.csv")
    
    return results_summary

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("results/ride_prediction", exist_ok=True)
    
    try:
        # Check if required packages are available
        import psutil
    except ImportError:
        print("Installing psutil for memory monitoring...")
        os.system("pip install psutil")
        import psutil
    
    # Run main pipeline
    results = main()
    
    print("\nFor more information, check:")
    print("- info/ride_prediction_guide.md (detailed documentation)")
    print("- results/full_prediction_summary.csv (performance results)")
    print("- results/ride_prediction/ (visual analysis plots)")
    print("- data/processed/ (preprocessed data for future use)") 