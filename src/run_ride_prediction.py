#!/usr/bin/env python3
"""
Unified script for CitiBike ride prediction
This script handles data preprocessing and machine learning prediction for:
- Number of rides (hourly and daily)
- Duration of rides (hourly and daily)

Usage:
    python src/run_ride_prediction.py
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

def main():
    """
    Main execution function
    """
    print("="*60)
    print("CitiBike Ride Prediction Pipeline")
    print("="*60)
    
    # Check if processed data exists
    processed_hourly = "data/processed/hourly_aggregated.parquet"
    processed_daily = "data/processed/daily_aggregated.parquet"
    
    if os.path.exists(processed_hourly) and os.path.exists(processed_daily):
        print("Processed data found. Skipping preprocessing...")
        
        # Load existing processed data
        preprocessor = RideDataPreprocessor()
        preprocessor.hourly_aggregated = pd.read_parquet(processed_hourly)
        preprocessor.daily_aggregated = pd.read_parquet(processed_daily)
        
        print(f"Loaded hourly data: {preprocessor.hourly_aggregated.shape}")
        print(f"Loaded daily data: {preprocessor.daily_aggregated.shape}")
        
    else:
        print("Processed data not found. Starting preprocessing...")
        
        # Initialize preprocessor
        preprocessor = RideDataPreprocessor(data_path="data/combined")
        
        # Load raw data with memory-efficient options
        print("\n1. Loading raw data...")
        try:
            # Memory-efficient loading options
            print("Choose loading option:")
            print("  1. Lightweight (500K rows, quick test)")
            print("  2. Single year sample (10% of 2024)")
            print("  3. Recent years chunked (2022-2024)")
            
            # Default to lightweight for better performance
            print("Using lightweight loading for better performance...")
            data = preprocessor.load_data_lightweight(max_total_rows=500000)
            
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Please ensure parquet files exist in data/combined/")
            return
        
        # Basic data exploration
        print("\n2. Exploring data...")
        sample_data = preprocessor.explore_data()
        
        # Create aggregated datasets
        print("\n3. Creating hourly aggregated data...")
        hourly_data = preprocessor.create_hourly_aggregated_data()
        print(f"Created hourly data: {hourly_data.shape}")
        
        print("\n4. Creating daily aggregated data...")
        daily_data = preprocessor.create_daily_aggregated_data()
        print(f"Created daily data: {daily_data.shape}")
        
        # Save processed data
        print("\n5. Saving processed data...")
        preprocessor.save_processed_data()
    
    print("\n" + "="*60)
    print("Starting Machine Learning Predictions")
    print("="*60)
    
    # Run predictions
    results_summary = {}
    
    try:
        # 1. Hourly ride count prediction
        print("\n" + "-"*50)
        print("1. HOURLY RIDE COUNT PREDICTION")
        print("-"*50)
        
        ride_count_trainer, ride_count_results = main_prediction_pipeline(
            preprocessor, target='ride_count', data_type='hourly'
        )
        
        # Get best model performance
        best_hourly_count = ride_count_results[ride_count_results['model'].str.contains('_test')].sort_values('rmse').iloc[0]
        results_summary['hourly_ride_count'] = {
            'best_model': best_hourly_count['model'].replace('_test', ''),
            'rmse': best_hourly_count['rmse'],
            'r2': best_hourly_count['r2']
        }
        
    except Exception as e:
        print(f"Error in hourly ride count prediction: {e}")
        results_summary['hourly_ride_count'] = {'error': str(e)}
    
    try:
        # 2. Hourly duration prediction
        print("\n" + "-"*50)
        print("2. HOURLY AVERAGE DURATION PREDICTION")
        print("-"*50)
        
        duration_trainer, duration_results = main_prediction_pipeline(
            preprocessor, target='avg_duration', data_type='hourly'
        )
        
        # Get best model performance
        best_hourly_duration = duration_results[duration_results['model'].str.contains('_test')].sort_values('rmse').iloc[0]
        results_summary['hourly_duration'] = {
            'best_model': best_hourly_duration['model'].replace('_test', ''),
            'rmse': best_hourly_duration['rmse'],
            'r2': best_hourly_duration['r2']
        }
        
    except Exception as e:
        print(f"Error in hourly duration prediction: {e}")
        results_summary['hourly_duration'] = {'error': str(e)}
    
    try:
        # 3. Daily ride count prediction
        print("\n" + "-"*50)
        print("3. DAILY RIDE COUNT PREDICTION")
        print("-"*50)
        
        daily_count_trainer, daily_count_results = main_prediction_pipeline(
            preprocessor, target='daily_ride_count', data_type='daily'
        )
        
        # Get best model performance
        best_daily_count = daily_count_results[daily_count_results['model'].str.contains('_test')].sort_values('rmse').iloc[0]
        results_summary['daily_ride_count'] = {
            'best_model': best_daily_count['model'].replace('_test', ''),
            'rmse': best_daily_count['rmse'],
            'r2': best_daily_count['r2']
        }
        
    except Exception as e:
        print(f"Error in daily ride count prediction: {e}")
        results_summary['daily_ride_count'] = {'error': str(e)}
    
    try:
        # 4. Daily duration prediction
        print("\n" + "-"*50)
        print("4. DAILY AVERAGE DURATION PREDICTION")
        print("-"*50)
        
        daily_duration_trainer, daily_duration_results = main_prediction_pipeline(
            preprocessor, target='daily_avg_duration', data_type='daily'
        )
        
        # Get best model performance
        best_daily_duration = daily_duration_results[daily_duration_results['model'].str.contains('_test')].sort_values('rmse').iloc[0]
        results_summary['daily_duration'] = {
            'best_model': best_daily_duration['model'].replace('_test', ''),
            'rmse': best_daily_duration['rmse'],
            'r2': best_daily_duration['r2']
        }
        
    except Exception as e:
        print(f"Error in daily duration prediction: {e}")
        results_summary['daily_duration'] = {'error': str(e)}
    
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
    summary_df = pd.DataFrame(results_summary).T
    summary_df.to_csv("results/prediction_summary.csv")
    print(f"Results summary saved to: results/prediction_summary.csv")
    
    return results_summary

def create_prediction_function_example():
    """
    Create an example function for making new predictions
    """
    example_code = '''
# Example: How to use trained models for new predictions

def predict_rides_for_conditions(weather_conditions, time_features):
    """
    Example function to predict rides given weather and time conditions
    
    Parameters:
    weather_conditions: dict with keys like 'temperature', 'wind_speed', etc.
    time_features: dict with keys like 'hour', 'day_of_week', etc.
    """
    
    # Load processed data and trained models
    import pandas as pd
    import pickle
    
    # This is a simplified example - you would need to:
    # 1. Load your trained model (saved from the training process)
    # 2. Prepare the input features in the same format as training
    # 3. Make predictions
    
    # Example input preparation
    input_features = {
        **weather_conditions,
        **time_features
    }
    
    # Convert to DataFrame with proper feature engineering
    input_df = pd.DataFrame([input_features])
    
    # Apply same preprocessing as training (scaling, encoding, etc.)
    # ... (preprocessing steps)
    
    # Make prediction with trained model
    # prediction = trained_model.predict(processed_input)
    
    # return prediction

# Example usage:
# weather = {
#     'temperature': 20.0,
#     'wind_speed': 5.0,
#     'relative_humidity': 60.0,
#     'precipitation': 0.0
# }
# 
# time_info = {
#     'hour': 8,
#     'day_of_week': 1,  # Monday
#     'month': 6,        # June
#     'is_weekend': 0
# }
# 
# predicted_rides = predict_rides_for_conditions(weather, time_info)
'''
    
    with open("src/prediction_example.py", "w") as f:
        f.write(example_code)
    
    print("Example prediction function saved to: src/prediction_example.py")

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("results/ride_prediction", exist_ok=True)
    
    # Run main pipeline
    results = main()
    
    # Create example prediction function
    create_prediction_function_example()
    
    print("\nNext steps:")
    print("1. Check results/prediction_summary.csv for model performance")
    print("2. See results/ride_prediction/ for visual analysis plots")
    print("3. See src/prediction_example.py for how to use trained models")
    print("4. Experiment with different features or model parameters")
    print("5. Consider using more years of data for better performance") 