#!/usr/bin/env python3
"""
Standalone User Type Infrastructure Analysis Plot Generator

This script generates only the user type infrastructure comparison plot
without running the full analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

def load_and_preprocess_single_year(year: str, file_path: str):
    """Load and preprocess data for a single year."""
    print(f"Loading {year} data...")
    data = pd.read_parquet(file_path)
    print(f"Loaded {year}: {len(data):,} rides")
    
    # Convert datetime columns
    data['start_time'] = pd.to_datetime(data['start_time'])
    data['stop_time'] = pd.to_datetime(data['stop_time'])
    
    # Extract temporal features
    data['hour'] = data['start_time'].dt.hour
    data['day_of_week'] = data['start_time'].dt.dayofweek
    data['month'] = data['start_time'].dt.month
    data['year_month'] = data['start_time'].dt.to_period('M')
    
    print(f"Preprocessed {year}: {len(data):,} rides")
    return data

def aggregate_monthly_data(data: pd.DataFrame, year: str) -> pd.DataFrame:
    """Aggregate data by station and month with user type data."""
    print(f"Aggregating {year} data by station-month...")
    
    # Standard aggregation for model training and infrastructure analysis
    monthly_data = data.groupby(['start_station_id', 'year_month']).agg({
        'trip_duration': 'count',  # Count of rides
        'start_station_latitude': 'first',
        'start_station_longitude': 'first',
        'temperature': 'mean',
        'relative_humidity': 'mean',
        'wind_speed': 'mean',
        'precipitation': 'mean',
        'weather_cat': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
        'utci_cat': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
    }).reset_index()
    
    # Add user type aggregation for separate analysis
    # Check for both possible column names
    user_type_col = None
    if 'user_type' in data.columns:
        user_type_col = 'user_type'
    elif 'usertype' in data.columns:
        user_type_col = 'usertype'
    
    print(f"Available columns: {list(data.columns)}")
    print(f"Looking for user type column... Found: {user_type_col}")
    
    if user_type_col is not None:
        # Check unique values in user type column
        unique_user_types = data[user_type_col].unique()
        print(f"Unique user types: {unique_user_types}")
        
        user_type_data = data.groupby(['start_station_id', 'year_month', user_type_col]).agg({
            'trip_duration': 'count'  # Count rides by user type
        }).reset_index()
        
        user_type_pivot = user_type_data.pivot_table(
            index=['start_station_id', 'year_month'], 
            columns=user_type_col, 
            values='trip_duration', 
            fill_value=0
        ).reset_index()
        user_type_pivot.columns.name = None
        
        # Merge user type data with monthly data
        monthly_data = monthly_data.merge(user_type_pivot, on=['start_station_id', 'year_month'], how='left')
        
        # Standardize column names and fill NaN values with 0
        # Map common user type variations to standard names
        for col in monthly_data.columns:
            if col.lower() in ['member', 'subscriber']:
                monthly_data = monthly_data.rename(columns={col: 'Member'})
            elif col.lower() in ['casual', 'customer']:
                monthly_data = monthly_data.rename(columns={col: 'Casual'})
        
        # Fill NaN values with 0 for user types
        for col in ['Member', 'Casual']:
            if col in monthly_data.columns:
                monthly_data[col] = monthly_data[col].fillna(0)
    else:
        print("Warning: No user type column found in data")
    
    # Rename trip_duration count column
    monthly_data = monthly_data.rename(columns={'trip_duration': 'monthly_rides'})
    
    # Extract month number
    monthly_data['month'] = monthly_data['year_month'].dt.month
    
    # One-hot encode categorical variables
    weather_dummies = pd.get_dummies(monthly_data['weather_cat'], prefix='weather')
    utci_dummies = pd.get_dummies(monthly_data['utci_cat'], prefix='utci')
    
    result = pd.concat([monthly_data, weather_dummies, utci_dummies], axis=1)
    
    print(f"Result: {len(result):,} station-month records")
    print(f"Final columns: {list(result.columns)}")
    return result

def simple_infrastructure_identification(data, infrastructure_path):
    """Simple infrastructure identification using fixed buffer."""
    # Load infrastructure data
    infrastructure = pd.read_csv(infrastructure_path)
    
    # Get unique stations
    stations = data[['start_station_id', 'start_station_latitude', 'start_station_longitude']].drop_duplicates()
    stations = stations.dropna()
    
    # Simple buffer approach (0.005 degrees ≈ 500m)
    buffer_degrees = 0.005
    affected_stations = set()
    
    for _, infra in infrastructure.iterrows():
        if pd.isna(infra['latitude']) or pd.isna(infra['longitude']):
            continue
            
        # Find stations within buffer
        lat_min = infra['latitude'] - buffer_degrees
        lat_max = infra['latitude'] + buffer_degrees
        lon_min = infra['longitude'] - buffer_degrees
        lon_max = infra['longitude'] + buffer_degrees
        
        nearby_stations = stations[
            (stations['start_station_latitude'] >= lat_min) &
            (stations['start_station_latitude'] <= lat_max) &
            (stations['start_station_longitude'] >= lon_min) &
            (stations['start_station_longitude'] <= lon_max)
        ]
        
        affected_stations.update(nearby_stations['start_station_id'].tolist())
    
    return list(affected_stations)

def train_simple_models(train_data, affected_stations):
    """Train simple baseline and enhanced models."""
    # Baseline features (exclude original categorical columns)
    baseline_features = [
        'month', 'start_station_latitude', 'start_station_longitude',
        'temperature', 'relative_humidity', 'wind_speed', 'precipitation'
    ]
    
    # Add only one-hot encoded weather dummy columns (exclude original categorical columns)
    weather_cols = [col for col in train_data.columns if col.startswith('weather_') and col != 'weather_cat']
    utci_cols = [col for col in train_data.columns if col.startswith('utci_') and col != 'utci_cat']
    baseline_features.extend(weather_cols)
    baseline_features.extend(utci_cols)
    
    # Add user type features if available
    if 'Member' in train_data.columns and 'Casual' in train_data.columns:
        train_data['total_user_rides'] = train_data['Member'] + train_data['Casual']
        train_data['member_ratio'] = train_data['Member'] / (train_data['total_user_rides'] + 1e-6)
        train_data['casual_ratio'] = train_data['Casual'] / (train_data['total_user_rides'] + 1e-6)
        baseline_features.extend(['member_ratio', 'casual_ratio'])
    
    # Enhanced features (add infrastructure proximity)
    enhanced_features = baseline_features.copy()
    train_data['near_infrastructure'] = train_data['start_station_id'].isin(affected_stations).astype(int)
    enhanced_features.append('near_infrastructure')
    
    print(f"Training features ({len(baseline_features)} baseline, {len(enhanced_features)} enhanced):")
    print(f"Baseline: {baseline_features}")
    print(f"Enhanced additional: ['near_infrastructure']")
    
    # Prepare data
    X_baseline = train_data[baseline_features].fillna(0)
    X_enhanced = train_data[enhanced_features].fillna(0)
    y = train_data['monthly_rides']
    
    # Train models
    baseline_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    enhanced_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    
    baseline_model.fit(X_baseline, y)
    enhanced_model.fit(X_enhanced, y)
    
    return baseline_model, enhanced_model, baseline_features, enhanced_features

def compare_predictions(baseline_model, enhanced_model, baseline_features, enhanced_features, test_data, affected_stations, year):
    """Compare predictions and create results dataframe."""
    # Add infrastructure feature to test data
    test_data['near_infrastructure'] = test_data['start_station_id'].isin(affected_stations).astype(int)
    
    # Add user type features if available
    if 'Member' in test_data.columns and 'Casual' in test_data.columns:
        test_data['total_user_rides'] = test_data['Member'] + test_data['Casual']
        test_data['member_ratio'] = test_data['Member'] / (test_data['total_user_rides'] + 1e-6)
        test_data['casual_ratio'] = test_data['Casual'] / (test_data['total_user_rides'] + 1e-6)
    
    # Ensure features exist in test data (filter out any that don't exist)
    baseline_features_filtered = [f for f in baseline_features if f in test_data.columns]
    enhanced_features_filtered = [f for f in enhanced_features if f in test_data.columns]
    
    print(f"Baseline features available: {len(baseline_features_filtered)}/{len(baseline_features)}")
    print(f"Enhanced features available: {len(enhanced_features_filtered)}/{len(enhanced_features)}")
    
    # Make predictions
    X_baseline = test_data[baseline_features_filtered].fillna(0)
    X_enhanced = test_data[enhanced_features_filtered].fillna(0)
    
    baseline_pred = baseline_model.predict(X_baseline)
    enhanced_pred = enhanced_model.predict(X_enhanced)
    
    # Create results dataframe
    results_df = test_data.copy()
    results_df['baseline_pred'] = baseline_pred
    results_df['enhanced_pred'] = enhanced_pred
    results_df['is_affected'] = results_df['start_station_id'].isin(affected_stations)
    
    # Calculate performance metrics
    actual = test_data['monthly_rides']
    baseline_r2 = r2_score(actual, baseline_pred)
    enhanced_r2 = r2_score(actual, enhanced_pred)
    baseline_mae = mean_absolute_error(actual, baseline_pred)
    enhanced_mae = mean_absolute_error(actual, enhanced_pred)
    
    return {
        'year': year,
        'baseline_r2': baseline_r2,
        'enhanced_r2': enhanced_r2,
        'baseline_mae': baseline_mae,
        'enhanced_mae': enhanced_mae,
        'results_df': results_df
    }

def plot_user_type_infrastructure_analysis(results_2023, results_2024, output_dir):
    """Generate the user type infrastructure analysis plot."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract data for analysis
    data_2023 = results_2023['results_df']
    data_2024 = results_2024['results_df']
    
    print(f"2023 data columns: {list(data_2023.columns)}")
    print(f"2024 data columns: {list(data_2024.columns)}")
    
    # Check if user type data is available
    if 'Member' not in data_2023.columns or 'Casual' not in data_2023.columns:
        # Create placeholder plot if no user type data
        for ax in [ax1, ax2, ax3, ax4]:
            ax.text(0.5, 0.5, 'User Type Data\nNot Available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('User Type Analysis')
        
        plt.suptitle('User Type Infrastructure Analysis\n(No User Type Data Available)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "user_type_infrastructure_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    # Calculate user type proportions for affected vs unaffected stations
    for year_idx, (data, year) in enumerate([(data_2023, '2023'), (data_2024, '2024')]):
        ax = ax1 if year_idx == 0 else ax2
        
        # Calculate total rides by user type for affected vs unaffected
        affected_data = data[data['is_affected']].copy()
        unaffected_data = data[~data['is_affected']].copy()
        
        print(f"{year} - Affected stations: {len(affected_data)}, Unaffected stations: {len(unaffected_data)}")
        
        affected_member = affected_data['Member'].sum() if len(affected_data) > 0 else 0
        affected_casual = affected_data['Casual'].sum() if len(affected_data) > 0 else 0
        unaffected_member = unaffected_data['Member'].sum() if len(unaffected_data) > 0 else 0
        unaffected_casual = unaffected_data['Casual'].sum() if len(unaffected_data) > 0 else 0
        
        print(f"{year} - Affected: Member={affected_member}, Casual={affected_casual}")
        print(f"{year} - Unaffected: Member={unaffected_member}, Casual={unaffected_casual}")
        
        # Create comparison data
        categories = ['Affected Stations', 'Unaffected Stations']
        member_values = [affected_member, unaffected_member]
        casual_values = [affected_casual, unaffected_casual]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, member_values, width, label='Member', color='steelblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, casual_values, width, label='Casual', color='orange', alpha=0.8)
        
        ax.set_xlabel('Station Type')
        ax.set_ylabel('Total Rides')
        ax.set_title(f'{year}: Member vs Casual Ridership\nby Infrastructure Status')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{int(height/1000)}K', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Member/Casual ratio analysis by infrastructure status
    affected_member_ratio_2023 = data_2023[data_2023['is_affected']]['Member'].sum() / (data_2023[data_2023['is_affected']]['Member'].sum() + data_2023[data_2023['is_affected']]['Casual'].sum()) if len(data_2023[data_2023['is_affected']]) > 0 else 0
    unaffected_member_ratio_2023 = data_2023[~data_2023['is_affected']]['Member'].sum() / (data_2023[~data_2023['is_affected']]['Member'].sum() + data_2023[~data_2023['is_affected']]['Casual'].sum()) if len(data_2023[~data_2023['is_affected']]) > 0 else 0
    
    affected_member_ratio_2024 = data_2024[data_2024['is_affected']]['Member'].sum() / (data_2024[data_2024['is_affected']]['Member'].sum() + data_2024[data_2024['is_affected']]['Casual'].sum()) if len(data_2024[data_2024['is_affected']]) > 0 else 0
    unaffected_member_ratio_2024 = data_2024[~data_2024['is_affected']]['Member'].sum() / (data_2024[~data_2024['is_affected']]['Member'].sum() + data_2024[~data_2024['is_affected']]['Casual'].sum()) if len(data_2024[~data_2024['is_affected']]) > 0 else 0
    
    categories = ['2023 Affected', '2023 Unaffected', '2024 Affected', '2024 Unaffected']
    member_ratios = [affected_member_ratio_2023, unaffected_member_ratio_2023, affected_member_ratio_2024, unaffected_member_ratio_2024]
    colors = ['darkred', 'lightcoral', 'darkgreen', 'lightgreen']
    
    bars = ax3.bar(categories, member_ratios, color=colors, alpha=0.8)
    ax3.set_ylabel('Member Ratio (Member / Total)')
    ax3.set_title('Member Usage Ratio:\nInfrastructure vs Non-Infrastructure Stations')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    ax3.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='50% Split')
    ax3.legend()
    
    # Add value labels
    for bar, ratio in zip(bars, member_ratios):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{ratio:.2f}', ha='center', va='bottom')
    
    # Plot 4: Infrastructure effect by user type (gap analysis)
    affected_2023 = data_2023[data_2023['is_affected']]
    unaffected_2023 = data_2023[~data_2023['is_affected']]
    affected_2024 = data_2024[data_2024['is_affected']]
    unaffected_2024 = data_2024[~data_2024['is_affected']]
    
    # Calculate infrastructure effect (difference between affected and unaffected) by user type
    member_effect_2023 = (affected_2023['Member'].mean() - unaffected_2023['Member'].mean()) if len(affected_2023) > 0 and len(unaffected_2023) > 0 else 0
    casual_effect_2023 = (affected_2023['Casual'].mean() - unaffected_2023['Casual'].mean()) if len(affected_2023) > 0 and len(unaffected_2023) > 0 else 0
    member_effect_2024 = (affected_2024['Member'].mean() - unaffected_2024['Member'].mean()) if len(affected_2024) > 0 and len(unaffected_2024) > 0 else 0
    casual_effect_2024 = (affected_2024['Casual'].mean() - unaffected_2024['Casual'].mean()) if len(affected_2024) > 0 and len(unaffected_2024) > 0 else 0
    
    print(f"Infrastructure effects - Member 2023: {member_effect_2023:.1f}, Casual 2023: {casual_effect_2023:.1f}")
    print(f"Infrastructure effects - Member 2024: {member_effect_2024:.1f}, Casual 2024: {casual_effect_2024:.1f}")
    
    years = ['2023', '2024']
    member_effects = [member_effect_2023, member_effect_2024]
    casual_effects = [casual_effect_2023, casual_effect_2024]
    
    x = np.arange(len(years))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, member_effects, width, label='Member Effect', color='steelblue', alpha=0.8)
    bars2 = ax4.bar(x + width/2, casual_effects, width, label='Casual Effect', color='orange', alpha=0.8)
    
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Infrastructure Effect (rides/month)')
    ax4.set_title('Infrastructure Effect by User Type\n(Affected - Unaffected Stations)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(years)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + (5 if height > 0 else -15),
                    f'{height:.1f}', ha='center', va='bottom' if height > 0 else 'top')
    
    plt.suptitle('User Type Infrastructure Impact Analysis:\nHow Infrastructure Affects Members vs Casual Users', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "user_type_infrastructure_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"User type analysis plot saved to {output_dir}")

def main():
    """Main execution function for standalone user type plot generation."""
    print("Generating User Type Infrastructure Analysis Plot...")
    
    # Define file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    citibike_2022_path = os.path.join(project_root, "data", "combined", "2022_combined_citibike_weather.parquet")
    citibike_2023_path = os.path.join(project_root, "data", "combined", "2023_combined_citibike_weather.parquet")
    citibike_2024_path = os.path.join(project_root, "data", "combined", "2024_combined_citibike_weather.parquet")
    infrastructure_path = os.path.join(project_root, "data", "nyc_streets_geocoded_with_years.csv")
    
    output_dir = os.path.join(project_root, "results", "clean_infrastructure_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    try:
        # Load and process data
        print("Loading data...")
        data_2022 = load_and_preprocess_single_year("2022", citibike_2022_path)
        data_2023 = load_and_preprocess_single_year("2023", citibike_2023_path) 
        data_2024 = load_and_preprocess_single_year("2024", citibike_2024_path)
        
        # Aggregate monthly data
        print("Aggregating monthly data...")
        train_data = aggregate_monthly_data(data_2022, "2022")
        test_data_2023 = aggregate_monthly_data(data_2023, "2023")
        test_data_2024 = aggregate_monthly_data(data_2024, "2024")
        
        # Identify affected stations
        print("Identifying infrastructure-affected stations...")
        affected_stations = simple_infrastructure_identification(train_data, infrastructure_path)
        print(f"Found {len(affected_stations)} affected stations")
        
        # Train models
        print("Training models...")
        baseline_model, enhanced_model, baseline_features, enhanced_features = train_simple_models(train_data, affected_stations)
        
        # Compare predictions
        print("Comparing predictions...")
        results_2023 = compare_predictions(baseline_model, enhanced_model, baseline_features, enhanced_features, test_data_2023, affected_stations, "2023")
        results_2024 = compare_predictions(baseline_model, enhanced_model, baseline_features, enhanced_features, test_data_2024, affected_stations, "2024")
        
        # Generate user type plot
        print("Generating user type analysis plot...")
        plot_user_type_infrastructure_analysis(results_2023, results_2024, output_dir)
        
        print("User type analysis plot generation complete!")
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
