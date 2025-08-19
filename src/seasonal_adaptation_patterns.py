"""
Seasonal Adaptation Patterns Analysis
====================================

This script analyzes how CitiBike ridership patterns adapt to seasonal changes
in weather conditions throughout 2024.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore')

# Create results directory
results_dir = os.path.join('results', 'seasonal_adaptation_patterns')
os.makedirs(results_dir, exist_ok=True)

def load_2024_data():
    """Load and prepare 2024 data for seasonal analysis"""
    print("Loading 2024 CitiBike weather data...")
    
    file_path = 'data/combined/2024_combined_citibike_weather.parquet'
    df = pd.read_parquet(file_path)
    
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['month'] = df['start_time'].dt.month
    df['day_of_year'] = df['start_time'].dt.dayofyear
    df['hour'] = df['start_time'].dt.hour
    
    # Add season classification
    df['season'] = df['month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })
    
    # Create month names
    df['month_name'] = df['month'].map({
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
        7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
    })
    
    print(f"Loaded {len(df):,} rides from 2024")
    return df

def analyze_seasonal_trends(df):
    """Analyze seasonal ridership and weather trends"""
    print("Analyzing seasonal trends...")
    
    # Monthly aggregations
    monthly_data = df.groupby(['month', 'month_name']).agg({
        'start_time': 'count',
        'trip_duration': 'mean',
        'temperature': 'mean',
        'relative_humidity': 'mean',
        'precipitation': 'mean'
    }).reset_index()
    
    monthly_data.columns = ['month', 'month_name', 'total_rides', 'avg_duration', 
                           'avg_temp', 'avg_humidity', 'avg_precipitation']
    
    # Seasonal aggregations
    seasonal_data = df.groupby('season').agg({
        'start_time': 'count',
        'trip_duration': 'mean',
        'temperature': ['mean', 'std'],
        'relative_humidity': ['mean', 'std'],
        'precipitation': ['mean', 'sum']
    }).reset_index()
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Seasonal Adaptation Patterns - 2024', fontsize=16, fontweight='bold')
    
    # 1. Monthly ridership trend
    ax1 = axes[0, 0]
    ax1.plot(monthly_data['month'], monthly_data['total_rides'], 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Total Rides')
    ax1.set_title('Monthly Ridership Trend')
    ax1.set_xticks(monthly_data['month'])
    ax1.set_xticklabels(monthly_data['month_name'], rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # 2. Temperature vs ridership
    ax2 = axes[0, 1]
    colors = plt.cm.RdYlBu_r(np.linspace(0, 1, 12))
    scatter = ax2.scatter(monthly_data['avg_temp'], monthly_data['total_rides'], 
                         c=monthly_data['month'], cmap='RdYlBu_r', s=100, alpha=0.7)
    
    # Add month labels
    for i, row in monthly_data.iterrows():
        ax2.annotate(row['month_name'], (row['avg_temp'], row['total_rides']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax2.set_xlabel('Average Temperature (°C)')
    ax2.set_ylabel('Total Rides')
    ax2.set_title('Temperature vs Ridership by Month')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='Month')
    
    # 3. Seasonal comparison
    ax3 = axes[1, 0]
    seasonal_rides = seasonal_data[('start_time', 'count')].values
    season_names = seasonal_data['season'].values
    season_colors = ['blue', 'green', 'red', 'orange']
    
    bars = ax3.bar(season_names, seasonal_rides, color=season_colors)
    ax3.set_xlabel('Season')
    ax3.set_ylabel('Total Rides')
    ax3.set_title('Ridership by Season')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, seasonal_rides):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(seasonal_rides)*0.01,
                f'{value:,}', ha='center', va='bottom', fontsize=10)
    
    # 4. Precipitation patterns
    ax4 = axes[1, 1]
    monthly_precip = monthly_data['avg_precipitation']
    bars = ax4.bar(monthly_data['month_name'], monthly_precip, color='skyblue')
    ax4.set_xlabel('Month')
    ax4.set_ylabel('Average Precipitation (mm)')
    ax4.set_title('Monthly Precipitation Patterns')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'seasonal_trends.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save data
    monthly_data.to_csv(os.path.join(results_dir, 'monthly_data.csv'), index=False)
    seasonal_data.to_csv(os.path.join(results_dir, 'seasonal_data.csv'), index=False)
    
    return monthly_data, seasonal_data

def analyze_temperature_adaptation(df):
    """Analyze temperature adaptation by season"""
    print("Analyzing temperature adaptation patterns...")
    
    # Create temperature bins for analysis
    temp_bins = np.arange(-20, 40, 5)
    df['temp_bin'] = pd.cut(df['temperature'], bins=temp_bins, labels=temp_bins[:-1])
    
    # Calculate ridership by temperature and season
    temp_seasonal = df.groupby(['temp_bin', 'season']).agg({
        'start_time': 'count',
        'trip_duration': 'mean'
    }).reset_index()
    temp_seasonal.columns = ['temperature', 'season', 'ride_count', 'avg_duration']
    temp_seasonal = temp_seasonal.dropna()
    
    # Calculate relative ridership within each season
    for season in ['Winter', 'Spring', 'Summer', 'Fall']:
        season_data = temp_seasonal[temp_seasonal['season'] == season]
        if len(season_data) > 0:
            max_rides = season_data['ride_count'].max()
            temp_seasonal.loc[temp_seasonal['season'] == season, 'relative_ridership'] = (
                temp_seasonal[temp_seasonal['season'] == season]['ride_count'] / max_rides
            )
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Temperature Adaptation by Season - 2024', fontsize=16, fontweight='bold')
    
    # 1. Temperature tolerance curves
    ax1 = axes[0, 0]
    season_colors = {'Winter': 'blue', 'Spring': 'green', 'Summer': 'red', 'Fall': 'orange'}
    
    for season in ['Winter', 'Spring', 'Summer', 'Fall']:
        season_data = temp_seasonal[temp_seasonal['season'] == season]
        if len(season_data) > 0:
            ax1.plot(season_data['temperature'], season_data['relative_ridership'], 
                    'o-', color=season_colors[season], label=season, markersize=6, linewidth=2)
    
    ax1.set_xlabel('Temperature (°C)')
    ax1.set_ylabel('Relative Ridership')
    ax1.set_title('Temperature Tolerance by Season')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Optimal temperatures by season
    ax2 = axes[0, 1]
    optimal_temps = []
    seasons = []
    
    for season in ['Winter', 'Spring', 'Summer', 'Fall']:
        season_data = temp_seasonal[temp_seasonal['season'] == season]
        if len(season_data) > 0 and 'relative_ridership' in season_data.columns:
            optimal_temp = season_data.loc[season_data['relative_ridership'].idxmax(), 'temperature']
            optimal_temps.append(optimal_temp)
            seasons.append(season)
    
    if optimal_temps:
        bars = ax2.bar(seasons, optimal_temps, color=[season_colors[s] for s in seasons])
        ax2.set_xlabel('Season')
        ax2.set_ylabel('Optimal Temperature (°C)')
        ax2.set_title('Optimal Temperature by Season')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, optimal_temps):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}°C', ha='center', va='bottom', fontsize=10)
    
    # 3. Temperature range heatmap
    ax3 = axes[1, 0]
    if len(temp_seasonal) > 0:
        pivot_temp = temp_seasonal.pivot_table(values='ride_count', 
                                              index='temperature', 
                                              columns='season', 
                                              fill_value=0)
        
        sns.heatmap(pivot_temp, cmap='YlOrRd', ax=ax3, cbar_kws={'label': 'Ride Count'})
        ax3.set_title('Ridership Heatmap: Temperature vs Season')
        ax3.set_xlabel('Season')
        ax3.set_ylabel('Temperature (°C)')
    
    # 4. Trip duration vs temperature
    ax4 = axes[1, 1]
    for season in ['Winter', 'Spring', 'Summer', 'Fall']:
        season_data = temp_seasonal[temp_seasonal['season'] == season]
        if len(season_data) > 0:
            ax4.scatter(season_data['temperature'], season_data['avg_duration']/60, 
                       color=season_colors[season], label=season, alpha=0.7, s=50)
    
    ax4.set_xlabel('Temperature (°C)')
    ax4.set_ylabel('Average Trip Duration (minutes)')
    ax4.set_title('Trip Duration vs Temperature by Season')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'temperature_adaptation.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    temp_seasonal.to_csv(os.path.join(results_dir, 'temperature_adaptation.csv'), index=False)
    
    return temp_seasonal

def analyze_precipitation_adaptation(df):
    """Analyze adaptation to precipitation by season"""
    print("Analyzing precipitation adaptation...")
    
    # Create precipitation categories
    df['precip_category'] = pd.cut(df['precipitation'], 
                                  bins=[0, 0.1, 2, 5, 100], 
                                  labels=['None', 'Light', 'Moderate', 'Heavy'])
    
    # Calculate impact by season
    precip_impact = df.groupby(['season', 'precip_category']).agg({
        'start_time': 'count',
        'trip_duration': 'mean'
    }).reset_index()
    precip_impact.columns = ['season', 'precip_category', 'ride_count', 'avg_duration']
    
    # Calculate relative impact
    for season in ['Winter', 'Spring', 'Summer', 'Fall']:
        season_data = precip_impact[precip_impact['season'] == season]
        no_precip = season_data[season_data['precip_category'] == 'None']['ride_count']
        if len(no_precip) > 0:
            baseline = no_precip.values[0]
            precip_impact.loc[precip_impact['season'] == season, 'relative_impact'] = (
                precip_impact[precip_impact['season'] == season]['ride_count'] / baseline
            )
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Precipitation Adaptation by Season - 2024', fontsize=16, fontweight='bold')
    
    # 1. Precipitation impact by season
    ax1 = axes[0, 0]
    if 'relative_impact' in precip_impact.columns:
        pivot_impact = precip_impact.pivot_table(values='relative_impact', 
                                                index='precip_category', 
                                                columns='season', 
                                                fill_value=1)
        pivot_impact.plot(kind='bar', ax=ax1, width=0.8)
        ax1.set_xlabel('Precipitation Category')
        ax1.set_ylabel('Relative Ridership Impact')
        ax1.set_title('Precipitation Impact by Season')
        ax1.legend(title='Season')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=1, color='red', linestyle='--', alpha=0.7)
    
    # 2. Precipitation distribution by season
    ax2 = axes[0, 1]
    precip_dist = df.groupby(['season', 'precip_category']).size().unstack(fill_value=0)
    precip_dist.plot(kind='bar', ax=ax2, stacked=True, width=0.8)
    ax2.set_xlabel('Season')
    ax2.set_ylabel('Number of Hours')
    ax2.set_title('Precipitation Distribution by Season')
    ax2.legend(title='Precipitation')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 3. Duration impact
    ax3 = axes[1, 0]
    if len(precip_impact) > 0:
        duration_pivot = precip_impact.pivot_table(values='avg_duration', 
                                                  index='precip_category', 
                                                  columns='season', 
                                                  fill_value=0)
        duration_pivot_min = duration_pivot / 60
        duration_pivot_min.plot(kind='bar', ax=ax3, width=0.8)
        ax3.set_xlabel('Precipitation Category')
        ax3.set_ylabel('Average Trip Duration (minutes)')
        ax3.set_title('Trip Duration by Precipitation and Season')
        ax3.legend(title='Season')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
    
    # 4. Seasonal resilience score
    ax4 = axes[1, 1]
    if 'relative_impact' in precip_impact.columns:
        resilience_scores = precip_impact.groupby('season')['relative_impact'].mean()
        season_colors = ['blue', 'red', 'green', 'orange']  # Fall, Spring, Summer, Winter
        
        bars = ax4.bar(resilience_scores.index, resilience_scores.values, 
                      color=season_colors[:len(resilience_scores)])
        ax4.set_xlabel('Season')
        ax4.set_ylabel('Average Resilience Score')
        ax4.set_title('Seasonal Weather Resilience')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=1, color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'precipitation_adaptation.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    precip_impact.to_csv(os.path.join(results_dir, 'precipitation_adaptation.csv'), index=False)
    
    return precip_impact

def create_adaptation_model(df):
    """Create a simple seasonal adaptation model"""
    print("Creating seasonal adaptation model...")
    
    # Create monthly features
    monthly_features = df.groupby('month').agg({
        'start_time': 'count',
        'temperature': 'mean',
        'relative_humidity': 'mean',
        'precipitation': 'mean',
        'wind_speed': 'mean'
    }).reset_index()
    
    monthly_features.columns = ['month', 'ride_count', 'temp_mean', 'humidity_mean', 
                               'precip_mean', 'wind_mean']
    
    # Add cyclical features
    monthly_features['month_sin'] = np.sin(2 * np.pi * monthly_features['month'] / 12)
    monthly_features['month_cos'] = np.cos(2 * np.pi * monthly_features['month'] / 12)
    
    # Prepare features
    feature_cols = ['temp_mean', 'humidity_mean', 'precip_mean', 'wind_mean', 'month_sin', 'month_cos']
    X = monthly_features[feature_cols].fillna(0)
    y = monthly_features['ride_count']
    
    # Fit model
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Seasonal Adaptation Model - 2024', fontsize=16, fontweight='bold')
    
    # 1. Model performance
    ax1 = axes[0, 0]
    ax1.scatter(y, y_pred, alpha=0.7, s=100)
    ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    ax1.set_xlabel('Actual Rides')
    ax1.set_ylabel('Predicted Rides')
    ax1.set_title(f'Model Performance (R² = {r2:.4f})')
    ax1.grid(True, alpha=0.3)
    
    # 2. Feature importance
    ax2 = axes[0, 1]
    importance = np.abs(model.coef_)
    bars = ax2.bar(feature_cols, importance)
    ax2.set_title('Feature Importance')
    ax2.set_ylabel('Absolute Coefficient')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 3. Monthly predictions vs actual
    ax3 = axes[1, 0]
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    x = np.arange(len(y))
    width = 0.35
    
    ax3.bar(x - width/2, y, width, label='Actual', alpha=0.8)
    ax3.bar(x + width/2, y_pred, width, label='Predicted', alpha=0.8)
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Total Rides')
    ax3.set_title('Monthly Predictions vs Actual')
    ax3.set_xticks(x)
    ax3.set_xticklabels(month_names, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Residuals
    ax4 = axes[1, 1]
    residuals = y - y_pred
    ax4.scatter(y_pred, residuals, alpha=0.7)
    ax4.axhline(y=0, color='r', linestyle='--')
    ax4.set_xlabel('Predicted Rides')
    ax4.set_ylabel('Residuals')
    ax4.set_title('Model Residuals')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'adaptation_model.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save results
    model_results = {
        'r2_score': r2,
        'coefficients': dict(zip(feature_cols, model.coef_)),
        'intercept': model.intercept_
    }
    
    with open(os.path.join(results_dir, 'model_results.txt'), 'w') as f:
        f.write("Seasonal Adaptation Model Results\n")
        f.write("=================================\n")
        f.write(f"R² Score: {r2:.6f}\n")
        f.write(f"Intercept: {model.intercept_:.2f}\n\n")
        f.write("Feature Coefficients:\n")
        for feature, coef in zip(feature_cols, model.coef_):
            f.write(f"  {feature}: {coef:.6f}\n")
    
    return model_results

def main():
    """Main execution function"""
    print("Starting Seasonal Adaptation Patterns Analysis")
    print("=" * 60)
    
    df = load_2024_data()
    
    monthly_data, seasonal_data = analyze_seasonal_trends(df)
    temp_data = analyze_temperature_adaptation(df)
    precip_data = analyze_precipitation_adaptation(df)
    model_results = create_adaptation_model(df)
    
    print(f"\nKey Findings:")
    peak_month = monthly_data.loc[monthly_data['total_rides'].idxmax(), 'month_name']
    print(f"- Peak ridership month: {peak_month}")
    print(f"- Model R² score: {model_results['r2_score']:.4f}")
    print(f"- Temperature coefficient: {model_results['coefficients']['temp_mean']:.2f}")
    print(f"- Precipitation coefficient: {model_results['coefficients']['precip_mean']:.2f}")
    
    print("\nSeasonal Adaptation Patterns Analysis Complete!")
    print(f"Results saved to: {results_dir}")

if __name__ == "__main__":
    main() 