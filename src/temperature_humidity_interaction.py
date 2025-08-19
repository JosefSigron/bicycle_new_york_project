"""
Temperature-Humidity Interaction Effects on Ridership
====================================================

This script analyzes the complex interactions between temperature and humidity
and their combined effects on CitiBike ridership patterns.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Create results directory
results_dir = os.path.join('results', 'temperature_humidity_interaction')
os.makedirs(results_dir, exist_ok=True)

def load_2024_data():
    """Load and prepare 2024 data"""
    print("Loading 2024 CitiBike weather data...")
    
    file_path = 'data/combined/2024_combined_citibike_weather.parquet'
    df = pd.read_parquet(file_path)
    
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['datetime_hour'] = df['start_time'].dt.floor('h')
    df['month'] = df['start_time'].dt.month
    df['season'] = df['month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })
    
    print(f"Loaded {len(df):,} rides from 2024")
    return df

def create_comfort_zone_analysis(df):
    """Create temperature-humidity comfort zone analysis"""
    print("Creating temperature-humidity comfort zone analysis...")
    
    # Create bins
    temp_bins = np.arange(-20, 45, 3)
    humidity_bins = np.arange(0, 101, 10)
    
    df['temp_bin'] = pd.cut(df['temperature'], bins=temp_bins)
    df['humidity_bin'] = pd.cut(df['relative_humidity'], bins=humidity_bins)
    
    # Calculate ridership by temperature-humidity combinations
    comfort_data = df.groupby(['temp_bin', 'humidity_bin']).agg({
        'start_time': 'count',
        'trip_duration': 'mean'
    }).reset_index()
    
    comfort_data.columns = ['temperature', 'humidity', 'total_rides', 'avg_duration']
    comfort_data = comfort_data.dropna()
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Temperature-Humidity Interaction Analysis - 2024', fontsize=16)
    
    # 1. Heatmap of ridership
    ax1 = axes[0, 0]
    temp_labels = [f"{int(interval.mid)}" for interval in comfort_data['temperature']]
    humidity_labels = [f"{int(interval.mid)}" for interval in comfort_data['humidity']]
    
    # Create pivot table for heatmap
    pivot_data = comfort_data.pivot_table(values='total_rides', 
                                         index='humidity', 
                                         columns='temperature', 
                                         fill_value=0)
    
    sns.heatmap(pivot_data, cmap='YlOrRd', ax=ax1, cbar_kws={'label': 'Total Rides'})
    ax1.set_title('Total Rides by Temperature-Humidity')
    ax1.set_xlabel('Temperature (°C)')
    ax1.set_ylabel('Relative Humidity (%)')
    
    # 2. Scatter plot with temperature and humidity
    ax2 = axes[0, 1]
    scatter = ax2.scatter(df['temperature'], df['relative_humidity'], 
                         c=df.groupby(['temperature', 'relative_humidity']).cumcount(), 
                         cmap='viridis', alpha=0.5, s=1)
    ax2.set_xlabel('Temperature (°C)')
    ax2.set_ylabel('Relative Humidity (%)')
    ax2.set_title('Temperature vs Humidity Distribution')
    
    # 3. Seasonal analysis
    ax3 = axes[1, 0]
    seasonal_data = df.groupby(['season', pd.cut(df['temperature'], bins=5)])['start_time'].count().unstack()
    seasonal_data.plot(kind='bar', ax=ax3, stacked=True)
    ax3.set_title('Ridership by Season and Temperature Range')
    ax3.set_xlabel('Season')
    ax3.set_ylabel('Total Rides')
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend(title='Temperature Range', bbox_to_anchor=(1.05, 1))
    
    # 4. User type comparison
    ax4 = axes[1, 1]
    if 'user_type' in df.columns:
        user_temp_data = df.groupby(['user_type', pd.cut(df['temperature'], bins=6)])['start_time'].count().unstack(fill_value=0)
        user_temp_data.plot(kind='bar', ax=ax4)
        ax4.set_title('Temperature Preferences by User Type')
        ax4.set_xlabel('User Type')
        ax4.set_ylabel('Total Rides')
        ax4.tick_params(axis='x', rotation=45)
        ax4.legend(title='Temperature Range', bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'temperature_humidity_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return comfort_data

def perform_interaction_modeling(df):
    """Perform statistical modeling of temperature-humidity interactions"""
    print("Performing temperature-humidity interaction modeling...")
    
    # Create hourly aggregations
    hourly_data = df.groupby('datetime_hour').agg({
        'temperature': 'mean',
        'relative_humidity': 'mean',
        'start_time': 'count'
    }).reset_index()
    hourly_data.columns = ['datetime_hour', 'temperature', 'humidity', 'ride_count']
    
    # Prepare features
    X = hourly_data[['temperature', 'humidity']].values
    X_interaction = np.column_stack([
        X[:, 0],  # temperature
        X[:, 1],  # humidity
        X[:, 0] * X[:, 1],  # interaction
        X[:, 0]**2,  # temperature squared
        X[:, 1]**2   # humidity squared
    ])
    
    y = hourly_data['ride_count'].values
    
    # Fit model
    model = LinearRegression()
    model.fit(X_interaction, y)
    y_pred = model.predict(X_interaction)
    
    r2 = r2_score(y, y_pred)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Temperature-Humidity Interaction Modeling - 2024', fontsize=16)
    
    # 1. Model performance
    ax1 = axes[0, 0]
    ax1.scatter(y, y_pred, alpha=0.6)
    ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    ax1.set_xlabel('Actual Rides per Hour')
    ax1.set_ylabel('Predicted Rides per Hour')
    ax1.set_title(f'Model Performance (R² = {r2:.4f})')
    ax1.grid(True, alpha=0.3)
    
    # 2. Feature importance
    ax2 = axes[0, 1]
    feature_names = ['Temperature', 'Humidity', 'Temp×Humidity', 'Temperature²', 'Humidity²']
    importance = np.abs(model.coef_)
    
    bars = ax2.bar(feature_names, importance)
    ax2.set_title('Feature Importance')
    ax2.set_ylabel('Absolute Coefficient')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Temperature vs ridership
    ax3 = axes[1, 0]
    ax3.scatter(hourly_data['temperature'], hourly_data['ride_count'], alpha=0.6)
    ax3.set_xlabel('Temperature (°C)')
    ax3.set_ylabel('Rides per Hour')
    ax3.set_title('Temperature vs Ridership')
    ax3.grid(True, alpha=0.3)
    
    # 4. Humidity vs ridership
    ax4 = axes[1, 1]
    ax4.scatter(hourly_data['humidity'], hourly_data['ride_count'], alpha=0.6)
    ax4.set_xlabel('Relative Humidity (%)')
    ax4.set_ylabel('Rides per Hour')
    ax4.set_title('Humidity vs Ridership')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'interaction_modeling.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save results
    results = {
        'r2_score': r2,
        'coefficients': dict(zip(feature_names, model.coef_)),
        'intercept': model.intercept_
    }
    
    with open(os.path.join(results_dir, 'model_results.txt'), 'w') as f:
        f.write("Temperature-Humidity Interaction Model Results\n")
        f.write("==============================================\n")
        f.write(f"R² Score: {r2:.6f}\n")
        f.write(f"Intercept: {model.intercept_:.2f}\n\n")
        f.write("Coefficients:\n")
        for feature, coef in zip(feature_names, model.coef_):
            f.write(f"  {feature}: {coef:.6f}\n")
    
    return results

def main():
    """Main execution function"""
    print("Starting Temperature-Humidity Interaction Analysis")
    print("=" * 60)
    
    df = load_2024_data()
    comfort_data = create_comfort_zone_analysis(df)
    model_results = perform_interaction_modeling(df)
    
    print(f"\nKey Findings:")
    print(f"- Model R² score: {model_results['r2_score']:.4f}")
    print(f"- Temperature coefficient: {model_results['coefficients']['Temperature']:.4f}")
    print(f"- Humidity coefficient: {model_results['coefficients']['Humidity']:.4f}")
    print(f"- Interaction coefficient: {model_results['coefficients']['Temp×Humidity']:.4f}")
    
    print("\nTemperature-Humidity Interaction Analysis Complete!")
    print(f"Results saved to: {results_dir}")

if __name__ == "__main__":
    main() 