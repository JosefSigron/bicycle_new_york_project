"""
UTCI Impact Modeling Analysis
============================

This script performs detailed Universal Thermal Climate Index (UTCI) impact modeling 
on CitiBike ridership data. UTCI is a comprehensive thermal comfort index that considers:
- Air temperature
- Relative humidity  
- Wind speed
- Mean radiant temperature

The analysis includes:
1. UTCI distribution analysis across different seasons
2. Ridership patterns by UTCI categories (thermal comfort zones)
3. Non-linear UTCI effects using polynomial regression
4. UTCI threshold analysis for optimal riding conditions
5. Temporal UTCI patterns and their correlation with ride volume
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import warnings
from tqdm import tqdm
import pyarrow.parquet as pq

warnings.filterwarnings('ignore')

# Create results directory
results_dir = os.path.join('results', 'utci_impact_modeling')
os.makedirs(results_dir, exist_ok=True)

# Set matplotlib parameters for publication-quality plots
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'figure.figsize': (12, 8)
})

def load_2024_data():
    """Load 2024 CitiBike weather data with memory optimization"""
    print("Loading 2024 CitiBike weather data...")
    
    # Load data in chunks to manage memory
    file_path = 'data/combined/2024_combined_citibike_weather.parquet'
    
    # Read parquet file
    df = pd.read_parquet(file_path)
    
    # Convert timestamps
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['datetime_hour'] = df['start_time'].dt.floor('h')
    
    # Add temporal features
    df['month'] = df['start_time'].dt.month
    df['day_of_year'] = df['start_time'].dt.dayofyear
    df['hour'] = df['start_time'].dt.hour
    df['weekday'] = df['start_time'].dt.weekday
    df['is_weekend'] = df['weekday'].isin([5, 6])
    
    # Add season classification
    df['season'] = df['month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })
    
    print(f"Loaded {len(df):,} rides from 2024")
    print(f"UTCI range: {df['utci'].min():.1f}°C to {df['utci'].max():.1f}°C")
    
    return df

def analyze_utci_distribution(df):
    """Analyze UTCI distribution across seasons and time periods"""
    print("Analyzing UTCI distribution patterns...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('UTCI Distribution Analysis - 2024', fontsize=20, fontweight='bold')
    
    # 1. UTCI distribution by season
    ax1 = axes[0, 0]
    for season in ['Winter', 'Spring', 'Summer', 'Fall']:
        season_data = df[df['season'] == season]['utci']
        ax1.hist(season_data, bins=50, alpha=0.6, label=season, density=True)
    
    ax1.set_xlabel('UTCI (°C)')
    ax1.set_ylabel('Density')
    ax1.set_title('UTCI Distribution by Season')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. UTCI vs Hour of Day
    ax2 = axes[0, 1]
    hourly_utci = df.groupby('hour')['utci'].agg(['mean', 'std']).reset_index()
    ax2.errorbar(hourly_utci['hour'], hourly_utci['mean'], 
                yerr=hourly_utci['std'], capsize=5, capthick=2)
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Mean UTCI (°C)')
    ax2.set_title('Daily UTCI Pattern')
    ax2.grid(True, alpha=0.3)
    
    # 3. UTCI categories distribution
    ax3 = axes[1, 0]
    utci_cat_counts = df['utci_cat'].value_counts()
    colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(utci_cat_counts)))
    bars = ax3.bar(range(len(utci_cat_counts)), utci_cat_counts.values, color=colors)
    ax3.set_xlabel('UTCI Category')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of UTCI Categories')
    ax3.set_xticks(range(len(utci_cat_counts)))
    ax3.set_xticklabels(utci_cat_counts.index, rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, utci_cat_counts.values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(utci_cat_counts.values)*0.01,
                f'{value:,}', ha='center', va='bottom', fontsize=10)
    
    # 4. Monthly UTCI trends
    ax4 = axes[1, 1]
    monthly_utci = df.groupby('month')['utci'].agg(['mean', 'min', 'max']).reset_index()
    ax4.plot(monthly_utci['month'], monthly_utci['mean'], 'o-', linewidth=2, markersize=8, label='Mean')
    ax4.fill_between(monthly_utci['month'], monthly_utci['min'], monthly_utci['max'], 
                    alpha=0.3, label='Min-Max Range')
    ax4.set_xlabel('Month')
    ax4.set_ylabel('UTCI (°C)')
    ax4.set_title('Monthly UTCI Trends')
    ax4.set_xticks(range(1, 13))
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'utci_distribution_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save distribution statistics
    distribution_stats = {
        'overall': df['utci'].describe(),
        'by_season': df.groupby('season')['utci'].describe(),
        'by_utci_category': df.groupby('utci_cat')['utci'].describe()
    }
    
    # Save to file
    with open(os.path.join(results_dir, 'utci_distribution_stats.txt'), 'w') as f:
        f.write("UTCI Distribution Statistics - 2024\n")
        f.write("=====================================\n\n")
        f.write("Overall UTCI Statistics:\n")
        f.write(str(distribution_stats['overall']) + '\n\n')
        f.write("UTCI Statistics by Season:\n")
        f.write(str(distribution_stats['by_season']) + '\n\n')
        f.write("UTCI Statistics by Category:\n")
        f.write(str(distribution_stats['by_utci_category']) + '\n')
    
    return distribution_stats

def analyze_ridership_by_utci_categories(df):
    """Analyze ridership patterns across UTCI thermal comfort categories"""
    print("Analyzing ridership by UTCI categories...")
    
    # Create hourly aggregations
    hourly_data = df.groupby(['datetime_hour', 'utci_cat', 'season']).size().reset_index(name='ride_count')
    
    # Calculate average rides per hour by UTCI category
    utci_ridership = hourly_data.groupby('utci_cat')['ride_count'].agg([
        'mean', 'median', 'std', 'count'
    ]).reset_index()
    utci_ridership = utci_ridership.sort_values('mean', ascending=False)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Ridership Analysis by UTCI Categories - 2024', fontsize=20, fontweight='bold')
    
    # 1. Average rides per hour by UTCI category
    ax1 = axes[0, 0]
    colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(utci_ridership)))
    bars = ax1.bar(range(len(utci_ridership)), utci_ridership['mean'], 
                  yerr=utci_ridership['std'], capsize=5, color=colors)
    ax1.set_xlabel('UTCI Category')
    ax1.set_ylabel('Average Rides per Hour')
    ax1.set_title('Average Hourly Ridership by UTCI Category')
    ax1.set_xticks(range(len(utci_ridership)))
    ax1.set_xticklabels(utci_ridership['utci_cat'], rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, utci_ridership['mean']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.01,
                f'{value:.1f}', ha='center', va='bottom', fontsize=10)
    
    # 2. Box plot of ridership distribution by UTCI category
    ax2 = axes[0, 1]
    utci_categories = hourly_data['utci_cat'].unique()
    ridership_by_category = [hourly_data[hourly_data['utci_cat'] == cat]['ride_count'].values 
                           for cat in utci_categories]
    
    box_plot = ax2.boxplot(ridership_by_category, labels=utci_categories, patch_artist=True)
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    ax2.set_xlabel('UTCI Category')
    ax2.set_ylabel('Rides per Hour')
    ax2.set_title('Ridership Distribution by UTCI Category')
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # 3. Seasonal ridership by UTCI category
    ax3 = axes[1, 0]
    seasonal_utci = hourly_data.groupby(['utci_cat', 'season'])['ride_count'].mean().unstack(fill_value=0)
    seasonal_utci.plot(kind='bar', ax=ax3, width=0.8)
    ax3.set_xlabel('UTCI Category')
    ax3.set_ylabel('Average Rides per Hour')
    ax3.set_title('Seasonal Ridership by UTCI Category')
    ax3.legend(title='Season')
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    # 4. Correlation heatmap
    ax4 = axes[1, 1]
    # Create a pivot table for heatmap
    heatmap_data = hourly_data.pivot_table(values='ride_count', 
                                          index='utci_cat', 
                                          columns='season', 
                                          aggfunc='mean',
                                          fill_value=0)
    
    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlBu_r', ax=ax4)
    ax4.set_title('Average Rides per Hour: UTCI vs Season')
    ax4.set_xlabel('Season')
    ax4.set_ylabel('UTCI Category')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'ridership_by_utci_categories.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save detailed results
    utci_ridership.to_csv(os.path.join(results_dir, 'ridership_by_utci_categories.csv'), index=False)
    seasonal_utci.to_csv(os.path.join(results_dir, 'seasonal_ridership_by_utci.csv'))
    
    return utci_ridership, seasonal_utci

def perform_polynomial_utci_modeling(df):
    """Perform non-linear UTCI impact modeling using polynomial regression"""
    print("Performing polynomial UTCI modeling...")
    
    # Create hourly aggregations
    hourly_data = df.groupby('datetime_hour').agg({
        'utci': 'mean',
        'start_time': 'count'  # Count rides per hour
    }).reset_index()
    hourly_data.rename(columns={'start_time': 'ride_count'}, inplace=True)
    
    # Prepare data for modeling
    X = hourly_data['utci'].values.reshape(-1, 1)
    y = hourly_data['ride_count'].values
    
    # Test different polynomial degrees
    degrees = [1, 2, 3, 4, 5]
    models = {}
    scores = {}
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Polynomial UTCI Impact Modeling - 2024', fontsize=20, fontweight='bold')
    
    for i, degree in enumerate(degrees):
        # Create polynomial features
        poly_features = PolynomialFeatures(degree=degree)
        X_poly = poly_features.fit_transform(X)
        
        # Fit model
        model = LinearRegression()
        model.fit(X_poly, y)
        
        # Predictions
        y_pred = model.predict(X_poly)
        
        # Calculate metrics
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        models[degree] = (model, poly_features)
        scores[degree] = {'R²': r2, 'RMSE': rmse}
        
        # Plot results
        if i < 5:  # Plot first 5 models
            ax = axes[i//3, i%3]
            
            # Sort data for smooth curve plotting
            sort_idx = np.argsort(X.flatten())
            ax.scatter(X[sort_idx], y[sort_idx], alpha=0.5, s=20, label='Data')
            ax.plot(X[sort_idx], y_pred[sort_idx], 'r-', linewidth=2, 
                   label=f'Degree {degree} (R²={r2:.3f})')
            
            ax.set_xlabel('UTCI (°C)')
            ax.set_ylabel('Rides per Hour')
            ax.set_title(f'Polynomial Degree {degree}')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # Model comparison plot
    ax_comparison = axes[1, 2]
    degrees_list = list(scores.keys())
    r2_scores = [scores[d]['R²'] for d in degrees_list]
    rmse_scores = [scores[d]['RMSE'] for d in degrees_list]
    
    ax2 = ax_comparison.twinx()
    line1 = ax_comparison.plot(degrees_list, r2_scores, 'bo-', linewidth=2, markersize=8, label='R²')
    line2 = ax2.plot(degrees_list, rmse_scores, 'ro-', linewidth=2, markersize=8, label='RMSE')
    
    ax_comparison.set_xlabel('Polynomial Degree')
    ax_comparison.set_ylabel('R² Score', color='b')
    ax2.set_ylabel('RMSE', color='r')
    ax_comparison.set_title('Model Performance Comparison')
    ax_comparison.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax_comparison.legend(lines, labels, loc='center right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'polynomial_utci_modeling.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Find optimal model (best R² with reasonable complexity)
    best_degree = max(scores.keys(), key=lambda x: scores[x]['R²'])
    best_model, best_poly = models[best_degree]
    
    # Save model performance results
    performance_df = pd.DataFrame(scores).T
    performance_df.to_csv(os.path.join(results_dir, 'polynomial_model_performance.csv'))
    
    # Generate predictions across UTCI range for optimal conditions analysis
    utci_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    utci_range_poly = best_poly.transform(utci_range)
    predictions = best_model.predict(utci_range_poly)
    
    # Find optimal UTCI for maximum ridership
    optimal_idx = np.argmax(predictions)
    optimal_utci = utci_range[optimal_idx, 0]
    
    return models, scores, optimal_utci, hourly_data

def main():
    """Main execution function"""
    print("Starting UTCI Impact Modeling Analysis")
    print("=" * 50)
    
    # Load data
    df = load_2024_data()
    
    # Perform analyses
    distribution_stats = analyze_utci_distribution(df)
    utci_ridership, seasonal_utci = analyze_ridership_by_utci_categories(df)
    models, scores, optimal_utci, hourly_data = perform_polynomial_utci_modeling(df)
    
    # Print key findings
    best_degree = max(scores.keys(), key=lambda x: scores[x]['R²'])
    print(f"\nKey Findings:")
    print(f"- Optimal UTCI for ridership: {optimal_utci:.1f}°C")
    print(f"- Best polynomial model: Degree {best_degree} (R² = {scores[best_degree]['R²']:.4f})")
    print(f"- Best UTCI category: {utci_ridership.iloc[0]['utci_cat']} ({utci_ridership.iloc[0]['mean']:.1f} rides/hour)")
    
    print("\nUTCI Impact Modeling Analysis Complete!")
    print(f"Results saved to: {results_dir}")
    print("=" * 50)

if __name__ == "__main__":
    main() 