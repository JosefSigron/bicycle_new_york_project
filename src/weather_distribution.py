import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import MaxNLocator

# Set style for better visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

# Load the NYC temperature data
file_path = 'data/weather/nyc_temp_2019_2024.csv'
df = pd.read_csv(file_path)

# Data cleaning
# Check if there are any missing values
print(f"Missing values in each column:\n{df.isnull().sum()}")

# Drop rows with missing temperature values
df = df.dropna(subset=['temp'])

# Convert date components to a datetime column
df['date'] = pd.to_datetime(df[['year', 'month', 'day']])

# Create a season column
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

df['season'] = df['month'].apply(get_season)

# Create a figure for subplots
plt.figure(figsize=(20, 16))

# 1. Temperature Distribution Histogram
plt.subplot(2, 2, 1)
sns.histplot(df['temp'], bins=30, kde=True)
plt.title('NYC Temperature Distribution (2019-2024)', fontsize=14)
plt.xlabel('Temperature (°C)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

# 2. Monthly Temperature Box Plot
plt.subplot(2, 2, 2)
sns.boxplot(x='month', y='temp', data=df)
plt.title('Monthly Temperature Distribution', fontsize=14)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

# 3. Seasonal Temperature Distribution
plt.subplot(2, 2, 3)
season_order = ['Winter', 'Spring', 'Summer', 'Fall']
sns.violinplot(x='season', y='temp', data=df, order=season_order)
plt.title('Seasonal Temperature Distribution', fontsize=14)
plt.xlabel('Season', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)

# 4. Yearly Temperature Trends
plt.subplot(2, 2, 4)
yearly_avg = df.groupby('year')['temp'].mean().reset_index()
sns.lineplot(x='year', y='temp', data=yearly_avg, marker='o', linewidth=2)
plt.title('Yearly Average Temperature Trend', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Average Temperature (°C)', fontsize=12)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.tight_layout()
plt.savefig('results/weather_distribution/nyc_temp_distribution_analysis.png')

# Create additional plots for more detailed analysis
plt.figure(figsize=(20, 12))

# 5. Monthly Temperature Trends Across Years
pivot_df = df.pivot_table(index='month', columns='year', values='temp', aggfunc='mean')
plt.subplot(2, 2, 1)
sns.heatmap(pivot_df, cmap='coolwarm', annot=True, fmt='.1f', linewidths=.5)
plt.title('Monthly Average Temperatures by Year', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Month', fontsize=12)
# Fix month labels alignment by using position + 0.5 to center labels
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
plt.yticks([i + 0.5 for i in range(12)], month_labels)

# 6. Temperature Range by Month
plt.subplot(2, 2, 2)
monthly_stats = df.groupby('month')['temp'].agg(['min', 'max', 'mean']).reset_index()
plt.fill_between(monthly_stats['month'], monthly_stats['min'], monthly_stats['max'], alpha=0.3)
plt.plot(monthly_stats['month'], monthly_stats['mean'], 'o-', linewidth=2)
plt.title('Monthly Temperature Range', fontsize=14)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(True)

# 7. Temperature Distribution by Season and Year
plt.subplot(2, 2, 3)
sns.boxplot(x='year', y='temp', hue='season', data=df, palette='viridis')
plt.title('Temperature Distribution by Season and Year', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.legend(title='Season')

# 8. Daily Temperature Variation for a Sample Year (2023)
plt.subplot(2, 2, 4)
sample_year = 2023
sample_df = df[df['year'] == sample_year].copy()
sample_df['day_of_year'] = sample_df['date'].dt.dayofyear
sns.scatterplot(x='day_of_year', y='temp', data=sample_df, alpha=0.6)

# Add a smoothed trend line
x = sample_df['day_of_year']
y = sample_df['temp']
z = np.polyfit(x, y, 10)
p = np.poly1d(z)
plt.plot(x, p(x), 'r-', linewidth=2)

plt.title(f'Daily Temperature Variation in {sample_year}', fontsize=14)
plt.xlabel('Day of Year', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.grid(True)

plt.tight_layout()
plt.savefig('results/weather_distribution/nyc_temp_detailed_analysis.png')

print("Analysis complete. Visualization images saved.")
