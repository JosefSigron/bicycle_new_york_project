import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as mcolors
from matplotlib.cm import get_cmap

# Set style for better visualizations
plt.style.use('seaborn-v0_8-whitegrid')

# Set a colorblind-friendly palette
sns.set_palette('colorblind')

# Load the NYC temperature data
file_path = 'data/weather/csv/nyc_weather_with_utci.csv'
df = pd.read_csv(file_path)

# Data cleaning
# Check if there are any missing values
print(f"Missing values in each column:\n{df.isnull().sum()}")

# Drop rows with missing temperature values
df = df.dropna(subset=['temperature'])

# Convert datetime column to datetime type
df['datetime'] = pd.to_datetime(df['datetime'])

# Extract date components from datetime
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['hour'] = df['datetime'].dt.hour

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
sns.histplot(df['temperature'], bins=30, kde=True)
plt.title('NYC Temperature Distribution (2019-2024)', fontsize=14)
plt.xlabel('Temperature (°C)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

# 2. Monthly Temperature Box Plot
plt.subplot(2, 2, 2)
sns.boxplot(x='month', y='temperature', data=df)
plt.title('Monthly Temperature Distribution', fontsize=14)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

# 3. Seasonal Temperature Distribution
plt.subplot(2, 2, 3)
season_order = ['Winter', 'Spring', 'Summer', 'Fall']
sns.violinplot(x='season', y='temperature', data=df, order=season_order)
plt.title('Seasonal Temperature Distribution', fontsize=14)
plt.xlabel('Season', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)

# 4. Yearly Temperature Trends
plt.subplot(2, 2, 4)
yearly_avg = df.groupby('year')['temperature'].mean().reset_index()
sns.lineplot(x='year', y='temperature', data=yearly_avg, marker='o', linewidth=2)
plt.title('Yearly Average Temperature Trend', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Average Temperature (°C)', fontsize=12)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.tight_layout()
plt.savefig('results/weather_distribution/nyc_temp_distribution_analysis.png')

# Create additional plots for more detailed analysis
plt.figure(figsize=(20, 12))

# 5. Monthly Temperature Trends Across Years
pivot_df = df.pivot_table(index='month', columns='year', values='temperature', aggfunc='mean')
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
monthly_stats = df.groupby('month')['temperature'].agg(['min', 'max', 'mean']).reset_index()
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
sns.boxplot(x='year', y='temperature', hue='season', data=df, palette='colorblind')
plt.title('Temperature Distribution by Season and Year', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.legend(title='Season')

# 8. Daily Temperature Variation for a Sample Year (2023)
plt.subplot(2, 2, 4)
sample_year = 2023
sample_df = df[df['year'] == sample_year].copy()
sample_df['day_of_year'] = sample_df['datetime'].dt.dayofyear
sns.scatterplot(x='day_of_year', y='temperature', data=sample_df, alpha=0.6)

# Add a smoothed trend line
x = sample_df['day_of_year']
y = sample_df['temperature']
z = np.polyfit(x, y, 10)
p = np.poly1d(z)
plt.plot(x, p(x), 'r-', linewidth=2)

plt.title(f'Daily Temperature Variation in {sample_year}', fontsize=14)
plt.xlabel('Day of Year', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.grid(True)

plt.tight_layout()
plt.savefig('results/weather_distribution/nyc_temp_detailed_analysis.png')

# Create a new figure for UTCI category distribution by hour
fig, axes = plt.subplots(2, 2, figsize=(30, 24))
axes = axes.flatten()  # Flatten the 2x2 array to make indexing easier

# Create subplots for each season
seasons = ['Winter', 'Spring', 'Summer', 'Fall']
season_months = {
    'Winter': [12, 1, 2],
    'Spring': [3, 4, 5],
    'Summer': [6, 7, 8],
    'Fall': [9, 10, 11]
}

# Create a custom color map for temperature categories
# Define colors that make sense for temperature: blue for cold, green for neutral, red/yellow for hot
# Print out the unique utci_cat values to check what's actually in the data
print("Unique UTCI Categories:", sorted(df['utci_cat'].unique().tolist()))

# Ensure all UTCI categories have proper colors that match their temperature meaning
custom_colors = {
    'Extreme cold stress': '#0000FF',         # Dark blue
    'Very strong cold stress': '#4444FF',     # Blue
    'Strong cold stress': '#8888FF',          # Light blue
    'Moderate cold stress': '#AAAAFF',        # Very light blue
    'Slight cold stress': '#CCCCFF',          # Pale blue
    'No thermal stress': '#00CC00',           # Green
    'Slight heat stress': '#FFFFAA',          # Pale yellow
    'Moderate heat stress': '#FFDD55',        # Yellow
    'Strong heat stress': '#FF8800',          # Orange
    'Very strong heat stress': '#FF4400',     # Orange-red
    'Extreme heat stress': '#FF0000'          # Red
}

# Hatches for accessibility - match them to the proper category names
hatches = {
    'Extreme cold stress': '/',
    'Very strong cold stress': '\\',
    'Strong cold stress': 'x',
    'Moderate cold stress': 'o',
    'Slight cold stress': '.',
    'No thermal stress': '*',
    'Slight heat stress': '+',
    'Moderate heat stress': '-',
    'Strong heat stress': '|',
    'Very strong heat stress': 'O',
    'Extreme heat stress': '.'
}

# Find all unique UTCI categories across the entire dataset
all_categories = df['utci_cat'].unique().tolist()

# Define the correct order from coldest to hottest
temperature_order = [
    'Extreme cold stress',
    'Very strong cold stress', 
    'Strong cold stress',
    'Moderate cold stress',
    'Slight cold stress',
    'No thermal stress',
    'Slight heat stress',
    'Moderate heat stress',
    'Strong heat stress',
    'Very strong heat stress',
    'Extreme heat stress'
]

# Filter and sort the categories based on the temperature order
ordered_categories = [cat for cat in temperature_order if cat in all_categories]
print("Categories in temperature order:", ordered_categories)

for idx, season in enumerate(seasons):
    # Filter data for the current season
    season_df = df[df['month'].isin(season_months[season])]
    
    # Calculate distribution of UTCI categories by hour
    hourly_utci_dist = season_df.groupby(['hour', 'utci_cat']).size().unstack(fill_value=0)
    
    # Create stacked bar plot with wider bars on the specific subplot
    ax = axes[idx]
    
    # Get the categories present in this season and sort them by temperature
    categories = hourly_utci_dist.columns.tolist()
    categories_ordered = [cat for cat in ordered_categories if cat in categories]
    
    # For each hour and category, plot with specific color and hatch pattern
    bottom = np.zeros(len(hourly_utci_dist.index))
    
    # Plot in temperature order
    for cat in categories_ordered:
        values = hourly_utci_dist[cat].values
        # Use temperature-appropriate color and hatch
        color = custom_colors.get(cat, 'gray')  # Use gray as last resort
        hatch = hatches.get(cat, '')  # Default to no hatch if not found
        
        ax.bar(hourly_utci_dist.index, values, bottom=bottom, 
               width=0.8, label=cat if idx == 0 else "", 
               color=color, hatch=hatch)
        bottom += values
    
    ax.set_title(f'Distribution of UTCI Categories by Hour of Day - {season}', 
                fontsize=24)
    ax.set_xlabel('Hour of Day', fontsize=20)
    ax.set_ylabel('Number of Records', fontsize=20)
    ax.grid(True, axis='y')
    ax.tick_params(axis='both', labelsize=18)
    ax.set_xticks(range(24))
    ax.set_xticklabels(range(24))

# Add a single legend for the entire figure, position it below the plots
# Create a custom legend with all categories in temperature order
handles = []
labels = []

# Create dummy patches for the legend in temperature order
for cat in ordered_categories:
    if cat in all_categories:  # Only include categories that exist in the data
        color = custom_colors.get(cat, 'gray')
        hatch = hatches.get(cat, '')
        handle = plt.Rectangle((0,0), 1, 1, facecolor=color, hatch=hatch)
        handles.append(handle)
        labels.append(cat)

fig.legend(handles, labels, title='UTCI Category', 
           loc='lower center', bbox_to_anchor=(0.5, 0.02),
           fontsize=20, title_fontsize=24, ncol=len(labels))

plt.tight_layout()
# Adjust the bottom margin to make room for the legend but reduce the gap
plt.subplots_adjust(bottom=0.1)
plt.savefig('results/weather_distribution/utci_category_by_hour_seasonal.png', dpi=300, bbox_inches='tight')

print("Analysis complete. Visualization images saved.")
