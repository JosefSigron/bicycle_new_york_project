"""
Weekly Ride Time Analysis
Analyzes median ride times by day of week across different UTCI and weather categories
Includes time range analysis and statistical significance testing
Uses chunked processing to handle large datasets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import kruskal
import pyarrow.parquet as pq
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend for non-interactive use
import matplotlib
matplotlib.use('Agg')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class WeeklyRideTimeAnalyzer:
    def __init__(self, data_path="data/combined", chunk_size=100000):
        self.data_path = Path(data_path)
        self.chunk_size = chunk_size
        self.aggregated_stats = {}
        self.raw_samples = {}  # Store samples for significance testing
        self.results = {}
        
    def process_chunk(self, chunk_df):
        """Process a single chunk of data and return aggregated statistics"""
        
        # Convert datetime and extract features
        chunk_df['start_time'] = pd.to_datetime(chunk_df['start_time'])
        chunk_df['day_name'] = chunk_df['start_time'].dt.day_name()
        chunk_df['hour'] = chunk_df['start_time'].dt.hour
        
        # Convert trip duration to minutes
        chunk_df['trip_duration_minutes'] = chunk_df['trip_duration'] / 60
        
        # Remove outliers
        chunk_df = chunk_df[
            (chunk_df['trip_duration_minutes'] > 0) & 
            (chunk_df['trip_duration_minutes'] <= 180)
        ]
        
        if len(chunk_df) == 0:
            return {}
        
        # Define time ranges
        chunk_df['time_range'] = 'Other'
        chunk_df.loc[(chunk_df['hour'] >= 6) & (chunk_df['hour'] <= 9), 'time_range'] = '6-9 AM'
        chunk_df.loc[(chunk_df['hour'] >= 10) & (chunk_df['hour'] <= 14), 'time_range'] = '10-14 PM'
        chunk_df.loc[(chunk_df['hour'] >= 15) & (chunk_df['hour'] <= 18), 'time_range'] = '15-18 PM'
        
        # Process different categories and time ranges
        categories = ['utci_cat', 'weather_cat']
        time_ranges = [None, '6-9 AM', '10-14 PM', '15-18 PM']
        
        chunk_results = {}
        
        for category in categories:
            if category not in chunk_df.columns:
                continue
                
            for time_range in time_ranges:
                time_label = "All Day" if time_range is None else time_range
                key = f"{category}_{time_label}"
                
                # Filter data
                if time_range:
                    data = chunk_df[chunk_df['time_range'] == time_range].copy()
                else:
                    data = chunk_df.copy()
                
                if len(data) == 0:
                    continue
                
                # Calculate statistics for this chunk
                grouped = data.groupby(['day_name', category])['trip_duration_minutes']
                
                # Get detailed statistics
                chunk_stats = []
                raw_data_sample = {}
                
                for (day, cat), group in grouped:
                    durations = group.values
                    if len(durations) > 0:
                        chunk_stats.append({
                            'day_name': day,
                            'category': cat,
                            'count': len(durations),
                            'sum': durations.sum(),
                            'sum_squared': (durations ** 2).sum(),
                            'median': np.median(durations),
                            'min': durations.min(),
                            'max': durations.max()
                        })
                        
                        # Sample for significance testing (limit to avoid memory issues)
                        if day not in raw_data_sample:
                            raw_data_sample[day] = {}
                        if cat not in raw_data_sample[day]:
                            raw_data_sample[day][cat] = []
                        
                        # Take a sample if too large
                        sample_size = min(1000, len(durations))
                        if len(durations) > sample_size:
                            sample = np.random.choice(durations, size=sample_size, replace=False)
                        else:
                            sample = durations
                        raw_data_sample[day][cat].extend(sample.tolist())
                
                if chunk_stats:
                    chunk_results[key] = {
                        'stats': pd.DataFrame(chunk_stats),
                        'raw_sample': raw_data_sample
                    }
        
        return chunk_results
    
    def read_parquet_in_chunks(self, file_path):
        """Read a parquet file in chunks using pyarrow"""
        print(f"Processing {file_path.name} in chunks of {self.chunk_size:,} rows...")
        
        # Open parquet file
        parquet_file = pq.ParquetFile(file_path)
        
        chunk_count = 0
        total_processed = 0
        
        # Process in batches
        for batch in parquet_file.iter_batches(batch_size=self.chunk_size):
            chunk_df = batch.to_pandas()
            chunk_count += 1
            total_processed += len(chunk_df)
            
            if chunk_count % 10 == 0:  # Progress update every 10 chunks
                print(f"  Processed {chunk_count} chunks ({total_processed:,} rows)...")
            
            # Process this chunk
            chunk_results = self.process_chunk(chunk_df)
            
            # Aggregate results
            self.aggregate_chunk_results(chunk_results)
            
            # Clear memory
            del chunk_df
        
        print(f"  Completed {file_path.name}: {chunk_count} chunks, {total_processed:,} total rows")
    
    def aggregate_chunk_results(self, chunk_results):
        """Aggregate results from a single chunk into overall statistics"""
        
        for key, data in chunk_results.items():
            if key not in self.aggregated_stats:
                self.aggregated_stats[key] = []
                self.raw_samples[key] = {}
            
            # Store chunk statistics
            self.aggregated_stats[key].append(data['stats'])
            
            # Aggregate raw samples for significance testing
            for day, day_data in data['raw_sample'].items():
                if day not in self.raw_samples[key]:
                    self.raw_samples[key][day] = {}
                
                for cat, cat_data in day_data.items():
                    if cat not in self.raw_samples[key][day]:
                        self.raw_samples[key][day][cat] = []
                    
                    # Limit sample size to prevent memory issues
                    current_size = len(self.raw_samples[key][day][cat])
                    max_sample_size = 5000
                    
                    if current_size < max_sample_size:
                        remaining_space = max_sample_size - current_size
                        if len(cat_data) <= remaining_space:
                            self.raw_samples[key][day][cat].extend(cat_data)
                        else:
                            # Random sample to fit remaining space
                            sample = np.random.choice(cat_data, size=remaining_space, replace=False)
                            self.raw_samples[key][day][cat].extend(sample.tolist())
    
    def finalize_statistics(self):
        """Combine all chunk statistics into final results"""
        print("Finalizing statistics from all chunks...")
        
        self.final_results = {}
        
        for key, stats_list in self.aggregated_stats.items():
            if not stats_list:
                continue
            
            # Combine all chunk statistics
            all_stats = pd.concat(stats_list, ignore_index=True)
            
            # Group by day and category to get final statistics
            grouped = all_stats.groupby(['day_name', 'category'])
            
            final_stats = []
            for (day, cat), group in grouped:
                total_count = group['count'].sum()
                total_sum = group['sum'].sum()
                total_sum_squared = group['sum_squared'].sum()
                
                # Calculate overall statistics
                mean_duration = total_sum / total_count if total_count > 0 else 0
                variance = (total_sum_squared / total_count - mean_duration ** 2) if total_count > 1 else 0
                std_duration = np.sqrt(max(0, variance))
                
                # For median, we'll use the median of chunk medians weighted by count
                weighted_median = np.average(group['median'], weights=group['count'])
                
                final_stats.append({
                    'day_name': day,
                    'category': cat,
                    'count': total_count,
                    'median_duration': weighted_median,
                    'mean_duration': mean_duration,
                    'std_duration': std_duration
                })
            
            self.final_results[key] = {
                'stats': pd.DataFrame(final_stats),
                'raw_data': self.raw_samples[key]
            }
    
    def load_and_process_all_files(self):
        """Load and process all parquet files in chunks"""
        print("Starting chunked processing of parquet files...")
        
        parquet_files = list(self.data_path.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {self.data_path}")
        
        print(f"Found {len(parquet_files)} files to process")
        
        # Process each file in chunks
        for file_path in parquet_files:
            try:
                self.read_parquet_in_chunks(file_path)
            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")
                continue
        
        # Finalize all statistics
        self.finalize_statistics()
        
        print(f"Completed processing all files. Categories found: {list(self.final_results.keys())}")
    
    def calculate_median_by_category(self, key):
        """Get median results for a specific category/time range combination"""
        if key not in self.final_results:
            return None, None, None
        
        data = self.final_results[key]
        stats_df = data['stats']
        
        if len(stats_df) == 0:
            return None, None, None
        
        # Create pivot tables
        pivot_result = stats_df.pivot(index='day_name', columns='category', values='median_duration')
        count_result = stats_df.pivot(index='day_name', columns='category', values='count')
        
        # Reorder days correctly
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot_result = pivot_result.reindex(day_order)
        count_result = count_result.reindex(day_order)
        
        return pivot_result, count_result, stats_df
    
    def perform_significance_testing(self, key):
        """Perform statistical significance testing for a specific category/time range combination"""
        if key not in self.final_results:
            return {}
        
        raw_data = self.final_results[key]['raw_data']
        significance_results = {}
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        for day in day_order:
            if day not in raw_data:
                continue
            
            day_data = raw_data[day]
            category_data = []
            
            for cat, cat_data in day_data.items():
                if len(cat_data) > 10:  # Minimum sample size
                    category_data.append(cat_data)
            
            # Perform Kruskal-Wallis test
            if len(category_data) >= 2:
                try:
                    statistic, p_value = kruskal(*category_data)
                    significance_results[day] = {
                        'statistic': statistic,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
                except Exception as e:
                    print(f"    Warning: Could not perform significance test for {day}: {e}")
                    significance_results[day] = {
                        'statistic': np.nan,
                        'p_value': np.nan,
                        'significant': False
                    }
        
        return significance_results
    
    def create_plot_with_table(self, pivot_result, count_result, significance_results, 
                              category_col, time_range_filter=None, figsize=None):
        """Create a plot with significance marking and data table"""
        
        # Adjust figure size based on number of categories
        num_categories = len(pivot_result.columns)
        if figsize is None:
            if num_categories > 6:  # UTCI has 9 categories
                figsize = (18, 12)
            else:  # Weather has 6 categories
                figsize = (14, 10)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                      gridspec_kw={'height_ratios': [3, 1]})
        
        # Main plot
        for category in pivot_result.columns:
            if not pivot_result[category].isna().all():
                line = ax1.plot(pivot_result.index, pivot_result[category], 
                               marker='o', linewidth=2.5, markersize=7, label=category)
                
                # Mark significant days with stars
                for i, day in enumerate(pivot_result.index):
                    if day in significance_results and significance_results[day]['significant']:
                        ax1.plot(i, pivot_result.loc[day, category], marker='*', 
                               markersize=16, color=line[0].get_color(), 
                               markeredgecolor='red', markeredgewidth=2)
        
        # Formatting
        ax1.set_ylabel('Median Trip Duration (minutes)', fontsize=13)
        ax1.set_title(f'Median Ride Time by Day of Week - {category_col}' + 
                     (f' ({time_range_filter})' if time_range_filter else ''), 
                     fontsize=15, fontweight='bold', pad=20)
        ax1.legend(title=category_col, bbox_to_anchor=(1.05, 1), loc='upper left', 
                  fontsize=10, title_fontsize=11)
        
        # Improved grid
        ax1.grid(True, alpha=0.7, linewidth=0.8, linestyle='-', color='gray')
        ax1.set_axisbelow(True)  # Put grid behind the data
        
        ax1.tick_params(axis='x', rotation=45, labelsize=11)
        ax1.tick_params(axis='y', labelsize=11)
        
        # Create summary table
        table_data = []
        for day in pivot_result.index:
            row = [day]
            for category in pivot_result.columns:
                value = pivot_result.loc[day, category]
                count = count_result.loc[day, category] if day in count_result.index else 0
                
                if pd.notna(value) and count > 0:
                    sig_marker = "*" if (day in significance_results and 
                                       significance_results[day]['significant']) else ""
                    row.append(f"{value:.1f}{sig_marker}\n(n={count:,})")
                else:
                    row.append("N/A")
            
            # Add p-value
            if day in significance_results:
                p_val = significance_results[day]['p_value']
                if pd.notna(p_val):
                    row.append(f"{p_val:.3f}")
                else:
                    row.append("N/A")
            else:
                row.append("N/A")
            
            table_data.append(row)
        
        # Table headers
        headers = ['Day'] + list(pivot_result.columns) + ['P-value']
        
        # Create table with adjusted font size based on number of categories
        if num_categories > 6:  # UTCI categories
            font_size = 7
            cell_height = 1.8
            table_scale = (1.2, cell_height)
        else:  # Weather categories
            font_size = 9
            cell_height = 2.0
            table_scale = (1, cell_height)
        
        table = ax2.table(cellText=table_data, colLabels=headers,
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(font_size)
        table.scale(*table_scale)
        
        # Style the table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white', fontsize=font_size)
        
        # Adjust cell heights for better readability
        for i in range(len(table_data) + 1):  # +1 for header
            for j in range(len(headers)):
                cell = table[(i, j)]
                cell.set_height(0.15 if num_categories > 6 else 0.12)
        
        ax2.axis('off')
        
        plt.tight_layout()
        return fig
    
    def analyze_all_categories(self):
        """Perform complete analysis for all categories and time ranges"""
        
        categories = ['utci_cat', 'weather_cat']
        time_ranges = ['All Day', '6-9 AM', '10-14 PM', '15-18 PM']
        
        for category in categories:
            print(f"\nAnalyzing {category}...")
            
            for time_range in time_ranges:
                print(f"  Time range: {time_range}")
                key = f"{category}_{time_range}"
                
                if key not in self.final_results:
                    print(f"    No data available for {key}")
                    continue
                
                # Calculate medians
                pivot_result, count_result, raw_result = self.calculate_median_by_category(key)
                
                if pivot_result is None or len(pivot_result.columns) == 0:
                    print(f"    No results for {key}")
                    continue
                
                # Perform significance testing
                significance_results = self.perform_significance_testing(key)
                
                # Create and save plot
                time_filter = None if time_range == 'All Day' else time_range
                fig = self.create_plot_with_table(
                    pivot_result, count_result, significance_results, 
                    category, time_filter)
                
                # Save plot
                filename = f"{category}_{time_range.replace(' ', '_').replace('-', '_')}_analysis.png"
                filepath = Path("results") / "weekly_ridetime_analysis" / filename
                filepath.parent.mkdir(parents=True, exist_ok=True)
                
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"    Saved plot: {filepath}")
                
                # Store results
                self.results[key] = {
                    'pivot_result': pivot_result,
                    'count_result': count_result,
                    'significance_results': significance_results,
                    'raw_result': raw_result
                }
                
                plt.close()
    
    def generate_summary_report(self):
        """Generate a summary report of all findings"""
        
        print("\n" + "="*80)
        print("SUMMARY REPORT")
        print("="*80)
        
        for analysis_name, results in self.results.items():
            print(f"\n{analysis_name.upper()}")
            print("-" * len(analysis_name))
            
            pivot_result = results['pivot_result']
            significance_results = results['significance_results']
            
            # Overall statistics
            print(f"Categories analyzed: {list(pivot_result.columns)}")
            print(f"Days with significant differences: {sum(1 for day, result in significance_results.items() if result['significant'])}/7")
            
            # Significant days
            sig_days = [day for day, result in significance_results.items() if result['significant']]
            if sig_days:
                print(f"Significant days: {', '.join(sig_days)}")
            else:
                print("No statistically significant differences found")
            
            # Range of median durations
            all_values = pivot_result.values.flatten()
            all_values = all_values[~pd.isna(all_values)]
            if len(all_values) > 0:
                print(f"Median duration range: {np.min(all_values):.1f} - {np.max(all_values):.1f} minutes")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting Weekly Ride Time Analysis with Chunked Processing...")
        print(f"Chunk size: {self.chunk_size:,} rows")
        
        # Load and process data in chunks
        self.load_and_process_all_files()
        
        # Analyze all categories and time ranges
        self.analyze_all_categories()
        
        # Generate summary report
        self.generate_summary_report()
        
        print(f"\nAnalysis complete! Results saved in 'results' folder.")
        print(f"Total plots generated: {len(self.results)}")

def main():
    """Main function to run the analysis"""
    # Use smaller chunk size for memory efficiency
    analyzer = WeeklyRideTimeAnalyzer(chunk_size=50000)
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main() 