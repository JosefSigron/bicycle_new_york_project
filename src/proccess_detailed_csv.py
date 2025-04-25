import os
import pandas as pd
import glob

def combine_csv_files():
    """
    Combines all CSV files in the data/weather/csv/detailed_weather_nyc folder
    into a single CSV file called combined_weather_nyc.csv
    """
    # Path to the directory containing CSV files
    csv_dir = 'data/weather/csv/detailed_weather_nyc'
    
    # Output file path
    output_file = 'data/weather/csv/combined_weather_nyc.csv'
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Get a list of all CSV files in the directory
    csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {csv_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files to combine")
    
    # Initialize an empty list to store dataframes
    all_dfs = []
    
    # Read each CSV file and append to the list
    for file in csv_files:
        print(f"Processing {os.path.basename(file)}...")
        df = pd.read_csv(file)
        all_dfs.append(df)
    
    # Combine all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Save the combined dataframe to a new CSV file
    combined_df.to_csv(output_file, index=False)
    
    print(f"Combined data saved to {output_file}")
    print(f"Total rows: {len(combined_df)}")

if __name__ == "__main__":
    combine_csv_files() 