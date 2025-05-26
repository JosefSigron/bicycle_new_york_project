import os
import pandas as pd
import glob

def combine_csv_files():
    """
    Combines all CSV files from each NYC borough (Manhattan, Bronx, Brooklyn, Queens)
    in the data/weather/csv/detailed_weather_nyc/ folder into separate CSV files 
    for each borough (e.g., combined_weather_manhattan.csv, combined_weather_bronx.csv, etc.)
    """
    # Base path to the directory containing borough folders
    base_dir = 'data/weather/csv/detailed_weather_nyc'
    
    # List of borough folders to process
    boroughs = ['manhatten', 'bronx', 'brooklyn', 'queens']
    
    # Output directory
    output_dir = 'data/weather/csv'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    total_files_processed = 0
    successful_boroughs = []
    
    # Process each borough separately
    for borough in boroughs:
        csv_dir = os.path.join(base_dir, borough)
        
        # Get a list of all CSV files in the current borough directory
        csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))
        
        if not csv_files:
            print(f"No CSV files found in {csv_dir}")
            continue
        
        print(f"\nProcessing {borough.upper()}:")
        print(f"Found {len(csv_files)} CSV files")
        
        # Initialize list for this borough's dataframes
        borough_dfs = []
        
        # Read each CSV file for this borough
        for file in csv_files:
            print(f"  Processing {os.path.basename(file)}...")
            # Use low_memory=False to avoid DtypeWarning messages
            df = pd.read_csv(file, low_memory=False)
            borough_dfs.append(df)
        
        # Combine all dataframes for this borough
        combined_df = pd.concat(borough_dfs, ignore_index=True)
        
        # Create output filename for this borough
        # Handle the manhattan spelling variation
        borough_name = 'manhattan' if borough == 'manhatten' else borough
        output_file = os.path.join(output_dir, f'combined_weather_{borough_name}.csv')
        
        # Save the combined dataframe for this borough
        combined_df.to_csv(output_file, index=False)
        
        print(f"  Combined data saved to {output_file}")
        print(f"  Total rows for {borough}: {len(combined_df)}")
        
        total_files_processed += len(csv_files)
        successful_boroughs.append(borough_name)
    
    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY:")
    print(f"Total files processed: {total_files_processed}")
    print(f"Boroughs processed: {len(successful_boroughs)}")
    print(f"Output files created:")
    for borough in successful_boroughs:
        print(f"  - combined_weather_{borough}.csv")

if __name__ == "__main__":
    combine_csv_files() 