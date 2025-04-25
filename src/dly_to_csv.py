#!/usr/bin/env python3

import pandas as pd
import argparse
import os
from read_dly import DLYReader

def convert_dly_to_csv(input_file, output_file, start_year=2019, end_year=2024):
    """
    Convert a DLY file to CSV format, filtering for a specific range of years.
    
    Args:
        input_file (str): Path to the input DLY file
        output_file (str): Path to the output CSV file
        start_year (int): Start year for filtering (inclusive)
        end_year (int): End year for filtering (inclusive)
    """
    # Read the DLY file
    reader = DLYReader(input_file)
    df = reader.read()
    
    # Reset index to make all columns available
    df = df.reset_index()
    
    # Filter for the specified years
    filtered_df = df[(df['year'] >= start_year) & (df['year'] <= end_year)]
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Write to CSV
    filtered_df.to_csv(output_file, index=False)
    
    return filtered_df

def main():
    """
    Parse command line arguments and convert DLY to CSV.
    """
    parser = argparse.ArgumentParser(description='Convert DLY file to CSV with year filtering.')
    parser.add_argument('input_file', type=str, help='Path to the input DLY file')
    parser.add_argument('output_file', type=str, help='Path to the output CSV file')
    parser.add_argument('--start_year', type=int, default=2019, help='Start year for filtering (inclusive)')
    parser.add_argument('--end_year', type=int, default=2024, help='End year for filtering (inclusive)')
    
    args = parser.parse_args()
    
    try:
        df = convert_dly_to_csv(args.input_file, args.output_file, args.start_year, args.end_year)
        print(f"Converted {args.input_file} to {args.output_file}")
        print(f"Filtered data from {args.start_year} to {args.end_year}")
        print(f"Total rows: {len(df)}")
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    main() 