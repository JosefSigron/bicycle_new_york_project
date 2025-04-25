import pandas as pd
import glob
import os
import argparse

# Common columns are: 'start_station_longitude', 'trip_duration', 'user_type',
#  'end_station_name', 'end_station_id', 'start_station_name', 'stop_time',
#  'start_station_id', 'end_station_longitude', 'end_station_latitude',
#  'start_time', 'start_station_latitude'

def combine_2019_data():
    # Get all CSV files from the 2019 directory
    csv_files = glob.glob('data/citibike/2019/*.csv')
    
    # Initialize an empty list to store DataFrames
    dfs = []
    
    # Read each CSV file and append to the list
    for file in csv_files:
        print(f"Reading {file}...")
        df = pd.read_csv(file)
        df.rename(columns={'tripduration': 'trip_duration'}, inplace=True)
        df.rename(columns={'starttime': 'start_time'}, inplace=True)
        df.rename(columns={'stoptime': 'stop_time'}, inplace=True)
        df.rename(columns={'start station id': 'start_station_id'}, inplace=True)
        df.rename(columns={'start station name': 'start_station_name'}, inplace=True)
        df.rename(columns={'start station latitude': 'start_station_latitude'}, inplace=True)
        df.rename(columns={'start station longitude': 'start_station_longitude'}, inplace=True)
        df.rename(columns={'end station id': 'end_station_id'}, inplace=True)
        df.rename(columns={'end station name': 'end_station_name'}, inplace=True)
        df.rename(columns={'end station latitude': 'end_station_latitude'}, inplace=True)
        df.rename(columns={'end station longitude': 'end_station_longitude'}, inplace=True)
        df.rename(columns={'usertype': 'user_type'}, inplace=True)
        df.drop(columns={'bikeid'}, inplace=True)
        df.drop(columns={'birth year'}, inplace=True)
        df.drop(columns={'gender'}, inplace=True)
        
        dfs.append(df)
    
    # Combine all DataFrames
    print("Combining all DataFrames...")
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Create output directory if it doesn't exist
    os.makedirs('data/citibike/combined', exist_ok=True)
    
    # Save as Parquet file
    output_path = 'data/citibike/combined/2019_citibike_data.parquet'
    print(f"Saving combined data to {output_path}...")
    combined_df.to_parquet(output_path, index=False)
    print("Done!")

def combine_2020_data():
    # Get all CSV files from the 2020 directory
    csv_files = glob.glob('data/citibike/2020/*.csv')
    
    # Initialize an empty list to store DataFrames
    dfs = []
    
    # Read each CSV file and append to the list
    for file in csv_files:
        print(f"Reading {file}...")
        # Read CSV with specific dtypes to handle mixed types
        df = pd.read_csv(file, 
                        dtype={
                            'start_station_id': str,
                            'end_station_id': str,
                            'ride_id': str,
                            'rideable_type': str,
                            'start_station_name': str,
                            'end_station_name': str
                        },
                        low_memory=False)
        
        df.rename(columns={'started_at': 'start_time'}, inplace=True)
        df.rename(columns={'ended_at': 'stop_time'}, inplace=True)
        df.rename(columns={'start_lat': 'start_station_latitude'}, inplace=True)
        df.rename(columns={'start_lng': 'start_station_longitude'}, inplace=True)
        df.rename(columns={'end_lat': 'end_station_latitude'}, inplace=True)
        df.rename(columns={'end_lng': 'end_station_longitude'}, inplace=True)
        df.rename(columns={'member_casual': 'user_type'}, inplace=True)
        df.drop(columns={'ride_id'}, inplace=True)
        df.drop(columns={'rideable_type'}, inplace=True)
        
        # Convert start_time and stop_time to datetime
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['stop_time'] = pd.to_datetime(df['stop_time'])
        
        # Calculate trip duration in seconds
        df['trip_duration'] = (df['stop_time'] - df['start_time']).dt.total_seconds()
        
        # Ensure numeric columns are float
        numeric_columns = ['start_station_latitude', 'start_station_longitude', 
                         'end_station_latitude', 'end_station_longitude']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        dfs.append(df)
    
    # Combine all DataFrames
    print("Combining all DataFrames...")
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Create output directory if it doesn't exist
    os.makedirs('data/citibike/combined', exist_ok=True)
    
    # Save as Parquet file
    output_path = 'data/citibike/combined/2020_citibike_data.parquet'
    print(f"Saving combined data to {output_path}...")
    combined_df.to_parquet(output_path, index=False)
    print("Done!")

def combine_2021_data():
    # Get all CSV files from the 2021 directory
    csv_files = glob.glob('data/citibike/2021/*.csv')
    
    # Initialize an empty list to store DataFrames
    dfs = []
    
    # Read each CSV file and append to the list
    for file in csv_files:
        print(f"Reading {file}...")
        # Read CSV with specific dtypes to handle mixed types
        df = pd.read_csv(file, 
                        dtype={
                            'start_station_id': str,
                            'end_station_id': str,
                            'ride_id': str,
                            'rideable_type': str,
                            'start_station_name': str,
                            'end_station_name': str
                        },
                        low_memory=False)
        
        df.rename(columns={'started_at': 'start_time'}, inplace=True)
        df.rename(columns={'ended_at': 'stop_time'}, inplace=True)
        df.rename(columns={'start_lat': 'start_station_latitude'}, inplace=True)
        df.rename(columns={'start_lng': 'start_station_longitude'}, inplace=True)
        df.rename(columns={'end_lat': 'end_station_latitude'}, inplace=True)
        df.rename(columns={'end_lng': 'end_station_longitude'}, inplace=True)
        df.rename(columns={'member_casual': 'user_type'}, inplace=True)
        df.drop(columns={'ride_id'}, inplace=True)
        df.drop(columns={'rideable_type'}, inplace=True)
        
        # Convert start_time and stop_time to datetime
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['stop_time'] = pd.to_datetime(df['stop_time'])
        
        # Calculate trip duration in seconds
        df['trip_duration'] = (df['stop_time'] - df['start_time']).dt.total_seconds()
        
        # Ensure numeric columns are float
        numeric_columns = ['start_station_latitude', 'start_station_longitude', 
                         'end_station_latitude', 'end_station_longitude']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        dfs.append(df)
    
    # Combine all DataFrames
    print("Combining all DataFrames...")
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Create output directory if it doesn't exist
    os.makedirs('data/citibike/combined', exist_ok=True)
    
    # Save as Parquet file
    output_path = 'data/citibike/combined/2021_citibike_data.parquet'
    print(f"Saving combined data to {output_path}...")
    combined_df.to_parquet(output_path, index=False)
    print("Done!")

def combine_2022_data():
    # Get all CSV files from the 2022 directory
    csv_files = glob.glob('data/citibike/2022/*.csv')
    
    # Initialize an empty list to store DataFrames
    dfs = []
    
    # Read each CSV file and append to the list
    for file in csv_files:
        print(f"Reading {file}...")
        # Read CSV with specific dtypes to handle mixed types
        df = pd.read_csv(file, 
                        dtype={
                            'start_station_id': str,
                            'end_station_id': str,
                            'ride_id': str,
                            'rideable_type': str,
                            'start_station_name': str,
                            'end_station_name': str
                        },
                        low_memory=False)
        
        df.rename(columns={'started_at': 'start_time'}, inplace=True)
        df.rename(columns={'ended_at': 'stop_time'}, inplace=True)
        df.rename(columns={'start_lat': 'start_station_latitude'}, inplace=True)
        df.rename(columns={'start_lng': 'start_station_longitude'}, inplace=True)
        df.rename(columns={'end_lat': 'end_station_latitude'}, inplace=True)
        df.rename(columns={'end_lng': 'end_station_longitude'}, inplace=True)
        df.rename(columns={'member_casual': 'user_type'}, inplace=True)
        df.drop(columns={'ride_id'}, inplace=True)
        df.drop(columns={'rideable_type'}, inplace=True)
        

        # Convert start_time and stop_time to datetime
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['stop_time'] = pd.to_datetime(df['stop_time'])
        
        # Calculate trip duration in seconds
        df['trip_duration'] = (df['stop_time'] - df['start_time']).dt.total_seconds()
        
        # Ensure numeric columns are float
        numeric_columns = ['start_station_latitude', 'start_station_longitude', 
                         'end_station_latitude', 'end_station_longitude']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        dfs.append(df)
    
    # Combine all DataFrames
    print("Combining all DataFrames...")
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Create output directory if it doesn't exist
    os.makedirs('data/citibike/combined', exist_ok=True)
    
    # Save as Parquet file
    output_path = 'data/citibike/combined/2022_citibike_data.parquet'
    print(f"Saving combined data to {output_path}...")
    combined_df.to_parquet(output_path, index=False)
    print("Done!")

def combine_2023_data():

    # Get all CSV files from the 2023 directory
    csv_files = glob.glob('data/citibike/2023/*.csv')
    
    # Initialize an empty list to store DataFrames
    dfs = []
    
    # Read each CSV file and append to the list
    for file in csv_files:
        print(f"Reading {file}...")
        # Read CSV with specific dtypes to handle mixed types
        df = pd.read_csv(file, 
                        dtype={
                            'start_station_id': str,
                            'end_station_id': str,
                            'ride_id': str,
                            'rideable_type': str,
                            'start_station_name': str,
                            'end_station_name': str
                        },
                        low_memory=False)
        
        df.rename(columns={'started_at': 'start_time'}, inplace=True)
        df.rename(columns={'ended_at': 'stop_time'}, inplace=True)
        df.rename(columns={'start_lat': 'start_station_latitude'}, inplace=True)
        df.rename(columns={'start_lng': 'start_station_longitude'}, inplace=True)
        df.rename(columns={'end_lat': 'end_station_latitude'}, inplace=True)
        df.rename(columns={'end_lng': 'end_station_longitude'}, inplace=True)
        df.rename(columns={'member_casual': 'user_type'}, inplace=True)
        df.drop(columns={'ride_id'}, inplace=True)
        df.drop(columns={'rideable_type'}, inplace=True)
        
        # Convert start_time and stop_time to datetime
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['stop_time'] = pd.to_datetime(df['stop_time'])
        
        # Calculate trip duration in seconds
        df['trip_duration'] = (df['stop_time'] - df['start_time']).dt.total_seconds()
        
        # Ensure numeric columns are float
        numeric_columns = ['start_station_latitude', 'start_station_longitude', 
                         'end_station_latitude', 'end_station_longitude']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        dfs.append(df)
    
    # Combine all DataFrames
    print("Combining all DataFrames...")
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Create output directory if it doesn't exist
    os.makedirs('data/citibike/combined', exist_ok=True)
    
    # Save as Parquet file
    output_path = 'data/citibike/combined/2023_citibike_data.parquet'
    print(f"Saving combined data to {output_path}...")
    combined_df.to_parquet(output_path, index=False)
    print("Done!")

def combine_2024_data():
    # Get all CSV files from the 2024 directory
    csv_files = glob.glob('data/citibike/2024/*.csv')
    
    # Initialize an empty list to store DataFrames
    dfs = []
    
    # Read each CSV file and append to the list
    for file in csv_files:
        print(f"Reading {file}...")
        # Read CSV with specific dtypes to handle mixed types
        df = pd.read_csv(file, 
                        dtype={
                            'start_station_id': str,
                            'end_station_id': str,
                            'ride_id': str,
                            'rideable_type': str,
                            'start_station_name': str,
                            'end_station_name': str
                        },
                        low_memory=False)
        
        df.rename(columns={'started_at': 'start_time'}, inplace=True)
        df.rename(columns={'ended_at': 'stop_time'}, inplace=True)
        df.rename(columns={'start_lat': 'start_station_latitude'}, inplace=True)
        df.rename(columns={'start_lng': 'start_station_longitude'}, inplace=True)
        df.rename(columns={'end_lat': 'end_station_latitude'}, inplace=True)
        df.rename(columns={'end_lng': 'end_station_longitude'}, inplace=True)
        df.rename(columns={'member_casual': 'user_type'}, inplace=True)
        df.drop(columns={'ride_id'}, inplace=True)
        df.drop(columns={'rideable_type'}, inplace=True)
        

        # Convert start_time and stop_time to datetime
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['stop_time'] = pd.to_datetime(df['stop_time'])
        
        # Calculate trip duration in seconds
        df['trip_duration'] = (df['stop_time'] - df['start_time']).dt.total_seconds()
        
        # Ensure numeric columns are float
        numeric_columns = ['start_station_latitude', 'start_station_longitude', 
                         'end_station_latitude', 'end_station_longitude']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        dfs.append(df)
    
    # Combine all DataFrames
    print("Combining all DataFrames...")
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Create output directory if it doesn't exist
    os.makedirs('data/citibike/combined', exist_ok=True)
    
    # Save as Parquet file
    output_path = 'data/citibike/combined/2024_citibike_data.parquet'
    print(f"Saving combined data to {output_path}...")
    combined_df.to_parquet(output_path, index=False)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combine Citibike data for specific years')
    parser.add_argument('--year', type=str, choices=['2019', '2020', '2021', '2022', '2023', '2024', 'all'], default='all',
                      help='Year to process (2019, 2020, 2021, 2022, 2023, 2024, or all)')
    
    args = parser.parse_args()
    
    if args.year == '2019' or args.year == 'all':
        print("Processing 2019 data...")
        combine_2019_data()
    
    if args.year == '2020' or args.year == 'all':
        print("Processing 2020 data...")
        combine_2020_data()
    
    if args.year == '2021' or args.year == 'all':
        print("Processing 2021 data...")
        combine_2021_data()

    if args.year == '2022' or args.year == 'all':
        print("Processing 2022 data...")
        combine_2022_data()

    if args.year == '2023' or args.year == 'all':
        print("Processing 2023 data...")
        combine_2023_data()

    if args.year == '2024' or args.year == 'all':
        print("Processing 2024 data...")
        combine_2024_data()


