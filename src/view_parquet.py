import pandas as pd
import sys

def view_parquet(file_path, n_rows=5):
    """
    View the contents of a parquet file.
    
    Args:
        file_path (str): Path to the parquet file
        n_rows (int): Number of rows to display (default: 5)
    """
    try:
        # Read the parquet file
        df = pd.read_parquet(file_path)
        
        # Display basic information
        print(f"\nFile: {file_path}")
        print(f"Number of rows: {len(df)}")
        print(f"Number of columns: {len(df.columns)}")
        print("\nColumns:")
        for col in df.columns:
            print(f"- {col}")
        
        # Display the first n_rows
        print(f"\nFirst {n_rows} rows:")
        print(df.head(n_rows))
        
        # Display the last n_rows
        print(f"\nLast {n_rows} rows:")
        print(df.tail(n_rows))
    except Exception as e:
        print(f"Error reading parquet file: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python view_parquet.py <path_to_parquet_file> [number_of_rows]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    n_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    view_parquet(file_path, n_rows) 