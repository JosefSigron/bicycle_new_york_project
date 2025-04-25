#!/usr/bin/env python3

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import os

# Example usage:
#   from read_dly import DLYReader
   
#   reader = DLYReader("path/to/your/file.dly")
#   df = reader.read()
#   Work with the data in df

# USW00094728.dly is the file for New York City Central Park


class DLYReader:
    """
    A class to read and parse DLY (daily) format files commonly used in climate data.
    Specifically handles GHCN (Global Historical Climatology Network) daily data format.
    """
    
    def __init__(self, file_path: str):
        """
        Initialize the DLY reader with a file path.
        
        Args:
            file_path (str): Path to the .dly file
        """
        self.file_path = file_path
        self.data = None
        
    def _parse_element(self, line: str, year: int, month: int) -> Dict:
        """
        Parse a single element line from the DLY file.
        
        Args:
            line (str): The line to parse
            year (int): The year of the data
            month (int): The month of the data
            
        Returns:
            dict: Dictionary containing the parsed data
        """
        station_id = line[0:11]
        element = line[15:19]
        
        # Parse daily values (each value is 8 characters)
        days_in_month = 31  # DLY format always includes 31 days
        daily_data = []
        
        for day in range(days_in_month):
            start_pos = 21 + (day * 8)
            value_str = line[start_pos:start_pos + 5]
            quality_flag = line[start_pos + 5:start_pos + 6]
            
            try:
                value = int(value_str)
                # Convert missing value indicators to NaN
                if value == -9999:
                    value = np.nan
            except ValueError:
                value = np.nan
                
            daily_data.append({
                'station_id': station_id,
                'element': element,
                'year': year,
                'month': month,
                'day': day + 1,
                'value': value,
                'quality_flag': quality_flag
            })
            
        return daily_data
    
    def read(self) -> pd.DataFrame:
        """
        Read and parse the DLY file into a pandas DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame containing the parsed climate data
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The file {self.file_path} does not exist")
            
        all_data = []
        
        with open(self.file_path, 'r') as f:
            for line in f:
                # Extract year and month from the line
                try:
                    year = int(line[11:15])
                    month = int(line[15:17])
                    
                    # Parse the element data
                    daily_data = self._parse_element(line, year, month)
                    all_data.extend(daily_data)
                except (ValueError, IndexError) as e:
                    print(f"Warning: Skipping malformed line: {line.strip()}")
                    continue
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        # Set multi-index for easier data access
        df.set_index(['station_id', 'element', 'year', 'month', 'day'], inplace=True)
        
        self.data = df
        return df

def main():
    """
    Example usage of the DLYReader class.
    """
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python read_dly.py <path_to_dly_file>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    
    try:
        reader = DLYReader(file_path)
        df = reader.read()
        print("\nLast few rows of the parsed data:")
        print(df.tail())
        
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 