import pandas as pd
import numpy as np
from math import pi
import pythermalcomfort

def estimate_solar_radiation(df):
    """
    Estimate solar radiation using Hargreaves-Samani method
    Requires temperature data
    """
    # Extract latitude for solar calculations
    lat = df['LATITUDE'].iloc[0]
    
    # Convert date to datetime
    df['datetime'] = pd.to_datetime(df['DATE'])
    
    # Extract day of year for solar calculations
    df['day_of_year'] = df['datetime'].dt.dayofyear
    
    # Calculate extraterrestrial radiation
    # Convert latitude to radians
    lat_rad = lat * pi / 180.0
    
    # Calculate solar declination for each day
    df['solar_declination'] = 0.409 * np.sin(2 * pi * df['day_of_year'] / 365 - 1.39)
    
    # Calculate sunset hour angle
    df['sunset_hour_angle'] = np.arccos(-np.tan(lat_rad) * np.tan(df['solar_declination']))
    
    # Calculate extraterrestrial radiation
    dr = 1 + 0.033 * np.cos(2 * pi * df['day_of_year'] / 365)
    df['Ra'] = (24 * 60 / pi) * 0.0820 * dr * (
        df['sunset_hour_angle'] * np.sin(lat_rad) * np.sin(df['solar_declination']) +
        np.cos(lat_rad) * np.cos(df['solar_declination']) * np.sin(df['sunset_hour_angle'])
    )
    
    # Basic Hargreaves-Samani method
    # Need to calculate temperature difference
    df_daily = df.groupby(df['datetime'].dt.date).agg({
        'temperature': ['max', 'min']
    })
    
    df_daily.columns = ['temp_max', 'temp_min']
    df_daily = df_daily.reset_index()
    df_daily['temp_diff'] = df_daily['temp_max'] - df_daily['temp_min']
    
    # Create a date string column to use for merging
    df['date_str'] = df['datetime'].dt.date.astype(str)
    df_daily['date_str'] = df_daily['datetime'].astype(str)
    
    # Join back to original dataframe
    df = pd.merge(df, df_daily[['date_str', 'temp_diff']], 
                 on='date_str', 
                 how='left')
    
    # Apply Hargreaves-Samani equation
    # Rs = KT * Ra * sqrt(Tmax - Tmin)
    # KT is a coefficient (0.16 for inland areas, 0.19 for coastal)
    KT = 0.17  # Assuming NYC is coastal but slightly inland
    df['solar_radiation_estimated'] = KT * df['Ra'] * np.sqrt(df['temp_diff'])
    
    return df

def calculate_utci(temperature, wind_speed, humidity, mrt):
    """
    Calculate UTCI using the pythermalcomfort library
    
    Parameters:
    temperature: Air temperature in °C
    wind_speed: Wind speed in m/s
    humidity: Relative humidity in %
    mrt: Mean radiant temperature in °C
    
    Returns:
    UTCI value in °C
    """
    from pythermalcomfort.models import utci
    
    # Calculate UTCI using the pythermalcomfort library
    utci_value = utci(tdb=temperature, tr=mrt, v=wind_speed, rh=humidity)
    
    return utci_value

def categorize_utci(utci_value):
    """
    Categorize UTCI values according to thermal stress levels
    
    Parameters:
    utci_value: UTCI value in °C
    
    Returns:
    Category string
    """
    if utci_value < -40:
        return "Extreme cold stress"
    elif utci_value < -27:
        return "Very strong cold stress"
    elif utci_value < -13:
        return "Strong cold stress"
    elif utci_value < 0:
        return "Moderate cold stress"
    elif utci_value < 9:
        return "Slight cold stress"
    elif utci_value < 26:
        return "No thermal stress"
    elif utci_value < 32:
        return "Moderate heat stress"
    elif utci_value < 38:
        return "Strong heat stress"
    elif utci_value < 46:
        return "Very strong heat stress"
    else:
        return "Extreme heat stress"

def categorize_weather(utci, rain, snow):
    """
    Categorize weather based on utci, rain and snow
    
    Parameters:
    utci: UTCI value in °C
    rain: Rain value (0 or 1)
    snow: Snow value (0 or 1)
    
    Returns:
    Weather category string
    """
    if snow == 1:
        return "Snow"
    elif rain == 1:
        return "Rain"
    elif utci < 9:
        return "Cold"
    elif utci > 26:
        return "Heat"
    else:
        return "Neutral"
    


def fill_missing_with_3month_avg(df, columns):
    """
    Fill missing values with a 3-month average (90 day window).
    
    Parameters:
    df: DataFrame with datetime column
    columns: List of columns to fill
    
    Returns:
    DataFrame with filled values
    """
    print(f"Filling missing values for {len(columns)} columns with 3-month averages...")
    
    # Make sure there's a datetime column
    if 'datetime' not in df.columns:
        df['datetime'] = pd.to_datetime(df['DATE'])
    
    # For each column that needs filling
    for column in columns:
        # Check if column has missing values
        missing_count = df[column].isna().sum()
        if missing_count > 0:
            print(f"  - {column}: {missing_count} missing values")
            
            # Create a copy of the dataframe with the column and datetime
            temp_df = df[['datetime', column]].copy()
            
            # For each row with a missing value
            for idx in df[df[column].isna()].index:
                # Get the date for this row
                current_date = df.loc[idx, 'datetime']
                
                # Calculate the 90-day window before and after (180 days total)
                start_date = current_date - pd.Timedelta(days=90)
                end_date = current_date + pd.Timedelta(days=90)
                
                # Get values in that date range (excluding missing values)
                window_values = df[(df['datetime'] >= start_date) & 
                                 (df['datetime'] <= end_date)][column].dropna()
                
                # If we have values in the window, use their mean
                if len(window_values) > 0:
                    df.loc[idx, column] = window_values.mean()
                else:
                    # If no values in window, use global mean
                    df.loc[idx, column] = df[column].mean()
    
    return df

def detect_rain_and_snow(df):
    """
    Add rain and snow columns based on present weather codes
    
    Rain: 
    - If any of pres_wx_AW, pres_wx_MW contains codes 23, 25, 50-69, 80-84, 91-99
    - If any of pres_wx_AU contains codes 01, 02
    
    Snow: 
    - If any of pres_wx_AW, pres_wx_MW contains codes 24, 27-29, 68-79, 83-88, 93, 94
    - If any of pres_wx_AU contains codes 03, 04
    
    Parameters:
    df: DataFrame with present weather columns
    
    Returns:
    DataFrame with added rain and snow columns
    """
    print("Detecting rain and snow events...")
    
    # Ensure present weather columns are strings to search within them
    wx_mw_aw_columns = ['pres_wx_AW1', 'pres_wx_AW2', 'pres_wx_AW3', 'pres_wx_MW1', 'pres_wx_MW2', 'pres_wx_MW3']
    wx_au_columns = ['pres_wx_AU1', 'pres_wx_AU2', 'pres_wx_AU3']
    all_wx_columns = wx_mw_aw_columns + wx_au_columns
    
    for col in all_wx_columns:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    # Initialize rain and snow columns
    df['rain'] = 0
    df['snow'] = 0
    
    # Define rain and snow codes for MW and AW columns
    mw_aw_rain_codes = [23, 25] + list(range(50, 70)) + list(range(80, 85)) + list(range(91, 100))
    mw_aw_snow_codes = [24] + list(range(27, 30)) + list(range(68, 80)) + list(range(83, 89)) + [93, 94]
    
    # Define rain and snow codes for AU columns
    au_rain_codes = ['01', '02']
    au_snow_codes = ['03', '04']
    
    # Convert codes to strings for pattern matching
    mw_aw_rain_codes_str = [str(code) for code in mw_aw_rain_codes]
    mw_aw_snow_codes_str = [str(code) for code in mw_aw_snow_codes]
    
    # Create functions to check if a cell contains any of the target codes
    def contains_mw_aw_code(cell_value, code_list):
        if pd.isna(cell_value) or cell_value == 'nan':
            return False
        
        # Split by colon to handle formats like "RA:62"
        parts = str(cell_value).replace(':', ' ').split()
        
        # Check each part to see if it matches a code
        for part in parts:
            # Try to extract just the digits from the part
            digits = ''.join(c for c in part if c.isdigit())
            if digits and digits in code_list:
                return True
        
        return False
    
    def contains_au_code(cell_value, code_list):
        if pd.isna(cell_value) or cell_value == 'nan':
            return False
        
        # Handle formats like "-RA:02", "+SN:03", "DZ:01", "SG:04"
        # We need to check for specific weather types with specific codes
        
        # Define patterns for each code
        patterns = {
            '01': ['DZ:01'],  # Drizzle
            '02': ['RA:02'],  # Rain
            '03': ['SN:03'],  # Snow
            '04': ['SG:04']   # Snow grains
        }
        
        # Check for each code we're looking for
        for code in code_list:
            pattern_list = patterns.get(code, [])
            for pattern in pattern_list:
                # Account for potential + or - prefix
                if pattern in cell_value or f"+{pattern}" in cell_value or f"-{pattern}" in cell_value:
                    return True
        
        return False
    
    # Check MW and AW columns
    for col in wx_mw_aw_columns:
        if col in df.columns:
            # Apply the function to each row
            df['has_rain_code'] = df[col].apply(lambda x: contains_mw_aw_code(x, mw_aw_rain_codes_str))
            df['has_snow_code'] = df[col].apply(lambda x: contains_mw_aw_code(x, mw_aw_snow_codes_str))
            
            # Update rain and snow columns
            df.loc[df['has_rain_code'], 'rain'] = 1
            df.loc[df['has_snow_code'], 'snow'] = 1
    
    # Check AU columns
    for col in wx_au_columns:
        if col in df.columns:
            # Apply the function to each row
            df['has_rain_code'] = df[col].apply(lambda x: contains_au_code(x, au_rain_codes))
            df['has_snow_code'] = df[col].apply(lambda x: contains_au_code(x, au_snow_codes))
            
            # Update rain and snow columns
            df.loc[df['has_rain_code'], 'rain'] = 1
            df.loc[df['has_snow_code'], 'snow'] = 1
    
    # Clean up temporary columns
    if 'has_rain_code' in df.columns:
        df.drop(['has_rain_code', 'has_snow_code'], axis=1, inplace=True)
    
    # Print counts
    rain_count = df['rain'].sum()
    snow_count = df['snow'].sum()
    print(f"Detected {rain_count} rain events and {snow_count} snow events")
    
    # Sample a few rows with rain or snow for validation
    if rain_count > 0 or snow_count > 0:
        print("\nSample rows with detected precipitation:")
        sample_rain = df[df['rain'] == 1].head(2)
        sample_snow = df[df['snow'] == 1].head(2)
        
        for sample, precip_type in [(sample_rain, "Rain"), (sample_snow, "Snow")]:
            if not sample.empty:
                print(f"\n{precip_type} samples:")
                for idx, row in sample.iterrows():
                    weather_info = []
                    for col in all_wx_columns:
                        if col in df.columns and not pd.isna(row[col]) and row[col] != 'nan':
                            weather_info.append(f"{col}={row[col]}")
                    weather_str = ", ".join(weather_info)
                    print(f"Row {idx}: Rain={row['rain']}, Snow={row['snow']}, Weather: {weather_str}")
    
    return df

def main():
    # Load the data
    print("Loading data...")
    df = pd.read_csv('data/weather/csv/combined_weather_nyc.csv', low_memory=False)

    # Convert columns to numeric
    print("Processing data...")
    df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce')
    df['wind_speed'] = pd.to_numeric(df['wind_speed'], errors='coerce')
    df['relative_humidity'] = pd.to_numeric(df['relative_humidity'], errors='coerce')

    # Convert date to datetime
    df['datetime'] = pd.to_datetime(df['DATE'])
    
    # Fill missing values with 3-month averages
    fill_columns = ['temperature', 'wind_speed', 'relative_humidity', 'LATITUDE', 'LONGITUDE']
    df = fill_missing_with_3month_avg(df, fill_columns)
    
    # Handle any remaining missing values
    remaining_missing = df[fill_columns].isna().sum().sum()
    if remaining_missing > 0:
        print(f"Warning: {remaining_missing} values still missing after 3-month average filling")
    
    # Estimate solar radiation
    print("Estimating solar radiation...")
    df = estimate_solar_radiation(df)
    
    # Calculate UTCI
    print("Calculating UTCI...")
    df['mean_radiant_temp'] = df['temperature'] + 0.7 * df['solar_radiation_estimated'] / 100
    
    # Apply the UTCI calculation
    print("Computing UTCI values...")
    df['utci'] = df.apply(lambda row: calculate_utci(
        row['temperature'], 
        row['wind_speed'], 
        row['relative_humidity'], 
        row['mean_radiant_temp']), axis=1)
    
    # Add UTCI category column
    print("Categorizing UTCI values...")
    df['utci_cat'] = df['utci'].apply(categorize_utci)
    
    # Add rain and snow columns
    df = detect_rain_and_snow(df)
    
    # Add weather category column
    print("Categorizing weather...")
    df['weather_cat'] = df.apply(lambda row: categorize_weather(
        row['utci'], 
        row['rain'], 
        row['snow']), axis=1)
    

    # Save the result
    print("Saving results...")
    output_columns = [
        'datetime', 'temperature', 'wind_speed', 'relative_humidity', 
        'solar_radiation_estimated', 'mean_radiant_temp', 'precipitation',
        'utci', 'utci_cat', 'rain', 'snow', 'weather_cat',
        'pres_wx_AW1', 'pres_wx_AW2', 'pres_wx_AW3',
        'pres_wx_MW1', 'pres_wx_MW2', 'pres_wx_MW3',
        'pres_wx_AU1', 'pres_wx_AU2', 'pres_wx_AU3',
        
    ]
    
    # Only include columns that exist in the dataframe
    output_columns = [col for col in output_columns if col in df.columns]
    df[output_columns].to_csv('data/weather/csv/nyc_weather_with_utci.csv', index=False)
    
    row_count = len(df)
    print(f"Done! UTCI values have been calculated and saved for all {row_count} rows.")

if __name__ == "__main__":
    main() 