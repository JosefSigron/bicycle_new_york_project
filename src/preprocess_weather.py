import pandas as pd
import numpy as np
from math import pi
import pythermalcomfort

def parse_sky_cover(df):
    """
    Parse sky cover codes from the dataset and convert to cloud cover fraction (0-1).
    
    The dataset contains up to 3 cloud layers (sky_cover_1, sky_cover_2, sky_cover_3).
    Per documentation, the last layer's value best represents the total sky state.
    
    Oktas values conversion:
    CLR:00 = 0.0 (Clear sky)
    FEW:01 = 0.1 (1/10 or less but not zero)
    FEW:02 = 0.25 (2/10 - 3/10)
    SCT:03 = 0.4 (4/10)
    SCT:04 = 0.5 (5/10)
    BKN:05 = 0.6 (6/10)
    BKN:06 = 0.75 (7/10 - 8/10)
    BKN:07 = 0.9 (9/10 or more but not 10/10)
    OVC:08 = 1.0 (10/10, completely overcast)
    VV:09 = 1.0 (Sky obscured)
    X:10 = 0.5 (Partial obscuration - assume 50%)
    
    Parameters:
    df: DataFrame with sky_cover_1, sky_cover_2, and/or sky_cover_3 columns
    
    Returns:
    DataFrame with added cloud_cover column (0-1 scale)
    """
    print("Parsing sky cover data...")
    
    # Check which sky cover columns exist
    sky_cover_columns = [col for col in ['sky_cover_1', 'sky_cover_2', 'sky_cover_3'] if col in df.columns]
    
    if not sky_cover_columns:
        print("  - No sky cover columns found in dataset")
        return df
    
    # Conversion dictionary for sky cover codes to cloud fraction
    sky_code_to_fraction = {
        'CLR:00': 0.0,
        'SKC': 0.0,  # Sometimes Clear is represented as SKC
        'CLR': 0.0,  # Sometimes Clear is represented as just CLR
        'FEW:01': 0.1,
        'FEW:02': 0.25,
        'FEW': 0.25,  # If only FEW is given, assume 2/10-3/10
        'SCT:03': 0.4,
        'SCT:04': 0.5,
        'SCT': 0.5,  # If only SCT is given, assume 5/10
        'BKN:05': 0.6,
        'BKN:06': 0.75,
        'BKN:07': 0.9,
        'BKN': 0.75,  # If only BKN is given, assume 7/10-8/10
        'OVC:08': 1.0,
        'OVC': 1.0,  # If only OVC is given, assume 10/10
        'VV:09': 1.0,
        'VV': 1.0,
        'X:10': 0.5,
        'X': 0.5
    }
    
    # Function to extract cloud fraction from a code
    def extract_cloud_fraction(code):
        if pd.isna(code) or code == 'nan' or not code:
            return np.nan
            
        code = str(code).strip().upper()
        
        # Direct lookup
        if code in sky_code_to_fraction:
            return sky_code_to_fraction[code]
            
        # Try to parse numeric part if format is different
        for key in sky_code_to_fraction:
            if code.startswith(key.split(':')[0]):
                return sky_code_to_fraction[key]
                
        # If we can't parse it, return NaN
        return np.nan
    
    # Apply conversion to all sky cover columns
    for col in sky_cover_columns:
        df[f'{col}_fraction'] = df[col].apply(extract_cloud_fraction)
    
    # Use the last available layer as the overall cloud cover
    # This follows the documentation that the last layer best represents total sky state
    df['cloud_cover'] = np.nan
    
    # Try sky_cover_3 first, then fall back to sky_cover_2, then sky_cover_1
    if 'sky_cover_3_fraction' in df.columns:
        df['cloud_cover'] = df['sky_cover_3_fraction']
        
    if 'sky_cover_2_fraction' in df.columns:
        df.loc[df['cloud_cover'].isna(), 'cloud_cover'] = df.loc[df['cloud_cover'].isna(), 'sky_cover_2_fraction']
        
    if 'sky_cover_1_fraction' in df.columns:
        df.loc[df['cloud_cover'].isna(), 'cloud_cover'] = df.loc[df['cloud_cover'].isna(), 'sky_cover_1_fraction']
    
    # Default to 0.5 (partly cloudy) if all values are missing
    df['cloud_cover'] = df['cloud_cover'].fillna(0.5)
    
    # Clean up temporary columns
    for col in sky_cover_columns:
        if f'{col}_fraction' in df.columns:
            df = df.drop(f'{col}_fraction', axis=1)
    
    cover_counts = df['cloud_cover'].value_counts(normalize=True).sort_index() * 100
    print(f"  - Cloud cover distribution: {', '.join([f'{v:.1f}% at {k:.1f}' for k, v in cover_counts.items()])}")
    
    return df

def calculate_mean_radiant_temperature(df):
    """
    Calculate Mean Radiant Temperature (MRT) using a simplified method based on air temperature
    and cloud cover fraction, following the approach from Thorsson et al. (2007).
    
    Parameters:
    df: DataFrame with temperature and cloud_cover columns
    
    Returns:
    DataFrame with added mean_radiant_temp column
    """
    print("Calculating Mean Radiant Temperature...")
    
    # If cloud cover is not available, use a default value of 50% (0.5)
    if 'cloud_cover' not in df.columns or df['cloud_cover'].isna().all():
        print("  - Cloud cover data not available, using default value of 0.5")
        df['cloud_cover'] = 0.5
    
    # Ensure cloud_cover is numeric and between 0-1
    df['cloud_cover'] = pd.to_numeric(df['cloud_cover'], errors='coerce')
    df['cloud_cover'] = df['cloud_cover'].clip(0, 1)
    
    # Calculate the solar adjustment factor based on cloud cover
    # This follows Lindberg et al. (2008) simplified approach
    # Clear sky = higher adjustment, overcast = lower adjustment
    solar_factor = (1 - 0.75 * df['cloud_cover'])
    
    # Calculate MRT using the simple formula:
    # MRT ≈ Tair + solar_adjustment
    # The solar adjustment is higher during the day and zero at night
    
    # Determine if it's day or night based on hour
    df['is_daytime'] = df['datetime'].dt.hour.between(6, 18)
    
    # Apply the adjustment only during daytime
    df['solar_adjustment'] = np.where(
        df['is_daytime'],
        solar_factor * 4.0,  # Up to 4°C adjustment for clear sky days
        0.0                  # No adjustment at night
    )
    
    # Calculate MRT as air temperature plus solar adjustment
    df['mean_radiant_temp'] = df['temperature'] + df['solar_adjustment']
    
    print(f"  - MRT calculated: ranges from {df['mean_radiant_temp'].min():.1f}°C to {df['mean_radiant_temp'].max():.1f}°C")
    
    # Clean up temporary columns
    df = df.drop(['is_daytime', 'solar_adjustment'], axis=1, errors='ignore')
    
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
    UTCI value in °C or NaN if calculation fails
    """
    from pythermalcomfort.models import utci
    
    # Check for invalid or missing input values
    if (pd.isna(temperature) or pd.isna(wind_speed) or 
        pd.isna(humidity) or pd.isna(mrt)):
        return np.nan
        
    try:
        # Ensure values are within valid ranges for UTCI calculation
        # Pythermalcomfort has valid ranges for each parameter
        
        # Air temperature range: -50°C to 50°C 
        if temperature < -50 or temperature > 50:
            return np.nan
            
        # Wind speed range: 0.5 to 17 m/s
        # Adjust wind speed to be within valid range
        v_adjusted = max(0.5, min(17, wind_speed))
            
        # Relative humidity range: 5% to 100%
        rh_adjusted = max(5, min(100, humidity))
            
        # Calculate UTCI using the pythermalcomfort library
        utci_value = utci(tdb=temperature, tr=mrt, v=v_adjusted, rh=rh_adjusted)
        
        return utci_value
    except Exception as e:
        # Log the error if needed
        print(f"Error calculating UTCI: {e}, temp={temperature}, wind={wind_speed}, rh={humidity}, mrt={mrt}")
        return np.nan

def categorize_utci(utci_value):
    """
    Categorize UTCI values according to thermal stress levels
    
    Parameters:
    utci_value: UTCI value in °C
    
    Returns:
    Category string
    """
    # Handle null values
    if pd.isna(utci_value):
        return "Unknown"
        
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
    # Handle the case where UTCI is null
    if pd.isna(utci):
        if snow == 1:
            return "Snow"
        elif rain == 1:
            return "Rain"
        else:
            return "Unknown"  # We can't determine thermal category without UTCI
    
    # Normal categorization when UTCI is available
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
    
    # Parse sky cover data to get cloud cover fraction
    df = parse_sky_cover(df)
    
    # Fill missing values with 3-month averages
    fill_columns = ['temperature', 'wind_speed', 'relative_humidity']
    df = fill_missing_with_3month_avg(df, fill_columns)
    
    # Handle any remaining missing values
    remaining_missing = df[fill_columns].isna().sum().sum()
    if remaining_missing > 0:
        print(f"Warning: {remaining_missing} values still missing after 3-month average filling")
        
        # Fill any remaining missing values with global means
        for column in fill_columns:
            if df[column].isna().sum() > 0:
                print(f"  - Filling {df[column].isna().sum()} remaining missing {column} values with global mean")
                df[column] = df[column].fillna(df[column].mean())
    
    # Calculate MRT using the simplified method
    df = calculate_mean_radiant_temperature(df)
    
    # Apply the UTCI calculation
    print("Computing UTCI values...")
    df['utci'] = df.apply(lambda row: calculate_utci(
        row['temperature'], 
        row['wind_speed'], 
        row['relative_humidity'], 
        row['mean_radiant_temp']), axis=1)
    
    # Report on null UTCI values
    null_utci_count = df['utci'].isna().sum()
    if null_utci_count > 0:
        print(f"Warning: {null_utci_count} null UTCI values ({null_utci_count/len(df)*100:.2f}% of data)")
        
        # Sample a few rows with null UTCI to help diagnose the issue
        print("\nSample rows with null UTCI values:")
        sample_nulls = df[df['utci'].isna()].head(3)
        for idx, row in sample_nulls.iterrows():
            print(f"Row {idx}: temp={row['temperature']}, wind={row['wind_speed']}, " +
                  f"rh={row['relative_humidity']}, mrt={row['mean_radiant_temp']}")
    
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
        'cloud_cover', 'mean_radiant_temp', 'precipitation',
        'utci', 'utci_cat', 'rain', 'snow', 'weather_cat',
        'sky_cover_1', 'sky_cover_2', 'sky_cover_3',
        'pres_wx_AW1', 'pres_wx_AW2', 'pres_wx_AW3',
        'pres_wx_MW1', 'pres_wx_MW2', 'pres_wx_MW3',
        'pres_wx_AU1', 'pres_wx_AU2', 'pres_wx_AU3'
    ]
    
    # Only include columns that exist in the dataframe
    output_columns = [col for col in output_columns if col in df.columns]
    df[output_columns].to_csv('data/weather/csv/nyc_weather_with_utci.csv', index=False)
    
    row_count = len(df)
    print(f"Done! UTCI values have been calculated and saved for all {row_count} rows.")
    print(f"  - {row_count - null_utci_count} rows with valid UTCI values")
    print(f"  - {null_utci_count} rows with null UTCI values")

if __name__ == "__main__":
    main() 