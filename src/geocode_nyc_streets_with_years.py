import pandas as pd
import requests
import time
from typing import Optional, Tuple, List, Dict


def geocode_with_nominatim(location: str, retries: int = 3) -> Tuple[Optional[float], Optional[float]]:
    """
    Geocode a location using OpenStreetMap Nominatim API.
    
    Args:
        location: Location string to geocode
        retries: Number of retry attempts
        
    Returns:
        Tuple of (latitude, longitude) or (None, None) if not found
    """
    # Add NYC context to improve accuracy
    query = f"{location}, New York City, NY, USA"
    
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        'q': query,
        'format': 'json',
        'addressdetails': 1,
        'limit': 1
    }
    
    headers = {
        'User-Agent': 'NYC-Street-Geocoder/1.0'
    }
    
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            if data and len(data) > 0:
                result = data[0]
                lat = float(result['lat'])
                lon = float(result['lon'])
                
                # Verify the result is in NYC area (rough bounds check)
                if 40.4774 <= lat <= 40.9176 and -74.2591 <= lon <= -73.7004:
                    return lat, lon
            
            # Rate limiting - be respectful to the free service
            time.sleep(1.5)
            return None, None
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for '{location}': {e}")
            if attempt < retries - 1:
                time.sleep(2)  # Wait longer between retries
            
    return None, None


def geocode_street_list(street_locations: List[str], year: int) -> List[Dict]:
    """
    Geocode a list of streets for a specific year.
    
    Args:
        street_locations: List of street names
        year: Year for the data
        
    Returns:
        List of dictionaries with geocoding results
    """
    print(f"Starting geocoding of {len(street_locations)} NYC locations for {year}...")
    
    results = []
    
    for i, location in enumerate(street_locations, 1):
        print(f"Geocoding {i}/{len(street_locations)}: {location}")
        
        lat, lon = geocode_with_nominatim(location)
        
        result = {
            'street_name': location,
            'year': year,
            'latitude': lat,
            'longitude': lon,
            'geocoded_successfully': lat is not None and lon is not None
        }
        
        results.append(result)
        
        # Show progress
        if lat and lon:
            print(f"  ✓ Found: {lat:.6f}, {lon:.6f}")
        else:
            print(f"  ✗ Not found")
    
    return results


def main():
    """
    Geocode NYC streets for both 2022 and 2023 and save to CSV.
    """
    # 2022 street names
    streets_2022 = [
        "Riverdale Ave.",
        "E 233rd St.",
        "Eastchester Rd.",
        "White Plains Rd.",
        "Mosholu Pkwy.",
        "Fordham Rd.",
        "University Ave.",
        "Bronxdale Ave.",
        "165th St.",
        "167th St.",
        "Kingsland Ave.",
        "Flushing Ave.",
        "Schermerhorn St.",
        "20th St. - 3rd Ave.",
        "21st St.",
        "Owl's Head",
        "Sunset Park",
        "Emmons Ave.",
        "Neptune Ave.",
        "Buffalo Ave.",
        "Williams Ave.",
        "Hinsdale St.",
        "Amsterdam Ave./Fort George Ave.",
        "7 Ave.",
        "W 3rd St.",
        "E Houston St.",
        "Church St.",
        "44th Dr.",
        "Roosevelt Island Bridge",
        "36th Ave./Vernon Blvd.",
        "Northern Blvd.",
        "Queens Blvd.",
        "Eliot Ave.",
        "62nd Dr.",
        "63rd Rd.",
        "71st Ave.",
        "149th St.",
        "Far Rockaway",
        "Seagirt Blvd.",
        "Beach 108th St.",
        "Hylan Blvd."
    ]
    
    # 2023 street names
    streets_2023 = [
        # Bronx bike-related projects
        "White Plains Road",  # Project 3: White Plains Road (East 226th Street to East 241st Street) - protected bike lanes
        "Burke Avenue",  # Project 10: Burke Avenue (Eastchester Road to Laconia Avenue) - bike lanes to connect greenways
        "Mosholu Parkway",  # Project 11: Mosholu Parkway (Van Cortlandt Avenue to Southern Boulevard) - two-way bike path
        "University Avenue",  # Project 14: University Avenue (Tremont Avenue to Kingsbridge Road) - protected bike lanes
        "Grand Concourse",  # Project 16: Grand Concourse (175th Street to East Fordham Road) - protected bike lanes
        "Park Avenue",  # Project 17: Park Avenue (East 173rd Street to East 188th Street) - protected bike lanes
        "East 180th Street",  # Project 19: East 180th Street (Webster Ave to Boston Road) - protected bike lanes
        "Sheridan Boulevard",  # Project 22: Sheridan Boulevard Network - bike lanes
        "Westchester Avenue",  # Project 24: Westchester Avenue (Southern Boulevard to Whitlock Avenue) - protected bike lanes
        "Lafayette Avenue",  # Project 25: Lafayette Avenue (White Plains Road to Havemeyer Avenue) - protected bike lanes
        "Soundview Avenue",  # Project 26: Soundview Avenue (Rosedale Avenue to Clason Point) - protected bike lane
        
        # Manhattan bike-related projects
        "Harlem River Driveway",  # Project 28: Harlem River Driveway - upgraded protected bike lane
        "3rd Avenue",  # Project 33: 3rd Avenue (East 59th Street to East 96th Street) - protected bike lanes
        "7th Avenue",  # Project 34: 7th Avenue (West 59th Street to West 56 Street) - protected bike lane
        "11th Avenue",  # Project 36: 11th Avenue (West 41st Street to West 42 Street) - protected bike lane
        "Broadway",  # Project 38: Broadway (West 25th Street to West 32nd Street) - two-way cycling
        "West 22nd Street",  # Project 40: West 22nd Street (8th Avenue to 7th Avenue) - bike corrals
        "West 13th Street",  # Project 41: West 13th Street - bike crossing
        "Varick Street",  # Project 43: Varick Street & West Broadway - protected bike lanes
        "West Broadway",  # Project 43: Varick Street & West Broadway - protected bike lanes
        "Centre Street",  # Project 44: Centre Street & Lafayette Street (Worth Street to Prince Street) - protected bike lanes
        "Lafayette Street",  # Project 44: Centre Street & Lafayette Street (Worth Street to Prince Street) - protected bike lanes
        "Grand Street",  # Project 47: Grand Street to Williamsburg Bridge - protected bike lane
        
        # Queens bike-related projects
        "44th Drive",  # Project 48: Long Island City Hunter's Point - 44th Drive between Vernon Boulevard and 23rd Street
        "11th Street",  # Project 48: Long Island City Hunter's Point - 11th Street between 44th Drive and Jackson Avenue
        "Jackson Avenue",  # Project 48: Long Island City Hunter's Point - Jackson Avenue between Vernon Boulevard and Pulaski Bridge
        "Vernon Boulevard",  # Project 48: Long Island City Hunter's Point - referenced as boundary/connection point
        "100th Street",  # Project 52: 100th Street & 101st Street (32nd Avenue to 37th Avenue) - bike lanes
        "101st Street",  # Project 52: 100th Street & 101st Street (32nd Avenue to 37th Avenue) - bike lanes
        "34th Avenue",  # Project 53: 34th Avenue - cyclist priority corridor
        "59th Street",  # Project 54: 59th Street & 60th Street (34th Avenue to 39th Avenue) - bike lanes
        "60th Street",  # Project 54: 59th Street & 60th Street (34th Avenue to 39th Avenue) - bike lanes
        "33rd Avenue",  # Project 59: 33rd Avenue Bike Boulevard (215th Place to Utopia Parkway) - bike markings
        "63rd Road",  # Project 62: 63rd Road & Grand Central Parkway - protected bike lane
        "Grand Central Parkway",  # Project 62: 63rd Road & Grand Central Parkway - protected bike lane
        "Queens Boulevard",  # Project 69: Queens Boulevard (Union Turnpike to Jamaica Avenue) - protected bike lanes
        "Cross Bay Boulevard",  # Project 80: Addabbo Bridge & Cross Bay Boulevard - protected bike lanes
        "Seagirt Boulevard",  # Project 83: Seagirt Boulevard (Rockaway Freeway to Beach 9th Street) - protected bike lane
        
        # Brooklyn bike-related projects
        "Berry Street",  # Project 85: Berry Street - transformed into Bike Boulevard
        "Meeker Avenue",  # Project 87: Meeker Avenue (Apollo Street to Graham Avenue) - protected bike lane and multi-use path
        "Jay Street",  # Project 89: Jay Street & Sands Street - ramps for cyclists
        "Sands Street",  # Project 89: Jay Street & Sands Street - ramps for cyclists
        "Ashland Place",  # Project 90: Ashland Place & Navy Street - protected bike lanes
        "Navy Street",  # Project 90: Ashland Place & Navy Street - protected bike lanes
        "Gates Avenue",  # Project 95: Gates Avenue - bike parking
        "9th Street",  # Project 96: 9th Street (Smith Street to 3rd Avenue) - protected bike lane
        "Linden Boulevard",  # Project 101: East New York School Safety - protected bike lanes (connects to Linden Boulevard)
        
        # Staten Island bike-related projects
        "Goethals Road North",  # Project 115: Goethals Road North, South Avenue, & Lamberts Lane - bike lanes
        "South Avenue",  # Project 115: Goethals Road North, South Avenue, & Lamberts Lane - bike lanes
        "Lamberts Lane",  # Project 115: Goethals Road North, South Avenue, & Lamberts Lane - bike lanes
        "Lincoln Avenue"  # Project 116: Lincoln Avenue (Father Capodanno Boulevard to Boundary Avenue) - bike lane
    ]
    
    # Geocode both years
    results_2022 = geocode_street_list(streets_2022, 2022)
    results_2023 = geocode_street_list(streets_2023, 2023)
    
    # Combine results
    all_results = results_2022 + results_2023
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Save to CSV
    output_file = './data/nyc_streets_geocoded_with_years.csv'
    df.to_csv(output_file, index=False)
    
    # Print summary
    total_locations = len(df)
    successful_geocodes = df['geocoded_successfully'].sum()
    success_rate = (successful_geocodes / total_locations) * 100
    
    print(f"\n{'='*60}")
    print(f"GEOCODING SUMMARY")
    print(f"{'='*60}")
    print(f"Total locations: {total_locations}")
    print(f"Successfully geocoded: {successful_geocodes}")
    print(f"Failed to geocode: {total_locations - successful_geocodes}")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"\nResults saved to: {output_file}")
    
    # Show summary by year
    print(f"\nSummary by year:")
    for year in [2022, 2023]:
        year_data = df[df['year'] == year]
        year_total = len(year_data)
        year_success = year_data['geocoded_successfully'].sum()
        year_rate = (year_success / year_total) * 100 if year_total > 0 else 0
        print(f"  {year}: {year_success}/{year_total} ({year_rate:.1f}%)")
    
    # Show the first few results
    print(f"\nFirst 10 results:")
    print(df.head(10).to_string(index=False))
    
    # Show failed geocodes if any
    failed_geocodes = df[~df['geocoded_successfully']]
    if not failed_geocodes.empty:
        print(f"\nFailed to geocode:")
        for _, row in failed_geocodes.iterrows():
            print(f"  - {row['street_name']} ({row['year']})")


if __name__ == "__main__":
    main() 