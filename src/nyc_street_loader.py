#!/usr/bin/env python3
"""
NYC LION Street Data Loader for Infrastructure Analysis

Loads and processes NYC LION (Linear Integrated Ordered Network) street data
to enhance infrastructure analysis with accurate street classification and sizing.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
import os
from typing import Dict, List, Tuple, Optional

class NYCStreetLoader:
    """Load and process NYC LION street data for infrastructure analysis."""
    
    def __init__(self, lion_path: str = None):
        if lion_path is None:
            # Create absolute path to LION database
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            lion_path = os.path.join(project_root, "data", "street_locations", "lion.gdb")
        
        self.lion_path = lion_path
        self.street_data = None
        self.processed_data = None
        
    def load_lion_streets(self, use_cache: bool = True, cache_file: str = None):
        """Load and process NYC LION street data."""
        
        if cache_file is None:
            # Create absolute path to cache file
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            cache_file = os.path.join(project_root, "data", "nyc_lion_processed.parquet")
        
        if use_cache and os.path.exists(cache_file):
            try:
                print("Loading cached LION street data...")
                self.processed_data = pd.read_parquet(cache_file)
                print(f"Loaded {len(self.processed_data):,} street segments from cache")
                return self.processed_data
            except Exception as e:
                print(f"Cache loading failed: {e}, downloading fresh data...")
        
        print("Loading NYC LION street data...")
        
        # Load the main LION layer
        self.street_data = gpd.read_file(self.lion_path, layer='lion')
        print(f"Loaded {len(self.street_data):,} total segments")
        
        # Process and classify streets
        self.processed_data = self._process_lion_data()
        
        # Cache the processed data
        if use_cache:
            try:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                # Convert to DataFrame for parquet (losing geometry but keeping classification)
                cache_df = pd.DataFrame(self.processed_data.drop(columns=['geometry']))
                cache_df.to_parquet(cache_file)
                print(f"Cached processed street data to {cache_file}")
            except Exception as e:
                print(f"Caching failed: {e}")
        
        print(f"Processed {len(self.processed_data):,} street segments")
        return self.processed_data
    
    def _process_lion_data(self):
        """Process raw LION data and add street classifications."""
        
        # Filter to actual streets (FeatureTyp 0 = streets)
        streets = self.street_data[self.street_data['FeatureTyp'] == 0].copy()
        print(f"Filtered to {len(streets):,} actual street segments")
        
        # Clean and process key columns
        streets['street_width'] = streets[['StreetWidth_Min', 'StreetWidth_Max']].mean(axis=1)
        streets['travel_lanes'] = streets['Number_Travel_Lanes'].fillna(1).astype(int)
        streets['segment_length'] = streets['SHAPE_Length']
        
        # Handle missing width data with reasonable defaults based on lanes
        streets['street_width'] = streets['street_width'].fillna(
            streets['travel_lanes'] * 12 + 18  # ~12ft per lane + 18ft for parking/sidewalks
        )
        
        # Classify streets based on width and lanes
        streets['street_class'] = streets.apply(self._classify_street, axis=1)
        
        # Add buffer sizes based on street classification
        buffer_map = {
            'major': 750,      # Major roads: larger impact zone
            'arterial': 600,   # Arterial roads: medium impact zone
            'local': 450,      # Local roads: smaller impact zone
            'service': 300     # Service roads: minimal impact zone
        }
        
        streets['buffer_meters'] = streets['street_class'].map(buffer_map)
        streets['buffer_degrees'] = streets['buffer_meters'] / 111000  # Convert to lat/lon degrees
        
        # Keep essential columns
        essential_cols = [
            'Street', 'SegmentTyp', 'street_width', 'travel_lanes', 'segment_length',
            'street_class', 'buffer_meters', 'buffer_degrees', 'geometry'
        ]
        
        processed = streets[essential_cols].copy()
        
        # Add summary statistics
        print("\nStreet Classification Summary:")
        print(processed['street_class'].value_counts())
        print("\nStreet Width Statistics by Class:")
        print(processed.groupby('street_class')['street_width'].agg(['count', 'mean', 'min', 'max']))
        
        return processed
    
    def _classify_street(self, row):
        """Classify a street based on width and travel lanes."""
        width = row['street_width']
        lanes = row['travel_lanes']
        segment_type = row['SegmentTyp']
        
        # Handle special segment types
        if segment_type in ['F']:  # Ferry routes
            return 'service'
        elif segment_type in ['C', 'T']:  # Circles, tunnels - often major
            return 'major'
        
        # Classify based on width and lanes
        if width >= 60 or lanes >= 3:
            return 'major'
        elif width >= 40 or lanes == 2:
            return 'arterial'
        elif width >= 25:
            return 'local'
        else:
            return 'service'
    
    def get_street_info_for_locations(self, locations_df: pd.DataFrame, 
                                    lat_col: str = 'latitude', 
                                    lon_col: str = 'longitude',
                                    max_distance: float = 0.002) -> pd.DataFrame:
        """Get street information for infrastructure locations."""
        
        if self.processed_data is None:
            self.load_lion_streets()
        
        results = []
        
        print(f"Finding street information for {len(locations_df)} locations...")
        
        for idx, location in locations_df.iterrows():
            if pd.isna(location[lat_col]) or pd.isna(location[lon_col]):
                results.append(self._get_default_street_info())
                continue
            
            # Create point for location
            location_point = Point(location[lon_col], location[lat_col])
            
            # Find nearest street within reasonable distance
            if hasattr(self.processed_data, 'geometry'):
                # If we have geometry data
                distances = self.processed_data.geometry.distance(location_point)
                nearest_idx = distances.idxmin()
                
                if distances.iloc[nearest_idx] <= max_distance:
                    nearest_street = self.processed_data.iloc[nearest_idx]
                    results.append({
                        'street_name': nearest_street.get('Street', 'unknown'),
                        'street_class': nearest_street['street_class'],
                        'street_width': nearest_street['street_width'],
                        'travel_lanes': nearest_street['travel_lanes'],
                        'buffer_meters': nearest_street['buffer_meters'],
                        'buffer_degrees': nearest_street['buffer_degrees'],
                        'distance_to_street': distances.iloc[nearest_idx] * 111000  # Convert to meters
                    })
                else:
                    results.append(self._get_default_street_info())
            else:
                # Fallback: use coordinate-based search
                results.append(self._get_default_street_info())
        
        return pd.DataFrame(results)
    
    def _get_default_street_info(self):
        """Get default street information when no match found."""
        return {
            'street_name': 'unknown',
            'street_class': 'local',
            'street_width': 30.0,
            'travel_lanes': 1,
            'buffer_meters': 450,
            'buffer_degrees': 0.004,
            'distance_to_street': 999.0
        }
    
    def get_buffer_size_for_infrastructure(self, infrastructure_df: pd.DataFrame) -> pd.DataFrame:
        """Get appropriate buffer sizes for infrastructure based on nearest streets."""
        
        street_info = self.get_street_info_for_locations(infrastructure_df)
        
        # Combine infrastructure with street information
        enhanced_infrastructure = infrastructure_df.copy()
        for col in street_info.columns:
            enhanced_infrastructure[col] = street_info[col].values
        
        print(f"\nBuffer size distribution:")
        print(enhanced_infrastructure['buffer_meters'].value_counts().sort_index())
        
        return enhanced_infrastructure

# Usage example and testing
if __name__ == "__main__":
    loader = NYCStreetLoader()
    
    # Load and process LION data
    streets = loader.load_lion_streets()
    
    print(f"\nProcessed {len(streets):,} street segments")
    print("\nStreet classification summary:")
    print(streets['street_class'].value_counts())
