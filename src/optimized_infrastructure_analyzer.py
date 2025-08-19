#!/usr/bin/env python3
"""
Optimized Infrastructure Analyzer

Simplified version of the infrastructure analysis that focuses on performance
and avoids processing millions of station records individually.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from enhanced_street_matcher import EnhancedStreetMatcher
import os


class OptimizedInfrastructureAnalyzer:
    """Optimized infrastructure analyzer for large datasets."""
    
    def __init__(self):
        self.enhanced_matcher = EnhancedStreetMatcher()
        
    def analyze_infrastructure_impact(self, infrastructure_df, citibike_df, max_stations=5000):
        """
        Analyze infrastructure impact with optimizations for large datasets.
        
        Args:
            infrastructure_df: DataFrame with infrastructure locations
            citibike_df: DataFrame with CitiBike ride data
            max_stations: Maximum number of stations to analyze (for performance)
        """
        
        print("🚀 Starting Optimized Infrastructure Impact Analysis")
        print("=" * 60)
        
        # Load street data
        print("📍 Loading street geometries...")
        self.enhanced_matcher.load_street_data()
        
        # Match infrastructure to streets using name-based matching
        print("🎯 Matching infrastructure to exact streets...")
        enhanced_infrastructure = self.enhanced_matcher.match_infrastructure_to_streets(
            infrastructure_df,
            name_col='street_name',
            lat_col='latitude',
            lon_col='longitude',
            max_coord_distance=200
        )
        
        if len(enhanced_infrastructure) == 0:
            print("❌ No infrastructure could be matched to streets")
            return pd.DataFrame()
        
        # Extract unique stations efficiently
        print("🚉 Extracting unique stations...")
        unique_stations = citibike_df.groupby('start_station_id').agg({
            'start_station_latitude': 'first',
            'start_station_longitude': 'first'
        }).reset_index()
        
        # Remove stations with invalid coordinates
        unique_stations = unique_stations.dropna()
        unique_stations = unique_stations[
            (unique_stations['start_station_longitude'].between(-74.5, -73.5)) &
            (unique_stations['start_station_latitude'].between(40.4, 41.0))
        ]
        
        print(f"   Found {len(unique_stations):,} unique stations")
        
        # Sample stations if too many
        if len(unique_stations) > max_stations:
            print(f"   Sampling {max_stations:,} stations for analysis...")
            unique_stations = unique_stations.sample(n=max_stations, random_state=42)
        
        # Find affected stations
        print("🔍 Finding affected stations...")
        affected_stations = self._find_affected_stations_optimized(
            enhanced_infrastructure, unique_stations
        )
        
        print(f"✅ Analysis complete: {len(affected_stations)} station-infrastructure relationships found")
        
        return affected_stations
    
    def _find_affected_stations_optimized(self, infrastructure_df, stations_df):
        """Find affected stations using optimized spatial operations."""
        
        affected_relationships = []
        
        for _, infra in infrastructure_df.iterrows():
            # Get street information
            street_class = infra['street_class']
            influence_distance = infra['influence_distance']
            
            # Convert influence distance from meters to approximate degrees
            # (rough approximation: 1 degree ≈ 111km)
            influence_degrees = influence_distance / 111000
            
            # Find stations within rough influence area using vectorized operations
            lat_diff = abs(stations_df['start_station_latitude'] - infra['latitude'])
            lon_diff = abs(stations_df['start_station_longitude'] - infra['longitude'])
            
            # Rough distance filter (much faster than precise geometry)
            rough_distance = np.sqrt(lat_diff**2 + lon_diff**2)
            nearby_stations = stations_df[rough_distance <= influence_degrees * 2]  # 2x buffer for safety
            
            if len(nearby_stations) == 0:
                continue
            
            # For nearby stations, calculate more precise distances
            for _, station in nearby_stations.iterrows():
                # Simple distance calculation (good enough for this analysis)
                lat_diff = (station['start_station_latitude'] - infra['latitude']) * 111000  # Convert to meters
                lon_diff = (station['start_station_longitude'] - infra['longitude']) * 111000 * np.cos(np.radians(infra['latitude']))
                distance = np.sqrt(lat_diff**2 + lon_diff**2)
                
                if distance <= influence_distance:
                    impact_strength = 1.0 - (distance / influence_distance)
                    
                    affected_relationships.append({
                        'station_id': station['start_station_id'],
                        'infrastructure_street': infra['matched_street_name'],
                        'street_class': infra['street_class'],
                        'distance_to_infrastructure': distance,
                        'influence_distance': influence_distance,
                        'impact_strength': impact_strength,
                        'match_quality': infra['match_quality']
                    })
        
        return pd.DataFrame(affected_relationships)


def main():
    """Test the optimized analyzer."""
    
    # Load data
    print("Loading infrastructure data...")
    infrastructure_df = pd.read_csv("data/nyc_streets_geocoded_with_years.csv")
    
    print("Loading sample CitiBike data...")
    citibike_df = pd.read_parquet("data/combined/2022_combined_citibike_weather.parquet")
    
    # Take a sample for testing
    citibike_sample = citibike_df.sample(n=100000, random_state=42)
    
    # Run analysis
    analyzer = OptimizedInfrastructureAnalyzer()
    results = analyzer.analyze_infrastructure_impact(
        infrastructure_df[infrastructure_df['year'] == 2022],
        citibike_sample,
        max_stations=2000
    )
    
    print("\n📊 Results Summary:")
    if len(results) > 0:
        print(f"   Affected stations: {results['station_id'].nunique()}")
        print(f"   Infrastructure locations with impact: {results['infrastructure_street'].nunique()}")
        print(f"   Street class distribution:")
        print(results['street_class'].value_counts().to_string())
        print(f"\n   Sample results:")
        print(results.head().to_string())
    else:
        print("   No impacts found")


if __name__ == "__main__":
    main()
