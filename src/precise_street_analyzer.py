#!/usr/bin/env python3
"""
Precise Street Geometry Analyzer for Infrastructure Analysis

Uses exact street geometries from NYC LION data for precise spatial analysis
instead of crude buffer zones. Calculates exact distances and geometric relationships.

Enhanced with name-based street matching to eliminate the need for 2km distance thresholds.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, LineString, box
from shapely.ops import nearest_points
import os
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

# Import the enhanced street matcher
from enhanced_street_matcher import EnhancedStreetMatcher

class PreciseStreetAnalyzer:
    """Precise geometric analysis using exact street coordinates with enhanced name matching."""
    
    def __init__(self, lion_path: str = None):
        if lion_path is None:
            # Create absolute path to LION database
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            lion_path = os.path.join(project_root, "data", "street_locations", "lion.gdb")
        
        self.lion_path = lion_path
        self.street_geometries = None
        self.processed_streets = None
        self.street_geometries_projected = None
        self.spatial_index = None
        self.crs_projected = 'EPSG:2263'  # NAD83 / New York Long Island
        
        # Initialize enhanced street matcher for name-based matching
        self.enhanced_matcher = EnhancedStreetMatcher(lion_path)
        
    def load_street_geometries(self, use_cache: bool = True, cache_file: str = None):
        """Load street geometries with full spatial data."""
        
        if cache_file is None:
            # Create absolute path to cache file
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            cache_file = os.path.join(project_root, "data", "nyc_streets_geometry.gpkg")
        
        if use_cache and os.path.exists(cache_file):
            try:
                print("Loading cached street geometries...")
                self.street_geometries = gpd.read_file(cache_file)
                print(f"Loaded {len(self.street_geometries):,} street geometries from cache")
                return self.street_geometries
            except Exception as e:
                print(f"Cache loading failed: {e}, loading fresh data...")
        
        print("Loading NYC LION street geometries...")
        
        # Load with full geometry data
        raw_streets = gpd.read_file(self.lion_path, layer='lion')
        print(f"Loaded {len(raw_streets):,} total segments")
        
        # Filter to actual streets and process
        self.street_geometries = self._process_street_geometries(raw_streets)
        
        # Cache the processed geometries
        if use_cache:
            try:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                self.street_geometries.to_file(cache_file, driver="GPKG")
                print(f"Cached street geometries to {cache_file}")
            except Exception as e:
                print(f"Caching failed: {e}")
        
        print(f"Processed {len(self.street_geometries):,} street geometries")
        
        # Prepare projected geometries and spatial index for efficient queries
        self._prepare_projected_geometries()
        
        return self.street_geometries
    
    def _prepare_projected_geometries(self, cache_file: str = None):
        """Prepare projected geometries and spatial index for efficient queries."""
        
        if cache_file is None:
            # Create absolute path to cache file
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            cache_file = os.path.join(project_root, "data", "nyc_streets_projected.gpkg")
        
        if os.path.exists(cache_file):
            try:
                print("Loading cached projected street geometries...")
                self.street_geometries_projected = gpd.read_file(cache_file)
                print(f"Loaded {len(self.street_geometries_projected):,} projected geometries from cache")
                
                # Create spatial index
                self.spatial_index = self.street_geometries_projected.sindex
                return
            except Exception as e:
                print(f"Error loading cached projected geometries: {e}")
        
        if self.street_geometries is not None:
            print(f"Converting {len(self.street_geometries):,} street geometries to projected CRS (one-time operation)...")
            try:
                # Add progress bar for CRS conversion
                tqdm.pandas(desc="Converting street geometries to projected CRS")
                self.street_geometries_projected = self.street_geometries.to_crs(self.crs_projected)
                
                # Create spatial index
                self.spatial_index = self.street_geometries_projected.sindex
                
                # Cache the projected geometries for future use
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                self.street_geometries_projected.to_file(cache_file, driver='GPKG')
                print(f"Cached projected geometries to {cache_file}")
                
            except Exception as e:
                print(f"Error creating projected geometries: {e}")
                self.street_geometries_projected = None
                self.spatial_index = None
    
    def _process_street_geometries(self, raw_streets):
        """Process raw LION data keeping full geometries."""
        
        # Debug: Check FeatureTyp values
        print(f"FeatureTyp value counts: {raw_streets['FeatureTyp'].value_counts().head()}")
        
        # Filter to actual streets (FeatureTyp '0' = streets, stored as string)
        streets = raw_streets[raw_streets['FeatureTyp'] == '0'].copy()
        print(f"Filtered to {len(streets):,} actual street segments")
        
        # Clean and process key columns
        print(f"Processing columns for {len(streets)} street segments...")
        
        # Handle street width calculation with debug info
        width_data = streets[['StreetWidth_Min', 'StreetWidth_Max']].copy()
        print(f"Width data info: Min has {width_data['StreetWidth_Min'].notna().sum()} non-null values, Max has {width_data['StreetWidth_Max'].notna().sum()} non-null values")
        
        streets['street_width'] = width_data.mean(axis=1)
        
        # Handle travel lanes with proper string cleaning
        travel_lanes_clean = streets['Number_Travel_Lanes'].astype(str).str.strip()
        travel_lanes_clean = travel_lanes_clean.replace(['', 'nan', 'None'], '1')
        travel_lanes_clean = pd.to_numeric(travel_lanes_clean, errors='coerce').fillna(1).astype(int)
        streets['travel_lanes'] = travel_lanes_clean
        
        streets['segment_length'] = streets['SHAPE_Length']
        
        # Handle missing width data
        streets['street_width'] = streets['street_width'].fillna(
            streets['travel_lanes'] * 12 + 18  # ~12ft per lane + 18ft for parking/sidewalks
        )
        
        print(f"After processing: {streets['street_width'].notna().sum()} streets have width data")
        
        # Classify streets
        print("Classifying streets...")
        streets['street_class'] = streets.apply(self._classify_street, axis=1)
        print(f"Street classification complete. Sample classes: {streets['street_class'].value_counts().head()}")
        
        # Calculate influence zones based on street class and width
        print("Calculating influence distances...")
        streets['influence_distance'] = streets.apply(self._calculate_influence_distance, axis=1)
        print(f"Influence distance calculation complete. Sample distances: {streets['influence_distance'].describe()}")
        
        # Keep essential columns including full geometry
        essential_cols = [
            'Street', 'SegmentTyp', 'street_width', 'travel_lanes', 'segment_length',
            'street_class', 'influence_distance', 'geometry'
        ]
        
        print(f"Available columns: {list(streets.columns)}")
        print(f"Selecting essential columns: {essential_cols}")
        
        # Check if all essential columns exist
        missing_cols = [col for col in essential_cols if col not in streets.columns]
        if missing_cols:
            print(f"⚠️  Missing columns: {missing_cols}")
        
        processed = streets[essential_cols].copy()
        print(f"After column selection: {len(processed)} segments")
        
        # Ensure we have valid geometries
        print(f"Before geometry filtering: {len(processed)} segments")
        print(f"Segments with null geometry: {processed.geometry.isna().sum()}")
        
        processed = processed[~processed.geometry.isna()]
        print(f"After removing null geometries: {len(processed)} segments")
        
        print(f"Segments with invalid geometry: {(~processed.geometry.is_valid).sum()}")
        processed = processed[processed.geometry.is_valid]
        print(f"After removing invalid geometries: {len(processed)} segments")
        
        if len(processed) > 0:
            print(f"\nStreet Classification Summary:")
            print(processed['street_class'].value_counts())
        else:
            print("\n⚠️  No valid street segments remaining after filtering!")
        print(f"\nInfluence Distance Summary:")
        print(processed.groupby('street_class')['influence_distance'].agg(['mean', 'min', 'max']))
        
        return processed
    
    def _classify_street(self, row):
        """Classify a street based on width and travel lanes."""
        try:
            width = row['street_width'] if not pd.isna(row['street_width']) else 30.0
            lanes = row['travel_lanes'] if not pd.isna(row['travel_lanes']) else 1
            segment_type = row['SegmentTyp'] if not pd.isna(row['SegmentTyp']) else 'U'
            
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
        except Exception as e:
            print(f"Error classifying street: {e}, using 'local' as default")
            return 'local'
    
    def _calculate_influence_distance(self, row):
        """Calculate precise influence distance based on street characteristics."""
        try:
            width = row['street_width'] if not pd.isna(row['street_width']) else 30.0
            street_class = row['street_class'] if 'street_class' in row else self._classify_street(row)
            
            # Base influence on street width and classification
            if street_class == 'major':
                return max(width * 8, 400)  # Major: 8x width or min 400m
            elif street_class == 'arterial':
                return max(width * 6, 300)  # Arterial: 6x width or min 300m  
            elif street_class == 'local':
                return max(width * 4, 200)  # Local: 4x width or min 200m
            else:
                return max(width * 2, 100)  # Service: 2x width or min 100m
        except Exception as e:
            print(f"Error calculating influence distance: {e}, using 300m as default")
            return 300.0
    
    def find_enhanced_infrastructure_impact(self, infrastructure_df: pd.DataFrame, 
                                          stations_df: pd.DataFrame,
                                          use_name_matching: bool = True) -> pd.DataFrame:
        """
        Find stations impacted by infrastructure using enhanced matching (name-first, then geometric).
        
        Args:
            infrastructure_df: DataFrame with infrastructure locations (must have 'street_name' column)
            stations_df: DataFrame with station locations
            use_name_matching: If True, use name-based matching first, then fallback to coordinates
            
        Returns:
            Tuple of (enhanced_infrastructure, affected_stations)
        """
        
        if use_name_matching and 'street_name' in infrastructure_df.columns:
            print("🎯 Using enhanced name-based street matching...")
            return self._find_impact_with_name_matching(infrastructure_df, stations_df)
        else:
            print("⚠️  Falling back to coordinate-based matching...")
            return self.find_precise_infrastructure_impact(infrastructure_df, stations_df)
    
    def _find_impact_with_name_matching(self, infrastructure_df: pd.DataFrame,
                                       stations_df: pd.DataFrame) -> pd.DataFrame:
        """Find infrastructure impact using name-based street matching."""
        
        # Use enhanced street matcher to match infrastructure to exact streets
        enhanced_infrastructure = self.enhanced_matcher.match_infrastructure_to_streets(
            infrastructure_df,
            name_col='street_name',
            lat_col='latitude', 
            lon_col='longitude',
            max_coord_distance=200  # Much smaller than 2km!
        )
        
        if len(enhanced_infrastructure) == 0:
            print("❌ No infrastructure locations could be matched to streets")
            return pd.DataFrame(), pd.DataFrame()
        
        print(f"✅ Successfully matched {len(enhanced_infrastructure)} infrastructure locations")
        
        # Extract unique stations to reduce processing load
        print("Extracting unique stations from ride data...")
        unique_stations = stations_df[['start_station_id', 'start_station_latitude', 'start_station_longitude']].drop_duplicates()
        unique_stations = unique_stations.dropna()
        print(f"Found {len(unique_stations):,} unique stations (reduced from {len(stations_df):,} ride records)")
        
        # Use all stations for accurate results - the distance-based approach is efficient
        print(f"Using all {len(unique_stations):,} stations for infrastructure impact analysis (no sampling)")
        
        # Convert unique stations to GeoDataFrame
        station_geometries = []
        valid_station_indices = []
        
        for idx, row in unique_stations.iterrows():
            try:
                if pd.notna(row['start_station_longitude']) and pd.notna(row['start_station_latitude']):
                    if -74.5 <= row['start_station_longitude'] <= -73.5 and 40.4 <= row['start_station_latitude'] <= 41.0:
                        station_geometries.append(Point(row['start_station_longitude'], row['start_station_latitude']))
                        valid_station_indices.append(idx)
            except Exception as e:
                print(f"Warning: Error creating point for station {idx}: {e}")
        
        if not station_geometries:
            print("❌ No valid station coordinates found!")
            return pd.DataFrame(), pd.DataFrame()
            
        station_points = gpd.GeoDataFrame(
            unique_stations.loc[valid_station_indices],
            geometry=station_geometries,
            crs='EPSG:4326'
        )
        
        # Convert to projected CRS for accurate calculations
        try:
            print(f"Converting {len(station_points)} station points to projected CRS...")
            stations_projected = station_points.to_crs(self.crs_projected)
            
            # Create spatial index for efficient queries
            try:
                print("Creating spatial index for stations...")
                stations_projected.sindex
                print("✅ Spatial index created successfully")
            except Exception as idx_error:
                print(f"Warning: Could not create spatial index: {idx_error}")
                
        except Exception as e:
            print(f"Error in CRS transformation: {e}")
            return pd.DataFrame(), pd.DataFrame()
        
        # Find affected stations using exact street geometries
        affected_stations = self._find_precisely_affected_stations_enhanced(
            enhanced_infrastructure, stations_projected
        )
        
        return enhanced_infrastructure, affected_stations
    
    def _find_precisely_affected_stations_enhanced(self, enhanced_infrastructure, stations_projected):
        """Find stations affected by infrastructure using simple, reliable distance calculations."""
        
        affected_station_details = []
        
        print(f"Finding affected stations for {len(enhanced_infrastructure)} infrastructure locations...")
        print("🔄 Using NEW simple distance-based approach for reliable results...")
        
        # Print infrastructure summary
        print(f"Influence distances ranging from {enhanced_infrastructure['influence_distance'].min():.0f}m to {enhanced_infrastructure['influence_distance'].max():.0f}m")
        
        # Convert stations back to lat/lon for simple distance calculations
        stations_latlon = stations_projected.to_crs('EPSG:4326')
        
        total_found = 0
        for _, infra in tqdm(enhanced_infrastructure.iterrows(), 
                            total=len(enhanced_infrastructure),
                            desc="Finding affected stations"):
            
            # Infrastructure coordinates
            infra_lat = infra['latitude']
            infra_lon = infra['longitude']
            influence_distance = infra['influence_distance']
            
            # Debug for first few
            if infra.name < 3:
                print(f"\n  Infrastructure {infra.name}: {infra.get('matched_street_name', 'Unknown')}")
                print(f"    Location: ({infra_lat:.6f}, {infra_lon:.6f})")
                print(f"    Influence distance: {influence_distance}m")
            
            # Calculate distances to all stations using simple lat/lon math
            # Convert influence distance from meters to approximate degrees
            # 1 degree latitude ≈ 111,000 meters
            # 1 degree longitude ≈ 111,000 * cos(latitude) meters
            lat_range = influence_distance / 111000.0
            lon_range = influence_distance / (111000.0 * np.cos(np.radians(infra_lat)))
            
            # Filter stations within bounding box first (fast vectorized operation)
            lat_diff = abs(stations_latlon.geometry.y - infra_lat)
            lon_diff = abs(stations_latlon.geometry.x - infra_lon)
            
            candidate_stations = stations_latlon[
                (lat_diff <= lat_range) & (lon_diff <= lon_range)
            ]
            
            if len(candidate_stations) == 0:
                if infra.name < 3:
                    print(f"    No stations within bounding box")
                continue
            
            # Calculate actual distances for candidates (vectorized for better performance)
            stations_within_distance = []
            if len(candidate_stations) > 0:
                # Vectorized distance calculation for all candidates at once
                station_lats = candidate_stations.geometry.y.values
                station_lons = candidate_stations.geometry.x.values
                
                # Vectorized Haversine calculation
                distances_m = self._calculate_haversine_distance_vectorized(
                    infra_lat, infra_lon, station_lats, station_lons
                )
                
                # Filter stations within influence distance
                within_distance_mask = distances_m <= influence_distance
                stations_in_range = candidate_stations[within_distance_mask]
                distances_in_range = distances_m[within_distance_mask]
                
                # Create records for all stations within range
                for idx, (_, station) in enumerate(stations_in_range.iterrows()):
                    distance_m = distances_in_range[idx]
                    # Calculate impact strength (1.0 = at infrastructure, 0.0 = at edge of influence)
                    impact_strength = max(0.0, 1.0 - (distance_m / influence_distance))
                    
                    affected_station_details.append({
                        'station_id': station['start_station_id'],
                        'infrastructure_idx': infra.name,
                        'exact_distance_to_street': distance_m,
                        'influence_distance': influence_distance,
                        'position_along_street_ratio': 0.5,  # Not meaningful for distance-based approach
                        'street_name': infra.get('matched_street_name', 'Unknown'),
                        'street_class': infra.get('street_class', 'local'),
                        'street_width': infra.get('street_width', 30.0),
                        'travel_lanes': infra.get('travel_lanes', 2),
                        'impact_strength': impact_strength,
                        'match_quality': infra.get('match_quality', 'distance_based')
                    })
                    stations_within_distance.append(station['start_station_id'])
            
            if infra.name < 3:
                print(f"    Found {len(stations_within_distance)} stations within {influence_distance}m")
                if len(stations_within_distance) > 0:
                    print(f"    Sample station IDs: {stations_within_distance[:3]}")
            
            total_found += len(stations_within_distance)
        
        result_df = pd.DataFrame(affected_station_details)
        
        if len(result_df) > 0:
            unique_stations = len(result_df['station_id'].unique())
            print(f"✅ Found {len(result_df)} station-infrastructure relationships affecting {unique_stations} unique stations")
            print(f"   Average distance: {result_df['exact_distance_to_street'].mean():.1f}m")
            print(f"   Distance range: {result_df['exact_distance_to_street'].min():.1f}m - {result_df['exact_distance_to_street'].max():.1f}m")
        else:
            print("⚠️  No affected stations found")
        
        return result_df
    
    def _calculate_haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate the great circle distance between two points on Earth in meters."""
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Radius of Earth in meters
        r = 6371000
        
        return c * r
    
    def _calculate_haversine_distance_vectorized(self, lat1, lon1, lat2_array, lon2_array):
        """Calculate the great circle distance between one point and an array of points (vectorized)."""
        # Convert to radians
        lat1, lon1 = np.radians(lat1), np.radians(lon1)
        lat2_array, lon2_array = np.radians(lat2_array), np.radians(lon2_array)
        
        # Haversine formula (vectorized)
        dlat = lat2_array - lat1
        dlon = lon2_array - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2_array) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Radius of Earth in meters
        r = 6371000
        
        return c * r

    def find_precise_infrastructure_impact(self, infrastructure_df: pd.DataFrame, 
                                         stations_df: pd.DataFrame) -> pd.DataFrame:
        """Find stations impacted by infrastructure using precise geometric analysis."""
        
        if self.street_geometries is None:
            self.load_street_geometries()
        
        print(f"Analyzing precise impact for {len(infrastructure_df)} infrastructure locations...")
        print(f"Infrastructure columns: {list(infrastructure_df.columns)}")
        print(f"Stations columns: {list(stations_df.columns)}")
        print(f"Street geometries available: {len(self.street_geometries)}")
        
        # Convert infrastructure to GeoDataFrame with error handling
        try:
            infra_geometries = []
            valid_infra_indices = []
            
            for idx, row in infrastructure_df.iterrows():
                try:
                    if pd.notna(row['longitude']) and pd.notna(row['latitude']):
                        # Check if coordinates are reasonable for NYC area
                        if -74.5 <= row['longitude'] <= -73.5 and 40.4 <= row['latitude'] <= 41.0:
                            infra_geometries.append(Point(row['longitude'], row['latitude']))
                            valid_infra_indices.append(idx)
                        else:
                            print(f"Warning: Infrastructure at ({row['latitude']}, {row['longitude']}) outside NYC bounds")
                    else:
                        print(f"Warning: Infrastructure with missing coordinates at index {idx}")
                except Exception as e:
                    print(f"Warning: Error creating point for infrastructure {idx}: {e}")
            
            if not infra_geometries:
                print("Error: No valid infrastructure coordinates found!")
                return pd.DataFrame(), pd.DataFrame()
            
            infra_points = gpd.GeoDataFrame(
                infrastructure_df.iloc[valid_infra_indices],
                geometry=infra_geometries,
                crs='EPSG:4326'
            )
            
        except Exception as e:
            print(f"Error creating infrastructure GeoDataFrame: {e}")
            return pd.DataFrame(), pd.DataFrame()
        
        # Convert stations to GeoDataFrame with error handling
        try:
            station_geometries = []
            valid_station_indices = []
            
            for idx, row in stations_df.iterrows():
                try:
                    if pd.notna(row['start_station_longitude']) and pd.notna(row['start_station_latitude']):
                        # Check if coordinates are reasonable for NYC area
                        if -74.5 <= row['start_station_longitude'] <= -73.5 and 40.4 <= row['start_station_latitude'] <= 41.0:
                            station_geometries.append(Point(row['start_station_longitude'], row['start_station_latitude']))
                            valid_station_indices.append(idx)
                        else:
                            print(f"Warning: Station at ({row['start_station_latitude']}, {row['start_station_longitude']}) outside NYC bounds")
                    else:
                        print(f"Warning: Station with missing coordinates at index {idx}")
                except Exception as e:
                    print(f"Warning: Error creating point for station {idx}: {e}")
            
            if not station_geometries:
                print("Error: No valid station coordinates found!")
                return pd.DataFrame(), pd.DataFrame()
                
            station_points = gpd.GeoDataFrame(
                stations_df.loc[valid_station_indices],
                geometry=station_geometries,
                crs='EPSG:4326'
            )
            
        except Exception as e:
            print(f"Error creating station GeoDataFrame: {e}")
            return pd.DataFrame(), pd.DataFrame()
        
        # Convert to projected CRS for accurate distance calculations (NYC State Plane)        
        try:
            print(f"Converting {len(infra_points)} infrastructure points to projected CRS...")
            infra_projected = infra_points.to_crs(self.crs_projected)
            
            print(f"Converting {len(station_points)} station points to projected CRS...")
            # Add progress bar for station conversion (this is usually the slowest part)
            tqdm.pandas(desc="Converting station points to projected CRS")
            stations_projected = station_points.to_crs(self.crs_projected)
            
            # Use pre-computed projected street geometries
            if self.street_geometries_projected is None:
                print("Projected street geometries not available, preparing them...")
                self._prepare_projected_geometries()
            
            if self.street_geometries_projected is None:
                print("Error: Could not load projected street geometries")
                return pd.DataFrame(), pd.DataFrame()
                
            streets_projected = self.street_geometries_projected
            print(f"Using {len(streets_projected):,} cached projected street geometries")
            
        except Exception as e:
            print(f"Error in CRS transformation: {e}")
            return pd.DataFrame(), pd.DataFrame()
        
        # For each infrastructure point, find the street it's on
        enhanced_infrastructure = []
        
        print(f"Processing {len(infra_projected)} infrastructure points...")
        for position_idx, (idx, infra) in enumerate(tqdm(infra_projected.iterrows(), 
                                                         total=len(infra_projected),
                                                         desc="Finding nearest streets for infrastructure")):
            infra_point = infra.geometry
            
            # Use spatial index for faster nearest neighbor search
            if self.spatial_index is not None:
                # Get nearby streets using spatial index (much faster)
                buffer_distance = 2000  # 2km buffer for initial filtering
                nearby_indices = list(self.spatial_index.intersection(infra_point.buffer(buffer_distance).bounds))
                
                if not nearby_indices:
                    print(f"Warning: No streets found near infrastructure at {infra_point}")
                    continue
                
                # Calculate distances only for nearby streets
                nearby_streets = streets_projected.iloc[nearby_indices]
                distances = nearby_streets.geometry.distance(infra_point)
                
                # Adjust indices to match the nearby streets
                if len(distances) == 0:
                    print(f"Warning: No valid distances calculated for infrastructure at {infra_point}")
                    continue
                    
            else:
                # Fallback to full distance calculation (slower)
                distances = streets_projected.geometry.distance(infra_point)
                nearby_streets = streets_projected
            
            if len(distances) == 0 or distances.empty:
                print(f"Warning: No streets found for infrastructure at {infra_point}")
                continue
            
            nearest_street_idx = distances.idxmin()
            
            # Check if the index is valid
            if pd.isna(nearest_street_idx) or nearest_street_idx not in nearby_streets.index:
                print(f"Warning: Invalid nearest street index for infrastructure at {infra_point}")
                continue
                
            nearest_street = nearby_streets.loc[nearest_street_idx]
            nearest_distance = distances.loc[nearest_street_idx]
            
            # Skip if the infrastructure is too far from any street (>2km)
            if nearest_distance > 2000:  # 2km threshold
                print(f"Warning: Infrastructure at {infra_point} is {nearest_distance:.0f}m from nearest street (too far)")
                continue
            
            # Get street information
            street_info = {
                'infrastructure_idx': position_idx,
                'nearest_street_name': nearest_street['Street'],
                'street_class': nearest_street['street_class'],
                'street_width': nearest_street['street_width'],
                'travel_lanes': nearest_street['travel_lanes'],
                'distance_to_street': nearest_distance,
                'influence_distance': nearest_street['influence_distance'],
                'street_geometry': nearest_street.geometry
            }
            
            # Add original infrastructure data
            for col in infrastructure_df.columns:
                street_info[col] = infrastructure_df.loc[idx][col]
            
            enhanced_infrastructure.append(street_info)
        
        enhanced_infra_df = pd.DataFrame(enhanced_infrastructure)
        
        # Now find affected stations using precise geometry
        affected_stations = self._find_precisely_affected_stations(
            enhanced_infra_df, stations_projected, streets_projected
        )
        
        return enhanced_infra_df, affected_stations
    
    def _find_precisely_affected_stations(self, enhanced_infrastructure, stations_projected, streets_projected):
        """Find stations affected by infrastructure using precise geometric calculations."""
        
        affected_station_details = []
        
        print(f"Finding affected stations for {len(enhanced_infrastructure)} infrastructure points...")
        for _, infra in tqdm(enhanced_infrastructure.iterrows(), 
                            total=len(enhanced_infrastructure),
                            desc="Finding affected stations"):
            street_geometry = infra['street_geometry']
            influence_distance = infra['influence_distance']
            
            # Create influence zone around the street segment
            influence_zone = street_geometry.buffer(influence_distance)
            
            # Find stations within influence zone using spatial indexing for efficiency
            try:
                # Use spatial indexing to find stations that might be affected
                station_candidates = stations_projected[stations_projected.intersects(influence_zone)]
                
                for station_idx, station in station_candidates.iterrows():
                    station_point = station.geometry
                    # Calculate exact distance to street centerline
                    exact_distance = street_geometry.distance(station_point)
                    
                    # Calculate position along street (for future analysis)
                    try:
                        nearest_point_on_street = nearest_points(station_point, street_geometry)[1]
                        position_along_street = street_geometry.project(nearest_point_on_street)
                        position_ratio = position_along_street / street_geometry.length
                    except:
                        position_ratio = 0.5  # Default to middle
                    
                    affected_station_details.append({
                        'station_id': station['start_station_id'],
                        'infrastructure_idx': infra['infrastructure_idx'],
                        'exact_distance_to_street': exact_distance,
                        'influence_distance': influence_distance,
                        'position_along_street_ratio': position_ratio,
                        'street_name': infra['nearest_street_name'],
                        'street_class': infra['street_class'],
                        'street_width': infra['street_width'],
                        'travel_lanes': infra['travel_lanes'],
                        'impact_strength': 1.0 - (exact_distance / influence_distance)  # 1.0 = on street, 0.0 = at edge
                    })
                    
            except Exception as e:
                print(f"Warning: Error processing infrastructure point: {e}")
                continue
        
        return pd.DataFrame(affected_station_details)
    
    def create_precise_features(self, stations_df: pd.DataFrame, 
                              enhanced_infrastructure: pd.DataFrame,
                              affected_stations: pd.DataFrame,
                              year: str) -> pd.DataFrame:
        """Create precise geometric features for model training."""
        
        enhanced_stations = stations_df.copy()
        
        # Initialize new precise features
        enhanced_stations['precise_near_infrastructure'] = 0
        enhanced_stations['exact_distance_to_nearest_street'] = 999.0
        enhanced_stations['nearest_street_class'] = 0  # Will be encoded later
        enhanced_stations['nearest_street_width'] = 0.0
        enhanced_stations['nearest_travel_lanes'] = 0
        enhanced_stations['impact_strength'] = 0.0
        enhanced_stations['infrastructure_count_in_influence'] = 0
        
        # Street type features
        enhanced_stations['on_major_street_infra'] = 0
        enhanced_stations['on_arterial_street_infra'] = 0
        enhanced_stations['on_local_street_infra'] = 0
        
        # Protection features for 2023+
        if year != "2022":
            enhanced_stations['protected_infra_impact'] = 0.0
            enhanced_stations['unprotected_infra_impact'] = 0.0
        
        # Create fast lookup dictionary for station impacts - MAJOR OPTIMIZATION
        print(f"   Creating fast lookup for {len(affected_stations)} station impacts...")
        station_impact_dict = {}
        if len(affected_stations) > 0:
            for _, impact in affected_stations.iterrows():
                station_id = impact['station_id']
                if station_id not in station_impact_dict:
                    station_impact_dict[station_id] = []
                station_impact_dict[station_id].append(impact)
        
        # MAJOR OPTIMIZATION: Only process stations that have impacts
        affected_station_ids = set(station_impact_dict.keys())
        stations_to_process = enhanced_stations[enhanced_stations['start_station_id'].isin(affected_station_ids)]
        
        print(f"   Processing features for {len(stations_to_process)} affected stations (out of {len(enhanced_stations)} total)...")
        
        # Process only affected stations
        for station_idx, station in stations_to_process.iterrows():
            station_id = station['start_station_id']
            
            # Fast lookup instead of pandas filtering
            station_impacts_list = station_impact_dict.get(station_id, [])
            
            if len(station_impacts_list) > 0:
                enhanced_stations.loc[station_idx, 'precise_near_infrastructure'] = 1
                enhanced_stations.loc[station_idx, 'infrastructure_count_in_influence'] = len(station_impacts_list)
                
                # Convert list of impacts to DataFrame for processing
                station_impacts = pd.DataFrame(station_impacts_list)
                
                # Get the closest/strongest impact
                strongest_impact_idx = station_impacts['impact_strength'].idxmax()
                strongest_impact = station_impacts.loc[strongest_impact_idx]
                
                # Convert to scalar if it's a Series
                distance_value = strongest_impact['exact_distance_to_street']
                if hasattr(distance_value, 'iloc'):
                    distance_value = distance_value.iloc[0] if len(distance_value) > 0 else 999.0
                
                enhanced_stations.loc[station_idx, 'exact_distance_to_nearest_street'] = distance_value
                
                # Encode street class as numeric
                street_class_encoding = {
                    'service': 1,
                    'local': 2, 
                    'arterial': 3,
                    'major': 4
                }
                # Extract scalar values safely
                street_class = strongest_impact['street_class']
                if hasattr(street_class, 'iloc'):
                    street_class = street_class.iloc[0] if len(street_class) > 0 else 'unknown'
                
                street_width = strongest_impact['street_width']
                if hasattr(street_width, 'iloc'):
                    street_width = street_width.iloc[0] if len(street_width) > 0 else 0.0
                
                travel_lanes = strongest_impact['travel_lanes']
                if hasattr(travel_lanes, 'iloc'):
                    travel_lanes = travel_lanes.iloc[0] if len(travel_lanes) > 0 else 0
                
                impact_strength = strongest_impact['impact_strength']
                if hasattr(impact_strength, 'iloc'):
                    impact_strength = impact_strength.iloc[0] if len(impact_strength) > 0 else 0.0
                
                enhanced_stations.loc[station_idx, 'nearest_street_class'] = street_class_encoding.get(street_class, 0)
                enhanced_stations.loc[station_idx, 'nearest_street_width'] = street_width
                enhanced_stations.loc[station_idx, 'nearest_travel_lanes'] = travel_lanes
                enhanced_stations.loc[station_idx, 'impact_strength'] = impact_strength
                
                # Street type binary features
                if street_class == 'major':
                    enhanced_stations.loc[station_idx, 'on_major_street_infra'] = 1
                elif street_class == 'arterial':
                    enhanced_stations.loc[station_idx, 'on_arterial_street_infra'] = 1
                elif street_class == 'local':
                    enhanced_stations.loc[station_idx, 'on_local_street_infra'] = 1
                
                # Protection impact for 2023+ (weighted by impact strength)
                if year != "2022":
                    protected_impact = 0.0
                    unprotected_impact = 0.0
                    
                    for _, impact in station_impacts.iterrows():
                        infra_idx = impact['infrastructure_idx']
                        # Add bounds checking for iloc access
                        if infra_idx >= len(enhanced_infrastructure):
                            print(f"Warning: Infrastructure index {infra_idx} out of bounds (max: {len(enhanced_infrastructure)-1})")
                            continue
                        infra_info = enhanced_infrastructure.iloc[infra_idx]
                        
                        if not pd.isna(infra_info.get('infrastructure_attributes')):
                            if infra_info['infrastructure_attributes'] == 'protected':
                                protected_impact += impact['impact_strength']
                            elif infra_info['infrastructure_attributes'] == 'unprotected':
                                unprotected_impact += impact['impact_strength']
                    
                    enhanced_stations.loc[station_idx, 'protected_infra_impact'] = protected_impact
                    enhanced_stations.loc[station_idx, 'unprotected_infra_impact'] = unprotected_impact
        
        # Street class is already encoded numerically (service=1, local=2, arterial=3, major=4)
        # No need for one-hot encoding as the numeric encoding preserves the ordinal relationship
        
        return enhanced_stations

# Usage example
if __name__ == "__main__":
    analyzer = PreciseStreetAnalyzer()
    
    # Load street geometries
    streets = analyzer.load_street_geometries()
    
    print(f"\nLoaded {len(streets):,} street geometries")
    print("\nStreet classification summary:")
    print(streets['street_class'].value_counts())
