#!/usr/bin/env python3
"""
Enhanced Street Matcher for Infrastructure Analysis

Uses street names first, then falls back to precise coordinate matching.
Eliminates the need for crude 2km distance thresholds by leveraging exact street names.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
import os
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
from difflib import SequenceMatcher
import re
from functools import lru_cache


class EnhancedStreetMatcher:
    """Enhanced street matching using name-based matching with coordinate fallback."""
    
    def __init__(self, lion_path: str = None):
        if lion_path is None:
            # Create absolute path to LION database
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            lion_path = os.path.join(project_root, "data", "street_locations", "lion.gdb")
        
        self.lion_path = lion_path
        self.street_geometries = None
        self.street_name_index = None
        self.street_geometries_projected = None
        self.spatial_index = None
        self.crs_projected = 'EPSG:2263'  # NAD83 / New York Long Island
        
        # Caching for expensive operations
        self._street_match_cache = {}
        self._coordinate_match_cache = {}
        
        # Street name normalization patterns - expanded for better matching
        self.normalization_patterns = {
            # Common abbreviations
            r'\bST\.?\b': 'STREET',
            r'\bAVE\.?\b': 'AVENUE', 
            r'\bRD\.?\b': 'ROAD',
            r'\bBLVD\.?\b': 'BOULEVARD',
            r'\bPKWY\.?\b': 'PARKWAY',
            r'\bPL\.?\b': 'PLACE',
            r'\bDR\.?\b': 'DRIVE',
            r'\bCT\.?\b': 'COURT',
            r'\bLN\.?\b': 'LANE',
            r'\bPK\.?\b': 'PARK',
            
            # Directions
            r'\bN\.?\b': 'NORTH',
            r'\bS\.?\b': 'SOUTH', 
            r'\bE\.?\b': 'EAST',
            r'\bW\.?\b': 'WEST',
            r'\bNE\.?\b': 'NORTHEAST',
            r'\bNW\.?\b': 'NORTHWEST',
            r'\bSE\.?\b': 'SOUTHEAST',
            r'\bSW\.?\b': 'SOUTHWEST',
            
            # Also handle reverse normalization (full words to abbreviations)
            r'\bSTREET\b': 'ST',
            r'\bAVENUE\b': 'AVE',
            r'\bROAD\b': 'RD',
            r'\bBOULEVARD\b': 'BLVD',
            r'\bPARKWAY\b': 'PKWY',
            r'\bDRIVEWAY\b': 'DRIVEWAY',  # Special case
            
            # Numbers
            r'\b1ST\b': '1',
            r'\b2ND\b': '2',
            r'\b3RD\b': '3',
            r'\bTH\b': '',  # Remove TH from numbers like 4TH -> 4
        }
        
    def load_street_data(self, use_cache: bool = True, cache_file: str = None):
        """Load street geometries with enhanced name indexing."""
        
        if cache_file is None:
            # Create absolute path to cache file
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            cache_file = os.path.join(project_root, "data", "enhanced_street_cache.gpkg")
        
        if use_cache and os.path.exists(cache_file):
            try:
                print("Loading cached enhanced street data...")
                self.street_geometries = gpd.read_file(cache_file)
                self._build_name_index()
                self._prepare_projected_geometries()
                print(f"Loaded {len(self.street_geometries):,} street geometries from cache")
                return self.street_geometries
            except Exception as e:
                print(f"Cache loading failed: {e}, loading fresh data...")
        
        print("Loading NYC LION street geometries with name enhancement...")
        
        # Load with full geometry data
        raw_streets = gpd.read_file(self.lion_path, layer='lion')
        print(f"Loaded {len(raw_streets):,} total segments")
        
        # Filter to actual streets and process
        self.street_geometries = self._process_street_geometries(raw_streets)
        
        # Build name index for fast lookup
        self._build_name_index()
        
        # Cache the processed geometries
        if use_cache:
            try:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                self.street_geometries.to_file(cache_file, driver="GPKG")
                print(f"Cached enhanced street geometries to {cache_file}")
            except Exception as e:
                print(f"Caching failed: {e}")
        
        print(f"Processed {len(self.street_geometries):,} street geometries")
        
        # Prepare projected geometries for coordinate fallback
        self._prepare_projected_geometries()
        
        return self.street_geometries
    
    def _process_street_geometries(self, raw_streets):
        """Process raw LION data with enhanced name processing."""
        
        # Filter to actual streets (FeatureTyp '0' = streets)
        streets = raw_streets[raw_streets['FeatureTyp'] == '0'].copy()
        print(f"Filtered to {len(streets):,} actual street segments")
        
        # Clean and process key columns
        streets['street_width'] = streets[['StreetWidth_Min', 'StreetWidth_Max']].mean(axis=1)
        
        # Handle travel lanes
        travel_lanes_clean = streets['Number_Travel_Lanes'].astype(str).str.strip()
        travel_lanes_clean = travel_lanes_clean.replace(['', 'nan', 'None'], '1')
        travel_lanes_clean = pd.to_numeric(travel_lanes_clean, errors='coerce').fillna(1).astype(int)
        streets['travel_lanes'] = travel_lanes_clean
        
        streets['segment_length'] = streets['SHAPE_Length']
        
        # Handle missing width data
        streets['street_width'] = streets['street_width'].fillna(
            streets['travel_lanes'] * 12 + 18  # ~12ft per lane + 18ft for parking/sidewalks
        )
        
        # Classify streets
        streets['street_class'] = streets.apply(self._classify_street, axis=1)
        
        # Calculate influence zones
        streets['influence_distance'] = streets.apply(self._calculate_influence_distance, axis=1)
        
        # Add normalized street names for matching
        streets['normalized_name'] = streets['Street'].apply(self._normalize_street_name)
        streets['saf_normalized_name'] = streets['SAFStreetName'].apply(self._normalize_street_name)
        
        # Keep essential columns
        essential_cols = [
            'Street', 'SAFStreetName', 'normalized_name', 'saf_normalized_name',
            'SegmentTyp', 'street_width', 'travel_lanes', 'segment_length',
            'street_class', 'influence_distance', 'geometry'
        ]
        
        processed = streets[essential_cols].copy()
        
        # Filter valid geometries
        processed = processed[~processed.geometry.isna()]
        processed = processed[processed.geometry.is_valid]
        
        print(f"After processing: {len(processed)} valid street segments")
        print(f"Street Classification Summary:")
        print(processed['street_class'].value_counts())
        
        return processed
    
    @lru_cache(maxsize=10000)
    def _normalize_street_name(self, name: str) -> str:
        """Normalize street name for consistent matching."""
        if pd.isna(name) or name == '':
            return ''
        
        # Convert to uppercase and clean
        normalized = str(name).upper().strip()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Apply standard abbreviation replacements
        for pattern, replacement in self.normalization_patterns.items():
            normalized = re.sub(pattern, replacement, normalized)
        
        # Remove common prefixes/suffixes that might vary
        normalized = re.sub(r'^(THE\s+)', '', normalized)
        normalized = re.sub(r'\s+(BRIDGE|TUNNEL|HIGHWAY|EXPRESSWAY|FREEWAY)$', '', normalized)
        
        return normalized.strip()
    
    def _build_name_index(self):
        """Build fast lookup index for street names."""
        self.street_name_index = {}
        
        print("Building street name index...")
        for idx, row in self.street_geometries.iterrows():
            # Index both main name and SAF name
            main_name = row['normalized_name']
            saf_name = row['saf_normalized_name']
            
            if main_name:
                if main_name not in self.street_name_index:
                    self.street_name_index[main_name] = []
                self.street_name_index[main_name].append(idx)
            
            if saf_name and saf_name != main_name:
                if saf_name not in self.street_name_index:
                    self.street_name_index[saf_name] = []
                self.street_name_index[saf_name].append(idx)
        
        print(f"Built name index with {len(self.street_name_index)} unique street names")
    
    def _prepare_projected_geometries(self, cache_file: str = None):
        """Prepare projected geometries for coordinate-based fallback."""
        
        if cache_file is None:
            # Create absolute path to cache file
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            cache_file = os.path.join(project_root, "data", "enhanced_streets_projected.gpkg")
        
        if os.path.exists(cache_file):
            try:
                print("Loading cached projected street geometries...")
                self.street_geometries_projected = gpd.read_file(cache_file)
                self.spatial_index = self.street_geometries_projected.sindex
                return
            except Exception as e:
                print(f"Error loading cached projected geometries: {e}")
        
        if self.street_geometries is not None:
            print(f"Converting {len(self.street_geometries):,} street geometries to projected CRS...")
            try:
                self.street_geometries_projected = self.street_geometries.to_crs(self.crs_projected)
                self.spatial_index = self.street_geometries_projected.sindex
                
                # Cache the projected geometries
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                self.street_geometries_projected.to_file(cache_file, driver='GPKG')
                print(f"Cached projected geometries to {cache_file}")
                
            except Exception as e:
                print(f"Error creating projected geometries: {e}")
                self.street_geometries_projected = None
                self.spatial_index = None
    
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
            return 300.0
    
    def find_exact_street_match(self, street_name: str, lat: float = None, lon: float = None, 
                               max_coord_distance: float = 500) -> Optional[pd.Series]:
        """
        Find exact street match using name-first approach with coordinate verification.
        
        Args:
            street_name: Name of the street to match
            lat, lon: Optional coordinates for verification/disambiguation
            max_coord_distance: Maximum distance for coordinate verification (meters)
            
        Returns:
            Best matching street segment or None
        """
        if not street_name or self.street_name_index is None:
            return None
        
        # Create cache key
        cache_key = (street_name, lat, lon, max_coord_distance)
        if cache_key in self._street_match_cache:
            return self._street_match_cache[cache_key]
        
        normalized_name = self._normalize_street_name(street_name)
        
        # Method 1: Exact name match
        exact_matches = self.street_name_index.get(normalized_name, [])
        if exact_matches:
            if len(exact_matches) == 1:
                result = self.street_geometries.loc[exact_matches[0]]
            elif lat is not None and lon is not None:
                # Multiple matches - use coordinates to disambiguate
                result = self._disambiguate_by_coordinates(exact_matches, lat, lon, max_coord_distance)
            else:
                # Return first match if no coordinates provided
                result = self.street_geometries.loc[exact_matches[0]]
            self._street_match_cache[cache_key] = result
            return result
        
        # Method 2: Fuzzy name matching
        fuzzy_match = self._find_fuzzy_match(normalized_name, lat, lon, max_coord_distance)
        if fuzzy_match is not None:
            self._street_match_cache[cache_key] = fuzzy_match
            return fuzzy_match
        
        # Method 3: Coordinate-only fallback (only if coordinates provided)
        if lat is not None and lon is not None:
            result = self._find_nearest_by_coordinates(lat, lon, max_coord_distance)
        else:
            result = None
        
        # Cache the result
        self._street_match_cache[cache_key] = result
        
        # Limit cache size to prevent memory issues
        if len(self._street_match_cache) > 10000:
            # Remove oldest entries (simple FIFO)
            oldest_keys = list(self._street_match_cache.keys())[:1000]
            for key in oldest_keys:
                del self._street_match_cache[key]
        
        return result
    
    def _find_fuzzy_match(self, normalized_name: str, lat: float = None, lon: float = None,
                         max_coord_distance: float = 500, min_similarity: float = 0.6) -> Optional[pd.Series]:
        """Find fuzzy match using string similarity with multiple strategies."""
        
        best_match = None
        best_similarity = 0
        
        # Strategy 1: Try alternative normalizations
        name_variations = [
            normalized_name,
            # Try converting full words to abbreviations
            normalized_name.replace('STREET', 'ST').replace('AVENUE', 'AVE').replace('ROAD', 'RD')
                         .replace('BOULEVARD', 'BLVD').replace('PARKWAY', 'PKWY'),
            # Try converting abbreviations to full words
            normalized_name.replace('ST', 'STREET').replace('AVE', 'AVENUE').replace('RD', 'ROAD')
                         .replace('BLVD', 'BOULEVARD').replace('PKWY', 'PARKWAY'),
            # Try removing direction prefixes
            normalized_name.replace('EAST ', '').replace('WEST ', '').replace('NORTH ', '').replace('SOUTH ', ''),
            # Try adding common direction prefixes
            'EAST ' + normalized_name,
            'WEST ' + normalized_name,
        ]
        
        # Look for fuzzy matches with all variations
        for variation in name_variations:
            if not variation:
                continue
                
            for street_name, indices in self.street_name_index.items():
                similarity = SequenceMatcher(None, variation, street_name).ratio()
                
                if similarity >= min_similarity and similarity > best_similarity:
                    # If coordinates provided, verify the match is geographically reasonable
                    if lat is not None and lon is not None:
                        candidate = self.street_geometries.loc[indices[0]]
                        if self._verify_coordinate_proximity(candidate, lat, lon, max_coord_distance):
                            best_match = candidate
                            best_similarity = similarity
                    else:
                        best_match = self.street_geometries.loc[indices[0]]
                        best_similarity = similarity
        
        return best_match
    
    def _disambiguate_by_coordinates(self, candidate_indices: List[int], lat: float, lon: float,
                                   max_distance: float) -> Optional[pd.Series]:
        """Disambiguate multiple name matches using coordinates."""
        
        point = Point(lon, lat)
        best_match = None
        best_distance = float('inf')
        
        for idx in candidate_indices:
            candidate = self.street_geometries.loc[idx]
            
            # Calculate distance to street centerline
            try:
                # Convert to projected coordinates for accurate distance
                point_proj = gpd.GeoSeries([point], crs='EPSG:4326').to_crs(self.crs_projected).iloc[0]
                street_proj = gpd.GeoSeries([candidate.geometry], crs='EPSG:4326').to_crs(self.crs_projected).iloc[0]
                distance = street_proj.distance(point_proj)
                
                if distance <= max_distance and distance < best_distance:
                    best_match = candidate
                    best_distance = distance
            except Exception as e:
                print(f"Error calculating distance for disambiguation: {e}")
                continue
        
        return best_match
    
    def _verify_coordinate_proximity(self, street_segment: pd.Series, lat: float, lon: float,
                                   max_distance: float) -> bool:
        """Verify that coordinates are reasonably close to the street segment."""
        
        try:
            point = Point(lon, lat)
            point_proj = gpd.GeoSeries([point], crs='EPSG:4326').to_crs(self.crs_projected).iloc[0]
            street_proj = gpd.GeoSeries([street_segment.geometry], crs='EPSG:4326').to_crs(self.crs_projected).iloc[0]
            distance = street_proj.distance(point_proj)
            return distance <= max_distance
        except Exception:
            return False
    
    def _find_nearest_by_coordinates(self, lat: float, lon: float, max_distance: float) -> Optional[pd.Series]:
        """Fallback to find nearest street using coordinates only."""
        
        if self.street_geometries_projected is None or self.spatial_index is None:
            return None
        
        try:
            point = Point(lon, lat)
            point_proj = gpd.GeoSeries([point], crs='EPSG:4326').to_crs(self.crs_projected).iloc[0]
            
            # Use spatial index for efficient search
            nearby_indices = list(self.spatial_index.intersection(point_proj.buffer(max_distance).bounds))
            
            if not nearby_indices:
                return None
            
            # Find the closest street
            nearby_streets = self.street_geometries_projected.iloc[nearby_indices]
            distances = nearby_streets.geometry.distance(point_proj)
            
            if len(distances) == 0 or distances.min() > max_distance:
                return None
            
            closest_idx = distances.idxmin()
            return self.street_geometries.loc[closest_idx]
        
        except Exception as e:
            print(f"Error in coordinate-based matching: {e}")
            return None
    
    def match_infrastructure_to_streets(self, infrastructure_df: pd.DataFrame, 
                                      name_col: str = 'street_name',
                                      lat_col: str = 'latitude', 
                                      lon_col: str = 'longitude',
                                      max_coord_distance: float = 200) -> pd.DataFrame:
        """
        Match infrastructure locations to exact streets using name-first approach.
        
        Args:
            infrastructure_df: DataFrame with infrastructure locations
            name_col: Column containing street names
            lat_col: Column containing latitude
            lon_col: Column containing longitude  
            max_coord_distance: Maximum distance for coordinate verification (meters)
            
        Returns:
            Enhanced infrastructure DataFrame with exact street matches
        """
        
        if self.street_geometries is None:
            self.load_street_data()
        
        print(f"Matching {len(infrastructure_df)} infrastructure locations to streets...")
        
        enhanced_infrastructure = []
        match_stats = {
            'exact_name_matches': 0,
            'fuzzy_name_matches': 0, 
            'coordinate_fallback_matches': 0,
            'no_matches': 0
        }
        
        for idx, row in tqdm(infrastructure_df.iterrows(), total=len(infrastructure_df), 
                           desc="Matching infrastructure to streets"):
            
            street_name = row.get(name_col, '')
            lat = row.get(lat_col, None)
            lon = row.get(lon_col, None)
            
            # Skip if no street name and no coordinates
            if (not street_name or pd.isna(street_name)) and (pd.isna(lat) or pd.isna(lon)):
                match_stats['no_matches'] += 1
                continue
            
            # Find the best street match
            matched_street = self.find_exact_street_match(
                street_name, lat, lon, max_coord_distance
            )
            
            if matched_street is not None:
                # Determine match type for statistics
                normalized_input = self._normalize_street_name(street_name)
                if normalized_input in self.street_name_index:
                    match_stats['exact_name_matches'] += 1
                elif normalized_input:
                    match_stats['fuzzy_name_matches'] += 1
                else:
                    match_stats['coordinate_fallback_matches'] += 1
                
                # Create enhanced record
                enhanced_record = row.to_dict()
                enhanced_record.update({
                    'matched_street_name': matched_street['Street'],
                    'matched_saf_name': matched_street['SAFStreetName'],
                    'street_class': matched_street['street_class'],
                    'street_width': matched_street['street_width'],
                    'travel_lanes': matched_street['travel_lanes'],
                    'influence_distance': matched_street['influence_distance'],
                    'exact_street_geometry': matched_street.geometry,
                    'match_quality': 'exact_name' if normalized_input in self.street_name_index else 'fuzzy_name' if normalized_input else 'coordinate_only'
                })
                enhanced_infrastructure.append(enhanced_record)
            else:
                match_stats['no_matches'] += 1
                # Debug: Store failed matches for analysis
                normalized_input = self._normalize_street_name(street_name)
                if len(enhanced_infrastructure) < 20:  # Limit debug output
                    print(f"    No match for '{street_name}' -> normalized: '{normalized_input}' at ({lat}, {lon})")
        
        enhanced_df = pd.DataFrame(enhanced_infrastructure)
        
        # Print matching statistics
        total_matched = len(enhanced_df)
        total_input = len(infrastructure_df)
        
        print(f"\nStreet Matching Results:")
        print(f"  Total infrastructure locations: {total_input}")
        print(f"  Successfully matched: {total_matched} ({total_matched/total_input*100:.1f}%)")
        print(f"  Exact name matches: {match_stats['exact_name_matches']}")
        print(f"  Fuzzy name matches: {match_stats['fuzzy_name_matches']}")
        print(f"  Coordinate fallback: {match_stats['coordinate_fallback_matches']}")
        print(f"  No matches found: {match_stats['no_matches']}")
        
        return enhanced_df


# Usage example
if __name__ == "__main__":
    matcher = EnhancedStreetMatcher()
    
    # Load street data
    streets = matcher.load_street_data()
    
    print(f"\nLoaded {len(streets):,} street geometries")
    print(f"Built name index with {len(matcher.street_name_index)} unique names")
    
    # Test with sample infrastructure
    test_infrastructure = pd.DataFrame({
        'street_name': ['Broadway', 'Amsterdam Ave', '7th Avenue', 'W 42nd St'],
        'latitude': [40.7831, 40.7891, 40.7500, 40.7590],
        'longitude': [-73.9712, -73.9441, -73.9973, -73.9845],
        'year': [2022, 2022, 2023, 2023]
    })
    
    enhanced = matcher.match_infrastructure_to_streets(test_infrastructure)
    print(f"\nMatched {len(enhanced)} out of {len(test_infrastructure)} test locations")
