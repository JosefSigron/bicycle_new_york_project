#!/usr/bin/env python3
"""
Clean Infrastructure Impact Analysis

A simplified, robust approach to measuring cycling infrastructure impact.
Uses controlled comparison methodology with proper validation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from nyc_street_loader import NYCStreetLoader
from precise_street_analyzer import PreciseStreetAnalyzer

warnings.filterwarnings('ignore')

class CleanInfrastructureAnalyzer:
    """
    Clean Infrastructure Impact Analyzer using controlled comparison.
    
    Methodology:
    1. Train baseline model on 2022 data (no infrastructure features)
    2. Train enhanced model on 2022 data (with infrastructure features)  
    3. Compare predictions on 2023 data
    4. Validate with 2023→2024 comparison
    """
    
    def __init__(self, output_dir: str = None):
        if output_dir is None:
            # Create output directory relative to project root
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            output_dir = os.path.join(project_root, "results", "clean_infrastructure_analysis")
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Data storage
        self.infrastructure_locations = None
        self.affected_stations = []
        
        # Street data loader for enhanced analysis
        self.street_loader = NYCStreetLoader()
        
        # Enhanced geometric analyzer with name-based street matching
        self.precise_analyzer = PreciseStreetAnalyzer()
        self.use_enhanced_matching = True  # Enable name-based matching
        
        # Model results
        self.models = {}
        self.predictions = {}
        self.results = {}
        
        # Feature caching to avoid recalculation
        self._cached_features = {}
        self._feature_cache_keys = set()
    
    def load_data(self, citibike_2022_path: str, citibike_2023_path: str, 
                  citibike_2024_path: str, infrastructure_path: str):
        """Load and prepare all required data."""
        print("Setting up data paths...")
        
        # Store file paths for lazy loading
        self.citibike_2022_path = citibike_2022_path
        self.citibike_2023_path = citibike_2023_path
        self.citibike_2024_path = citibike_2024_path
        
        # Load infrastructure data (small file)
        self.infrastructure_locations = pd.read_csv(infrastructure_path)
        print(f"   Total infrastructure locations: {len(self.infrastructure_locations)}")
        
        # Filter infrastructure by year
        self.infrastructure_2022 = self.infrastructure_locations[self.infrastructure_locations['year'] == 2022].copy()
        self.infrastructure_2023 = self.infrastructure_locations[self.infrastructure_locations['year'] == 2023].copy()
        
        print(f"   Infrastructure 2022: {len(self.infrastructure_2022)} locations")
        print(f"   Infrastructure 2023: {len(self.infrastructure_2023)} locations")
        
        # Load 2022 data to identify affected stations
        print("   Loading 2022 data to identify affected stations...")
        self.data_2022 = self._load_and_preprocess_single_year("2022", citibike_2022_path)
        
        # Identify affected stations based on appropriate infrastructure for each year
        self._identify_affected_stations()
        
        print("Data setup complete")
    
    def _load_and_preprocess_single_year(self, year: str, file_path: str):
        """Load and preprocess data for a single year."""
        print(f"   Loading {year} data...")
        data = pd.read_parquet(file_path)
        print(f"   Loaded {year}: {len(data):,} rides")
        
        # Convert datetime columns
        data['start_time'] = pd.to_datetime(data['start_time'])
        data['stop_time'] = pd.to_datetime(data['stop_time'])
        
        # Extract temporal features
        data['hour'] = data['start_time'].dt.hour
        data['day_of_week'] = data['start_time'].dt.dayofweek
        data['month'] = data['start_time'].dt.month
        data['year_month'] = data['start_time'].dt.to_period('M')
        
        print(f"   Preprocessed {year}: {len(data):,} rides")
        return data
    
    def _identify_affected_stations(self):
        """Identify stations affected by infrastructure using precise geometric analysis."""
        print("Identifying affected stations using precise street geometry...")
        
        # Get unique stations from 2022 (baseline year)
        stations_2022 = self.data_2022[['start_station_id', 'start_station_latitude', 'start_station_longitude']].drop_duplicates()
        stations_2022 = stations_2022.dropna()
        
        # Load precise street geometries ONCE
        print("   Loading precise street geometries...")
        self.precise_analyzer.load_street_geometries()
        
        # Process all infrastructure at once to avoid redundant calculations
        print("   Analyzing enhanced geometric impact for all infrastructure...")
        all_infrastructure = pd.concat([self.infrastructure_2022, self.infrastructure_2023], ignore_index=True)
        all_infrastructure['year_group'] = all_infrastructure['year']  # Track which year each belongs to
        
        if self.use_enhanced_matching:
            all_infrastructure_enhanced, all_affected_stations_details = self.precise_analyzer.find_enhanced_infrastructure_impact(
                all_infrastructure, stations_2022, use_name_matching=True)
        else:
            all_infrastructure_enhanced, all_affected_stations_details = self.precise_analyzer.find_precise_infrastructure_impact(
                all_infrastructure, stations_2022)
        
        # Split results back by year
        infra_2022_mask = all_infrastructure_enhanced['year'] == 2022
        infra_2023_mask = all_infrastructure_enhanced['year'] == 2023
        
        self.infrastructure_2022_enhanced = all_infrastructure_enhanced[infra_2022_mask].copy()
        self.infrastructure_2023_enhanced = all_infrastructure_enhanced[infra_2023_mask].copy()
        
        # Split affected stations details by matching infrastructure years
        if len(all_affected_stations_details) > 0:
            # Get infrastructure indices for each year
            infra_2022_indices = all_infrastructure_enhanced[infra_2022_mask].index.tolist()
            infra_2023_indices = all_infrastructure_enhanced[infra_2023_mask].index.tolist()
            
            # Split affected stations based on infrastructure they're affected by
            self.affected_stations_2022_details = all_affected_stations_details[
                all_affected_stations_details['infrastructure_idx'].isin(infra_2022_indices)
            ].copy()
            self.affected_stations_2023_details = all_affected_stations_details[
                all_affected_stations_details['infrastructure_idx'].isin(infra_2023_indices)
            ].copy()
        else:
            self.affected_stations_2022_details = pd.DataFrame()
            self.affected_stations_2023_details = pd.DataFrame()
        
        # Get affected station IDs (for compatibility with existing code)
        if len(self.affected_stations_2022_details) > 0 and 'station_id' in self.affected_stations_2022_details.columns:
            self.affected_stations_2022 = list(self.affected_stations_2022_details['station_id'].unique())
        else:
            self.affected_stations_2022 = []
            print("   Warning: No affected stations found for 2022 infrastructure")
        
        # For 2023+: combine both years' impacts
        if len(self.affected_stations_2023_details) > 0 and 'station_id' in self.affected_stations_2023_details.columns:
            if len(self.affected_stations_2022_details) > 0:
                combined_affected_details = pd.concat([self.affected_stations_2022_details, self.affected_stations_2023_details])
            else:
                combined_affected_details = self.affected_stations_2023_details
            self.affected_stations_2023_plus = list(combined_affected_details['station_id'].unique())
        else:
            self.affected_stations_2023_plus = self.affected_stations_2022.copy()
            print("   Warning: No affected stations found for 2023 infrastructure, using 2022 stations only")
        
        print(f"   2022 affected stations (precise geometry): {len(self.affected_stations_2022)}")
        print(f"   2023+ affected stations (precise geometry): {len(self.affected_stations_2023_plus)}")
        print(f"   Total unique stations: {len(stations_2022)}")
        
        # Store detailed impact data for enhanced features
        self.impact_details_2022 = self.affected_stations_2022_details if len(self.affected_stations_2022_details) > 0 else pd.DataFrame()
        
        # Handle empty combined details case
        if len(self.affected_stations_2023_details) > 0 and 'station_id' in self.affected_stations_2023_details.columns:
            if len(self.affected_stations_2022_details) > 0:
                self.impact_details_2023_plus = pd.concat([self.affected_stations_2022_details, self.affected_stations_2023_details])
            else:
                self.impact_details_2023_plus = self.affected_stations_2023_details
        else:
            self.impact_details_2023_plus = self.impact_details_2022.copy()
    
    def _find_affected_stations_for_infrastructure(self, stations, infrastructure, buffer_degrees):
        """Find stations affected by given infrastructure with fixed buffer."""
        affected_stations = set()
        
        for _, infra in infrastructure.iterrows():
            if pd.isna(infra['latitude']) or pd.isna(infra['longitude']):
                continue
                
            # Find stations within buffer
            lat_min = infra['latitude'] - buffer_degrees
            lat_max = infra['latitude'] + buffer_degrees
            lon_min = infra['longitude'] - buffer_degrees
            lon_max = infra['longitude'] + buffer_degrees
            
            nearby_stations = stations[
                (stations['start_station_latitude'] >= lat_min) &
                (stations['start_station_latitude'] <= lat_max) &
                (stations['start_station_longitude'] >= lon_min) &
                (stations['start_station_longitude'] <= lon_max)
            ]
            
            affected_stations.update(nearby_stations['start_station_id'].tolist())
        
        return list(affected_stations)
    
    def _find_affected_stations_with_street_buffers(self, stations, infrastructure):
        """Find stations affected by infrastructure using street-specific buffer sizes."""
        affected_stations = set()
        
        for _, infra in infrastructure.iterrows():
            if pd.isna(infra['latitude']) or pd.isna(infra['longitude']):
                continue
                
            # Use street-specific buffer size
            buffer_degrees = infra.get('buffer_degrees', 0.0045)  # Default fallback
            
            # Find stations within adaptive buffer
            lat_min = infra['latitude'] - buffer_degrees
            lat_max = infra['latitude'] + buffer_degrees
            lon_min = infra['longitude'] - buffer_degrees
            lon_max = infra['longitude'] + buffer_degrees
            
            nearby_stations = stations[
                (stations['start_station_latitude'] >= lat_min) &
                (stations['start_station_latitude'] <= lat_max) &
                (stations['start_station_longitude'] >= lon_min) &
                (stations['start_station_longitude'] <= lon_max)
            ]
            
            affected_stations.update(nearby_stations['start_station_id'].tolist())
        
        return list(affected_stations)
    
    def _aggregate_monthly_data(self, data: pd.DataFrame, year: str) -> pd.DataFrame:
        """Aggregate data by station and month."""
        print(f"Aggregating {year} data by station-month...")
        
        # Standard aggregation for model training and infrastructure analysis
        monthly_data = data.groupby(['start_station_id', 'year_month']).agg({
            'trip_duration': 'count',  # Count of rides
            'start_station_latitude': 'first',
            'start_station_longitude': 'first',
            'temperature': 'mean',
            'relative_humidity': 'mean',
            'wind_speed': 'mean',
            'precipitation': 'mean',
            'weather_cat': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
            'utci_cat': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
        }).reset_index()
        
        # Add user type aggregation for separate analysis
        if 'user_type' in data.columns:
            user_type_data = data.groupby(['start_station_id', 'year_month', 'user_type']).agg({
                'trip_duration': 'count'  # Count rides by user type
            }).reset_index()
            user_type_pivot = user_type_data.pivot_table(
                index=['start_station_id', 'year_month'], 
                columns='user_type', 
                values='trip_duration', 
                fill_value=0
            ).reset_index()
            user_type_pivot.columns.name = None
            
            # Merge user type data with monthly data
            monthly_data = monthly_data.merge(user_type_pivot, on=['start_station_id', 'year_month'], how='left')
            
            # Standardize column names (map lowercase to capitalized)
            for col in monthly_data.columns:
                if col.lower() == 'member':
                    monthly_data = monthly_data.rename(columns={col: 'Member'})
                elif col.lower() == 'casual':
                    monthly_data = monthly_data.rename(columns={col: 'Casual'})
                elif col.lower() == 'subscriber':
                    monthly_data = monthly_data.rename(columns={col: 'Member'})
                elif col.lower() == 'customer':
                    monthly_data = monthly_data.rename(columns={col: 'Casual'})
            
            # Fill NaN values with 0 for user types
            for col in ['Member', 'Casual']:
                if col in monthly_data.columns:
                    monthly_data[col] = monthly_data[col].fillna(0)
        
        # PRESERVE DETAILED WEATHER DATA for weather resilience analysis
        weather_details = data.groupby(['start_station_id', 'year_month']).apply(
            lambda group: pd.Series({
                'weather_detail': group['weather_cat'].value_counts().to_dict(),
                'utci_detail': group['utci_cat'].value_counts().to_dict(),
                'total_rides_detailed': len(group)
            })
        ).reset_index()
        
        # Rename count column
        base_columns = [
            'start_station_id', 'year_month', 'monthly_rides',
            'start_station_latitude', 'start_station_longitude',
            'avg_temp', 'avg_humidity', 'avg_wind_speed', 'avg_precipitation',
            'weather_cat', 'utci_cat'
        ]
        
        # Handle dynamic column renaming based on what's actually present
        current_columns = monthly_data.columns.tolist()
        if len(current_columns) == len(base_columns):
            monthly_data.columns = base_columns
        else:
            # More columns present (likely user type data), rename base columns only
            monthly_data = monthly_data.rename(columns={
                current_columns[0]: 'start_station_id',
                current_columns[1]: 'year_month', 
                current_columns[2]: 'monthly_rides',
                current_columns[3]: 'start_station_latitude',
                current_columns[4]: 'start_station_longitude',
                current_columns[5]: 'avg_temp',
                current_columns[6]: 'avg_humidity',
                current_columns[7]: 'avg_wind_speed',
                current_columns[8]: 'avg_precipitation',
                current_columns[9]: 'weather_cat',
                current_columns[10]: 'utci_cat'
            })
        
        # Extract month number
        monthly_data['month'] = monthly_data['year_month'].dt.month
        
        # One-hot encode categorical variables (for model training)
        weather_dummies = pd.get_dummies(monthly_data['weather_cat'], prefix='weather')
        utci_dummies = pd.get_dummies(monthly_data['utci_cat'], prefix='utci')
        
        # Combine standard aggregation with detailed weather data
        result = pd.concat([monthly_data, weather_dummies, utci_dummies], axis=1)
        result = result.merge(weather_details, on=['start_station_id', 'year_month'], how='left')
        
        print(f"   Result: {len(result):,} station-month records")
        print(f"   Weather detail preservation: Complete")
        return result
    
    def _add_infrastructure_features(self, data: pd.DataFrame, year: str) -> pd.DataFrame:
        """Add precise infrastructure features based on exact geometric analysis."""
        
        # Create cache key for this specific data and year combination
        cache_key = f"{year}_{len(data)}_{hash(tuple(data['start_station_id'].values))}"
        
        # Check if we already computed features for this exact data
        if cache_key in self._cached_features:
            print(f"   Using cached precise geometric features for {year}...")
            cached_features = self._cached_features[cache_key]
            
            # Merge cached features with current data
            enhanced_data = data.copy()
            for feature_col in cached_features.columns:
                if feature_col not in enhanced_data.columns:
                    enhanced_data[feature_col] = 0  # Initialize
            
            # Map features by station ID
            feature_dict = cached_features.set_index('start_station_id').to_dict('index')
            for idx, row in enhanced_data.iterrows():
                station_id = row['start_station_id']
                if station_id in feature_dict:
                    for feature, value in feature_dict[station_id].items():
                        enhanced_data.loc[idx, feature] = value
        else:
            # Choose correct infrastructure and impact details for the year
            if year == "2022":
                infrastructure_enhanced = self.infrastructure_2022_enhanced
                impact_details = self.impact_details_2022
            else:  # 2023, 2024
                # Use pre-combined infrastructure to avoid repeated concatenation
                if not hasattr(self, '_combined_infrastructure_enhanced'):
                    self._combined_infrastructure_enhanced = pd.concat([self.infrastructure_2022_enhanced, self.infrastructure_2023_enhanced])
                infrastructure_enhanced = self._combined_infrastructure_enhanced
                impact_details = self.impact_details_2023_plus
            
            # Create precise features using the geometric analyzer
            print(f"   Creating precise geometric features for {year}...")
            enhanced_data = self.precise_analyzer.create_precise_features(
                data, infrastructure_enhanced, impact_details, year
            )
            
            # Cache the computed features (only infrastructure-related columns)
            infrastructure_columns = [
                'start_station_id', 'precise_near_infrastructure', 'exact_distance_to_nearest_street',
                'nearest_street_class', 'nearest_street_width', 'nearest_travel_lanes', 'impact_strength',
                'infrastructure_count_in_influence', 'on_major_street_infra', 'on_arterial_street_infra',
                'on_local_street_infra'
            ]
            if year != "2022":
                infrastructure_columns.extend(['protected_infra_impact', 'unprotected_infra_impact'])
            
            # Only cache columns that exist
            cache_columns = [col for col in infrastructure_columns if col in enhanced_data.columns]
            self._cached_features[cache_key] = enhanced_data[cache_columns].copy()
            print(f"   Cached features for future use (cache size: {len(self._cached_features)})")
        
        # Legacy feature mapping for compatibility with existing model training
        enhanced_data['near_infrastructure'] = enhanced_data['precise_near_infrastructure']
        enhanced_data['min_infra_distance'] = enhanced_data['exact_distance_to_nearest_street']
        
        # Map new precise features to existing feature names for model compatibility
        enhanced_data['near_major_street_infra'] = enhanced_data['on_major_street_infra']
        enhanced_data['near_arterial_street_infra'] = enhanced_data['on_arterial_street_infra'] 
        enhanced_data['near_local_street_infra'] = enhanced_data['on_local_street_infra']
        enhanced_data['avg_street_width'] = enhanced_data['nearest_street_width']
        enhanced_data['max_travel_lanes'] = enhanced_data['nearest_travel_lanes']
        
        # Protection features for 2023+
        if year != "2022":
            enhanced_data['near_protected_infra'] = (enhanced_data['protected_infra_impact'] > 0).astype(int)
            enhanced_data['near_unprotected_infra'] = (enhanced_data['unprotected_infra_impact'] > 0).astype(int)
            
            # Calculate protection ratio from impact strengths
            total_protection_impact = enhanced_data['protected_infra_impact'] + enhanced_data['unprotected_infra_impact']
            enhanced_data['protected_infra_ratio'] = np.where(
                total_protection_impact > 0,
                enhanced_data['protected_infra_impact'] / total_protection_impact,
                0.0
            )
        else:
            enhanced_data['near_protected_infra'] = 0
            enhanced_data['near_unprotected_infra'] = 0  
            enhanced_data['protected_infra_ratio'] = 0.0
        
        print(f"   Added precise geometric features. Affected stations: {enhanced_data['near_infrastructure'].sum()}")
        
        return enhanced_data
    
    def run_analysis(self):
        """Run the complete infrastructure impact analysis."""
        print("\n" + "="*70)
        print("CLEAN INFRASTRUCTURE IMPACT ANALYSIS")
        print("="*70)
        
        # Step 1: Prepare training data (2022)
        print("\n1. Preparing 2022 training data...")
        train_data = self._aggregate_monthly_data(self.data_2022, "2022")
        
        # Step 2: Train baseline model (no infrastructure features)
        print("\n2. Training baseline model...")
        baseline_model, baseline_features = self._train_baseline_model(train_data.copy())
        
        # Step 3: Train enhanced model (with infrastructure features)
        print("\n3. Training enhanced model...")
        enhanced_model, enhanced_features = self._train_enhanced_model(train_data.copy())
        
        # Free training data memory
        del train_data
        del self.data_2022
        
        # Step 4: Test on 2023 data
        print("\n4. Testing on 2023 data...")
        data_2023 = self._load_and_preprocess_single_year("2023", self.citibike_2023_path)
        test_data_2023 = self._aggregate_monthly_data(data_2023, "2023")
        del data_2023  # Free memory
        results_2023 = self._compare_predictions(
            baseline_model, enhanced_model, 
            baseline_features, enhanced_features, 
            test_data_2023, "2023"
        )
        
        # Step 5: Validate on 2024 data
        print("\n5. Validating on 2024 data...")
        data_2024 = self._load_and_preprocess_single_year("2024", self.citibike_2024_path)
        test_data_2024 = self._aggregate_monthly_data(data_2024, "2024")
        del data_2024  # Free memory
        results_2024 = self._compare_predictions(
            baseline_model, enhanced_model,
            baseline_features, enhanced_features,
            test_data_2024, "2024"
        )
        
        # Step 6: Weather Resilience Analysis
        print("\n6. Analyzing weather resilience...")
        weather_results = self._analyze_weather_resilience(test_data_2023, test_data_2024)
        
        # Step 7: Network Effects and Spillover Analysis
        print("\n7. Analyzing network effects and spillover...")
        network_results = self._analyze_network_spillover_effects(test_data_2023, test_data_2024)
        
        # Step 8: Seasonal Validation
        print("\n8. Performing seasonal validation...")
        seasonal_results = self._perform_seasonal_validation(baseline_model, enhanced_model, 
                                                           baseline_features, enhanced_features,
                                                           test_data_2023, test_data_2024)
        
        # Step 9: Analyze infrastructure impact
        print("\n9. Analyzing infrastructure impact...")
        final_results = self._analyze_infrastructure_impact(results_2023, results_2024)
        
        # Step 10: Create comprehensive visualizations
        print("\n10. Creating comprehensive visualizations...")
        self._create_comprehensive_visualizations(results_2023, results_2024, final_results, 
                                                weather_results, network_results, seasonal_results)
        
        print("\nAnalysis complete!")
        
        # Combine all results
        comprehensive_results = {
            **final_results,
            'weather_resilience': weather_results,
            'network_effects': network_results,
            'seasonal_validation': seasonal_results
        }
        
        return comprehensive_results
    
    def _train_baseline_model(self, train_data: pd.DataFrame):
        """Train baseline model without infrastructure features."""
        # Exclude dictionary columns from model training (they're only for weather analysis)
        dict_columns = ['weather_detail', 'utci_detail', 'total_rides_detailed']
        feature_columns = [col for col in train_data.columns if col not in dict_columns]
        model_data = train_data[feature_columns]
        
        # Define baseline features (exclude original categorical columns)
        baseline_features = [
            'month', 'start_station_latitude', 'start_station_longitude',
            'avg_temp', 'avg_humidity', 'avg_wind_speed', 'avg_precipitation'
        ]
        
        # Add only one-hot encoded weather features (exclude original categorical columns)
        weather_cols = [col for col in model_data.columns if col.startswith('weather_') and col != 'weather_cat']
        utci_cols = [col for col in model_data.columns if col.startswith('utci_') and col != 'utci_cat']
        baseline_features.extend(weather_cols)
        baseline_features.extend(utci_cols)
        
        # Add user type features if available
        if 'Member' in model_data.columns and 'Casual' in model_data.columns:
            baseline_features.extend(['member_ratio', 'casual_ratio'])
            # Calculate user type ratios
            model_data['total_user_rides'] = model_data['Member'] + model_data['Casual']
            model_data['member_ratio'] = model_data['Member'] / (model_data['total_user_rides'] + 1e-6)  # Avoid division by zero
            model_data['casual_ratio'] = model_data['Casual'] / (model_data['total_user_rides'] + 1e-6)
        
        print(f"   Using {len(baseline_features)} baseline features")
        
        # Prepare data
        X_train = model_data[baseline_features].fillna(0)
        y_train = train_data['monthly_rides']
        
        # Train model
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = model.predict(X_train)
        train_r2 = r2_score(y_train, train_pred)
        train_mae = mean_absolute_error(y_train, train_pred)
        
        print(f"   Baseline model R²: {train_r2:.3f}, MAE: {train_mae:.1f}")
        
        return model, baseline_features
    
    def _train_enhanced_model(self, train_data: pd.DataFrame):
        """Train enhanced model with infrastructure features."""
        # Add infrastructure features for 2022 (training year)
        enhanced_data = self._add_infrastructure_features(train_data, "2022")
        
        # Exclude dictionary columns from model training (they're only for weather analysis)
        dict_columns = ['weather_detail', 'utci_detail', 'total_rides_detailed']
        feature_columns = [col for col in enhanced_data.columns if col not in dict_columns]
        enhanced_data = enhanced_data[feature_columns]
        
        # Define enhanced features (baseline + infrastructure)
        baseline_features = [
            'month', 'start_station_latitude', 'start_station_longitude',
            'avg_temp', 'avg_humidity', 'avg_wind_speed', 'avg_precipitation'
        ]
        
        # Add only one-hot encoded weather features (exclude original categorical columns)
        weather_cols = [col for col in enhanced_data.columns if col.startswith('weather_') and col != 'weather_cat']
        utci_cols = [col for col in enhanced_data.columns if col.startswith('utci_') and col != 'utci_cat']
        
        # Add user type features if available
        user_type_features = []
        if 'Member' in enhanced_data.columns and 'Casual' in enhanced_data.columns:
            user_type_features = ['member_ratio', 'casual_ratio']
            # Calculate user type ratios
            enhanced_data['total_user_rides'] = enhanced_data['Member'] + enhanced_data['Casual']
            enhanced_data['member_ratio'] = enhanced_data['Member'] / (enhanced_data['total_user_rides'] + 1e-6)
            enhanced_data['casual_ratio'] = enhanced_data['Casual'] / (enhanced_data['total_user_rides'] + 1e-6)
        
        # Infrastructure features - basic for all years
        infrastructure_features = ['near_infrastructure', 'min_infra_distance']
        
        # Add protection features that exist in the data (they're 0 for 2022)
        if 'near_protected_infra' in enhanced_data.columns:
            infrastructure_features.extend(['near_protected_infra', 'near_unprotected_infra', 'protected_infra_ratio'])
        
        # Add street classification features
        street_features = []
        if 'near_major_street_infra' in enhanced_data.columns:
            street_features = [
                'near_major_street_infra', 'near_arterial_street_infra', 'near_local_street_infra',
                'avg_street_width', 'max_travel_lanes'
            ]
        
        # Add precise geometric features
        precise_features = []
        if 'impact_strength' in enhanced_data.columns:
            precise_features = [
                'impact_strength', 'infrastructure_count_in_influence'
            ]
            # Add one-hot encoded street class features if they exist
            street_class_cols = [col for col in enhanced_data.columns if col.startswith('nearest_street_')]
            precise_features.extend(street_class_cols)
        
        enhanced_features = baseline_features + weather_cols + utci_cols + user_type_features + infrastructure_features + street_features + precise_features
        
        print(f"   Using {len(enhanced_features)} enhanced features")
        
        # Prepare data
        X_train = enhanced_data[enhanced_features].fillna(0)
        y_train = enhanced_data['monthly_rides']
        
        # Train model
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = model.predict(X_train)
        train_r2 = r2_score(y_train, train_pred)
        train_mae = mean_absolute_error(y_train, train_pred)
        
        print(f"   Enhanced model R²: {train_r2:.3f}, MAE: {train_mae:.1f}")
        
        return model, enhanced_features
    
    def _compare_predictions(self, baseline_model, enhanced_model, 
                           baseline_features, enhanced_features, test_data, year):
        """Compare baseline vs enhanced model predictions."""
        print(f"   Comparing predictions for {year}...")
        
        # Choose correct affected stations for the year
        affected_stations = self.affected_stations_2022 if year == "2022" else self.affected_stations_2023_plus
        
        # Exclude dictionary columns from test data (they're only for weather analysis)
        dict_columns = ['weather_detail', 'utci_detail', 'total_rides_detailed']
        feature_columns = [col for col in test_data.columns if col not in dict_columns]
        clean_test_data = test_data[feature_columns]
        
        # Add infrastructure features to test data
        enhanced_test_data = self._add_infrastructure_features(clean_test_data, year)
        
        # Add user type features if available for both baseline and enhanced
        if 'Member' in clean_test_data.columns and 'Casual' in clean_test_data.columns:
            # For baseline test data
            clean_test_data['total_user_rides'] = clean_test_data['Member'] + clean_test_data['Casual']
            clean_test_data['member_ratio'] = clean_test_data['Member'] / (clean_test_data['total_user_rides'] + 1e-6)
            clean_test_data['casual_ratio'] = clean_test_data['Casual'] / (clean_test_data['total_user_rides'] + 1e-6)
            
            # For enhanced test data (copy the calculated ratios)
            enhanced_test_data['total_user_rides'] = enhanced_test_data['Member'] + enhanced_test_data['Casual']
            enhanced_test_data['member_ratio'] = enhanced_test_data['Member'] / (enhanced_test_data['total_user_rides'] + 1e-6)
            enhanced_test_data['casual_ratio'] = enhanced_test_data['Casual'] / (enhanced_test_data['total_user_rides'] + 1e-6)
        
        # Align features for both models
        baseline_test = clean_test_data[baseline_features].fillna(0)
        enhanced_test = enhanced_test_data[enhanced_features].fillna(0)
        
        # Make predictions
        baseline_pred = baseline_model.predict(baseline_test)
        enhanced_pred = enhanced_model.predict(enhanced_test)
        actual = test_data['monthly_rides']
        
        # Calculate performance metrics
        baseline_r2 = r2_score(actual, baseline_pred)
        enhanced_r2 = r2_score(actual, enhanced_pred)
        baseline_mae = mean_absolute_error(actual, baseline_pred)
        enhanced_mae = mean_absolute_error(actual, enhanced_pred)
        
        print(f"     Baseline - R²: {baseline_r2:.3f}, MAE: {baseline_mae:.1f}")
        print(f"     Enhanced - R²: {enhanced_r2:.3f}, MAE: {enhanced_mae:.1f}")
        print(f"     Improvement: {enhanced_r2 - baseline_r2:.3f} R², {baseline_mae - enhanced_mae:.1f} MAE")
        
        # Analyze by affected vs unaffected stations (using correct year's affected stations)
        results_df = test_data.copy()
        results_df['baseline_pred'] = baseline_pred
        results_df['enhanced_pred'] = enhanced_pred
        results_df['is_affected'] = results_df['start_station_id'].isin(affected_stations)
        
        # Calculate gaps (actual - predicted)
        results_df['baseline_gap'] = results_df['monthly_rides'] - results_df['baseline_pred']
        results_df['enhanced_gap'] = results_df['monthly_rides'] - results_df['enhanced_pred']
        
        # Infrastructure impact analysis
        affected_baseline_gap = results_df[results_df['is_affected']]['baseline_gap'].mean()
        unaffected_baseline_gap = results_df[~results_df['is_affected']]['baseline_gap'].mean()
        affected_enhanced_gap = results_df[results_df['is_affected']]['enhanced_gap'].mean()
        unaffected_enhanced_gap = results_df[~results_df['is_affected']]['enhanced_gap'].mean()
        
        infrastructure_effect_baseline = affected_baseline_gap - unaffected_baseline_gap
        infrastructure_effect_enhanced = affected_enhanced_gap - unaffected_enhanced_gap
        
        print(f"     Using {len(affected_stations)} affected stations for {year}")
        print(f"     Infrastructure effect (baseline model): {infrastructure_effect_baseline:.1f} rides/month")
        print(f"     Infrastructure effect (enhanced model): {infrastructure_effect_enhanced:.1f} rides/month")
        
        return {
            'year': year,
            'baseline_r2': baseline_r2,
            'enhanced_r2': enhanced_r2,
            'baseline_mae': baseline_mae,
            'enhanced_mae': enhanced_mae,
            'results_df': results_df,
            'infrastructure_effect_baseline': infrastructure_effect_baseline,
            'infrastructure_effect_enhanced': infrastructure_effect_enhanced,
            'affected_baseline_gap': affected_baseline_gap,
            'unaffected_baseline_gap': unaffected_baseline_gap,
            'affected_enhanced_gap': affected_enhanced_gap,
            'unaffected_enhanced_gap': unaffected_enhanced_gap
        }
    
    def _analyze_infrastructure_impact(self, results_2023, results_2024):
        """Analyze overall infrastructure impact."""
        
        # Primary metric: Infrastructure effect detected by baseline model
        # (This shows what the model WITHOUT infrastructure knowledge detects)
        effect_2023 = results_2023['infrastructure_effect_baseline']
        effect_2024 = results_2024['infrastructure_effect_baseline']
        
        # Model improvement: How much better does the enhanced model perform?
        model_improvement_2023 = results_2023['enhanced_r2'] - results_2023['baseline_r2']
        model_improvement_2024 = results_2024['enhanced_r2'] - results_2024['baseline_r2']
        
        # Calculate consistency
        effect_consistency = 1 - abs(effect_2023 - effect_2024) / (abs(effect_2023) + abs(effect_2024) + 1e-6)
        
        results = {
            'infrastructure_effect_2023': effect_2023,
            'infrastructure_effect_2024': effect_2024,
            'average_infrastructure_effect': (effect_2023 + effect_2024) / 2,
            'effect_consistency': effect_consistency,
            'model_improvement_2023': model_improvement_2023,
            'model_improvement_2024': model_improvement_2024,
            'average_model_improvement': (model_improvement_2023 + model_improvement_2024) / 2,
            'affected_stations_2022': len(self.affected_stations_2022),
            'affected_stations_2023_plus': len(self.affected_stations_2023_plus),
            'total_stations_analyzed': len(results_2023['results_df']['start_station_id'].unique())
        }
        
        print(f"\nINFRASTRUCTURE IMPACT SUMMARY:")
        print(f"   2023 Effect: {effect_2023:.1f} rides/month")
        print(f"   2024 Effect: {effect_2024:.1f} rides/month")
        print(f"   Average Effect: {results['average_infrastructure_effect']:.1f} rides/month")
        print(f"   Effect Consistency: {effect_consistency:.3f}")
        print(f"   Model Improvement (2023): {model_improvement_2023:.3f} R²")
        print(f"   Model Improvement (2024): {model_improvement_2024:.3f} R²")
        print(f"   Affected Stations (2022): {len(self.affected_stations_2022)}")
        print(f"   Affected Stations (2023+): {len(self.affected_stations_2023_plus)}")
        
        return results
    
    def _create_visualizations(self, results_2023, results_2024, final_results):
        """Create individual visualizations as separate PNG files."""
        
        # Set up the plotting style
        plt.style.use('default')
        
        # Create individual plots (10 total)
        self._plot_individual_r2_comparison(results_2023, results_2024)
        self._plot_individual_mae_comparison(results_2023, results_2024)
        self._plot_individual_infrastructure_effects_by_year(results_2023, results_2024)
        self._plot_individual_effect_consistency(final_results)
        self._plot_individual_2023_affected_gaps(results_2023)
        self._plot_individual_2023_unaffected_gaps(results_2023)
        self._plot_individual_2024_affected_gaps(results_2024)
        self._plot_individual_2024_unaffected_gaps(results_2024)
        self._plot_individual_2023_comparison_gaps(results_2023)
        self._plot_individual_2024_comparison_gaps(results_2024)
        
        print(f"   Created 10 individual visualization files in: {self.output_dir}")
        print("     Individual plot files:")
        print("     1. r2_comparison.png")
        print("     2. mae_comparison.png") 
        print("     3. infrastructure_effects_by_year.png")
        print("     4. effect_consistency.png")
        print("     5. gaps_2023_affected.png")
        print("     6. gaps_2023_unaffected.png")
        print("     7. gaps_2024_affected.png")
        print("     8. gaps_2024_unaffected.png")
        print("     9. gaps_comparison_2023.png")
        print("     10. gaps_comparison_2024.png")
    
    def _plot_individual_r2_comparison(self, results_2023, results_2024):
        """Plot R² comparison as individual file."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        years = ['2023', '2024']
        baseline_r2 = [results_2023['baseline_r2'], results_2024['baseline_r2']]
        enhanced_r2 = [results_2023['enhanced_r2'], results_2024['enhanced_r2']]
        
        x = np.arange(len(years))
        width = 0.35
        
        ax.bar(x - width/2, baseline_r2, width, label='Baseline Model', alpha=0.8, color='skyblue')
        ax.bar(x + width/2, enhanced_r2, width, label='Enhanced Model', alpha=0.8, color='lightcoral')
        
        ax.set_xlabel('Test Year')
        ax.set_ylabel('R² Score')
        ax.set_title('R² Score Comparison: Baseline vs Enhanced Models')
        ax.set_xticks(x)
        ax.set_xticklabels(years)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (baseline, enhanced) in enumerate(zip(baseline_r2, enhanced_r2)):
            ax.text(i - width/2, baseline + 0.01, f'{baseline:.3f}', ha='center', va='bottom')
            ax.text(i + width/2, enhanced + 0.01, f'{enhanced:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "r2_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_individual_mae_comparison(self, results_2023, results_2024):
        """Plot MAE comparison as individual file."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        years = ['2023', '2024']
        baseline_mae = [results_2023['baseline_mae'], results_2024['baseline_mae']]
        enhanced_mae = [results_2023['enhanced_mae'], results_2024['enhanced_mae']]
        
        x = np.arange(len(years))
        width = 0.35
        
        ax.bar(x - width/2, baseline_mae, width, label='Baseline Model', alpha=0.8, color='skyblue')
        ax.bar(x + width/2, enhanced_mae, width, label='Enhanced Model', alpha=0.8, color='lightcoral')
        
        ax.set_xlabel('Test Year')
        ax.set_ylabel('Mean Absolute Error (rides/month)')
        ax.set_title('Mean Absolute Error Comparison: Baseline vs Enhanced Models')
        ax.set_xticks(x)
        ax.set_xticklabels(years)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (baseline, enhanced) in enumerate(zip(baseline_mae, enhanced_mae)):
            ax.text(i - width/2, baseline + 10, f'{baseline:.1f}', ha='center', va='bottom')
            ax.text(i + width/2, enhanced + 10, f'{enhanced:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "mae_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_individual_infrastructure_effects_by_year(self, results_2023, results_2024):
        """Plot infrastructure effects by year as individual file."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        years = ['2023', '2024']
        effects = [results_2023['infrastructure_effect_baseline'], results_2024['infrastructure_effect_baseline']]
        
        bars = ax.bar(years, effects, color=['green' if e > 0 else 'red' for e in effects], alpha=0.7)
        ax.set_xlabel('Year')
        ax.set_ylabel('Infrastructure Effect (rides/month)')
        ax.set_title('Infrastructure Impact by Year')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add value labels
        for bar, effect in zip(bars, effects):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (5 if height > 0 else -15),
                   f'{effect:.1f}', ha='center', va='bottom' if height > 0 else 'top')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "infrastructure_effects_by_year.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_individual_effect_consistency(self, final_results):
        """Plot effect consistency analysis as individual file."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        categories = ['2023 Effect', '2024 Effect', 'Average Effect']
        values = [
            final_results['infrastructure_effect_2023'],
            final_results['infrastructure_effect_2024'],
            final_results['average_infrastructure_effect']
        ]
        colors = ['lightblue', 'lightgreen', 'gold']
        
        bars = ax.bar(categories, values, alpha=0.8, color=colors)
        ax.set_xlabel('Metric')
        ax.set_ylabel('Infrastructure Effect (rides/month)')
        ax.set_title(f'Infrastructure Effect Consistency (Consistency Score: {final_results["effect_consistency"]:.3f})')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                   f'{value:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "effect_consistency.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_individual_2023_affected_gaps(self, results_2023):
        """Plot 2023 affected stations gap distribution."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        data_2023 = results_2023['results_df']
        affected_gaps_2023 = data_2023[data_2023['is_affected']]['baseline_gap']
        
        ax.hist(affected_gaps_2023, bins=50, alpha=0.7, color='red', edgecolor='black')
        ax.set_title('2023 Affected Stations: Baseline Gap Distribution')
        ax.set_xlabel('Gap (Actual - Predicted rides/month)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax.axvline(x=affected_gaps_2023.mean(), color='red', linestyle='-', alpha=0.8, 
                  label=f'Mean: {affected_gaps_2023.mean():.1f}')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "gaps_2023_affected.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_individual_2023_unaffected_gaps(self, results_2023):
        """Plot 2023 unaffected stations gap distribution."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        data_2023 = results_2023['results_df']
        unaffected_gaps_2023 = data_2023[~data_2023['is_affected']]['baseline_gap']
        
        ax.hist(unaffected_gaps_2023, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax.set_title('2023 Unaffected Stations: Baseline Gap Distribution')
        ax.set_xlabel('Gap (Actual - Predicted rides/month)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax.axvline(x=unaffected_gaps_2023.mean(), color='blue', linestyle='-', alpha=0.8,
                  label=f'Mean: {unaffected_gaps_2023.mean():.1f}')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "gaps_2023_unaffected.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_individual_2024_affected_gaps(self, results_2024):
        """Plot 2024 affected stations gap distribution."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        data_2024 = results_2024['results_df']
        affected_gaps_2024 = data_2024[data_2024['is_affected']]['baseline_gap']
        
        ax.hist(affected_gaps_2024, bins=50, alpha=0.7, color='red', edgecolor='black')
        ax.set_title('2024 Affected Stations: Baseline Gap Distribution')
        ax.set_xlabel('Gap (Actual - Predicted rides/month)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax.axvline(x=affected_gaps_2024.mean(), color='red', linestyle='-', alpha=0.8,
                  label=f'Mean: {affected_gaps_2024.mean():.1f}')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "gaps_2024_affected.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_individual_2024_unaffected_gaps(self, results_2024):
        """Plot 2024 unaffected stations gap distribution."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        data_2024 = results_2024['results_df']
        unaffected_gaps_2024 = data_2024[~data_2024['is_affected']]['baseline_gap']
        
        ax.hist(unaffected_gaps_2024, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax.set_title('2024 Unaffected Stations: Baseline Gap Distribution')
        ax.set_xlabel('Gap (Actual - Predicted rides/month)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax.axvline(x=unaffected_gaps_2024.mean(), color='blue', linestyle='-', alpha=0.8,
                  label=f'Mean: {unaffected_gaps_2024.mean():.1f}')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "gaps_2024_unaffected.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_individual_2023_comparison_gaps(self, results_2023):
        """Plot 2023 affected vs unaffected comparison."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        data_2023 = results_2023['results_df']
        affected_gaps_2023 = data_2023[data_2023['is_affected']]['baseline_gap']
        unaffected_gaps_2023 = data_2023[~data_2023['is_affected']]['baseline_gap']
        
        ax.hist([affected_gaps_2023, unaffected_gaps_2023], bins=50, alpha=0.7, 
               label=['Affected Stations', 'Unaffected Stations'], color=['red', 'blue'])
        ax.set_title('2023 Gap Distribution: Affected vs Unaffected Stations')
        ax.set_xlabel('Gap (Actual - Predicted rides/month)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "gaps_comparison_2023.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_individual_2024_comparison_gaps(self, results_2024):
        """Plot 2024 affected vs unaffected comparison."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        data_2024 = results_2024['results_df']
        affected_gaps_2024 = data_2024[data_2024['is_affected']]['baseline_gap']
        unaffected_gaps_2024 = data_2024[~data_2024['is_affected']]['baseline_gap']
        
        ax.hist([affected_gaps_2024, unaffected_gaps_2024], bins=50, alpha=0.7,
               label=['Affected Stations', 'Unaffected Stations'], color=['red', 'blue'])
        ax.set_title('2024 Gap Distribution: Affected vs Unaffected Stations')
        ax.set_xlabel('Gap (Actual - Predicted rides/month)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "gaps_comparison_2024.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _analyze_weather_resilience(self, data_2023, data_2024):
        """Analyze weather resilience factor and weather-specific impacts."""
        print("   Calculating Weather Resilience Factor...")
        
        results = {}
        
        # Phase 1 (2023) analysis
        affected_2023 = data_2023[data_2023['start_station_id'].isin(self.affected_stations_2023_plus)]
        unaffected_2023 = data_2023[~data_2023['start_station_id'].isin(self.affected_stations_2023_plus)]
        
        # Phase 2 (2024) analysis
        affected_2024 = data_2024[data_2024['start_station_id'].isin(self.affected_stations_2023_plus)]
        unaffected_2024 = data_2024[~data_2024['start_station_id'].isin(self.affected_stations_2023_plus)]
        
        # Weather Resilience Factor calculation using detailed weather data
        def calculate_wrf(affected_data, unaffected_data, weather_condition):
            """Calculate Weather Resilience Factor for specific weather condition using detailed data."""
            
            def extract_rides_for_condition(data, condition):
                """Extract rides for specific weather/UTCI condition from detailed data."""
                total_rides = 0
                total_weighted_rides = 0
                
                for _, row in data.iterrows():
                    if condition == 'bad_weather':
                        # Count rides in Cold, Rain, Snow, Very strong cold stress, Extreme cold stress
                        weather_detail = row.get('weather_detail', {})
                        utci_detail = row.get('utci_detail', {})
                        
                        bad_weather_rides = (
                            weather_detail.get('Cold', 0) + 
                            weather_detail.get('Rain', 0) +
                            weather_detail.get('Snow', 0) +
                            utci_detail.get('Very strong cold stress', 0) +
                            utci_detail.get('Extreme cold stress', 0)
                        )
                        total_rides += bad_weather_rides
                        total_weighted_rides += row['monthly_rides']
                    else:
                        # Count rides for specific weather condition
                        weather_detail = row.get('weather_detail', {})
                        condition_rides = weather_detail.get(condition, 0)
                        
                        if condition_rides > 0:
                            # Weight by proportion of rides in this condition
                            proportion = condition_rides / row['total_rides_detailed'] if row['total_rides_detailed'] > 0 else 0
                            total_rides += condition_rides
                            total_weighted_rides += row['monthly_rides'] * proportion
                
                return total_weighted_rides / max(1, len(data)) if len(data) > 0 else 0.0
            
            affected_avg = extract_rides_for_condition(affected_data, weather_condition)
            unaffected_avg = extract_rides_for_condition(unaffected_data, weather_condition)
            
            if unaffected_avg == 0:
                return 0.0
            
            wrf = (affected_avg - unaffected_avg) / unaffected_avg
            return wrf
        
        # Calculate WRF for phases
        wrf_2023 = calculate_wrf(affected_2023, unaffected_2023, 'bad_weather')
        wrf_2024 = calculate_wrf(affected_2024, unaffected_2024, 'bad_weather')
        
        results['weather_resilience_factor'] = {
            'phase_1_2023': wrf_2023,
            'phase_2_2024': wrf_2024,
            'improvement': wrf_2024 - wrf_2023
        }
        
        # Weather-specific analysis
        weather_categories = ['Cold', 'Heat', 'Mist/Fog', 'Neutral', 'Rain', 'Snow']
        weather_analysis = {}
        
        for weather in weather_categories:
            phase1_wrf = calculate_wrf(affected_2023, unaffected_2023, weather)
            phase2_wrf = calculate_wrf(affected_2024, unaffected_2024, weather)
            
            weather_analysis[weather] = {
                'phase_1': phase1_wrf,
                'phase_2': phase2_wrf,
                'improvement': phase2_wrf - phase1_wrf
            }
        
        results['weather_specific'] = weather_analysis
        
        # UTCI thermal stress analysis - categories with sufficient data
        utci_categories = [
            'No thermal stress',
            'Moderate cold stress', 
            'Strong cold stress', 
            'Very strong cold stress',
            'Moderate heat stress',
            'Strong heat stress'
        ]
        utci_analysis = {}
        
        # Check what UTCI categories actually exist in detailed data
        all_utci_found = set()
        for data in [affected_2023, unaffected_2023, affected_2024, unaffected_2024]:
            for _, row in data.iterrows():
                utci_detail = row.get('utci_detail', {})
                all_utci_found.update(utci_detail.keys())
        
        print(f"   UTCI categories found in detailed data: {sorted(all_utci_found)}")
        
        # Only analyze UTCI categories that actually exist
        utci_categories_actual = [cat for cat in utci_categories if cat in all_utci_found]
        
        for utci in utci_categories_actual:
            def calc_utci_wrf(affected_data, unaffected_data, utci_condition):
                """Calculate WRF for specific UTCI condition using detailed data."""
                def extract_utci_rides(data, condition):
                    total_weighted_rides = 0
                    count = 0
                    
                    for _, row in data.iterrows():
                        utci_detail = row.get('utci_detail', {})
                        condition_rides = utci_detail.get(condition, 0)
                        
                        if condition_rides > 0:
                            # Weight by proportion of rides in this UTCI condition
                            proportion = condition_rides / row['total_rides_detailed'] if row['total_rides_detailed'] > 0 else 0
                            total_weighted_rides += row['monthly_rides'] * proportion
                            count += 1
                    
                    return total_weighted_rides / max(1, count) if count > 0 else 0.0
                
                affected_avg = extract_utci_rides(affected_data, utci_condition)
                unaffected_avg = extract_utci_rides(unaffected_data, utci_condition)
                
                if unaffected_avg == 0:
                    return 0.0
                
                return (affected_avg - unaffected_avg) / unaffected_avg
            
            phase1_utci = calc_utci_wrf(affected_2023, unaffected_2023, utci)
            phase2_utci = calc_utci_wrf(affected_2024, unaffected_2024, utci)
            
            utci_analysis[utci] = {
                'phase_1': phase1_utci,
                'phase_2': phase2_utci,
                'improvement': phase2_utci - phase1_utci
            }
        
        results['utci_analysis'] = utci_analysis
        
        print(f"   Weather Resilience Factor Phase 1: {wrf_2023:.3f}")
        print(f"   Weather Resilience Factor Phase 2: {wrf_2024:.3f}")
        print(f"   WRF Improvement: {(wrf_2024 - wrf_2023):.3f}")
        
        return results
    
    def _analyze_network_spillover_effects(self, data_2023, data_2024):
        """Analyze network effects and spillover benefits using precise geometric analysis."""
        print("   Calculating precise network spillover effects...")
        
        results = {}
        
        def calculate_spillover_for_year(data, year):
            """Calculate spillover effects using exact street-based distances."""
            spillover_results = {}
            
            # Get all station coordinates  
            stations = data[['start_station_id', 'start_station_latitude', 'start_station_longitude']].drop_duplicates()
            
            # Use appropriate infrastructure for the year
            if year == '2024':
                infrastructure_enhanced = pd.concat([self.infrastructure_2022_enhanced, self.infrastructure_2023_enhanced])
                impact_details = self.impact_details_2023_plus
            else:
                infrastructure_enhanced = self.infrastructure_2022_enhanced
                impact_details = self.impact_details_2022
            
            # Create precise distance-based spillover zones using street geometry impact data
            station_impacts = {}
            
            # For each station, find its closest infrastructure impact
            for _, station in stations.iterrows():
                station_id = station['start_station_id']
                station_impacts[station_id] = {
                    'min_distance': float('inf'),
                    'impact_strength': 0.0,
                    'street_class': 'none',
                    'influence_distance': 0.0
                }
                
                # Find the station's impact data if it exists
                station_impact_data = impact_details[impact_details['station_id'] == station_id]
                
                if len(station_impact_data) > 0:
                    # Get the strongest impact (closest infrastructure)
                    strongest_impact = station_impact_data.loc[station_impact_data['impact_strength'].idxmax()]
                    station_impacts[station_id] = {
                        'min_distance': strongest_impact['exact_distance_to_street'],
                        'impact_strength': strongest_impact['impact_strength'],
                        'street_class': strongest_impact['street_class'], 
                        'influence_distance': strongest_impact['influence_distance']
                    }
            
            # Define dynamic spillover zones based on street-specific influence distances
            direct_stations = []
            spillover_1_stations = []
            spillover_2_stations = []
            baseline_stations = []
            
            for station_id, impact_data in station_impacts.items():
                distance = impact_data['min_distance']
                influence_dist = impact_data['influence_distance']
                
                if distance == float('inf') or influence_dist == 0:
                    # No infrastructure impact - this is a baseline station
                    baseline_stations.append(station_id)
                else:
                    # Calculate relative position within influence zone
                    relative_distance = distance / influence_dist
                    
                    if relative_distance <= 0.3:  # Within 30% of influence zone
                        direct_stations.append(station_id)
                    elif relative_distance <= 0.6:  # 30-60% of influence zone
                        spillover_1_stations.append(station_id)
                    elif relative_distance <= 1.0:  # 60-100% of influence zone  
                        spillover_2_stations.append(station_id)
                    else:
                        # Beyond influence zone but close enough for extended spillover
                        if distance <= influence_dist * 1.5:  # Within 150% of influence
                            spillover_2_stations.append(station_id)
                        else:
                            baseline_stations.append(station_id)
            
            # Calculate ridership gains for each zone with proper error handling
            direct_rides = data[data['start_station_id'].isin(direct_stations)]['monthly_rides'].mean() if direct_stations else np.nan
            spillover_1_rides = data[data['start_station_id'].isin(spillover_1_stations)]['monthly_rides'].mean() if spillover_1_stations else np.nan
            spillover_2_rides = data[data['start_station_id'].isin(spillover_2_stations)]['monthly_rides'].mean() if spillover_2_stations else np.nan
            baseline_rides = data[data['start_station_id'].isin(baseline_stations)]['monthly_rides'].mean() if baseline_stations else np.nan
            
            # Debug information with precise methodology  
            print(f"     Precise Spillover Zone Counts:")
            print(f"       Direct Impact (≤30% of influence): {len(direct_stations)}")
            print(f"       Spillover 1 (30-60% of influence): {len(spillover_1_stations)}")
            print(f"       Spillover 2 (60-150% of influence): {len(spillover_2_stations)}")
            print(f"       Baseline (>150% of influence): {len(baseline_stations)}")
            
            if not pd.isna(direct_rides) and not pd.isna(baseline_rides):
                print(f"     Average Monthly Rides - Direct: {direct_rides:.1f}, Baseline: {baseline_rides:.1f}")
            
            # Calculate gains - only use real data when available
            if pd.isna(direct_rides) or pd.isna(baseline_rides) or len(direct_stations) == 0 or len(baseline_stations) == 0:
                # Insufficient data for spillover analysis - return null results
                print(f"     Warning: Insufficient spillover data for {year} - zones too small for reliable analysis")
                direct_impact_gain = 0
                spillover_1_gain = 0  
                spillover_2_gain = 0
            else:
                # Calculate relative gains using precise methodology
                direct_impact_gain = direct_rides - baseline_rides
                spillover_1_gain = spillover_1_rides - baseline_rides if not pd.isna(spillover_1_rides) else 0
                spillover_2_gain = spillover_2_rides - baseline_rides if not pd.isna(spillover_2_rides) else 0
            
            # Additional precise spillover metrics
            spillover_results = {
                'direct_impact_gain': direct_impact_gain,
                'spillover_1_gain': spillover_1_gain,
                'spillover_2_gain': spillover_2_gain,
                'network_multiplier': 0,
                'zone_counts': {
                    'direct': len(direct_stations),
                    'spillover_1': len(spillover_1_stations),
                    'spillover_2': len(spillover_2_stations),
                    'baseline': len(baseline_stations)
                },
                'impact_distribution': {
                    'high_impact_stations': len([sid for sid, data in station_impacts.items() if data['impact_strength'] > 0.7]),
                    'medium_impact_stations': len([sid for sid, data in station_impacts.items() if 0.3 < data['impact_strength'] <= 0.7]),
                    'low_impact_stations': len([sid for sid, data in station_impacts.items() if 0 < data['impact_strength'] <= 0.3])
                }
            }
            
            # Calculate network effect multiplier with safety checks
            total_gain = spillover_results['direct_impact_gain'] + spillover_results['spillover_1_gain'] + spillover_results['spillover_2_gain']
            direct_gain = spillover_results['direct_impact_gain']
            
            if direct_gain > 0:
                multiplier = total_gain / direct_gain
                # Cap unrealistic multipliers (network effects shouldn't exceed 4x)
                spillover_results['network_multiplier'] = max(1.0, min(multiplier, 4.0))
            else:
                # No direct gain means no measurable network effect
                spillover_results['network_multiplier'] = 1.0
            
            return spillover_results
        
        # Merge distance data
        data_2023_with_distance = data_2023.copy()
        data_2024_with_distance = data_2024.copy()
        
        results['phase_1_2023'] = calculate_spillover_for_year(data_2023_with_distance, '2023')
        results['phase_2_2024'] = calculate_spillover_for_year(data_2024_with_distance, '2024')
        
        print(f"   Network Multiplier Phase 1: {results['phase_1_2023']['network_multiplier']:.2f}x")
        print(f"   Network Multiplier Phase 2: {results['phase_2_2024']['network_multiplier']:.2f}x")
        print(f"   Direct Impact: +{results['phase_2_2024']['direct_impact_gain']:.0f} rides/month")
        print(f"   Spillover Zone 1: +{results['phase_2_2024']['spillover_1_gain']:.0f} rides/month")
        print(f"   Spillover Zone 2: +{results['phase_2_2024']['spillover_2_gain']:.0f} rides/month")
        
        return results
    
    def _perform_seasonal_validation(self, baseline_model, enhanced_model, baseline_features, enhanced_features, data_2023, data_2024):
        """Perform seasonal validation and model robustness analysis for all 4 model combinations."""
        print("   Performing seasonal validation...")
        
        results = {}
        
        # Test all 4 model combinations
        test_combinations = [
            ('2023', data_2023, 'Baseline 2022→2023'),
            ('2023', data_2023, 'Enhanced 2022→2023'),
            ('2024', data_2024, 'Baseline 2022→2024'),
            ('2024', data_2024, 'Enhanced 2022→2024')
        ]
        
        # Monthly accuracy analysis for all combinations
        monthly_results = {}
        
        for month in range(1, 13):
            monthly_results[month] = {}
            
            # Test each model combination
            for combination_idx, (year, test_data, model_name) in enumerate(test_combinations):
                month_data = test_data[test_data['year_month'].dt.month == month]
                
                if len(month_data) == 0:
                    continue
                
                # Determine which model and features to use
                if 'Baseline' in model_name:
                    model = baseline_model
                    month_data_processed = month_data.copy()
                    features = baseline_features
                else:  # Enhanced
                    model = enhanced_model
                    month_data_processed = self._add_infrastructure_features(month_data.copy(), year)
                    features = enhanced_features
                
                # Prepare features and targets
                X = month_data_processed[features].fillna(0)
                y_actual = month_data_processed['monthly_rides']
                
                # Make predictions
                y_pred = model.predict(X)
                
                # Calculate metrics
                mae = mean_absolute_error(y_actual, y_pred)
                mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
                
                monthly_results[month][model_name] = {
                    'mae': mae,
                    'mape': mape,
                    'sample_size': len(month_data)
                }
        
        results['monthly_accuracy'] = monthly_results
        
        # Calculate overall validation metrics for each model
        model_names = ['Baseline 2022→2023', 'Enhanced 2022→2023', 'Baseline 2022→2024', 'Enhanced 2022→2024']
        
        overall_validation = {}
        for model_name in model_names:
            model_months = []
            for month in monthly_results:
                if model_name in monthly_results[month]:
                    model_months.append(monthly_results[month][model_name])
            
            if model_months:
                overall_validation[model_name] = {
                    'average_mae': np.mean([m['mae'] for m in model_months]),
                    'average_mape': np.mean([m['mape'] for m in model_months]),
                    'best_month': None,  # Will be calculated in plotting
                    'worst_month': None  # Will be calculated in plotting
                }
        
        results['overall_validation'] = overall_validation
        
        # Print summary for enhanced model on 2024 (main comparison)
        if 'Enhanced 2022→2024' in overall_validation:
            main_model = overall_validation['Enhanced 2022→2024']
            print(f"   Average MAE (Enhanced 2022→2024): {main_model['average_mae']:.1f} rides/month")
            print(f"   Average MAPE (Enhanced 2022→2024): {main_model['average_mape']:.1f}%")
        
        # Analyze actual best and worst performing months from data
        if monthly_results:
            month_avg_mae = {}
            for month in monthly_results:
                month_maes = []
                for model_name in monthly_results[month]:
                    if 'mae' in monthly_results[month][model_name]:
                        month_maes.append(monthly_results[month][model_name]['mae'])
                if month_maes:
                    month_avg_mae[month] = np.mean(month_maes)
            
            if month_avg_mae:
                best_month = min(month_avg_mae.keys(), key=lambda k: month_avg_mae[k])
                worst_month = max(month_avg_mae.keys(), key=lambda k: month_avg_mae[k])
                print(f"   Best performing month: {best_month} (MAE: {month_avg_mae[best_month]:.1f})")
                print(f"   Worst performing month: {worst_month} (MAE: {month_avg_mae[worst_month]:.1f})")
            else:
                print("   No month-wise performance data available for analysis")
        else:
            print("   No seasonal validation data available")
        
        return results
    
    def _create_comprehensive_visualizations(self, results_2023, results_2024, final_results, 
                                           weather_results, network_results, seasonal_results):
        """Create comprehensive visualizations including weather, network, and seasonal analysis."""
        
        # Create existing individual plots
        self._create_visualizations(results_2023, results_2024, final_results)
        
        # Create additional comprehensive plots
        self._plot_weather_resilience_factor(weather_results)
        self._plot_weather_specific_analysis(weather_results)
        self._plot_utci_thermal_stress(weather_results)
        self._plot_network_spillover_effects(network_results)
        self._plot_seasonal_validation(seasonal_results)
        
        # Create new advanced analysis plots
        self._plot_infrastructure_effectiveness_summary(final_results, weather_results, network_results)
        self._plot_model_performance_evolution(results_2023, results_2024)
        self._plot_street_class_impact_analysis(network_results)
        
        # Create user type infrastructure analysis
        self._plot_user_type_infrastructure_analysis(results_2023, results_2024)
        
        print(f"   Created comprehensive visualization suite with 22 total plots")
        print(f"   All plots saved to: {self.output_dir}")
        print("     New plots added:")
        print("     16. impact_strength_distribution.png")
        print("     17. spillover_zone_composition.png") 
        print("     18. infrastructure_effectiveness_summary.png")
        print("     19. model_performance_evolution.png")
        print("     20. street_class_impact_analysis.png")
        print("     21. user_type_infrastructure_comparison.png")
    
    def _plot_infrastructure_effectiveness_summary(self, final_results, weather_results, network_results):
        """Create comprehensive infrastructure effectiveness summary dashboard."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Quadrant 1: Infrastructure Effect Overview
        effects = [
            final_results['infrastructure_effect_2023'],
            final_results['infrastructure_effect_2024'],
            final_results['average_infrastructure_effect']
        ]
        categories = ['2023 Effect', '2024 Effect', 'Average']
        colors = ['lightblue', 'lightgreen', 'gold']
        
        bars1 = ax1.bar(categories, effects, color=colors, alpha=0.8)
        ax1.set_ylabel('Infrastructure Effect (rides/month)')
        ax1.set_title('Infrastructure Effect Analysis')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        for bar, value in zip(bars1, effects):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{value:.1f}', ha='center', va='bottom')
        
        # Quadrant 2: Model Improvement
        model_improvements = [
            final_results['model_improvement_2023'],
            final_results['model_improvement_2024']
        ]
        years = ['2023', '2024']
        
        bars2 = ax2.bar(years, model_improvements, color=['coral', 'darkseagreen'], alpha=0.8)
        ax2.set_ylabel('R² Improvement')
        ax2.set_title('Model Performance Improvement')
        ax2.grid(True, alpha=0.3)
        
        for bar, value in zip(bars2, model_improvements):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Quadrant 3: Weather Resilience
        wrf_phases = ['Phase 1 (2023)', 'Phase 2 (2024)']
        wrf_values = [
            weather_results['weather_resilience_factor']['phase_1_2023'],
            weather_results['weather_resilience_factor']['phase_2_2024']
        ]
        
        bars3 = ax3.bar(wrf_phases, wrf_values, color=['lightsteelblue', 'mediumseagreen'], alpha=0.8)
        ax3.set_ylabel('Weather Resilience Factor')
        ax3.set_title('Weather Resilience Enhancement')
        ax3.grid(True, alpha=0.3)
        
        for bar, value in zip(bars3, wrf_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # Quadrant 4: Network Multiplier Effect
        network_phases = ['Phase 1 (2023)', 'Phase 2 (2024)']
        network_multipliers = [
            network_results['phase_1_2023']['network_multiplier'],
            network_results['phase_2_2024']['network_multiplier']
        ]
        
        bars4 = ax4.bar(network_phases, network_multipliers, color=['plum', 'mediumorchid'], alpha=0.8)
        ax4.set_ylabel('Network Effect Multiplier')
        ax4.set_title('Network Spillover Effects')
        ax4.grid(True, alpha=0.3)
        
        for bar, value in zip(bars4, network_multipliers):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{value:.2f}x', ha='center', va='bottom')
        
        plt.suptitle('Infrastructure Effectiveness Comprehensive Summary', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "infrastructure_effectiveness_summary.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_model_performance_evolution(self, results_2023, results_2024):
        """Plot model performance evolution across years."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Performance metrics evolution
        years = ['2023', '2024']
        baseline_r2 = [results_2023['baseline_r2'], results_2024['baseline_r2']]
        enhanced_r2 = [results_2023['enhanced_r2'], results_2024['enhanced_r2']]
        baseline_mae = [results_2023['baseline_mae'], results_2024['baseline_mae']]
        enhanced_mae = [results_2023['enhanced_mae'], results_2024['enhanced_mae']]
        
        # R² Evolution
        x = np.arange(len(years))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, baseline_r2, width, label='Baseline Model', color='lightcoral', alpha=0.8)
        bars2 = ax1.bar(x + width/2, enhanced_r2, width, label='Enhanced Model', color='lightblue', alpha=0.8)
        
        ax1.set_xlabel('Test Year')
        ax1.set_ylabel('R² Score')
        ax1.set_title('Model Performance Evolution: R² Scores')
        ax1.set_xticks(x)
        ax1.set_xticklabels(years)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                        f'{height:.3f}', ha='center', va='bottom')
        
        # MAE Evolution
        bars3 = ax2.bar(x - width/2, baseline_mae, width, label='Baseline Model', color='lightcoral', alpha=0.8)
        bars4 = ax2.bar(x + width/2, enhanced_mae, width, label='Enhanced Model', color='lightblue', alpha=0.8)
        
        ax2.set_xlabel('Test Year')
        ax2.set_ylabel('Mean Absolute Error')
        ax2.set_title('Model Performance Evolution: MAE Scores')
        ax2.set_xticks(x)
        ax2.set_xticklabels(years)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
                        f'{height:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "model_performance_evolution.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_street_class_impact_analysis(self, network_results):
        """Plot analysis of infrastructure impact by street classification using real data."""
        
        # Check if we have actual street class data from the precise analyzer
        if hasattr(self, 'impact_details_2022') and hasattr(self, 'impact_details_2023_plus'):
            # Extract real street class data
            try:
                # Get street class distribution from actual impact data
                details_2022 = self.impact_details_2022
                details_2023_plus = self.impact_details_2023_plus
                
                # Count by street class for each phase
                street_class_2022 = details_2022['street_class'].value_counts()
                street_class_2023_plus = details_2023_plus['street_class'].value_counts()
                
                # Get all unique street classes
                all_classes = set(street_class_2022.index.tolist() + street_class_2023_plus.index.tolist())
                street_classes = sorted(list(all_classes))
                
                if street_classes:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                    
                    # Phase 1 (2022) - Real data
                    phase1_counts = [street_class_2022.get(sc, 0) for sc in street_classes]
                    colors = ['darkred', 'orange', 'gold', 'lightgray'][:len(street_classes)]
                    
                    ax1.bar(street_classes, phase1_counts, color=colors, alpha=0.8)
                    ax1.set_ylabel('Number of Affected Stations')
                    ax1.set_title('2022: Infrastructure Impact by Street Class')
                    ax1.grid(True, alpha=0.3)
                    ax1.tick_params(axis='x', rotation=45)
                    
                    for i, (street, count) in enumerate(zip(street_classes, phase1_counts)):
                        ax1.text(i, count + 0.5, f'{count}', ha='center', va='bottom')
                    
                    # Phase 2 (2023+) - Real data
                    phase2_counts = [street_class_2023_plus.get(sc, 0) for sc in street_classes]
                    
                    ax2.bar(street_classes, phase2_counts, color=colors, alpha=0.8)
                    ax2.set_ylabel('Number of Affected Stations')
                    ax2.set_title('2023+: Infrastructure Impact by Street Class')
                    ax2.grid(True, alpha=0.3)
                    ax2.tick_params(axis='x', rotation=45)
                    
                    for i, (street, count) in enumerate(zip(street_classes, phase2_counts)):
                        ax2.text(i, count + 0.5, f'{count}', ha='center', va='bottom')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.output_dir, "street_class_impact_analysis.png"), dpi=300, bbox_inches='tight')
                    plt.close()
                    return
                    
            except Exception as e:
                print(f"Warning: Could not extract real street class data: {e}")
        
        # Fallback: Create informative placeholder if no real data available
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.text(0.5, 0.5, 'Street Class Analysis\nRequires Running Full\nPrecise Geometric Analysis', 
               ha='center', va='center', transform=ax.transAxes, fontsize=16)
        ax.set_title('Street Class Impact Analysis\n(Real Data Not Available)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "street_class_impact_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_user_type_infrastructure_analysis(self, results_2023, results_2024):
        """Analyze and plot how infrastructure affects Member vs Casual users differently."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Extract data for analysis
        data_2023 = results_2023['results_df']
        data_2024 = results_2024['results_df']
        
        # Check if user type data is available
        if 'Member' not in data_2023.columns or 'Casual' not in data_2023.columns:
            # Create placeholder plot if no user type data
            for ax in [ax1, ax2, ax3, ax4]:
                ax.text(0.5, 0.5, 'User Type Data\nNot Available', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
                ax.set_title('User Type Analysis')
            
            plt.suptitle('User Type Infrastructure Analysis\n(No User Type Data Available)', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "user_type_infrastructure_comparison.png"), dpi=300, bbox_inches='tight')
            plt.close()
            return
        
        # Calculate user type proportions for affected vs unaffected stations
        for year_idx, (data, year) in enumerate([(data_2023, '2023'), (data_2024, '2024')]):
            ax = ax1 if year_idx == 0 else ax2
            
            # Calculate total rides by user type for affected vs unaffected
            affected_data = data[data['is_affected']].copy()
            unaffected_data = data[~data['is_affected']].copy()
            
            affected_member = affected_data['Member'].sum() if len(affected_data) > 0 else 0
            affected_casual = affected_data['Casual'].sum() if len(affected_data) > 0 else 0
            unaffected_member = unaffected_data['Member'].sum() if len(unaffected_data) > 0 else 0
            unaffected_casual = unaffected_data['Casual'].sum() if len(unaffected_data) > 0 else 0
            
            # Create comparison data
            categories = ['Affected Stations', 'Unaffected Stations']
            member_values = [affected_member, unaffected_member]
            casual_values = [affected_casual, unaffected_casual]
            
            x = np.arange(len(categories))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, member_values, width, label='Member', color='steelblue', alpha=0.8)
            bars2 = ax.bar(x + width/2, casual_values, width, label='Casual', color='orange', alpha=0.8)
            
            ax.set_xlabel('Station Type')
            ax.set_ylabel('Total Rides')
            ax.set_title(f'{year}: Member vs Casual Ridership\nby Infrastructure Status')
            ax.set_xticks(x)
            ax.set_xticklabels(categories)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{int(height/1000)}K', ha='center', va='bottom', fontsize=9)
        
        # Plot 3: Member/Casual ratio analysis by infrastructure status
        affected_member_ratio_2023 = data_2023[data_2023['is_affected']]['Member'].sum() / (data_2023[data_2023['is_affected']]['Member'].sum() + data_2023[data_2023['is_affected']]['Casual'].sum()) if len(data_2023[data_2023['is_affected']]) > 0 else 0
        unaffected_member_ratio_2023 = data_2023[~data_2023['is_affected']]['Member'].sum() / (data_2023[~data_2023['is_affected']]['Member'].sum() + data_2023[~data_2023['is_affected']]['Casual'].sum()) if len(data_2023[~data_2023['is_affected']]) > 0 else 0
        
        affected_member_ratio_2024 = data_2024[data_2024['is_affected']]['Member'].sum() / (data_2024[data_2024['is_affected']]['Member'].sum() + data_2024[data_2024['is_affected']]['Casual'].sum()) if len(data_2024[data_2024['is_affected']]) > 0 else 0
        unaffected_member_ratio_2024 = data_2024[~data_2024['is_affected']]['Member'].sum() / (data_2024[~data_2024['is_affected']]['Member'].sum() + data_2024[~data_2024['is_affected']]['Casual'].sum()) if len(data_2024[~data_2024['is_affected']]) > 0 else 0
        
        categories = ['2023 Affected', '2023 Unaffected', '2024 Affected', '2024 Unaffected']
        member_ratios = [affected_member_ratio_2023, unaffected_member_ratio_2023, affected_member_ratio_2024, unaffected_member_ratio_2024]
        colors = ['darkred', 'lightcoral', 'darkgreen', 'lightgreen']
        
        bars = ax3.bar(categories, member_ratios, color=colors, alpha=0.8)
        ax3.set_ylabel('Member Ratio (Member / Total)')
        ax3.set_title('Member Usage Ratio:\nInfrastructure vs Non-Infrastructure Stations')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        ax3.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='50% Split')
        ax3.legend()
        
        # Add value labels
        for bar, ratio in zip(bars, member_ratios):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{ratio:.2f}', ha='center', va='bottom')
        
        # Plot 4: Infrastructure effect by user type (gap analysis)
        affected_2023 = data_2023[data_2023['is_affected']]
        unaffected_2023 = data_2023[~data_2023['is_affected']]
        affected_2024 = data_2024[data_2024['is_affected']]
        unaffected_2024 = data_2024[~data_2024['is_affected']]
        
        # Calculate infrastructure effect (difference between affected and unaffected) by user type
        member_effect_2023 = (affected_2023['Member'].mean() - unaffected_2023['Member'].mean()) if len(affected_2023) > 0 and len(unaffected_2023) > 0 else 0
        casual_effect_2023 = (affected_2023['Casual'].mean() - unaffected_2023['Casual'].mean()) if len(affected_2023) > 0 and len(unaffected_2023) > 0 else 0
        member_effect_2024 = (affected_2024['Member'].mean() - unaffected_2024['Member'].mean()) if len(affected_2024) > 0 and len(unaffected_2024) > 0 else 0
        casual_effect_2024 = (affected_2024['Casual'].mean() - unaffected_2024['Casual'].mean()) if len(affected_2024) > 0 and len(unaffected_2024) > 0 else 0
        
        years = ['2023', '2024']
        member_effects = [member_effect_2023, member_effect_2024]
        casual_effects = [casual_effect_2023, casual_effect_2024]
        
        x = np.arange(len(years))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, member_effects, width, label='Member Effect', color='steelblue', alpha=0.8)
        bars2 = ax4.bar(x + width/2, casual_effects, width, label='Casual Effect', color='orange', alpha=0.8)
        
        ax4.set_xlabel('Year')
        ax4.set_ylabel('Infrastructure Effect (rides/month)')
        ax4.set_title('Infrastructure Effect by User Type\n(Affected - Unaffected Stations)')
        ax4.set_xticks(x)
        ax4.set_xticklabels(years)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + (5 if height > 0 else -15),
                        f'{height:.1f}', ha='center', va='bottom' if height > 0 else 'top')
        
        plt.suptitle('User Type Infrastructure Impact Analysis:\nHow Infrastructure Affects Members vs Casual Users', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "user_type_infrastructure_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_weather_resilience_factor(self, weather_results):
        """Plot Weather Resilience Factor by phase."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        phases = ['Phase 1 (2023)', 'Phase 2 (2024)']
        wrf_values = [
            weather_results['weather_resilience_factor']['phase_1_2023'],
            weather_results['weather_resilience_factor']['phase_2_2024']
        ]
        
        bars = ax.bar(phases, wrf_values, color=['lightblue', 'lightgreen'], alpha=0.8)
        ax.set_ylabel('Weather Resilience Factor')
        ax.set_title('Weather Resilience Factor by Infrastructure Implementation Phase')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, wrf_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "weather_resilience_factor.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_weather_specific_analysis(self, weather_results):
        """Plot weather-specific gap analysis."""
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        weather_cats = list(weather_results['weather_specific'].keys())
        phase1_values = [weather_results['weather_specific'][w]['phase_1'] for w in weather_cats]
        phase2_values = [weather_results['weather_specific'][w]['phase_2'] for w in weather_cats]
        
        x = np.arange(len(weather_cats))
        width = 0.35
        
        # Color coding for different weather types
        colors_phase1 = []
        colors_phase2 = []
        for cat in weather_cats:
            if cat == 'Cold':
                colors_phase1.append('lightblue')
                colors_phase2.append('steelblue')
            elif cat == 'Heat':
                colors_phase1.append('lightcoral')
                colors_phase2.append('darkred')
            elif cat == 'Rain':
                colors_phase1.append('lightsteelblue')
                colors_phase2.append('darkslateblue')
            elif cat == 'Snow':
                colors_phase1.append('lightyellow')
                colors_phase2.append('darkblue')    
            elif cat == 'Mist/Fog':
                colors_phase1.append('lightgray')
                colors_phase2.append('gray')
            elif cat == 'Neutral':
                colors_phase1.append('lightgreen')
                colors_phase2.append('darkgreen')
            else:
                colors_phase1.append('wheat')
                colors_phase2.append('orange')
        
        bars1 = ax.bar(x - width/2, phase1_values, width, alpha=0.8, color=colors_phase1)
        bars2 = ax.bar(x + width/2, phase2_values, width, alpha=0.8, color=colors_phase2)
        
        ax.set_xlabel('Weather Category', fontsize=12)
        ax.set_ylabel('Weather Resilience Factor', fontsize=12)
        ax.set_title('Weather-Specific Infrastructure Impact Analysis:\nResilience Across Different Weather Conditions\n(Left bars: 2023, Right bars: 2024)', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(weather_cats)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        

        
        # Add value labels on bars for better readability
        for i, (bar1, bar2, val1, val2) in enumerate(zip(bars1, bars2, phase1_values, phase2_values)):
            if abs(val1) > 0.1:  # Only show labels for significant values
                ax.text(bar1.get_x() + bar1.get_width()/2., val1 + 0.05 if val1 > 0 else val1 - 0.05, 
                       f'{val1:.2f}', ha='center', va='bottom' if val1 > 0 else 'top', fontsize=10)
            if abs(val2) > 0.1:
                ax.text(bar2.get_x() + bar2.get_width()/2., val2 + 0.05 if val2 > 0 else val2 - 0.05, 
                       f'{val2:.2f}', ha='center', va='bottom' if val2 > 0 else 'top', fontsize=10)
        

        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "weather_specific_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_utci_thermal_stress(self, weather_results):
        """Plot comprehensive UTCI thermal stress analysis."""
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        
        utci_cats = list(weather_results['utci_analysis'].keys())
        phase1_values = [weather_results['utci_analysis'][u]['phase_1'] for u in utci_cats]
        phase2_values = [weather_results['utci_analysis'][u]['phase_2'] for u in utci_cats]
        
        x = np.arange(len(utci_cats))
        width = 0.35
        
        # Color coding: cool colors for cold stress, warm colors for heat stress, neutral for no stress
        colors_phase1 = []
        colors_phase2 = []
        for cat in utci_cats:
            if 'cold' in cat.lower():
                colors_phase1.append('lightblue')
                colors_phase2.append('steelblue')
            elif 'heat' in cat.lower():
                colors_phase1.append('lightcoral')
                colors_phase2.append('darkred')
            else:  # No thermal stress
                colors_phase1.append('lightgray')
                colors_phase2.append('gray')
        
        bars1 = ax.bar(x - width/2, phase1_values, width, alpha=0.8, color=colors_phase1)
        bars2 = ax.bar(x + width/2, phase2_values, width, alpha=0.8, color=colors_phase2)
        
        ax.set_xlabel('UTCI Thermal Stress Category', fontsize=12)
        ax.set_ylabel('Infrastructure Resilience Factor', fontsize=12)
        ax.set_title('Comprehensive UTCI Thermal Stress Analysis:\nInfrastructure Protection Across All Thermal Conditions\n(Left bars: 2023, Right bars: 2024)', fontsize=14)
        ax.set_xticks(x)
        
        # Improve label readability with rotation and shorter labels
        short_labels = []
        for cat in utci_cats:
            if cat == 'No thermal stress':
                short_labels.append('No Stress')
            elif 'Moderate cold' in cat:
                short_labels.append('Mod. Cold')
            elif 'Strong cold' in cat and 'Very strong' not in cat:
                short_labels.append('Strong Cold')
            elif 'Very strong cold' in cat:
                short_labels.append('V.Strong Cold')
            elif 'Moderate heat' in cat:
                short_labels.append('Mod. Heat')
            elif 'Strong heat' in cat:
                short_labels.append('Strong Heat')
            else:
                short_labels.append(cat)
        
        ax.set_xticklabels(short_labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        

        
        # Add value labels on bars for better readability
        for i, (bar1, bar2, val1, val2) in enumerate(zip(bars1, bars2, phase1_values, phase2_values)):
            if abs(val1) > 0.1:  # Only show labels for significant values
                ax.text(bar1.get_x() + bar1.get_width()/2., val1 + 0.05 if val1 > 0 else val1 - 0.05, 
                       f'{val1:.2f}', ha='center', va='bottom' if val1 > 0 else 'top', fontsize=9)
            if abs(val2) > 0.1:
                ax.text(bar2.get_x() + bar2.get_width()/2., val2 + 0.05 if val2 > 0 else val2 - 0.05, 
                       f'{val2:.2f}', ha='center', va='bottom' if val2 > 0 else 'top', fontsize=9)
        
        # Add section separators
        ax.axvline(x=0.5, color='black', linestyle=':', alpha=0.3)  # After "No stress"
        
        # Find the position between cold and heat stress categories
        cold_stress_count = sum(1 for cat in utci_cats if 'cold' in cat.lower())
        separator_pos = cold_stress_count + 0.5  # Position after all cold stress categories
        ax.axvline(x=separator_pos, color='black', linestyle=':', alpha=0.3)  # Between cold and heat
        
        # Add section labels - center them properly within their sections
        cold_center = (1 + cold_stress_count) / 2  # Center of cold stress section
        heat_center = (cold_stress_count + 1 + len(utci_cats) - 1) / 2  # Center of heat stress section
        
        ax.text(cold_center, max(max(phase1_values), max(phase2_values)) * 0.9, 'COLD STRESS', 
               ha='center', va='center', fontweight='bold', fontsize=10, alpha=0.7)
        ax.text(heat_center, max(max(phase1_values), max(phase2_values)) * 0.9, 'HEAT STRESS', 
               ha='center', va='center', fontweight='bold', fontsize=10, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "utci_thermal_stress.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_network_spillover_effects(self, network_results):
        """Plot precise network spillover effects analysis as multiple comprehensive files."""
        
        # Plot 1: Network multiplier comparison
        fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        
        phases = ['Phase 1 (2023)', 'Phase 2 (2024)']
        multipliers = [
            network_results['phase_1_2023']['network_multiplier'],
            network_results['phase_2_2024']['network_multiplier']
        ]
        
        bars = ax1.bar(phases, multipliers, color=['lightblue', 'lightgreen'], alpha=0.8)
        ax1.set_ylabel('Network Effect Multiplier')
        ax1.set_title('Precise Network Effect Multiplier by Phase\n(Based on Street-Specific Influence Zones)')
        ax1.grid(True, alpha=0.3)
        
        for bar, value in zip(bars, multipliers):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{value:.2f}x', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "network_multiplier_effects.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Precise spillover zones impact
        fig2, ax2 = plt.subplots(1, 1, figsize=(12, 6))
        
        zones = ['Direct Impact\n(≤30% of influence)', 'Spillover 1\n(30-60% of influence)', 'Spillover 2\n(60-150% of influence)']
        gains = [
            network_results['phase_2_2024']['direct_impact_gain'],
            network_results['phase_2_2024']['spillover_1_gain'],
            network_results['phase_2_2024']['spillover_2_gain']
        ]
        
        bars = ax2.bar(zones, gains, color=['darkgreen', 'orange', 'lightcoral'], alpha=0.8)
        ax2.set_ylabel('Ridership Gain (rides/month)')
        ax2.set_title('Precise Spatial Impact Zones:\nRidership Gains by Relative Distance to Street Infrastructure')
        ax2.grid(True, alpha=0.3)
        
        for bar, value in zip(bars, gains):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'+{value:.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "spatial_spillover_zones.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 3: NEW - Impact strength distribution
        fig3, ax3 = plt.subplots(1, 1, figsize=(12, 6))
        
        impact_dist = network_results['phase_2_2024']['impact_distribution']
        categories = ['High Impact\n(>0.7 strength)', 'Medium Impact\n(0.3-0.7 strength)', 'Low Impact\n(0-0.3 strength)']
        counts = [
            impact_dist['high_impact_stations'],
            impact_dist['medium_impact_stations'], 
            impact_dist['low_impact_stations']
        ]
        
        bars = ax3.bar(categories, counts, color=['darkred', 'orange', 'yellow'], alpha=0.8)
        ax3.set_ylabel('Number of Stations')
        ax3.set_title('Infrastructure Impact Strength Distribution\n(Based on Precise Distance to Street Centerlines)')
        ax3.grid(True, alpha=0.3)
        
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{count}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "impact_strength_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 4: NEW - Zone composition comparison
        fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Phase 1 zone counts
        zone_counts_2023 = network_results['phase_1_2023']['zone_counts']
        labels_2023 = list(zone_counts_2023.keys())
        values_2023 = list(zone_counts_2023.values())
        colors = ['darkgreen', 'orange', 'lightcoral', 'lightgray']
        
        ax4a.pie(values_2023, labels=labels_2023, colors=colors, autopct='%1.1f%%', startangle=90)
        ax4a.set_title('2023 Station Distribution by Spillover Zone')
        
        # Phase 2 zone counts  
        zone_counts_2024 = network_results['phase_2_2024']['zone_counts']
        labels_2024 = list(zone_counts_2024.keys())
        values_2024 = list(zone_counts_2024.values())
        
        ax4b.pie(values_2024, labels=labels_2024, colors=colors, autopct='%1.1f%%', startangle=90)
        ax4b.set_title('2024 Station Distribution by Spillover Zone')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "spillover_zone_composition.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_seasonal_validation(self, seasonal_results):
        """Plot seasonal validation and monthly accuracy for all 4 model combinations."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 12))
        
        # Monthly data and month labels
        monthly_data = seasonal_results['monthly_accuracy']
        months = sorted(monthly_data.keys())
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_labels = [month_names[m-1] for m in months]
        
        # Model configurations
        model_configs = [
            ('Baseline 2022→2023', 'blue', 'o', '--'),
            ('Enhanced 2022→2023', 'darkblue', 'o', '-'),
            ('Baseline 2022→2024', 'red', 's', '--'),
            ('Enhanced 2022→2024', 'darkred', 's', '-')
        ]
        
        # Plot MAE for all models
        for model_name, color, marker, linestyle in model_configs:
            mae_values = []
            for month in months:
                if model_name in monthly_data[month]:
                    mae_values.append(monthly_data[month][model_name]['mae'])
                else:
                    mae_values.append(np.nan)
            
            axes[0].plot(month_labels, mae_values, marker=marker, linewidth=2, markersize=6, 
                        color=color, linestyle=linestyle, label=model_name, alpha=0.8)
        
        axes[0].set_ylabel('Mean Absolute Error (rides/month)')
        axes[0].set_title('Four-Model Seasonal Validation: MAE Performance')
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot MAPE for all models
        for model_name, color, marker, linestyle in model_configs:
            mape_values = []
            for month in months:
                if model_name in monthly_data[month]:
                    mape_values.append(monthly_data[month][model_name]['mape'])
                else:
                    mape_values.append(np.nan)
            
            axes[1].plot(month_labels, mape_values, marker=marker, linewidth=2, markersize=6, 
                        color=color, linestyle=linestyle, label=model_name, alpha=0.8)
        
        axes[1].set_xlabel('Month')
        axes[1].set_ylabel('Mean Absolute Percentage Error (%)')
        axes[1].set_title('Four-Model Seasonal Validation: MAPE Performance')
        axes[1].grid(True, alpha=0.3)
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "seasonal_validation.png"), dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main execution function."""
    print("Starting Clean Infrastructure Impact Analysis...")
    
    # Initialize analyzer
    analyzer = CleanInfrastructureAnalyzer()
    
    # Define file paths (work from both src directory and project root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    citibike_2022_path = os.path.join(project_root, "data", "combined", "2022_combined_citibike_weather.parquet")
    citibike_2023_path = os.path.join(project_root, "data", "combined", "2023_combined_citibike_weather.parquet")
    citibike_2024_path = os.path.join(project_root, "data", "combined", "2024_combined_citibike_weather.parquet")
    infrastructure_path = os.path.join(project_root, "data", "nyc_streets_geocoded_with_years.csv")
    
    # Debug: Print the actual paths being used
    print(f"Project root: {project_root}")
    print(f"Infrastructure path: {infrastructure_path}")
    print(f"Infrastructure file exists: {os.path.exists(infrastructure_path)}")
    
    try:
        # Load data
        analyzer.load_data(
            citibike_2022_path=citibike_2022_path,
            citibike_2023_path=citibike_2023_path,
            citibike_2024_path=citibike_2024_path,
            infrastructure_path=infrastructure_path
        )
        
        # Run analysis
        results = analyzer.run_analysis()
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"Infrastructure Effect: {results['average_infrastructure_effect']:.1f} rides/month")
        print(f"Effect Consistency: {results['effect_consistency']:.3f}")
        print(f"Model Improvement: {results['average_model_improvement']:.3f} R²")
        print(f"Affected Stations (2022): {results['affected_stations_2022']}")
        print(f"Affected Stations (2023+): {results['affected_stations_2023_plus']}")
        print(f"Results saved to: {analyzer.output_dir}")
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()