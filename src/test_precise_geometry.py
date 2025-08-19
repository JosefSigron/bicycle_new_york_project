#!/usr/bin/env python3
"""
Test script for precise geometric infrastructure analysis.

Tests the new geometric approach that uses exact street coordinates
instead of simple buffer zones.
"""

import pandas as pd
import sys
import os

# Add src directory to path
sys.path.append('.')

from precise_street_analyzer import PreciseStreetAnalyzer

def test_precise_analyzer():
    """Test the PreciseStreetAnalyzer functionality."""
    print("Testing Precise Street Geometric Analyzer...")
    
    try:
        # Initialize analyzer with correct path
        analyzer = PreciseStreetAnalyzer("../data/street_locations/lion.gdb")
        
        # Test loading street geometries (disable cache for debugging)
        print("\n1. Testing street geometry loading...")
        streets = analyzer.load_street_geometries(use_cache=False)
        
        if streets is not None and len(streets) > 0:
            print(f"✓ Successfully loaded {len(streets):,} street geometries")
            print(f"✓ Street classes: {streets['street_class'].value_counts().to_dict()}")
            print(f"✓ Geometry types: {streets.geometry.type.value_counts().to_dict()}")
        else:
            print("✗ Failed to load street geometries")
            return False
        
        # Test precise infrastructure impact analysis
        print("\n2. Testing precise infrastructure impact analysis...")
        
        # Create sample infrastructure data (NYC coordinates)
        sample_infrastructure = pd.DataFrame({
            'latitude': [40.7589, 40.7614, 40.7505],  # Times Square, Central Park, Brooklyn Bridge areas
            'longitude': [-73.9851, -73.9776, -73.9934],
            'year': [2022, 2023, 2023],
            'infrastructure_attributes': ['', 'protected', 'unprotected']
        })
        
        # Create sample stations data
        sample_stations = pd.DataFrame({
            'start_station_id': [1, 2, 3, 4, 5],
            'start_station_latitude': [40.7580, 40.7620, 40.7500, 40.7590, 40.7510],
            'start_station_longitude': [-73.9860, -73.9770, -73.9940, -73.9840, -73.9920]
        })
        
        try:
            enhanced_infra, affected_stations = analyzer.find_precise_infrastructure_impact(
                sample_infrastructure, sample_stations
            )
            print("✓ Infrastructure impact analysis completed")
        except Exception as e:
            print(f"✗ Error in infrastructure impact analysis: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        if len(enhanced_infra) > 0 and len(affected_stations) > 0:
            print("✓ Successfully analyzed precise infrastructure impact")
            print(f"✓ Enhanced infrastructure records: {len(enhanced_infra)}")
            print(f"✓ Affected station details: {len(affected_stations)}")
            print(f"✓ Unique affected stations: {affected_stations['station_id'].nunique()}")
            
            # Show sample impact details
            print("\nSample impact analysis:")
            for idx, impact in affected_stations.head(3).iterrows():
                print(f"  Station {impact['station_id']}: {impact['exact_distance_to_street']:.1f}m from {impact['street_class']} street")
                print(f"    Impact strength: {impact['impact_strength']:.3f}, Street: {impact['street_name']}")
        else:
            print("✗ Failed to analyze infrastructure impact")
            return False
        
        # Test precise feature creation
        print("\n3. Testing precise feature creation...")
        
        precise_features = analyzer.create_precise_features(
            sample_stations, enhanced_infra, affected_stations, "2023"
        )
        
        new_feature_cols = [
            'precise_near_infrastructure', 'exact_distance_to_nearest_street',
            'impact_strength', 'infrastructure_count_in_influence',
            'on_major_street_infra', 'on_arterial_street_infra', 'on_local_street_infra'
        ]
        
        if all(col in precise_features.columns for col in new_feature_cols):
            print("✓ Successfully created precise geometric features")
            print(f"✓ Total features: {len(precise_features.columns)}")
            print(f"✓ Stations with infrastructure impact: {precise_features['precise_near_infrastructure'].sum()}")
            
            # Show feature summary
            print("\nFeature summary:")
            print(f"  Average impact strength: {precise_features['impact_strength'].mean():.3f}")
            print(f"  Max infrastructure count: {precise_features['infrastructure_count_in_influence'].max()}")
            print(f"  Street class distribution: {precise_features[['on_major_street_infra', 'on_arterial_street_infra', 'on_local_street_infra']].sum().to_dict()}")
        else:
            missing_cols = [col for col in new_feature_cols if col not in precise_features.columns]
            print(f"✗ Missing expected features: {missing_cols}")
            return False
        
        print("\n✓ All precise geometry tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_integration_with_main_analysis():
    """Test integration with the main analysis."""
    print("\n" + "="*50)
    print("Testing Integration with Main Analysis...")
    
    try:
        # Test importing the enhanced analyzer
        from clean_infrastructure_analysis import CleanInfrastructureAnalyzer
        
        # Initialize analyzer
        analyzer = CleanInfrastructureAnalyzer()
        
        # Check if precise analyzer is initialized
        if hasattr(analyzer, 'precise_analyzer'):
            print("✓ CleanInfrastructureAnalyzer has precise_analyzer attribute")
        else:
            print("✗ Missing precise_analyzer attribute")
            return False
        
        # Check if the precise analyzer can be initialized
        if analyzer.precise_analyzer.lion_path:
            print(f"✓ Precise analyzer configured with LION path: {analyzer.precise_analyzer.lion_path}")
        else:
            print("✗ Precise analyzer not properly configured")
            return False
        
        print("✓ Integration test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*70)
    print("PRECISE GEOMETRIC STREET ANALYSIS TEST")
    print("="*70)
    
    # Check if LION data exists (adjust path for src directory)
    lion_path = "../data/street_locations/lion.gdb"
    if not os.path.exists(lion_path):
        print(f"✗ LION data not found at {lion_path}")
        print("Please ensure the NYC LION data is downloaded to data/street_locations/")
        sys.exit(1)
    
    print(f"✓ LION data found at {lion_path}")
    
    # Run tests
    test1_passed = test_precise_analyzer()
    test2_passed = test_integration_with_main_analysis()
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Precise Geometry Test: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"Integration Test: {'PASSED' if test2_passed else 'FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Your infrastructure analysis now uses precise street geometry.")
        print("\nKey improvements:")
        print("✓ Exact distance calculations to street centerlines")
        print("✓ Precise geometric impact zones based on street characteristics")
        print("✓ Impact strength calculations (closer to street = stronger impact)")
        print("✓ Accurate identification of which specific street affects each station")
        print("✓ No more crude buffer zones - uses actual street geometries")
        print("\nNext steps:")
        print("1. Run: python src/clean_infrastructure_analysis.py")
        print("2. Compare results with previous buffer-based analysis")
        print("3. Analyze new precise features like 'impact_strength' and 'exact_distance_to_nearest_street'")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)
