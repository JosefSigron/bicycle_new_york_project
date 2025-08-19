#!/usr/bin/env python3
"""
Test script for NYC street data integration with infrastructure analysis.

This tests the new street classification features without running the full analysis.
"""

import pandas as pd
import sys
import os

# Add src directory to path
sys.path.append('src')

from nyc_street_loader import NYCStreetLoader

def test_street_loader():
    """Test the NYCStreetLoader functionality."""
    print("Testing NYC Street Loader...")
    
    try:
        # Initialize loader
        loader = NYCStreetLoader()
        
        # Test loading streets (will use cache if available)
        print("\n1. Testing street data loading...")
        streets = loader.load_lion_streets()
        
        if streets is not None and len(streets) > 0:
            print(f"✓ Successfully loaded {len(streets):,} street segments")
            print(f"✓ Street classes: {streets['street_class'].value_counts().to_dict()}")
        else:
            print("✗ Failed to load street data")
            return False
        
        # Test infrastructure enhancement
        print("\n2. Testing infrastructure enhancement...")
        
        # Create sample infrastructure data
        sample_infrastructure = pd.DataFrame({
            'latitude': [40.7589, 40.7614, 40.7505],  # NYC coordinates
            'longitude': [-73.9851, -73.9776, -73.9934],
            'year': [2022, 2023, 2023],
            'infrastructure_attributes': ['', 'protected', 'unprotected']
        })
        
        enhanced_infra = loader.get_buffer_size_for_infrastructure(sample_infrastructure)
        
        if 'street_class' in enhanced_infra.columns:
            print("✓ Successfully enhanced infrastructure with street data")
            print(f"✓ Buffer sizes: {enhanced_infra['buffer_meters'].value_counts().to_dict()}")
            print(f"✓ Street classes: {enhanced_infra['street_class'].value_counts().to_dict()}")
        else:
            print("✗ Failed to enhance infrastructure data")
            return False
        
        print("\n✓ All tests passed! Street integration is working correctly.")
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_integration_compatibility():
    """Test compatibility with the main analysis file."""
    print("\n" + "="*50)
    print("Testing Integration Compatibility...")
    
    try:
        # Test importing the enhanced analyzer
        from clean_infrastructure_analysis import CleanInfrastructureAnalyzer
        
        # Initialize analyzer
        analyzer = CleanInfrastructureAnalyzer()
        
        # Check if street loader is initialized
        if hasattr(analyzer, 'street_loader'):
            print("✓ CleanInfrastructureAnalyzer has street_loader attribute")
        else:
            print("✗ Missing street_loader attribute")
            return False
        
        print("✓ Integration compatibility test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*70)
    print("NYC STREET DATA INTEGRATION TEST")
    print("="*70)
    
    # Check if LION data exists
    lion_path = "data/street_locations/lion.gdb"
    if not os.path.exists(lion_path):
        print(f"✗ LION data not found at {lion_path}")
        print("Please ensure the NYC LION data is downloaded to data/street_locations/")
        sys.exit(1)
    
    print(f"✓ LION data found at {lion_path}")
    
    # Run tests
    test1_passed = test_street_loader()
    test2_passed = test_integration_compatibility()
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Street Loader Test: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"Integration Test: {'PASSED' if test2_passed else 'FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Your infrastructure analysis is ready to use street classification.")
        print("\nNext steps:")
        print("1. Run: python src/clean_infrastructure_analysis.py")
        print("2. The analysis will now use street-specific buffer sizes")
        print("3. Enhanced models will include street classification features")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)
