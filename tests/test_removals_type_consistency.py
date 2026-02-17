"""Test removals column type consistency across the codebase."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from network_dismantling.common.removal import Removal


def test_removals_as_numpy_array():
    """Test that removals column is consistently np.ndarray."""
    # Create sample removals from dismantler output
    removals_list = [
        Removal(1, 10, 0.9, 100, 5),
        Removal(2, 20, 0.8, 95, 8),
        Removal(3, 30, 0.7, 90, 12),
    ]
    
    # Convert to numpy array (as done in production code)
    removals_array = np.array(removals_list, dtype=object)
    
    # Verify it's a numpy array
    assert isinstance(removals_array, np.ndarray)
    assert removals_array.dtype == object
    assert len(removals_array) == 3
    
    # Verify we can iterate over it
    for removal in removals_array:
        assert isinstance(removal, Removal)
    
    # Verify we can index it
    assert removals_array[0].node_id == 10
    assert removals_array[1].lcc_size == 95
    
    # Verify we can convert to list if needed
    removals_as_list = removals_array.tolist()
    assert isinstance(removals_as_list, list)
    assert len(removals_as_list) == 3


def test_removals_in_dataframe():
    """Test that removals column works in DataFrame as np.ndarray."""
    removals1 = np.array([
        Removal(1, 10, 0.9, 100, 5),
        Removal(2, 20, 0.8, 95, 8),
    ], dtype=object)
    
    removals2 = np.array([
        Removal(1, 15, 0.95, 100, 4),
        Removal(2, 25, 0.85, 94, 9),
    ], dtype=object)
    
    # Create DataFrame with removals as numpy arrays
    df = pd.DataFrame({
        'network': ['net1', 'net2'],
        'heuristic': ['h1', 'h2'],
        'removals': [removals1, removals2],
        'rem_num': [2, 2],
    })
    
    # Verify type is preserved in DataFrame
    assert isinstance(df['removals'].iloc[0], np.ndarray)
    assert isinstance(df['removals'].iloc[1], np.ndarray)
    
    # Verify we can iterate over removals from DataFrame
    for idx, row in df.iterrows():
        removals = row['removals']
        assert isinstance(removals, np.ndarray)
        assert len(removals) == 2
        
        for removal in removals:
            assert isinstance(removal, Removal)


def test_removals_empty_array():
    """Test empty removals are handled correctly."""
    # Empty array (as done in CoreGDM when rem_num == 0)
    empty_removals = np.array([], dtype=object)
    
    assert isinstance(empty_removals, np.ndarray)
    assert len(empty_removals) == 0
    
    # Create DataFrame with empty removals
    df = pd.DataFrame({
        'network': ['net1'],
        'removals': [empty_removals],
        'rem_num': [0],
    })
    
    assert isinstance(df['removals'].iloc[0], np.ndarray)
    assert len(df['removals'].iloc[0]) == 0


def test_removals_conversion_from_list():
    """Test that list can be easily converted to numpy array."""
    # Simulate output from dismantler (List[Removal])
    removals_list = [
        Removal(1, 10, 0.9, 100, 5),
        Removal(2, 20, 0.8, 95, 8),
    ]
    
    # Convert to numpy array (production code pattern)
    removals_array = np.array(removals_list, dtype=object)
    
    # Verify conversion worked
    assert isinstance(removals_array, np.ndarray)
    assert len(removals_array) == 2
    assert all(isinstance(r, Removal) for r in removals_array)


def test_removals_compatibility_with_operations():
    """Test that numpy array removals work with common operations."""
    removals = np.array([
        Removal(1, 10, 0.9, 100, 5),
        Removal(2, 20, 0.8, 95, 8),
        Removal(3, 30, 0.7, 90, 12),
    ], dtype=object)
    
    # Test max() with lambda (common pattern in codebase for Removal objects)
    peak_slcc = max(removals, key=lambda r: r.slcc_size)
    assert peak_slcc.slcc_size == 12
    
    # Test list comprehension (common pattern)
    lcc_sizes = [r.lcc_size for r in removals]
    assert lcc_sizes == [100, 95, 90]
    
    # Test len()
    assert len(removals) == 3
    
    # Test indexing with integers (Removal supports this via __getitem__)
    assert removals[0].node_id == 10
    assert removals[-1].node_id == 30
    
    # Test slicing
    first_two = removals[:2]
    assert len(first_two) == 2
    assert first_two[0].node_id == 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
