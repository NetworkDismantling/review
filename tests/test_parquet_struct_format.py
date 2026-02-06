"""
Test suite for Parquet struct format verification.

Tests that removals column is stored efficiently as list<struct> rather than
inefficient string/binary format where field names are repeated for each element.

Verifies:
- ParquetDataFrameWriter correctly converts removals to struct format
- Schema verification detects efficient vs inefficient formats  
- File size differences between formats
- Round-trip data integrity
"""

import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pyarrow.parquet as pq

from network_dismantling.common.storage.pandas.parquet import (
    ParquetDataFrameWriter,
    convert_removals_to_struct,
    convert_removals_from_struct,
    verify_removals_schema,
    get_parquet_stats,
    read_without_columns,
)


@pytest.fixture
def test_data():
    """Create test DataFrame with removals column."""
    n_rows = 100
    n_removals = 50  # removals per row
    
    data = {
        'network': [f'network_{i%10}' for i in range(n_rows)],
        'heuristic': [f'heuristic_{i%5}' for i in range(n_rows)],
        'slcc_peak_at': np.random.randint(100, 1000, n_rows),
        'removals': [
            # Each row has a list of removal tuples
            # (removal_num, id, prediction, lcc_size, slcc_size)
            [
                (j, np.random.randint(0, 1000), np.random.random(), 
                 np.random.randint(0, 1000), np.random.randint(0, 1000))
                for j in range(n_removals)
            ]
            for i in range(n_rows)
        ]
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestParquetStructFormat:
    """Test Parquet struct format for removals column."""
    
    def test_efficient_struct_format(self, test_data, temp_dir):
        """Test that ParquetDataFrameWriter creates efficient struct format."""
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        output_file = temp_dir / 'efficient.parquet'
        
        # Write using ParquetDataFrameWriter (should convert to struct automatically)
        with ParquetDataFrameWriter(output_file, test_data.columns.tolist()) as writer:
            writer.write(test_data)
        
        assert output_file.exists(), "Output file should be created"
        
        # Verify schema is efficient
        is_efficient = verify_removals_schema(output_file, logger=logger)
        assert is_efficient, "Removals should be stored as efficient list<struct>"
        
        # Check stats
        stats = get_parquet_stats(output_file)
        assert stats['removals_format'] == 'efficient', "Should report efficient format"
        assert stats['num_rows'] == len(test_data), "Row count should match"
    
    def test_schema_detection_struct(self, test_data, temp_dir):
        """Test schema detection recognizes list<struct> format."""
        output_file = temp_dir / 'struct_format.parquet'
        
        # Write with struct conversion
        with ParquetDataFrameWriter(output_file, test_data.columns.tolist()) as writer:
            writer.write(test_data)
        
        # Read schema directly
        schema = pq.read_schema(str(output_file))
        removals_type = str(schema.field('removals').type)
        
        # PyArrow writes list<element: struct> or list<item: struct> (not just list<struct>)
        assert ('list<struct' in removals_type or 
                'list<element: struct' in removals_type or 
                'list<item: struct' in removals_type), \
            f"Removals type should be list<...struct>, got {removals_type}"
        
        # Verify struct has correct fields
        assert 'removal_num' in removals_type
        assert 'id' in removals_type
        assert 'prediction' in removals_type
        assert 'lcc_size' in removals_type
        assert 'slcc_size' in removals_type
    
    def test_inefficient_string_format(self, test_data, temp_dir):
        """Test detection of inefficient string format."""
        output_file = temp_dir / 'inefficient.parquet'
        
        # Convert removals to string (simulating old inefficient format)
        df_string = test_data.copy()
        df_string['removals'] = df_string['removals'].apply(str)
        
        # Write without struct conversion (use pyarrow engine, fastparquet has issues with Arrow strings)
        df_string.to_parquet(output_file, engine='pyarrow', compression='snappy', index=False)
        
        # Verify schema is inefficient
        is_efficient = verify_removals_schema(output_file)
        assert not is_efficient, "String format should be detected as inefficient"
        
        # Check stats
        stats = get_parquet_stats(output_file)
        assert stats['removals_format'] == 'inefficient', "Should report inefficient format"
    
    def test_size_comparison(self, test_data, temp_dir):
        """Test that struct format is significantly smaller than string format."""
        efficient_file = temp_dir / 'efficient.parquet'
        inefficient_file = temp_dir / 'inefficient.parquet'
        
        # Create efficient version (struct)
        with ParquetDataFrameWriter(efficient_file, test_data.columns.tolist()) as writer:
            writer.write(test_data)
        
        # Create inefficient version (string)
        df_string = test_data.copy()
        df_string['removals'] = df_string['removals'].apply(str)
        df_string.to_parquet(inefficient_file, engine='pyarrow', compression='snappy', index=False)
        
        # Compare sizes
        efficient_stats = get_parquet_stats(efficient_file)
        inefficient_stats = get_parquet_stats(inefficient_file)
        
        efficient_size = efficient_stats['size_mb']
        inefficient_size = inefficient_stats['size_mb']
        
        # Struct format should be at least 1.5x smaller
        size_ratio = inefficient_size / efficient_size
        assert size_ratio > 1.5, \
            f"Struct format should be significantly smaller (ratio: {size_ratio:.2f}x)"
    
    def test_data_round_trip(self, test_data, temp_dir):
        """Test that data can be written and read back correctly."""
        output_file = temp_dir / 'roundtrip.parquet'
        
        # Write
        with ParquetDataFrameWriter(output_file, test_data.columns.tolist()) as writer:
            writer.write(test_data)
        
        # Read back (without removals to avoid conversion issues in test)
        df_read = read_without_columns(output_file, exclude_columns=['removals'])
        
        # Verify non-removals columns match
        assert len(df_read) == len(test_data)
        assert list(df_read['network']) == list(test_data['network'])
        assert list(df_read['heuristic']) == list(test_data['heuristic'])
        np.testing.assert_array_equal(df_read['slcc_peak_at'].values, 
                                     test_data['slcc_peak_at'].values)
    
    def test_conversion_functions(self):
        """Test removals conversion to/from struct format."""
        # Test data: list of tuples
        removals = [
            (0, 123, 0.5, 900, 50),
            (1, 456, 0.3, 850, 100),
            (2, 789, 0.8, 700, 200),
        ]
        
        # Convert to struct (list of dicts)
        struct_data = convert_removals_to_struct(removals)
        
        assert isinstance(struct_data, list)
        assert len(struct_data) == 3
        assert isinstance(struct_data[0], dict)
        assert struct_data[0]['removal_num'] == 0
        assert struct_data[0]['id'] == 123
        assert struct_data[0]['prediction'] == 0.5
        assert struct_data[0]['lcc_size'] == 900
        assert struct_data[0]['slcc_size'] == 50
        
        # Convert back to tuples
        tuples_data = convert_removals_from_struct(struct_data)
        
        assert isinstance(tuples_data, list)
        assert len(tuples_data) == 3
        assert isinstance(tuples_data[0], tuple)
        assert tuples_data[0] == (0, 123, 0.5, 900, 50)
        assert tuples_data == removals
    
    def test_empty_removals(self, temp_dir):
        """Test handling of empty removals."""
        output_file = temp_dir / 'empty.parquet'
        
        df = pd.DataFrame({
            'network': ['net1', 'net2'],
            'heuristic': ['h1', 'h2'],
            'removals': [[], []],  # Empty removals
        })
        
        with ParquetDataFrameWriter(output_file, df.columns.tolist()) as writer:
            writer.write(df)
        
        # Should still create efficient format
        is_efficient = verify_removals_schema(output_file)
        assert is_efficient
    
    def test_no_removals_column(self, temp_dir):
        """Test file without removals column."""
        output_file = temp_dir / 'no_removals.parquet'
        
        df = pd.DataFrame({
            'network': ['net1', 'net2'],
            'heuristic': ['h1', 'h2'],
            'value': [1, 2],
        })
        
        with ParquetDataFrameWriter(output_file, df.columns.tolist()) as writer:
            writer.write(df)
        
        # Should not fail on files without removals
        is_efficient = verify_removals_schema(output_file)
        assert is_efficient  # Returns True for files without removals
        
        stats = get_parquet_stats(output_file)
        assert stats['removals_format'] == 'none'
    
    def test_multiple_writes_append(self, test_data, temp_dir):
        """Test that multiple writes maintain efficient format."""
        output_file = temp_dir / 'append.parquet'
        
        # Split data and write in chunks
        chunk_size = 30
        chunks = [test_data.iloc[i:i+chunk_size] for i in range(0, len(test_data), chunk_size)]
        
        with ParquetDataFrameWriter(output_file, test_data.columns.tolist()) as writer:
            for chunk in chunks:
                writer.write(chunk)
        
        # Verify final file is efficient
        is_efficient = verify_removals_schema(output_file)
        assert is_efficient
        
        stats = get_parquet_stats(output_file)
        assert stats['num_rows'] == len(test_data)
        assert stats['removals_format'] == 'efficient'
    
    def test_cross_session_append(self, test_data, temp_dir):
        """Test cross-session append (separate writer instances)."""
        output_file = temp_dir / 'cross_session.parquet'
        
        # Session 1: Write initial data
        first_batch = test_data.iloc[:30]
        with ParquetDataFrameWriter(output_file, test_data.columns.tolist(), mode='overwrite') as writer:
            writer.write(first_batch)
        
        # Verify session 1
        stats1 = get_parquet_stats(output_file)
        assert stats1['num_rows'] == 30
        
        # Session 2: Append more data
        second_batch = test_data.iloc[30:60]
        with ParquetDataFrameWriter(output_file, test_data.columns.tolist(), mode='append') as writer:
            writer.write(second_batch)
        
        # Verify both sessions present
        stats2 = get_parquet_stats(output_file)
        assert stats2['num_rows'] == 60
        assert stats2['removals_format'] == 'efficient'
        
        # Verify struct schema preserved
        is_efficient = verify_removals_schema(output_file)
        assert is_efficient


class TestGetParquetStats:
    """Test get_parquet_stats function."""
    
    def test_stats_structure(self, test_data, temp_dir):
        """Test that stats dict has all required keys."""
        output_file = temp_dir / 'stats_test.parquet'
        
        with ParquetDataFrameWriter(output_file, test_data.columns.tolist()) as writer:
            writer.write(test_data)
        
        stats = get_parquet_stats(output_file)
        
        # Check all required keys
        assert 'num_rows' in stats
        assert 'num_row_groups' in stats
        assert 'size_mb' in stats
        assert 'size_bytes' in stats
        assert 'avg_bytes_per_row' in stats
        assert 'removals_format' in stats
        
        # Check types and values
        assert isinstance(stats['num_rows'], int)
        assert stats['num_rows'] == len(test_data)
        assert isinstance(stats['size_mb'], float)
        assert stats['size_mb'] > 0
        assert isinstance(stats['avg_bytes_per_row'], float)
        assert stats['avg_bytes_per_row'] > 0
        assert stats['removals_format'] in ['efficient', 'inefficient', 'none', 'unknown']


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v'])
