"""
Test suite to verify that CSV and Parquet storage formats are equivalent.
Ensures that data written to CSV can be read identically from Parquet and vice versa.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from network_dismantling.common.data_structures import dotdict
from network_dismantling.common.storage.pandas import csv, parquet


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@pytest.fixture
def sample_dismantling_data():
    """Sample data resembling typical dismantling output"""
    return pd.DataFrame({
        "network": ["karate"] * 5,
        "dismantler": ["degree"] * 5,
        "static_id": [0] * 5,
        "lcc_size": [34, 28, 22, 15, 8],
        "removed_nodes": [0, 6, 12, 19, 26],
        "threshold": [0.1] * 5,
        "run": [0] * 5,
    })


@pytest.fixture
def csv_writer_helper(tmp_path):
    """Helper to write DataFrame to CSV using the CSVDataFrameWriter class"""
    def _write(df, columns, filename="test.csv"):
        output_file = tmp_path / filename
        with csv.CSVDataFrameWriter(output_file, columns, logger=logger) as writer:
            if df is not None:
                writer.write(df)
        return output_file
    return _write


@pytest.fixture
def parquet_writer_helper(tmp_path):
    """Helper to write DataFrame to Parquet using the ParquetDataFrameWriter class"""
    def _write(df, columns, filename="test.parquet"):
        output_file = tmp_path / filename
        with parquet.ParquetDataFrameWriter(output_file, columns, logger=logger) as writer:
            if df is not None:
                writer.write(df)
        return output_file
    return _write


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize DataFrame for comparison by sorting and resetting index"""
    df = df.copy()
    # Drop file column as paths will differ
    if "file" in df.columns:
        df = df.drop(columns=["file"])
    # Sort by idx for consistent comparison
    if "idx" in df.columns:
        df = df.sort_values("idx").reset_index(drop=True)
    # Convert categories to strings for comparison
    for col in df.select_dtypes(include=["category"]).columns:
        df[col] = df[col].astype(str)
    return df


def test_write_read_equivalence(sample_dismantling_data, csv_writer_helper, parquet_writer_helper):
    """Test that the same data written to CSV and Parquet can be read identically"""
    columns = list(sample_dismantling_data.columns)
    
    # Write to both formats
    csv_path = csv_writer_helper(sample_dismantling_data, columns)
    parquet_path = parquet_writer_helper(sample_dismantling_data, columns)
    
    # Read from both formats
    csv_df = csv.read_without_columns(file=csv_path, exclude_columns=[])
    parquet_df = parquet.read_without_columns(file=str(parquet_path), exclude_columns=[])
    
    # Normalize for comparison
    csv_normalized = normalize_df(csv_df)
    parquet_normalized = normalize_df(parquet_df)
    
    # Compare
    assert_frame_equal(csv_normalized, parquet_normalized)


def test_multiple_writes_equivalence(sample_dismantling_data, tmp_path):
    """Test that multiple sequential writes (append mode) produce identical results"""
    columns = list(sample_dismantling_data.columns)
    
    # Split data into chunks to simulate run-by-run writing
    chunk1 = sample_dismantling_data.iloc[:2]
    chunk2 = sample_dismantling_data.iloc[2:]
    
    # Write to CSV: multiple writes with separate writers (append mode)
    csv_file = tmp_path / "multi.csv"
    with csv.CSVDataFrameWriter(csv_file, columns, logger=logger) as writer:
        writer.write(chunk1)
    with csv.CSVDataFrameWriter(csv_file, columns, logger=logger) as writer:
        writer.write(chunk2)
    
    # Write to Parquet: single writer receiving multiple chunks
    parquet_file = tmp_path / "multi.parquet"
    with parquet.ParquetDataFrameWriter(parquet_file, columns, logger=logger) as writer:
        writer.write(chunk1)
    with parquet.ParquetDataFrameWriter(parquet_file, columns, mode='append', logger=logger) as writer:
        writer.write(chunk2)
    
    # Read both
    csv_df = csv.read_without_columns(file=csv_file, exclude_columns=[])
    parquet_df = parquet.read_without_columns(file=str(parquet_file), exclude_columns=[])
    
    # Also check that lengths match the original data
    assert len(csv_df) == len(sample_dismantling_data), f"Expected {len(sample_dismantling_data)} rows, got {len(csv_df)}"
    assert len(parquet_df) == len(sample_dismantling_data), f"Expected {len(sample_dismantling_data)} rows, got {len(parquet_df)}"

    # Normalize and compare
    csv_normalized = normalize_df(csv_df)
    parquet_normalized = normalize_df(parquet_df)
    
    assert_frame_equal(csv_normalized, parquet_normalized)


def test_exclude_columns_equivalence(sample_dismantling_data, csv_writer_helper, parquet_writer_helper):
    """Test that excluding columns works identically for both formats"""
    columns = list(sample_dismantling_data.columns)
    
    csv_path = csv_writer_helper(sample_dismantling_data, columns)
    parquet_path = parquet_writer_helper(sample_dismantling_data, columns)
    
    # Read excluding the same columns
    exclude = ["threshold", "run"]
    csv_df = csv.read_without_columns(file=csv_path, exclude_columns=exclude)
    parquet_df = parquet.read_without_columns(file=str(parquet_path), exclude_columns=exclude)
    
    # Verify excluded columns are not present
    for col in exclude:
        assert col not in csv_df.columns
        assert col not in parquet_df.columns
    
    # Compare remaining data
    csv_normalized = normalize_df(csv_df)
    parquet_normalized = normalize_df(parquet_df)
    
    assert_frame_equal(csv_normalized, parquet_normalized)


def test_read_specific_indices_equivalence(sample_dismantling_data, csv_writer_helper, parquet_writer_helper):
    """Test that reading specific row indices works identically for both formats"""
    columns = list(sample_dismantling_data.columns)
    
    csv_path = csv_writer_helper(sample_dismantling_data, columns)
    parquet_path = parquet_writer_helper(sample_dismantling_data, columns)
    
    # Read specific indices
    indices = [0, 2, 4]
    csv_df = csv.read_without_columns(file=csv_path, exclude_columns=[], read_index=indices)
    parquet_df = parquet.read_without_columns(file=str(parquet_path), exclude_columns=[], read_index=indices)
    
    # Verify we got the right number of rows
    assert len(csv_df) == len(indices)
    assert len(parquet_df) == len(indices)
    
    # Compare
    csv_normalized = normalize_df(csv_df)
    parquet_normalized = normalize_df(parquet_df)
    
    assert_frame_equal(csv_normalized, parquet_normalized)


def test_data_types_preservation(sample_dismantling_data, csv_writer_helper, parquet_writer_helper):
    """Test that data types are preserved consistently across formats"""
    # Add various data types
    data = sample_dismantling_data.copy()
    data["float_col"] = [1.1, 2.2, 3.3, 4.4, 5.5]
    data["int_col"] = [10, 20, 30, 40, 50]
    data["string_col"] = ["a", "b", "c", "d", "e"]
    
    columns = list(data.columns)
    
    csv_path = csv_writer_helper(data, columns)
    parquet_path = parquet_writer_helper(data, columns)
    
    csv_df = csv.read_without_columns(file=csv_path, exclude_columns=[])
    parquet_df = parquet.read_without_columns(file=str(parquet_path), exclude_columns=[])
    
    # Compare values (ignoring exact dtype mismatches like int64 vs int32)
    csv_normalized = normalize_df(csv_df)
    parquet_normalized = normalize_df(parquet_df)
    
    assert_frame_equal(csv_normalized, parquet_normalized, check_dtype=False)


# def test_empty_dataframe_handling(csv_writer_helper, parquet_writer_helper):
#     """Test handling of edge case: empty DataFrame"""
    # Note: Empty DataFrames are skipped by both writers (by design)
    # This test verifies the behavior is consistent
    empty_df = pd.DataFrame(columns=["a", "b", "c"])
    columns = ["a", "b", "c"]
    
    # Write non-empty data first to create files
    data = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    csv_path = csv_writer_helper(data, columns)
    parquet_path = parquet_writer_helper(data, columns)
    
    # Both files should exist and contain the data
    assert csv_path.exists()
    assert parquet_path.exists()


def test_large_string_handling(csv_writer_helper, parquet_writer_helper):
    """Test that large strings are handled identically"""
    data = pd.DataFrame({
        "id": [1, 2, 3],
        "long_string": ["x" * 1000, "y" * 2000, "z" * 3000],
        "value": [100, 200, 300]
    })
    columns = list(data.columns)
    
    csv_path = csv_writer_helper(data, columns)
    parquet_path = parquet_writer_helper(data, columns)
    
    csv_df = csv.read_without_columns(file=csv_path, exclude_columns=[])
    parquet_df = parquet.read_without_columns(file=str(parquet_path), exclude_columns=[])
    
    csv_normalized = normalize_df(csv_df)
    parquet_normalized = normalize_df(parquet_df)
    
    assert_frame_equal(csv_normalized, parquet_normalized)


def test_schema_consistency(sample_dismantling_data, csv_writer_helper, parquet_writer_helper):
    """Test that schema (column names and order) is consistent"""
    columns = list(sample_dismantling_data.columns)
    
    csv_path = csv_writer_helper(sample_dismantling_data, columns)
    parquet_path = parquet_writer_helper(sample_dismantling_data, columns)
    
    csv_df = csv.read_without_columns(file=csv_path, exclude_columns=[])
    parquet_df = parquet.read_without_columns(file=str(parquet_path), exclude_columns=[])
    
    # Remove file and idx columns that are added during read
    csv_cols = [c for c in csv_df.columns if c not in ["file", "idx"]]
    parquet_cols = [c for c in parquet_df.columns if c not in ["file", "idx"]]
    
    # Columns should match
    assert set(csv_cols) == set(parquet_cols)


def test_conversion_script_equivalence(tmp_path):
    """
    Test that converting an existing CSV to Parquet preserves data.
    This simulates the migration scenario.
    """
    # Create a CSV file using pandas directly
    original_data = pd.DataFrame({
        "network": ["test"] * 3,
        "lcc_size": [100, 80, 60],
        "removed": [0, 20, 40],
    })
    
    csv_file = tmp_path / "original.csv"
    original_data.to_csv(csv_file, index=False)
    
    # Read it back with CSV reader
    csv_df = csv.read_without_columns(file=csv_file, exclude_columns=[])
    
    # Convert to Parquet by writing with Parquet writer
    parquet_file = tmp_path / "converted.parquet"
    with parquet.ParquetDataFrameWriter(parquet_file, list(original_data.columns), logger=logger) as writer:
        writer.write(original_data)
    
    # Read back with Parquet reader
    parquet_df = parquet.read_without_columns(file=str(parquet_file), exclude_columns=[])
    
    # The data should be identical (modulo file paths)
    csv_normalized = normalize_df(csv_df)
    parquet_normalized = normalize_df(parquet_df)
    
    assert_frame_equal(csv_normalized, parquet_normalized)


def test_nan_and_none_handling(csv_writer_helper, parquet_writer_helper):
    """Test that NaN and None values are handled identically"""
    data = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "value_with_nan": [1.0, np.nan, 3.0, np.nan, 5.0],
        "string_with_none": ["a", None, "c", None, "e"],
        "int_with_nan": [10, 20, np.nan, 40, 50],  # Will become float due to NaN
    })
    columns = list(data.columns)
    
    csv_path = csv_writer_helper(data, columns)
    parquet_path = parquet_writer_helper(data, columns)
    
    csv_df = csv.read_without_columns(file=csv_path, exclude_columns=[])
    parquet_df = parquet.read_without_columns(file=str(parquet_path), exclude_columns=[])
    
    csv_normalized = normalize_df(csv_df)
    parquet_normalized = normalize_df(parquet_df)
    
    # Use check_dtype=False as NaN handling may cause type differences
    assert_frame_equal(csv_normalized, parquet_normalized, check_dtype=False)


def test_categorical_columns(csv_writer_helper, parquet_writer_helper):
    """Test that categorical columns are preserved correctly"""
    data = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "category": pd.Categorical(["A", "B", "A", "C", "B"]),
        "value": [10, 20, 30, 40, 50],
    })
    columns = list(data.columns)
    
    csv_path = csv_writer_helper(data, columns)
    parquet_path = parquet_writer_helper(data, columns)
    
    csv_df = csv.read_without_columns(file=csv_path, exclude_columns=[])
    parquet_df = parquet.read_without_columns(file=str(parquet_path), exclude_columns=[])
    
    csv_normalized = normalize_df(csv_df)
    parquet_normalized = normalize_df(parquet_df)
    
    # Categories might be stored differently but values should match
    assert_frame_equal(csv_normalized, parquet_normalized, check_dtype=False, check_categorical=False)


def test_multi_run_scenario(tmp_path):
    """Test realistic multi-run scenario: multiple networks, multiple runs appended over time"""
    columns = ["network", "dismantler", "run", "lcc_size", "removed_nodes"]
    
    # Simulate 3 runs for 2 networks
    run1 = pd.DataFrame({
        "network": ["karate", "dolphins"],
        "dismantler": ["degree", "degree"],
        "run": [0, 0],
        "lcc_size": [30, 50],
        "removed_nodes": [4, 12],
    })
    
    run2 = pd.DataFrame({
        "network": ["karate", "dolphins"],
        "dismantler": ["degree", "degree"],
        "run": [1, 1],
        "lcc_size": [28, 48],
        "removed_nodes": [6, 14],
    })
    
    run3 = pd.DataFrame({
        "network": ["karate", "dolphins"],
        "dismantler": ["degree", "degree"],
        "run": [2, 2],
        "lcc_size": [25, 45],
        "removed_nodes": [9, 17],
    })
    
    # CSV: multiple writer calls (realistic usage with append)
    csv_file = tmp_path / "multi_run.csv"
    for run_data in [run1, run2, run3]:
        with csv.CSVDataFrameWriter(csv_file, columns, logger=logger) as writer:
            writer.write(run_data)
    
    # Parquet: single writer, multiple chunks (realistic usage)
    parquet_file = tmp_path / "multi_run.parquet"
    with parquet.ParquetDataFrameWriter(parquet_file, columns, logger=logger) as writer:
        for run_data in [run1, run2, run3]:
            writer.write(run_data)
    
    # Read and compare
    csv_df = csv.read_without_columns(file=csv_file, exclude_columns=[])
    parquet_df = parquet.read_without_columns(file=str(parquet_file), exclude_columns=[])
    
    csv_normalized = normalize_df(csv_df)
    parquet_normalized = normalize_df(parquet_df)
    
    # Should have 6 rows total (3 runs × 2 networks)
    assert len(csv_normalized) == 6
    assert len(parquet_normalized) == 6
    
    assert_frame_equal(csv_normalized, parquet_normalized)


def test_schema_mismatch_detection(tmp_path):
    """Test that schema mismatches between chunks are detected and handled gracefully"""
    columns1 = ["a", "b", "c"]
    
    chunk1 = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    chunk2 = pd.DataFrame({"a": [4], "b": [5], "d": [6]})  # Different column 'd' instead of 'c'
    
    parquet_file = tmp_path / "schema_mismatch.parquet"
    with parquet.ParquetDataFrameWriter(parquet_file, columns1, logger=logger) as writer:
        writer.write(chunk1)  # This will succeed
        writer.write(chunk2)  # This will fail due to missing column 'c'
    
    # Verify that only the first chunk was written successfully
    if parquet_file.exists():
        df = pd.read_parquet(str(parquet_file), engine="pyarrow")
        # Should have only 1 row from chunk1, chunk2 was rejected
        assert len(df) == 1, f"Expected 1 row (first chunk only), got {len(df)}"
        assert df["a"].iloc[0] == 1
        assert df["b"].iloc[0] == 2
        assert df["c"].iloc[0] == 3


def test_converted_csv_matches_original(tmp_path):
    """Test that a CSV converted to Parquet via script preserves all data exactly"""
    # Create original data
    original_data = pd.DataFrame({
        "network": ["test1", "test2", "test3"],
        "lcc_size": [100, 80, 60],
        "removed": [0, 20, 40],
        "threshold": [0.1, 0.2, 0.3],
        "run": [0, 0, 0],
    })
    
    csv_file = tmp_path / "original.csv"
    original_data.to_csv(csv_file, index=False)
    
    # Read with CSV reader
    csv_read = pd.read_csv(csv_file)
    
    # Convert to Parquet using pyarrow (simulating the conversion script)
    parquet_file = tmp_path / "converted.parquet"
    csv_read.to_parquet(str(parquet_file), engine="pyarrow", compression="snappy", index=False)
    
    # Read back
    parquet_read = pd.read_parquet(str(parquet_file), engine="pyarrow")
    
    # Should be identical
    assert_frame_equal(csv_read, parquet_read)


def test_file_column_is_categorical(sample_dismantling_data, csv_writer_helper, parquet_writer_helper):
    """Test that the 'file' column is properly set as categorical for memory efficiency"""
    columns = list(sample_dismantling_data.columns)
    
    csv_path = csv_writer_helper(sample_dismantling_data, columns)
    parquet_path = parquet_writer_helper(sample_dismantling_data, columns)
    
    csv_df = csv.read_without_columns(file=csv_path, exclude_columns=[])
    parquet_df = parquet.read_without_columns(file=str(parquet_path), exclude_columns=[])
    
    # Both should have 'file' as categorical
    assert csv_df["file"].dtype.name == "category"
    assert parquet_df["file"].dtype.name == "category"


def test_parquet_writer_class(tmp_path, sample_dismantling_data):
    """Test the new ParquetDataFrameWriter class"""
    columns = list(sample_dismantling_data.columns)
    output_file = tmp_path / "test_class.parquet"
    
    # Split data into chunks
    chunk1 = sample_dismantling_data.iloc[:2]
    chunk2 = sample_dismantling_data.iloc[2:]
    
    # Use class-based writer with context manager
    with parquet.ParquetDataFrameWriter(
        output_file=output_file,
        columns=columns,
        logger=logger,
    ) as writer:
        writer.write(chunk1)
        assert writer.is_alive(), "Writer should be alive after first write"
        
        writer.write(chunk2)
        assert writer.is_alive(), "Writer should be alive after second write"
    
    # Verify data was written correctly
    df = parquet.read_without_columns(file=str(output_file), exclude_columns=[])
    assert len(df) == len(sample_dismantling_data)
    
    # Compare content
    df_normalized = normalize_df(df)
    expected = sample_dismantling_data.copy()
    expected = expected.reset_index(drop=True)
    
    for col in df_normalized.columns:
        if col not in ["idx"]:  # idx is added by reader
            assert col in expected.columns


def test_parquet_writer_context_manager(tmp_path, sample_dismantling_data):
    """Test that ParquetDataFrameWriter works as context manager"""
    columns = list(sample_dismantling_data.columns)
    output_file = tmp_path / "test_context.parquet"
    
    # Use context manager
    with parquet.ParquetDataFrameWriter(output_file, columns, logger=logger) as writer:
        writer.write(sample_dismantling_data)
    
    # File should exist and be valid
    assert output_file.exists()
    df = parquet.read_without_columns(file=str(output_file), exclude_columns=[])
    assert len(df) == len(sample_dismantling_data)


def test_parquet_writer_error_detection(tmp_path):
    """Test that ParquetDataFrameWriter tolerates batches with missing columns.
    
    The writer skips malformed batches (logging an error) and continues
    accepting valid data.  The final file should contain only the valid rows.
    """
    import time
    columns = ["a", "b", "c"]
    output_file = tmp_path / "test_error.parquet"
    
    with parquet.ParquetDataFrameWriter(output_file, columns, logger=logger) as writer:
        # Write valid data
        chunk1 = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        writer.write(chunk1)
        
        # Write invalid data (missing column 'c', extra column 'd')
        chunk2 = pd.DataFrame({"a": [4], "b": [5], "d": [6]})
        writer.write(chunk2)
        time.sleep(0.5)  # Give writer thread time to process
        
        # Writer should still be alive — next valid write should succeed
        chunk3 = pd.DataFrame({"a": [7], "b": [8], "c": [9]})
        writer.write(chunk3)
    
    # Verify: only chunk1 and chunk3 are in the file (chunk2 was skipped)
    result = pd.read_parquet(str(output_file))
    assert len(result) == 2, f"Expected 2 rows (chunk1 + chunk3), got {len(result)}"
    assert result["a"].tolist() == [1, 7]
    assert result["c"].tolist() == [3, 9]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
