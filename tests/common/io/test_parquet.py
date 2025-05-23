# tests/common/io/test_parquet.py
import os
import multiprocessing

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from network_dismantling.common.data_structures import dotdict
from network_dismantling.common.storage.pandas.parquet import (
    read_without_columns,
    start_df_writer,
)

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@pytest.fixture
def write_parquet(tmp_path):
    """
    Helper to write a DataFrame (or None) to parquet using df_writer.
    Returns the pathlib.Path to the output file.
    """
    def _write(df, columns):
        output_file = tmp_path / "test.parquet"
        queue = multiprocessing.Queue()
        writer = start_df_writer(
            args=dotdict(output_file=str(output_file),
                         output_df_columns=columns,
                         logger=logger),
            df_queue=queue,
        )
        # push data block or just end
        if df is not None:
            queue.put(df)
        queue.put(None)
        writer.join()
        return output_file
    return _write


def test_full_write_and_read(write_parquet):
    original = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    path = write_parquet(original, ["a", "b"])

    df = read_without_columns(file=str(path), exclude_columns=[])
    df = df.sort_values("idx").reset_index(drop=True)

    expected = original.copy()
    expected["idx"] = [0, 1, 2]
    expected["file"] = str(path)
    expected["file"] = expected["file"].astype("category")

    assert_frame_equal(df, expected)


def test_append_mode(write_parquet):
    first = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    write_parquet(first, ["a", "b"])

    second = pd.DataFrame({"a": [3, 4], "b": ["u", "v"]})
    write_parquet(second, ["a", "b"])

    path = (write_parquet.__closure__[0].cell_contents)  # recover tmp_path/test.parquet
    file = path / "test.parquet"
    df = read_without_columns(file=str(file), exclude_columns=[])
    df = df.sort_values("idx").reset_index(drop=True)

    expected = pd.concat([first, second], ignore_index=True)
    expected["idx"] = list(range(len(expected)))
    expected["file"] = str(file)
    expected["file"] = expected["file"].astype("category")

    logger.info(f"Expected:\n{expected}")
    logger.info(f"Result:\n{df}")
    print(f"Expected:\n{expected}\n{expected.dtypes}\nfile unique:{df['file'].unique()}\n")
    print(f"Result:\n{df}\n{df.dtypes}\nfile unique:{df['file'].unique()}\n")

    # Check if the DataFrame is sorted by 'idx'
    # assert df["idx"].is_monotonic_increasing

    assert_frame_equal(df, expected)


def test_read_specific_indices(write_parquet):
    data = pd.DataFrame({"a": list(range(6)), "b": list("abcdef")})
    path = write_parquet(data, ["a", "b"])

    df = read_without_columns(file=str(path), exclude_columns=[], read_index=[1, 5])
    df = df.sort_values("idx").reset_index(drop=True)

    expected = data.loc[[1, 5]].reset_index(drop=True)
    expected["idx"] = [1, 5]
    expected["file"] = str(path)
    expected["file"] = expected["file"].astype("category")

    assert_frame_equal(df, expected)


def test_exclude_columns(write_parquet):
    original = pd.DataFrame({"a": [7, 8], "b": ["m", "n"], "c": [True, False]})
    path = write_parquet(original, ["a", "b", "c"])

    df = read_without_columns(file=str(path), exclude_columns="b")
    df = df.sort_values("idx").reset_index(drop=True)

    expected = original.drop(columns=["b"]).copy()
    expected["idx"] = [0, 1]
    expected["file"] = str(path)
    expected["file"] = expected["file"].astype("category")

    assert_frame_equal(df, expected)


def test_empty_write(write_parquet):
    # Only sentinel -> no rows
    path = write_parquet(None, ["a", "b"])
    df = read_without_columns(file=str(path), exclude_columns=[])
    assert df.empty
    # columns 'a', 'b', 'idx', 'file' should exist
    assert list(df.columns) == ["a", "b", "idx", "file"]


def test_dtype_casting(tmp_path):
    # write with pandas directly
    path = tmp_path / "dtype.parquet"
    df_orig = pd.DataFrame({"x": [1, 2, 3]})
    df_orig.to_parquet(str(path), engine="pyarrow")

    # read and cast 'x' to float
    df = read_without_columns(file=str(path),
                              exclude_columns=[],
                              dtype_dict={"x": "float64"}
                              )

    assert df["x"].dtype == "float64"


def test_invalid_index_raises(write_parquet):
    data = pd.DataFrame({"a": [0], "b": ["z"]})
    path = write_parquet(data, ["a", "b"])
    with pytest.raises(ValueError):
        read_without_columns(file=str(path),
                             exclude_columns=[],
                             read_index=[5])  # out of range


def test_missing_file(tmp_path):
    missing = tmp_path / "nope.parquet"
    with pytest.raises(FileNotFoundError):
        read_without_columns(file=str(missing), exclude_columns=[])