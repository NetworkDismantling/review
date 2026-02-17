import multiprocessing
import pandas as pd
from pandas.testing import assert_frame_equal
from network_dismantling.common.data_structures import dotdict
from network_dismantling.common.storage.pandas.parquet import (
    read_without_columns,
    ParquetDataFrameWriter,
)
import logging

logger = logging.getLogger()


def _write_parquet(output_file, columns, df, mode='overwrite'):
    """Helper: write a DataFrame using ParquetDataFrameWriter."""
    with ParquetDataFrameWriter(
        output_file=output_file,
        columns=columns,
        logger=logger,
        mode=mode,
    ) as writer:
        if df is not None:
            writer.write(df)


def test_full_write_and_read(tmp_path):
    output_file = tmp_path / "test.parquet"

    original = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    _write_parquet(output_file, ["a", "b"], original)

    result = read_without_columns(file=str(output_file), exclude_columns=[]).sort_values("idx").reset_index(drop=True)

    expected = original.copy()
    expected["idx"] = [0, 1, 2]
    expected["file"] = str(output_file)
    expected["file"] = expected["file"].astype("category")

    assert_frame_equal(result, expected)


def test_append_mode(tmp_path):
    output_file = tmp_path / "test.parquet"

    # first write
    first = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    _write_parquet(output_file, ["a", "b"], first, mode='overwrite')

    # append second block
    second = pd.DataFrame({"a": [3, 4], "b": ["u", "v"]})
    _write_parquet(output_file, ["a", "b"], second, mode='append')

    result = read_without_columns(file=str(output_file), exclude_columns=[]).sort_values("idx").reset_index(drop=True)

    expected = pd.concat([first, second]).reset_index(drop=True)
    expected["idx"] = list(range(len(expected)))
    expected["file"] = str(output_file)
    expected["file"] = expected["file"].astype("category")

    assert_frame_equal(result, expected)


def test_read_specific_indices(tmp_path):
    output_file = tmp_path / "test.parquet"

    data = pd.DataFrame({
        "a": list(range(6)),
        "b": list("abcdef"),
    })
    _write_parquet(output_file, ["a", "b"], data)

    result = read_without_columns(file=str(output_file), exclude_columns=[], read_index=[1, 5]).sort_values("idx").reset_index(drop=True)

    sub_expected = data.loc[[1, 5]].reset_index(drop=True)
    sub_expected["idx"] = [1, 5]
    sub_expected["file"] = str(output_file)
    sub_expected["file"] = sub_expected["file"].astype("category")

    assert_frame_equal(result, sub_expected)


def test_exclude_columns(tmp_path):
    output_file = tmp_path / "test.parquet"

    original = pd.DataFrame({"a": [7, 8], "b": ["m", "n"], "c": [True, False]})
    _write_parquet(output_file, ["a", "b", "c"], original)

    result = read_without_columns(file=str(output_file), exclude_columns="b") \
        .sort_values("idx") \
        .reset_index(drop=True)

    expected = original.drop(columns=["b"]).copy()
    expected["idx"] = [0, 1]
    expected["file"] = str(output_file)
    expected["file"] = expected["file"].astype("category")

    assert_frame_equal(result, expected)