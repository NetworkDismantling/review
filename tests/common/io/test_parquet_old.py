import multiprocessing
import pandas as pd
from pandas.testing import assert_frame_equal
from network_dismantling.common.data_structures import dotdict
from network_dismantling.common.storage.pandas.parquet import (
    read_without_columns,
    start_df_writer,
)
import logging

logger = logging.getLogger()


def test_full_write_and_read(tmp_path):
    output_file = tmp_path / "test.parquet"
    queue = multiprocessing.Queue()
    writer = start_df_writer(
        args=dotdict(output_file=str(output_file), output_df_columns=["a", "b"], logger=logger),
        df_queue=queue,
    )

    original = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    queue.put(original)
    queue.put(None)
    writer.join()

    result = read_without_columns(file=str(output_file), exclude_columns=[]).sort_values("idx").reset_index(drop=True)

    expected = original.copy()
    expected["idx"] = [0, 1, 2]
    expected["file"] = str(output_file)
    expected["file"] = expected["file"].astype("category")

    assert_frame_equal(result, expected)


def test_append_mode(tmp_path):
    output_file = tmp_path / "test.parquet"

    # first write
    queue = multiprocessing.Queue()
    writer = start_df_writer(
        args=dotdict(output_file=str(output_file), output_df_columns=["a", "b"], logger=logger),
        df_queue=queue,
    )
    first = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    queue.put(first)
    queue.put(None)
    writer.join()

    # append second block
    queue = multiprocessing.Queue()
    writer = start_df_writer(
        args=dotdict(output_file=str(output_file), output_df_columns=["a", "b"], logger=logger),
        df_queue=queue,
    )
    second = pd.DataFrame({"a": [3, 4], "b": ["u", "v"]})
    queue.put(second)
    queue.put(None)
    writer.join()

    result = read_without_columns(file=str(output_file), exclude_columns=[]).sort_values("idx").reset_index(drop=True)

    expected = pd.concat([first, second]).reset_index(drop=True)
    expected["idx"] = list(range(len(expected)))
    expected["file"] = str(output_file)
    expected["file"] = expected["file"].astype("category")

    assert_frame_equal(result, expected)


def test_read_specific_indices(tmp_path):
    output_file = tmp_path / "test.parquet"
    queue = multiprocessing.Queue()
    writer = start_df_writer(
        args=dotdict(output_file=str(output_file), output_df_columns=["a", "b"], logger=logger),
        df_queue=queue,
    )

    data = pd.DataFrame({
        "a": list(range(6)),
        "b": list("abcdef"),
    })
    queue.put(data)
    queue.put(None)
    writer.join()

    result = read_without_columns(file=str(output_file), exclude_columns=[], idxs=[1, 5]).sort_values("idx").reset_index(drop=True)

    sub_expected = data.loc[[1, 5]].reset_index(drop=True)
    sub_expected["idx"] = [1, 5]
    sub_expected["file"] = str(output_file)
    sub_expected["file"] = sub_expected["file"].astype("category")

    assert_frame_equal(result, sub_expected)


def test_exclude_columns(tmp_path):
    output_file = tmp_path / "test.parquet"
    queue = multiprocessing.Queue()
    writer = start_df_writer(
        args=dotdict(output_file=str(output_file), output_df_columns=["a", "b"], logger=logger),
        df_queue=queue,
    )

    original = pd.DataFrame({"a": [7, 8], "b": ["m", "n"], "c": [True, False]})
    queue.put(original)
    queue.put(None)
    writer.join()

    result = read_without_columns(file=str(output_file), exclude_columns="b") \
        .sort_values("idx") \
        .reset_index(drop=True)

    expected = original.drop(columns=["b"]).copy()
    expected["idx"] = [0, 1]
    expected["file"] = str(output_file)
    expected["file"] = expected["file"].astype("category")

    assert_frame_equal(result, expected)