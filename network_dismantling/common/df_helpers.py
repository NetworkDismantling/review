import logging
from pathlib import Path
from typing import Callable, List, Union, Dict

import numpy as np
import pandas as pd


def get_df_columns(file: Path):
    # Read column names from file
    cols = list(pd.read_csv(str(file), nrows=1))

    return cols


def read_without_removals(file,
                          exclude_columns: Union[str, List[str]] = None,
                          **kwargs,
                          ):
    if exclude_columns is None:
        exclude_columns = ["removals"]
    elif isinstance(exclude_columns, str):
        exclude_columns = [exclude_columns]
    elif "removals" not in exclude_columns:
        exclude_columns = exclude_columns + ["removals"]

    return read_without_columns(
        file=file,
        exclude_columns=exclude_columns,
        **kwargs,
    )


def read_without_columns(
        file,
        exclude_columns: Union[str, List[str]],
        read_index: Union[None, int, List[int]] = None,
        dtype_dict=None,
):
    if exclude_columns is None:
        exclude_columns = []

    if isinstance(exclude_columns, str):
        exclude_columns = [exclude_columns]

    # Read column names from file
    cols = get_df_columns(file)
    usecols = [i for i in cols if i not in exclude_columns]

    indices_to_read = None
    if read_index is not None:
        if isinstance(read_index, int):
            indices_to_read = [read_index]

        buffer = []
        for index_to_read in indices_to_read:
            read_df = pd.read_csv(
                str(file),
                skiprows=index_to_read + 1,
                nrows=1,
                # usecols=usecols,
                names=usecols,
                dtype=dtype_dict,
            )
            read_df["idx"] = index_to_read

            buffer.append(read_df)

        df = pd.concat(buffer,
                       ignore_index=True,
                       )
    else:
        # Use list comprehension to remove the unwanted column in **usecol**
        df = pd.read_csv(
            str(file),
            usecols=usecols,
            dtype=dtype_dict,
        )
        df["idx"] = df.index

    df["file"] = f"{file}"
    df["file"] = df["file"].astype("category")

    return df


def df_reader(
        files: Union[Union[Path, str], List[Union[Path, str]]],
        include_removals: bool = False,
        file_callbacks: Union[Callable, List[Callable]] = None,
        raise_on_missing_file: bool = True,
        expected_columns: Union[str, List[str]] = None,
        exclude_columns: Union[str, List[str]] = None,
        at_least_one_file: bool = False,
        dtype_dict: Dict = None,
        read_index: Union[None, int, List[int], Dict[Union[str, Path], List[int]]] = None,
        logger: logging.Logger = logging.getLogger("dummy"),
):
    from pathlib import Path

    if not isinstance(files, list):
        files = [files]

    for i, file in enumerate(files):
        if not isinstance(file, Path):
            file = Path(file)

        file = file.resolve()

        files[i] = file

    if expected_columns is not None:
        if isinstance(expected_columns, str):
            expected_columns = [expected_columns]

    if dtype_dict is None:
        dtype_dict = {}
    dtype_dict.setdefault("network", "category")

    if read_index is not None:
        if isinstance(read_index, list):
            if len(read_index) != len(files):
                raise ValueError(
                    f"read_index must have the same length as files. Found {len(read_index)} read_index values and {len(files)} files."
                )

            read_index = {file: [index] if isinstance(index, int) else index
                          for file, index in
                          zip(files, read_index)
                          }

        elif isinstance(read_index, dict):
            for file in files:
                if file not in read_index:
                    raise ValueError(
                        f"read_index must have a value for each file. Missing value for {file}."
                    )
        elif (isinstance(read_index, int) or
              np.issubdtype(read_index, np.integer)):
            read_index = {file: int(read_index) for file in files}
        else:
            raise ValueError(f"Invalid read_index {read_index} (type {type(read_index)}.")

    df_buffer = []
    for file in files:
        if (not file.exists()) or (not file.is_file()):
            if raise_on_missing_file:
                raise FileNotFoundError(f"Input file {file} does not exist.")
            else:
                continue

        if include_removals is False:
            read_function = read_without_removals

        else:
            read_function = read_without_columns

        df = read_function(
            file,
            exclude_columns=exclude_columns,
            read_index=read_index[file] if read_index is not None else None,
        )

        if expected_columns is not None:
            for column in ["idx", "file"]:
                if column not in expected_columns:
                    expected_columns += [column]

            if (len(df.columns) != len(expected_columns)) or (df.columns != expected_columns).all():
                raise ValueError(
                    f"Input file columns {df.columns} do not match the expected columns {expected_columns}."
                )

        if file_callbacks is not None:
            if not isinstance(file_callbacks, List):
                file_callbacks = [file_callbacks]

            for file_callback in file_callbacks:
                if not isinstance(file_callback, Callable):
                    raise ValueError(
                        f"file_callbacks must be a list of callables. Found {type(file_callback)}."
                    )

                df = file_callback(
                    file=file,
                    df=df,
                )

        # df["idx"] = df.index
        # df["file"] = f"{file}"

        df_buffer.append(df)

    if len(df_buffer) == 0:
        if at_least_one_file:
            raise FileNotFoundError(f"No input files found.")
        else:
            df = pd.DataFrame(
                columns=expected_columns,
            )

            # TODO coherence with dtype_dict
    else:

        df = pd.concat(
            df_buffer,
            ignore_index=True,
        )

        df.drop_duplicates(inplace=True)

    # if "network" in df and df["network"].dtype != str:
    #     df["network"] = df["network"].astype(str)

    return df


# def read_index(file,
#                index_col="idx",
#                ):
#     # Read column names from file
#     cols = get_df_columns(file)
#
#     # Use list comprehension to remove the unwanted column in **usecol**
#     df = pd.read_csv(
#         str(file),
#         usecols=[i for i in cols if i not in exclude_columns],
#         dtype=dtype_dict,
#     )
#
#     return pd.read_csv(
#         str(file),
#         index_col=index_col,
#     )
class RemovalsColumns:
    REMOVAL_NUM = 0
    ID = 1
    PREDICTION = 2
    LCC_SIZE = 3
    SLCC_SIZE = 4
