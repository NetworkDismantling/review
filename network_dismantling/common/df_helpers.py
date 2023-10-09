from pathlib import Path
from typing import Callable, List, Union, Dict

import pandas as pd


def get_df_columns(file: Path):
    # Read column names from file
    cols = list(pd.read_csv(str(file), nrows=1))

    return cols


def read_without_removals(file, exclude_columns: Union[str, List[str]] = None):
    if exclude_columns is None:
        exclude_columns = ["removals"]
    elif isinstance(exclude_columns, str):
        exclude_columns = [exclude_columns]
    elif "removals" not in exclude_columns:
        exclude_columns = exclude_columns + ["removals"]

    return read_without_columns(
        file=file,
        exclude_columns=exclude_columns,
    )


def read_without_columns(
    file,
    exclude_columns: Union[str, List[str]],
    dtype_dict=None,
):
    if exclude_columns is None:
        exclude_columns = []
        
    if isinstance(exclude_columns, str):
        exclude_columns = [exclude_columns]

    # Read column names from file
    cols = get_df_columns(file)

    # Use list comprehension to remove the unwanted column in **usecol**
    df = pd.read_csv(
        str(file),
        usecols=[i for i in cols if i not in exclude_columns],
        dtype=dtype_dict,
    )

    return df


def df_reader(
    files,
    include_removals: bool = False,
    file_callbacks: Union[Callable, List[Callable]] = None,
    raise_on_missing_file: bool = True,
    expected_columns: Union[str, List[str]] = None,
    exclude_columns: Union[str, List[str]] = None,
    at_least_one_file: bool = False,
    dtype_dict: Dict = None,
):
    from pathlib import Path

    if not isinstance(files, list):
        files = [files]

    if expected_columns is not None:
        if isinstance(expected_columns, str):
            expected_columns = [expected_columns]

    if dtype_dict is None:
        dtype_dict = {}
    dtype_dict.setdefault("network", "category")

    df_buffer = []
    for file in files:
        if not isinstance(file, Path):
            file = Path(file)

        file = file.resolve()

        if (not file.exists()) or (not file.is_file()):
            if raise_on_missing_file:
                raise FileNotFoundError(f"Input file {file} does not exist.")
            else:
                continue

        if include_removals is False:
            df = read_without_removals(
                file,
                exclude_columns=exclude_columns,
            )
        else:
            df = read_without_columns(
                file,
                exclude_columns=exclude_columns,
            )

        # for expected_column in expected_columns:
        #     if expected_column not in df:
        #         raise ValueError(
        #             f"Expected column {expected_column} not found in {file}."
        #         )
        if expected_columns is not None:
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
