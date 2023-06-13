from pathlib import Path
from typing import Callable

import pandas as pd


def get_df_columns(file: Path):
    # Read column names from file
    cols = list(pd.read_csv(str(file), nrows=1))

    return cols


def read_without_removals(file):
    # Read column names from file
    cols = get_df_columns(file)

    # Use list comprehension to remove the unwanted column in **usecol**
    df = pd.read_csv(str(file),
                     usecols=[i for i in cols if i != 'removals'],
                     )

    return df


def df_reader(files, include_removals=False, file_callback: Callable = None):
    from pathlib import Path

    if not isinstance(files, list):
        files = [files]

    df_buffer = []
    for file in files:
        if not isinstance(file, Path):
            file = Path(file)

        file = file.resolve()

        if not file.exists():
            raise FileNotFoundError(f"Input file {file} does not exist.")
        elif not file.is_file():
            raise FileNotFoundError(f"Input file {file} is not a file.")

        if include_removals is False:
            df = read_without_removals(file)
        else:
            df = pd.read_csv(str(file))

        if file_callback is not None:
            df = file_callback(file=file,
                               df=df,
                               )

        df_buffer.append(df)

    df = pd.concat(df_buffer, ignore_index=True)

    return df
