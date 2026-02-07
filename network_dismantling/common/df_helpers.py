import logging
from pathlib import Path
from typing import Callable, List, Union, Dict, Optional

import numpy as np
import pandas as pd

# Import direttamente le funzioni parquet - niente wrapper
from network_dismantling.common.storage.pandas.parquet import (
    df_reader,
    read_without_columns,
    read_without_removals,
    get_df_columns,
)



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

# class RemovalsColumns:
#     REMOVAL_NUM = 0
#     ID = 1
#     PREDICTION = 2
#     LCC_SIZE = 3
#     SLCC_SIZE = 4

class RemovalsColumns:
    REMOVAL_NUM = "removal_num"
    ID = "id"
    PREDICTION = "prediction"
    LCC_SIZE = "lcc_size"
    SLCC_SIZE = "slcc_size"
