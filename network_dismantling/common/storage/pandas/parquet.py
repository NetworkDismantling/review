import logging
import multiprocessing
import threading
from bisect import bisect_right
from collections import defaultdict
from itertools import accumulate
from pathlib import Path
from typing import Callable, List, Union, Dict, Optional, Tuple, Literal

import numpy as np
import pandas as pd
import fastparquet as fp
import pyarrow as pa
import pyarrow.parquet as pq
# Use PyArrow for reading (better performance)
from pyarrow.parquet import ParquetFile  

from network_dismantling.common.storage.pandas.base import BaseDataFrameWriter
from network_dismantling.common.removal import RemovalsList, Removal


# append to parquet
# https://stackoverflow.com/questions/47191675/pandas-write-dataframe-to-parquet-format-with-append
#   Append could be inefficient if you write too many small row groups.
#   Typically recommended size of a row group is closer to 100,000 or 1,000,000 rows.
#   This has a few benefits over very small row groups. Compression will work better,
#   since compression operates within a row group only.
#   There will also be less overhead spent on storing statistics, since each row group
#   stores its own statistics.

# Best practices for Parquet append:
# - Use fastparquet for writing (native append support)
# - Use pyarrow for reading (better performance for complex queries)
# - Row groups should be 100,000 to 1,000,000 rows for optimal compression
# - Snappy compression offers best balance between speed and compression ratio
# - Files are compatible between fastparquet and pyarrow (both follow Apache Parquet standard)


def prepare_dataframe_for_fastparquet(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare DataFrame for fastparquet writing by converting problematic types.
    
    Fastparquet doesn't handle PyArrow-backed dtypes well. This function converts:
    - PyArrow string types -> object dtype
    - StringDtype -> object dtype  
    - Category dtypes with PyArrow storage -> object dtype
    
    Args:
        df: Input DataFrame (will be modified in place)
        
    Returns:
        The same DataFrame with converted types
    """
    for col in df.columns:
        dtype = df[col].dtype
        
        # Handle pandas StringDtype (PyArrow-backed by default in pandas 2.x)
        if isinstance(dtype, pd.StringDtype):
            df[col] = df[col].astype(object)
        if isinstance(dtype, pd.CategoricalDtype) and isinstance(dtype.categories.dtype, pd.StringDtype):
            df[col] = df[col].astype(object)
        # Handle PyArrow-backed types
        elif hasattr(dtype, 'pyarrow_dtype'):
            df[col] = df[col].astype(object)
        # Handle categorical with PyArrow storage
        elif isinstance(dtype, pd.CategoricalDtype):
            # Convert to string first to avoid issues
            df[col] = df[col].astype(str)
    
    return df


def get_removals_schema() -> pa.DataType:
    """Get the PyArrow schema for the removals column.
    
    Removals are stored as list of structs with proper types:
    - removal_num: uint32 (removal index, 0 to 4.3B)
    - id: uint32 (node ID, 0 to 4.3B)
    - prediction: float32 (predicted value)
    - lcc_size: uint32 (absolute LCC node count, not fraction)
    - slcc_size: uint32 (absolute SLCC node count, not fraction)
    
    Fractions are computed at runtime: lcc_fraction = lcc_size / network_size
    
    Returns:
        PyArrow list of struct type.
    """
    return pa.list_(pa.struct([
        ('removal_num', pa.uint32()),
        ('id', pa.uint32()),
        ('prediction', pa.float32()),
        ('lcc_size', pa.uint32()),      # absolute count, not fraction
        ('slcc_size', pa.uint32()),     # absolute count, not fraction
    ]))


def convert_removals_to_struct(removals_list: RemovalsList) -> List[Dict[str, Union[int, float]]]:
    """Convert removals from Removal objects or tuples to list of dicts for PyArrow struct.
    
    Expects removals with ABSOLUTE counts (not fractions).
    Accepts both Removal dataclass objects and tuples.
    
    Args:
        removals_list: List of Removal objects or tuples (removal_num, id, prediction, lcc_size_absolute, slcc_size_absolute)
        
    Returns:
        List of dicts with named fields, or None if input is None/empty.
    """
    if removals_list is None or not removals_list:
        return None
    
    if isinstance(removals_list, str):
        # Should not happen, but handle it
        from ast import literal_eval
        removals_list = literal_eval(removals_list)
    
    # Import Removal to check type
    from network_dismantling.common.removal import Removal
    
    # # Convert Removal objects to tuples if needed
    if removals_list and isinstance(removals_list[0], Removal):
        removals_list = [r.to_tuple() for r in removals_list]
    # elif removals_list and isinstance(removals_list[0], tuple):

    # Convert list of tuples to list of dicts (values already absolute)
    return [
        {
            'removal_num': int(r[0]),
            'id': int(r[1]),
            'prediction': float(r[2]),
            'lcc_size': int(r[3]),      # already absolute
            'slcc_size': int(r[4]),     # already absolute
        }
        for r in removals_list
    ]


def convert_removals_from_struct(removals_list):
    """Convert removals from PyArrow struct (list of dicts) back to list of tuples.
    
    Values remain as absolute counts (no conversion to fractions).
    
    Args:
        removals_list: List of dicts from PyArrow struct with absolute counts.
        
    Returns:
        List of tuples (removal_num, id, prediction, lcc_size_absolute, slcc_size_absolute).
    """
    if removals_list is None or not removals_list:
        return []
    
    # Convert list of dicts to list of tuples (values stay absolute)
    return [
        (
            int(r['removal_num']),
            int(r['id']),
            float(r['prediction']),
            int(r['lcc_size']),       # absolute (no division)
            int(r['slcc_size']),      # absolute (no division)
        )
        for r in removals_list
    ]


def convert_numpy_to_python(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numpy types to Python types in DataFrame (in place)."""
    return df

def df_writer(queue: multiprocessing.Queue,
              output_file: Union[Path, str],
              output_columns: Optional[List[str]] = None,
              mode: Literal['overwrite', 'append'] = 'overwrite',
              row_group_size: int = 100000,
              error_event: Optional[threading.Event] = None,
              logger: logging.Logger = logging.getLogger("dummy"),
              ):
    """Write dataframes to a parquet file using fastparquet with native append support.
    
    Args:
        queue: A multiprocessing queue to receive dataframes.
        output_file: The path to the output file.
        output_columns: The columns to write to the file.
        row_group_size: Target row group size for compression optimization.
        error_event: Optional Event to signal fatal errors to the caller.
        mode: Write mode - 'overwrite' (default) or 'append'.
        logger: A logger to log messages.
    """
    output_file = Path(output_file).resolve()
    if not output_file.parent.exists():
        output_file.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    first_write = True
    
    try:
        while True:
            record: Union[pd.DataFrame, None] = queue.get()

            if record is None:
                logger.debug(f"Received sentinel. Closing writer for {output_file}.")
                break

            if not isinstance(record, pd.DataFrame):
                logger.error(f"Received non-DataFrame: {type(record)}. Skipping.")
                continue

            if len(record) == 0:
                logger.warning(f"Received empty DataFrame. Skipping write.")
                continue

            # Reorder columns if specified
            if output_columns:
                try:
                    record = record[output_columns]
                except KeyError as e:
                    logger.error(f"Missing column(s) in DataFrame: {e}")
                    raise

            # Convert removals column to struct format if present (values already absolute)
            if 'removals' in record.columns:
                record['removals'] = record['removals'].apply(convert_removals_to_struct)

            # Prepare DataFrame for fastparquet (convert PyArrow types to native types)
            record = prepare_dataframe_for_fastparquet(record)

            # Determine write mode for this iteration
            if first_write:
                # First write: use mode parameter (overwrite or append)
                # If file doesn't exist, we must use overwrite regardless of mode
                if output_file.exists() and mode == 'append':
                    write_mode = 'append'
                else:
                    write_mode = 'overwrite'
                
                first_write = False
                logger.info(f"First write to {output_file} with mode='{write_mode}'")
            else:
                # Subsequent writes: always append
                write_mode = 'append'

            logger.info(f"Writing {len(record)} rows to {output_file} (mode={write_mode})\n{record.dtypes}\n{record.head()}")

            # Write using fastparquet
            record.to_parquet(
                str(output_file),
                engine='fastparquet',
                compression='snappy',
                append=(write_mode == 'append'),
                row_group_offsets=row_group_size,
                index=False,
            )
            
            rows_written += len(record)
            logger.debug(f"Wrote {len(record)} rows to {output_file} (mode={write_mode}). Total: {rows_written}")

    except Exception as e:
        logger.exception(f"Fatal error in df_writer for {output_file}: {e}")
        if error_event is not None:
            error_event.set()  # Signal error to caller
        # Don't re-raise - would cause unhandled thread exception warning

    finally:
        logger.info(f"Writer finished for {output_file}. Total rows written: {rows_written}")


def start_df_writer(output_file: Path,
                    output_df_columns: Union[str, List[str]],
                    df_queue: multiprocessing.Queue,
                    mode: str = 'overwrite',
                    row_group_size: int = 100000,
                    logger: logging.Logger = logging.getLogger("dummy"),
                    ) -> threading.Thread:
    """Start a non-daemon thread to write dataframes to a parquet file.
    
    .. deprecated::
        Use ParquetDataFrameWriter class instead for better error handling:
        
        Old way::
            queue = multiprocessing.Queue()
            writer = start_df_writer(output_file=path, output_df_columns=cols, df_queue=queue)
            queue.put(data)
            queue.put(None)
            writer.join()
        
        New way::
            with ParquetDataFrameWriter(path, cols, logger) as writer:
                writer.write(data)
    
    The writer thread reads DataFrames from a queue and appends them to a single
    Parquet file using fastparquet for efficient native append operations.
    
    Args:
        output_file: Path to the output .parquet file.
        output_df_columns: Columns to write to the .parquet file.
        df_queue: Queue to read DataFrames from. Put None to signal end of data.
        row_group_size: Target row group size for compression optimization (default: 100k).
        mode: Write mode - 'overwrite' (default) or 'append'.
        logger: Logger for logging messages.

    Returns:
        The thread writing the dataframe to the .parquet file.
        IMPORTANT: Caller should check thread.is_alive() periodically to detect errors.
    """
    # Create error event for signaling fatal errors
    error_event = threading.Event()

    dp = threading.Thread(
        target=df_writer,
        kwargs=dict(
            queue=df_queue,
            output_file=output_file,
            output_columns=output_df_columns,
            row_group_size=row_group_size,
            mode=mode,
            error_event=error_event,
            logger=logger,
        ),
        daemon=False,  # Changed to False to ensure data is written before shutdown
        name=f"ParquetWriter-{Path(output_file).name}",
    )

    # Attach error_event to thread for caller to check
    dp.error_event = error_event  # type: ignore

    dp.start()
    logger.info(f"Started ParquetWriter thread for {output_file}")
    return dp


def safe_queue_put(df_queue: multiprocessing.Queue,
                   data: Union[pd.DataFrame, None],
                   writer_thread: threading.Thread,
                   logger: logging.Logger = logging.getLogger("dummy"),
                   ) -> bool:
    """Safely put data in queue, checking if writer thread is still alive.
    
    This helper prevents silently pushing data to a dead writer thread.
    
    Args:
        df_queue: The queue to put data into.
        data: DataFrame to write, or None for sentinel.
        writer_thread: The writer thread to monitor.
        logger: Logger for error messages.
        
    Returns:
        True if data was queued successfully, False if writer is dead.
        
    Raises:
        RuntimeError: If writer thread has died with an error.
        
    Example:
        writer = start_df_writer(...)
        for run_data in runs:
            if not safe_queue_put(queue, run_data, writer, logger):
                raise RuntimeError("Writer died, aborting")
        safe_queue_put(queue, None, writer, logger)  # Sentinel
        writer.join()
    """
    # Check if writer is alive
    if not writer_thread.is_alive():
        logger.error(f"Writer thread '{writer_thread.name}' has died!")
        # Check if it signaled an error
        if hasattr(writer_thread, 'error_event') and writer_thread.error_event.is_set():
            logger.error("Writer encountered a fatal error. Check logs above.")
        raise RuntimeError(f"Writer thread '{writer_thread.name}' terminated unexpectedly")

    # Check if error was signaled
    if hasattr(writer_thread, 'error_event') and writer_thread.error_event.is_set():
        logger.error(f"Writer thread '{writer_thread.name}' signaled an error!")
        raise RuntimeError(f"Writer thread '{writer_thread.name}' encountered an error")

    # Safe to queue
    df_queue.put(data)
    return True


class ParquetDataFrameWriter(BaseDataFrameWriter):
    """Thread-safe Parquet writer for incremental DataFrame writing.
    
    This class manages a background thread that writes DataFrames to a Parquet file
    using fastparquet for efficient native append support. It provides automatic
    error detection and prevents silently losing data if the writer thread fails.
    
    Usage with ProcessPoolExecutor (workers return DataFrames):
        with ParquetDataFrameWriter(output_file, columns, logger) as writer:
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(worker_func, ...) for ...]
                for future in futures:
                    result_df = future.result()
                    writer.write(result_df)
    
    Usage for sequential writing:
        with ParquetDataFrameWriter(output_file, columns, logger) as writer:
            for run_data in runs:
                writer.write(run_data)
    """

    def __init__(self,
                 output_file: Union[Path, str],
                 columns: Union[str, List[str]],
                 row_group_size: int = 100000,
                 mode: str = "overwrite",
                 logger: logging.Logger = logging.getLogger("dummy"),
                 ):
        """Initialize the Parquet writer.
        
        Args:
            output_file: Path to output .parquet file.
            columns: Column names to write.
            logger: Logger for messages.
            row_group_size: Target row group size (default: 100k).
            mode: Write mode - 'overwrite' (default) or 'append'.
                  With fastparquet, append is natively supported (no file re-reading).
        """
        self.row_group_size = row_group_size
        self.mode = mode

        # Create queue and error event before calling super().__init__
        self._queue: multiprocessing.Queue = multiprocessing.Queue()
        self._error_event = threading.Event()

        # Call base class constructor (will call _create_writer_thread)
        super().__init__(output_file, columns, logger)

    def _create_writer_thread(self) -> threading.Thread:
        """Create the Parquet writer thread."""
        return threading.Thread(
            target=df_writer,
            kwargs=dict(
                queue=self._queue,
                output_file=self.output_file,
                output_columns=self.columns,
                mode=self.mode,
                row_group_size=self.row_group_size,
                error_event=self._error_event,
                logger=self.logger,
            ),
            daemon=False,
            name=f"ParquetWriter-{self.output_file.name}",
        )

    def _send_sentinel(self):
        """Send sentinel to stop the writer thread."""
        self._queue.put(None)

    def _check_writer_alive(self):
        """Check if writer thread is still alive and raise if not."""
        if not self._thread.is_alive() and not self._closed:
            self.logger.error(f"Writer thread died unexpectedly!")
            raise RuntimeError(f"Writer thread '{self._thread.name}' terminated")

        if self._error_event.is_set():
            self.logger.error(f"Writer thread encountered an error!")
            raise RuntimeError(f"Writer thread '{self._thread.name}' signaled error")

    def write(self, df: pd.DataFrame):
        """Write a DataFrame to the Parquet file.
        
        Args:
            df: DataFrame to write.
            
        Raises:
            RuntimeError: If writer thread has died or encountered an error.
            ValueError: If writer is already closed.
        """
        if self._closed:
            raise ValueError("Cannot write to closed ParquetDataFrameWriter")

        # TODO check df columns match self.columns?
        # TODO validate df is a DataFrame?
        # TODO check df is not empty?
        # TODO if the thread is dead, raise error or try to restart it?
        self._check_writer_alive()
        self._queue.put(df)
        self.logger.debug(f"Queued {len(df)} rows for writing")

    def close(self, timeout: float = 30.0):
        """Close the writer and wait for all data to be written.
        
        This sends a sentinel to the writer thread and waits for it to finish.
        Should be called when all data has been written.
        
        Args:
            timeout: Maximum seconds to wait for thread to finish.
        """
        # Call parent close method
        super().close(timeout=timeout)

        # Check if error occurred during shutdown
        if self._error_event.is_set():
            self.logger.error("Writer encountered an error during shutdown")
            raise RuntimeError("Writer thread failed during shutdown")


def read_parquet_schema_df(uri: str) -> pd.DataFrame:
    """Return a Pandas DataFrame with the schema of a parquet file.

    Returns a dataframe with columns: column, pa_dtype

    Args:
        uri: Path to the parquet file.

    Returns:
        DataFrame with schema information.
    """
    schema = pq.read_schema(uri, memory_map=True)
    schema_df = pd.DataFrame({
        "column": schema.names,
        "pa_dtype": [str(dtype) for dtype in schema.types]
    })

    return schema_df


def get_df_columns(file: Path):
    schema = read_parquet_schema_df(str(file))

    cols = schema["column"].tolist()

    return cols


def read_without_removals(file,
                          exclude_columns: Union[str, List[str], None] = None,
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
        logger: logging.Logger = logging.getLogger("dummy"),
):
    if not isinstance(file, Path):
        file = Path(file)
    file = file.resolve()
    if not file.exists():
        raise FileNotFoundError(f"Input file {file} does not exist.")
    if not file.is_file():
        raise FileNotFoundError(f"Input file {file} is not a file.")
    if not file.suffix == ".parquet":
        raise ValueError(f"Input file {file} is not a parquet file.")

    if exclude_columns is None:
        exclude_columns = []

    elif isinstance(exclude_columns, str):
        exclude_columns = [exclude_columns]

    elif not isinstance(exclude_columns, list):
        raise ValueError(f"Invalid exclude_columns {exclude_columns} (type {type(exclude_columns)}.")

    # 1. determine columns to load
    cols = get_df_columns(file)
    usecols = [c for c in cols if c not in exclude_columns]

    if read_index is not None:
        # 2. prepare sorted list of global indices
        if isinstance(read_index, int):
            indices = [read_index]
        elif isinstance(read_index, (list, np.ndarray, pd.Series, set, tuple)):
            indices = sorted(read_index)
        else:
            raise ValueError(f"Invalid read_index {read_index} (type {type(read_index)}.")

        pf = ParquetFile(str(file))
        logger.info(f"Attributes of ParquetFile: {dir(pf)}")
        meta = pf.metadata

        # 3. collect row counts and compute cumulative offsets
        rg_counts = [meta.row_group(i).num_rows for i in range(meta.num_row_groups)]
        offsets = list(accumulate(rg_counts, initial=0))

        total_rows = offsets[-1]

        # 4. map each index to (group, local_index)
        group_map = defaultdict(list)
        for idx in indices:
            if idx < 0 or idx >= total_rows:
                raise ValueError(f"Index {idx} out of bounds for file {file}.")

            # Bisect right will find the first group with a start index greater than idx
            grp = bisect_right(offsets, idx) - 1
            local_idx = idx - offsets[grp]
            # group_map.setdefault(grp, []).append((idx, local_idx))

            group_map[grp].append((idx, local_idx))

        # 5. read each group once and slice all needed rows
        rows = []
        for grp in sorted(group_map):
            tbl = pf.read_row_group(grp,
                                    columns=usecols,
                                    use_threads=True,
                                    )

            # Slice the table to get the rows we need
            # df_grp = tbl.to_pandas()
            for global_idx, local_idx in group_map[grp]:
                row = tbl.slice(local_idx, 1).to_pandas()
                row["idx"] = global_idx
                rows.append(row)
                # row = df_grp.iloc[[local_idx]].copy()
                # row["idx"] = global_idx
                # rows.append(row)

        # Concatenate all rows into a single dataframe
        df = pd.concat(rows, ignore_index=True)
    else:
        # full read
        df = pd.read_parquet(str(file),
                             columns=usecols,
                             engine="pyarrow",
                             # dtype=dtype_dict,
                             )
        df["idx"] = df.index

    df["file"] = f"{file}"
    # df["file"] = df["file"].astype("category")

    # 6. cast dtypes
    if dtype_dict is None:
        dtype_dict = {}

    dtype_dict.setdefault("network", "category")
    dtype_dict.setdefault("file", "category")

    for dtype_col, dtype in dtype_dict.items():
        if dtype_col not in df.columns:
            continue

        if dtype == "category":
            df[dtype_col] = df[dtype_col].astype("category")
        else:
            df[dtype_col] = df[dtype_col].astype(dtype)

    return df


# def read_without_columns(
#         file,
#         exclude_columns: Union[str, List[str]],
#         read_index: Union[None, int, List[int]] = None,
#         dtype_dict=None,
# ):
#     if exclude_columns is None:
#         exclude_columns = []
#
#     if isinstance(exclude_columns, str):
#         exclude_columns = [exclude_columns]
#
#     # Read column names from file
#     cols = get_df_columns(file)
#     usecols = [i for i in cols if i not in exclude_columns]
#
#     read_kwargs = {
#         "columns": usecols,
#         "engine": "pyarrow",
#         "use_threads": True,
#         "dtype": dtype_dict,
#     }
#     # Read the schema of the file
#     indices_to_read = None
#     if read_index is not None:
#         if isinstance(read_index, int):
#             indices_to_read = [read_index]
#
#         indices_to_read = sorted(indices_to_read)
#
#         buffer = []
#         first_row_of_group = 0 #-1
#         parquet_file = ParquetFile(str(file))
#
#         # Get metadata
#         metadata = parquet_file.metadata
#         if metadata is None:
#             raise ValueError(f"Metadata is None for file {file}.")
#         if metadata.num_row_groups < 1:
#             raise ValueError(f"Metadata has no row groups for file {file}.")
#
#         # Define a groups generator:
#         groups_iterator = enumerate(iter(range(parquet_file.num_row_groups)))
#         i, group = next(groups_iterator)
#         row_group_metadata = metadata.row_group(i)
#         row_group = None
#         current_row_group = None
#
#         # # Print number of rows in each row group
#         # for i in range(metadata.num_row_groups):
#         #     row_group = metadata.row_group(i)
#         for index_to_read in indices_to_read:
#             if index_to_read < 0:
#                 raise ValueError(f"Invalid read_index {index_to_read} (type {type(index_to_read)}.")
#
#             # if index_to_read >= metadata.num_rows:
#             #     raise ValueError(f"Index {index_to_read} is out of bounds for file {file}.")
#
#             # Check if the index_to_read is NOT in the current row group
#             if not (first_row_of_group <= index_to_read < first_row_of_group + row_group_metadata.num_rows):
#                 # Navigate to the row group that contains the index_to_read
#                 for i, group in groups_iterator:
#                     # Get the row group that contains the index_to_read
#                     row_group_metadata = metadata.row_group(i)
#                     first_row_of_group += row_group_metadata.num_rows
#                     print(f"Row group {i} has {row_group_metadata.num_rows} rows.")
#                     print(f"First row of group {i} is {first_row_of_group}.")
#
#                 if first_row_of_group <= index_to_read < first_row_of_group + row_group_metadata.num_rows:
#                     break
#
#                 else: # This else is executed when the for loop is exhausted and break is not executed
#                     # We have exhausted all row groups and didn't find the index_to_read
#                     # If we reach here, it means we didn't find the index_to_read in any group
#                     raise ValueError(f"Index {index_to_read} is out of bounds for file {file}.")
#
#             if current_row_group != group: # Check if we are in a new row group or we have never loaded one
#                 # If we reach here, it means we found the index_to_read in group i
#                 current_row_group = group
#                 print(f"Reading row group {group}.")
#                 # Get the row group that contains the index_to_read
#                 row_group = parquet_file.read_row_group(group,
#                                                         columns=usecols,
#                                                         use_threads=True,
#                                                         # memory_map=True,
#                                                         )
#
#
#             # Filter the desired row
#             read_df = row_group.slice(index_to_read, 1)
#
#             # Convert to pandas dataframe
#             read_df = read_df.to_pandas()
#
#             # # read_df = parquet_file.read_row_group(index_to_read, columns=usecols).to_pandas()
#             # # first_ten_rows = next(pf.iter_batches(batch_size=10))
#             # # read_df = pa.Table.from_batches([first_ten_rows]).to_pandas()
#             #
#             # read_df = pd.read_parquet(
#             #     str(file),
#             #     skiprows=index_to_read + 1,
#             #     nrows=1,
#             #     # columns=usecols,
#             #     # usecols=usecols,
#             #     # names=usecols,
#             #     # dtype=dtype_dict,
#             #     # engine="pyarrow",
#             #     **read_kwargs,
#             # )
#             read_df["idx"] = index_to_read
#
#             buffer.append(read_df)
#
#         df = pd.concat(buffer,
#                        ignore_index=True,
#                        )
#     else:
#         df = pd.read_parquet(
#             str(file),
#             # usecols=usecols,
#             # dtype=dtype_dict,
#             # engine="pyarrow",
#             **read_kwargs,
#         )
#         df["idx"] = df.index
#
#     df["file"] = f"{file}"
#     df["file"] = df["file"].astype("category")
#
#     return df


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
            logger=logger,
        )

        if (not include_removals) and (expected_columns):
            if ("removals" in expected_columns):
                expected_columns.remove("removals")

        if expected_columns is not None:
            for column in ["idx", "file"]:
                if column not in expected_columns:
                    expected_columns += [column]

            if (len(df.columns) != len(expected_columns)) or (df.columns != expected_columns).all():
                raise ValueError(
                    f"Input file {file} columns {list(df.columns)} "
                    f"do not match the expected columns {expected_columns}."
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
    # if "network" in df and df["network"].dtype != str:
    #     df["network"] = df["network"].astype(str)

    return df
