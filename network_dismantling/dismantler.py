#   This file is part of the Network Dismantling review,
#   proposed in the paper "Robustness and resilience of complex networks"
#   by Oriol Artime, Marco Grassia, Manlio De Domenico, James P. Gleeson,
#   Hernán A. Makse, Giuseppe Mangioni, Matjaž Perc and Filippo Radicchi.
#
#   This is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   The project is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with the code.  If not, see <http://www.gnu.org/licenses/>.

# TODO move the todos to the GitHub discussion and issues

# TODO: Parameters:
#       - allow to pass parameters to the heuristics
#       - define a way to configure the parameters of the heuristics, and show information / errors
#           if the parameters are not correct
#       - allow heuristics to re-run the same network with different parameters
#           and/or to complete some missing runs
#       - when providing dependencies, check if the dependency was already run with the requested parameters!
# TODO allow parallel execution of the heuristics
# TODO improve pool performance by using a single pool for all the heuristics.
#       Can we spawn a worker for each network and heuristic?
#       A worker for each heuristic is not a good idea, if they have multiple parameters.
#       Moreover, the network processing should be done only once anyway to avoid overhead...
#       Plus, I really don't like the idea of having a ton of tasks submitted to the pool at the same time.
#       I would like to submit them in batches.
# TODO DataFrames:
#       - handle common errors like broken dataframes, missing columns, etc...
#       - store dataframes in binary format to reduce the size of the output file?
#       - compress the output file? Cleanup the removals of suboptimal solutions? Of intermediate results?
#       - make filtering the DataFrames faster.
# TODO (big todo actually): it would be nice to use boost data structures and pass the data to the heuristics
#       without using text edge lists.

import argparse
import logging
from ast import literal_eval
from datetime import timedelta
from logging.handlers import QueueHandler
from operator import itemgetter
from pathlib import Path
from time import time
from typing import Callable, Union, Dict, List

import numpy as np
import pandas as pd
from graph_tool import Graph
from tqdm.auto import tqdm


try:
    from torch import multiprocessing, cuda
except ImportError:
    import multiprocessing

    # TODO maybe improve this cuda mock?
    from types import SimpleNamespace

    cuda = SimpleNamespace(is_available=lambda: False)

from network_dismantling.common.dataset_providers import (
    list_files,
    load_single_network,
)

# # Remove the OpenMP threads. Use data parallelism instead
# from graph_tool.all import openmp_set_num_threads
# from os import environ
# environ["OMP_NUM_THREADS"] = "1"
# openmp_set_num_threads(1)

logger = None


def pool_initializer(log_queue,
                     log_level=logging.INFO,
                     lock: Union[multiprocessing.Lock, None] = None,
                     ):
    global logger

    logging.basicConfig(
        format="%(message)s",
        # stream=sys.stdout,
        level=log_level,
        handlers=[QueueHandler(log_queue)],
        # datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger(__name__)
    # queue_logger.setLevel(logging.DEBUG)

    if lock is not None:
        tqdm.set_lock(lock)


def get_predictions(
        network: Graph,
        sorting_function: Callable,
        logger=logging.getLogger("dummy"),
        **kwargs,
):
    logger.debug(f"Calling the sorting function...")
    start_time = time()

    values = sorting_function(network, **kwargs)

    time_spent = time() - start_time
    logger.debug(f"Heuristics returned. Took {timedelta(seconds=(time_spent))}")

    return values, time_spent


def validate_heuristic_imports(heuristics: List[str],
                               logger: logging.Logger = logging.getLogger("dummy"),
                               ) -> List[str]:
    """Test if imports needed by heuristics are available."""
    from network_dismantling import dismantling_methods, DismantlingMethod
    valid_heuristics = []
    for heuristic in heuristics:
        # Check if heuristic exists
        if heuristic not in dismantling_methods:
            logger.error(f"Heuristic '{heuristic}' not found in available methods")
            continue

        dismantling_method: DismantlingMethod = dismantling_methods[heuristic]

        # Check if the heuristic has required imports
        if hasattr(dismantling_method, 'required_imports') and dismantling_method.required_imports:
            missing_imports = []
            for module_name in dismantling_method.required_imports:
                try:
                    __import__(module_name)
                except ImportError:
                    missing_imports.append(module_name)

            if missing_imports:
                logger.warning(
                    f"Heuristic {dismantling_method.short_name} requires missing imports: {', '.join(missing_imports)}. "
                    f"This heuristic will be skipped."
                )
                continue

        valid_heuristics.append(heuristic)

    return valid_heuristics


def check_dependencies(heuristics: List[str],
                       logger: logging.Logger = logging.getLogger("dummy"),
                       ):
    """Check and resolve dependencies between heuristics, including cyclic dependency detection."""
    from network_dismantling import dismantling_methods, DismantlingMethod
    
    # Track dependency chain to detect cycles
    dependency_chain = set()
    
    def check_cyclic_dependency(heuristic_key: str, chain: set) -> bool:
        """Recursively check for cyclic dependencies."""
        if heuristic_key in chain:
            return True

        method = dismantling_methods.get(heuristic_key)
        if method is None or method.depends_on is None:
            return False

        dependency_key = method.depends_on.key if hasattr(method.depends_on, 'key') else method.depends_on
        return check_cyclic_dependency(dependency_key, chain | {heuristic_key})

    # Check for cyclic dependencies before processing
    for heuristic in heuristics:
        # Check if heuristic exists
        if heuristic not in dismantling_methods:
            logger.error(f"Heuristic '{heuristic}' not found in available methods")
            raise KeyError(f"Heuristic '{heuristic}' not found in available methods")

        if check_cyclic_dependency(heuristic, set()):
            logger.error(f"Cyclic dependency detected for heuristic {heuristic}")
            raise ValueError(f"Cyclic dependency detected in heuristics chain starting from {heuristic}")

    # Reverse the list to check the dependencies in the correct order
    heuristics = heuristics[::-1]
    for i, heuristic in enumerate(heuristics):
        dismantling_method: DismantlingMethod = dismantling_methods[heuristic]
        display_name: str = dismantling_method.short_name

        depends_on: DismantlingMethod | str | None = dismantling_method.depends_on
        logger.debug(f"Checking dependencies for heuristic {display_name}")
        logger.debug(f"Depends on: {depends_on} type {type(depends_on)}")

        if depends_on is not None:
            logger.debug(f"Dismantling method {display_name} depends on {depends_on}")
            if isinstance(depends_on, str):
                depends_on = dismantling_methods.get(depends_on, None)

                if depends_on is None:
                    logger.error(f"Dependency {dismantling_method.depends_on} not found for heuristic {display_name}")
                    continue

            if depends_on.key not in heuristics:
                heuristics.insert(i + 1, depends_on.key)
                logger.info(f"Added dependency {depends_on.short_name} for heuristic {display_name}")

            elif heuristics.index(depends_on.key) < i:
                heuristics.remove(depends_on.key)
                heuristics.insert(i, depends_on.key)
                logger.debug(f"Moved dependency {depends_on.short_name} for heuristic {display_name}")
        else:
            logger.debug(f"Heuristic {display_name} does not depend on any other heuristic")

    # Reverse the list to run the heuristics in the correct order
    heuristics = heuristics[::-1]

    logger.debug(f"Final heuristics list: {heuristics}")

    return heuristics


def main(args: argparse.Namespace,
         logger: logging.Logger = logging.getLogger("dummy")
         ):
    from multiprocessing.managers import SyncManager

    try:
        from deadpool import Deadpool as ProcessPoolExecutor

        pool_kwargs = {
            # "max_tasks_per_child": 25,
        }

    except ImportError:
        logger.warning("Deadpool not found. Using ProcessPoolExecutor instead.")

        from concurrent.futures import ProcessPoolExecutor

        pool_kwargs = {}

    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # Create the Multiprocessing Manager
    mp_context: multiprocessing.context = multiprocessing.get_context("spawn")

    mp_manager: multiprocessing.Manager = SyncManager(ctx=mp_context)
    mp_manager.start()
    # mp_manager: multiprocessing.Manager = multiprocessing.Manager()

    # Create the Dataset Queue
    df_queue: Queue = mp_manager.Queue()

    # Create the Log Queue
    log_queue: Queue = mp_manager.Queue()

    # Create and start the Logger Thread
    lp = threading.Thread(
        target=logger_thread,
        args=(
            logger,
            log_queue,
        ),
    )
    lp.start()

    # Create and start the Dataset Writer Thread
    dp: threading.Thread = start_df_writer(#args=args,
                                           output_file=args.output_file,
                                           output_df_columns=args.output_df_columns,
                                           df_queue=df_queue,
                                           logger=logger,
                                           )

    # List the networks. Do not load them yet to save memory and CPU time.
    test_networks_list = list_files(
        args.location,
        max_num_vertices=args.max_num_vertices,
        filter=args.filter,
        targets=None,
        # manager=mp_manager,
    )

    if len(test_networks_list) == 0:
        logger.info(
            f"No networks found in {[str(loc) for loc in args.location]} with filters {args.filter} ."
        )

    reader_kwargs = dict(
        # expected_columns=args.output_df_columns,
        at_least_one_file=False,
        raise_on_missing_file=False,
        dtype_dict={
            "network": "category",
            "heuristic": "category",
            # "rem_num"
        },
        logger=logger,
    )
    if args.input is not None:
        expected_columns = args.output_df_columns.copy()
        expected_columns.remove("removals")

        files = args.input

    else:
        expected_columns = args.output_df_columns
        files = args.output_file

    df: pd.DataFrame = df_reader(
        files=files,

        expected_columns=expected_columns,
        include_removals=False,

        **reader_kwargs,
    )

    # Keep only the rows with the same threshold
    df = df[df["threshold"] == args.threshold]

    # Create the pool
    with (
        ProcessPoolExecutor(
            max_workers=args.jobs,
            mp_context=mp_context,
            initializer=pool_initializer,
            initargs=(log_queue,
                      logger.level,
                      multiprocessing.Lock(),
                      ),
            # initializer=tqdm.set_lock,
            # initargs=(multiprocessing.Lock(),),
            **pool_kwargs,
        ) as executor,

        ParquetDataFrameWriter(
            output_file=args.output_file,
            columns=args.output_df_columns,
            logger=logger,
        ) as parquet_writer,

        tqdm(
            test_networks_list,
            desc="Networks",
            position=0,
        ) as tqdm_test_network_list
    ):
        # noinspection PyTypeChecker
        for network_path in tqdm_test_network_list:
            network: Union[Graph, None] = None
            network_loaded: bool = False
            network_size: Union[int, None] = None
            stop_condition: Union[int, None] = None
            generator_args: Union[Dict, None] = None

            network_name: str = network_path.stem

            tqdm_test_network_list.set_description(f"Networks ({network_name})")

            # Check if the network was already tested
            # Get the rows of the dataframe with the same network network_name
            # Note that the network is a categorical column
            # Avoid .loc for performance reasons
            network_df = df[df["network"] == network_name]

            logger.debug(f"Network {network_name} has {network_df.shape[0]} rows in the dataframe\n{network_df}")

            with tqdm(args.heuristics,
                      desc="Heuristics",
                      position=1,
                      ) as tqdm_heuristics:
                # Iterate over the heuristics
                for heuristic in tqdm_heuristics:
                    dismantling_method: DismantlingMethod = dismantling_methods[heuristic]

                    logger.info(f"Running heuristic {dismantling_method.short_name} "
                                f"with threshold {args.threshold}"
                                )
                    df_filtered = network_df[network_df["heuristic"] == dismantling_method.key]

                    # TODO also check if all the requested metrics are present?
                    if len(df_filtered) != 0:
                        # Nothing to do. The network was already tested
                        continue

                    if not network_loaded:
                        # Delay the network loading until the heuristic is actually run.
                        # This is meant to avoid loading the network if it is not needed,
                        # e.g., all the heuristics have been already run on the network.

                        network = load_single_network(
                            network_name,
                            network_path,
                            max_num_vertices=args.max_num_vertices,
                            logger=logger,
                        )

                        if network is None:
                            continue
                        
                        # Compute network properties
                        network_size = network.num_vertices()
                        stop_condition = np.ceil(network_size * args.threshold)

                        generator_args = {
                            "network_name": network_name,
                            "stop_condition": int(stop_condition),
                            "threshold": args.threshold,
                        }

                        # Mark that we've loaded the network
                        network_loaded = True

                    dismantling_method_kwargs = {}

                    if dismantling_method.depends_on is not None:
                        # Check if the dependency was already tested
                        df_dependency_filtered = network_df[
                            network_df["heuristic"] == dismantling_method.depends_on.key
                            ]

                        if len(df_dependency_filtered) == 0:
                            logger.error(
                                f"Dependency {dismantling_method.depends_on.short_name} not found "
                                f"for heuristic {dismantling_method.short_name}"
                            )
                            continue

                        if len(df_dependency_filtered) > 1:
                            logger.error(
                                f"More than one dependency {dismantling_method.depends_on.short_name} "
                                f"found for heuristic {dismantling_method.short_name}"
                            )
                            continue

                        # Get the removals from the dependency
                        df_dependency_filtered = df_dependency_filtered.iloc[0]

                        logger.debug(f"df_dependency_filtered: {df_dependency_filtered}")

                        # Check if removals are missing or invalid
                        removals = df_dependency_filtered.get("removals", None)
                        needs_reload = (
                                not removals or  # Handles None, "", [], etc.
                                removals == "[]"  # String representation of empty list
                        )

                        if needs_reload:

                            try:
                                df_dependency_row = df_reader(df_dependency_filtered["file"],
                                                              expected_columns=args.output_df_columns,
                                                              read_index=int(df_dependency_filtered["idx"]),
                                                              include_removals=True,
                                                              **reader_kwargs,
                                                              )
                            except Exception as e:
                                logger.error(
                                    f"Error while reading the dependency {dismantling_method.depends_on.display_name} "
                                    f"for heuristic {dismantling_method.display_name} from file {df_dependency_filtered['file']}:\n"
                                    f"{e}",
                                    exc_info=True,
                                )
                                continue

                            if (df_dependency_row.shape[0] != 1):
                                logger.error(
                                    f"Dependency {dismantling_method.depends_on.short_name} not found "
                                    f"for heuristic {dismantling_method.short_name}"
                                )
                                continue

                            dependency_run = df_dependency_row.iloc[0]

                            dependency_removals = dependency_run.pop("removals")

                            # logger.debug(f"Dependency run: {dependency_run}")
                            # logger.debug(f"Dependency df_dependency_filtered: {df_dependency_filtered}")
                            # logger.debug(f"Dependency removals: {dependency_removals}")

                            if not df_dependency_filtered.equals(dependency_run):
                                logger.error(
                                    f"Dependency {dismantling_method.depends_on.short_name} mismatch "
                                    f"for heuristic {dismantling_method.short_name}:\n"
                                    f"Original:\n{df_dependency_filtered}\n"
                                    f"Read:\n{dependency_run}"
                                )
                                continue

                            logger.debug(f"Dependency {dismantling_method.depends_on.display_name} "
                                         f"found for heuristic {dismantling_method.display_name}:\n"
                                         f"{df_dependency_filtered}")
                        else:
                            dependency_removals = df_dependency_filtered["removals"]

                        if dependency_removals is None:
                            logger.error(
                                f"Dependency {dismantling_method.depends_on.short_name} not found "
                                f"for heuristic {dismantling_method.short_name}"
                            )
                            continue

                        try:

                            if isinstance(dependency_removals, str):
                                dependency_removals = literal_eval(dependency_removals)

                            dependency_removals = list(map(itemgetter(RemovalsColumns.ID), dependency_removals))
                        except Exception as e:
                            logger.error(
                                f"Error while parsing the removals for the dependency {dismantling_method.depends_on.short_name} "
                                f"for heuristic {dismantling_method.short_name}:\n"
                                f"{e}"
                                f"Dependency removals: {dependency_removals}",
                                exc_info=True,
                            )
                            continue

                        dismantling_method_kwargs[dismantling_method.depends_on.key] = dependency_removals
                        if generator_args is not None:
                            generator_args[dismantling_method.depends_on.key] = dependency_removals

                    if network is None or stop_condition is None or network_size is None or generator_args is None:
                        logger.error(f"Network {network_name} was not properly loaded. Skipping heuristic {dismantling_method.short_name}")
                        continue

                    logger.debug(
                        f"Dismantling {network_name} according to {dismantling_method.short_name}. "
                        f"Aiming to LCC size {stop_condition} ({stop_condition / network_size:.3f})"
                    )
                    # logger.debug(f"dismantling_method_kwargs: {dismantling_method_kwargs}")

                    generator_args["executor"] = executor
                    generator_args["pool_size"] = args.jobs
                    generator_args["mp_manager"] = mp_manager

                    try:
                        # TODO REMOVE THE COPY OF THE NETWORK, and move where its actually needed
                        run = dismantling_method(
                            network=network.copy(),
                            threshold=args.threshold,

                            stop_condition=stop_condition,
                            generator_args=generator_args,
                            **dismantling_method_kwargs,

                            executor=executor,
                            pool_size=args.jobs,
                            mp_manager=mp_manager,
                            logger=logger,
                        )

                        run["network"] = network_name
                        run["threshold"] = args.threshold
                        # run["heuristic"] = dismantling_method.key

                        if isinstance(run, pd.Series):
                            run = run.to_dict()

                        if isinstance(run, dict):
                            logger.info(f"{dismantling_method.short_name} run info on {network_name}: "
                                        f"{run['rem_num']} removals, AUC {run['r_auc']:.3f}")

                        if isinstance(run, pd.DataFrame):
                            logger.info(f"{dismantling_method.short_name} run(s) for {network_name}:\n"
                                        f"{run}")
                            runs_dataframe = run[args.output_df_columns]

                        else:  # not isinstance(run, pd.DataFrame):
                            runs_dataframe = pd.DataFrame(
                                data=[run],
                                columns=args.output_df_columns,
                            )

                        if "file" in runs_dataframe.columns:
                            runs_dataframe = runs_dataframe.drop(columns=["file"])

                        # Update the dataframe with the new run(s)
                        network_df = pd.concat([network_df, runs_dataframe],
                                               ignore_index=True,
                                               )
                        # Write directly with the writer (checks if thread is alive)
                        parquet_writer.write(runs_dataframe)

                    except Exception as e:
                        logger.exception(
                            f"Error while dismantling network {network_name} with {dismantling_method.short_name}:\n"
                            f"{e}",
                            exc_info=True,
                        )

                        continue

        # Close the pool
        executor.shutdown(
            wait=True,
            cancel_futures=False,
        )
    df_queue.put(None)
    log_queue.put(None)

    dp.join()
    lp.join()


def get_df_columns():
    return [
        "network",
        "heuristic",
        "slcc_peak_at",
        "lcc_size_at_peak",
        "slcc_size_at_peak",
        "removals",
        "static",
        "r_auc",
        "rem_num",
        "prediction_time",
        "dismantle_time",
        "threshold",
    ]


if __name__ == "__main__":
    from network_dismantling import dismantling_methods, DismantlingMethod

    # Create the logger
    logging.basicConfig(
        format="%(asctime)s :: %(levelname)-8s :: %(processName)s :: %(message)s",
        # stream=sys.stdout,
        level=logging.DEBUG,
        handlers=[TqdmLoggingHandler()],
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=None,
        nargs="*",
        required=False,
        help="Heuristics input file. Will be used to skip already tested algorithms.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        required=True,
        help="Heuristics output file. Will be used to store the results of the runs.",
    )

    parser.add_argument(
        "-l",
        "--location",
        type=Path,
        default=None,
        required=True,
        nargs="+",
        help="Location of the dataset (directory)",
    )

    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.1,
        help="Dismantling target Largest Connected Component threshold [0,1]",
    )

    parser.add_argument(
        "-H",
        "--heuristics",
        type=str,
        choices=sorted(dismantling_methods.keys()) + ["all"],
        # default="all",
        required=True,
        nargs="+",
        help="Dismantling heuristics to run. "
             "See the repository README for more information.",
    )

    parser.add_argument(
        "-F",
        "--filter",
        type=str,
        default="*",
        nargs="*",
        required=False,
        help="Test folder filter",
    )

    parser.add_argument(
        "-mnv",
        "--max_num_vertices",
        type=int,
        default=float("inf"),
        help="Filter the networks given the maximum number of vertices.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        type=str.upper,
        choices=["INFO", "DEBUG", "WARNING", "ERROR"],
        default="info",
        help="Verbosity level (case insensitive)",
    )

    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="Number of jobs.",
    )

    # parser.add_argument(
    #     "-sa",
    #     "--simultaneous_access",
    #     type=int,
    #     default=float('inf'),
    #     help="Maximum number of simultaneous predictions on torch CUDA device.",
    # )

    args, cmdline_args = parser.parse_known_args()
    if cmdline_args:
        logger.warning(f"Unused arguments: {cmdline_args}")

    logger.setLevel(logging.getLevelName(args.verbose))

    if not args.output.is_absolute():
        args.output = args.output.resolve()

    elif args.output.exists():
        if not args.output.is_file():
            raise FileNotFoundError("Output file {} is not a file.".format(args.output))

    args.output_file = args.output

    if args.input is None:
        if args.output_file.exists():
            args.input = args.output_file

    else:
        if not isinstance(args.input, list):
            args.input = [args.input]

        args.input = [i.resolve() for i in args.input]

        if args.output not in args.input:
            args.input.append(args.output)

    if not args.output_file.parent.exists():
        try:
            args.output_file.parent.mkdir(parents=True)
            logger.info(f"Created output directory: {args.output_file.parent}")
        except OSError as e:
            logger.error(f"Failed to create output directory {args.output_file.parent}: {e}")
            raise

    args.output_df_columns = get_df_columns()

    logger.info(f"Output file {args.output_file}")

    if "all" in args.heuristics:
        args.heuristics = list(dismantling_methods.keys())

    logger.info(f"Running the following heuristics: {', '.join(args.heuristics)}")

    # Validate that required imports are available
    args.heuristics = validate_heuristic_imports(args.heuristics, logger=logger)

    if not args.heuristics:
        logger.error("No valid heuristics to run after import validation.")
        import sys

        sys.exit(1)

    # Check the dependencies of the heuristics
    args.heuristics = check_dependencies(args.heuristics, logger=logger)

    # Show the citation for the heuristics
    for heuristic in args.heuristics:
        dismantling_method: DismantlingMethod = dismantling_methods[heuristic]
        display_name: str = dismantling_method.name
        display_name_short: str = dismantling_method.short_name
        logger.info(
            f"\n"
            f"==================================\n"
            f"Cite {display_name} ({display_name_short}) as:\n"
            f"{dismantling_method.citation.strip()}\n"
            f"==================================\n"
        )

    main(args=args,
         logger=logger,
         )
