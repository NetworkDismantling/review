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
# TODO define a way to test the imports needed by the heuristics, and show a warning if they are not installed
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
import threading
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
from network_dismantling.common.logger import logger_thread
from tqdm.auto import tqdm

try:
    from torch import multiprocessing, cuda
except ImportError:
    import multiprocessing

    # TODO maybe improve this cuda mock?
    cuda = object()
    cuda.is_available = lambda: False

from network_dismantling.common.dataset_providers import (
    list_files,
    init_network_provider,
)
from network_dismantling.common.df_helpers import df_reader, RemovalsColumns
from network_dismantling.common.multiprocessing import (
    TqdmLoggingHandler,
    dataset_writer,
)

# # Remove the OpenMP threads. Use data parallelism instead
# from graph_tool.all import openmp_set_num_threads
# from os import environ
# environ["OMP_NUM_THREADS"] = "1"
# openmp_set_num_threads(1)

logger = None


def pool_initializer(log_queue,
                     log_level=logging.INFO,
                     lock: multiprocessing.Lock = None,
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


def check_dependencies(heuristics: List[str],
                       logger: logging.Logger = logging.getLogger("dummy"),
                       ):
    # Reverse the list to check the dependencies in the correct order
    # TODO cyclic dependencies are not detected
    heuristics = heuristics[::-1]
    for i, heuristic in enumerate(heuristics):
        dismantling_method: DismantlingMethod = dismantling_methods[heuristic]
        display_name: str = dismantling_method.short_name

        depends_on = dismantling_method.depends_on
        logger.debug(f"Dismantling method {display_name} depends on {depends_on}")
        if depends_on is not None:
            if isinstance(depends_on, str):
                depends_on = dismantling_methods.get(depends_on.key, None)

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

    # Reverse the list to run the heuristics in the correct order
    heuristics = heuristics[::-1]

    logger.debug(f"Final heuristics list: {heuristics}")

    return heuristics


def main(args, logger=logging.getLogger("dummy")):
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
        if cuda.is_available():
            multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    multiprocessing.set_start_method("spawn", force=True)

    # Create the Multiprocessing Manager
    mp_context: multiprocessing.context = multiprocessing.get_context("spawn")

    mp_manager: multiprocessing.Manager = SyncManager(ctx=mp_context)
    mp_manager.start()
    # mp_manager: multiprocessing.Manager = multiprocessing.Manager()

    # Create the Dataset Queue
    df_queue = mp_manager.Queue()

    # Create the Log Queue
    log_queue = mp_manager.Queue()

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
    dp = threading.Thread(
        target=dataset_writer, args=(df_queue, args.output_file), daemon=True
    )
    dp.start()

    logger: logging.Logger

    # List the networks. Do not load them yet to save memory and CPU time.
    test_networks_list = []

    if not isinstance(args.location, list):
        args.location = [args.location]

    for loc in args.location:
        try:
            test_networks_list += list_files(
                loc,
                # max_num_vertices=args.max_num_vertices,
                filter=args.filter,
                targets=None,
                # manager=mp_manager,
            )
        except FileNotFoundError:
            pass

    if len(test_networks_list) == 0:
        logger.info(
            f"No networks found in {str(args.location)} with filter {args.filter} ."
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

        df = df_reader(args.input,
                       expected_columns=expected_columns,
                       include_removals=False,

                       **reader_kwargs,
                       )
    else:
        df = df_reader(
            args.output_file,
            include_removals=False,

            expected_columns=args.output_df_columns,
            **reader_kwargs,
        )

    # Keep only the rows with the same threshold
    df = df[df["threshold"] == args.threshold]

    # Create the pool
    with ProcessPoolExecutor(
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
    ) as executor:
        with tqdm(
                test_networks_list,
                desc="Networks",
                position=0,
        ) as tqdm_test_network_list:
            # noinspection PyTypeChecker
            for network_path in tqdm_test_network_list:
                network_name = network_path.stem

                tqdm_test_network_list.set_description(f"Networks ({network_name})")

                # Check if the network was already tested
                # Get the rows of the dataframe with the same network network_name
                # Note that the network is a categorical column
                # Avoid .loc for performance reasons
                network_df = df[df["network"] == network_name]

                logger.debug(f"Network {network_name} has {network_df.shape[0]} rows in the dataframe\n{network_df}")
                networks_provider: Union[Dict, None] = None
                network_size: int = None
                generator_args: Union[Dict, None] = None

                with tqdm(args.heuristics,
                          desc="Heuristics",
                          position=1,
                          ) as tqdm_heuristics:
                    # Iterate over the heuristics
                    for heuristic in tqdm_heuristics:
                        dismantling_method: DismantlingMethod = dismantling_methods[heuristic]

                        logger.info(
                            f"Running heuristic {dismantling_method.short_name} with threshold {args.threshold}")
                        df_filtered = network_df[network_df["heuristic"] == dismantling_method.key]

                        # TODO also check if all the requested metrics are present?
                        if len(df_filtered) != 0:
                            # Nothing to do. The network was already tested
                            continue

                        if networks_provider is None:
                            # Delay the network loading until the heuristic is actually run
                            # (to avoid loading the network if it is not needed, e.g., the heuristics have been already run)

                            logger.debug(f"Loading network: {network_name}")

                            networks_provider = init_network_provider(
                                location=network_path.parent,
                                filter=f"{network_name}",
                                logger=logger,
                            )

                            if len(networks_provider) == 0:
                                logger.error(f"Network {network_name} not found!")
                                continue

                            elif len(networks_provider) > 1:
                                logger.error(f"More than one network found for {network_name}!")
                                continue

                            network_name, network = networks_provider[0]

                            if network_name != network_name:
                                logger.error(
                                    f"Loaded network with filename {network_name} does not match the expected filename {network_name}!"
                                )
                                continue

                            network_size = network.num_vertices()

                            # Compute stop condition
                            stop_condition = np.ceil(network_size * args.threshold)

                            generator_args = {
                                "network_name": network_name,
                                "stop_condition": int(stop_condition),
                                "threshold": args.threshold,
                            }

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
                            if ("removals" not in df_dependency_filtered or # missing column
                                    df_dependency_filtered["removals"] is None or # None
                                    df_dependency_filtered["removals"] == "" or # empty string
                                    df_dependency_filtered["removals"] == "[]" or # empty list
                                    not isinstance(df_dependency_filtered["removals"], list) or # not a list
                                    len(df_dependency_filtered["removals"]) == 0 # empty list
                            ):

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
                            generator_args[dismantling_method.depends_on.key] = dependency_removals

                        logger.debug(
                            f"Dismantling {network_name} according to {display_name}. "
                            f"Aiming to LCC size {stop_condition} ({stop_condition / network_size:.3f})"
                        )
                        
                        if network_size > args.max_num_vertices:
                            continue
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
                            df_queue.put(runs_dataframe)

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

    parser.add_argument(
        "-mnv",
        "--max_num_vertices",
        type=int,
        default=np.inf,
        help="Filter for network size based on number of vertices.",
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
        args.output_file.parent.mkdir(parents=True)

    args.output_df_columns = get_df_columns()

    logger.info(f"Output file {args.output_file}")

    if "all" in args.heuristics:
        args.heuristics = list(dismantling_methods.keys())

    logger.info(f"Running the following heuristics: {', '.join(args.heuristics)}")

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
