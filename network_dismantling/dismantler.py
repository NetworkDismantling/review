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

import argparse
import logging
import threading
from datetime import timedelta
from pathlib import Path
from time import time
from typing import Callable

import numpy as np
import pandas as pd
from graph_tool import Graph
from torch import multiprocessing, cuda
from tqdm.auto import tqdm

from network_dismantling.common.dataset_providers import list_files, init_network_provider
from network_dismantling.common.df_helpers import df_reader
from network_dismantling.common.multiprocessing import TqdmLoggingHandler, dataset_writer

import os
from graph_tool.all import openmp_set_num_threads

# Remove the OpenMP threads. Use data parallelism instead
# os.environ["OMP_NUM_THREADS"] = "1"
openmp_set_num_threads(1)


def get_predictions(network: Graph, sorting_function: Callable, logger=logging.getLogger('dummy'), **kwargs):
    logger.debug(f"Sorting the predictions...")
    start_time = time()

    values = sorting_function(network, **kwargs)

    time_spent = time() - start_time
    logger.debug("Heuristics returned. Took {}".format(timedelta(seconds=(time_spent))))

    return values, time_spent


def main(args):
    try:
        from deadpool import Deadpool as ProcessPoolExecutor

        pool_kwargs = {
            "max_tasks_per_child": 10
        }

    except ImportError:
        logger.warning("Deadpool not found. Using ProcessPoolExecutor instead.")
        from concurrent.futures import ProcessPoolExecutor

        # if __version__ == "0.5.0":
        pool_kwargs = {
        }

    try:
        if cuda.is_available():
            multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # Create the Multiprocessing Manager
    mp_manager: multiprocessing.Manager = multiprocessing.Manager()
    # Create the Dataset Queue
    df_queue = mp_manager.Queue()

    # Create and start the Dataset Writer Thread
    dp = threading.Thread(target=dataset_writer, args=(df_queue, args.output_file), daemon=True)
    dp.start()

    test_networks_list = []

    if not isinstance(args.location, list):
        args.location = [args.location]

    for loc in args.location:
        try:
            test_networks_list += list_files(loc,
                                             # max_num_vertices=args.max_num_vertices,
                                             filter=args.filter,
                                             targets=None,
                                             # manager=mp_manager,
                                             )
        except FileNotFoundError:
            pass

    if len(test_networks_list) == 0:
        logger.info(f"No networks found in {str(args.location)} with filter {args.filter} .")

    if args.input is not None:
        df = df_reader(args.input, include_removals=False)

        expected_columns = args.output_df_columns.copy()
        expected_columns.remove("removals")

        if (df.columns != expected_columns).all():
            raise ValueError(
                f"Input file columns {df.columns} do not match the expected columns {args.output_df_columns}.")

    elif args.output_file.exists():
        df = df_reader(args.output_file, include_removals=False)
    else:
        df = pd.DataFrame(columns=args.output_df_columns)

    # with multiprocessing.Pool(processes=args.jobs,
    #                           initializer=tqdm.set_lock,
    #                           initargs=(multiprocessing.Lock(),)
    #                           ) as p:

    # Create the pool
    # mp_context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
            max_workers=args.jobs,
            mp_context=multiprocessing.get_context("spawn"),
            # initializer=tqdm.set_lock,
            # initargs=(multiprocessing.Lock(),),
            **pool_kwargs,
    ) as executor:

        with tqdm(test_networks_list,
                  desc="Networks",
                  position=0,
                  ) as tqdm_test_network_list:

            # noinspection PyTypeChecker
            for network_path in tqdm_test_network_list:
                name = network_path.stem

                # logger.info(f"Loading network: {name}")
                tqdm_test_network_list.set_description(f"Networks ({name})")

                networks_provider = init_network_provider(location=network_path.parent,
                                                          filter=f"{name}",
                                                          logger=logger,
                                                          )

                if len(networks_provider) == 0:
                    logger.info(f"Network {name} not found!")
                    continue

                elif len(networks_provider) > 1:
                    logger.info(f"More than one network found for {name}!")
                    continue

                network_name, network = networks_provider[0]

                assert network_name == name

                network_df = df.loc[(df["network"] == name)]

                network_size = network.num_vertices()

                # Compute stop condition
                stop_condition = np.ceil(network_size * args.threshold)

                generator_args = {
                    "network_name": name,
                    "stop_condition": int(stop_condition),
                    "threshold": args.threshold,
                    "logger": logger,
                }

                # noinspection PyTypeChecker
                for heuristic in tqdm(
                        args.heuristics,
                        position=1,
                        desc="Heuristics",
                ):
                    dismantling_method = dismantling_methods[heuristic]
                    display_name = dismantling_method.name
                    display_name_short = dismantling_method.short_name

                    # generator_args["sorting_function"] = dismantling_method.function

                    logger.info(f"\n"
                                f"==================================\n"
                                f"Running {display_name} ({display_name_short}) heuristic. Cite as:\n"
                                f"{dismantling_method.citation.strip()}\n"
                                f"==================================\n"
                                )

                    filter = {
                        "heuristic": heuristic,
                    }

                    df_filtered = network_df.loc[
                        (network_df[list(filter.keys())] == list(filter.values())).all(axis='columns'),
                        ["network"]
                    ]

                    filtered_network_df = df_filtered.loc[(df_filtered["network"] == name)]

                    if len(filtered_network_df) != 0:
                        # Nothing to do. The network was already tested
                        continue

                    # logger.info(f"Dismantling {name} according to {display_name}. "
                    #             f"Aiming to LCC size {stop_condition} ({stop_condition / network_size:.3f})"
                    #             )

                    try:

                        # TODO REMOVE THE COPY OF THE NETWORK, and move where its actually needed
                        runs = dismantling_method(network=network.copy(),
                                                  stop_condition=stop_condition,
                                                  generator_args=generator_args,

                                                  executor=executor,
                                                  # pool=p,
                                                  pool_size=args.jobs,
                                                  mp_manager=mp_manager,

                                                  logger=logger,
                                                  )

                        runs["network"] = name

                        if not isinstance(runs, pd.DataFrame):
                            runs_dataframe = pd.DataFrame(data=[runs],
                                                          columns=args.output_df_columns,
                                                          )
                        else:  # isinstance(runs, pd.DataFrame):
                            runs_dataframe = runs[args.output_df_columns]

                        df_queue.put(runs_dataframe)

                    except Exception as e:
                        logger.exception(e, exc_info=True)

                        continue

        # Close the pool
        executor.shutdown(wait=True,
                          cancel_futures=False,
                          )
        # p.close()
        # p.join()

    df_queue.put(None)

    dp.join()


def get_df_columns():
    return ["network", "heuristic", "slcc_peak_at", "lcc_size_at_peak",
            "slcc_size_at_peak", "removals", "static", "r_auc", "rem_num",
            "prediction_time", "dismantle_time",
            ]


if __name__ == "__main__":
    from network_dismantling import dismantling_methods

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(TqdmLoggingHandler())

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=None,
        nargs="*",
        required=False,
        help="Heuristics input file. Will be used to filter the algorithms already tested.",
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
        default="all",
        nargs="+",
        help="Dismantling heuristics to run. Default: all. See the repository README for more information.",
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

    # parser.add_argument(
    #     "-mnv",
    #     "--max_num_vertices",
    #     type=int,
    #     default=float("inf"),
    #     help="Filter the networks given the maximum number of vertices.",
    # )

    args, cmdline_args = parser.parse_known_args()

    logger.setLevel(logging.getLevelName(args.verbose))
    if not args.output.is_absolute():
        # args.output_file = base_dataframes_path / args.output
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

    main(args)
