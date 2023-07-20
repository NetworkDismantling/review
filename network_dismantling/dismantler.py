import argparse
import logging
import multiprocessing
import threading
from datetime import timedelta
from pathlib import Path
from time import time
from typing import Callable

import numpy as np
import pandas as pd
from graph_tool import Graph
from network_dismantling.common.dataset_providers import list_files, init_network_provider
from network_dismantling.common.df_helpers import df_reader
from network_dismantling.common.multiprocessing import TqdmLoggingHandler, dataset_writer
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(TqdmLoggingHandler())


def get_predictions(network: Graph, sorting_function: Callable, logger=logging.getLogger('dummy'), **kwargs):
    logger.info(f"Sorting the predictions...")
    start_time = time()

    values = sorting_function(network, **kwargs)

    time_spent = time() - start_time
    logger.info("Heuristics returned. Took {}".format(timedelta(seconds=(time_spent))))

    return values, time_spent


def main(args):
    # Create the Multiprocessing Manager
    mp_manager = multiprocessing.Manager()

    df_queue = mp_manager.Queue()
    # Create and start the Dataset Writer Thread
    dp = threading.Thread(target=dataset_writer, args=(df_queue, args.output_file), daemon=True)
    dp.start()

    # time_queue = mp_manager.Queue()
    # tp = threading.Thread(target=dataset_writer, args=(time_queue, args.time_output_file), daemon=True)
    # tp.start()

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

    # noinspection PyTypeChecker
    for name in tqdm(test_networks_list,
                     desc="Networks",
                     position=0,
                     ):
        logger.info(f"Loading network: {name}")

        networks_provider = init_network_provider(args.location,
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

            # generator_args["sorting_function"] = dismantling_method.function

            logger.info(f"\n"
                        f"==================================\n"
                        f"Running {display_name} heuristic. Cite as:\n"
                        f"{dismantling_method.citation}\n"
                        f"==================================\n"
                        )

            filter = {
                "heuristic": heuristic
            }

            df_filtered = network_df.loc[
                (network_df[list(filter.keys())] == list(filter.values())).all(axis='columns'),
                ["network"]
            ]

            filtered_network_df = df_filtered.loc[(df_filtered["network"] == name)]

            if len(filtered_network_df) != 0:
                # Nothing to do. Network was already tested
                continue

            logger.info(f"Dismantling {name} according to {display_name}. "
                        f"Aiming to LCC size {stop_condition} ({stop_condition / network_size:.3f})"
                        )

            try:
                # runs, time_runs = dismantler_wrapper(network,
                # runs = dismantler_wrapper(network,
                #                           # name,
                #                           # heuristic,
                #                           stop_condition,
                #                           generator_args,
                #                           )

                # TODO REMOVE THE COPY OF THE NETWORK, and move where its actually needed
                runs = dismantling_method(network=network.copy(),
                                          stop_condition=stop_condition,
                                          generator_args=generator_args,
                                          logger=logger,
                                          )

                runs["network"] = name

                if not isinstance(runs, pd.DataFrame):
                    runs_dataframe = pd.DataFrame(data=[runs],
                                        columns=args.output_df_columns,
                                        )
                else: # isinstance(runs, pd.DataFrame):
                    runs_dataframe = runs[args.output_df_columns]
                # time_dataframe = pd.DataFrame(data=[time_runs], columns=args.output_time_df_columns)

                df_queue.put(runs_dataframe)
                # time_queue.put(time_dataframe)

            except Exception as e:
                logger.exception(e, exc_info=True)

                continue

    df_queue.put(None)
    # time_queue.put(None)

    dp.join()
    # tp.join()


def get_df_columns():
    return ["network", "heuristic", "slcc_peak_at", "lcc_size_at_peak",
            "slcc_size_at_peak", "removals", "static", "r_auc", "rem_num",
            "prediction_time", "dismantle_time",
            ]


# def get_time_df_columns():
#     return ["network", "heuristic", "static", "prediction_time", "dismantle_time"]


if __name__ == "__main__":
    from network_dismantling import dismantling_methods

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
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="Verbosity level",
    )

    # parser.add_argument(
    #     "-mnv",
    #     "--max_num_vertices",
    #     type=int,
    #     default=float("inf"),
    #     help="Filter the networks given the maximum number of vertices.",
    # )

    args, cmdline_args = parser.parse_known_args()

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
    # args.output_time_df_columns = get_time_df_columns()

    # args.time_output_file = extend_filename(args.output_file,
    #                                         "_reinserted",
    #                                         postfixes=["time"],
    #                                         )

    logger.info(f"Output file {args.output_file}")
    # logger.info(f"Time output file {args.time_output_file}")

    if "all" in args.heuristics:
        args.heuristics = list(dismantling_methods.keys())

    logger.info(f"Running the following heuristics: {', '.join(args.heuristics)}")

    main(args)
