import argparse
import logging
import multiprocessing
import threading
from datetime import timedelta
from operator import itemgetter
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
from scipy.integrate import simps
from tqdm.auto import tqdm

from network_dismantling.common.dataset_providers import list_files, init_network_provider
from network_dismantling.common.df_helpers import df_reader
from network_dismantling.common.external_dismantlers.lcc_threshold_dismantler import \
    threshold_dismantler as external_threshold_dismantler
from network_dismantling.common.helpers import extend_filename
from network_dismantling.common.multiprocessing import TqdmLoggingHandler, dataset_writer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(TqdmLoggingHandler())


def static_generator(network, sorting_function, *args, logger=logging.getLogger('dummy'), **kwargs):
    values, _ = get_predictions(network, sorting_function, logger)
    pairs = list(zip([network.vertex_properties["static_id"][v] for v in network.vertices()], values))

    # Get the highest predicted value
    sorted_predictions = sorted(pairs, key=itemgetter(1), reverse=True)

    for node, value in sorted_predictions:
        yield node, value


def get_predictions(network, sorting_function, *args, logger=logging.getLogger('dummy'), **kwargs):
    logger.info("Sorting the predictions...")
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

    time_queue = mp_manager.Queue()
    tp = threading.Thread(target=dataset_writer, args=(time_queue, args.time_output_file), daemon=True)
    tp.start()

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
        logging.info(f"No networks found in {str(args.location)} with filter {args.filter} .")

    if args.input is not None:
        df = df_reader(args.input, include_removals=False)
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
                                                  # max_num_vertices=args.max_num_vertices,
                                                  filter=f"{name}",
                                                  # manager=mp_manager,
                                                  )

        if len(networks_provider) == 0:
            tqdm.write(f"Network {name} not found!")
            continue

        elif len(networks_provider) > 1:
            tqdm.write(f"More than one network found for {name}!")
            continue

        network_name, network = networks_provider[0]

        assert network_name == name

        network_df = df.loc[(df["network"] == name)]

        network_size = network.num_vertices()

        # Compute stop condition
        stop_condition = np.ceil(network_size * args.threshold)

        generator_args = {
            "logger": logger,
            "network_name": name,
            "stop_condition": int(stop_condition),
            "threshold": args.threshold,
        }

        # noinspection PyTypeChecker
        for heuristic in tqdm(
                args.heuristics,
                position=1,
                desc="Heuristics",
        ):
            heuristic_info = dismantling_methods[heuristic]
            display_name = heuristic_info.name

            generator_args["sorting_function"] = heuristic_info.function

            logger.info(f"\n"
                        f"==================================\n"
                        f"Running {display_name} heuristic. Cite as:\n"
                        f"{heuristic_info.citation}\n"
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

            logger.info("Dismantling {} according to {}. Aiming to LCC size {} ({})".format(name,
                                                                                            display_name,
                                                                                            stop_condition,
                                                                                            stop_condition / network_size)
                        )
            try:
                # external_generator
                removals, prediction_time, dismantle_time = external_threshold_dismantler(network.copy(),
                                                                                          get_predictions,
                                                                                          generator_args,
                                                                                          stop_condition,
                                                                                          )

                peak_slcc = max(removals, key=itemgetter(4))

                rem_num = len(removals)

                if rem_num > 0:
                    assert removals[0][0] > -1, "First removal is just the LCC size!"

                run = {
                    "network": name,
                    "removals": removals,
                    "slcc_peak_at": peak_slcc[0],
                    "lcc_size_at_peak": peak_slcc[3],
                    "slcc_size_at_peak": peak_slcc[4],
                    "heuristic": heuristic,
                    "static": None,
                    "r_auc": simps(list(r[3] for r in removals), dx=1),
                    "rem_num": rem_num,
                }

                if args.verbose == 2:
                    for removal in run["removals"]:
                        logger.info("\t{}-th removal: node {} ({}). LCC size: {}, SLCC size: {}".format(removal[0],
                                                                                                        removal[1],
                                                                                                        removal[2],
                                                                                                        removal[3],
                                                                                                        removal[4]
                                                                                                        )
                                    )
                if removals[-1][2] == 0:
                    raise RuntimeError("ERROR: removed more nodes than predicted!")

            except Exception as e:
                # print("ERROR: {}".format(e))
                # print_tb(e.__traceback__)
                logger.exception(e)

                continue
                # raise e

            runs_dataframe = pd.DataFrame(data=[run], columns=args.output_df_columns)

            df_queue.put(runs_dataframe)

            time_run = {
                "network": name,
                "heuristic": heuristic,
                "static": None,
                "prediction_time": prediction_time,
                "dismantle_time": dismantle_time
            }
            time_dataframe = pd.DataFrame(data=[time_run],
                                          columns=["network", "heuristic", "static", "prediction_time",
                                                   "dismantle_time"]
                                          )

            time_queue.put(time_dataframe)

    df_queue.put(None)
    time_queue.put(None)

    dp.join()
    tp.join()


def get_df_columns():
    return ["network", "heuristic", "slcc_peak_at", "lcc_size_at_peak",
            "slcc_size_at_peak", "removals", "static", "r_auc"]


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

    args.time_output_file = extend_filename(args.output_file,
                                            "_reinserted",
                                            postfixes=["time"],
                                            )

    logger.info(f"Output file {args.output_file}")
    logger.info(f"Time output file {args.time_output_file}")

    if "all" in args.heuristics:
        args.heuristics = list(dismantling_methods.keys())

    logger.info(f"Running the following heuristics: {', '.join(args.heuristics)}")

    main(args)
