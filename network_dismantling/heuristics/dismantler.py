import argparse
import logging
from datetime import timedelta
from operator import itemgetter
from pathlib import Path
from time import time
from types import GeneratorType

import numpy as np
import pandas as pd
from scipy.integrate import simpson
from tqdm.auto import tqdm

from network_dismantling.common.dataset_providers import list_files, init_network_provider
from network_dismantling.common.df_helpers import df_reader
from network_dismantling.common.dismantlers import threshold_dismantler
from network_dismantling.common.external_dismantlers.lcc_threshold_dismantler import \
    threshold_dismantler as external_threshold_dismantler
from network_dismantling.heuristics import sorters


def incremental_dynamic_generator(network, sorting_function, *args, logger=logging.getLogger('dummy'), **kwargs):
    generator = sorting_function(network)  # , generator=True)

    last_removal = None
    for _ in range(network.num_vertices()):
        values = generator.send(last_removal)

        # Get largest predicted value
        index = np.argmax(values)

        node, value = network.vertex_properties["static_id"][index], values[index]

        last_removal = index

        # print("REMOVING {}".format(node))

        yield node, value


def dynamic_generator(network, sorting_function, *args, logger=logging.getLogger('dummy'), **kwargs):
    for _ in range(network.num_vertices()):
        values = sorting_function(network)

        # Get largest predicted value
        index = np.argmax(values)
        node, value = network.vertex_properties["static_id"][index], values[index]

        del values

        yield node, value


def static_generator(network, sorting_function, *args, logger=logging.getLogger('dummy'), **kwargs):
    values, _ = get_predictions(network, sorting_function, logger)
    pairs = list(zip([network.vertex_properties["static_id"][v] for v in network.vertices()], values))

    # Get the highest predicted value
    sorted_predictions = sorted(pairs, key=itemgetter(1), reverse=True)

    for node, value in sorted_predictions:
        yield node, value


def get_predictions(network, sorting_function, *args, logger=logging.getLogger('dummy'), **kwargs):
    sorting_function_name = sorting_function.__name__
    if sorting_function_name in network.vertex_properties.keys():
        logger.info("{} values already computed!".format(sorting_function_name))

        values = network.vertex_properties[sorting_function_name].get_array()

        time_spent = None
    else:
        logger("{} computing values!".format(sorting_function_name))

        start_time = time()

        values = sorting_function(network)

        time_spent = time() - start_time

        if isinstance(values, GeneratorType):
            values = next(values)

        logger.info("Heuristics returned. Took {}".format(timedelta(seconds=(time_spent))))

    # pairs = list(zip([network.vertex_properties["static_id"][v] for v in network.vertices()], values))
    # return pairs, time_spent
    return values, time_spent


def main(args):
    print = tqdm.write

    # TODO
    static_modes = []
    if args.static_dismantling:
        static_modes.append(True)
    if args.dynamic_dismantling:
        static_modes.append(False)

    # TODO
    if len(static_modes) == 0:
        exit("No generators chosen!")

    test_networks_list = []

    if not isinstance(args.location, list):
        args.location = [args.location]

    for loc in args.location:
        test_networks_list += list_files(loc,
                                         max_num_vertices=args.max_num_vertices,
                                         filter=args.test_filter,
                                         targets=None,
                                         # manager=mp_manager,
                                         )

    # networks_provider = init_network_provider(args.location,
    #                                           max_num_vertices=args.max_num_vertices,
    #                                           filter=args.test_filter,
    #                                           )

    if args.input is not None:
        # df = pd.read_csv(args.output_file)
        df = df_reader(args.input, include_removals=False)
    elif args.output_file.exists():
        df = df_reader(args.output_file, include_removals=False)
    else:
        df = pd.DataFrame(columns=args.output_df_columns)

    for name in tqdm(test_networks_list,
                     desc="Networks",
                     position=0,
                     ):
        tqdm.write(f"Loading network: {name}")

        networks_provider = init_network_provider(args.location,
                                                  max_num_vertices=args.max_num_vertices,
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

        for heuristic in tqdm(
                args.heuristics,
                position=1,
                desc="Heuristics",
        ):
            # for heuristic in progressbar(
            #         # list(product_dict(_callback=sorter.parameters_combination_validator, **parameters_to_try)),
            #         args.heuristics,
            #         redirect_stdout=True
            # ):
            display_name = ' '.join(heuristic.split("_")).upper()

            # TODO improve me
            filter = {
                "heuristic": heuristic
            }
            # add_run_parameters(params, filter)
            # sorter.add_run_parameters(filter)

            df_filtered = network_df.loc[
                (network_df[list(filter.keys())] == list(filter.values())).all(axis='columns'),
                ["network", "static"]  # , "seed" ]
            ]

            generator_args = {
                "sorting_function": sorters.__all_dict__[heuristic],
                "logger": print,
                "network_name": name,
            }
            # for name, network in tqdm(networks_provider,
            #                           desc="Networks",
            #                           position=1,
            #                           ascii=True,
            #                           ):

            runs = []
            for mode in static_modes:
                filtered_network_df = df_filtered.loc[(df_filtered["static"] == mode)]

                if len(filtered_network_df) != 0:
                    # Nothing to do. Network was already tested
                    continue

                if mode:
                    # External (fast) C++ dismantler.
                    # WARNING: It won't work if you try to remove nodes that only have self loops in the original network.
                    generator = get_predictions
                    dismantler = external_threshold_dismantler

                    # generator = static_generator
                    # dismantler = threshold_dismantler
                else:
                    # TODO REMOVE THIS. IT'S ONLY FOR DEBUGGING
                    if "collective_influence" in heuristic:
                        generator = incremental_dynamic_generator
                    else:
                        generator = dynamic_generator

                    # dismantler = external_iterative_threshold_dismantler
                    dismantler = threshold_dismantler

                print("Dismantling {} according to {}. Aiming to LCC size {} ({})".format(name,
                                                                                          (
                                                                                              "STATIC" if mode is True else "DYNAMIC") + " " + display_name,
                                                                                          stop_condition,
                                                                                          stop_condition / network_size))
                removals, prediction_time, dismantle_time, _ = dismantler(network=network.copy(),
                                                                       predictor=generator,
                                                                       generator_args=generator_args,
                                                                       stop_condition=stop_condition,
                                                                       )

                peak_slcc = max(removals, key=itemgetter(4))

                run = {
                    "network": name,
                    "removals": removals,
                    "slcc_peak_at": peak_slcc[0],
                    "lcc_size_at_peak": peak_slcc[3],
                    "slcc_size_at_peak": peak_slcc[4],
                    "heuristic": heuristic,
                    "static": mode,
                    "r_auc": simpson(list(r[3] for r in removals), dx=1)
                }

                runs.append(run)

                if args.verbose == 2:
                    for removal in run["removals"]:
                        print("\t{}-th removal: node {} ({}). LCC size: {}, SLCC size: {}".format(removal[0],
                                                                                                  removal[1],
                                                                                                  removal[2],
                                                                                                  removal[3],
                                                                                                  removal[4]
                                                                                                  ))

            runs_dataframe = pd.DataFrame(data=runs, columns=args.output_df_columns)

            if args.output_file is not None:

                kwargs = {
                    "path_or_buf": Path(args.output_file),
                    "index": False,
                    # header='column_names',
                    "columns": args.output_df_columns
                }

                # If dataframe exists append without writing the header
                if kwargs["path_or_buf"].exists():
                    kwargs["mode"] = "a"
                    kwargs["header"] = False

                runs_dataframe.to_csv(**kwargs)


def get_df_columns():
    return ["network", "heuristic", "slcc_peak_at", "lcc_size_at_peak",
            "slcc_size_at_peak", "removals", "static", "r_auc"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=None,
        nargs="*",
        required=False,
        help="Heuristics input file. Will be used to filter the runs that have already been performed.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("heuristics.csv"),
        required=False,
        help="Heuristics output file. Will be used to store the results of the runs.",
    )

    parser.add_argument(
        "-l",
        "--location",
        type=Path,
        default=None,
        nargs="*",
        help="Location of the dataset (directory)",
    )

    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.1,
        help="Dismantling stop threshold",
    )

    parser.add_argument(
        "-H",
        "--heuristics",
        type=str,
        choices=sorted(sorters.__all_dict__.keys()),
        default=sorted(sorters.__all_dict__.keys()),
        nargs="+",
        help="Dismantling stop threshold",
    )

    parser.add_argument(
        "-SD",
        "--static_dismantling",
        default=False,
        action="store_true",
        help="[Test only] Static removal of nodes",
    )

    parser.add_argument(
        "-DD",
        "--dynamic_dismantling",
        default=False,
        action="store_true",
        help="[Test only] Static removal of nodes",
    )

    parser.add_argument(
        "-Ft",
        "--test_filter",
        type=str,
        default="*",
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

    parser.add_argument(
        "-mnv",
        "--max_num_vertices",
        type=int,
        default=float("inf"),
        help="Filter the networks given the maximum number of vertices.",
    )

    args, cmdline_args = parser.parse_known_args()

    if not args.output.is_absolute():
        # args.output_file = base_dataframes_path / args.output
        args.output = args.output.resolve()

    elif args.output.exists():
        if not args.output.is_file():
            raise FileNotFoundError("Output file {} is not a file.".format(args.output))

    args.output_file = args.output

    if args.input is None:
        args.input = args.output_file
    else:

        if not isinstance(args.input, list):
            args.input = [args.input]

        args.input = [i.resolve() for i in args.input]

        if args.output not in args.input:
            args.input.append(args.output)

    # else:
    #
    #     if not args.input.is_absolute():
    #         args.input = args.input.resolve()
    #
    #     if not args.input.exists():
    #         raise FileNotFoundError("Input file {} does not exist.".format(args.input))
    #     elif not args.input.is_file():
    #         raise FileNotFoundError("Input file {} is not a file.".format(args.input))

    if not args.output_file.parent.exists():
        args.output_file.parent.mkdir(parents=True)

    args.output_df_columns = get_df_columns()

    print("Output file {}".format(args.output_file))

    main(args)
