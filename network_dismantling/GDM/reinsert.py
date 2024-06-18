#   This file is part of GDM (Graph Dismantling with Machine learning),
#   proposed in the paper "Machine learning dismantling and
#   early-warning signals of disintegration in complex systems"
#   by M. Grassia, M. De Domenico and G. Mangioni.
#
#   GDM is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   GDM is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with GDM.  If not, see <http://www.gnu.org/licenses/>.

import logging
from ast import literal_eval
from errno import ENOSPC
from operator import itemgetter
from os import remove, close
from pathlib import Path
from subprocess import check_output
from tempfile import NamedTemporaryFile
from time import time

import numpy as np
import pandas as pd
from graph_tool import Graph
from tqdm import tqdm

from network_dismantling.GDM.dataset_providers import list_files
from network_dismantling.common.dataset_providers import init_network_provider
from network_dismantling.common.df_helpers import df_reader
from network_dismantling.common.external_dismantlers.lcc_threshold_dismantler import lcc_threshold_dismantler
from network_dismantling.common.helpers import extend_filename
from network_dismantling.common.multiprocessing import TqdmLoggingHandler

folder = "network_dismantling/GDM/reinsertion/"
cd_cmd = "cd {} && ".format(folder)
reinsertion_strategy = 2
reinsertion_executable = "reinsertion"

# Define run columns to match the runs
run_columns = [
    # "removals",
    "slcc_peak_at",
    "lcc_size_at_peak",
    "slcc_size_at_peak",
    "r_auc",
    # TODO
    "seed",
    "average_dmg",
    "rem_num",
    "idx",
    "file",
]

cached_networks = {}


def get_predictions(
        network, removals, stop_condition, logger=logging.getLogger("dummy"), **kwargs
):
    start_time = time()

    predictions = reinsert(
        network=network,
        removals=removals,
        stop_condition=stop_condition,
        logger=logger,
    )

    time_spent = time() - start_time

    return predictions, time_spent


def reinsert(
        network,
        removals,
        stop_condition,
        logger=logging.getLogger("dummy"),
):
    network_path = get_network_file(network)

    # broken_fd, broken_path = tempfile.mkstemp()
    # output_fd, output_path = tempfile.mkstemp()
    #
    # tmp_file_handles = [broken_fd, output_fd]
    # tmp_file_paths = [broken_path, output_path]

    nodes = []

    with (
        NamedTemporaryFile("w+") as broken_fd,
        NamedTemporaryFile("w+") as output_fd
    ):

        broken_path = broken_fd.name
        output_path = output_fd.name

        cmds = [
            # 'make clean && make',
            "make",
            f"./{reinsertion_executable} "
            f"--NetworkFile {network_path} "
            f'--IDFile "{broken_path}" '
            f'--OutFile "{output_path}" '
            f"--TargetSize {stop_condition} "
            f"--SortStrategy {reinsertion_strategy} ",
        ]

        # try:
        # with open(broken_fd, "w+") as tmp:
        for removal in removals:
            broken_fd.write(f"{removal}\n")

        for cmd in cmds:
            try:
                check_output(cd_cmd + cmd, shell=True, text=True)  # , stderr=STDOUT))
            except Exception as e:
                raise RuntimeError("ERROR! {}".format(e))

        with open(output_path, "r+") as tmp:
            num_lines = sum(1 for _ in tmp.readlines())

        # output_fd.rewind()

        for line in output_fd.readlines():
            num_lines -= 1
            node = int(line.strip())

            nodes.append(node)

        assert num_lines == 0

    # finally:
    #     for fd, path in zip(tmp_file_handles, tmp_file_paths):
    #         try:
    #             close(fd)
    #
    #         except:
    #             pass
    #
    #         try:
    #             remove(path)
    #
    #         except:
    #             pass

    output = np.zeros(network.num_vertices())

    filtered_removals = []
    removed_removals = []
    for x in removals:
        if x in nodes:
            filtered_removals.append(x)
        else:
            removed_removals.append(x)

    nodes = filtered_removals + removed_removals

    # TODO improve this loop
    for n, p in zip(nodes, list(reversed(range(1, len(nodes) + 1)))):
        output[n] = p

    return output


def get_network_file(network: Graph) -> str:
    from tempfile import mkstemp

    network_file_path = network.graph_properties.get("filepath", None)
    cached_network_path = cached_networks.get(network_file_path, None)
    if (cached_network_path is not None) and (Path(cached_network_path).exists()):
        network_path = cached_network_path

    else:
        try:
            try:
                network_fd, network_path = mkstemp()
            except OSError as e:
                # If there is no space left on the device
                #  remove the cached networks and try again
                if e.errno == ENOSPC:
                    cleanup_cache()

                    network_fd, network_path = mkstemp()

            static_id = network.vertex_properties["static_id"]

            with open(network_fd, "w+") as tmp:
                for edge in network.edges():
                    tmp.write(f"{static_id[edge.source()]} {static_id[edge.target()]}\n")

            if network_file_path is not None:
                cached_networks[network_file_path] = network_path

        finally:
            try:
                close(network_fd)
            except:
                pass

    return network_path


def cleanup_cache():
    for path in cached_networks.values():
        try:
            remove(path)
        except:
            pass

    cached_networks.clear()


def main(
        args,
        df=None,
        test_networks=None,
        predictor=get_predictions,
        dismantler=lcc_threshold_dismantler,
        threshold: float = None,
        logger=logging.getLogger("dummy"),
):
    from scipy.integrate import simpson

    if df is None:
        # Load the runs dataframe...
        df = df_reader(args.file,
                       include_removals=True,
                       raise_on_missing_file=True,
                       )

        if args.query is not None:
            # ... and query it
            df.query(args.query, inplace=True)

    df_columns = df.columns

    if args.output_file.exists():
        output_df = pd.read_csv(args.output_file)
    else:
        output_df = pd.DataFrame(columns=df.columns)

    if test_networks is None:
        # TODO defer the loading of the networks
        # Get the list of networks in the folder
        test_networks_list = list_files(
            args.location_test,
            filter=args.test_filter,
        )
        test_networks_list = np.intersect1d(
            [file.stem for file in test_networks_list],
            df["network"].unique(),
        )

        # TODO create the mapping preserving the file location
        # that was lost in the previous step

        # # Load the networks
        # test_networks = dict(
        #     storage_provider(args.location_test,
        #                      filter=args.test_filter,
        #                      )
        # )

    # Filter the networks in the folder
    df = df.loc[(df["network"].isin(test_networks.keys()))]

    if df.shape[0] == 0:
        logger.warning("No networks to reinsert!")

    if args.sort_column == "average_dmg":
        df["average_dmg"] = (1 - df["lcc_size_at_peak"]) / df["slcc_peak_at"]

    # Sort dataframe
    df.sort_values(
        by=[args.sort_column],
        ascending=(not args.sort_descending),
        inplace=True,
    )

    all_runs = []
    groups = df.groupby("network")
    for network_name, network_df in groups:
        network_df = network_df.head(args.reinsert_first)

        with tqdm(
                network_df.iterrows(),
                ascii=False,
                desc="Reinserting",
                leave=False,
        ) as runs_iterable:
            runs_iterable.set_description(network_name)

            for _, run in runs_iterable:
                network = test_networks[network_name]

                # Get the removals
                removals = run.pop("removals")
                removals = literal_eval(removals)

                # Remove the columns that are not needed
                run.drop(run_columns,
                         inplace=True,
                         errors="ignore",
                         )
                run = run.to_dict()

                reinserted_run_df = output_df.loc[
                    (output_df[list(run.keys())] == list(run.values())).all(
                        axis="columns"
                    ),
                    ["network", "seed"],
                ]

                if len(reinserted_run_df) != 0:
                    # Nothing to do. The network was already tested
                    continue

                if threshold is None:
                    threshold = run.get("threshold",
                                        removals[-1][3],
                                        )

                stop_condition = int(np.ceil(threshold * network.num_vertices()))
                generator_args = {
                    "removals": list(map(itemgetter(1), removals)),
                    "stop_condition": stop_condition,
                    "network_name": network_name,
                    "logger": logger,
                }

                removals, _, _ = dismantler(
                    network=network.copy(),
                    predictor=predictor,
                    generator_args=generator_args,
                    stop_condition=stop_condition,
                    dismantler=dismantler,
                    logger=logger,
                )

                peak_slcc = max(removals, key=itemgetter(4))

                _run = {
                    "network": network_name,
                    "removals": removals,
                    "slcc_peak_at": peak_slcc[0],
                    "lcc_size_at_peak": peak_slcc[3],
                    "slcc_size_at_peak": peak_slcc[4],
                    "r_auc": simpson(list(r[3] for r in removals), dx=1),
                    "rem_num": len(removals),
                    "threshold": threshold,
                }

                for key, value in _run.items():
                    run[key] = value

                # Check if something is wrong with the removals
                if removals[-1][2] == 0:
                    for removal in removals:
                        logger.info(
                            "\t{}-th removal: node {} ({}). LCC size: {}, SLCC size: {}".format(
                                *removal
                            )
                        )

                    raise RuntimeError

                all_runs.append(run)

                run_df = pd.DataFrame(data=[run], columns=network_df.columns)

                if args.output_file is not None:
                    kwargs = {
                        "path_or_buf": Path(args.output_file),
                        "index": False,
                        # header='column_names',
                        "columns": df_columns,
                    }

                    # If dataframe exists append without writing the header
                    if kwargs["path_or_buf"].exists():
                        kwargs["mode"] = "a"
                        kwargs["header"] = False

                    run_df.to_csv(**kwargs)

        cleanup_cache()

    return all_runs


def parse_parameters(parse_string=None, logger=logging.getLogger("dummy")):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        type=Path,
        default=None,
        required=True,
        help="Output DataFrame file location",
    )
    parser.add_argument(
        "-lt",
        "--location_test",
        type=Path,
        default=None,
        required=True,
        help="Location of the dataset (directory)",
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
        "-q",
        "--query",
        type=str,
        default=None,
        required=False,
        help="Query the dataframe",
    )
    parser.add_argument(
        "-rf",
        "--reinsert_first",
        type=int,
        default=15,
        required=False,
        help="Show first N dismantligs",
    )
    parser.add_argument(
        "-s",
        "--sort_column",
        type=str,
        default="r_auc",
        required=False,
        help="Column used to sort the entries",
    )
    parser.add_argument(
        "-sa",
        "--sort_descending",
        default=False,
        required=False,
        action="store_true",
        help="Descending sorting",
    )
    args, cmdline_args = parser.parse_known_args(parse_string)

    if not args.location_test.is_absolute():
        args.location_test = args.location_test.resolve()

    logger.debug(f"Reinsertion input file {args.file}")
    args.output_file = extend_filename(
        args.file,
        filename_extension="_reinserted",
    )

    logger.debug(f"Reinsertion output file {args.output_file}")

    return args


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(TqdmLoggingHandler())

    args = parse_parameters()

    networks_provider = init_network_provider(
        location=args.location_test,
        filter=args.test_filter,
        logger=logger,
    )
    test_networks = dict(networks_provider)

    main(
        args,
        test_networks=test_networks,
        logger=logger,
    )
