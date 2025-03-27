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
from subprocess import run, CalledProcessError
from tempfile import NamedTemporaryFile
from time import time
from typing import Dict

import numpy as np
import pandas as pd
from graph_tool import Graph
from network_dismantling.common.dataset_providers import init_network_provider
from network_dismantling.common.df_helpers import df_reader
from network_dismantling.common.external_dismantlers.lcc_threshold_dismantler import lcc_threshold_dismantler, \
    threshold_dismantler
from network_dismantling.common.helpers import extend_filename
from network_dismantling.common.logging.pipe import LogPipe
from network_dismantling.common.multiprocessing import TqdmLoggingHandler
from scipy.integrate import simpson
from tqdm import tqdm

folder = "network_dismantling/multiscale_entanglement/reinsertion/"
cd_cmd = r"cd {} && ".format(folder)
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
    "average_dmg",
    "rem_num",
    "idx",
    "file",
]

lcc_threshold_dismantler
threshold_dismantler

cached_networks: Dict[Path, str] = {}


def get_predictions(
        network,
        removals,
        stop_condition,
        logger: logging.Logger = logging.getLogger("dummy"),
        **kwargs
):
    start_time = time()

    logger.debug("Running reinsertion algorithm")

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

    nodes = []
    output = np.zeros(network.num_vertices(),
                      dtype=int,
                      )
    with (
        NamedTemporaryFile("w+") as broken_fd,
        NamedTemporaryFile("w+") as output_fd,
        LogPipe(logger=logger,
                level=logging.INFO,
                ) as stdout_pipe,
        LogPipe(logger=logger,
                level=logging.ERROR,
                ) as stderr_pipe
    ):

        broken_path = broken_fd.name
        output_path = output_fd.name

        for removal in removals:
            broken_fd.write(f"{removal}\n")

        broken_fd.flush()

        cmds = [
            # 'make clean && make',

            # Build the reinsertion program, if necessary
            "make",

            # Run the reinsertion algorithm
            f"./{reinsertion_executable} "
            f"--NetworkFile {network_path} "
            f"--IDFile \"{broken_path}\" "
            f"--OutFile \"{output_path}\" "
            f"--TargetSize {stop_condition} "
            f"--SortStrategy {reinsertion_strategy} ",
        ]

        for cmd in cmds:
            try:
                logger.debug(f"Running command: {cd_cmd + cmd}")
                run(cd_cmd + cmd,
                    shell=True,
                    stdout=stdout_pipe,
                    stderr=stderr_pipe,
                    text=True,
                    check=True,
                    )
            except CalledProcessError as e:
                logger.error(f"ERROR while running reinsertion algorithm: {e}", exc_info=True)
                raise RuntimeError(f"ERROR! {e}")
            except Exception as e:
                raise RuntimeError("ERROR! {}".format(e))

        with open(output_path, "r") as output_fd:
            # Read the output file
            # Count the number of lines
            num_removals = 0
            for _ in output_fd.readlines():
                num_removals += 1

            if num_removals <= 0:
                raise RuntimeError("No removals found!")

            output_fd.seek(0)
            for i, line in enumerate(output_fd.readlines(), start=0):
                node = int(line.strip())
                # node -= 1

                nodes.append(node)

                output[node] = num_removals - i

                if output[node] <= 0:
                    logger.error(f"Node {node} has a non-positive value: {output[node]}")
                    raise RuntimeError(f"Node {node} has a non-positive value: {output[node]}")

    logger.debug("Reinsertion algorithm finished")
    logger.debug(f"Original number of removals: {len(removals)}")
    logger.debug(f"Number of final removals: {num_removals}")

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

                tmp.flush()

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
        # dismantler=lcc_threshold_dismantler,
        dismantler=threshold_dismantler,
        threshold: float = None,
        logger=logging.getLogger("dummy"),
):
    logger.info(f"Using dismantler {dismantler.__name__}")

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
        with tqdm(
                network_df.iterrows(),
                ascii=False,
                desc="Reinserting",
                leave=False,
        ) as runs_iterable:
            runs_iterable.set_description(network_name)

            run: pd.Series
            for _, run in runs_iterable:
                network = test_networks[network_name]

                run.drop(run_columns, inplace=True, errors="ignore")
                # Get the removals
                removals = run.pop("removals")
                removals = literal_eval(removals)

                # Remove the columns that are not needed
                run.drop(run_columns,
                         inplace=True,
                         errors="ignore",
                         )

                run: dict = run.to_dict()

                run["heuristic"] += "_reinsertion"

                reinserted_run_df = output_df.loc[
                    (output_df[list(run.keys())] == list(run.values())).all(
                        axis="columns"
                    ),
                    ["network"],
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

                removals, _, _, _ = dismantler(
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

                    # for removal in removals:
                    #     logger.info(
                    #         "\t{}-th removal: node {} ({}). LCC size: {}, SLCC size: {}".format(
                    #             *removal
                    #         )
                    #     )

                    logger.error(f"Had to remove too many nodes ({len(removals)})")
                    last_valid_index = 0
                    for i, removal in enumerate(removals):
                        if removal[2] > 0:
                            last_valid_index = i
                        else:
                            break

                    logger.error(f"Last valid index: {last_valid_index}: {removals[last_valid_index]}")
                    raise RuntimeError(f"Had to remove too many nodes ({len(removals)})")

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

    parser.add_argument(
        "-v",
        "--verbose",
        type=str.upper,
        choices=["INFO", "DEBUG", "WARNING", "ERROR"],
        default="info",
        help="Verbosity level (case insensitive)",
    )

    args, cmdline_args = parser.parse_known_args(parse_string)

    logger.setLevel(logging.getLevelName(args.verbose))

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

    logger.setLevel(logging.getLevelName(args.verbose))

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
