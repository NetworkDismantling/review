import argparse
import tempfile
from ast import literal_eval
from operator import itemgetter
from os import remove
from os.path import relpath, dirname, realpath
from pathlib import Path
from subprocess import check_output, STDOUT
from time import time

import numpy as np
import pandas as pd
from network_dismantling.machine_learning.pytorch.common import extend_filename
from scipy.integrate import simps
from tqdm import tqdm

# from network_dismantling.common.dismantlers import threshold_dismantler, lcc_threshold_dismantler
from network_dismantling.common.external_dismantlers.lcc_threshold_dismantler import threshold_dismantler
from network_dismantling.common.multiprocessing import null_logger
from network_dismantling.machine_learning.pytorch.dataset_providers import storage_provider

# Define run columns to match the runs
run_columns = [
    # "removals",
    "slcc_peak_at",
    "lcc_size_at_peak",
    "slcc_size_at_peak",
    "r_auc",
    # TODO
    # "seed",
    "average_dmg",
    "rem_num",
    "idx"
]


def get_predictions(network, removals, stop_condition, logger=null_logger):
    start_time = time()

    predictions = reinsert(network, removals, stop_condition)

    time_spent = time() - start_time

    predictions = list(zip([network.vertex_properties["static_id"][v] for v in network.vertices()], predictions))
    return predictions, time_spent


def static_predictor(network, removals, stop_condition, logger=null_logger):
    predictions = get_predictions(network, removals, stop_condition)

    # Get highest predicted value
    sorted_predictions = sorted(predictions, key=itemgetter(1), reverse=True)

    for v, p in sorted_predictions:
        yield v, p


def reinsert(network, removals, stop_condition):
    folder = 'network_dismantling/heuristics/reinsertion/'
    cd_cmd = 'cd {} && '.format(folder)

    config_r_file = "config_r.h"
    reinsertion_strategy = 2

    static_id = network.vertex_properties["static_id"]

    network_fd, network_path = tempfile.mkstemp()
    broken_fd, broken_path = tempfile.mkstemp()
    output_fd, output_path = tempfile.mkstemp()

    nodes = []
    try:
        with open(network_fd, 'w+') as tmp:
            for edge in network.edges():
                tmp.write("{} {}\n".format(
                    static_id[edge.source()],
                    static_id[edge.target()]
                    )
                )
            # for edge in network.get_edges():
            #     # TODO STATIC ID?
            #     tmp.write("{} {}\n".format(int(edge[0]) + 1, int(edge[1]) + 1))

        with open(broken_fd, "w+") as tmp:
            for removal in removals:
                tmp.write("{}\n".format(removal))

        cmds = [
            'make clean && make',
            './reinsertion -t {}'.format(
                stop_condition,
            )
        ]

        with open(folder + config_r_file, "w+") as f:
            f.write(("const char* fileNet = \"{}\";  // input format of each line: id1 id2\n"
                     "const char* fileId = \"{}\";   // output the id of the removed nodes in order\n"
                     "const char* outputFile = \"{}\";   // output the id of the removed nodes after reinserting\n"
                     "const int strategy = {}; // removing order\n"
                     "                             // 0: keep the original order\n"
                     "                             // 1: ascending order - better strategy for weighted case\n"
                     "                             // 2: descending order - better strategy for unweighted case\n"
                     ).format("../" + relpath(network_path, dirname(realpath(__file__))),
                              "../" + relpath(broken_path, dirname(realpath(__file__))),
                              "../" + relpath(output_path, dirname(realpath(__file__))),
                              reinsertion_strategy
                              )
                    )

        for cmd in cmds:
            try:
                print(f"Running cmd: {cmd}")

                check_output(cd_cmd + cmd,
                             shell=True,
                             text=True,
                             close_fds=True,
                             stderr=STDOUT,
                             )
            except Exception as e:
                raise RuntimeError(f"ERROR! When running cmd: {cmd} {e}")

        with open(output_fd, 'r+') as tmp:
            for line in tmp.readlines():
                node = int(line.strip())

                nodes.append(node)

    finally:
        # os.close(network_fd)
        # os.close(broken_fd)
        # os.close(output_fd)

        remove(network_path)
        remove(broken_path)
        remove(output_path)

    output = np.zeros(network.num_vertices())

    for n, p in zip(nodes, list(reversed(range(1, len(nodes) + 1)))):
        output[n] = p

    return output


def main(args):
    print = tqdm.write

    # Load the runs dataframe...
    df = pd.read_csv(args.file)

    df_columns = df.columns

    if args.output_file.exists():
        output_df = pd.read_csv(args.output_file)
    else:
        output_df = pd.DataFrame(columns=df.columns)

    # ... and query it
    if args.query is not None:
        df.query(args.query, inplace=True)

    # Load the networks
    test_networks = dict(
        storage_provider(args.location_test, filter=args.test_filter)
    )

    # Filter the networks in the folder
    df = df.loc[(df["network"].isin(test_networks.keys()))]

    if args.sort_column == "average_dmg":
        df["average_dmg"] = (1 - df["lcc_size_at_peak"]) / df["slcc_peak_at"]

    # Sort dataframe
    df.sort_values(by=[args.sort_column], ascending=(not args.sort_descending), inplace=True)

    predictor = get_predictions  # static_predictor
    dismantler = threshold_dismantler

    groups = df.groupby(["network"])
    for network_name, network_df in groups:
        runs_iterable = tqdm(network_df.iterrows(), ascii=True)
        runs_iterable.set_description(network_name)

        # runs = []
        for _, run in runs_iterable:
            print("Reinserting {} {}".format("static" if run["static"] else "dynamic", run["heuristic"]))
            network = test_networks[network_name]

            removals = literal_eval(run.pop("removals"))

            run.drop(run_columns, inplace=True, errors="ignore")

            run = run.to_dict()

            reinserted_run_df = output_df.loc[
                (output_df[list(run.keys())] == list(run.values())).all(axis='columns'), ["network", "seed"]
            ]

            if len(reinserted_run_df) != 0:
                # Nothing to do. Network was already tested
                continue

            stop_condition = int(np.ceil(removals[-1][3] * network.num_vertices()))
            generator_args = {
                "removals": list(map(itemgetter(1), removals)),
                "stop_condition": stop_condition,
                "logger": print,
            }

            removals, prediction_time, dismantle_time = dismantler(network.copy(), predictor, generator_args, stop_condition)

            peak_slcc = max(removals, key=itemgetter(4))

            _run = {
                "network": network_name,
                "removals": removals,
                "slcc_peak_at": peak_slcc[0],
                "lcc_size_at_peak": peak_slcc[3],
                "slcc_size_at_peak": peak_slcc[4],
                "r_auc": simps(list(r[3] for r in removals), dx=1),
            }

            for key, value in _run.items():
                run[key] = value

            # Check if something is wrong with the removals
            if removals[-1][2] == 0:
                for removal in removals:
                    print("\t{}-th removal: node {} ({}). LCC size: {}, SLCC size: {}".format(*removal))

                raise RuntimeError

            # runs.append(run)

            run_df = pd.DataFrame(data=[run], columns=network_df.columns)

            if args.output_file is not None:
                kwargs = {
                    "path_or_buf": Path(args.output_file),
                    "index": False,
                    # header='column_names',
                    "columns": df_columns
                }

                # If dataframe exists append without writing the header
                if kwargs["path_or_buf"].exists():
                    kwargs["mode"] = "a"
                    kwargs["header"] = False

                run_df.to_csv(**kwargs)


def parse_parameters():
    parser = argparse.ArgumentParser(
        description="Graph node classification using GraphSAGE"
    )
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
    args, cmdline_args = parser.parse_known_args()

    return args


if __name__ == "__main__":
    args = parse_parameters()

    if not args.location_test.is_absolute():
        args.location_test = args.location_test.resolve()

    reinsertions_file = extend_filename(args.file, "_reinserted")

    print("Output file {}".format(args.output_file))

    main(args)
