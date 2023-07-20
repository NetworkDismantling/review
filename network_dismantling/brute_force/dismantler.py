import argparse
import multiprocessing
from itertools import combinations, chain
from pathlib import Path

import numpy as np
from graph_tool.topology import label_largest_component
from tqdm import tqdm

from network_dismantling.common.dataset_providers import storage_provider


def init_network_provider(location, filter="*"):
    networks = storage_provider(location, filter=filter)

    for _, network in networks:
        # Store node ids of the original network as property
        # TODO move this to dataset generation
        network.vertex_properties["static_id"] = network.new_vertex_property("int", vals=network.vertex_index)

    return networks


def optimal_threshold_dismantler(network, stop_condition, k_range, target_property_name="target"):
    targets = network.new_vertex_property("string")
    network.vertex_properties[target_property_name] = targets

    best_combinations = set()

    network_size = network.num_vertices()

    for k in k_range:
        print("Trying with {} / {} vertices long combinations".format(k, network_size))
        best_score = network_size

        # Generate all the possible combinations of length k
        for combination in combinations(network.get_vertices(), k):
            local_network = network.copy()
            local_network.set_fast_edge_removal(fast=True)

            local_network.remove_vertex(combination, fast=True)
            # local_network.clear_vertex(combination)

            local_network_lcc_size = (np.count_nonzero(label_largest_component(local_network).get_array()))

            if local_network_lcc_size < best_score:
                best_score = local_network_lcc_size
                best_combinations = set()
                best_combinations.add(combination)
            elif local_network_lcc_size == best_score:
                best_combinations.add(combination)

        if best_score <= stop_condition:
            print("Found that {} vertices break the network apart".format(k))
            break

    num_combinations = len(best_combinations)

    # Count number of occurrences:
    all_occurrences = list(chain.from_iterable(best_combinations))
    unique, counts = np.unique(all_occurrences, return_counts=True)

    if len(unique) == 0:
        # Handle networks that already satisfy the requirement
        for v in network.get_vertices():
            targets[v] = 0
    else:

        for key, value in zip(list(unique), list(counts)):
            targets[key] = (value / num_combinations)

    return targets


t = tqdm([])


def _callback(x):
    t.update()


def main(args):
    networks_provider = init_network_provider(args.location, filter=args.test_filter)

    # Create the Log Queue
    mp_manager = multiprocessing.Manager()

    if args.removals_num:
        k_range = [args.removals_num]
        threshold = 0
        target_property_name = "k_{}".format(args.removals_num)
    else:
        k_range = None
        target_property_name = "t_{}".format(args.threshold)
        threshold = args.threshold

    with mp_manager.Pool(processes=args.jobs, initializer=tqdm.set_lock, initargs=(mp_manager.RLock(),)) as p:

        t.total = len(networks_provider)
        t.refresh()

        for name, network in networks_provider:
            output_file = args.output / (name + ".graphml")

            # if output_file.exists():
            if target_property_name in network.vertex_properties.keys():
                print("{} already processed. Skipping it.".format(name))

            else:
                p.apply_async(func=bruteforce_wrapper,
                              args=(network, name, k_range, target_property_name, threshold, str(output_file)),
                              kwds=dict(), callback=_callback, error_callback=print)

        p.close()
        p.join()


def bruteforce_wrapper(network, name, k_range, target_property_name, threshold, output_file):
    print("Processing network {}".format(name))

    network_size = network.num_vertices()

    if k_range is None:
        k_range = range(0, network_size)

    # Compute stop condition
    stop_condition = np.ceil(network_size * threshold)

    print("Dismantling {} optimally. Aiming to LCC size {} ({})".format(name,
                                                                        stop_condition,
                                                                        stop_condition / network_size))

    optimal_set_belongings = optimal_threshold_dismantler(network, stop_condition, k_range=k_range,
                                                          target_property_name=target_property_name)

    if output_file is not None:
        # Store static ID of the nodes
        network.vertex_properties["static_id"] = network.new_vertex_property("int", vals=network.vertex_index)

        print("Storing the graph")
        network.save(output_file, fmt='graphml')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-l",
        "--location",
        type=Path,
        default=None,
        help="Location of the dataset (directory)",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Location of the output folder (directory)",
    )

    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.1,
        help="Dismantling stop threshold",
    )
    parser.add_argument(
        "-k",
        "--removals_num",
        type=int,
        default=None,
        required=False,
        help="[TARGET] Exact num of removals to perform",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="Number of jobs to dispatch",
    )

    parser.add_argument(
        "-Ft",
        "--test_filter",
        type=str,
        default="*",
        required=False,
        help="Test folder filter",
    )

    args, cmdline_args = parser.parse_known_args()

    if not args.location.is_absolute():
        args.location = args.location.resolve()

    if args.output is None:
        args.output = args.location

    if not args.output.is_absolute():
        args.output = args.output.resolve()

    main(args)
