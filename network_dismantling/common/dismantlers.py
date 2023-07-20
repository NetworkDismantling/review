import logging
from _operator import itemgetter
from functools import wraps
from operator import itemgetter
from typing import Dict, Callable, Union

import numpy as np
from graph_tool import Graph, VertexPropertyMap, GraphView
from graph_tool.topology import label_components, kcore_decomposition
from scipy.integrate import simps

from network_dismantling.common.external_dismantlers.lcc_threshold_dismantler import \
    threshold_dismantler as external_threshold_dismantler
from network_dismantling.dismantler import get_predictions


def get_lcc_slcc(network):
    # Networks are undirected and this is checked after load phase
    # Forcing directed = False triggers a GraphView call which is expensive
    belongings, counts = label_components(network)  # , directed=False)
    counts = counts.astype(int, copy=False)

    if len(counts) == 0:
        local_network_lcc_size, local_network_slcc_size = 0, 0
        lcc_index = 0
    elif len(counts) < 2:
        local_network_lcc_size, local_network_slcc_size = counts[0], 0
        lcc_index = 0
    else:
        lcc_index, slcc_index = np.argpartition(np.negative(counts), 1)[:2]
        local_network_lcc_size, local_network_slcc_size = counts[[lcc_index, slcc_index]]

    return belongings, local_network_lcc_size, local_network_slcc_size, lcc_index


def threshold_dismantler(network: Graph, node_generator: Callable, generator_args: Dict, stop_condition: int,
                         early_stopping_auc=np.inf, early_stopping_removals=np.inf, logger=logging.getLogger("dummy"),):
    removals = []

    network.set_fast_edge_removal(fast=True)

    network_size = network.num_vertices()

    # Get static and dynamic vertex IDs
    # static_id = network.vertex_properties["static_id"].get_array()
    # dynamic_id = np.arange(start=0, stop=network_size, dtype=np.int64)[static_id]

    # dynamic_id = np.empty(shape=network_size, dtype=np.int64)
    # for v in network.get_vertices():
    #     dynamic_id[static_id[v]] = network.vertex_index[v]
    #     assert dynamic_id[static_id[v]] == v

    # Init last valid vertex
    # last_vertex = network_size - 1

    for i, (v_i_static, p) in enumerate(
            node_generator(network, **generator_args),
            start=1
    ):

        # Find the vertex in graph-tool and remove it
        v_gt = network.vertex(v_i_static, use_index=True, add_missing=False)

        # To improve performance, we can "clear" the vertex instead of removing it (i.e. remove all edges)
        network.clear_vertex(v_gt)

        # Compute connected component sizes
        _, local_network_lcc_size, local_network_slcc_size, _ = get_lcc_slcc(network)

        removals.append(
            (i, v_i_static, float(p), local_network_lcc_size / network_size, local_network_slcc_size / network_size)
        )

        if local_network_lcc_size <= stop_condition:
            break

        current_auc = simps(list(map(itemgetter(3), removals)), dx=1)
        if (i > early_stopping_removals) and (current_auc > early_stopping_auc):
            # if current_auc > early_stopping_auc:
            removals.append(
                (-1, -1, -1, -1, -1)
            )

            logging.debug("EARLY STOPPING")
            break

    return removals, None, None


# TODO REMOVE THIS FROM THE REVIEW. IT IS NOT USED!
def kcore_lcc_threshold_dismantler(network: Graph, node_generator: Callable, generator_args: Dict, stop_condition: int,
                                   early_stopping_auc=np.inf, early_stopping_removals=np.inf, logger=logging.getLogger("dummy"),):
    removals = []

    network.set_fast_edge_removal(fast=True)
    network_size = network.num_vertices()

    # Init generator
    generator = node_generator(network, **generator_args)
    response = None

    # Get static and dynamic vertex IDs
    static_id = network.vertex_properties["static_id"].get_array()
    dynamic_id = np.arange(start=0, stop=network_size, dtype=np.int64)[static_id]

    # Compute connected component sizes
    belongings, local_network_lcc_size, local_network_slcc_size, lcc_index = get_lcc_slcc(network)

    kcore: Union[VertexPropertyMap, None] = None

    # Init removals counter
    i = 0
    while True:
        v_i_static, p = generator.send(response)

        # Find the vertex in graph-tool and remove it
        v_i_dynamic = dynamic_id[v_i_static]

        # assert v_i_dynamic == v_i_static
        kcore = kcore_decomposition(network, vprop=kcore)

        # Extract 2-core of the network
        # two_core_mask = lambda v: kcore[v] > 1
        two_core_mask = kcore.a > 1

        network_view = GraphView(network,
                                 vfilt=two_core_mask
                                 )

        # Check if is there any node left in the 2-core
        # Otherwise go to tree-breaking
        if network_view.num_vertices() == 0:
            break

        if (belongings[v_i_dynamic] != lcc_index) or (kcore[v_i_dynamic] < 2):
            response = False
        else:
            response = True

            v_gt = network.vertex(v_i_dynamic, use_index=True, add_missing=False)

            network.clear_vertex(v_gt)

            i += 1

            # Compute connected component sizes
            belongings, local_network_lcc_size, local_network_slcc_size, lcc_index = get_lcc_slcc(network)

            removals.append(
                (i, v_i_static, float(p), local_network_lcc_size / network_size, local_network_slcc_size / network_size)
            )

        if local_network_lcc_size <= stop_condition:
            generator.close()
            break

        current_auc = simps(list(map(itemgetter(3), removals)), dx=1)
        if (i > early_stopping_removals) and (current_auc > early_stopping_auc):
            # if current_auc > early_stopping_auc:

            # print("EARLY STOPPING")
            break

    return removals, None, None


def lcc_threshold_dismantle(network: Graph, node_generator: Callable, generator_args: Dict, stop_condition: int, logger=logging.getLogger("dummy")):
    removals = []

    network.set_fast_edge_removal(fast=True)
    network_size = network.num_vertices()

    # Init generator
    generator = node_generator(network, **generator_args)
    response = None

    # Get static and dynamic vertex IDs
    static_id = network.vertex_properties["static_id"].get_array()
    dynamic_id = np.arange(start=0, stop=network_size, dtype=np.int64)[static_id]

    # dynamic_id = np.empty(shape=network_size, dtype=np.int64)
    # for v in network.get_vertices():
    #     dynamic_id[static_id[v]] = network.vertex_index[v]
    #     assert dynamic_id[static_id[v]] == v

    # # Init last valid vertex
    # last_vertex = network_size - 1

    # Compute connected component sizes
    belongings, local_network_lcc_size, local_network_slcc_size, lcc_index = get_lcc_slcc(network)

    # Init removals counter
    i = 0
    while True:
        v_i_static, p = generator.send(response)

        # Find the vertex in graph-tool and remove it
        v_i_dynamic = dynamic_id[v_i_static]

        # assert v_i_dynamic == v_i_static

        if belongings[v_i_dynamic] != lcc_index:
            response = False
        else:
            response = True

            v_gt = network.vertex(v_i_dynamic, use_index=True, add_missing=False)

            # try:
            #     assert static_id[v_i_dynamic] == v_i_static
            #     # assert dynamic_id[static_id[v_i_dynamic]] == v_i_dynamic
            #
            # except Exception as e:
            #     print("ASSERT FAILED: static_id", static_id[v_i_dynamic], "==", "v_i_static", v_i_static)
            #     # print("A2", dynamic_id[static_id[v_i_dynamic]], "==", v_i_dynamic)
            #     raise e

            # dynamic_id[static_id[last_vertex]] = v_i_dynamic
            # network.remove_vertex(v_gt, fast=True)
            # last_vertex -= 1
            network.clear_vertex(v_gt)

            i += 1

            # Compute connected component sizes
            belongings, local_network_lcc_size, local_network_slcc_size, lcc_index = get_lcc_slcc(network)

            removals.append(
                (i, v_i_static, float(p), local_network_lcc_size / network_size, local_network_slcc_size / network_size)
            )

        if local_network_lcc_size <= stop_condition:
            generator.close()
            break

    return removals, None, None


def lcc_peak_dismantler(network: Graph, node_generator: Callable, generator_args: Dict, stop_condition: int,
                        logger: Callable = logging.getLogger("dummy")):
    removals = []

    network.set_fast_edge_removal(fast=True)
    network_size = network.num_vertices()

    # Init generator
    generator = node_generator(network, **generator_args)
    response = None

    # Init removals counter
    i = 0

    # Get static and dynamic vertex IDs
    static_id = network.vertex_properties["static_id"].get_array()
    dynamic_id = np.arange(start=0, stop=network_size, dtype=np.int64)[static_id]

    # dynamic_id = np.empty(shape=network_size, dtype=np.int64)
    # for v in network.get_vertices():
    #     dynamic_id[static_id[v]] = network.vertex_index[v]
    #     assert dynamic_id[static_id[v]] == v

    # # Init last valid vertex
    last_vertex = network_size - 1

    # Compute connected component sizes
    belongings, local_network_lcc_size, local_network_slcc_size, lcc_index = get_lcc_slcc(network)

    # Init peak SLCC value
    peak_network_slcc_size = local_network_slcc_size

    while True:
        v_i_static, p = generator.send(response)

        # Find the vertex in graph-tool and remove it
        v_i_dynamic = dynamic_id[v_i_static]

        if belongings[v_i_dynamic] != lcc_index:
            response = False
        else:
            response = True

            v_gt = network.vertex(v_i_dynamic, use_index=True, add_missing=False)

            # try:
            #     assert static_id[v_i_dynamic] == v_i_static
            #     # assert dynamic_id[static_id[v_i_dynamic]] == v_i_dynamic
            #
            # except Exception as e:
            #     print("ASSERT FAILED: static_id", static_id[v_i_dynamic], "==", "v_i_static", v_i_static)
            #     # print("A2", dynamic_id[static_id[v_i_dynamic]], "==", v_i_dynamic)
            #     raise e

            # dynamic_id[static_id[last_vertex]] = v_i_dynamic
            # network.remove_vertex(v_gt, fast=True)
            last_vertex -= 1
            network.clear_vertex(v_gt)

            i += 1

            # Compute connected component sizes
            belongings, local_network_lcc_size, local_network_slcc_size, lcc_index = get_lcc_slcc(network)

            if peak_network_slcc_size < local_network_slcc_size:
                peak_network_slcc_size = local_network_slcc_size

            removals.append(
                (i, v_i_static, float(p), local_network_lcc_size / network_size, local_network_slcc_size / network_size)
            )

        if (peak_network_slcc_size >= local_network_lcc_size) or \
                (local_network_lcc_size <= stop_condition):
            break

    # TODO REMOVE ME
    for v, p in generator:
        removals.append(
            (i, v, float(p), local_network_lcc_size / network_size, local_network_slcc_size / network_size)
        )

        last_vertex -= 1

        if last_vertex < 0:
            break

    # TODO END REMOVE ME

    generator.close()

    return removals, None, None  # prediction_time, dismantle_time


def dismantler_wrapper(function: Callable):
    @wraps(function)
    def wrapper(network: Graph,
                predictor: Callable = get_predictions,
                dismantler: Callable = external_threshold_dismantler,
                **kwargs
                ):

        generator_args = kwargs.pop("generator_args")
        try:
            logger = kwargs.get("logger")
        except KeyError:
            logger = generator_args.get("logger", logging.getLogger("dummy"))

        generator_args["sorting_function"] = function

        print("WRAPPING FUNCTION", function.__name__)

        removals, prediction_time, dismantle_time = dismantler(network=network,
                                                               predictor=predictor,
                                                               generator_args=generator_args,
                                                               # stop_condition=stop_condition,
                                                               # *args,
                                                               **kwargs
                                                               )

        peak_slcc = max(removals, key=itemgetter(4))
        rem_num = len(removals)

        if rem_num > 0:
            if removals[0][0] < 0:
                raise RuntimeError("First removal is just the LCC size!")

            if removals[-1][2] == 0:
                raise RuntimeError("ERROR: removed more nodes than predicted!")

        run = {
            # "network": name,
            "removals": removals,
            "slcc_peak_at": peak_slcc[0],
            "lcc_size_at_peak": peak_slcc[3],
            "slcc_size_at_peak": peak_slcc[4],
            # "heuristic": heuristic,
            # "static": None,
            "r_auc": simps(list(r[3] for r in removals), dx=1),
            "rem_num": rem_num,

            "prediction_time": prediction_time,
            "dismantle_time": dismantle_time
        }

        return run  # , time_run

    return wrapper
