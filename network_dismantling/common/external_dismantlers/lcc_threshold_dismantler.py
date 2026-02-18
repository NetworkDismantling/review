import logging
from datetime import timedelta
from time import time, perf_counter_ns
from typing import Callable, Dict, List, Tuple, Optional

import numpy as np
from graph_tool import Graph
from network_dismantling.common.external_dismantlers.dismantler import Graph as ExternalGraph
from network_dismantling.common.removal import Removal, RemovalsList


# from traceback import print_tb


def test_network_callback(network: Graph):
    from graph_tool.all import remove_parallel_edges, remove_self_loops

    remove_parallel_edges(network)
    remove_self_loops(network)

    static_id = network.vertex_properties["static_id"]

    edges = list(
        map(lambda e: (static_id[e.source()], static_id[e.target()]), network.edges())
    )

    print(f"External dismantler: loaded network with {network.num_vertices()} vertices and {len(edges)} edges.",
          flush=True)
    # print(f"Edges (type {type(edges)}): {edges}", flush=True)

    # if len(edges) == 0:
    #     raise RuntimeError("No edges in network")

    eg = ExternalGraph(network.graph_properties["filename"])
    eg.addNodes(static_id.a.tolist())
    eg.addEdgeList(edges)

    assert eg.getNumNodes() == network.num_vertices()
    assert eg.getNumEdges() == network.num_edges()

    return eg


cache: Dict[str, ExternalGraph] = dict()


# def add_dismantling_edges(filename: str, network: Graph) -> ExternalGraph:
#     cache[filename] = test_network_callback(network)

#     return cache[filename]

def getExternalGraph(network: Graph, logger: logging.Logger) -> ExternalGraph:
    from network_dismantling.common.external_dismantlers.dismantler import Graph as ExternalGraph

    filename = network.graph_properties["filename"]

    try:
        external_network = cache[filename]
        logger.debug(f"Using cached ExternalGraph for {filename}, creating deep copy")
        # Create a deep copy to avoid side effects from dismantling operations
        return external_network.deepCopy()
    except KeyError:
        logger.debug(f"Creating new ExternalGraph for {filename}")
        external_network = test_network_callback(network)
        cache[filename] = external_network
        # Return a deep copy, keeping the original in cache
        return external_network.deepCopy()


# def _threshold_dismantler(network, predictions, generator_args, stop_condition, dismantler):
def _threshold_dismantler(
        network: Graph,
        predictor: Callable[[Graph], List[Tuple[int, float]]],
        generator_args: Dict,
        stop_condition: int,
        dismantler: Callable[[ExternalGraph, List[int], int], List[Tuple[int, int, int]]],
        logger: logging.Logger = logging.getLogger("dummy"),
        **kwargs,
) -> Tuple[RemovalsList, float, float]:
    """Core threshold dismantler using external C++ implementation.
    
    Returns removals with ABSOLUTE counts (lcc_size, slcc_size are node counts, not fractions).
    """
    from network_dismantling.common.external_dismantlers.dismantler import Graph as ExternalGraph

    network_name = generator_args.get(
        "network_name",
        network.graph_properties.get("filename", "unknown"),
    )

    predictions, prediction_time = predictor(network, **generator_args)

    # Get the highest predicted value
    logger.debug(f"{network_name}: Sorting the predictions...")
    start_time = time()
    removal_indices = np.argsort(-predictions, kind="stable")

    logger.debug(
        f"{network_name}: Done sorting. Took {timedelta(seconds=(time() - start_time))}"
    )

    removal_order = network.vertex_properties["static_id"].a[removal_indices]
    removal_order = removal_order.tolist()

    # network_size = network.num_vertices()

    external_network: ExternalGraph = getExternalGraph(network, logger)

    logger.debug(f"{network_name}: Invoking the external dismantler.")
    start_time = perf_counter_ns()

    try:
        raw_removals = dismantler(external_network, removal_order, stop_condition)
    except Exception as e:
        logger.exception(f"{network_name}: ERROR {e}", exc_info=True)

        raise e
    finally:
        try:
            del external_network
        except Exception as e:
            logger.exception(
                f"{network_name}: ERROR when deleting external_network {e}",
                exc_info=True,
            )

    dismantle_time = perf_counter_ns() - start_time  # in ns
    dismantle_time /= 1e9  # in s

    logger.debug(f"{network_name}: External dismantler returned in {dismantle_time}s")

    # predictions_dict = dict(predictions)
    predictions_dict = dict(
        zip(network.vertex_properties["static_id"].a.tolist(), predictions.tolist())
    )

    removals: RemovalsList = []
    for i, (s_id, lcc_size, slcc_size) in enumerate(raw_removals, start=1):
        removals.append(
            Removal(
                removal_num=i,
                node_id=s_id,
                prediction=float(predictions_dict[s_id]),
                lcc_size=lcc_size,      # C++ returns absolute count
                slcc_size=slcc_size,    # C++ returns absolute count
            )
        )

    del predictions_dict

    return removals, prediction_time, dismantle_time


def lcc_threshold_dismantler(
        network: Graph,
        predictor: Callable,
        generator_args: Dict,
        stop_condition: int,
        **kwargs
) -> Tuple[RemovalsList, float, float]:
    """Dismantle network using LCC threshold strategy (external C++ implementation).
    
    Returns removals with ABSOLUTE counts (lcc_size, slcc_size are node counts, not fractions).
    """
    from network_dismantling.common.external_dismantlers.dismantler import (
        lccThresholdDismantler,
    )

    kwargs["dismantler"] = lccThresholdDismantler

    return _threshold_dismantler(
        network, predictor, generator_args, stop_condition, **kwargs
    )


def threshold_dismantler(
        network: Graph,
        predictor: Callable,
        generator_args: Dict,
        stop_condition: int,
        **kwargs
) -> Tuple[RemovalsList, float, float]:
    """Dismantle network using threshold strategy (external C++ implementation).
    
    Returns removals with ABSOLUTE counts (lcc_size, slcc_size are node counts, not fractions).
    """
    from network_dismantling.common.external_dismantlers.dismantler import (
        thresholdDismantler,
    )

    kwargs["dismantler"] = thresholdDismantler

    # assert "generator_args" in kwargs, "threshold_dismantler: generator_args must be provided"

    return _threshold_dismantler(
        network, predictor, generator_args, stop_condition, **kwargs
    )


def _iterative_threshold_dismantler(
        network: Graph,
        predictor: Callable[[Graph], List[Tuple[int, float]]],
        generator_args: Dict,
        stop_condition: int
) -> Tuple[RemovalsList, Optional[float], Optional[float]]:
    """Iterative threshold dismantler (external C++ implementation).
    
    Returns removals with ABSOLUTE counts (lcc_size, slcc_size are node counts, not fractions).
    """
    from network_dismantling.common.external_dismantlers.dismantler import (
        Graph,
        thresholdDismantler,
    )

    # network = network.copy()
    network.set_fast_edge_removal(fast=True)

    logger = generator_args.get("logger", logging.getLogger("dummy"))
    network_name = generator_args.get(
        "network_name",
        network.graph_properties.get("filename", "unknown"),
    )

    external_network: ExternalGraph = getExternalGraph(network, logger)

    start_time = perf_counter_ns()

    removals: RemovalsList = []
    try:
        for i, (removal_static_id, removal_value) in enumerate(
                predictor(network, **generator_args), start=1
        ):
            # Get the highest predicted value
            for s_id, lcc_size, slcc_size in thresholdDismantler(
                    external_network, [removal_static_id], stop_condition
            ):
                assert s_id == removal_static_id

                v_gt = network.vertex(
                    removal_static_id,
                    use_index=True,
                    add_missing=False,
                )

                network.clear_vertex(v_gt)

                removals.append(
                    Removal(
                        removal_num=i,
                        node_id=removal_static_id,
                        prediction=float(removal_value),
                        lcc_size=lcc_size,      # C++ returns absolute count
                        slcc_size=slcc_size,    # C++ returns absolute count
                    )
                )

                if lcc_size <= stop_condition:
                    raise StopIteration

    except StopIteration:
        pass

    except Exception as e:
        logger.error(f"{network_name}: ERROR: {e}")
        logger.exception(e)

        raise e
    finally:
        try:
            del external_network
        except Exception as e:
            logger.info(f"{network_name}: ERROR: {e}")
            logger.exception(e)

    dismantle_time = perf_counter_ns() - start_time  # in ns
    dismantle_time /= 1e9  # in s

    logger.info(
        f"{network_name}: iterative external dismantler returned in {dismantle_time}s"
    )

    return removals, None, None


def iterative_threshold_dismantler(
        network: Graph,
        predictor: Callable,
        generator_args: Dict,
        stop_condition: int
) -> Tuple[RemovalsList, Optional[float], Optional[float]]:
    """Iterative threshold dismantler (external C++ implementation).
    
    Returns removals with ABSOLUTE counts (lcc_size, slcc_size are node counts, not fractions).
    """
    return _iterative_threshold_dismantler(
        network, predictor, generator_args, stop_condition
    )
