import logging
from datetime import timedelta
from time import time, perf_counter_ns
# from traceback import print_tb

import numpy as np
from network_dismantling.common.external_dismantlers.dismantler import Graph, lccThresholdDismantler, thresholdDismantler


def test_network_callback(network):
    from graph_tool.all import remove_parallel_edges, remove_self_loops

    remove_parallel_edges(network)
    remove_self_loops(network)

    static_id = network.vertex_properties["static_id"]
    edges = list(map(
        lambda e: (
            static_id[e.source()],
            static_id[e.target()]
        ),
        network.edges()
    ))

    if len(edges) == 0:
        raise RuntimeError

    return Graph(edges)


cache = dict()


def add_dismantling_edges(filename, network):
    cache[filename] = test_network_callback(network)

    return cache[filename]


# def _threshold_dismantler(network, predictions, generator_args, stop_condition, dismantler):
def _threshold_dismantler(network, predictor, generator_args, stop_condition, dismantler, **kwargs):
    logger = generator_args.get("logger", logging.getLogger('dummy'))

    network_name = generator_args["network_name"]

    predictions, prediction_time = predictor(network, **generator_args)

    # Get the highest predicted value
    logger.info(f"{network_name}: Sorting the predictions...")
    start_time = time()
    removal_indices = np.argsort(-predictions, kind="stable")

    logger.info(f"{network_name}: Done sorting. Took {timedelta(seconds=(time() - start_time))}")

    removal_order = network.vertex_properties["static_id"].a[removal_indices]
    removal_order = removal_order.tolist()

    network_size = network.num_vertices()
    filename = network.graph_properties["filename"]

    try:
        external_network = cache[filename]
    except:
        external_network = add_dismantling_edges(filename, network)

    external_network = Graph(external_network)

    logger.info(f"{network_name}: Invoking the external dismantler.")
    start_time = perf_counter_ns()

    try:
        raw_removals = dismantler(external_network, removal_order, stop_condition)
    except Exception as e:
        logger.info(f"{network_name}: ERROR: {e}")
        logger.exception(e)

        raise e
    finally:
        try:
            del external_network
        except Exception as e:
            logger.info(f"{network_name}: ERROR when deleting external_network: {e}")
            logger.exception(e)

    dismantle_time = perf_counter_ns() - start_time  # in ns
    dismantle_time /= 1e9 # in s

    logger.info(f"{network_name}: External dismantler returned in {dismantle_time}s")

    # predictions_dict = dict(predictions)
    predictions_dict = dict(zip(network.vertex_properties["static_id"].a.tolist(), predictions.tolist()))

    removals = []
    for i, (s_id, lcc_size, slcc_size) in enumerate(raw_removals, start=1):
        removals.append(
            (i, s_id, float(predictions_dict[s_id]), lcc_size / network_size, slcc_size / network_size)
        )

    del predictions_dict

    return removals, prediction_time, dismantle_time


def lcc_threshold_dismantler(network, predictor, generator_args, stop_condition, **kwargs):
    kwargs["dismantler"] = lccThresholdDismantler

    return _threshold_dismantler(network, predictor, generator_args, stop_condition, **kwargs)


def threshold_dismantler(network, predictor, generator_args, stop_condition, **kwargs):
    kwargs["dismantler"] = thresholdDismantler

    # assert "generator_args" in kwargs, "threshold_dismantler: generator_args must be provided"

    return _threshold_dismantler(network, predictor, generator_args, stop_condition, **kwargs)


def _iterative_threshold_dismantler(network, predictor, generator_args, stop_condition):
    # network = network.copy()
    network.set_fast_edge_removal(fast=True)

    logger = generator_args["logger"]
    network_name = generator_args["network_name"]
    filename = network.graph_properties["filename"]

    try:
        external_network = cache[filename]
    except:
        external_network = add_dismantling_edges(filename, network)

    external_network = Graph(external_network)

    start_time = perf_counter_ns()

    removals = []
    try:

        for i, (removal_static_id, removal_value) in enumerate(predictor(network, **generator_args), start=1):
            # Get the highest predicted value
            for s_id, lcc_size, slcc_size in thresholdDismantler(external_network, [removal_static_id], stop_condition):

                assert s_id == removal_static_id

                network_size = network.num_vertices()

                v_gt = network.vertex(removal_static_id,
                                      use_index=True,
                                      add_missing=False,
                                      )

                network.clear_vertex(v_gt)

                removals.append(
                    (i, removal_static_id, float(removal_value), lcc_size / network_size, slcc_size / network_size)
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

    logger.info(f"{network_name}: iterative external dismantler returned in {dismantle_time}s")

    return removals, None, None


def iterative_threshold_dismantler(network, predictor, generator_args, stop_condition):
    return _iterative_threshold_dismantler(network, predictor, generator_args, stop_condition)
