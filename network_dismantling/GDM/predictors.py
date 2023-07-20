import logging
from _operator import itemgetter
from datetime import timedelta
from time import time

import numpy as np

from network_dismantling.GDM.dataset_providers import prepare_graph
from network_dismantling.common.multiprocessing import clean_up_the_pool


def static_predictor(network, model, lock, data, features, device, logger=logging.getLogger('dummy'), ):
    logger.info("Predicting dismantling order. ")
    start_time = time()

    predictions, _ = get_predictions(network, model, lock, data=data, features=features, device=device)
    # predictions = list(zip(network.vertex_properties["static_id"].a, predictions))

    # Sort by highest prediction value
    removal_indices = np.argsort(-predictions, kind="stable")
    static_id = network.vertex_properties["static_id"].a

    logger.info("Done predicting dismantling order. Took {} (including GPU access time and sorting)".format(
        timedelta(seconds=(time() - start_time))))

    for i in removal_indices:
        yield static_id[i], predictions[i]


def get_predictions(network, model, lock, device=None, data=None, features=None, logger=logging.getLogger('dummy'),
                    **kwargs):
    network_name = kwargs["network_name"]

    logger.info(f"Getting predictions for {network_name}...")

    start_time = time()

    if data is None:
        data = prepare_graph(network, features=features)

    with lock:
        if device:
            data.x = data.x.to(device)
            data.edge_index = data.edge_index.to(device)

        try:
            predictions = model(data.x, data.edge_index)
            predictions = predictions.cpu()
            predictions = predictions.numpy()

        finally:
            # Fix OOM
            del data.x
            del data.edge_index
            del data
            clean_up_the_pool()

    time_spent = time() - start_time
    logger.info(f"{network_name}: Done predicting dismantling order."
                f"Took {timedelta(seconds=time_spent)} (including GPU access time, if any)"
                )

    return predictions, time_spent


def lcc_static_predictor(network, model, lock, data, features, device, logger=logging.getLogger('dummy')):
    logger.info("Predicting dismantling order. ")
    start_time = time()

    predictions, _ = get_predictions(network, model, lock, data=data, features=features, device=device)

    # TODO IMPROVE SORTING!
    # Sort by highest prediction value
    sorted_predictions = sorted(predictions, key=itemgetter(1), reverse=True)
    logger.info("Done predicting dismantling order. Took {} (including GPU access time and sorting)".format(
        timedelta(seconds=(time() - start_time)))
    )

    i = 0
    while True:
        if i >= len(sorted_predictions):
            break

        removed = yield sorted_predictions[i]
        if removed is not False:
            # Vertex was removed, remove it from predictions
            del sorted_predictions[i]

            # ... and start over
            i = 0

        else:
            i += 1

    raise RuntimeError("No more vertices to remove!")
