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
from datetime import timedelta
from time import time

import numpy as np

from network_dismantling.GDM.dataset_providers import prepare_graph
from network_dismantling.common.multiprocessing import clean_up_the_pool
from network_dismantling.machine_learning.pytorch.training_data_extractor import (
    training_data_extractor,
)


def get_predictions(
    network,
    model,
    lock,
    device=None,
    data=None,
    features=None,
    logger=logging.getLogger("dummy"),
    **kwargs,
):
    network_name = kwargs["network_name"]

    logger.debug(f"Getting predictions for {network_name}...")

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
    logger.debug(
        f"{network_name}: Done predicting dismantling order."
        f"Took {timedelta(seconds=time_spent)} (including GPU access time, if any)"
    )

    return predictions, time_spent


def dynamic_predictor(network, model, lock, device, features, data=None):
    for _ in range(network.num_vertices()):
        static_id = network.vertex_properties["static_id"].a

        training_data_extractor(
            network, compute_targets=False, features=features
        )  # , logger=print)

        predictions, _ = get_predictions(
            network, model, lock, data=data, features=features, device=device
        )

        while predictions.shape[0] > 0:
            # Get the largest predicted value
            # i = predictions[mask, 1].argmax()
            # i = predictions[:, 1].argmax()
            i = predictions.argmax()

            removed = yield static_id[i], predictions[i]

            if removed is not False:
                # Vertex was removed. Get new predictions.
                break

            # Otherwise, just mask out the vertex
            predictions[i] = 0
            # predictions = np.delete(predictions, i, 0)
            # mask[i] = False

    raise RuntimeError("No more vertices to remove!")


def block_dynamic_predictor(network, model, lock, features, device, k, data=None):
    for _ in range(network.num_vertices()):
        static_id = network.vertex_properties["static_id"].a

        num_removals = 0
        training_data_extractor(
            network, compute_targets=False, features=features
        )  # , logger=print)

        predictions, _ = get_predictions(
            network, model, lock, data=data, features=features, device=device
        )

        while predictions.shape[0] > 0:
            # Get the largest predicted value
            # i = predictions[mask, 1].argmax()
            # i = predictions[:, 1].argmax()
            i = predictions.argmax()

            removed = yield static_id[i], predictions[i]

            if removed is not False:
                # Vertex was removed.
                num_removals += 1

                # If we removed at least k vertices from the current predictions, refresh them.
                if num_removals > k:
                    break

            # Otherwise, just mask out the vertex
            predictions[i] = 0
            # predictions = np.delete(predictions, i, 0)
            # mask[i] = False

    raise RuntimeError("No more vertices to remove!")


def static_predictor(
    network,
    model,
    lock,
    data,
    features,
    device,
    logger=logging.getLogger("dummy"),
):
    logger.info("Predicting dismantling order. ")
    start_time = time()

    predictions, _ = get_predictions(
        network, model, lock, data=data, features=features, device=device
    )
    # predictions = list(zip(network.vertex_properties["static_id"].a, predictions))

    # Sort by highest prediction value
    removal_indices = np.argsort(-predictions, kind="stable")
    static_id = network.vertex_properties["static_id"].a

    logger.info(
        "Done predicting dismantling order. Took {} (including GPU access time and sorting)".format(
            timedelta(seconds=(time() - start_time))
        )
    )

    for i in removal_indices:
        yield static_id[i], predictions[i]


# def lcc_static_predictor(network, model, lock, data, features, device, logger=logging.getLogger('dummy')):
#     logger.info("Predicting dismantling order. ")
#     start_time = time()
#
#     predictions, _ = get_predictions(network, model, lock, data=data, features=features, device=device)
#
#     # TODO IMPROVE SORTING!
#     # Sort by highest prediction value
#     sorted_predictions = sorted(predictions, key=itemgetter(1), reverse=True)
#     logger.info("Done predicting dismantling order. Took {} (including GPU access time and sorting)".format(
#         timedelta(seconds=(time() - start_time)))
#     )
#
#     i = 0
#     while True:
#         if i >= len(sorted_predictions):
#             break
#
#         removed = yield sorted_predictions[i]
#         if removed is not False:
#             # Vertex was removed, remove it from predictions
#             del sorted_predictions[i]
#
#             # ... and start over
#             i = 0
#
#         else:
#             i += 1
#
#     raise RuntimeError("No more vertices to remove!")


def lcc_static_predictor(
    network, model, lock, data, features, device, logger=logging.getLogger("dummy")
):
    logger.warning(
        "WARNING!! Using the LCC static predictor. THIS NEW VERSION WAS NOT TESTED YET!!!"
    )
    logger.info("Predicting dismantling order. ")
    start_time = time()

    predictions, _ = get_predictions(
        network, model, lock, data=data, features=features, device=device
    )
    # predictions = list(zip(network.vertex_properties["static_id"].a, predictions))
    masked_predictions: np.ma.masked_array = np.ma.masked_array(predictions, mask=False)

    static_id = network.vertex_properties["static_id"].a
    while True:
        # Sort by highest prediction value
        # removal_indices = np.argsort(-masked_predictions, kind="stable")
        i = masked_predictions.argmax()

        removed = yield static_id[i], predictions[i]

        if removed is not False:
            # Vertex was removed, remove it from predictions
            predictions[i] = 0

            # ... and start over
            masked_predictions.mask = False

        else:
            masked_predictions.mask[i] = True
