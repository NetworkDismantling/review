#   This file is part of CoreGDM (Core Graph Dismantling with Machine learning),
#   proposed in the paper " CoreGDM: Geometric Deep Learning Network Decycling
#   and Dismantling"  by M. Grassia and G. Mangioni.
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
#   along with CoreGDM.  If not, see <http://www.gnu.org/licenses/>.

import logging
from copy import deepcopy
from datetime import timedelta
from operator import itemgetter
from time import time
from typing import Union

import numpy as np
import torch
from graph_tool import Graph, GraphView, VertexPropertyMap
from graph_tool.topology import kcore_decomposition, label_largest_component
from scipy.integrate import simpson
from torch_geometric import seed_everything
from tqdm.auto import tqdm

from network_dismantling.GDM.dataset_providers import prepare_graph
from network_dismantling.GDM.network_dismantler import init_network_provider, \
    add_run_parameters, get_df_columns, train_wrapper
from network_dismantling.GDM.training_data_extractor import training_data_extractor
from network_dismantling.common.dismantlers import threshold_dismantler
from network_dismantling.common.multiprocessing import clean_up_the_pool

get_df_columns
train_wrapper


def tree_breaker(network: Graph,
                 stop_condition: int,
                 logger: logging.Logger=logging.getLogger("dummy"),
                 ) -> np.ndarray:
    logger.debug("Running tree breaker!")
    from network_dismantling.CoreGDM.treebreaker import Graph, tree_breaker as minsum_tree_breaker, \
        CyclesError

    static_id = network.vertex_properties["static_id"]

    G = Graph()

    for edge in network.edges():
        G.add_edge(int(edge.source()) + 1,
                   int(edge.target()) + 1,
                   )

    try:
        nodes = minsum_tree_breaker(G,
                                    stop_condition=int(stop_condition)
                                    )
    except CyclesError as e:
        raise e

    output = np.zeros(network.num_vertices())

    for n, p in zip(nodes, list(reversed(range(1, len(nodes) + 1)))):
        output[static_id[int(n) - 1]] = p

    predictions_property = network.vertex_properties.get("predictions", network.new_vertex_property("float"))
    predictions_property.fa = output

    return predictions_property


def get_predictions(network, model, lock, device=None, data=None, features=None, logger=logging.getLogger("dummy")):
    logger.debug("Converting the graph and computing the predictions...")
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

    predictions_property = network.vertex_properties.get("predictions", network.new_vertex_property("float"))
    predictions_property.fa = predictions

    time_spent = time() - start_time
    logger.debug("Done predicting dismantling order. Took {} (including GPU access time, if any)".format(
        timedelta(seconds=time_spent))
    )

    return predictions_property, time_spent


def kcore_static_predictor(network, model, lock, stop_condition, device, features,
                           logger=logging.getLogger("dummy"),
                           **kwargs):
    # logger.debug("Running kcore_static_predictor")

    kcore: Union[VertexPropertyMap, None] = None

    kcore = kcore_decomposition(network, vprop=kcore)

    # Extract 2-core of the network
    two_core_mask = kcore.a > 1

    network_view = GraphView(network,
                             vfilt=two_core_mask
                             )

    training_data_extractor(network_view,
                            compute_targets=False,
                            features=features,
                            # logger=logger,
                            )

    # Get the predictions
    predictions, _ = get_predictions(network_view,
                                     # data=data,
                                     model=model,
                                     lock=lock,
                                     features=features,
                                     device=device,
                                     logger=logger,
                                     )

    # Store them into a vertex property
    network.vertex_properties["predictions"] = predictions

    # Perform at most |N| removals
    for _ in range(network.num_vertices()):
        # Compute k-core
        kcore = kcore_decomposition(network, vprop=kcore)

        # Extract 2-core of the network
        two_core_mask = kcore.a > 1
        if not (two_core_mask >= 0).any():
            # No more vertices to remove!
            break

        network_view = GraphView(network,
                                 # TODO improve this!
                                 vfilt=two_core_mask
                                 )

        # Check if is there any node left in the 2-core
        # Otherwise go to tree-breaking
        if network_view.num_vertices(ignore_filter=False) == 0:
            # logger.debug("network_view is empty!")
            break
        # else:
        #     logger.debug(f"network_view has {network_view.num_vertices()} vertices")

        # Extract the largest connected component of the 2-core
        largest_component_mask = label_largest_component(network_view, directed=False)
        network_view = GraphView(network_view,
                                 vfilt=largest_component_mask
                                 )

        static_id: np.ma.MaskedArray = network_view.vertex_properties["static_id"].ma
        predictions: np.ma.MaskedArray = network_view.vertex_properties["predictions"].ma

        # Sort predictions in descending order
        removal_order = (-predictions).argsort(kind="stable",
                                               fill_value=0,
                                               )
        for i in range(removal_order.shape[0]):
            idx = removal_order[i]
            # assert largest_component_mask.a[idx] == 1, "Node is not in the LCC!"
            # assert kcore.a[idx] > 1, "Node is not in the 2-core!"

            v = static_id[idx]
            p = predictions[idx]

            removed = yield int(v), float(p)

            if removed is not False:
                # logger.info(
                #     f"Removed {v} ({p}) with Kcore value {kcore.a[idx]} and in LCC? {largest_component_mask.a[idx]}"
                #     )
                break
            # else:
            raise RuntimeError("Dismantler rejected node in 2-core LCC.")
    else:
        raise RuntimeError("Should have removed all nodes by now")

    # Tree breaking phase
    logger.debug("Running tree breaker")

    predictions: np.ndarray = tree_breaker(network=network,
                                           stop_condition=stop_condition,
                                           logger=logger,
                                           )
    network.vertex_properties["predictions"] = predictions

    # Perform at most |N| removals
    for _ in range(network.num_vertices()):

        # Extract the largest connected component of the 2-core
        network_view = GraphView(network,
                                 vfilt=label_largest_component(network)
                                 )

        static_id: VertexPropertyMap = network_view.vertex_properties["static_id"].ma
        predictions: np.ma.MaskedArray = network_view.vertex_properties["predictions"].ma

        removal_order = (-predictions).argsort(kind="stable",
                                               fill_value=0,
                                               )

        for i in range(removal_order.shape[0]):
            idx = removal_order[i]

            v = static_id[idx]
            p = predictions[idx]
            # logger.debug(f"Removing {idx} {v} ({p})")

            if p <= 0:
                raise RuntimeError("Removing node that was not predicted by the tree breaker!")

            removed = yield int(v), float(p)

            if removed is not False:
                break
            else:
                raise RuntimeError("Dismantler rejected tree breaker node in LCC.")

    raise RuntimeError("No more vertices to remove!")


@torch.no_grad()
def test(args, model, early_stopping_dict: dict = None, networks_provider=None, print_model=True,
         logger=logging.getLogger("dummy")
         ):
    if early_stopping_dict is None:
        early_stopping_dict = {}

    if print_model:
        logger.info(model)

    seed_everything(args.seed_test)
    # torch.manual_seed(args.seed_test)
    # np.random.seed(args.seed_test)
    # seed(args.seed_test)

    if model.is_affected_by_seed():
        model.set_seed(args.seed_test)

    model.eval()

    # TODO allow to choose the dismantler (e.g., dynamic or block dynamic)
    predictor = kcore_static_predictor
    dismantler = threshold_dismantler
    # dismantler = lcc_threshold_dismantler

    if networks_provider is None:
        networks_provider = init_network_provider(args.location_test, features=args.features, targets=None)
    else:
        networks_provider = deepcopy(networks_provider)

    generator_args = {
        "model": model,
        "features": args.features,
        "device": args.device,
        "lock": args.lock,
        "logger": logger,
    }

    # Init runs buffer
    runs = []

    # noinspection PyTypeChecker
    for filename, network, data in tqdm(networks_provider,
                                        desc="Testing",
                                        leave=False,
                                        # position=1,
                                        ):

        network_size = network.num_vertices()

        # Compute stop condition
        stop_condition = int(np.ceil(network_size * float(args.threshold)))

        generator_args["stop_condition"] = stop_condition
        generator_args["data"] = data

        # logger.info(f"Dismantling {filename} according to the predictions. "
        #             f"Aiming to reach LCC size {stop_condition} ({stop_condition * 100 / network_size:.3f}%)"
        #             )

        early_stopping = early_stopping_dict.get(filename, {
            "auc": np.inf,
            "rem_num": np.inf,
        })

        removals, prediction_time, dismantle_time, _ = dismantler(network=network.copy(),
                                                               node_generator=predictor,
                                                               generator_args=generator_args,
                                                               stop_condition=stop_condition,
                                                               early_stopping_auc=early_stopping["auc"],
                                                               # if early_stopping is not None else np.inf,
                                                               early_stopping_removals=early_stopping["rem_num"],
                                                               # if early_stopping is not None else np.inf,
                                                               logger=logger,
                                                               )

        best_dismantling = removals[-1]

        r_auc = simpson(list(r[3] for r in removals), dx=1)
        rem_num = best_dismantling[0]
        # rem_num = len(removals)
        min_lcc_size = best_dismantling[3]

        if 0 <= min_lcc_size <= stop_condition:
            # logger.debug("UPDATING EARLY STOPPING VALUES")

            default_value = early_stopping_dict.get(filename, {
                "auc": np.inf,
                "rem_num": np.inf
            })
            early_stopping_dict[filename] = {
                "auc": min(default_value["auc"], r_auc),
                "rem_num": min(default_value["rem_num"], rem_num),
            }

        peak_slcc = max(removals, key=itemgetter(4))

        run = {
            "network": filename,

            "removals": removals if rem_num > 0 else [],
            "slcc_peak_at": peak_slcc[0] if rem_num > 0 else np.inf,
            "lcc_size_at_peak": peak_slcc[3] if rem_num > 0 else np.inf,
            "slcc_size_at_peak": peak_slcc[4] if rem_num > 0 else np.inf,

            "r_auc": r_auc if rem_num > 0 else np.inf,
            "rem_num": rem_num if rem_num > 0 else np.inf,

            "prediction_time": prediction_time if rem_num > 0 else np.inf,
            "dismantle_time": dismantle_time if rem_num > 0 else np.inf,
        }
        add_run_parameters(args, run, model)

        runs.append(run)

        if args.verbose > 1:
            logger.info(
                f"Percolation at {run['slcc_peak_at']}: "
                f"LCC {run['lcc_size_at_peak']}, SLCC {run['slcc_size_at_peak']}, R {run['r_auc']}"
            )

        if args.verbose == 2:
            for removal in run["removals"]:
                logger.info("\t{}-th removal: node {} ({}). LCC size: {}, SLCC size: {}".format(*removal))

        # Fix OOM
        clean_up_the_pool()

    return runs
