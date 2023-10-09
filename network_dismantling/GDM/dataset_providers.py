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

from glob import glob
from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch_geometric.data import Data

from network_dismantling.GDM.config import all_features
from network_dismantling.GDM.training_data_extractor import training_data_extractor
from network_dismantling.common.dataset_providers import storage_provider


def list_files(location, filter="*", extensions: Union[list, str] = ("graphml", "gt"), **kwargs):
    if not isinstance(filter, list):
        filter = [filter]

    # extension = "gt"
    if not isinstance(extensions, (list, tuple)):
        extensions = [extensions]

    files = []
    for extension in extensions:
        for f in filter:
            loc = location / f"{f}.{extension}"
            files += glob(str(loc))

    files = sorted([Path(file).stem for file in files])

    if len(files) == 0:
        raise FileNotFoundError

    return files


def prepare_graph(network, features=None, targets=None):
    # Retrieve node features and targets

    # TODO IMPROVE ME
    if features is None:
        features = all_features

    if "None" in features:
        x = np.ones((network.num_vertices(), 1))
    else:
        features_to_compute = np.setdiff1d(features, network.vertex_properties.keys())

        training_data_extractor(network,
                                compute_targets=False,
                                features=features_to_compute,
                                # logger=print,
                                )

        x = np.column_stack(tuple(
            network.vertex_properties[feature].get_array() for feature in features
        ))

    x = torch.from_numpy(x).to(torch.float)

    if targets is None:
        y = None
    else:
        targets = network.vertex_properties[targets]

        y = targets.get_array()
        y = torch.from_numpy(y).to(torch.float)

    edge_index = np.empty((2, 2 * network.num_edges()), dtype=np.int32)
    i = 0
    for e in network.edges():
        # TODO Can we replace the index here?
        # network.edge_index[e]
        edge_index[:, i] = (network.vertex_index[e.source()], network.vertex_index[e.target()])
        edge_index[:, i + 1] = (network.vertex_index[e.target()], network.vertex_index[e.source()])

        i += 2

    edge_index = torch.from_numpy(edge_index).to(torch.long)

    data = Data(x=x, y=y, edge_index=edge_index)

    return data


def init_network_provider(location, targets, features_list, max_num_vertices=None, filter="*", callback=None,
                          manager=None):
    # if isinstance(location, (str, Path)):
    networks = storage_provider(location,
                                max_num_vertices=max_num_vertices,
                                filter=filter,
                                extensions=["graphml", "gt"],
                                callback=callback,
                                )
    # else:
    #     networks = location

    networks_names, networks = zip(*networks)

    if manager is not None:
        pp_networks = manager.dict()
    else:
        pp_networks = {}

    list_class = list if manager is None else manager.list

    for features in features_list:
        key = '_'.join(features)

        # TODO REMOVE THIS LIST
        pp_networks[key] = list_class(list(zip(networks_names, networks,
                                               map(lambda n: prepare_graph(n,
                                                                           features=features,
                                                                           targets=targets,
                                                                           ),
                                                   networks
                                                   )
                                               )
                                           )
                                      )

    return pp_networks
