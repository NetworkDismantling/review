#   This file is part of GDM (Graph Dismantling with Machine learning),
#   proposed in the paper "Machine learning dismantling and
#   early-warning signals of disintegration in complex systems"
#   by M. Grassia, M. De Domenico and G. Mangioni.
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
from network_dismantling.common.loaders import load_graph


def storage_provider(location, max_num_vertices=None, filter="*", extensions: Union[list, str] = ("graphml", "gt"),
                     callback=None):
    if not location.is_absolute():
        location = location.resolve()

    if not location.exists():
        raise FileNotFoundError(f"Location {location} does not exist.")
    elif not location.is_dir():
        raise FileNotFoundError(f"Location {location} is not a directory.")

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

    files = sorted(files)

    if len(files) == 0:
        raise FileNotFoundError

    networks = list()
    for file in files:
        filename = Path(file).stem

        network = load_graph(file)

        if (max_num_vertices is not None) and (network.num_vertices() > max_num_vertices):
            continue

        assert not network.is_directed()

        network.graph_properties["filename"] = network.new_graph_property("string", filename)
        network.graph_properties["filepath"] = network.new_graph_property("string", str(file))

        if callback:
            callback(filename, network)

        networks.append((filename, network))

    return networks


def prepare_graph(network, features=None, targets=None):
    # Retrieve node features and targets

    # TODO IMPROVE ME
    if features is None:
        features = all_features

    if "None" in features:
        x = np.ones((network.num_vertices(), 1))
    else:
        x = np.column_stack(
            tuple(
                network.vertex_properties[feature].get_array() for feature in features
            )
        )
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
