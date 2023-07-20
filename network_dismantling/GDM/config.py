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

from decimal import Decimal

from network_dismantling.common.config import output_path

threshold = {
    "train": Decimal("0.18"),
    "test": Decimal("0.1")  # 0.05
}

all_features = ["num_vertices", "num_edges", "degree", "clustering_coefficient", "eigenvectors", "chi_degree",
            "chi_lcc", "pagerank_out", "betweenness_centrality", "kcore"]

# "mean_chi_degree", "mean_chi_lcc",

features_indices = dict(zip(all_features, range(len(all_features))))

base_models_path = output_path / "models/"
