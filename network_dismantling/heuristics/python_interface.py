#   This file is part of the Network Dismantling review,
#   proposed in the paper "Robustness and resilience of complex networks"
#   by Oriol Artime, Marco Grassia, Manlio De Domenico, James P. Gleeson,
#   Hernán A. Makse, Giuseppe Mangioni, Matjaž Perc and Filippo Radicchi.
#
#   This is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   The project is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with the code.  If not, see <http://www.gnu.org/licenses/>.

"""Auto-register node-metric heuristics as dismantling methods.

Each scoring function from :mod:`~network_dismantling.heuristics.sorters` is
wrapped with :func:`~network_dismantling.common.dismantlers.dismantler_wrapper`
and registered via :func:`~network_dismantling._sorters.dismantling_method` so
that they are automatically available in the main CLI (``dismantler.py``).

Both **static** (scored once, then removed in order) and **dynamic** (re-scored
after every removal) variants are registered where applicable.
"""

from graph_tool import Graph

from network_dismantling._sorters import dismantling_method
from network_dismantling.common.dismantlers import dismantler_wrapper

from network_dismantling.heuristics.sorters import (
    get_degree,
    get_eigenvector_centrality,
    get_pagerank,
    get_betweenness_centrality,
    get_random,
)

# ---------------------------------------------------------------------------
# Static heuristics (scored once, then removed by decreasing score)
# ---------------------------------------------------------------------------

@dismantling_method(
    name="Degree",
    short_name="Degree",
    method_type="heuristic",
    description="Remove nodes by decreasing normalised degree.",
)
@dismantler_wrapper
def degree(network: Graph, **kwargs):
    return get_degree(network)


@dismantling_method(
    name="Eigenvector Centrality",
    short_name="EigC",
    method_type="heuristic",
    description="Remove nodes by decreasing eigenvector centrality.",
)
@dismantler_wrapper
def eigenvector_centrality(network: Graph, **kwargs):
    return get_eigenvector_centrality(network)


@dismantling_method(
    name="PageRank",
    short_name="PR",
    method_type="heuristic",
    description="Remove nodes by decreasing PageRank score.",
)
@dismantler_wrapper
def pagerank(network: Graph, **kwargs):
    return get_pagerank(network)


@dismantling_method(
    name="Betweenness Centrality",
    short_name="BC",
    method_type="heuristic",
    description="Remove nodes by decreasing betweenness centrality.",
)
@dismantler_wrapper
def betweenness_centrality(network: Graph, **kwargs):
    return get_betweenness_centrality(network)


@dismantling_method(
    name="Random",
    short_name="Rnd",
    method_type="heuristic",
    description="Remove nodes in random order (baseline).",
)
@dismantler_wrapper
def random(network: Graph, **kwargs):
    return get_random(network)
