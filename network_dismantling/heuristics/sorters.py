import numpy as np


def get_degree(network):
    degree = network.get_out_degrees(network.get_vertices())
    max_degree = np.max(degree)

    degree = np.divide(degree, max_degree)

    return degree


def get_eigenvector_centrality(network):
    from graph_tool.centrality import eigenvector

    _, eigenvectors = eigenvector(network)

    return eigenvectors.get_array()


def get_pagerank(network):
    from graph_tool.centrality import pagerank

    return pagerank(network).get_array()


def get_betweenness_centrality(network):
    from graph_tool.centrality import betweenness

    betweenness_out, _ = betweenness(network)

    return betweenness_out.get_array()


def get_random(network):
    return np.random.rand(network.num_vertices())


__all__ = [
    (name.replace("get_", ""), thing)
    for (name, thing) in locals().items()
    if callable(thing) and thing.__module__ == __name__ and thing.__name__[0] != "_"
]
__all_dict__ = dict(__all__)
