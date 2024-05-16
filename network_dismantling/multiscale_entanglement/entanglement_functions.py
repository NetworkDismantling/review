import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from functools import partial
from logging import Logger, getLogger

import numpy as np
from graph_tool import Graph, VertexPropertyMap
from graph_tool.spectral import laplacian
from scipy.linalg import eigvalsh
from scipy.sparse import csr_matrix

# Beta values
beta_small = 0.9
beta_mid = 0.33
beta_large = 0.01

# Float zero threshold
zero_threshold = 10 ** -12
p_threshold = 10 ** -20


def get_sorted_eigvals(G: Graph):
    # Ls = np.sort(nx.laplacian_spectrum(G))
    L: csr_matrix = laplacian(G)
    L: np.matrix = L.toarray()
    eigvals = eigvalsh(L)

    Ls = np.sort(eigvals)

    return Ls


def get_first_non_zero_laplacian_eig(Ls: np.ndarray,
                                     zero_threshold: float = zero_threshold,
                                     ):
    return Ls[np.where(Ls > zero_threshold)[0]][0]


# def entropy_diff_time_small(G: Graph,
#                             beta=0.9,
#                             zero_threshold=zero_threshold,
#                             ):
#     Ls = get_sorted_eigvals(G)
#     diff_time = get_first_non_zero_laplacian_eig(Ls,
#                                                  zero_threshold=zero_threshold,
#                                                  )
#     # k_ = len(G.edges()) / len(G.nodes())
#     # # max_S = np.log(len(G.nodes))/100
#     # # N = np.log(len(G.nodes()))
#     # # n_ = np.log(len(G.nodes))
#     beta = -np.log(beta) / diff_time
#     # # beta = -np.log(10/(len(G.edges())/len(G.nodes())))/diff_time
#     # # beta = (diff_time + k_)/2
#     # print('beta = ', beta)
#     sp = np.exp(-beta * Ls)
#     Z = np.sum(sp)
#     p = np.exp(-beta * Ls) / Z
#     p = p[p > 10 ** -20]
#     S = np.sum(-p * np.log2(p))
#     return S, beta
#
#
# def entropy_diff_time_mid(G,
#                           beta=0.33,
#                           ):
#     Ls = np.sort(nx.laplacian_spectrum(G))
#     diff_time = Ls[np.where(Ls > 10 ** -12)[0]][0]
#     # k_ = len(G.edges()) / len(G.nodes())
#     # # max_S = np.log(len(G.nodes))/100
#     # # N = np.log(len(G.nodes()))
#     # # n_ = np.log(len(G.nodes))
#     beta = -np.log(.33) / diff_time
#     # beta = -np.log(10/(len(G.edges())/len(G.nodes())))/diff_time
#     # beta = (diff_time + k_)/2
#     print('beta = ', beta)
#     sp = np.exp(-beta * Ls)
#     Z = np.sum(sp)
#     p = np.exp(-beta * Ls) / Z
#     p = p[p > 10 ** -20]
#     S = np.sum(-p * np.log2(p))
#     return S, beta
#
#
# def entropy_diff_time_large(G,
#                             beta=0.01,
#                             ):
#     Ls = np.sort(nx.laplacian_spectrum(G))
#     diff_time = Ls[np.where(Ls > 10 ** -12)[0]][0]
#     k_ = len(G.edges()) / len(G.nodes())
#     beta = -np.log(beta) / diff_time
#     print('beta = ', beta)
#     sp = np.exp(-beta * Ls)
#     Z = np.sum(sp)
#     p = np.exp(-beta * Ls) / Z
#     p = p[p > 10 ** -20]
#     S = np.sum(-p * np.log2(p))
#     return S, beta


def entropy_diff_time(G: Graph,
                      beta: float,
                      p_threshold: float = p_threshold,
                      zero_threshold: float = zero_threshold,
                      logger: Logger = getLogger("dummy"),
                      **kwargs) -> tuple[float, float]:
    Ls = get_sorted_eigvals(G)
    diff_time = get_first_non_zero_laplacian_eig(Ls,
                                                 zero_threshold=zero_threshold,
                                                 )

    beta = -np.log(beta) / diff_time

    logger.debug(f"beta = {beta}")

    S = get_entropy(Ls, beta, p_threshold=p_threshold)

    return S, beta


def get_entropy(Ls: np.ndarray,
                beta: float,
                p_threshold=p_threshold,
                ) -> float:
    sp = np.exp(-beta * Ls)

    Z = np.sum(sp)
    p = np.exp(-beta * Ls) / Z
    p = p[p > p_threshold]
    S = np.sum(-p * np.log2(p))

    return S


def entropy(G: Graph,
            beta: float,
            ):
    Ls = get_sorted_eigvals(G)

    return get_entropy(Ls, beta)


# def entanglement_small(G):
#     S_1, beta = entropy_diff_time_small(G)
#     ent = {}
#     nodes = list(G.nodes())
#     # for i in pbar(range(len(nodes))):
#
#     for i in range(len(nodes)):
#         # for i in range(len(nodes)):
#         G_i = G.copy()
#         k = G_i.degree[nodes[i]]
#         G_i.remove_node(nodes[i])
#         S_2 = entropy(G_i, beta)
#         G_star = nx.star_graph(k + 1)
#         S_star = entropy(G_star, beta)
#
#         S_2 = S_2 + S_star
#         ent[nodes[i]] = S_2 - S_1
#     return ent
#
#
# def entanglement_mid(G):
#     # pbar = ProgressBar()
#     S_1, beta = entropy_diff_time_mid(G)
#     ent = {}
#     nodes = list(G.nodes())
#     for i in range(len(nodes)):
#         # for i in range(len(nodes)):
#         G_i = G.copy()
#         k = G_i.degree[nodes[i]]
#         G_i.remove_node(nodes[i])
#         S_2 = entropy(G_i, beta)
#         G_star = nx.star_graph(k + 1)
#         S_star = entropy(G_star, beta)
#
#         S_2 = S_2 + S_star
#         ent[nodes[i]] = S_2 - S_1
#     return ent
#
#
# def entanglement_large(G):
#     S_1, beta = entropy_diff_time_large(G)
#     ent = {}
#     nodes = list(G.nodes())
#     for i in range(len(nodes)):
#         # for i in range(len(nodes)):
#         G_i = G.copy()
#         k = G_i.degree[nodes[i]]
#         G_i.remove_node(nodes[i])
#         S_2 = entropy(G_i, beta)
#         G_star = nx.star_graph(k + 1)
#         S_star = entropy(G_star, beta)
#
#         S_2 = S_2 + S_star
#         ent[nodes[i]] = S_2 - S_1
#     return ent


def compute_entropy_value(
        i: int,
        G: Graph,
        beta: float,
        degree_property: VertexPropertyMap,
):
    k = degree_property[i]

    G_i: Graph = deepcopy(G)
    G_i.set_fast_edge_removal(True)

    G_i.remove_vertex(i,
                      fast=True,
                      )

    S_2 = entropy(G_i, beta)

    G_star = star_graph(k + 1)
    S_star = entropy(G_star, beta)

    S_2 = S_2 + S_star

    return S_2


def star_graph(k: int) -> Graph:
    G: Graph = Graph(directed=False)

    # Add the central node
    G.add_vertex(k + 1)

    # Add the star edges
    for i in range(1, k + 1):
        G.add_edge(0, i)

    return G


def entanglement(G: Graph,
                 beta: float,
                 pool_size: int = None,
                 mp_manager: multiprocessing.Manager = multiprocessing.get_context("spawn"),
                 executor: ProcessPoolExecutor = None,
                 chunksize: int = 16,
                 logger: Logger = getLogger("dummy"),
                 **kwargs
                 ) -> VertexPropertyMap:
    logger.debug("Kwargs: %s", kwargs)
    S_1, beta = entropy_diff_time(G=G,
                                  beta=beta,
                                  logger=logger,
                                  )
    degree_property = G.degree_property_map("out")

    partial_compute_entropy_value = partial(compute_entropy_value,
                                            G=G,
                                            beta=beta,
                                            degree_property=degree_property,
                                            # **kwargs,
                                            )

    entropy_values = list(  # tqdm(
        executor.map(partial_compute_entropy_value,
                     list(G.iter_vertices()),
                     chunksize=chunksize,
                     ),
        # desc="Computing entanglement",
        # total=G.num_vertices(),
        # )
    )

    entropy_values = np.array(entropy_values)
    entropy_values -= S_1

    entropy_property = G.new_vertex_property("double",
                                             vals=entropy_values,
                                             )
    return entropy_property.get_array()


def entanglement_small(G: Graph,
                       beta: float = beta_small,
                       ):
    return entanglement(G, beta)


def entanglement_mid(G: Graph,
                     beta: float = beta_mid,
                     ):
    return entanglement(G, beta)


def entanglement_large(G: Graph,
                       beta: float = beta_large,
                       ):
    return entanglement(G, beta)
