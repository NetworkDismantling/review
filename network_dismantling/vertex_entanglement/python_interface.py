from copy import deepcopy
from logging import Logger, getLogger
from typing import Union

import numpy as np
from graph_tool import Graph

from network_dismantling import dismantler_wrapper
from network_dismantling._sorters import dismantling_method
from network_dismantling.vertex_entanglement.VE import new_generate_beta

method_info = {
    "source": "https://github.com/Yiminghh/VertexEntanglement",
    "citation": "Huang, Y., Wang, H., Ren, XL. et al. Identifying key players in complex networks via network entanglement. Commun Phys 7, 19 (2024). https://doi.org/10.1038/s42005-023-01483-8",
}


def to_networkx(g: Graph,
                return_mapping=False,
                logger: Logger = getLogger("dummy")
                ):
    from io import BytesIO
    from networkx import read_graphml, relabel_nodes, Graph as nxGraph

    logger.debug("Converting graph to NetworkX")
    with BytesIO() as io_buffer:
        g.save(io_buffer, fmt='graphml')

        io_buffer.seek(0)

        try:
            gn: nxGraph = read_graphml(io_buffer, node_type=str)
        except Exception as e:
            raise e

    # Map nodes to consecutive IDs to avoid issues with FINDER
    mapping = {k: i for i, k in enumerate(gn.nodes)}
    # reverse_mapping = {v: k for k, v in mapping.items()}
    gn = relabel_nodes(gn, mapping)

    if not return_mapping:
        return gn
    else:
        return gn, mapping


def VertexEnt(G: Graph, belta=None,
              perturb_strategy='default',
              printLog=False,
              logger: Logger = getLogger("dummy"),
              ):
    """
    近似计算的方法计算节点纠缠度
    :param G:
    :return:
    """
    # 邻接矩阵
    # nodelist: list, optional:
    # The rows and columns are ordered according to the nodes in nodelist.
    # If nodelist is None, then the ordering is produced by G.nodes().
    from graph_tool.spectral import adjacency
    from graph_tool.topology import label_components
    from tqdm.auto import tqdm

    assert not G.is_directed()
    A = adjacency(G).todense()
    # A = nx.adjacency_matrix(G, nodelist=sorted(G.nodes())).todense()

    # assert 0 in G.vertex_index, "Node 0 should be in the input graph!"
    assert np.allclose(A, A.T), "adjacency matrix should be symmetric"
    # 拉普拉斯矩阵
    L = np.diag(np.array(sum(A)).flatten()) - A
    # N = G.number_of_nodes()
    N = G.num_vertices()

    eigValue, eigVector = np.linalg.eigh(L)
    logger.debug("VE: done computing eigen values!")
    eigValue = eigValue.real
    eigVector = eigVector.real  # 每个特征向量一列

    # sort_idx = np.argsort(eigValue)
    # eigValue=eigValue[sort_idx]
    # eigVector=eigVector[:,sort_idx]

    if printLog:
        logger.debug(f"VE: eigValue:\n{eigValue}")

    if belta is None:
        # belta = generate_Belta(eigValue)

        # num_components = nx.number_connected_components(G)
        comp, _ = label_components(G)
        num_components = np.unique(comp.a).size

        logger.debug(f"VE There are {num_components} components.")
        belta = new_generate_beta(eigValue[num_components])

    S = np.zeros(len(belta))
    for i in range(0, len(belta)):
        b = belta[i]
        Z_ = np.sum(np.exp(eigValue * -b))
        t = -b * eigValue
        S[i] = -sum(np.exp(t) * (t / np.log(2) - np.log2(Z_)) / Z_)

    logger.debug(f"VE: done computing spectral entropy!")
    if printLog:
        print(S)

    lambda_ral = np.zeros((N, N))
    if perturb_strategy == 'default':
        for v_id in tqdm(range(0, N),
                         desc="Computing eigenValues",
                         unit="node",
                         ):
            # dA=np.zeros((N,N))
            # neibour = list(G.neighbors(v_id))
            neibour = G.get_out_neighbors(v_id).tolist()
            # kx = G.degree(v_id)
            kx = G.get_out_degrees([v_id]).item()
            A_loc = A[neibour][:, neibour]
            N_loc = kx + np.sum(A_loc) / 2
            weight = 2 * N_loc / kx / (kx + 1)
            if weight == 1:
                lambda_ral[v_id] = eigValue
            else:
                neibour.append(v_id)
                neibour = sorted(neibour)
                dA = weight - A[neibour][:, neibour]
                dA = dA - np.diag([weight] * (kx + 1))
                dL = np.diag(np.array(sum(dA)).flatten()) - dA
                for j in range(0, N):
                    t__ = eigVector[neibour, j].T @ dL @ eigVector[neibour, j]
                    if isinstance(t__, float):
                        lambda_ral[v_id, j] = eigValue[j] + t__
                    else:
                        lambda_ral[v_id, j] = eigValue[j] + t__[0, 0]

    elif perturb_strategy == 'remove':
        for v_id in tqdm(range(0, N),
                         desc="Computing eigenValues for removed networks",
                         unit="node",
                         position=0,
                         leave=True,
                         ):
            # neibour = list(G.neighbors(v_id))
            neibour = G.get_out_neighbors([v_id]).tolist()

            pt_A = deepcopy(A)
            pt_A[v_id, :] = 0
            pt_A[:, v_id] = 0
            dA = pt_A - A
            dA = dA[neibour][:, neibour]
            dL = np.diag(np.array(sum(dA)).flatten()) - dA
            for j in range(0, N):
                lambda_ral[v_id, j] = eigValue[j] + (eigVector[neibour, j].T @ dL @ eigVector[neibour, j])[0, 0]

    E = np.zeros((len(belta), N))
    for x in tqdm(range(0, N),
                  desc="Searching minium entanglement",
                  unit="node",
                  ):
        xl_ = lambda_ral[x, :]
        for i in range(0, len(belta)):
            b = belta[i]
            Z_ = np.sum(np.exp(-b * xl_))
            t = -b * xl_
            E[i, x] = -sum(np.exp(t) * (t / np.log(2) - np.log2(Z_)) / Z_) - S[i]

    VE = np.min(E, axis=0)
    mean_tau = np.mean(np.array(belta)[np.argmin(E, axis=0)])
    logger.debug(f"VE: mean_tau={mean_tau}")

    return VE


@dismantling_method(
    name=r"Vertex Entanglement",
    short_name=r"$\mathrm{VE}$",

    includes_reinsertion=False,
    plot_color="#34eb46",

    **method_info,
)
@dismantler_wrapper
def vertex_entanglement(network: Graph,
                        logger: Logger = getLogger("dummy"),
                        **kwargs):
    # from networkx import Graph as nxGraph
    # G: nxGraph
    # mapping: Dict[int, int]
    # G, mapping = to_networkx(network, return_mapping=True)
    # reverse_mapping: Dict[int, int] = {v: k for k, v in mapping.items()}

    # assert G.number_of_nodes() == network.num_vertices(), "Number of nodes mismatch"
    # assert G.number_of_edges() == network.num_edges(), "Number of edges mismatch"
    # assert set(G.nodes) == set(range(G.number_of_nodes())), "Nodes are not consecutive"

    # VE_cent: np.ndarray = VertexEnt(G)
    VE_cent: np.ndarray = VertexEnt(network)

    if not isinstance(VE_cent, np.ndarray):
        VE_cent = np.ndarray(VE_cent)

    # Convert to dismantling metric
    VE_cent = -VE_cent
    # # Map back to original node IDs
    # VE_cent = VE_cent[[int(reverse_mapping[i]) for i in G.nodes]]

    return VE_cent


@dismantling_method(
    name=r"Vertex Entanglement + Reinsertion",
    short_name=r"$\mathrm{VE}$ + R",

    includes_reinsertion=True,
    plot_color="#34eb46",

    depends_on=vertex_entanglement,

    **method_info,
)
@dismantler_wrapper
def vertex_entanglement_reinsertion(network: Graph,
                                    stop_condition: int,

                                    vertex_entanglement: Union[list, np.ndarray],

                                    logger: Logger = getLogger("dummy"),
                                    **kwargs):
    from network_dismantling.vertex_entanglement.reinsertion import reinsert

    predictions = reinsert(
        network=network,
        removals=vertex_entanglement,
        stop_condition=stop_condition,
        logger=logger,
    )

    return predictions
