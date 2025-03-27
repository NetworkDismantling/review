# import faulthandler
import logging
import os
import sys
from pathlib import Path
from typing import Union

from graph_tool import Graph

from network_dismantling import dismantler_wrapper
from network_dismantling.FINDER_ND.FINDER import FINDER
from network_dismantling._sorters import dismantling_method

local_dir = os.path.dirname(__file__) + os.sep

sys.path.append(local_dir)

model_file_path = local_dir + 'models/'
model_file_path = Path(model_file_path)
model_file_path = model_file_path.resolve()

if not model_file_path.exists():
    raise FileNotFoundError(f"Model file path {model_file_path} does not exist")
elif not model_file_path.is_dir():
    raise NotADirectoryError(f"Model file path {model_file_path} is not a directory")

dqn = None


def to_networkx(g: Graph):
    from io import BytesIO
    from networkx import read_graphml, relabel_nodes

    print("Converting graph to NetworkX")
    with BytesIO() as io_buffer:
        g.save(io_buffer, fmt='graphml')

        io_buffer.seek(0)

        try:
            gn = read_graphml(io_buffer, node_type=str)
        except Exception as e:
            raise e

    # Map nodes to consecutive IDs to avoid issues with FINDER
    mapping = {k: i for i, k in enumerate(gn.nodes)}

    gn = relabel_nodes(gn, mapping)

    return gn


@dismantler_wrapper
def _finder_nd(network: Graph, reinsertion=True, strategy_id=0,
               model_file_ckpt: Union[str, Path] = 'nrange_30_50_iter_78000.ckpt',
               step_ratio=0.01, reinsert_step=0.001,
               logger=logging.getLogger("dummy"), **kwargs
               ):
    """
    Implements interface to FINDER ND (no cost).
    This function merges the GetSolution and EvaluateSolution functions.
    Note that the default parameters are the same as provided in the author's code.

    :param network:
    :param reinsertion:
    :param model_file_ckpt:
    :param strategy_id:
    :param step_ratio:
    :param reinsert_step:
    :param kwargs:
    :return:
    """
    global dqn

    dqn = FINDER()

    from graph_tool.all import remove_parallel_edges, remove_self_loops
    import time
    import networkx as nx
    import numpy as np

    remove_parallel_edges(network)
    remove_self_loops(network)

    # # Convert the network to NetworkX Graph
    nx_graph = to_networkx(network)

    print("Getting static ids")
    static_id = nx.get_node_attributes(nx_graph, "static_id")

    # GetSolution BEGIN
    model_file = model_file_path / model_file_ckpt

    print("Loading model")
    print('The best model is :%s' % (model_file))
    dqn.LoadModel(model_file)

    print("Getting solution")
    solution, solution_time = dqn.EvaluateRealData(g=nx_graph,
                                                   # model_file=model_file,
                                                   # data_test=data_test,
                                                   # save_dir=save_dir,
                                                   stepRatio=step_ratio,
                                                   )

    # GetSolution END
    # print("Done getting solution")
    if reinsertion is True:
        print("Reinserting nodes")
        # EvaluateSolution BEGIN
        t1 = time.time()
        # strategyID: 0:no insert; 1:count; 2:rank; 3:multiply
        # This function returns the solution after the reinsertion steps
        reinsert_solution, Robustness, MaxCCList = dqn.EvaluateSol(g=nx_graph,
                                                                   # data_test=data_test,
                                                                   # sol_file=solution,
                                                                   solution=solution,
                                                                   strategyID=strategy_id,
                                                                   reInsertStep=reinsert_step,
                                                                   )
        t2 = time.time()

        solution_time = t2 - t1

        solution = reinsert_solution

        # EvaluateSolution END

    # print("Done reinserting nodes")
    output = np.zeros(network.num_vertices())

    for n, p in zip(solution, list(reversed(range(1, len(solution))))):
        # output[static_id[f"n{n}"]] = p
        # assert static_id[n] == n, f"static_id[n] = {static_id[n]} != n = {n}"
        output[static_id[n]] = p

    # print("Done ordering nodes")

    return output


method_info = {
    "source": "https://github.com/FFrankyy/FINDER/tree/master/code/FINDER_ND",
}


@dismantling_method(name="FINDER ND",
                    short_name="FINDER",

                    includes_reinsertion=False,
                    plot_color="#9467bd",
                    **method_info)
def FINDER_ND(network, **kwargs):
    return _finder_nd(network, reinsertion=True, **kwargs)
