import os
import sys
import tempfile
from operator import itemgetter
from subprocess import check_output, STDOUT

import numpy as np
from graph_tool import Graph

from network_dismantling._sorters import dismantling_method

local_dir = os.path.dirname(__file__) + os.sep

sys.path.append(local_dir)


def _collective_influence_l(network: Graph, l, stop_condition, **kwargs):
    """
    Implements interface to Collective Influence.
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

    folder = f'network_dismantling/CI/'
    cd_cmd = f'cd {folder} && '

    from graph_tool.stats import remove_parallel_edges, remove_self_loops

    remove_parallel_edges(network)
    remove_self_loops(network)

    # TODO map static node ids to CONTIGOUS ones
    static_id = network.vertex_properties["static_id"]
    node_id_mapping = {n: i for i, n in enumerate(static_id.a, start=1)}
    reverse_node_id_mapping = {v: k for k, v in node_id_mapping.items()}

    network_fd, network_path = tempfile.mkstemp()
    output_fd, output_path = tempfile.mkstemp()

    cmds = [
        'make clean',
        'make',
        f'./CI {network_path} {l} {stop_condition} {output_path}'
        # f'./CI {network_path} {l} {stop_condition / network.num_vertices()} {output_path}'
    ]

    try:
        with open(network_fd, 'w+') as tmp:
            # for node in network.vertices():
            for node, node_id in network.iter_vertices(vprops=[network.vp["static_id"]]):
                # node_id = static_id[node]
                node_id = node_id_mapping[node_id]
                tmp.write(f"{node_id}")

                for out_neighbor, out_neighbor_id in sorted(
                        network.iter_out_neighbors(node, vprops=[network.vp["static_id"]]), key=itemgetter(1)):
                    # out_neighbor_id = static_id[out_neighbor]
                    out_neighbor_id = node_id_mapping[out_neighbor_id]
                    tmp.write(f" {out_neighbor_id}")

                tmp.write("\n")

        for cmd in cmds:
            try:
                print(f"Running cmd: {cmd}")
                print(
                    check_output(cd_cmd + cmd,
                                 shell=True,
                                 text=True,
                                 stderr=STDOUT,
                                 )
                )
            except Exception as e:
                raise RuntimeError(f"ERROR! When running cmd: {cmd}. {e}")

        nodes = []

        with open(output_fd, 'r+') as tmp:
            for line in tmp.readlines():
                _, node = line.strip().split(" ")
                node = reverse_node_id_mapping[int(node)]

                nodes.append(node)

    except Exception as e:
        raise e

    finally:
        os.remove(network_path)
        os.remove(output_path)

    output = np.zeros(network.num_vertices())
    for n, p in zip(nodes, list(reversed(range(1, len(nodes) + 1)))):
        output[n] = p

    return output

method_info = {
    # "name": "Collective Influence",
    # "short_name": "CI",
    "source": "https://github.com/makselab/Collective-Influence",
    "authors": "",
    "citation": "",
    "includes_reinsertion": True,
}

@dismantling_method(**method_info)
def CollectiveInfluenceL1(network, stop_condition, **kwargs):
    return _collective_influence_l(network, l=1, stop_condition=stop_condition, **kwargs)


@dismantling_method(**method_info)
def CollectiveInfluenceL2(network, stop_condition, **kwargs):
    return _collective_influence_l(network, l=2, stop_condition=stop_condition, **kwargs)


@dismantling_method(**method_info)
def CollectiveInfluenceL3(network, stop_condition, **kwargs):
    return _collective_influence_l(network, l=3, stop_condition=stop_condition, **kwargs)
