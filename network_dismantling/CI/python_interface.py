import logging
from pathlib import Path

from graph_tool import Graph

from network_dismantling import dismantler_wrapper
from network_dismantling._sorters import dismantling_method

_CI_DIR = Path(__file__).resolve().parent


# @dismantler_wrapper
def _collective_influence_l(network: Graph,
                            l: int,
                            stop_condition: int,
                            logger: logging.Logger = logging.getLogger("dummy"),
                            **kwargs):
    """
    Implements interface to Collective Influence.
    This function merges the GetSolution and EvaluateSolution functions.
    Note that the default parameters are the same as provided in the author's code.

    :param network:
    :param l:
    :param stop_condition:
    :param logger:
    :param kwargs:
    :return:
    """

    import numpy as np

    from operator import itemgetter
    from subprocess import run, CalledProcessError
    from tempfile import NamedTemporaryFile

    from graph_tool.all import remove_parallel_edges, remove_self_loops

    from network_dismantling.common.logging.pipe import LogPipe

    cd_cmd = f"cd {_CI_DIR} && "

    remove_parallel_edges(network)
    remove_self_loops(network)

    # Map static node ids to contiguous 1-based ids (required by CI binary)
    static_id = network.vertex_properties["static_id"]
    node_id_mapping = {n: i for i, n in enumerate(static_id.a, start=1)}
    reverse_node_id_mapping = {v: k for k, v in node_id_mapping.items()}

    nodes = []

    with (
        NamedTemporaryFile("w+", suffix=".ci_net") as network_f,
        NamedTemporaryFile("r+", suffix=".ci_out") as output_f,
        LogPipe(logger=logger, level=logging.INFO) as stdout_pipe,
        LogPipe(logger=logger, level=logging.ERROR) as stderr_pipe,
    ):
        # Write adjacency list
        for node, node_id in network.iter_vertices(
                vprops=[network.vp["static_id"]]
        ):
            node_id = node_id_mapping[node_id]
            network_f.write(f"{node_id}")

            for out_neighbor, out_neighbor_id in sorted(
                    network.iter_out_neighbors(node, vprops=[network.vp["static_id"]]),
                    key=itemgetter(1),
            ):
                out_neighbor_id = node_id_mapping[out_neighbor_id]
                network_f.write(f" {out_neighbor_id}")

            network_f.write("\n")

        network_f.flush()

        cmds = [
            "make clean",
            "make",
            f"./CI {network_f.name} {l} {stop_condition} {output_f.name}",
        ]

        for cmd in cmds:
            try:
                logger.debug(f"Running: {cd_cmd + cmd}")
                run(
                    cd_cmd + cmd,
                    shell=True,
                    stdout=stdout_pipe,
                    stderr=stderr_pipe,
                    text=True,
                    check=True,
                )
            except CalledProcessError as e:
                raise RuntimeError(f"CI binary failed on cmd '{cmd}': {e}") from e

        # Read output
        output_f.seek(0)
        for line in output_f:
            _, node = line.strip().split(" ")
            node = reverse_node_id_mapping[int(node)]
            nodes.append(node)

    output = np.zeros(network.num_vertices())
    for n, p in zip(nodes, list(reversed(range(1, len(nodes) + 1)))):
        output[n] = p

    return output


method_info = {
    # "name": "Collective Influence",
    # "short_name": "CI",
    "source": "https://github.com/makselab/Collective-Influence",
    # "authors": "",
    # "citation": "",
    "includes_reinsertion": True,
}


@dismantling_method(
    name=r"Collective Influence $\ell-1$",
    # display_name="GND",
    short_name=r"CI $\ell-1$",
    plot_color="#eb6434",
    **method_info,
)
@dismantler_wrapper
def CollectiveInfluenceL1(network, stop_condition, **kwargs):
    return _collective_influence_l(
        network, l=1, stop_condition=stop_condition, **kwargs
    )


@dismantling_method(
    name=r"Collective Influence $\ell-2$",
    # display_name="GND",
    short_name=r"CI $\ell-2$",
    plot_color="#d62728",
    **method_info,
)
@dismantler_wrapper
def CollectiveInfluenceL2(network, stop_condition, **kwargs):
    return _collective_influence_l(
        network, l=2, stop_condition=stop_condition, **kwargs
    )


@dismantling_method(
    name=r"Collective Influence $\ell-3$",
    # display_name="GND",
    short_name=r"CI $\ell-3$",
    plot_color="#eb3434",
    **method_info,
)
@dismantler_wrapper
def CollectiveInfluenceL3(network, stop_condition, **kwargs):
    return _collective_influence_l(
        network, l=3, stop_condition=stop_condition, **kwargs
    )
