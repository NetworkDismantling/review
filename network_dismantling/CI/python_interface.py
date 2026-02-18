import logging

from graph_tool import Graph

from network_dismantling import dismantler_wrapper
from network_dismantling._sorters import dismantling_method

# TODO use tempfile.NamedTemporaryFile?

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

    import tempfile
    import numpy as np

    from operator import itemgetter
    from os import remove, close
    from subprocess import run, CalledProcessError

    from graph_tool.all import remove_parallel_edges, remove_self_loops

    from network_dismantling.common.logging.pipe import LogPipe

    folder = f"network_dismantling/CI/"
    cd_cmd = f"cd {folder} && "

    remove_parallel_edges(network)
    remove_self_loops(network)

    # TODO map static node ids to CONTIGOUS ones
    static_id = network.vertex_properties["static_id"]
    node_id_mapping = {n: i for i, n in enumerate(static_id.a, start=1)}
    reverse_node_id_mapping = {v: k for k, v in node_id_mapping.items()}

    network_fd, network_path = tempfile.mkstemp()
    output_fd, output_path = tempfile.mkstemp()

    tmp_file_handles = [network_fd, output_fd]
    tmp_file_paths = [network_path, output_path]

    cmds = [
        "make clean",
        "make",
        f"./CI {network_path} {l} {stop_condition} {output_path}"
        # f'./CI {network_path} {l} {stop_condition / network.num_vertices()} {output_path}'
    ]

    nodes = []
    try:
        with open(network_fd, "w+") as tmp:
            # for node in network.vertices():
            for node, node_id in network.iter_vertices(
                    vprops=[network.vp["static_id"]]
            ):
                # node_id = static_id[node]
                node_id = node_id_mapping[node_id]
                tmp.write(f"{node_id}")

                for out_neighbor, out_neighbor_id in sorted(
                        network.iter_out_neighbors(node, vprops=[network.vp["static_id"]]),
                        key=itemgetter(1),
                ):
                    # out_neighbor_id = static_id[out_neighbor]
                    out_neighbor_id = node_id_mapping[out_neighbor_id]
                    tmp.write(f" {out_neighbor_id}")

                tmp.write("\n")

        with (
            LogPipe(logger=logger, level=logging.INFO) as stdout_pipe,
            LogPipe(logger=logger, level=logging.ERROR) as stderr_pipe,
        ):
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

        with open(output_fd, "r+") as tmp:
            for line in tmp.readlines():
                _, node = line.strip().split(" ")
                node = reverse_node_id_mapping[int(node)]

                nodes.append(node)

    except Exception as e:
        raise e

    finally:
        for fd, path in zip(tmp_file_handles, tmp_file_paths):
            try:
                close(fd)

            except:
                pass

            try:
                remove(path)

            except:
                pass

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
