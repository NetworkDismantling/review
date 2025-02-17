import logging
from subprocess import run
from tempfile import NamedTemporaryFile
from typing import List

from graph_tool.all import Graph
from network_dismantling import dismantler_wrapper
from network_dismantling._sorters import dismantling_method
from network_dismantling.common.logging.pipe import LogPipe

cd_cmd = "cd network_dismantling/decycler/ && "
executable = "decycler"
reverse_greedy_executable = "reverse-greedy"


# TODO avoid re-running the same code for MS+R and use the dependency system

def _decycler(network: Graph,
              stop_condition: int,
              reinsertion: bool,
              logger: logging.Logger = logging.getLogger("dummy"),
              **kwargs):
    import numpy as np

    network_type = "D" if network.is_directed() else "E"

    static_id = network.vertex_properties["static_id"]

    nodes = []

    with (
        NamedTemporaryFile("w+",
                           # delete=False,
                           ) as network_fd,
        NamedTemporaryFile("w+",
                           # delete=False,
                           ) as seeds_fd,
        NamedTemporaryFile("w+",
                           # delete=False,
                           ) as broken_fd,
        NamedTemporaryFile("w+",
                           # delete=False,
                           ) as output_fd,
        LogPipe(logger=logger,
                level=logging.INFO,
                ) as stdout_pipe,
        LogPipe(logger=logger,
                level=logging.ERROR,
                ) as stderr_pipe
    ):

        network_path = network_fd.name
        seeds_path = seeds_fd.name
        broken_path = broken_fd.name
        output_path = output_fd.name

        for edge in network.edges():
            if edge.source() != edge.target():
                network_fd.write(
                    f"{network_type} "
                    f"{static_id[edge.source()] + 1} "
                    f"{static_id[edge.target()] + 1}\n"
                )
        network_fd.flush()

        cmds = [
            "make",

            f"cat {network_path} | ./{executable} -o > {seeds_path}",

            f"(cat {network_path} {seeds_path}) | "
            f"python treebreaker.py {stop_condition} > {broken_path}",
        ]

        output: List[NamedTemporaryFile]
        if reinsertion is True:
            cmds.append(
                f"(cat {network_path} {seeds_path} {broken_path}) | "
                f"./{reverse_greedy_executable} --threshold {stop_condition} > {output_path}"
            )

            output = [output_fd]
        else:
            output = [seeds_fd, broken_fd]

        for cmd in cmds:
            try:
                logger.debug(f"Running command: {cd_cmd + cmd}")
                run(cd_cmd + cmd,
                    shell=True,
                    stdout=stdout_pipe,
                    stderr=stderr_pipe,
                    text=True,
                    check=True,
                    )

            # except CalledProcessError as e:
            #     logger.error(f"ERROR while running cmd {cmd}: {e}", exc_info=True)
            #     raise e

            except Exception as e:
                logger.error(f"ERROR while running cmd {cmd}: {e}", exc_info=True)

                # raise RuntimeError("ERROR! {}".format(e))
                raise e

        output_fd.seek(0)
        logger.debug(f"Output: {output_fd.readlines()}")

        # Iterate over seeds
        for tmp_file in output:
            tmp_file.seek(0)

            for line in tmp_file.readlines():
                line = line.strip().split(" ")

                node_type, seed = line[0], line[1]

                if node_type != "S":
                    continue
                    # raise ValueError("Unexpected output: {}".format(line))

                nodes.append(seed)

    output = np.zeros(network.num_vertices())

    for n, p in zip(nodes, list(reversed(range(1, len(nodes) + 1)))):
        output[int(n) - 1] = p

    return output


method_info = {
    # "name": "MinSum",
    # "description": "MinSum",
    # "module": "decycler",
    "source": "https://github.com/abraunst/decycler",
}


@dismantling_method(
    name="Min-Sum",
    short_name="MS",
    plot_color="#98df8a",
    includes_reinsertion=False,
    **method_info,
)
@dismantler_wrapper
def MS(network, **kwargs):
    return _decycler(network, reinsertion=False, **kwargs)


@dismantling_method(
    name="Min-Sum + Reinsertion",
    short_name="MS +R",
    plot_color="#2ca02c",
    includes_reinsertion=True,
    **method_info,
)
@dismantler_wrapper
def MSR(network, **kwargs):
    return _decycler(network, reinsertion=True, **kwargs)
