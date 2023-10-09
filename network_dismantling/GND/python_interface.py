import logging
import tempfile
from os import remove, close
from subprocess import check_output, STDOUT

import numpy as np

from network_dismantling import dismantler_wrapper
from network_dismantling._sorters import dismantling_method

folder = "network_dismantling/GND/"
cd_cmd = f"cd {folder} && "
executable = "GND"
reinsertion_executable = "reinsertion"

# TODO use tempfile.NamedTemporaryFile?
# TODO use logger instead of print


def _generalized_network_dismantling(
    network,
    stop_condition: int,
    reinsertion=False,
    remove_strategy=3,
    reinsertion_strategy=2,
    logger=logging.getLogger("dummy"),
    **kwargs,
):
    static_id = network.vertex_properties["static_id"]

    network_fd, network_path = tempfile.mkstemp()
    broken_fd, broken_path = tempfile.mkstemp()
    output_fd, output_path = tempfile.mkstemp()

    tmp_file_handles = [network_fd, broken_fd, output_fd]
    tmp_file_paths = [network_path, broken_path, output_path]

    nodes = []

    try:
        with open(network_fd, "w+") as tmp:
            for edge in network.edges():
                tmp.write(
                    "{} {}\n".format(
                        static_id[edge.source()] + 1, static_id[edge.target()] + 1
                    )
                )

        cmds = [
            # TODO move build to setup.py?
            # 'make clean && make',
            "make",
            f"./{executable} "
            f"--NetworkFile {network_path} "
            f'--IDFile "{broken_path}" '
            f"--NodeNum {network.num_vertices()} "
            f"--TargetSize {stop_condition} "
            f"--RemoveStrategy {remove_strategy} ",
        ]

        if reinsertion is True:
            cmds.append(
                f"./{reinsertion_executable} "
                f"--NetworkFile {network_path} "
                f'--IDFile "{broken_path}" '
                f'--OutFile "{output_path}" '
                f"--TargetSize {stop_condition} "
                f"--SortStrategy {reinsertion_strategy} "
            )

            output = output_fd
        else:
            output = broken_fd

        for cmd in cmds:
            try:
                logger.debug(f"Running cmd: {cmd}")
                logger.debug(
                    check_output(
                        cd_cmd + cmd,
                        shell=True,
                        text=True,
                        stderr=STDOUT,
                    )
                )
            except Exception as e:
                raise RuntimeError(f"ERROR! When running cmd: {cmd} {e}")

        with open(output, "r+") as tmp:
            for line in tmp.readlines():
                node = line.strip()

                nodes.append(node)

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
        output[int(n) - 1] = p

    return output


method_info = {
    "source": "https://github.com/renxiaolong/Generalized-Network-Dismantling",
}


@dismantling_method(
    name="Generalized Network Dismantling",
    # display_name="GND",
    short_name="GND",
    plot_color="#ffbb78",
    includes_reinsertion=False,
    **method_info,
)
@dismantler_wrapper
def GND(network, **kwargs):
    return _generalized_network_dismantling(network, reinsertion=False, **kwargs)


@dismantling_method(
    name="Generalized Network Dismantling + Reinsertion",
    # display_name="GND +R",
    short_name="GND +R",
    plot_color="#ff7f0e",
    # plot_marker="s",
    includes_reinsertion=True,
    **method_info,
)
@dismantler_wrapper
def GNDR(network, **kwargs):
    return _generalized_network_dismantling(network, reinsertion=True, **kwargs)
