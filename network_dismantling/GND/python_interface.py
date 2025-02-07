import logging
from os import remove, close
from subprocess import check_output, STDOUT, CalledProcessError, run
from tempfile import mkstemp, NamedTemporaryFile
from typing import Union

import numpy as np
from graph_tool import Graph
from network_dismantling import dismantler_wrapper
from network_dismantling._sorters import dismantling_method
from network_dismantling.common.logging.pipe import LogPipe

folder = "network_dismantling/GND/"
cd_cmd = f"cd {folder} && "
executable = "GND"
reinsertion_executable = "reinsertion"


# TODO use NamedTemporaryFile?


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

    network_fd, network_path = mkstemp()
    broken_fd, broken_path = mkstemp()
    output_fd, output_path = mkstemp()

    tmp_file_handles = [network_fd, broken_fd, output_fd]
    tmp_file_paths = [network_path, broken_path, output_path]

    nodes = []

    assert network.num_vertices() == len(static_id), "Number of vertices mismatch"
    assert network.num_vertices() == static_id.a.max() + 1, "Node IDs are not consecutive"
    try:
        with open(network_fd, "w+") as tmp:
            for edge in network.edges():
                tmp.write(f"{static_id[edge.source()] + 1} {static_id[edge.target()] + 1}\n")

            tmp.flush()

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
                logger.exception(f"ERROR! When running cmd: {cmd}", exc_info=True)
                raise e

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
def GND(network: Graph,
        stop_condition: int,
        # reinsertion=False,
        remove_strategy: int = 3,
        # reinsertion_strategy: int=2,
        logger=logging.getLogger("dummy"),
        **kwargs,
        ):
    static_id = network.vertex_properties["static_id"]

    # network_fd, network_path = mkstemp()
    # broken_fd, broken_path = mkstemp()
    # output_fd, output_path = mkstemp()
    #
    # tmp_file_handles = [network_fd, broken_fd, output_fd]
    # tmp_file_paths = [network_path, broken_path, output_path]

    nodes = []
    output = np.zeros(network.num_vertices())

    with (
        NamedTemporaryFile("w+") as network_fd,
        NamedTemporaryFile("r+") as broken_fd,
        LogPipe(logger=logger,
                level=logging.INFO,
                ) as stdout_pipe,
        LogPipe(logger=logger,
                level=logging.ERROR,
                ) as stderr_pipe
    ):
        network_path = network_fd.name
        broken_path = broken_fd.name

        for edge in network.edges():
            network_fd.write(
                f"{static_id[edge.source()] + 1} {static_id[edge.target()] + 1}\n"
            )

        network_fd.flush()

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
            except CalledProcessError as e:
                logger.error(f"ERROR while running reinsertion algorithm: {e}", exc_info=True)
                raise RuntimeError(f"ERROR! {e}")
            except Exception as e:
                raise RuntimeError("ERROR! {}".format(e))

        broken_fd.seek(0)
        for line in broken_fd.readlines():
            node = line.strip()

            nodes.append(node)

        if len(nodes) < 0:
            raise RuntimeError("No removals found!")

    for n, p in zip(nodes, list(reversed(range(1, len(nodes) + 1)))):
        output[int(n) - 1] = p

    return output


@dismantling_method(
    name="Generalized Network Dismantling + Reinsertion",
    # display_name="GND +R",
    short_name="GND +R",
    depends_on=GND,
    plot_color="#ff7f0e",
    # plot_marker="s",
    includes_reinsertion=True,
    **method_info,
)
@dismantler_wrapper
def GNDR(network: Graph,
         stop_condition: int,
         GND: Union[list, np.ndarray],
         remove_strategy: int = 3,
         reinsertion_strategy: int = 2,

         logger=logging.getLogger("dummy"),
         **kwargs,
         ):
    removals = GND

    static_id = network.vertex_properties["static_id"]

    nodes = []
    output = np.zeros(network.num_vertices(),
                      dtype=int,
                      )
    with (
        NamedTemporaryFile("w+") as network_fd,
        NamedTemporaryFile("w+") as broken_fd,
        NamedTemporaryFile("r+") as output_fd,
        LogPipe(logger=logger,
                level=logging.INFO,
                ) as stdout_pipe,
        LogPipe(logger=logger,
                level=logging.ERROR,
                ) as stderr_pipe
    ):
        network_path = network_fd.name
        broken_path = broken_fd.name
        output_path = output_fd.name

        for edge in network.edges():
            network_fd.write(
                f"{static_id[edge.source()] + 1} {static_id[edge.target()] + 1}\n"
            )

        network_fd.flush()

        for removal in removals:
            broken_fd.write(f"{removal + 1}\n")

        broken_fd.flush()

        cmds = [
            # TODO move build to setup.py?
            # 'make clean && make',
            "make",

            f"./{reinsertion_executable} "
            f"--NetworkFile {network_path} "
            f'--IDFile "{broken_path}" '
            f'--OutFile "{output_path}" '
            f"--TargetSize {stop_condition} "
            f"--SortStrategy {reinsertion_strategy} "
        ]

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
            except CalledProcessError as e:
                logger.error(f"ERROR while running reinsertion algorithm: {e}", exc_info=True)
                raise RuntimeError(f"ERROR! {e}")
            except Exception as e:
                raise RuntimeError("ERROR! {}".format(e))

        # Reset the file pointer to the beginning of the file
        output_fd.seek(0)

        # Read the output file
        # Count the number of lines
        num_removals = 0
        for _ in output_fd.readlines():
            num_removals += 1

        if num_removals <= 0:
            raise RuntimeError("No removals found!")

        output_fd.seek(0)
        for i, line in enumerate(output_fd.readlines(), start=0):
            node = int(line.strip())
            node -= 1

            nodes.append(node)

            output[node] = num_removals - i

            if output[node] <= 0:
                logger.error(f"Node {node} has a non-positive value: {output[node]}")
                raise RuntimeError(f"Node {node} has a non-positive value: {output[node]}")

    logger.debug("Reinsertion algorithm finished")
    logger.debug(f"Original number of removals: {len(removals)}")
    logger.debug(f"Number of final removals: {num_removals}")

    return output
