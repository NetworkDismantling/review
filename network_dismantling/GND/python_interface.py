import tempfile
from os import remove
from os.path import relpath, dirname, realpath
from subprocess import check_output, STDOUT

import numpy as np

from network_dismantling._sorters import dismantling_method


def _generalized_network_dismantling(network, reinsertion=False, remove_strategy=3, reinsertion_strategy=2, **kwargs):
    folder = 'network_dismantling/GND/'
    cd_cmd = f'cd {folder} && '
    executable = 'GND'
    reinsertion_executable = 'reinsertion'

    static_id = network.vertex_properties["static_id"]

    network_fd, network_path = tempfile.mkstemp()
    broken_fd, broken_path = tempfile.mkstemp()
    output_fd, output_path = tempfile.mkstemp()

    nodes = []

    try:
        with open(network_fd, 'w+') as tmp:
            for edge in network.edges():
                tmp.write("{} {}\n".format(
                    static_id[edge.source()] + 1,
                    static_id[edge.target()] + 1
                )
                )

        cmds = [
            # TODO move build to setup.py?
            # 'make clean && make',
            'make',
            f'./{executable} '
            f'--NetworkFile {network_path} '
            f'--IDFile "{broken_path}" '
            f'--NodeNum {network.num_vertices()} '
            f'--TargetSize {kwargs["stop_condition"]} '
            f'--RemoveStrategy {remove_strategy} '
        ]

        if reinsertion is True:

            cmds.append(
                f'./{reinsertion_executable} '
                f'--NetworkFile {network_path} '
                f'--IDFile "{broken_path}" '
                f'--OutFile "{output_path}" '
                f'--TargetSize {kwargs["stop_condition"]} '
                f'--SortStrategy {reinsertion_strategy} '
            )

            output = output_fd
        else:
            output = broken_fd

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
                raise RuntimeError(f"ERROR! When running cmd: {cmd} {e}")

        with open(output, 'r+') as tmp:
            for line in tmp.readlines():
                node = line.strip()

                nodes.append(node)

    finally:
        # os.close(network_fd)
        # os.close(broken_fd)
        # os.close(output_fd)

        remove(network_path)
        remove(broken_path)
        remove(output_path)

    output = np.zeros(network.num_vertices())

    for n, p in zip(nodes, list(reversed(range(1, len(nodes) + 1)))):
        output[int(n) - 1] = p

    return output


method_info = {
    "source": "https://github.com/renxiaolong/Generalized-Network-Dismantling",
}


@dismantling_method(**method_info)
def GND(network, **kwargs):
    return _generalized_network_dismantling(network, reinsertion=False, **kwargs)


@dismantling_method(**method_info)
def GNDR(network, **kwargs):
    return _generalized_network_dismantling(network, reinsertion=True, **kwargs)
