import tempfile
from os import remove
from subprocess import check_output, STDOUT

import numpy as np

from network_dismantling._sorters import dismantling_method


def _decycler(network, reinsertion=True, **kwargs):
    cd_cmd = 'cd network_dismantling/decycler/ && '

    network_type = "D" if network.is_directed() else "E"

    static_id = network.vertex_properties["static_id"]

    network_fd, network_path = tempfile.mkstemp()
    seeds_fd, seeds_path = tempfile.mkstemp()
    broken_fd, broken_path = tempfile.mkstemp()
    output_fd, output_path = tempfile.mkstemp()

    nodes = []

    try:
        with open(network_fd, 'w+') as tmp:
            for edge in network.edges():
                if edge.source() != edge.target():
                    tmp.write("{} {} {}\n".format(
                        network_type,
                        static_id[edge.source()] + 1,
                        static_id[edge.target()] + 1
                    )
                    )
            # for edge in network.get_edges():
            #     if edge[0] != edge[1]:
            #         tmp.write("{} {} {}\n".format(network_type, int(edge[0]) + 1, int(edge[1]) + 1))

        cmds = [
            'make',
            'cat {} | ./decycler -o > {}'.format(
                network_path,
                seeds_path
            ),
            '(cat {} {}) | python treebreaker.py {} > {}'.format(
                network_path,
                seeds_path,
                kwargs["stop_condition"],
                broken_path
            ),
        ]

        if reinsertion is True:
            cmds.append(
                '(cat {} {} {}) | ./reverse-greedy -t {} > {}'.format(
                    network_path,
                    seeds_path,
                    broken_path,
                    kwargs["stop_condition"],
                    output_path
                )
            )

            output = [output_fd]
        else:
            output = [seeds_fd, broken_fd]

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

        # Iterate over seeds
        for tmp_file in output:
            with open(tmp_file, 'r+') as tmp:
                for line in tmp.readlines():
                    line = line.strip().split(" ")

                    node_type, seed = line[0], line[1]

                    if node_type != "S":
                        continue
                        # raise ValueError("Unexpected output: {}".format(line))

                    nodes.append(seed)
    finally:
        # os.close(network_fd)
        # os.close(seeds_fd)
        # os.close(broken_fd)
        # os.close(output_fd)

        remove(network_path)
        remove(seeds_path)
        remove(broken_path)
        remove(output_path)

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


@dismantling_method(**method_info)
def MS(network, **kwargs):
    return _decycler(network, reinsertion=False, **kwargs)


@dismantling_method(**method_info)
def MSR(network, **kwargs):
    return _decycler(network, reinsertion=True, **kwargs)
