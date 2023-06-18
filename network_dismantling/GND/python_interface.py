import tempfile
from os import remove
from os.path import relpath, dirname, realpath
from subprocess import check_output, STDOUT

import numpy as np

from network_dismantling._sorters import dismantling_method


def _generalized_network_dismantling(network, reinsertion=False, remove_strategy=3, reinsertion_strategy=2, **kwargs):
    folder = 'network_dismantling/GND/'
    cd_cmd = 'cd {} && '.format(folder)
    config_file = "config.h"

    config_r_file = "config_r.h"

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
            # for edge in network.get_edges():
            #     tmp.write("{} {}\n".format(int(edge[0]) + 1, int(edge[1]) + 1))

        with open(folder + config_file, "w+") as f:
            f.write(("const int NODE_NUM = {};                  // the number of nodes\n"
                     "const char* FILE_NET = \"{}\";            // input format of each line: id1 id2\n"
                     "const char* FILE_ID = \"{}\";             // output the id of the removed nodes in order\n"
                     "const int TARGET_SIZE = {};               // If the gcc size is smaller than TARGET_SIZE, the dismantling will stop. Default value can be 0.01*NODE_NUM  OR  1000\n"
                     "const int REMOVE_STRATEGY = {};           // 1: weighted method: powerIterationB(); vertex_cover_2() -- remove the node with smaller degree first\n"
                     "                                          // 3: unweighted method with one-degree in vertex cover： powerIteration; vertex_cover() -- remove the node with larger degree first\n"
                     ).format(network.num_vertices(),
                              "../" + relpath(network_path, dirname(realpath(__file__))),
                              "../" + relpath(broken_path, dirname(realpath(__file__))),
                              kwargs["stop_condition"],
                              remove_strategy
                              )
                    )

        cmds = [
            'make clean && make',
            './GND'
        ]

        if reinsertion is True:
            with open(folder + config_r_file, "w+") as f:
                f.write(("const char* fileNet = \"{}\";  // input format of each line: id1 id2\n"
                         "const char* fileId = \"{}\";   // output the id of the removed nodes in order\n"
                         "const char* outputFile = \"{}\";   // output the id of the removed nodes after reinserting\n"
                         "const int strategy = {}; // removing order\n"
                         "                             // 0: keep the original order\n"
                         "                             // 1: ascending order - better strategy for weighted case\n"
                         "                             // 2: descending order - better strategy for unweighted case\n"
                         ).format("../" + relpath(network_path, dirname(realpath(__file__))),
                                  "../" + relpath(broken_path, dirname(realpath(__file__))),
                                  "../" + relpath(output_path, dirname(realpath(__file__))),
                                  reinsertion_strategy
                                  )
                        )
            cmds.append(
                './reinsertion -t {}'.format(
                    kwargs["stop_condition"],
                )
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
