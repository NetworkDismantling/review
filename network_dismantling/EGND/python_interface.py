from network_dismantling import dismantler_wrapper
from network_dismantling._sorters import dismantling_method

folder = "network_dismantling/EGND/"
cd_cmd = "cd {} && ".format(folder)
config_file = "config.h"

config_r_file = "config_r.h"
reinsertion_strategy = 2

# TODO USE BOOST COMMAND LINE PARSER
# TODO use tempfile.NamedTemporaryFile?
# TODO use logger instead of print


@dismantler_wrapper
def _ensemble_generalized_network_dismantling(
    network, reinsertion=False, remove_strategy=3, runs=1000, **kwargs
):
    import tempfile
    from os import close, remove
    from os.path import relpath, dirname, realpath
    from subprocess import check_output, STDOUT

    import numpy as np

    static_id = network.vertex_properties["static_id"]

    network_fd, network_path = tempfile.mkstemp()
    broken_fd, broken_path = tempfile.mkstemp()
    output_fd, output_path = tempfile.mkstemp()
    plot_fd, plot_path = tempfile.mkstemp()
    seed_fd, seed_path = tempfile.mkstemp()

    tmp_file_handles = [network_fd, broken_fd, output_fd, plot_fd, seed_fd]
    tmp_file_paths = [network_path, broken_path, output_path, plot_path, seed_path]

    nodes = []

    try:
        with open(network_fd, "w+") as tmp:
            for edge in network.edges():
                tmp.write(
                    "{} {}\n".format(
                        static_id[edge.source()] + 1, static_id[edge.target()] + 1
                    )
                )

        with open(folder + config_file, "w+") as f:
            f.write(
                (
                    "const int NODE_NUM = {};                  // the number of nodes\n"
                    'const char* FILE_NET = "{}";            // input format of each line: id1 id2\n'
                    'const char* FILE_ID = "{}";             // output the id of the removed nodes in order\n'
                    'const char* FILE_PLOT = "{}";           // format of each line: gcc removed_cost removed_nodes\n'
                    'const char* FILE_SEED = "{}";\n'
                    "const int TARGET_SIZE = {};               // If the gcc size is smaller than TARGET_SIZE, the dismantling will stop. Default value can be 0.01*NODE_NUM  OR  1000\n"
                    "const int REMOVE_STRATEGY = {};           // 1: weighted method: powerIterationB(); vertex_cover_2() -- remove the node with smaller degree first\n"
                    "                                          // 3: unweighted method with one-degree in vertex cover： powerIteration; vertex_cover() -- remove the node with larger degree first\n"
                    "const int PLOT_SIZE = {};                 // the removal size of each line in FILE_PLOT. E.g. PLOT_SIZE=2 means each line of FILE_PLOT is the result that remove two nodes from the network\n"
                    "int C = {};                               // number of different run of GND\n"
                ).format(
                    network.num_vertices(),
                    "../" + relpath(network_path, dirname(realpath(__file__))),
                    "../" + relpath(broken_path, dirname(realpath(__file__))),
                    "../" + relpath(plot_path, dirname(realpath(__file__))),
                    "../" + relpath(seed_path, dirname(realpath(__file__))),
                    kwargs["stop_condition"],
                    remove_strategy,
                    network.num_vertices()
                    + 1,  # (never output to the plot file. We don't need it!)
                    runs,  # Default value is 1000
                )
            )

        cmds = ["make clean && make", "./EnsembleGND"]

        if reinsertion is True:
            raise NotImplementedError
        else:
            output = broken_fd

        for cmd in cmds:
            try:
                print(f"Running cmd: {cmd}")

                print(
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


@dismantling_method(
    name="Ensemble Generalized Network Dismantling",
    # display_name="EGND",
    short_name="EGND",
    plot_color="#ffbb78",
    includes_reinsertion=False,
    source="https://github.com/renxiaolong/2019-Ensemble-approach-for-generalized-network-dismantling",
)
def EGND(network, **kwargs):
    return _ensemble_generalized_network_dismantling(
        network, reinsertion=False, **kwargs
    )
