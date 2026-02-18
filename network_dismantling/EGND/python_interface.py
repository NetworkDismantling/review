import logging
from pathlib import Path

from network_dismantling import dismantler_wrapper
from network_dismantling._sorters import dismantling_method

_EGND_DIR = Path(__file__).resolve().parent
config_file = "config.h"

config_r_file = "config_r.h"
reinsertion_strategy = 2


# TODO USE BOOST COMMAND LINE PARSER


@dismantler_wrapper
def _ensemble_generalized_network_dismantling(
        network, reinsertion=False, remove_strategy=3, runs=1000,
        logger: logging.Logger = logging.getLogger("dummy"), **kwargs
):
    from os.path import relpath
    from subprocess import run, CalledProcessError
    from tempfile import NamedTemporaryFile

    import numpy as np

    from network_dismantling.common.logging.pipe import LogPipe

    cd_cmd = f"cd {_EGND_DIR} && "

    static_id = network.vertex_properties["static_id"]

    nodes = []

    with (
        NamedTemporaryFile("w+", suffix=".egnd_net") as network_f,
        NamedTemporaryFile("r+", suffix=".egnd_brk") as broken_f,
        NamedTemporaryFile("r+", suffix=".egnd_out") as output_f,
        NamedTemporaryFile("w+", suffix=".egnd_plt") as plot_f,
        NamedTemporaryFile("w+", suffix=".egnd_seed") as seed_f,
        LogPipe(logger=logger, level=logging.INFO) as stdout_pipe,
        LogPipe(logger=logger, level=logging.ERROR) as stderr_pipe,
    ):
        # Write edge list (1-indexed)
        for edge in network.edges():
            network_f.write(
                "{} {}\n".format(
                    static_id[edge.source()] + 1, static_id[edge.target()] + 1
                )
            )
        network_f.flush()

        # Write config.h with relative paths for the C++ binary
        config_path = _EGND_DIR / config_file
        with open(config_path, "w") as f:
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
                    "../" + relpath(network_f.name, _EGND_DIR),
                    "../" + relpath(broken_f.name, _EGND_DIR),
                    "../" + relpath(plot_f.name, _EGND_DIR),
                    "../" + relpath(seed_f.name, _EGND_DIR),
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
                raise RuntimeError(f"EGND binary failed on cmd '{cmd}': {e}") from e

        # Read output (broken file contains removed node IDs)
        broken_f.seek(0)
        for line in broken_f:
            node = line.strip()
            nodes.append(node)

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
