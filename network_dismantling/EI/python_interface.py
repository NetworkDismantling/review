from graph_tool import Graph
from parse import compile

from network_dismantling import dismantler_wrapper
from network_dismantling._sorters import dismantling_method

targets_num_expression = compile("Vaccinated nodes {num:d}")

folder = "network_dismantling/EI/"
cd_cmd = "cd {} && ".format(folder)
executable = "exploimmun"

# TODO use tempfile.NamedTemporaryFile?
# TODO use logger instead of print


def _explosive_immunization(
    network: Graph, stop_condition: int, sigma: int, candidates: int, **kwargs
):
    import tempfile
    from os import close, remove
    from subprocess import check_output, STDOUT

    import numpy as np
    from graph_tool.all import remove_parallel_edges, remove_self_loops

    # Not sure if EI supports parallel edges or self-loops.
    # Remove them as this fixes a bug and as they do not alter the dismantling set
    remove_parallel_edges(network)
    remove_self_loops(network)

    static_id = network.vertex_properties["static_id"]

    assert static_id.a.min() == 0, "Static id must start from 0"
    assert (
        static_id.a.max() == network.num_vertices() - 1
    ), "Static id must be consecutive"

    network_fd, network_path = tempfile.mkstemp()
    output_fd, output_path = tempfile.mkstemp()
    threshold_condition_fd, threshold_condition_path = tempfile.mkstemp()

    tmp_file_handles = [network_fd, output_fd, threshold_condition_fd]
    tmp_file_paths = [network_path, output_path, threshold_condition_path]

    # vaccinated_nodes = []
    unvaccinated_nodes = []
    try:
        with open(network_fd, "w+") as tmp:
            tmp.write("{}\n".format(network.num_vertices()))
            for edge in network.edges():
                tmp.write(
                    "{} {}\n".format(static_id[edge.source()], static_id[edge.target()])
                )

        cmds = [
            # 'make clean && make',
            "make -C Library",
            f"./{executable} {candidates} {network_path} {output_path} {stop_condition} {sigma} {threshold_condition_path}",
        ]

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
                exit("ERROR! {}".format(e))

        # # Safety check
        # with open(threshold_condition_fd, 'r+') as tmp:
        #     for line in tmp.readlines():
        #         node, vaccinated = line.strip().split()
        #
        #         vaccinated = int(vaccinated)
        #
        #         if vaccinated:
        #             vaccinated_nodes.append(int(node))

        with open(output_fd, "r+") as tmp:
            for line in tmp.readlines():
                node = line.strip()

                unvaccinated_nodes.append(node)

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

    output = np.arange(start=1, stop=network.num_vertices() + 1)

    for n in unvaccinated_nodes:
        output[int(n)] = 0

    return output


method_info = {
    # "name": "Explosive Immunization",
    # "short_name": "EI",
    # "description": "Explosive Immunization",
    "source": "https://github.com/pclus/explosive-immunization/",
}


@dismantling_method(
    name="Explosive Immunization $\sigma=1$",
    short_name="EI $\sigma=1$",
    includes_reinsertion=False,
    # plot_color="",
    **method_info,
)
@dismantler_wrapper
def EI_s1(network, **kwargs):
    return _explosive_immunization(network, candidates=1000, sigma=1, **kwargs)


@dismantling_method(
    name="Explosive Immunization $\sigma=2$",
    short_name="EI $\sigma=2$",
    includes_reinsertion=False,
    # plot_color="",
    **method_info,
)
@dismantler_wrapper
def EI_s2(network, **kwargs):
    return _explosive_immunization(network, candidates=1000, sigma=2, **kwargs)
