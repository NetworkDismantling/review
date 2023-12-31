from parse import compile

from network_dismantling import dismantler_wrapper
from network_dismantling._sorters import dismantling_method

targets_num_expression = compile("Targets  {targets:d}")

folder = "network_dismantling/CoreHD/"
cd_cmd = f"cd {folder} && "
executable = "coreHD"

# TODO use tempfile.NamedTemporaryFile?
# TODO use logger instead of print


@dismantler_wrapper
def _coreHD(network, **kwargs):
    import tempfile
    from os import close, remove
    from subprocess import check_output

    import numpy as np
    from graph_tool.all import remove_parallel_edges, remove_self_loops

    # CoreHD does not support parallel edges or self-loops.
    # Remove them.
    remove_parallel_edges(network)
    remove_self_loops(network)

    static_id = network.vertex_properties["static_id"]

    network_fd, network_path = tempfile.mkstemp()
    output_fd, output_path = tempfile.mkstemp()
    feedback_fd, feedback_path = tempfile.mkstemp()
    time_fd, time_path = tempfile.mkstemp()

    tmp_file_handles = [network_fd, output_fd, feedback_fd, time_fd]
    tmp_file_paths = [network_path, output_path, feedback_path, time_path]

    nodes = []

    try:
        with open(network_fd, "w+") as tmp:
            tmp.write(f"{network.num_vertices()} {network.num_edges()}\n")

            for edge in network.edges():
                tmp.write(
                    f"{static_id[edge.source()] + 1} {static_id[edge.target()] + 1}\n"
                )

            # for edge in network.get_edges():
            #     tmp.write("{} {}\n".format(int(edge[0]) + 1, int(edge[1]) + 1))

        cmds = [
            # TODO move build to setup.py?
            # 'make clean && make',
            "make",
            f"./{executable} "
            f'--NetworkFile "{network_path}" '
            f"--VertexNumber {network.num_vertices()} "
            f"--EdgeNumber {network.num_edges()} "
            f'--Afile "{output_path}" '
            f'--FVSfile "{feedback_path}" '
            f'--Timefile "{time_path}" '
            f'--Csize {kwargs["stop_condition"]} '
            # f'--seed {kwargs["seed"]} '
            #     int rdseed = 93276792; //you can set this seed to another value
            #     int prerun = 14000000; //you can set it to another value
        ]

        for cmd in cmds:
            try:
                print(f"Running cmd: {cmd}")

                print(
                    check_output(
                        cd_cmd + cmd,
                        shell=True,
                        text=True,
                        # close_fds=True,
                        # stderr=STDOUT,
                    )
                )
            except Exception as e:
                raise RuntimeError(f"ERROR! When running cmd: {cmd} {e}")

        with open(output_fd, "r+") as tmp:
            lines = tmp.readlines()
            num_targets = targets_num_expression.parse(lines[0].strip())["targets"]

            for line in lines[2:]:
                node = line.strip()

                nodes.append(node)

            if len(nodes) > num_targets:
                raise RuntimeError

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
    name="CoreHD",
    short_name="CoreHD",
    plot_color="#3080bd",
    includes_reinsertion=True,
)
def CoreHD(*args, **kwargs):
    return _coreHD(*args, **kwargs)
