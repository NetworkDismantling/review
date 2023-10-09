from network_dismantling import dismantler_wrapper
from network_dismantling._sorters import dismantling_method

cd_cmd = "cd network_dismantling/decycler/ && "
executable = "decycler"
reverse_greedy_executable = "reverse-greedy"

# TODO use tempfile.NamedTemporaryFile?
# TODO use logger instead of print


@dismantler_wrapper
def _decycler(network, stop_condition: int, reinsertion=True, **kwargs):
    import tempfile
    from os import close, remove
    from subprocess import check_output, STDOUT

    import numpy as np

    network_type = "D" if network.is_directed() else "E"

    static_id = network.vertex_properties["static_id"]

    network_fd, network_path = tempfile.mkstemp()
    seeds_fd, seeds_path = tempfile.mkstemp()
    broken_fd, broken_path = tempfile.mkstemp()
    output_fd, output_path = tempfile.mkstemp()

    tmp_file_handles = [network_fd, seeds_fd, broken_fd, output_fd]
    tmp_file_paths = [network_path, seeds_path, broken_path, output_path]

    nodes = []

    try:
        with open(network_fd, "w+") as tmp:
            for edge in network.edges():
                if edge.source() != edge.target():
                    tmp.write(
                        "{} {} {}\n".format(
                            network_type,
                            static_id[edge.source()] + 1,
                            static_id[edge.target()] + 1,
                        )
                    )
            # for edge in network.get_edges():
            #     if edge[0] != edge[1]:
            #         tmp.write("{} {} {}\n".format(network_type, int(edge[0]) + 1, int(edge[1]) + 1))

        cmds = [
            "make",
            f"cat {network_path} | ./{executable} -o > {seeds_path}",
            f"(cat {network_path} {seeds_path}) | python treebreaker.py {stop_condition} > {broken_path}",
        ]

        if reinsertion is True:
            cmds.append(
                f"(cat {network_path} {seeds_path} {broken_path}) | "
                f"./{reverse_greedy_executable} -t {stop_condition} > {output_path}"
            )

            output = [output_fd]
        else:
            output = [seeds_fd, broken_fd]

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

        # Iterate over seeds
        for tmp_file in output:
            with open(tmp_file, "r+") as tmp:
                for line in tmp.readlines():
                    line = line.strip().split(" ")

                    node_type, seed = line[0], line[1]

                    if node_type != "S":
                        continue
                        # raise ValueError("Unexpected output: {}".format(line))

                    nodes.append(seed)
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
def MS(network, **kwargs):
    return _decycler(network, reinsertion=False, **kwargs)


@dismantling_method(
    name="Min-Sum + Reinsertion",
    short_name="MS +R",
    plot_color="#2ca02c",
    includes_reinsertion=True,
    **method_info,
)
def MSR(network, **kwargs):
    return _decycler(network, reinsertion=True, **kwargs)
