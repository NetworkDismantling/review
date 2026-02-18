import logging
from pathlib import Path

from graph_tool import Graph
from parse import compile

from network_dismantling import dismantler_wrapper
from network_dismantling._sorters import dismantling_method

targets_num_expression = compile("Vaccinated nodes {num:d}")

_EI_DIR = Path(__file__).resolve().parent
executable = "exploimmun"


def _explosive_immunization(
        network: Graph,
        stop_condition: int,
        sigma: int,
        candidates: int,
        logger: logging.Logger = logging.getLogger("dummy"),
        **kwargs
):
    from subprocess import run, CalledProcessError
    from tempfile import NamedTemporaryFile

    import numpy as np
    from graph_tool.all import remove_parallel_edges, remove_self_loops

    from network_dismantling.common.logging.pipe import LogPipe

    cd_cmd = f"cd {_EI_DIR} && "

    # Not sure if EI supports parallel edges or self-loops.
    # Remove them as this fixes a bug and as they do not alter the dismantling set
    remove_parallel_edges(network)
    remove_self_loops(network)

    static_id = network.vertex_properties["static_id"]

    assert static_id.a.min() == 0, "Static id must start from 0"
    assert (
            static_id.a.max() == network.num_vertices() - 1
    ), "Static id must be consecutive"

    unvaccinated_nodes = []

    with (
        NamedTemporaryFile("w+", suffix=".ei_net") as network_f,
        NamedTemporaryFile("r+", suffix=".ei_out") as output_f,
        NamedTemporaryFile("w+", suffix=".ei_thr") as threshold_f,
        LogPipe(logger=logger, level=logging.INFO) as stdout_pipe,
        LogPipe(logger=logger, level=logging.ERROR) as stderr_pipe,
    ):
        network_f.write("{}\n".format(network.num_vertices()))
        for edge in network.edges():
            network_f.write(
                "{} {}\n".format(static_id[edge.source()], static_id[edge.target()])
            )
        network_f.flush()

        cmds = [
            "make -C Library",
            f"./{executable} {candidates} {network_f.name} {output_f.name} {stop_condition} {sigma} {threshold_f.name}",
        ]

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
                raise RuntimeError(f"EI binary failed on cmd '{cmd}': {e}") from e

        output_f.seek(0)
        for line in output_f:
            node = line.strip()
            unvaccinated_nodes.append(node)

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
    name=r"Explosive Immunization $\sigma=1$",
    short_name=r"EI $\sigma=1$",
    includes_reinsertion=False,
    # plot_color="",
    **method_info,
)
@dismantler_wrapper
def EI_s1(network, **kwargs):
    return _explosive_immunization(network, candidates=1000, sigma=1, **kwargs)


@dismantling_method(
    name=r"Explosive Immunization $\sigma=2$",
    short_name=r"EI $\sigma=2$",
    includes_reinsertion=False,
    # plot_color="",
    **method_info,
)
@dismantler_wrapper
def EI_s2(network, **kwargs):
    return _explosive_immunization(network, candidates=1000, sigma=2, **kwargs)
