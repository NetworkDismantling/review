from graph_tool import Graph

from network_dismantling import dismantler_wrapper
from network_dismantling._sorters import dismantling_method
from network_dismantling.multiscale_entanglement.entanglement_functions import (
    # entanglement_small,
    # entanglement_mid,
    # entanglement_large,

    entanglement, beta_small, beta_large, beta_mid)

method_info = {
    "source": "",
    # "authors": "",
    # "citation": "",
    "includes_reinsertion": False,
    "plot_color": "#d62728",
}


@dismantling_method(
    name=r"Small-Scale Entanglement",
    short_name=r"$\mathrm{MSE}_s$",
    **method_info,
)
@dismantler_wrapper
def entanglement_small(network: Graph, **kwargs):
    return entanglement(G=network, beta=beta_small, **kwargs)


@dismantling_method(
    name=r"Mid-Scale Entanglement",
    short_name=r"$\mathrm{MSE}_m$",
    **method_info,
)
@dismantler_wrapper
def entanglement_mid(network: Graph, **kwargs):
    return entanglement(G=network, beta=beta_mid, **kwargs)


@dismantling_method(
    name=r"Large-Scale Entanglement",
    short_name=r"$\mathrm{MSE}_l$",
    **method_info,
)
@dismantler_wrapper
def entanglement_large(network: Graph, **kwargs):
    return entanglement(G=network, beta=beta_large, **kwargs)
