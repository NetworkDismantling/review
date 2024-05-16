from logging import Logger, getLogger
from typing import Callable

from graph_tool import Graph

from network_dismantling import dismantler_wrapper
from network_dismantling._sorters import dismantling_method

method_info = {
    "source": "",
    # "authors": "",
    # "citation": "",
    # "includes_reinsertion": False,
    # "plot_color": "#d62728",
    # 1f77b4
    # 9e1309
    # ebae07
}


# @dismantling_method(
#     name=r"Small-Scale Entanglement",
#     short_name=r"$\mathrm{MSE}_s$",
#     **method_info,
# )
# @dismantler_wrapper()
# def entanglement_small(network: Graph, **kwargs):
#     logger = kwargs.get("logger", getLogger("dummy"))
#     logger.debug("Small-Scale Entanglement")
#     logger.debug(f"kwargs: {kwargs}")
#     return entanglement(G=network, beta=beta_small, **kwargs)
#
#
# @dismantling_method(
#     name=r"Mid-Scale Entanglement",
#     short_name=r"$\mathrm{MSE}_m$",
#     **method_info,
# )
# @dismantler_wrapper()
# def entanglement_mid(network: Graph, **kwargs):
#     return entanglement(G=network, beta=beta_mid, **kwargs)
#
#
# @dismantling_method(
#     name=r"Large-Scale Entanglement",
#     short_name=r"$\mathrm{MSE}_l$",
#     **method_info,
# )
# @dismantler_wrapper()
# def entanglement_large(network: Graph, **kwargs):
#     return entanglement(G=network, beta=beta_large, **kwargs)


# @dismantling_method(
#     name=r"Large-Scale Entanglement + Reinsertion",
#     short_name=r"$\mathrm{MSE}_l$ + R",
#     depends_on=entanglement_large,
#     **method_info,
# )
# @dismantler_wrapper
# def entanglement_large_reinsertion(network: Graph,
#                                    stop_condition: int,
#                                    entanglement_large: Union[list, np.ndarray],
#                                    logger: Logger = getLogger("dummy"),
#                                    **kwargs):
#     from network_dismantling.multiscale_entanglement.reinsertion import reinsert
#
#     predictions = reinsert(
#         network=network,
#         removals=entanglement_large,
#         stop_condition=stop_condition,
#         logger=logger,
#     )
#
#     return predictions


def entanglement_reinsertion(network: Graph,
                             stop_condition: int,
                             entanglement_function: Callable,
                             logger: Logger = getLogger("dummy"),
                             **kwargs):
    from network_dismantling.multiscale_entanglement.reinsertion import reinsert

    logger.debug(f"entanglement_function: {entanglement_function} kwargs: {kwargs}")
    run_info = entanglement_function(network=network,
                                     stop_condition=stop_condition,
                                     logger=logger,
                                     **kwargs)

    removals = run_info["predictions"]

    predictions = reinsert(
        network=network,
        removals=removals,
        stop_condition=stop_condition,
        logger=logger,
    )

    return predictions


# @dismantling_method(
#     name=r"Small-Scale Entanglement + Reinsertion",
#     short_name=r"$\mathrm{MSE}_s$ + R",
#     **method_info,
# )
# @dismantler_wrapper
# def entanglement_small_reinsertion(network: Graph,
#                                    stop_condition: int,
#                                    logger: Logger = getLogger("dummy"),
#                                    **kwargs):
#     return entanglement_reinsertion(network=network,
#                                     stop_condition=stop_condition,
#                                     entanglement_function=entanglement_small,
#                                     logger=logger,
#                                     **kwargs)
#
#
# @dismantling_method(
#     name=r"Mid-Scale Entanglement + Reinsertion",
#     short_name=r"$\mathrm{MSE}_m$ + R",
#     **method_info,
# )
# @dismantler_wrapper
# def entanglement_mid_reinsertion(network: Graph,
#                                  stop_condition: int,
#                                  logger: Logger = getLogger("dummy"),
#                                  **kwargs):
#     return entanglement_reinsertion(network=network,
#                                     stop_condition=stop_condition,
#                                     entanglement_function=entanglement_mid,
#                                     logger=logger,
#                                     **kwargs)
#
#
# @dismantling_method(
#     name=r"Large-Scale Entanglement + Reinsertion",
#     short_name=r"$\mathrm{MSE}_l$ + R",
#     **method_info,
# )
# @dismantler_wrapper
# def entanglement_large_reinsertion(network: Graph,
#                                    stop_condition: int,
#                                    logger: Logger = getLogger("dummy"),
#                                    **kwargs):
#     return entanglement_reinsertion(network=network,
#                                     stop_condition=stop_condition,
#                                     entanglement_function=entanglement_large,
#                                     logger=logger,
#                                     **kwargs)


@dismantling_method(
    name=r"Small-Scale Network Entanglement",
    short_name=r"$\mathrm{MSNE}_s$",
    includes_reinsertion=False,
    plot_color="#ebae07",
    **method_info,
)
@dismantler_wrapper
def network_entanglement_small(network: Graph,
                               stop_condition: int,
                               logger: Logger = getLogger("dummy"),
                               **kwargs):
    return network.vertex_properties["NE_small"].get_array()


@dismantling_method(
    name=r"Mid-Scale Network Entanglement",
    short_name=r"$\mathrm{MSNE}_m$",
    includes_reinsertion=False,
    plot_color="#34e8eb",
    **method_info,
)
@dismantler_wrapper
def network_entanglement_mid(network: Graph,
                             stop_condition: int,
                             logger: Logger = getLogger("dummy"),
                             **kwargs):
    return network.vertex_properties["NE_mid"].get_array()


@dismantling_method(
    name=r"Large-Scale Network Entanglement",
    short_name=r"$\mathrm{MSNE}_l$",
    includes_reinsertion=False,
    plot_color="#ed02e9",
    **method_info,
)
@dismantler_wrapper
def network_entanglement_large(network: Graph,
                               stop_condition: int,
                               logger: Logger = getLogger("dummy"),
                               **kwargs):
    return network.vertex_properties["NE_large"].get_array()


@dismantling_method(
    name=r"Small-Scale Network Entanglement + Reinsertion",
    short_name=r"$\mathrm{MSNE}_s$ + R",
    includes_reinsertion=True,
    plot_color="#ebae07",
    **method_info,
)
@dismantler_wrapper
def network_entanglement_small_reinsertion(network: Graph,
                                           stop_condition: int,
                                           logger: Logger = getLogger("dummy"),
                                           **kwargs):
    raise NotImplementedError


@dismantling_method(
    name=r"Mid-Scale Network Entanglement + Reinsertion",
    short_name=r"$\mathrm{MSNE}_m$ + R",
    includes_reinsertion=True,
    plot_color="#34e8eb",
    **method_info,
)
@dismantler_wrapper
def network_entanglement_mid_reinsertion(network: Graph,
                                         stop_condition: int,
                                         logger: Logger = getLogger("dummy"),
                                         **kwargs):
    raise NotImplementedError


@dismantling_method(
    name=r"Large-Scale Network Entanglement + Reinsertion",
    short_name=r"$\mathrm{MSNE}_l$ + R",
    includes_reinsertion=True,
    plot_color="#ed02e9",
    **method_info,
)
@dismantler_wrapper
def network_entanglement_large_reinsertion(network: Graph,
                                           stop_condition: int,
                                           logger: Logger = getLogger("dummy"),
                                           **kwargs):
    raise NotImplementedError


@dismantling_method(
    name=r"Vertex Entanglement",
    short_name=r"$\mathrm{VE}$",
    includes_reinsertion=False,
    plot_color="#34eb46",
    # **method_info,
)
@dismantler_wrapper
def vertex_entanglement(network: Graph,
                        stop_condition: int,
                        logger: Logger = getLogger("dummy"),
                        **kwargs):
    return network.vertex_properties["VE"].get_array()


@dismantling_method(
    name=r"Vertex Entanglement + Reinsertion",
    short_name=r"$\mathrm{VE}$ + R",
    includes_reinsertion=True,
    plot_color="#34eb46",
    # **method_info,
)
@dismantler_wrapper
def vertex_entanglement_reinsertion(network: Graph,
                                    stop_condition: int,
                                    logger: Logger = getLogger("dummy"),
                                    **kwargs):
    raise NotImplementedError
