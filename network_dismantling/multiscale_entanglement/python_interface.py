from logging import Logger, getLogger
from typing import Callable, Union

import numpy as np
from graph_tool import Graph

from network_dismantling import dismantler_wrapper
from network_dismantling._sorters import dismantling_method
from network_dismantling.multiscale_entanglement.entanglement_functions import entanglement_small, entanglement_mid, \
    entanglement_large, entanglement, beta_small, beta_mid, beta_large

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


@dismantling_method(
    name=r"Small-Scale Entanglement",
    short_name=r"$\mathrm{MSE}_s$",
    **method_info,
)
@dismantler_wrapper
def entanglement_small(network: Graph, **kwargs):
    logger = kwargs.get("logger", getLogger("dummy"))
    logger.debug("Small-Scale Entanglement")
    logger.debug(f"kwargs: {kwargs}")
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


@dismantling_method(
    name=r"Large-Scale Entanglement + Reinsertion",
    short_name=r"$\mathrm{NE}_l$ + R",
    depends_on=entanglement_large,
    **method_info,
)
@dismantler_wrapper
def entanglement_large_reinsertion(network: Graph,
                                   stop_condition: int,
                                   entanglement_large: Union[list, np.ndarray],
                                   logger: Logger = getLogger("dummy"),
                                   **kwargs):
    from network_dismantling.multiscale_entanglement.reinsertion import reinsert

    predictions = reinsert(
        network=network,
        removals=entanglement_large,
        stop_condition=stop_condition,
        logger=logger,
    )

    return predictions


# def entanglement_reinsertion(network: Graph,
#                              stop_condition: int,
#                              entanglement_function: Callable,
#                              logger: Logger = getLogger("dummy"),
#                              **kwargs):
#     from network_dismantling.multiscale_entanglement.reinsertion import reinsert
#
#     logger.debug(f"entanglement_function: {entanglement_function} kwargs: {kwargs}")
#     run_info = entanglement_function(network=network,
#                                      stop_condition=stop_condition,
#                                      logger=logger,
#                                      **kwargs)
#
#     removals = run_info["predictions"]
#
#     predictions = reinsert(
#         network=network,
#         removals=removals,
#         stop_condition=stop_condition,
#         logger=logger,
#     )
#
#     return predictions
#
#
# @dismantling_method(
#     name=r"Small-Scale Entanglement + Reinsertion",
#     short_name=r"$\mathrm{NE}_s$ + R",
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
#     short_name=r"$\mathrm{NE}_m$ + R",
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
#     short_name=r"$\mathrm{NE}_l$ + R",
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

