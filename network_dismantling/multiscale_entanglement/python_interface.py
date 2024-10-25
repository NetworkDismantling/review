from logging import Logger, getLogger
from typing import Union

import numpy as np
from graph_tool import Graph

from network_dismantling import dismantler_wrapper
from network_dismantling._sorters import dismantling_method
from network_dismantling.multiscale_entanglement.entanglement_functions import (entanglement,
                                                                                beta_small, beta_mid, beta_large)

method_info = {
    # "source": "",
    # "authors": "",
    "citation": "Ghavasieh, A., Stella, M., Biamonte, J. et al. Unraveling the effects of multiscale network entanglement on empirical systems. Commun Phys 4, 129 (2021). https://doi.org/10.1038/s42005-021-00633-0",
}


@dismantling_method(
    name=r"Small-scale Network Entanglement",
    short_name=r"$\mathrm{NE}_s$",

    includes_reinsertion=False,
    plot_color="#ebae07",

    **method_info,
)
@dismantler_wrapper
def network_entanglement_small(network: Graph, **kwargs):
    return entanglement(G=network, beta=beta_small, **kwargs)


@dismantling_method(
    name=r"Small-scale Network Entanglement + Reinsertion",
    short_name=r"$\mathrm{NE}_s$ + R",
    includes_reinsertion=True,
    plot_color="#ebae07",

    depends_on=network_entanglement_small,
    **method_info,
)
@dismantler_wrapper
def network_entanglement_small_reinsertion(network: Graph,
                                           stop_condition: int,
                                           network_entanglement_small: Union[list, np.ndarray],
                                           logger: Logger = getLogger("dummy"),
                                           **kwargs):
    from network_dismantling.multiscale_entanglement.reinsertion import reinsert

    predictions = reinsert(
        network=network,
        removals=network_entanglement_small,
        stop_condition=stop_condition,
        logger=logger,
    )

    return predictions


@dismantling_method(
    name=r"Mid-scale Network Entanglement",
    short_name=r"$\mathrm{NE}_m$",

    includes_reinsertion=False,
    plot_color="#34e8eb",

    **method_info,
)
@dismantler_wrapper
def network_entanglement_mid(network: Graph, **kwargs):
    return entanglement(G=network, beta=beta_mid, **kwargs)


@dismantling_method(
    name=r"Mid-scale Network Entanglement + Reinsertion",
    short_name=r"$\mathrm{NE}_m$ + R",

    includes_reinsertion=True,
    plot_color="#34e8eb",

    depends_on=network_entanglement_mid,
    **method_info,
)
@dismantler_wrapper
def network_entanglement_mid_reinsertion(network: Graph,
                                         stop_condition: int,
                                         network_entanglement_mid: Union[list, np.ndarray],
                                         logger: Logger = getLogger("dummy"),
                                         **kwargs):
    from network_dismantling.multiscale_entanglement.reinsertion import reinsert

    predictions = reinsert(
        network=network,
        removals=network_entanglement_mid,
        stop_condition=stop_condition,
        logger=logger,
    )

    return predictions


@dismantling_method(
    name=r"Large-scale Network Entanglement",
    short_name=r"$\mathrm{MSE}_l$",

    includes_reinsertion=False,
    plot_color="#ed02e9",

    **method_info,
)
@dismantler_wrapper
def network_entanglement_large(network: Graph, **kwargs):
    return entanglement(G=network, beta=beta_large, **kwargs)


@dismantling_method(
    name=r"Large-scale Network Entanglement + Reinsertion",
    short_name=r"$\mathrm{NE}_l$ + R",

    includes_reinsertion=True,
    plot_color="#ed02e9",

    depends_on=network_entanglement_large,
    **method_info,
)
@dismantler_wrapper
def network_entanglement_large_reinsertion(network: Graph,
                                           stop_condition: int,
                                           network_entanglement_large: Union[list, np.ndarray],
                                           logger: Logger = getLogger("dummy"),
                                           **kwargs):
    from network_dismantling.multiscale_entanglement.reinsertion import reinsert

    predictions = reinsert(
        network=network,
        removals=network_entanglement_large,
        stop_condition=stop_condition,
        logger=logger,
    )

    return predictions

# def network_entanglement_reinsertion(network: Graph,
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
#     name=r"Small-scale Entanglement + Reinsertion",
#     short_name=r"$\mathrm{NE}_s$ + R",
#     **method_info,
# )
# @dismantler_wrapper
# def network_entanglement_small_reinsertion(network: Graph,
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
#     name=r"Mid-scale Entanglement + Reinsertion",
#     short_name=r"$\mathrm{NE}_m$ + R",
#     **method_info,
# )
# @dismantler_wrapper
# def network_entanglement_mid_reinsertion(network: Graph,
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
#     name=r"Large-scale Entanglement + Reinsertion",
#     short_name=r"$\mathrm{NE}_l$ + R",
#     **method_info,
# )
# @dismantler_wrapper
# def network_entanglement_large_reinsertion(network: Graph,
#                                    stop_condition: int,
#                                    logger: Logger = getLogger("dummy"),
#                                    **kwargs):
#     return entanglement_reinsertion(network=network,
#                                     stop_condition=stop_condition,
#                                     entanglement_function=entanglement_large,
#                                     logger=logger,
#                                     **kwargs)
