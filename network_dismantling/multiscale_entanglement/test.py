import logging
import unittest
from pathlib import Path
from typing import Dict, List

import numpy as np
from graph_tool import VertexPropertyMap, Graph

import network_dismantling
from network_dismantling.common.dataset_providers import (
    # list_files,
    init_network_provider,
)
from network_dismantling.multiscale_entanglement.entanglement_functions import (
    entanglement_small as entanglement_small_new,
)
from network_dismantling.multiscale_entanglement.original_entanglement_functions import (
    entanglement_small as entanglement_small_original,
)


def to_networkx(g: Graph):
    from io import BytesIO
    from networkx import read_graphml, relabel_nodes

    logger.info("Converting graph to NetworkX")
    with BytesIO() as io_buffer:
        g.save(io_buffer, fmt='graphml')

        io_buffer.seek(0)

        try:
            gn = read_graphml(io_buffer, node_type=str)
        except Exception as e:
            raise e

    # Map nodes to consecutive IDs to avoid issues with FINDER
    mapping = {k: i for i, k in enumerate(gn.nodes)}

    gn = relabel_nodes(gn, mapping)

    return gn


test_data_path = "./dataset/unit_test_data/"

base_path = Path(network_dismantling.__file__)
base_path = base_path.parent  # Remove __init__.py
base_path = base_path.parent  # Remove network_dismantling

test_data_path = Path(test_data_path)
test_data_path = base_path / test_data_path
test_data_path = test_data_path.resolve()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

print(f"Test data path: {test_data_path}")
networks_provider = init_network_provider(
    location=test_data_path,
    filter=f"*",
    logger=logger,
)


class MyTestCase(unittest.TestCase):
    def test_entanglement_small(self):
        if (networks_provider is None) or (len(networks_provider) == 0):
            self.fail(f"No networks found in the dataset directory {test_data_path}")

        for network_name, network in networks_provider:

            with self.subTest(network_name=network_name):
                print(f"{network_name}: starting testing")

                print(f"{network_name}: calculating new entanglement")
                new_entanglement: VertexPropertyMap = entanglement_small_new(network)
                new_entanglement: np.ndarray = new_entanglement.get_array()
                new_entanglement: List[float] = new_entanglement.tolist()

                print(f"{network_name}: converting to networkx")
                networkx_network = to_networkx(network)

                print(f"{network_name}: calculating original entanglement")
                original_entanglement: Dict[int, float] = entanglement_small_original(networkx_network)
                original_entanglement: List[float] = list(original_entanglement.values())

                try:
                    np.testing.assert_almost_equal(original_entanglement,
                                                   new_entanglement,
                                                   decimal=7,
                                                   err_msg='',
                                                   verbose=True,
                                                   )

                except AssertionError as e:
                    logger.error(f"{network_name}: original and new entanglement are not equal")
                    print(f"{network_name}: original and new entanglement are not equal")

                    for i, (o, n) in enumerate(zip(original_entanglement, new_entanglement)):
                        if not np.isclose(o, n, atol=1e-7):
                            print(f"{i}: {o} != {n}")

                    self.fail(f"{network_name}: "
                              f"original entanglement: {original_entanglement} != \n"
                              f"new entanglement: {new_entanglement}"
                              )

                logger.info(f"{network_name}: original and new entanglement are equal")
                print(f"{network_name}: original and new entanglement are equal")


if __name__ == '__main__':
    unittest.main()
