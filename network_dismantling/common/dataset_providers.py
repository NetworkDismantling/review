import logging
from glob import glob
from pathlib import Path
from typing import List, Union, Callable

from network_dismantling.common.loaders import load_graph


def list_files(location: Union[Path, List[Path]],
               filter: str = "*",
               max_num_vertices: Union[int, None] = None,
               extensions: Union[list, str] = ("graphml", "gt"),
               **kwargs) -> List[Path]:
    files: List[Path] = []

    if isinstance(location, list):
        files = []
        for loc in location:
            try:
                files += list_files(
                    loc,
                    max_num_vertices=max_num_vertices,
                    filter=filter,
                    targets=None,
                    # manager=mp_manager,
                )
            except FileNotFoundError:
                pass

        return files

    if not isinstance(filter, list):
        filter = [filter]

    # extension = "gt"
    if not isinstance(extensions, (list, tuple)):
        extensions = [extensions]

    for extension in extensions:
        for f in filter:
            l = location / (f"{f}.{extension}")

            files += glob(str(l))

    # files = sorted([Path(file).stem for file in files])
    files = sorted([Path(file).resolve() for file in files])

    if max_num_vertices is not None:
        files = [file for file in files if load_graph(str(file)).num_vertices() <= max_num_vertices]

    if len(files) == 0:
        raise FileNotFoundError

    return files


def storage_provider(location,
                     max_num_vertices=None,
                     filter="*", extensions: Union[list, str] = ("graphml", "gt"),
                     callback: Union[Callable, None] = None,
                     logger=logging.getLogger("dummy"),
                     ):
    if not location.is_absolute():
        location = location.resolve()

    if not location.exists():
        raise FileNotFoundError(f"Location {location} does not exist.")
    elif not location.is_dir():
        raise FileNotFoundError(f"Location {location} is not a directory.")

    if not isinstance(filter, list):
        filter = [filter]

    # extension = "gt"
    if not isinstance(extensions, (list, tuple)):
        extensions = [extensions]

    files = []
    for extension in extensions:
        for f in filter:
            loc = location / f"{f}.{extension}"
            files += glob(str(loc))

    files = sorted(files)

    if len(files) == 0:
        raise FileNotFoundError

    networks = list()
    for file in files:
        filename = Path(file).stem

        network = load_graph(file)

        if (max_num_vertices is not None) and (network.num_vertices() > max_num_vertices):
            continue

        # assert not network.is_directed()
        if network.is_directed():
            logger.warning(f"Network {filename} is directed. Converting to undirected by ignoring edge directions.")
            network.set_directed(False)

        if "static_id" not in network.vertex_properties:
            # TODO turn into a unsigned int?
            network.vertex_properties["static_id"] = network.new_vertex_property("int",
                                                                                 vals=network.vertex_index,
                                                                                 )

        network.graph_properties["filename"] = network.new_graph_property("string", filename)

        if callback:
            callback(filename, network)

        networks.append((filename, network))

    return networks


def load_single_network(network_name: str,
                        network_path: Path,
                        max_num_vertices=None,
                        logger=logging.getLogger("dummy")):
    """Load a single network by name and validate it.
    
    Args:
        network_name: The name/stem of the network file
        network_path: The path to the network file
        max_num_vertices: Optional maximum number of vertices filter
        logger: Logger instance
        
    Returns:
        The loaded Graph or None if loading failed
    """
    logger.debug(f"Loading network: {network_name}")

    networks_provider = init_network_provider(
        location=network_path.parent,
        max_num_vertices=max_num_vertices,
        filter=f"{network_name}",
        logger=logger,
    )

    if len(networks_provider) == 0:
        logger.error(f"Network {network_name} not found!")
        return None

    if len(networks_provider) > 1:
        logger.error(f"More than one network found for {network_name}!")
        return None

    loaded_network_name, network = networks_provider[0]

    if loaded_network_name != network_name:
        logger.error(
            f"Loaded network with filename {loaded_network_name} does not match the expected filename {network_name}!"
        )
        return None

    return network


def init_network_provider(location: Union[Path, List[Path]],
                          max_num_vertices=None,
                          filter="*",
                          logger=logging.getLogger("dummy"),
                          **kwargs):
    if not isinstance(location, list):
        location = [location]

    # location = [Path(loc).resolve() for loc in location]
    # logger.debug(f"Loading networks with filter {filter} from: {', '.join([str(l) for l in location])}")

    networks = []
    for loc in location:
        loc = Path(loc).resolve()

        # logger.info(f"Loading networks from: {loc}")
        try:
            networks += storage_provider(loc,
                                         max_num_vertices=max_num_vertices,
                                         filter=filter,
                                         **kwargs)
        except FileNotFoundError:
            # Assume we can find the file somewhere else
            pass

    return networks
