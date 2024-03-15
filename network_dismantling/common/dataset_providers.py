import logging
from glob import glob
from pathlib import Path
from typing import Union, List

from network_dismantling.common.loaders import load_graph


def list_files(location, filter="*", extensions: Union[list, str] = ("graphml", "gt"), **kwargs):
    if not isinstance(filter, list):
        filter = [filter]

    # extension = "gt"
    if not isinstance(extensions, (list, tuple)):
        extensions = [extensions]

    files = []
    for extension in extensions:
        for f in filter:
            l = location / (f"{f}.{extension}")

            files += glob(str(l))

    # files = sorted([Path(file).stem for file in files])
    files = sorted([Path(file) for file in files])

    if len(files) == 0:
        raise FileNotFoundError

    return files


def storage_provider(location, max_num_vertices=None, filter="*", extensions: Union[list, str] = ("graphml", "gt"),
                     callback=None):
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

        assert not network.is_directed()

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


def init_network_provider(location: Union[Path, List[Path]], max_num_vertices=None, filter="*",
                          logger=logging.getLogger("dummy"), **kwargs):
    if not isinstance(location, list):
        location = [location]

    # location = [Path(loc).resolve() for loc in location]
    # logger.debug(f"Loading networks with filter {filter} from: {', '.join([str(l) for l in location])}")

    networks = []
    for loc in location:
        loc = Path(loc).resolve()

        # logger.info(f"Loading networks from: {loc}")
        try:
            networks += storage_provider(loc, max_num_vertices=max_num_vertices, filter=filter)
        except FileNotFoundError:
            # Assume we can find the file somewhere else
            pass

    return networks
