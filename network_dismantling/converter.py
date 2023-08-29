#   This file is part of the Network Dismantling review,
#   proposed in the paper "Robustness and resilience of complex networks"
#   by Oriol Artime, Marco Grassia, Manlio De Domenico, James P. Gleeson,
#   Hernán A. Makse, Giuseppe Mangioni, Matjaž Perc and Filippo Radicchi.
#
#   This is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   The project is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with the code.  If not, see <http://www.gnu.org/licenses/>.

import argparse
import logging
from glob import glob
from pathlib import Path

import networkx as nx

from network_dismantling.common.multiprocessing import TqdmLoggingHandler

_format_mapping = {
    # "ncol":       ("Read_Ncol", "write_ncol"),
    # "lgl":        ("Read_Lgl", "write_lgl"),
    # "graphdb":    ("Read_GraphDB", None),
    # "graphmlz":     ("read_graphml", "write_graphmlz"),
    "graphml": ("read_graphml", "write_graphml"),
    # "gml":        ("Read_GML", "write_gml"),
    # "dot":          (None, "write_dot"),
    # "graphviz":   (None, "write_dot"),
    "net": ("read_pajek", "write_pajek"),
    "pajek": ("read_pajek", "write_pajek"),
    # "dimacs":     ("Read_DIMACS", "write_dimacs"),
    "adjacency": ("read_adjlist", "write_adjlist"),
    "adj": ("read_adjlist", "write_adjlist"),
    "edgelist": ("read_edgelist", "write_edgelist"),
    "edge": ("read_edgelist", "write_edgelist"),
    "edges": ("read_edgelist", "write_edgelist"),
    "el": ("read_edgelist", "write_edgelist"),
    "pickle": ("read_gpickle", "write_gpickle"),
    "picklez": ("read_gpickle", "write_gpickle"),
    # "svg":        (None, "write_svg"),
    # "gw":         (None, "write_leda"),
    # "leda":       (None, "write_leda"),
    # "lgr":        (None, "write_leda")
}


def get_supported_exts():
    return _format_mapping.keys()


def get_io_helpers(file=None, ext=None):
    from pathlib import Path

    if file is not None:
        ext = Path(file).suffix[1:]
    elif ext is None:
        raise ValueError("No parameter is provided")

    try:
        methods = _format_mapping[ext]
    except KeyError as e:
        raise ValueError("Format not supported {}".format(e))

    return getattr(nx.readwrite, methods[0]), getattr(nx.readwrite, methods[1])


def to_graphtool(g):
    from io import BytesIO
    from networkx import write_graphml
    from network_dismantling.common.loaders import load_graph

    with BytesIO() as io_buffer:
        write_graphml(g, io_buffer)
        io_buffer.seek(0)

        try:
            gt = load_graph(io_buffer, fmt="graphml")
        except Exception as e:
            raise e
    return gt


def main(args):
    if args.output is None:
        args.output = args.input

    if not args.output.is_dir():
        exit("Error: output must be a directory!")

    if not args.output.exists():
        args.output.mkdir(parents=True)

    if not isinstance(args.input_ext, list):
        args.input_ext = [args.input_ext]

    if not args.output.exists():
        args.output.mkdir(parents=True)

    files = []
    for ext in args.input_ext:
        files.extend(glob(str(args.input / ("*." + ext))))

    create_using = nx.Graph

    for file in files:
        logger.info("----------\nfile {}".format(file))

        network = None

        file = Path(file)

        reader, _ = get_io_helpers(ext=file.suffix[1:])

        # TODO IMPROVE ME
        try:
            try:
                network = reader(str(file), create_using=create_using, data=(('weight', float),))
            except TypeError as e:
                # print(e)
                try:
                    network = reader(str(file), create_using=create_using)
                except TypeError as e:

                    try:
                        network = reader(str(file))
                    except TypeError as e:
                        logger.exception(e)
        except Exception as e:
            logger.exception(e)
            continue
            # exit("Error reading file {}".format(e))

        output_file = Path(args.output) / file.with_suffix("." + args.output_ext)

        if args.output_ext == "gt":
            gt = to_graphtool(network)

            assert gt.num_vertices() == network.number_of_nodes(), "Number of nodes does not match after graph-tool conversion"
            assert gt.num_edges() == network.number_of_edges(), "Number of edges does not match after graph-tool conversion"

            gt.save(str(output_file))

        else:
            try:
                _, writer = get_io_helpers(ext=args.output_ext)

                writer(network, str(output_file), data=(args.no_weights is False))
            except ValueError as e:
                logger.exception(e)
                continue


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(TqdmLoggingHandler())

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=None,
        help="Location of the input networks (directory)",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        required=False,
        help="Location of the output directory",
    )

    parser.add_argument(
        "-nw",
        "--no_weights",
        default=False,
        action='store_true',
        help="Discard weights",
    )

    parser.add_argument(
        "-ei",
        "--input_ext",
        type=str,
        nargs="*",
        default=sorted(get_supported_exts()),
        required=False,
        help="Input extension without dot",
    )

    parser.add_argument(
        "-eo",
        "--output_ext",
        type=str,
        default=None,
        required=True,
        help="Output file extension without dot",
    )

    args, cmdline_args = parser.parse_known_args()

    main(args)
