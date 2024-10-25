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
from typing import Dict

import networkx as nx
import numpy as np
from network_dismantling.common.multiprocessing import TqdmLoggingHandler
from network_dismantling.converter import get_io_helpers, get_supported_exts


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
        if args.input.is_dir():
            args.output = args.input / "converted"
        elif args.input.is_file():
            args.output = args.input.with_suffix(f".converted.{args.output_ext}")

        args.output = args.input

    if args.output.is_dir() and not args.output.exists():
        args.output.mkdir(parents=True)

    if not isinstance(args.input_ext, list):
        args.input_ext = [args.input_ext]

    if not args.output.exists():
        args.output.mkdir(parents=True)

    files = []
    if args.input.is_file():
        files.append(args.input)
    else:
        for ext in args.input_ext:
            files.extend(glob(str(args.input / ("*." + ext))))

    create_using = nx.Graph

    for file in files:
        logger.info("----------\nfile {}".format(file))

        network = None

        file = Path(file)

        e = np.loadtxt(file, dtype=int)  # read edge_list

        edge_list = []
        G = nx.Graph()
        for m in range(len(e)):
            G.add_edge(*e[m])
            edge_list.append(e[m])

        A = nx.to_numpy_array(G)
        network = nx.from_numpy_array(A)

        # print("edge_list", edge_list)
        # A = np.array(edge_list)
        #
        # A = A - A.min()
        # network = nx.from_edgelist(A, create_using=create_using)
        #
        # print("G:", G)
        # print("A:", A)
        # print("network:", network)

        assert network.number_of_edges() == A.shape[0], \
            f"Number of edges does not match after loading the network: {network.number_of_edges()} != {A.shape[0]}"

        assert network.number_of_nodes() == A.max() + 1, \
            "Number of nodes does not match after loading the network"

        assert A.min() == 0, "Minimum node id is not 0 but {}".format(A.min())

        assert A.max() == network.number_of_nodes() - 1, \
            "Maximum node id is not the number of nodes minus 1 but {}".format(A.max())

        # Check for index contiguity
        assert np.unique(A).shape[0] == network.number_of_nodes(), \
            "Node ids are not contiguous"

        centrality_file = file.parent / "centrality.npy"
        numpy_centrality: Dict = np.load(str(centrality_file),
                                         allow_pickle=True,
                                         ).item()

        print(numpy_centrality["VE"])
        logger.info(f"Loaded centrality measures from {centrality_file}:\n"
                    # f"{numpy_centrality.dtype}\n"
                    # f"{numpy_centrality}"
                    )

        for centrality_measure, centrality_dict in numpy_centrality.items():
            logger.info(f"Adding {centrality_measure} with values {centrality_dict} to the network")

            assert len(centrality_dict) == network.number_of_nodes(), \
                "Number of nodes does not match after loading centrality measures"
            assert all([node in network for node in centrality_dict.keys()]), \
                "Nodes in centrality measures not found in the network"

            nx.set_node_attributes(network,
                                   values=centrality_dict,
                                   name=centrality_measure,
                                   )

        output_file = Path(args.output) / file.with_suffix("." + args.output_ext)

        if args.output_ext == "gt":
            gt = to_graphtool(network)

            assert gt.num_vertices() == network.number_of_nodes(), "Number of nodes does not match after graph-tool conversion"
            assert gt.num_edges() == network.number_of_edges(), "Number of edges does not match after graph-tool conversion"

            logger.info(f"Saving to {output_file}")
            logger.info(f"Number of nodes: {gt.num_vertices()}")
            logger.info(f"Number of edges: {gt.num_edges()}")
            logger.info(f"Vertex properties: {list(gt.vertex_properties.keys())}")

            gt.save(str(output_file))

        else:
            try:
                _, writer = get_io_helpers(ext=args.output_ext)

                writer(network, str(output_file))  # , data=(args.no_weights is False))
            except ValueError as e:
                logger.exception(e)
                continue


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(TqdmLoggingHandler())

    parser = argparse.ArgumentParser(
        prog="python network_dismantling/converter.py",
        description="Converts undirected and unweighted networks between different formats",
    )

    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=None,
        help="Location of the input network (single file) or of the directory containing the networks to convert",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        required=False,
        help="Location of the output directory. By default, it will be the same as the input directory",
    )

    # parser.add_argument(
    #     "-nw",
    #     "--no_weights",
    #     default=False,
    #     action='store_true',
    #     help="Discard weights",
    # )

    parser.add_argument(
        "-ei",
        "--input_ext",
        type=str,
        nargs="*",
        default=["txt"],
        choices=["txt"],
        # default=sorted(get_supported_exts()),
        # choices=sorted(get_supported_exts()),
        required=False,
        help="Input extension without dot. Required if input is a directory",
    )

    parser.add_argument(
        "-eo",
        "--output_ext",
        type=str,
        default=None,
        choices=sorted(list(get_supported_exts()) + ["gt"]),
        required=True,
        help="Output file extension without dot",
    )

    args, cmdline_args = parser.parse_known_args()

    for i, input_format in enumerate(args.input_ext):
        if input_format.count(".") > 0:
            args.input_ext[i] = input_format.replace(".", "")

        # if input_format not in get_supported_exts():
        #     exit(f"Input format {input_format} not supported")

    if args.output_ext.count(".") > 0:
        args.output_ext = args.output_ext.replace(".", "")

    # if args.output_ext not in get_supported_exts():
    #     exit(f"Output format {args.output_ext} not supported")

    main(args)
