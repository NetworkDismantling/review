#   This file is part of GDM (Graph Dismantling with Machine learning),
#   proposed in the paper "Machine learning dismantling and
#   early-warning signals of disintegration in complex systems"
#   by M. Grassia, M. De Domenico and G. Mangioni.
#
#   GDM is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   GDM is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with GDM.  If not, see <http://www.gnu.org/licenses/>.

import argparse
from decimal import Decimal
from glob import glob
from pathlib import Path

import torch.multiprocessing as multiprocessing
from tqdm import tqdm

from network_dismantling.GDM.training_data_extractor import training_data_extractor
from network_dismantling.common.loaders import load_graph


def main(args):
    # Create the Log Queue
    mp_manager = multiprocessing.Manager()

    def logger(msg):
        print(msg)

    logger("Generating dataset from files in {}".format(args.directory))

    output_path = args.output if args.output is not None else args.directory / "dataset"

    if args.removals_num:
        k_range = [args.removals_num]
        threshold = 0
        target_property_name = "k_{}".format(args.removals_num)
    else:
        k_range = None
        target_property_name = "t_{}".format(args.threshold)
        threshold = args.threshold

    if not output_path.exists():
        output_path.mkdir(parents=True)

    if not isinstance(args.filter, list):
        args.filter = [args.filter]

    extensions = ["graphml", "gt"]

    if not isinstance(extensions, (list, tuple)):
        extensions = [extensions]

    files = []
    for extension in extensions:
        for f in args.filter:
            l = args.directory / (f"{f}.{extension}")
            files += glob(str(l))

    with mp_manager.Pool(processes=args.jobs, initializer=tqdm.set_lock, initargs=(mp_manager.RLock(),)) as p:

        for file in files:
            file = Path(file)

            output_file = file  # output_path / (file.stem + ".graphml")

            if output_file.exists():
                net_file = str(output_file)

                if args.update_features:
                    print("Updating features of file {}".format(file.stem))

                    compute_targets = False
                else:
                    # print("File {} already processed!".format(file.stem))
                    print("Processing file {}".format(file.stem))
                    compute_targets = args.targets

            else:
                print("Processing file {}".format(file.stem))
                net_file = str(file)
                compute_targets = args.targets

            logger("Reading file")
            g = load_graph(net_file)

            # Store static ID of the nodes
            g.vertex_properties["static_id"] = g.new_vertex_property("int", vals=g.vertex_index)

            if "features" in g.vertex_properties.keys():
                del g.vertex_properties["features"]

            if "features" in g.graph_properties.keys():
                del g.graph_properties["features"]

            if target_property_name in g.vertex_properties.keys():
                compute_targets = False

                if g.vertex_properties[target_property_name].value_type() == "string":
                    target_property = g.vertex_properties[target_property_name]
                    new_target_property = g.new_vertex_property("float")

                    for v in g.get_vertices():
                        if len(target_property[v]) > 0:
                            new_target_property[v] = float(target_property[v])

                    g.vertex_properties[target_property_name] = new_target_property
                    print("Target was string!")

                print("Target already found in file {}. Skipping computation!".format(file.stem))

            # TODO !

            from network_dismantling.common.data_structures import DefaultDict
            features_to_compute = DefaultDict(False)
            features_to_compute["degree"] = True
            features_to_compute["clustering_coefficient"] = True
            features_to_compute["kcore"] = True
            features_to_compute["chi_degree"] = True

            p.apply_async(func=training_data_extractor,
                          args=(
                              g,
                              threshold,
                              str(output_file),
                          ),
                          kwds=dict(
                              compute_targets=compute_targets,
                              target_property_name=target_property_name,
                              k_range=k_range,
                              features=features_to_compute,
                              logger=print
                          ),
                          callback=print,
                          error_callback=print)

        p.close()
        p.join()


if __name__ == '__main__':
    # multiprocessing.freeze_support()  # for Windows support

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     argument_default=argparse.SUPPRESS)
    required = parser.add_argument_group('Required arguments')
    optional = parser.add_argument_group('Optional arguments')

    parser.add_argument(
        "-d",
        "--directory",
        type=Path,
        required=True,
        help="Location of the dataset (directory)",
    )
    parser.add_argument(
        "-F",
        "--filter",
        type=str,
        required=False,
        default="*",
        help="Filter input folder files by pattern provided",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        required=False,
        help="Location of the output (directory)",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        required=False,
        help="Number of parallel jobs to dispatch",
    )
    parser.add_argument(
        "-t",
        "--targets",
        default=False,
        required=False,
        action="store_true",
        help="Compute targets",
    )
    parser.add_argument(
        "-uf",
        "--update-features",
        default=False,
        required=False,
        action="store_true",
        help="Updates features of already existing files",
    )
    parser.add_argument(
        "-k",
        "--removals_num",
        type=int,
        default=None,
        required=False,
        help="[TARGET] Exact num of removals to perform",
    )

    parser.add_argument(
        "-T",
        "--threshold",
        type=Decimal,
        default=Decimal("0.1"),
        required=False,
        help="[TARGET] Dismantling threshold",
    )

    args = parser.parse_args()

    main(args)
