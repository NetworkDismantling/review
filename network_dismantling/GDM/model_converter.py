#   This file is part of GDM (Graph Dismantling with Machine learning),
#   proposed in the paper "Machine learning dismantling and
#   early-warning signals of disintegration in complex systems"
#   by M. Grassia, M. De Domenico and G. Mangioni.
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

import logging
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch import Tensor
from tqdm.auto import tqdm

from network_dismantling.common.multiprocessing import TqdmLoggingHandler


def main(args):
    args.input = args.input.resolve()
    args.output = args.output.resolve()

    if not args.output.exists():
        args.output.mkdir(parents=True)

    logger.info(f"Converting models from: {args.input}")
    logger.info(f"Storing converted models to {args.output}")

    files = args.input.glob(f"*.{args.extension}")
    files = sorted(files)

    # noinspection PyTypeChecker
    for model_path in tqdm(files,
                           desc="Converting models",
                           leave=False,
                           position=0,
                           ):
        logger.info(f"Converting model: {model_path}")

        weights = torch.load(str(model_path),
                             map_location="cpu",
                             # map_location=args.device,
                             )

        logger.info(weights.keys())

        for key, value in list(weights.items()):
            if not key.startswith("convolutional_layers."):
                continue

            if key.endswith(".weight"):
                new_key = key.replace(".weight", ".lin_l.weight")
                # value.data = value.data.t()
                value = value.t()
                weights[new_key] = value

                del weights[key]

                continue

            if key.endswith(".bias"):
                # Nothing to do here
                continue

            if key.endswith(".att"):
                print(f"weights[{key}] = {weights[key].shape}")

                new_key_l = key.replace(".att", ".att_l")
                att_l: Tensor = value
                # att_l.data = att_l.data[:, :, :att_l.shape[2] // 2]
                att_l = att_l[:, :, :att_l.shape[2] // 2]
                weights[new_key_l] = att_l

                new_key_r = key.replace(".att", ".att_r")
                att_r = value
                # att_r.data = att_r.data[:, :, att_r.shape[2] // 2:]
                att_r = att_r[:, :, att_r.shape[2] // 2:]
                weights[new_key_r] = att_r

                del weights[key]

                continue

            if key.endswith(".lin_l"):
                new_key = key.replace(".lin_l", ".lin_r")
                weights[new_key] = value

                continue

        model_path: Path
        # Store the model
        model_name = model_path.name
        model_path = args.output / model_name

        if model_path.exists():
            logger.warning(f"Model {model_path} already exists. Skipping. Remove it to convert again.")

            # continue
        else:
            pass
        torch.save(weights, str(model_path))

        # exit()
        # # Lets try to load the model
        # weights = torch.load(str(model_path),
        #                      map_location="cpu",
        #                      # map_location=args.device,
        #                      )
        #
        # print(model_path.stem)
        #
        # # Instantiate the corresponding GATModel
        # model = GAT_Model.from_model_name(model_path.stem)
        #
        # model.load_state_dict(weights)
        #
        # break
        # # for conv_layer, conv_layer_old in zip(model.convolutional_layers_new, model.convolutional_layers):
        # #     out_channels = conv_layer.out_channels
        # #
        # #     assert out_channels == conv_layer_old.out_channels, "Mismatching output size"
        # #
        # #     conv_layer.bias.data = conv_layer_old.bias.data
        # #     conv_layer.lin_l.weight.data = conv_layer_old.weight.data.t()
        # #
        # #     conv_layer.att_l.data = conv_layer_old.att[:, :, out_channels:].data
        # #     conv_layer.att_r.data = conv_layer_old.att[:, :, :out_channels].data
        # #
        # #     conv_layer.lin_r = conv_layer.lin_l
        # #     # conv_layer.att_l.data = conv_layer_old.att[:, :, :out_channels].data
        # #     # conv_layer.att_r.data = conv_layer_old.att[:, :, out_channels:].data
        # #
        # # del model.convolutional_layers


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(TqdmLoggingHandler())

    parser = ArgumentParser(
        description="Converts a GDM model file into a the new PyG model format"
    )

    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=None,
        required=True,
        help="Input file location",
    )

    parser.add_argument(
        "-e",
        "--extension",
        type=str,
        default="h5",
        required=False,
        help="Input model file extension",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        required=True,
        help="Output file location",
    )

    args = parser.parse_args()

    assert args.input.exists(), f"Input file does not exist: {args.input}"
    assert args.input.is_dir(), f"Input file is not a directory: {args.input}"
    assert args.input != args.output, f"Input and output directories are the same: {args.input}"

    main(args)
