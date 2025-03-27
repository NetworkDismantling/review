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
import logging
from copy import deepcopy
from operator import itemgetter
from pathlib import Path
from random import seed

import numpy as np
import torch
from graph_tool import Graph
from scipy.integrate import simpson
from torch_geometric import seed_everything
from torch_geometric.data import Data

from network_dismantling.GDM.config import all_features, threshold
from network_dismantling.GDM.dataset_providers import storage_provider, prepare_graph
from network_dismantling.GDM.predictors import static_predictor, get_predictions, lcc_static_predictor
from network_dismantling.common.dismantlers import lcc_peak_dismantler
from network_dismantling.common.external_dismantlers.lcc_threshold_dismantler import \
    lcc_threshold_dismantler as external_lcc_threshold_dismantler, threshold_dismantler as external_threshold_dismantler
from network_dismantling.common.multiprocessing import clean_up_the_pool


class ModelWeightsNotFoundError(FileNotFoundError):
    pass


def train(args, model, networks_provider=None, logger=logging.getLogger('dummy')):
    logger.info(model)

    # TODO
    loss_op = torch.nn.MSELoss()

    if args.device:
        _model = model
        try:
            model.to(args.device)
        finally:
            del _model

    torch.manual_seed(args.seed_train)
    np.random.seed(args.seed_train)
    seed(args.seed_train)

    if model.is_affected_by_seed():
        model.set_seed(args.seed_train)

    # Load training networks
    if networks_provider is None:
        networks_provider = init_network_provider(args.location_train, features=args.features, targets=args.target)

    assert networks_provider is not None, "No networks provider"
    assert len(networks_provider) > 0, "No networks found."

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    model.train()

    for epoch in range(1, args.num_epochs + 1):
        total_loss = 0
        i = 0

        for i, (_, _, data) in enumerate(networks_provider, start=1):
            # num_graphs = data.num_graphs
            data.batch = None
            data = data.to(args.device)

            optimizer.zero_grad()
            loss = loss_op(model(data.x, data.edge_index), data.y)
            total_loss += loss.item()  # * num_graphs
            loss.backward()
            optimizer.step()

        loss = total_loss / i

        logger.info('Epoch: {:02d}, Loss: {}, Acc: {:.4f}'.format(epoch, loss, 0.0))


def init_network_provider(location, targets, features=None, filter="*"):
    networks_provider = storage_provider(location, filter=filter)
    network_names, networks = zip(*networks_provider)
    networks_provider = list(
        zip(network_names, networks, map(lambda n: prepare_graph(n, features=features, targets=targets), networks))
    )

    return networks_provider


@torch.no_grad()
def test(args, model, networks_provider, print_model=True, logger=logging.getLogger('dummy')):
    if print_model:
        logger.info(model)

    # torch.manual_seed(args.seed_test)
    # np.random.seed(args.seed_test)
    # seed(args.seed_test)
    seed_everything(args.seed_test)

    if model.is_affected_by_seed():
        model.set_seed(args.seed_test)

    model.eval()

    if args.peak_dismantling:
        if args.lcc_only:
            dismantler = lcc_peak_dismantler
        else:
            raise NotImplementedError
    else:
        if args.lcc_only:
            # dismantler = lcc_threshold_dismantler

            dismantler = external_lcc_threshold_dismantler
        else:
            # dismantler = threshold_dismantler

            dismantler = external_threshold_dismantler

    if args.static_dismantling:
        if args.lcc_only:
            if args.peak_dismantling:
                predictor = lcc_static_predictor
            else:
                predictor = get_predictions
        else:
            if args.peak_dismantling:
                predictor = static_predictor
            else:
                predictor = get_predictions

        args.removals_num = 0

    elif args.removals_num > 1:
        raise NotImplementedError

        # # TODO
        # def block_dynamic_predictor_wrapper(network, model, features, device, data=None):
        #     return block_dynamic_predictor(network, model, features, device, k=args.removals_num)
        #
        # predictor = block_dynamic_predictor_wrapper
    else:
        raise NotImplementedError

        # predictor = dynamic_predictor
        # args.removals_num = 1

    generator_args = {
        "model": model,
        "features": args.features,
        "device": args.device,
        "lock": args.lock,
        "logger": logger,
    }

    with torch.no_grad():

        # Init runs buffer
        runs = []

        # # noinspection PyTypeChecker
        # for filename, network, data in tqdm(networks_provider,
        #                                     desc="Testing",
        #                                     leave=False,
        #                                     # position=1,
        #                                     ):

        # TODO remove this loop, it is not used anymore in the review version
        for filename, network, data in networks_provider:
            filename: str
            network: Graph
            data: Data

            # TODO avoid deepcopy?
            # network = deepcopy(network)
            network = network.copy()
            data = deepcopy(data)

            network_size = network.num_vertices()

            generator_args["network_name"] = filename
            generator_args["data"] = data

            # Compute stop condition
            stop_condition = int(np.ceil(network_size * float(args.threshold)))

            logger.debug(f"Dismantling {filename} according to the predictions. "
                         f"Aiming to reach LCC size {stop_condition} ({stop_condition * 100 / network_size:.3f}%)"
                         )

            removals, prediction_time, dismantle_time, _ = dismantler(network, predictor, generator_args, stop_condition)

            peak_slcc = max(removals, key=itemgetter(4))

            run = {
                "network": filename,

                "removals": removals,

                "slcc_peak_at": peak_slcc[0],
                "lcc_size_at_peak": peak_slcc[3],
                "slcc_size_at_peak": peak_slcc[4],

                "r_auc": simpson(list(r[3] for r in removals), dx=1),
                "rem_num": len(removals),

                "prediction_time": prediction_time,
                "dismantle_time": dismantle_time,
            }
            add_run_parameters(args, run, model)

            runs.append(run)

            if args.verbose > 1:
                logger.info(
                    "Percolation at {}: LCC {}, SLCC {}, R {}".format(run["slcc_peak_at"], run["lcc_size_at_peak"],
                                                                      run["slcc_size_at_peak"], run["r_auc"]))

            if args.verbose == 2:
                for removal in run["removals"]:
                    logger.info("\t{}-th removal: node {} ({}). LCC size: {}, SLCC size: {}".format(*removal))

            # Fix OOM
            clean_up_the_pool()

        # runs_dataframe = pd.DataFrame(data=runs, columns=args.output_df_columns)

    return runs


def parse_parameters(nn_model):
    parser = argparse.ArgumentParser(
        description="Graph node classification using GraphSAGE"
    )
    add_arguments(nn_model, parser)

    args, cmdline_args = parser.parse_known_args()
    # args.feature_indices = [indices[x] for x in args.features]

    arguments_processing(args)

    return args


def arguments_processing(args):
    if args.seed_train is None:
        args.seed_train = args.seed
    if args.seed_test is None:
        args.seed_test = args.seed
    if args.removals_num is None:
        if args.static_dismantling:
            args.removals_num = 0
        else:
            args.removals_num = 1

    args.features = sorted(args.features)


def add_arguments(nn_model, parser):
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        dest="num_epochs",
        type=int,
        default=30,
        help="The number of epochs to train the model",
    )
    parser.add_argument(
        "-r",
        "--learning_rate",
        type=float,
        default=0.005,
        help="Initial learning rate for model training",
    )
    parser.add_argument(
        "-wd",
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay",
    )
    parser.add_argument(
        "-lm",
        "--location_train",
        type=Path,
        default=None,
        help="Location of the dataset (directory)",
    )
    parser.add_argument(
        "-lt",
        "--location_test",
        type=Path,
        default=None,
        help="Location of the dataset (directory)",
    )
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        default=None,
        required=True,
        help="The target node property",
    )
    parser.add_argument(
        "-T",
        "--threshold",
        type=float,
        default=float(threshold["test"]),
        required=False,
        help="The target threshold",
    )
    parser.add_argument(
        "-Sm",
        "--seed_train",
        nargs="*",
        type=int,
        default=None,
        help="Pseudo Random Number Generator Seed to use during training",
    )
    parser.add_argument(
        "-St",
        "--seed_test",
        nargs="*",
        type=int,
        default=None,
        help="Pseudo Random Number Generator Seed to use during tests",
    )
    parser.add_argument(
        "-S",
        "--seed",
        type=int,
        default=0,
        help="Pseudo Random Number Generator Seed",
    )
    parser.add_argument(
        "-f",
        "--features",
        type=str,
        default=["degree", "clustering_coefficient", "kcore"],
        choices=all_features + ["None"],
        nargs="+",
        help="The features to use",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        default=None,
        help="Output DataFrame file location",
    )
    parser.add_argument(
        "-SD",
        "--static_dismantling",
        default=False,
        action="store_true",
        help="[Test only] Static removal of nodes",
    )
    parser.add_argument(
        "-PD",
        "--peak_dismantling",
        default=False,
        action="store_true",
        help="[Test only] Stops the dismantling when the max SLCC size is larger than the current LCC",
    )
    parser.add_argument(
        "-LO",
        "--lcc_only",
        default=False,
        action="store_true",
        help="[Test only] Remove nodes only from LCC",
    )
    parser.add_argument(
        "-k",
        "--removals_num",
        type=int,
        default=None,
        required=False,
        help="Block size before recomputing predictions",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="Verbosity level",
    )
    parser.add_argument(
        "-FCPU",
        "--force_cpu",
        default=False,
        action="store_true",
        help="Disables ",
    )
    nn_model.add_model_parameters(parser)


def train_wrapper(args, nn_model, train_ne=True, networks_provider=None, logger=logging.getLogger('dummy'),
                  print_model=True):
    seed_everything(args.seed_train)

    model = nn_model(args)

    if print_model:
        logger.info(model)

    model.set_seed(args.seed_train)

    model_name = "F{}_{}_L{}_WD{}_E{}_S{}".format(
        '_'.join(args.features),
        model.model_name(),
        args.learning_rate,
        args.weight_decay,
        args.num_epochs,
        args.seed_train if model.is_affected_by_seed() else None
    )

    if args.verbose == 2:
        logger.info(model_name)

    models_path = args.models_location / args.location_train.name / f"{args.target}" / model.get_name()
    model_weights_file = models_path / (model_name + ".h5")

    if model_weights_file.is_file():
        try:
            weights = torch.load(str(model_weights_file),
                                 map_location=args.device,
                                 )
        except RuntimeError as e:
            logger.error(f"Error loading model weights from file: {model_weights_file}\n"
                         f"{model}",
                         exc_info=True,
                         )
            raise e

        try:
            model.load_state_dict(weights,
                                  strict=False,
                                  )
        except RuntimeError as e:
            logger.error(f"Error loading model weights from file: {model_weights_file}\n{model}", exc_info=True)
            raise e

    elif train_ne:
        if not models_path.exists():
            models_path.mkdir(parents=True)

        if networks_provider is None:
            raise ValueError("No train networks provided!")

        # Init model
        train(args, model, networks_provider, logger=logger)
        torch.save(model.state_dict(), str(model_weights_file))
    else:
        raise ModelWeightsNotFoundError(f"Model {model_weights_file} not found!")

    return model


def get_df_columns(nn_model):
    # "model",
    return \
            ["network", "features", "slcc_peak_at", "lcc_size_at_peak",
             "slcc_size_at_peak", "removals", "static", "removals_num"] + \
            nn_model.get_parameters() + \
            ["model_seed", "num_epochs", "learning_rate", "weight_decay",
             "seed", "r_auc", "rem_num", "prediction_time", "dismantle_time",
             ]


def add_run_parameters(params, run, model):
    run["learning_rate"] = params.learning_rate
    run["weight_decay"] = params.weight_decay
    run["num_epochs"] = params.num_epochs
    run["static"] = params.static_dismantling
    run["model_seed"] = params.seed_train
    run["features"] = ','.join(params.features)
    # run["seed"] = params.seed_test

    run["removals_num"] = params.removals_num

    model.add_run_parameters(run)
