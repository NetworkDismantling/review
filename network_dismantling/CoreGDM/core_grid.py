#   This file is part of CoreGDM (Core Graph Dismantling with Machine learning),
#   proposed in the paper " CoreGDM: Geometric Deep Learning Network Decycling
#   and Dismantling"  by M. Grassia and G. Mangioni.
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
#   along with CoreGDM.  If not, see <http://www.gnu.org/licenses/>.

import argparse
import logging
import threading
from collections import defaultdict
from functools import partial
from itertools import combinations
from pathlib import Path
from queue import Queue

import network_dismantling
import numpy as np
import pandas as pd
from graph_tool.all import remove_parallel_edges, remove_self_loops
from network_dismantling.CoreGDM.core_network_dismantler import get_df_columns
from network_dismantling.GDM.config import all_features
from network_dismantling.GDM.config import base_models_path
from network_dismantling.GDM.dataset_providers import init_network_provider
from network_dismantling.GDM.models import models_mapping
from network_dismantling.GDM.network_dismantler import ModelWeightsNotFoundError
from network_dismantling.GDM.training_data_extractor import training_data_extractor
from network_dismantling.common.config import output_path, base_dataframes_path
from network_dismantling.common.data_structures import product_dict
from network_dismantling.common.dataset_providers import list_files
from network_dismantling.common.multiprocessing import (
    dataset_writer,
    apply_async,
    progressbar_thread,
    TqdmLoggingHandler,
)
from torch import multiprocessing, cuda
from tqdm import tqdm


def process_parameters_wrapper(
        args,
        df,
        nn_model,
        params_queue,
        train_networks,
        test_networks,
        df_queue,
        iterations_queue,
        early_stopping_dict,
        logger=logging.getLogger("dummy"),
):
    import logging

    from queue import Empty
    from torch import device
    from tqdm.auto import tqdm

    from network_dismantling.common.multiprocessing import clean_up_the_pool
    from network_dismantling.common.multiprocessing import get_position
    from network_dismantling.CoreGDM.core_network_dismantler import (
        add_run_parameters,
        train_wrapper,
        test,
    )

    logger.setLevel(logging.INFO)
    # logger.addHandler(TqdmLoggingHandler())
    # logger.propagate = False

    child_number = get_position()

    all_runs = []

    runtime_exceptions = 0
    while True:
        try:
            params = params_queue.get_nowait()
        except Empty:
            break

        if params is None:
            break

        for key in vars(args):
            value = getattr(args, key)
            params.setdefault(key, value)

        key = "_".join(params.features)
        # key = '_'.join(sorted(params.features))

        available_device = child_number % len(args.devices)
        available_device = args.devices[available_device]
        # logger.info(f"Using device {available_device}")

        params.lock = args.locks[available_device]
        params.device = device(available_device)

        # TODO new data structure instead of dict[key] ?

        # Train the model
        # Delay model loading to improve performances if everything is already done
        # model = None
        #
        try:
            model = train_wrapper(
                params,
                nn_model=nn_model,
                networks_provider=train_networks[key],
                train_ne=(not args.dont_train),
                print_model=False,
                logger=logger,
            )

            if params.device:
                model.to(params.device)

        except ModelWeightsNotFoundError as e:
            runtime_exceptions += 1
            iterations_queue.put(1)

            continue
            # raise e
        except (RuntimeError, FileNotFoundError) as e:
            logger.error("ERROR: {}".format(e), exc_info=True)

            runtime_exceptions += 1

            iterations_queue.put(1)

            continue
            # raise e

        # TODO improve me
        filter = {}
        add_run_parameters(params, filter, model)
        df_filtered = df.loc[
            (df[list(filter.keys())] == list(filter.values())).all(axis="columns"),
            ["network", "seed"],
        ]

        # noinspection PyTypeChecker
        for name, network, data in tqdm(
                test_networks[key],
                desc="Networks",
                leave=False,
        ):
            network_df = df_filtered.loc[(df_filtered["network"] == name)]

            if nn_model.is_affected_by_seed():
                tested_seeds = network_df["seed"].unique()

                seeds_to_test = set(args.seed_test) - set(tested_seeds)
                seeds_to_test = sorted(seeds_to_test)

            else:
                if len(network_df) == 0:
                    seeds_to_test = [next(iter(args.seed_test))]
                else:
                    # Nothing to do. Network was already tested (and seed doesn't matter)
                    continue

            for seed_test in seeds_to_test:
                params.seed_test = seed_test

                try:
                    # if model is None:
                    #     model = train_wrapper(params, nn_model=nn_model, networks_provider=train_networks[key], print=logger)

                    # Test
                    runs = test(
                        params,
                        model=model,
                        networks_provider=[
                            (name, network, data),
                        ],
                        early_stopping_dict=early_stopping_dict,
                        logger=logger,
                        # print_model=True,
                        print_model=False,
                    )
                    all_runs += runs

                except RuntimeError as e:
                    logger.error(f"Runtime error: {e}", exc_info=True)

                    runtime_exceptions += 1

                    # raise e
                    continue

                runs_dataframe = pd.DataFrame(data=runs, columns=args.output_df_columns)

                if "file" in runs_dataframe.columns:
                    runs_dataframe = runs_dataframe.drop(columns=["file"])

                df_queue.put(runs_dataframe)

                # runs_dataframe["idx"] = range(len(runs_dataframe))
                #
                # pd.concat(objs=[network_df,
                #                 runs_dataframe,
                #                 ],
                #           axis=1,
                #           )
                clean_up_the_pool()

        # TODO fix OOM
        del model
        clean_up_the_pool()

        iterations_queue.put(1)

    if runtime_exceptions > 0:
        logger.warning(
            "\n\n\n"
            "WARNING: Some runs did not complete due to some runtime exception (most likely CUDA OOM)."
            "Try again with lower GPU load."
            "\n\n\n"
        )

    return all_runs


def main(args, nn_model):
    # try:
    #     if cuda.is_available():
    #         multiprocessing.set_start_method('spawn')
    # except RuntimeError:
    #     pass

    parameters_to_try = args.parameters + nn_model.get_parameters() + ["seed_train"]

    # Get subset of args dictionary
    parameters_to_try = {k: vars(args)[k] for k in parameters_to_try}

    if args.output_file.exists():
        df = pd.read_csv(args.output_file)
    else:
        df = pd.DataFrame(columns=args.output_df_columns)

    del df["removals"]

    # Init network providers
    if not args.dont_train:
        train_networks = init_network_provider(
            args.location_train,
            max_num_vertices=None,
            features_list=args.features,
            targets=args.target,
            # manager=mp_manager
        )
        # logger.debug(f"Train networks {len(train_networks)}: {train_networks}")
    else:
        train_networks = defaultdict(list)

    test_networks_list = list_files(
        args.location_test,
        max_num_vertices=args.max_num_vertices,
        features_list=args.features,
        filter=args.test_filter,
        targets=None,
        # manager=mp_manager,
    )

    # logger.debug(f"Test network list: {len(test_networks_list)} {test_networks_list}")

    # List the parameters to try
    params_list = list(
        product_dict(
            _callback=nn_model.parameters_combination_validator, **parameters_to_try
        )
    )

    # Create the Multiprocessing Manager
    mp_manager = multiprocessing.Manager()

    # Init queues
    df_queue: Queue = mp_manager.Queue()
    params_queue: Queue = mp_manager.Queue()
    # device_queue: Queue = mp_manager.Queue()
    iterations_queue: Queue = mp_manager.Queue()

    early_stopping_dict = mp_manager.dict()

    for network_name in df["network"].unique():
        network_df = df.loc[(df["network"] == network_name)]

        early_stopping_dict[network_name] = {
            "auc": network_df["r_auc"].min() or np.inf,
            "rem_num": network_df["rem_num"].min() or np.inf,
        }

    # Create and start the Dataset Writer Thread
    dp = threading.Thread(
        target=dataset_writer,
        args=(df_queue, args.output_file),
        daemon=True,
    )
    dp.start()

    devices = []
    locks = dict()

    logger.info(f"Using package {network_dismantling.__file__}")
    if cuda.is_available() and not args.force_cpu:
        logger.info("Using GPU(s).")
        for device in range(cuda.device_count()):
            device = "cuda:{}".format(device)
            devices.append(device)
            locks[device] = mp_manager.BoundedSemaphore(args.simultaneous_access)
    else:
        logger.info("Using CPU.")
        device = "cpu"
        devices.append(device)
        locks[device] = mp_manager.BoundedSemaphore(args.simultaneous_access)

    # # Put all available devices in the queue
    # for device in devices:
    #     for _ in range(args.simultaneous_access):
    #         # Allow simultaneous access to the same device
    #         device_queue.put(device)

    args.devices = devices
    args.locks = locks

    for network_name in tqdm(
            test_networks_list,
            desc="Networks",
    ):
        for features in tqdm(
                args.features,
                desc="Features",
        ):
            logger.info(f"Loading network: {network_name} with features: {features}")

            def storage_provider_callback(filename, network):
                network_size = network.num_vertices()
                stop_condition = int(np.ceil(network_size * float(args.threshold)))
                logger.info(
                    f"Dismantling {filename} according to the predictions. "
                    f"Aiming to reach LCC size {stop_condition} ({stop_condition * 100 / network_size:.3f}%)"
                )

                # CoreHD does not support nor parallel edges nor self loops.
                # Remove them.
                remove_parallel_edges(network)
                remove_self_loops(network)

                training_data_extractor(
                    network,
                    compute_targets=False,
                    features=features,
                    # logger=print,
                )

            pp_test_networks = init_network_provider(
                args.location_test,
                features_list=[features],
                filter=f"{network_name}",
                targets=None,
                # manager=mp_manager,
                callback=storage_provider_callback,
            )
            # logger.info(f"Test network LOADED: {len(test_networks)}: {test_networks}")
            # Fill the params queue
            # TODO ANY BETTER WAY?
            for i, params in enumerate(params_list):
                # device = devices[i % len(devices)]
                # params.device = device
                # params.lock = locks[device]
                params_queue.put(params)

            # Create the pool
            with multiprocessing.Pool(
                    processes=args.jobs,
                    initializer=tqdm.set_lock,
                    initargs=(multiprocessing.Lock(),),
            ) as p:
                with tqdm(
                        total=len(params_list),
                        leave=False,
                        desc="Parameters",
                ) as pb:
                    # Create and start the ProgressBar Thread
                    pbt = threading.Thread(
                        target=progressbar_thread,
                        args=(
                            iterations_queue,
                            pb,
                        ),
                        daemon=True,
                    )
                    pbt.start()

                    for i in range(args.jobs):
                        # torch.cuda._lazy_init()

                        # p.apply_async(
                        apply_async(
                            pool=p,
                            func=process_parameters_wrapper,
                            kwargs=dict(
                                args=args,
                                df=df,
                                nn_model=nn_model,
                                params_queue=params_queue,
                                train_networks=train_networks,
                                test_networks=pp_test_networks,
                                df_queue=df_queue,
                                iterations_queue=iterations_queue,
                                early_stopping_dict=early_stopping_dict,
                                logger=logger,
                            ),
                            # callback=_callback,
                            error_callback=partial(logger.exception, exc_info=True),
                        )

                    # Close the pool
                    p.close()
                    p.join()

                    # Close the progress bar thread
                    iterations_queue.put(None)
                    pbt.join()

    # Gracefully close the daemons
    df_queue.put(None)

    dp.join()


def parse_parameters(
        parse_args=None,
        base_dataframes_path=base_dataframes_path,
        base_models_path=base_models_path,
        logger=logging.getLogger("dummy"),
):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--parameters",
        type=str,
        nargs="*",
        default=[
            "batch_size",
            "num_epochs",
            "learning_rate",
            "weight_decay",
            "features",
        ],
        help="The features to use",
        # action="append",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=[32],
        nargs="+",
        help="Batch size for training",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        dest="num_epochs",
        type=int,
        default=[30],
        nargs="+",
        help="The number of epochs to train the model",
    )
    parser.add_argument(
        "-r",
        "--learning_rate",
        type=float,
        default=[0.005],
        nargs="+",
        help="Initial learning rate for model training",
    )
    parser.add_argument(
        "-wd",
        "--weight_decay",
        type=float,
        default=[1e-3],
        nargs="+",
        help="Weight decay",
    )
    parser.add_argument(
        "-lm",
        "--location_train",
        type=Path,
        default=None,
        required=True,
        help="Location of the dataset (directory)",
    )
    parser.add_argument(
        "-lM",
        "--models_location",
        type=Path,
        default=Path(base_models_path),
        required=False,
        help="Location of the dataset (directory)",
    )
    parser.add_argument(
        "-lt",
        "--location_test",
        type=Path,
        default=None,
        required=True,
        help="Location of the dataset (directory)",
    )
    parser.add_argument(
        "-Ft",
        "--test_filter",
        type=str,
        default="*",
        nargs="*",
        required=False,
        help="Test folder filter",
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
        default=None,
        required=False,
        help="The target threshold",
    )
    parser.add_argument(
        "-Sm",
        "--seed_train",
        type=int,
        default=set(),
        nargs="*",
        help="Pseudo Random Number Generator Seed to use during training",
    )
    parser.add_argument(
        "-St",
        "--seed_test",
        type=int,
        default=set(),
        nargs="*",
        help="Pseudo Random Number Generator Seed to use during tests",
    )
    parser.add_argument(
        "-S",
        "--seed",
        type=int,
        default={0},
        nargs="+",
        help="Pseudo Random Number Generator Seed",
    )
    parser.add_argument(
        "-f",
        "--features",
        type=str,
        default=["chi_degree", "clustering_coefficient", "degree", "kcore"],
        choices=all_features + ["None"],
        nargs="+",
        help="The features to use",
    )
    parser.add_argument(
        "-sf",
        "--static_features",
        type=str,
        default=["degree"],
        choices=all_features + ["None"],
        nargs="*",
        help="The features to use",
    )
    parser.add_argument(
        "-mf",
        "--features_min",
        type=int,
        default=1,
        help="The minimum number of features to use",
    )
    parser.add_argument(
        "-Mf",
        "--features_max",
        type=int,
        default=None,
        help="The maximum number of features to use",
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
        help="[Test only] Stops the dimantling when the max SLCC size is larger than the current LCC",
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
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="Number of jobs.",
    )
    parser.add_argument(
        "-sa",
        "--simultaneous_access",
        type=int,
        default=float("inf"),
        help="Maximum number of simultaneous predictions on CUDA device.",
    )
    parser.add_argument(
        "-FCPU",
        "--force_cpu",
        default=False,
        action="store_true",
        help="Disables ",
    )
    parser.add_argument(
        "-OE",
        "--output_extension",
        type=str,
        default=None,
        required=False,
        help="Log output file extension [for testing purposes]",
    )

    parser.add_argument(
        "-mnv",
        "--max_num_vertices",
        type=int,
        default=float("inf"),
        help="Filter the networks given the maximum number of vertices.",
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="GAT_Model",
        help="Model to use",
        choices=models_mapping.keys()
        # action="append",
    )

    parser.add_argument(
        "-DT",
        "--dont_train",
        default=False,
        action="store_true",
    )

    args, unrecognized = parser.parse_known_args(args=parse_args)

    if (not hasattr(args, "model")) or (args.model is None):
        exit("No Model selected")

    nn_model = models_mapping[args.model]

    nn_model.add_model_parameters(parser, grid=True)
    args, _ = parser.parse_known_args(args=parse_args)

    if len(args.seed_train) == 0:
        args.seed_train = args.seed.copy()
    if len(args.seed_test) == 0:
        args.seed_test = args.seed.copy()

    args.static_features = set(args.static_features)
    args.features = set(args.features) - args.static_features
    args.static_features = list(args.static_features)

    if args.features_max is None:
        args.features_max = len(args.static_features) + len(args.features)

    args.features_min -= len(args.static_features)
    args.features_max -= len(args.static_features)

    args.features = [
        sorted(args.static_features + list(c))
        for i in range(args.features_min, args.features_max + 1)
        for c in combinations(args.features, i)
    ]

    if args.removals_num is None:
        if args.static_dismantling:
            args.removals_num = 0
        else:
            args.removals_num = 1

    logger.debug(f"Cuda device count {cuda.device_count()}")
    logger.debug(f"Output folder {output_path}")

    if args.simultaneous_access is None:
        args.simultaneous_access = args.jobs

    logger.debug(f"Simultaneous access to PyTorch device {args.simultaneous_access}")

    base_dataframes_path = (
            base_dataframes_path
            / args.location_train.name
            / args.target
            / "T_{}".format(float(args.threshold) if not args.peak_dismantling else "PEAK")
    )

    if not base_dataframes_path.exists():
        base_dataframes_path.mkdir(parents=True)

    args.models_location = args.models_location.resolve()

    # logger.debug("Models location {}".format(args.models_location))

    if not args.models_location.exists():
        # args.models_location.mkdir(parents=True)
        pass
    args.output_df_columns = get_df_columns(nn_model)

    suffix = "_CORE"
    if args.lcc_only:
        suffix += "_LCC_ONLY"

    if not args.static_dismantling:
        suffix += "_DYNAMIC"

    args.output_file = base_dataframes_path / (nn_model.get_name() + suffix + ".csv")
    if args.output_extension is not None:
        args.output_file = args.output_file.with_suffix(f".{args.output_extension}.csv")

    logger.info(f"Output DF: {args.output_file}")

    return args, nn_model


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])
    logger.addHandler(TqdmLoggingHandler())
    logger.propagate = False

    multiprocessing.freeze_support()  # for Windows support

    args, nn_model = parse_parameters()

    main(args, nn_model)
