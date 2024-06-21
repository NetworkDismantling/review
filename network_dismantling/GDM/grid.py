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

import logging

from network_dismantling.GDM.config import base_models_path
from network_dismantling.common.config import base_dataframes_path


def process_parameters_wrapper(
    args,
    df,
    nn_model,
    params_queue,
    test_networks,
    train_networks,
    df_queue,
    iterations_queue,
    logger=logging.getLogger("dummy"),
):
    # import sys
    #
    # logging.basicConfig(
    #     level=logging.DEBUG,
    #     format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
    #     datefmt="%m-%d %H:%M",
    #     stream=sys.stdout,
    # )
    # logger = logging.getLogger(__name__)

    import pandas as pd
    from torch import device
    from os import getpid
    from queue import Empty
    from network_dismantling.common.multiprocessing import clean_up_the_pool
    from network_dismantling.common.multiprocessing import get_position
    from network_dismantling.GDM.network_dismantler import (
        add_run_parameters,
        train_wrapper,
        test,
        ModelWeightsNotFoundError,
    )

    child_number = get_position()

    runtime_exceptions = 0
    models_not_found = 0

    logger.debug(f"Executing on Process {getpid()}")

    all_runs = []
    while True:
        try:
            params = params_queue.get_nowait()
        except Empty:
            logger.debug(f"Empty queue. Exiting.")
            break

        if params is None:
            logger.debug(f"Processing empty (None) parameters. Exiting.")
            break

        logger.debug(f"Processing parameters: {params}")

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

        if train_networks is not None:
            train_networks_provider = train_networks.get(key, None)
        else:
            train_networks_provider = None

        # Train the model
        try:
            model = train_wrapper(
                params,
                nn_model=nn_model,
                networks_provider=train_networks_provider,
                train_ne=(not args.dont_train),
                print_model=False,
                logger=logger,
            )

            if params.device:
                try:
                    _model = model
                    model.to(params.device)
                finally:
                    del _model

        except ModelWeightsNotFoundError as e:
            runtime_exceptions += 1
            models_not_found += 1

            iterations_queue.put_nowait(1)

            continue

        except (RuntimeError, FileNotFoundError) as e:
            logger.exception(f"ERROR: {e}", exc_info=True)

            runtime_exceptions += 1
            iterations_queue.put_nowait(1)

            continue
            # raise e

        except OSError as e:
            raise e

        logger.debug(f"process_parameters_wrapper df columns: {df.columns}")

        # TODO improve me
        filter = {}
        add_run_parameters(params, filter, model)
        df_filtered = df.loc[
            (df[list(filter.keys())] == list(filter.values())).all(axis="columns"),
            ["network", "seed"],
        ]

        # # noinspection PyTypeChecker
        # for name, network, data in tqdm(test_networks[key],
        #                                 desc="Networks",
        #                                 leave=False,
        #                                 ):

        # TODO remove this loop, it is not used anymore in the review version
        for name, network, data in test_networks[key]:
            network_df = df_filtered.loc[(df_filtered["network"] == name)]

            if nn_model.is_affected_by_seed():
                tested_seeds = network_df["seed"].unique()

                seeds_to_test = set(args.seed_test) - set(tested_seeds)
                seeds_to_test = sorted(seeds_to_test)

            else:
                if len(network_df) == 0:
                    seeds_to_test = [next(iter(args.seed_test))]
                else:
                    # Nothing to do.
                    # The network was already tested (and seed doesn't matter)
                    continue

            for seed_test in seeds_to_test:
                params.seed_test = seed_test

                try:
                    # if model is None:
                    #     model = train_wrapper(params, nn_model=nn_model, networks_provider=train_networks[key], print=logger)

                    # logger.info(f"{current_process_name}: testing network {name} with seed {seed_test}")
                    # print(f"{current_process_name}: print testing network {name} with seed {seed_test}")
                    # Test
                    runs = test(
                        params,
                        model=model,
                        networks_provider=[
                            (name, network, data),
                        ],
                        print_model=False,
                        logger=logger,
                    )

                    # logger.info(f"{current_process_name}: DONE testing network {name} with seed {seed_test}")
                    # print(f"{current_process_name}: print DONE testing network {name} with seed {seed_test}")
                    all_runs += runs

                    runs_dataframe = pd.DataFrame(
                        data=runs,
                        columns=args.output_df_columns
                    )

                    if "file" in runs_dataframe.columns:
                        runs_dataframe = runs_dataframe.drop(columns=["file"])

                except RuntimeError as e:
                    logger.error(f"Runtime error: {e}", exc_info=True)

                    runtime_exceptions += 1

                    continue

                df_queue.put(runs_dataframe)

                clean_up_the_pool()

        # TODO fix OOM
        del model
        clean_up_the_pool()

        iterations_queue.put_nowait(1)

    # if runtime_exceptions > 0:
    #     logger.warning("\n\n\n")
    #
    #     logger.warning(
    #         f"WARNING: {runtime_exceptions} runs did not complete due to some runtime exception (most likely CUDA OOM). "
    #         "Try again with lower GPU load."
    #     )
    #
    #     if models_not_found > 0:
    #         logger.warning(
    #             f"\tNOTE: {models_not_found} runs did not complete because the model weights were not found. "
    #         )
    #
    #     logger.warning("\n\n\n")

    return all_runs, runtime_exceptions, models_not_found


def main(args, nn_model):
    import threading
    from functools import partial
    from queue import Queue

    import pandas as pd

    # from torch import multiprocessing, cuda
    import multiprocessing
    from torch import cuda
    from tqdm import tqdm

    import network_dismantling
    from network_dismantling.common.data_structures import product_dict
    from network_dismantling.GDM.dataset_providers import init_network_provider
    from network_dismantling.common.dataset_providers import list_files
    from network_dismantling.common.df_helpers import df_reader
    from network_dismantling.common.multiprocessing import (
        dataset_writer,
        progressbar_thread,
        apply_async,
    )

    parameters_to_try = args.parameters + nn_model.get_parameters() + ["seed_train"]

    # Get subset of args dictionary
    parameters_to_try = {k: vars(args)[k] for k in parameters_to_try}

    if args.output_file.exists():
        df_reader(args.output_file, include_removals=False)
    else:
        df = pd.DataFrame(columns=args.output_df_columns)
        del df["removals"]

    # try:
    #     multiprocessing.set_start_method('spawn')
    # except RuntimeError:
    #     pass

    # Create the Multiprocessing Manager
    # mp_manager = multiprocessing
    mp_manager = multiprocessing.Manager()

    # Init network providers
    train_networks = init_network_provider(
        args.location_train,
        max_num_vertices=None,
        features_list=args.features,
        targets=args.target,
        # manager=mp_manager,
    )

    test_networks_list = list_files(
        args.location_test,
        max_num_vertices=args.max_num_vertices,
        features_list=args.features,
        filter=args.test_filter,
        targets=None,
        # manager=mp_manager,
    )

    # logger.info(f"Test networks: {len(test_networks_list)} {test_networks_list}")

    # List the parameters to try
    params_list = list(
        product_dict(
            _callback=nn_model.parameters_combination_validator, **parameters_to_try
        )
    )

    # Init queues
    df_queue: Queue = mp_manager.Queue()
    params_queue: Queue = mp_manager.Queue()
    iterations_queue: Queue = mp_manager.Queue()

    # Create and start the Dataset Writer Thread
    dp = threading.Thread(
        target=dataset_writer, args=(df_queue, args.output_file), daemon=True
    )
    dp.start()

    # mpl = multiprocessing.log_to_stderr()
    # mpl.setLevel(logging.INFO)

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
        leave=False,
    ):
        logger.info(f"Loading network: {network_name}")

        test_networks = init_network_provider(
            args.location_test,
            max_num_vertices=args.max_num_vertices,
            features_list=args.features,
            filter=f"{network_name}",
            targets=None,
            # manager=mp_manager,
        )

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
            with tqdm(total=len(params_list),
                      ascii=True) as pb:
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
                            test_networks=test_networks,
                            train_networks=train_networks,
                            df_queue=df_queue,
                            iterations_queue=iterations_queue,
                            # device_queue=device_queue,
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
    import argparse
    from itertools import combinations
    from pathlib import Path

    from torch import cuda

    from network_dismantling.GDM.config import all_features, threshold
    from network_dismantling.GDM.models import models_mapping
    from network_dismantling.GDM.network_dismantler import get_df_columns
    from network_dismantling.common.config import output_path

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
        default=float(threshold["test"]),
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
        choices=models_mapping.keys(),
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

    if args.static_dismantling:
        args.removals_num = 0
    elif args.removals_num is None:
        args.removals_num = 1

    logger.debug(f"Cuda device count {cuda.device_count()}")
    logger.debug(f"Output folder {output_path}")

    if args.simultaneous_access is None:
        args.simultaneous_access = args.jobs

    logger.debug(f"Simultaneous access to PyTorch device {args.simultaneous_access}")

    dataframes_path = (
        base_dataframes_path
        / args.location_train.name
        / args.target
        / "T_{}".format(float(args.threshold) if not args.peak_dismantling else "PEAK")
    )

    if not dataframes_path.exists():
        dataframes_path.mkdir(parents=True)

    args.models_location = args.models_location.resolve()

    args.output_df_columns = get_df_columns(nn_model)

    suffix = ""
    if args.lcc_only:
        suffix += "_LCC_ONLY"

    if not args.static_dismantling:
        suffix += "_DYNAMIC"

    args.output_file = dataframes_path / (nn_model.get_name() + suffix + ".csv")
    if args.output_extension is not None:
        args.output_file = args.output_file.with_suffix(f".{args.output_extension}.csv")

    logger.debug(f"Output DF: {args.output_file}")

    return args, nn_model


if __name__ == "__main__":
    import argparse
    import logging
    import threading
    from functools import partial
    from itertools import combinations
    from pathlib import Path
    from queue import Queue

    import pandas as pd
    from torch import multiprocessing, cuda
    from tqdm import tqdm

    import network_dismantling
    from network_dismantling.common.data_structures import product_dict
    from network_dismantling.GDM.config import all_features, threshold, base_models_path
    from network_dismantling.GDM.dataset_providers import init_network_provider
    from network_dismantling.GDM.models import models_mapping
    from network_dismantling.GDM.network_dismantler import (
        get_df_columns,
        ModelWeightsNotFoundError,
    )
    from network_dismantling.common.config import output_path, base_dataframes_path
    from network_dismantling.common.dataset_providers import list_files
    from network_dismantling.common.multiprocessing import (
        dataset_writer,
        progressbar_thread,
        apply_async,
        TqdmLoggingHandler,
    )

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])
    logger.addHandler(TqdmLoggingHandler())
    logger.propagate = False

    multiprocessing.freeze_support()  # for Windows support

    args, nn_model = parse_parameters()

    main(args, nn_model)
