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


# This script should:
# 1. Take the input networks
# 2. Activate the right conda environment?
# 2. Compute the node features for each network
# 3. Run the grid search for the best parameters
# 4. Store the results in a CSV file
# 5. Run the reinsertion script
# 6. Extract the best runs from the CSV file and store them in the heuristics.csv file

import logging
from pathlib import Path

from graph_tool import Graph
from network_dismantling.GDM.models import BaseModel
from network_dismantling.GDM.python_interface import models_folder_path
from network_dismantling._sorters import dismantling_method
from network_dismantling.common.data_structures import dotdict, product_dict

folder = 'network_dismantling/CoreGDM/'
cd_cmd = f'cd {folder} && '
executable = 'CoreGDM'

output_folder = 'out/'
dataframes_folder = 'df/'
# models_folder = 'models/'

folder_path = Path(folder)
folder_path = folder_path.resolve()

dataframe_folder_path = folder_path / output_folder / dataframes_folder
# models_folder_path = folder_path / output_folder / models_folder

default_gdm_params = (f'--epochs 50 '
                      f'--weight_decay 1e-5 '
                      f'--learning_rate 0.003 '
                      f'--features degree clustering_coefficient kcore chi_degree '
                      f'--features_min 4 '
                      f'--features_max 4 '
                      f'--target t_0.18 '
                      f'--simultaneous_access 1 '
                      f'--static_dismantling '
                      f'--lcc_only '
                      f'--location_train GDM/dataset/synth_train_NEW/ '
                      f'--location_test NONE '
                      f'--model GAT_Model '
                      f'--models_location {models_folder_path} '
                      f'--jobs 1 '
                      f'--seed 0 '
                      f'--dont_train '
                      f'-CL 5 -CL 10 -CL 20 -CL 50 -CL 5 5 -CL 10 10 -CL 20 20 -CL 50 50 -CL 100 100 -CL 5 5 5 -CL 10 10 10 -CL 20 20 20 -CL 20 10 5 '
                      f'-CL 30 20 -CL 30 20 10 -CL 40 30 -CL 5 5 5 5 -CL 10 10 10 10 -CL 20 20 20 20 -CL 40 30 20 10 '
                      f'-FCL 100 -FCL 40 40 40 -FCL 50 30 30 -FCL 100 100 -FCL 40 30 20 -FCL 50 30 30 30 '
                      f'-H 1 -H 5 -H 10 -H 15 -H 20 -H 30 -H 1 1 -H 5 5 -H 10 10 -H 15 15 -H 20 20 -H 30 30 -H 1 1 1 -H 10 10 10 -H 20 20 20 -H 30 30 30 -H 1 1 1 1 -H 5 5 5 5 -H 10 10 10 10 -H 20 20 20 20 '
                      "--threshold {threshold} "
                      )
default_reinsertion_params = '--file NONE ' \
                             '--location_test NONE ' \
                             '--test_filter "*" ' \
                             '--sort_column r_auc rem_num ' \
                             '--reinsert_first 15 '

run_extractor_params = dotdict({
    'sort_column': ['r_auc', 'rem_num'],
    'sort_descending': False,
})

# TODO IMPROVE THIS
# Make the DF global so that it won't be loaded for every network...
df = None


def grid(df,
         args,
         nn_model: BaseModel,
         network: Graph,
         logger: logging.Logger = logging.getLogger("dummy"),
         ):
    import threading
    from collections import defaultdict
    from functools import partial
    from queue import Queue

    import numpy as np
    import pandas as pd
    from graph_tool.all import remove_parallel_edges, remove_self_loops
    from torch import cuda
    from torch import multiprocessing
    from tqdm.auto import tqdm

    import network_dismantling
    from network_dismantling.CoreGDM.core_grid import process_parameters_wrapper
    from network_dismantling.GDM.dataset_providers import init_network_provider
    from network_dismantling.GDM.dataset_providers import prepare_graph
    # from network_dismantling.GDM.models import models_mapping
    from network_dismantling.GDM.training_data_extractor import training_data_extractor
    from network_dismantling.common.multiprocessing import progressbar_thread, dataset_writer, apply_async

    # try:
    #     if cuda.is_available():
    #         multiprocessing.set_start_method('spawn', force=True)
    # except RuntimeError:
    #     pass

    network_name = network.graph_properties["filename"]

    parameters_to_try = args.parameters + nn_model.get_parameters() + ["seed_train"]

    # Get subset of args dictionary
    parameters_to_try = {k: vars(args)[k] for k in parameters_to_try}

    # Init network providers
    if not args.dont_train:
        train_networks = init_network_provider(args.location_train,
                                               max_num_vertices=None,
                                               features_list=args.features,
                                               targets=args.target,
                                               # manager=mp_manager
                                               )
        # logger.debug(f"Train networks {len(train_networks)}: {train_networks}")
    else:
        train_networks = defaultdict(list)

    # logger.debug(f"Test network list: {len(test_networks_list)} {test_networks_list}")

    # List the parameters to try
    params_list = list(product_dict(_callback=nn_model.parameters_combination_validator, **parameters_to_try))

    # Create the Multiprocessing Manager
    mp_manager = multiprocessing.Manager()

    # Init queues
    df_queue: Queue = mp_manager.Queue()
    params_queue: Queue = mp_manager.Queue()
    # device_queue: Queue = mp_manager.Queue()
    iterations_queue: Queue = mp_manager.Queue()

    early_stopping_dict = mp_manager.dict()

    network_df = df.loc[(df["network"] == network_name)]
    # network_df = df.filter()

    early_stopping_dict[network_name] = {
        "auc": network_df["r_auc"].min() or np.inf,
        "rem_num": network_df["rem_num"].min() or np.inf,
    }

    # Create and start the Dataset Writer Thread
    dp = threading.Thread(target=dataset_writer, args=(df_queue, args.output_file), daemon=True)
    dp.start()

    devices = []
    locks = dict()

    logger.debug(f"Using package {network_dismantling.__file__}")
    if cuda.is_available() and not args.force_cpu:
        logger.info("Using GPU(s).")
        for device in range(cuda.device_count()):
            device = "cuda:{}".format(device)
            devices.append(device)
            locks[device] = mp_manager.BoundedSemaphore(args.simultaneous_access)
    else:
        logger.info("Using CPU.")
        device = 'cpu'
        devices.append(device)
        locks[device] = mp_manager.BoundedSemaphore(args.simultaneous_access)

    args.devices = devices
    args.locks = locks

    new_runs_buffer = []
    for features in tqdm(args.features,
                         desc="Features",
                         ):
        key = '_'.join(features)

        def storage_provider_callback(filename, network):
            network_size = network.num_vertices()

            stop_condition = int(np.ceil(network_size * float(args.threshold)))

            logger.info(f"Dismantling {filename} according to the predictions. "
                        f"Aiming to reach LCC size {stop_condition} ({stop_condition * 100 / network_size:.3f}%)"
                        )

            # CoreHD does not support nor parallel edges nor self-loops.
            # Remove them.
            remove_parallel_edges(network)
            remove_self_loops(network)

            training_data_extractor(network,
                                    compute_targets=False,
                                    features=features,
                                    # logger=print,
                                    )

        # Prepare the network for GDM
        networks_provider = {}

        networks_provider[key] = [(network_name, network,
                                   prepare_graph(network,
                                                 features=features,
                                                 targets=None,
                                                 ),
                                   )
                                  ]

        # logger.info(f"Test network LOADED: {len(test_networks)}: {test_networks}")
        # Fill the params queue
        # TODO ANY BETTER WAY?
        for i, params in enumerate(params_list):
            # device = devices[i % len(devices)]
            # params.device = device
            # params.lock = locks[device]
            params_queue.put(params)

        # Create the pool
        with multiprocessing.Pool(processes=args.jobs, initializer=tqdm.set_lock,
                                  initargs=(multiprocessing.Lock(),)) as p:

            with tqdm(total=len(params_list),
                      leave=False,
                      desc="Parameters",
                      ) as pb:
                # Create and start the ProgressBar Thread
                pbt = threading.Thread(target=progressbar_thread,
                                       args=(iterations_queue, pb,),
                                       daemon=True,
                                       )
                pbt.start()

                for i in range(args.jobs):
                    # torch.cuda._lazy_init()

                    # p.apply_async(
                    apply_async(pool=p,
                                func=process_parameters_wrapper,
                                kwargs=dict(
                                    args=args,
                                    df=df,
                                    nn_model=nn_model,
                                    params_queue=params_queue,
                                    train_networks=train_networks,
                                    test_networks=networks_provider,
                                    df_queue=df_queue,
                                    iterations_queue=iterations_queue,
                                    early_stopping_dict=early_stopping_dict,
                                    logger=logger,
                                ),
                                callback=new_runs_buffer.extend,
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

    new_df_runs = pd.DataFrame(new_runs_buffer, columns=df.columns)

    if new_df_runs["network"].dtype != str:
        new_df_runs["network"] = new_df_runs["network"].astype(str)

    return new_df_runs


def _CoreGDM(network: Graph,
             stop_condition: int,
             threshold: float,
             parameters=default_gdm_params,
             logger: logging.Logger = logging.getLogger("dummy"),
             **kwargs):
    import pandas as pd

    from network_dismantling.CoreGDM.core_grid import parse_parameters
    from network_dismantling.GDM.extract_gdm_best import extract_best_runs as best_run_extractor
    from network_dismantling.GDM.reinsert import main as reinsert, parse_parameters as reinsert_parse_parameters
    from network_dismantling.common.df_helpers import df_reader
    from network_dismantling.common.helpers import extend_filename

    parameters = parameters.replace("{threshold}", str(threshold))

    global df

    # Run the grid script
    args, nn_model = parse_parameters(parse_args=parameters.split(),
                                      base_dataframes_path=dataframe_folder_path,
                                      logger=logger,
                                      )

    # Prepare the network for GDM
    network_name = network.graph_properties["filename"]
    args.output_file = extend_filename(args.output_file, f"_{network_name}")

    # args.threshold = stop_condition / network.num_vertices()
    args.threshold = threshold

    if df is None:
        if (args.output_file.exists()) and (args.output_file.is_file()):
            df = df_reader(args.output_file,
                           include_removals=False,
                           )
        else:
            df = pd.DataFrame(columns=args.output_df_columns)

    new_df_runs = grid(df=df,
                       args=args,
                       nn_model=nn_model,
                       network=network,
                       logger=logger,
                       )

    # Run the reinsertion script
    reinsertion_args = reinsert_parse_parameters(parse_string=default_reinsertion_params.split())
    reinsertion_args.file = args.output_file
    reinsertion_args.location_test = args.location_test
    reinsertion_args.test_filter = f"{network_name}"
    reinsertion_args.output_file = extend_filename(reinsertion_args.file, f"_reinserted")

    logger.debug("Reinsert parameters:"
                 f"\n\tFile: {reinsertion_args.file}"
                 f"\n\tLocation test: {reinsertion_args.location_test}"
                 f"\n\tTest filter: {reinsertion_args.test_filter}"
                 f"\n\tOutput file: {reinsertion_args.output_file}"
                 f"\n\tDF: {new_df_runs}")

    # TODO: Fix this...
    #  This will load the whole DF again for every new network we test... :(
    try:
        reinsert(args=reinsertion_args,
                 # df=new_df_runs,
                 test_networks={network_name: network},
                 logger=logger,
                 )
    except FileNotFoundError as e:
        raise RuntimeError(f"File {reinsertion_args.file} was not generated. Something went wrong. ")

    reinsertion_output_df_file = reinsertion_args.output_file

    reinsert_df = df_reader(reinsertion_output_df_file,
                            include_removals=True,
                            )
    run_extractor_params.file = reinsertion_output_df_file
    run_extractor_params.query = f"network == '{network_name}'"
    best_reinsertion_runs = best_run_extractor(args=run_extractor_params,
                                               df=reinsert_df,
                                               heuristic_name="CoreGDM"
                                               )

    return best_reinsertion_runs


method_info = dict(
    description=None,
    citation=None,
    authors=None,
    source="https://github.com/NetworkScienceLab/CoreGDM/",
    # plot_color=None,
    # plot_marker=None,
)


@dismantling_method(name="CoreGDM",
                    short_name="CoreGDM",

                    plot_color="#13334b",

                    includes_reinsertion=True,
                    **method_info)
def CoreGDM(network, **kwargs):
    return _CoreGDM(network, **kwargs)
