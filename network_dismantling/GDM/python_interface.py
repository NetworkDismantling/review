# This script should:
# 1. Take the input networks
# 2. Activate the right conda environment?
# 2. Compute the node features for each network
# 3. Run the grid search for the best parameters
# 4. Store the results in a CSV file
# 5. Run the reinsertion script
# 6. Extract the best runs from the CSV file and store them in the heuristics.csv file
import logging
import multiprocessing
import threading
from functools import partial
from pathlib import Path

import pandas as pd
from graph_tool import Graph
from torch import cuda
from tqdm.auto import tqdm

import network_dismantling
from network_dismantling.GDM.common import product_dict, dotdict
from network_dismantling.GDM.dataset_providers import prepare_graph
from network_dismantling.GDM.extract_gdm_best import extract_best_runs as best_run_extractor
from network_dismantling.GDM.grid import process_parameters_wrapper, parse_parameters
from network_dismantling.GDM.reinsert import main as reinsert, parse_parameters as reinsert_parse_parameters
from network_dismantling._sorters import dismantling_method
from network_dismantling.common.df_helpers import df_reader
from network_dismantling.common.helpers import extend_filename
from network_dismantling.common.multiprocessing import progressbar_thread, dataset_writer, apply_async

folder = 'network_dismantling/GDM/'
cd_cmd = f'cd {folder} && '
executable = 'GDM'

output_folder = 'out/'
dataframes_folder = 'df/'
models_folder = 'models/'

folder_path = Path(folder)
folder_path = folder_path.resolve()

dataframe_folder_path = folder_path / output_folder / dataframes_folder
models_folder_path = folder_path / output_folder / models_folder

# def parse_parameters():
#     parser = argparse.ArgumentParser(
#         description="Graph node classification using GraphSAGE"
#     )
#     parser.add_argument(
#         "-p",
#         "--parameters",
#         type=str,
#         nargs="*",
#         default=["batch_size", "num_epochs", "learning_rate", "weight_decay", "features"],
#         help="The features to use",
#         # action="append",
#     )
#     parser.add_argument(
#         "-b",
#         "--batch_size",
#         type=int,
#         default=[32],
#         nargs="+",
#         help="Batch size for training",
#     )
#     parser.add_argument(
#         "-e",
#         "--epochs",
#         dest="num_epochs",
#         type=int,
#         default=[30],
#         nargs="+",
#         help="The number of epochs to train the model",
#     )
#     parser.add_argument(
#         "-r",
#         "--learning_rate",
#         type=float,
#         default=[0.005],
#         nargs="+",
#         help="Initial learning rate for model training",
#     )
#     parser.add_argument(
#         "-wd",
#         "--weight_decay",
#         type=float,
#         default=[1e-3],
#         nargs="+",
#         help="Weight decay",
#     )
#     parser.add_argument(
#         "-lm",
#         "--location_train",
#         type=Path,
#         default=None,
#         required=True,
#         help="Location of the dataset (directory)",
#     )
#     parser.add_argument(
#         "-lM",
#         "--models_location",
#         type=Path,
#         default=Path(base_models_path),
#         required=False,
#         help="Location of the dataset (directory)",
#     )
#     parser.add_argument(
#         "-lt",
#         "--location_test",
#         type=Path,
#         default=None,
#         required=True,
#         help="Location of the dataset (directory)",
#     )
#     parser.add_argument(
#         "-Ft",
#         "--test_filter",
#         type=str,
#         default="*",
#         nargs="*",
#         required=False,
#         help="Test folder filter",
#     )
#     parser.add_argument(
#         "-t",
#         "--target",
#         type=str,
#         default=None,
#         required=True,
#         help="The target node property",
#     )
#     parser.add_argument(
#         "-T",
#         "--threshold",
#         type=float,
#         default=float(threshold["test"]),
#         required=False,
#         help="The target threshold",
#     )
#     parser.add_argument(
#         "-Sm",
#         "--seed_train",
#         type=int,
#         default=set(),
#         nargs="*",
#         help="Pseudo Random Number Generator Seed to use during training",
#     )
#     parser.add_argument(
#         "-St",
#         "--seed_test",
#         type=int,
#         default=set(),
#         nargs="*",
#         help="Pseudo Random Number Generator Seed to use during tests",
#     )
#     parser.add_argument(
#         "-S",
#         "--seed",
#         type=int,
#         default={0},
#         nargs="+",
#         help="Pseudo Random Number Generator Seed",
#     )
#     parser.add_argument(
#         "-f",
#         "--features",
#         type=str,
#         default=["chi_degree", "clustering_coefficient", "degree", "kcore"],
#         choices=all_features + ["None"],
#         nargs="+",
#         help="The features to use",
#     )
#     parser.add_argument(
#         "-sf",
#         "--static_features",
#         type=str,
#         default=["degree"],
#         choices=all_features + ["None"],
#         nargs="*",
#         help="The features to use",
#     )
#     parser.add_argument(
#         "-mf",
#         "--features_min",
#         type=int,
#         default=1,
#         help="The minimum number of features to use",
#     )
#     parser.add_argument(
#         "-Mf",
#         "--features_max",
#         type=int,
#         default=None,
#         help="The maximum number of features to use",
#     )
#     parser.add_argument(
#         "-SD",
#         "--static_dismantling",
#         default=False,
#         action="store_true",
#         help="[Test only] Static removal of nodes",
#     )
#     parser.add_argument(
#         "-PD",
#         "--peak_dismantling",
#         default=False,
#         action="store_true",
#         help="[Test only] Stops the dimantling when the max SLCC size is larger than the current LCC",
#     )
#     parser.add_argument(
#         "-LO",
#         "--lcc_only",
#         default=False,
#         action="store_true",
#         help="[Test only] Remove nodes only from LCC",
#     )
#     parser.add_argument(
#         "-k",
#         "--removals_num",
#         type=int,
#         default=None,
#         required=False,
#         help="Block size before recomputing predictions",
#     )
#     parser.add_argument(
#         "-v",
#         "--verbose",
#         type=int,
#         choices=[0, 1, 2],
#         default=1,
#         help="Verbosity level",
#     )
#     parser.add_argument(
#         "-j",
#         "--jobs",
#         type=int,
#         default=1,
#         help="Number of jobs.",
#     )
#     parser.add_argument(
#         "-sa",
#         "--simultaneous_access",
#         type=int,
#         default=float('inf'),
#         help="Maximum number of simultaneous predictions on CUDA device.",
#     )
#     parser.add_argument(
#         "-FCPU",
#         "--force_cpu",
#         default=False,
#         action="store_true",
#         help="Disables ",
#     )
#     parser.add_argument(
#         "-OE",
#         "--output_extension",
#         type=str,
#         default=None,
#         required=False,
#         help="Log output file extension [for testing purposes]",
#     )
#
#     parser.add_argument(
#         "-mnv",
#         "--max_num_vertices",
#         type=int,
#         default=float("inf"),
#         help="Filter the networks given the maximum number of vertices.",
#     )
#
#     parser.add_argument(
#         "-m",
#         "--model",
#         type=str,
#         default="GATModel",
#         help="Model to use",
#         choices=models_mapping.keys()
#         # action="append",
#     )
#
#     args, _ = parser.parse_known_args()
#
#     if (not hasattr(args, "model")) or (args.model is None):
#         exit("No Model selected")
#
#     nn_model = models_mapping[args.model]
#
#     nn_model.add_model_parameters(parser, grid=True)
#     args, _ = parser.parse_known_args(namespace=args)
#
#     if len(args.seed_train) == 0:
#         args.seed_train = args.seed.copy()
#     if len(args.seed_test) == 0:
#         args.seed_test = args.seed.copy()
#
#     args.static_features = set(args.static_features)
#     args.features = set(args.features) - args.static_features
#     args.static_features = list(args.static_features)
#
#     if args.features_max is None:
#         args.features_max = len(args.static_features) + len(args.features)
#
#     args.features_min -= len(args.static_features)
#     args.features_max -= len(args.static_features)
#
#     args.features = [sorted(args.static_features + list(c)) for i in range(args.features_min, args.features_max + 1)
#                      for c in combinations(args.features, i)]
#
#     if args.removals_num is None:
#         if args.static_dismantling:
#             args.removals_num = 0
#         else:
#             args.removals_num = 1
#
#     return args, nn_model

# TODO improve parameter handling...
# default_gdm_params = {
#     'features': ['chi_degree', 'degree', 'clustering_coefficient', 'kcore', ],
#     'static_features': ['degree', ],
#     'features_min': 4,
#     'features_max': 4,
#     'static_dismantling': True,
#     'peak_dismantling': False,
#     'lcc_only': True,
#     'removals_num': 0,
#     'verbose': 0,
#     'jobs': None,
#     'simultaneous_access': 1,
#     'force_cpu': False,
#     'output_extension': None,
#     # 'max_num_vertices': float('inf'),
#     'model': 'GAT_Model',
#     'seed_train': [0, ],
#     'seed_test': [0, ],
#     'seed': [0, ],
#     'epochs': 50,
#     'batch_size': 32,
#     'lr': 0.001,
#     'weight_decay': 0.0005,
#     'location_train': "",
#     'location_test': "",
#     'models_location': "",
#     "dont_train": True,
# }


default_gdm_params = '--epochs 50 ' \
                     '--weight_decay 1e-5 ' \
                     '--learning_rate 0.003 ' \
                     '--features degree clustering_coefficient kcore chi_degree ' \
                     '--features_min 4 ' \
                     '--features_max 4 ' \
                     '--target t_0.18 ' \
                     '--threshold 0.10 ' \
                     '--simultaneous_access 1 ' \
                     '--static_dismantling ' \
                     '--lcc_only ' \
                     '--location_train GDM/dataset/synth_train_NEW/ ' \
                     '--location_test NONE ' \
                     '--model GAT_Model ' \
                     f'--models_location {models_folder_path} ' \
                     '--jobs 1 ' \
                     '--seed 0 ' \
                     '--dont_train ' \
                     '-CL 5 -CL 10 -CL 20 -CL 50 -CL 5 5 -CL 10 10 -CL 20 20 -CL 50 50 -CL 100 100 -CL 5 5 5 -CL 10 10 10 -CL 20 20 20 -CL 20 10 5 ' \
                     '-CL 30 20 -CL 30 20 10 -CL 40 30 -CL 5 5 5 5 -CL 10 10 10 10 -CL 20 20 20 20 -CL 40 30 20 10 ' \
                     '-FCL 100 -FCL 40 40 40 -FCL 50 30 30 -FCL 100 100 -FCL 40 30 20 -FCL 50 30 30 30 ' \
                     '-H 1 -H 5 -H 10 -H 15 -H 20 -H 30 -H 1 1 -H 5 5 -H 10 10 -H 15 15 -H 20 20 -H 30 30 -H 1 1 1 -H 10 10 10 -H 20 20 20 -H 30 30 30 -H 1 1 1 1 -H 5 5 5 5 -H 10 10 10 10 -H 20 20 20 20 '

# parser.add_argument(
#         "-f",
#         "--file",
#         type=Path,
#         default=None,
#         required=True,
#         help="Output DataFrame file location",
#     )
#     parser.add_argument(
#         "-lt",
#         "--location_test",
#         type=Path,
#         default=None,
#         required=True,
#         help="Location of the dataset (directory)",
#     )
#     parser.add_argument(
#         "-Ft",
#         "--test_filter",
#         type=str,
#         default="*",
#         required=False,
#         help="Test folder filter",
#     )
#     parser.add_argument(
#         "-q",
#         "--query",
#         type=str,
#         default=None,
#         required=False,
#         help="Query the dataframe",
#     )
#     parser.add_argument(
#         "-rf",
#         "--reinsert_first",
#         type=int,
#         default=15,
#         required=False,
#         help="Show first N dismantligs",
#     )
#     parser.add_argument(
#         "-s",
#         "--sort_column",
#         type=str,
#         default="r_auc",
#         required=False,
#         help="Column used to sort the entries",
#     )
#     parser.add_argument(
#         "-sa",
#         "--sort_descending",
#         default=False,
#         required=False,
#         action="store_true",
#         help="Descending sorting",
#     )

default_reinsertion_params = '--file NONE ' \
                             '--location_test NONE ' \
                             '--test_filter "*" ' \
                             '--sort_column r_auc rem_num ' \
                             '--reinsert_first 15 '
# '--sort_descending ' \

# Make the DF global so that it won't be for every network...
df = None


def grid(df,
         args,
         nn_model,
         test_networks_provider,
         logger=logging.getLogger("dummy")
         ):
    logger.info(f"Using package {network_dismantling.__file__}")

    parameters_to_try = args.parameters + nn_model.get_parameters() + ["seed_train"]

    # Get subset of args dictionary
    parameters_to_try = {k: vars(args)[k] for k in parameters_to_try}

    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    # Create the Multiprocessing Manager
    mp_manager = multiprocessing.Manager()
    # mp_manager = multiprocessing

    # List the parameters to try
    params_list = list(product_dict(_callback=nn_model.parameters_combination_validator, **parameters_to_try))

    # Init queues
    df_queue = mp_manager.Queue()
    params_queue = mp_manager.Queue()
    # device_queue = mp_manager.Queue()
    iterations_queue = mp_manager.Queue()

    # Create and start the Dataset Writer Thread
    dp = threading.Thread(target=dataset_writer, args=(df_queue, args.output_file), daemon=True)
    dp.start()

    devices = []
    locks = dict()
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

    # # Put all available devices in the queue
    # for device in devices:
    #     for _ in range(args.simultaneous_access):
    #         # Allow simultaneous access to the same device
    #         device_queue.put(device)

    args.devices = devices
    args.locks = locks

    # Fill the params queue
    # TODO ANY BETTER WAY?
    for i, params in enumerate(params_list):
        # device = devices[i % len(devices)]
        # params.device = device
        # params.lock = locks[device]
        params_queue.put(params)

    new_runs_buffer = []
    # Create the pool
    with multiprocessing.Pool(processes=args.jobs, initializer=tqdm.set_lock,
                              initargs=(multiprocessing.Lock(),)) as p:

        with tqdm(total=len(params_list), ascii=True) as pb:
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
                            kwargs=dict(args=args,
                                        df=df,
                                        nn_model=nn_model,
                                        params_queue=params_queue,
                                        train_networks=None,
                                        test_networks=test_networks_provider,
                                        df_queue=df_queue,
                                        iterations_queue=iterations_queue,
                                        # device_queue=device_queue,
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
    # new_runs = pd.concat(new_runs_buffer, ignore_index=True)

    return new_df_runs


def _GDM(network: Graph, stop_condition: int, reinsertion: bool, parameters=default_gdm_params,
         logger=logging.getLogger("dummy"), **kwargs):
    global df

    # Run the grid script
    args, nn_model = parse_parameters(parse_args=parameters.split(),
                                      base_dataframes_path=dataframe_folder_path,
                                      )

    # Prepare the network for GDM
    network_name = network.graph_properties["filename"]
    args.output_file = extend_filename(args.output_file, f"_{network_name}")
    args.threshold = stop_condition / network.num_vertices()

    # Prepare the network for GDM
    networks_provider = {}
    for features in args.features:
        key = '_'.join(features)

        # TODO REMOVE THIS LIST
        networks_provider[key] = [(network_name, network,
                                   prepare_graph(network,
                                                 features=features,
                                                 targets=None,
                                                 ),
                                   )
                                  ]

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
                       test_networks_provider=networks_provider,
                       logger=logger,
                       )

    # TODO: Fix this...
    #  This will load the whole DF again for every new network we test... :(
    run_extractor_params = dotdict({
        'query': f"network == '{network_name}'",
        'sort_column': ['r_auc', 'rem_num'],
        'sort_descending': False,
    })
    if reinsertion is True:
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
        reinsert(args=reinsertion_args,
                 # df=new_df_runs,
                 test_networks={network_name: network},
                 logger=logger,
                 )
        reinsertion_output_df_file = reinsertion_args.output_file

        run_extractor_params.file = reinsertion_output_df_file

        reinsert_df = df_reader(reinsertion_output_df_file,
                                include_removals=True,
                                )
        best_reinsertion_runs = best_run_extractor(args=run_extractor_params,
                                                   df=reinsert_df,
                                                   heuristic_name="GDMR"
                                                   )

        return best_reinsertion_runs

    else:
        # Get the best solution for the current network
        output_df_file = args.output_file

        df = df_reader(output_df_file,
                       include_removals=True,
                       )

        run_extractor_params.file = output_df_file
        best_runs = best_run_extractor(args=run_extractor_params,
                                       df=df,
                                       heuristic_name="GDM"
                                       )
        return best_runs

    # return pd.concat([best_runs, best_reinsertion_runs], ignore_index=True)
    # return best_runs


method_info = dict(
    description=None,
    citation=None,
    authors=None,
    source="https://github.com/NetworkScienceLab/GDM/",
    # plot_color=None,
    # plot_marker=None,
)


@dismantling_method(name="Graph Dismantling Machine",
                    short_name="GDM",

                    plot_color="#3080bd",

                    includes_reinsertion=False,
                    **method_info)
def GDM(network, **kwargs):
    return _GDM(network, reinsertion=False, **kwargs)


@dismantling_method(name="Graph Dismantling Machine + Reinsertion",
                    short_name="GDM +R",
                    plot_color="#084488",

                    includes_reinsertion=True,
                    **method_info)
def GDMR(network, **kwargs):
    return _GDM(network, reinsertion=True, **kwargs)
