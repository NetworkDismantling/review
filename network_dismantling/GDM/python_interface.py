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
from network_dismantling._sorters import dismantling_method
from torch import multiprocessing

try:
    from deadpool import as_completed, Executor

except ImportError:
    from concurrent.futures import as_completed, Executor

folder = "network_dismantling/GDM/"
cd_cmd = f"cd {folder} && "
executable = "GDM"

# output_folder = 'out/'
output_folder = "network_dismantling/out/GDM/"
dataframes_folder = "df/"
# models_folder = 'models/'
models_folder = "models_newpg/"

folder_path = Path(folder)
folder_path = folder_path.resolve()

output_folder_path = Path(output_folder)
output_folder_path = output_folder_path.resolve()

dataframe_folder_path = output_folder_path / dataframes_folder
models_folder_path = folder_path / "out" / models_folder

default_gdm_params = (
    "--epochs 50 "
    "--weight_decay 1e-5 "
    "--learning_rate 0.003 "
    "--features degree clustering_coefficient kcore chi_degree "
    "--features_min 4 "
    "--features_max 4 "
    "--target t_0.18 "
    # "--threshold 0.10 "
    "--threshold {threshold} "
    "--simultaneous_access 1 "
    "--static_dismantling "
    "--lcc_only "
    "--location_train GDM/dataset/synth_train_NEW/ "
    "--location_test NONE "
    "--model GAT_Model "
    f"--models_location {models_folder_path} "
    "--seed 0 "
    "--dont_train "
    "-CL 5 -CL 10 -CL 20 -CL 50 -CL 5 5 -CL 10 10 -CL 20 20 -CL 50 50 -CL 100 100 -CL 5 5 5 -CL 10 10 10 -CL 20 20 20 -CL 20 10 5 "
    "-CL 30 20 -CL 30 20 10 -CL 40 30 -CL 5 5 5 5 -CL 10 10 10 10 -CL 20 20 20 20 -CL 40 30 20 10 "
    "-FCL 100 -FCL 40 40 40 -FCL 50 30 30 -FCL 100 100 -FCL 40 30 20 -FCL 50 30 30 30 "
    "-H 1 -H 5 -H 10 -H 15 -H 20 -H 30 -H 1 1 -H 5 5 -H 10 10 -H 15 15 -H 20 20 -H 30 30 -H 1 1 1 -H 10 10 10 -H 20 20 20 -H 30 30 30 -H 1 1 1 1 -H 5 5 5 5 -H 10 10 10 10 -H 20 20 20 20 "
)
#                      '--jobs 1 ' \
#                      '--simultaneous_access 1 ' \

default_reinsertion_params = (
    "--file NONE "
    "--location_test NONE "
    '--test_filter "*" '
    "--sort_column r_auc rem_num "
    "--reinsert_first 15 "
)

# TODO IMPROVE THIS
# Make the DF global so that it won't be loaded for every network...
df = None


def grid(
        df,
        args,
        nn_model,
        test_networks_provider,
        executor: Executor,
        pool_size: int,
        mp_manager: multiprocessing.Manager,
        logger: logging.Logger = logging.getLogger("dummy"),
        **kwargs,
):
    import threading
    from queue import Queue

    import pandas as pd
    from torch import cuda
    from tqdm.auto import tqdm

    # import network_dismantling
    from network_dismantling.common.data_structures import product_dict
    from network_dismantling.GDM.grid import process_parameters_wrapper
    from network_dismantling.common.multiprocessing import (
        progressbar_thread,
        dataset_writer,
        submit,
    )

    # logger.debug(f"Using package {network_dismantling.__file__}")

    parameters_to_try = args.parameters + nn_model.get_parameters() + ["seed_train"]

    # Get subset of args dictionary
    parameters_to_try = {k: vars(args)[k] for k in parameters_to_try}

    # List the parameters to try
    params_list = list(
        product_dict(
            _callback=nn_model.parameters_combination_validator,
            **parameters_to_try
        )
    )

    # Init queues
    df_queue: Queue = mp_manager.Queue()
    params_queue: Queue = mp_manager.Queue()
    # device_queue: Queue = mp_manager.Queue()
    iterations_queue: Queue = mp_manager.Queue()

    # Create and start the Dataset Writer Thread
    dp = threading.Thread(
        target=dataset_writer,
        kwargs=dict(
            queue=df_queue,
            output_file=args.output_file,
        ),
        daemon=True,
    )
    dp.start()

    devices = []
    locks = dict()
    if cuda.is_available() and not args.force_cpu:
        logger.debug("Using GPU(s).")
        for device in range(cuda.device_count()):
            device = "cuda:{}".format(device)
            devices.append(device)
            locks[device] = mp_manager.BoundedSemaphore(args.simultaneous_access)
    else:
        logger.debug("Using CPU.")
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

    # Fill the params queue
    for i, params in enumerate(params_list):
        # device = devices[i % len(devices)]
        # params.device = device
        # params.lock = locks[device]
        params_queue.put(params)

    new_runs_buffer = []

    # def callback(future):
    #     try:
    #         new_runs_buffer.extend(future.result())
    #     except Exception as e:
    #         logger.exception(e, exc_info=True)

    runtime_exceptions = 0
    models_not_found = 0

    logger.debug(f"df columns: {df.columns}")
    with tqdm(
            total=len(params_list),
            # ascii=True,
            ascii=False,
            desc="Parameters",
            leave=False,
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

        # TODO improve this
        futures = []
        for i in range(pool_size):
            futures.append(
                # executor.submit(process_parameters_wrapper,
                submit(
                    executor=executor,
                    func=process_parameters_wrapper,
                    args=args,
                    df=df,
                    nn_model=nn_model,
                    params_queue=params_queue,
                    train_networks=None,
                    test_networks=test_networks_provider,
                    df_queue=df_queue,
                    iterations_queue=iterations_queue,
                    logger=logger,
                )
                # .add_done_callback(callback)
            )

        for future in as_completed(futures):
            try:
                # new_runs_buffer.extend(future.result())
                worker_new_runs, worker_runtime_exceptions, worker_models_not_found = future.result()

                new_runs_buffer.extend(worker_new_runs)
                runtime_exceptions += worker_runtime_exceptions
                models_not_found += worker_models_not_found

            except Exception as e:
                logger.exception(e, exc_info=True)

        # Close the progress bar thread
        iterations_queue.put(None)
        pbt.join()

    # Gracefully close the daemons
    df_queue.put(None)

    dp.join()

    new_df_runs = pd.DataFrame(new_runs_buffer, columns=df.columns)

    if new_df_runs["network"].dtype != str:
        new_df_runs["network"] = new_df_runs["network"].astype(str)

    if runtime_exceptions > 0:
        logger.warning("\n\n\n")

        logger.warning(
            f"WARNING: {runtime_exceptions} runs did not complete due to some runtime exception (most likely CUDA OOM). "
            "Try again with lower GPU load."
        )

        if models_not_found > 0:
            logger.warning(
                f"\tNOTE: {models_not_found} runs did not complete because the model weights were not found. "
            )

        logger.warning("\n\n\n")

    return new_df_runs


def _GDM(
        network: Graph,
        stop_condition: int,
        reinsertion: bool,
        threshold: float,
        parameters=default_gdm_params,
        logger=logging.getLogger("dummy"),
        **kwargs,
):
    import pandas as pd

    from network_dismantling.common.data_structures import dotdict
    from network_dismantling.GDM.dataset_providers import prepare_graph
    from network_dismantling.GDM.extract_gdm_best import (
        extract_best_runs as best_run_extractor,
    )
    from network_dismantling.GDM.grid import parse_parameters
    from network_dismantling.GDM.reinsert import (
        main as reinsert,
        parse_parameters as reinsert_parse_parameters,
    )
    from network_dismantling.common.df_helpers import df_reader
    from network_dismantling.common.helpers import extend_filename

    global df
    parameters = parameters.replace("{threshold}", str(threshold))

    # Run the grid script
    args, nn_model = parse_parameters(
        parse_args=parameters.split(),
        base_dataframes_path=dataframe_folder_path,
    )

    # Prepare the network for GDM
    network_name = network.graph_properties["filename"]
    args.output_file = extend_filename(args.output_file, f"_{network_name}")

    # args.threshold = stop_condition / network.num_vertices()
    args.threshold = threshold

    # Prepare the network for GDM
    networks_provider = {}
    for features in args.features:
        key = "_".join(features)

        # TODO REMOVE THIS LIST
        networks_provider[key] = [
            (
                network_name,
                network,
                prepare_graph(
                    network,
                    features=features,
                    targets=None,
                ),
            )
        ]

    if df is None:
        if (args.output_file.exists()) and (args.output_file.is_file()):
            df = df_reader(
                args.output_file,
                include_removals=False,
                raise_on_missing_file=True,
                # expected_columns=args.output_df_columns,
            )
        else:
            df = pd.DataFrame(columns=args.output_df_columns)

    new_df_runs = grid(
        df=df,
        args=args,
        nn_model=nn_model,
        test_networks_provider=networks_provider,
        logger=logger,
        **kwargs,
    )

    # TODO: Fix this...
    #  This will load the whole DF again for every new network we test... :(
    run_extractor_params = dotdict(
        {
            "query": f"network == '{network_name}'",
            "sort_column": ["r_auc", "rem_num"],
            "sort_descending": False,
        }
    )
    if reinsertion is True:
        reinsertion_args = reinsert_parse_parameters(
            parse_string=default_reinsertion_params.split()
        )
        reinsertion_args.file = args.output_file
        # reinsertion_args.location_test = args.location_test
        # reinsertion_args.test_filter = f"{network_name}"
        reinsertion_args.output_file = extend_filename(
            file=reinsertion_args.file,
            filename_extension=f"_reinserted"
        )

        logger.debug(
            "Reinsert parameters:"
            f"\n\tFile: {reinsertion_args.file}"
            # f"\n\tLocation test: {reinsertion_args.location_test}"
            # f"\n\tTest filter: {reinsertion_args.test_filter}"
            f"\n\tOutput file: {reinsertion_args.output_file}"
            f"\n\tDF: {new_df_runs}"
        )

        try:
            reinsert(
                args=reinsertion_args,
                # df=new_df_runs,
                test_networks={f"{network_name}": network},
                logger=logger,
                threshold=args.threshold,
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                f"File {reinsertion_args.file} was not generated. Something went wrong. "
            )

        reinsertion_output_df_file = reinsertion_args.output_file

        run_extractor_params.file = reinsertion_output_df_file

        reinsert_df = df_reader(
            reinsertion_output_df_file,
            include_removals=True,
        )
        best_reinsertion_runs = best_run_extractor(
            args=run_extractor_params,
            df=reinsert_df,
            heuristic_name="GDMR",
        )

        return best_reinsertion_runs

    else:
        # Get the best solution for the current network
        output_df_file = args.output_file

        df = df_reader(
            output_df_file,
            include_removals=True,
        )

        run_extractor_params.file = output_df_file
        best_runs = best_run_extractor(
            args=run_extractor_params,
            df=df,
            heuristic_name="GDM",
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


@dismantling_method(
    name="Graph Dismantling Machine",
    short_name="GDM",
    plot_color="#3080bd",
    includes_reinsertion=False,
    **method_info,
)
def GDM(network, **kwargs):
    return _GDM(network, reinsertion=False, **kwargs)


@dismantling_method(
    name="Graph Dismantling Machine + Reinsertion",
    short_name="GDM +R",
    plot_color="#084488",
    includes_reinsertion=True,
    **method_info,
)
def GDMR(network, **kwargs):
    return _GDM(network, reinsertion=True, **kwargs)
