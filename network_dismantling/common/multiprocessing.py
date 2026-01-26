import multiprocessing
from concurrent import futures
from typing import Callable

from parse import compile
from multiprocessing import current_process

child_num_format = compile("{}-{number:d}")


def progressbar_thread(q, progressbar):
    while True:
        record = q.get()

        if record is None:
            return

        progressbar.update()


def tqdm_logger_thread(q, logger=None):
    from tqdm import tqdm

    if logger is None:
        logger = tqdm.write

    while True:
        record = q.get()

        if record is None:
            return
        logger(record)


def run_dill_encoded(payload):
    """
    https://stackoverflow.com/questions/8804830/python-multiprocessing-picklingerror-cant-pickle-type-function
    """
    from dill import loads

    fun, args, kwargs = loads(payload, ignore=False)

    return fun(*args, **kwargs)


def submit(executor: futures.Executor,
           func: Callable,
           *args, **kwargs) -> futures.Future:
    from dill import dumps, HIGHEST_PROTOCOL

    payload = dumps(
        (func, args, kwargs),
        byref=False,
        protocol=HIGHEST_PROTOCOL,
        recurse=True,
    )
    return executor.submit(run_dill_encoded, payload)


def apply_async(pool, func, args=None, kwargs=None, callback=None, error_callback=None):
    import dill

    if args is None:
        args = ()

    if kwargs is None:
        kwargs = {}

    payload = dill.dumps((func, args, kwargs))
    return pool.apply_async(
        run_dill_encoded, (payload,), callback=callback, error_callback=error_callback
    )


def map(pool: multiprocessing.Pool, func, args=None, kwargs=None, callback=None, error_callback=None):
    import dill

    if args is None:
        args = ()

    if kwargs is None:
        kwargs = {}

    payload = dill.dumps((func, args, kwargs))
    return pool.map(
        run_dill_encoded, (payload,)
    )


def clean_up_the_pool(*args, **kwargs):
    """https://discuss.pytorch.org/t/pytorch-multiprocessing-cuda-out-of-memory/53417"""
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_position():
    try:
        position = child_num_format.parse(current_process().name)["number"]
    except:
        position = 2

    return position
