"""Build hook for greedy reinsertion C++ targets.

Invoked during ``pip install -e .`` via the :mod:`_setup_hook` registry.
Builds both the subprocess binary (``reinsertion``) and the graph-tool
C++ extension (``libreinsertion_gt.so``) using CMake.
"""

import logging

from network_dismantling._setup_hook import setup_hook

folder = "network_dismantling/greedy_reinsertion/"


@setup_hook
def setup(*args,
          logger: logging.Logger = logging.getLogger("dummy"),
          **kwargs):
    from subprocess import check_output

    cd_cmd = f"cd {folder} && "
    cmd = "mkdir -p build && cd build && cmake .. && make"

    try:
        logger.info(check_output(cd_cmd + cmd, shell=True, text=True))
    except Exception as e:
        logger.warning(f"Failed to build greedy_reinsertion: {e}")
