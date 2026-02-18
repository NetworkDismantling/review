"""Reverse-greedy reinsertion algorithm (subprocess-based).

This module provides the subprocess-based implementation of the
reverse-greedy reinsertion.  The C++ source (``reinsertion.cpp``) is
compiled via CMake into the ``reinsertion`` binary in this folder.

Prefer :func:`greedy_reinsertion.reverse_greedy_reinsertion` (the
auto-selecting wrapper in ``__init__.py``) over importing this module
directly — it transparently picks the graph-tool C++ extension when
available.
"""

import logging
from errno import ENOSPC
from os import close
from pathlib import Path
from subprocess import run, CalledProcessError
from tempfile import NamedTemporaryFile, mkstemp
from typing import Dict, List, Union

import numpy as np
from graph_tool import Graph

from network_dismantling.common.logging.pipe import LogPipe

logger = logging.getLogger(__name__)

# Default to the canonical reinsertion binary in this directory.
_DEFAULT_REINSERTION_DIR = Path(__file__).resolve().parent
_DEFAULT_EXECUTABLE = "reinsertion"
_DEFAULT_SORT_STRATEGY = 2

# Cache: original network filepath → temp-file path (edge-list)
_network_cache: Dict[str, str] = {}


def cleanup_network_cache() -> None:
    """Remove all cached network temp-files from disk."""
    import os

    for path in list(_network_cache.values()):
        try:
            os.remove(path)
        except OSError:
            pass

    _network_cache.clear()


def get_network_tempfile(network: Graph) -> str:
    """Write *network* as a space-separated edge-list to a temp-file.

    Uses ``static_id`` vertex property for node identifiers.
    Results are cached per ``graph_properties["filepath"]``.
    """
    network_file_path = network.graph_properties.get("filepath", None)
    if network_file_path is not None:
        cached = _network_cache.get(str(network_file_path))
        if cached is not None and Path(cached).exists():
            return cached

    fd = None
    try:
        try:
            fd, path = mkstemp(suffix=".edgelist")
        except OSError as e:
            if e.errno == ENOSPC:
                cleanup_network_cache()
                fd, path = mkstemp(suffix=".edgelist")
            else:
                raise

        static_id = network.vertex_properties["static_id"]
        with open(path, "w") as f:
            for edge in network.edges():
                f.write(f"{static_id[edge.source()]} {static_id[edge.target()]}\n")

        if network_file_path is not None:
            _network_cache[str(network_file_path)] = path

    finally:
        if fd is not None:
            try:
                close(fd)
            except OSError:
                pass

    return path


def reverse_greedy_reinsertion(
        network: Graph,
        removals: List[Union[int, Dict]],
        stop_condition: int,
        sort_strategy: int = _DEFAULT_SORT_STRATEGY,
        reinsertion_dir: Union[str, Path, None] = None,
        executable: str = _DEFAULT_EXECUTABLE,
        logger: logging.Logger = logging.getLogger("dummy"),
) -> np.ndarray:
    """Run the reverse-greedy reinsertion via the compiled C++ binary.

    Args:
        network: The full network (before any removals).
        removals: Static IDs of nodes to consider for reinsertion.
                  Can be plain ints or dicts with an ``"id"`` key.
        stop_condition: Target LCC size at which dismantling stops.
        sort_strategy: Sort strategy for the C++ binary (default 2).
        reinsertion_dir: Directory containing the compiled ``reinsertion``
            binary.  Defaults to ``GDM/reinsertion/``.
        executable: Name of the compiled binary.
        logger: Logger instance.

    Returns:
        An array of length ``network.num_vertices()`` where ``output[v]``
        is the reinsertion priority of node ``v`` (higher = remove first),
        or 0 if the node was not selected for removal.
    """
    if reinsertion_dir is None:
        reinsertion_dir = _DEFAULT_REINSERTION_DIR
    reinsertion_dir = Path(reinsertion_dir)

    network_path = get_network_tempfile(network)

    output = np.zeros(network.num_vertices(), dtype=int)

    # Normalise removals to plain ints (static_ids)
    normalised_removals: List[int] = []
    for r in removals:
        if isinstance(r, dict):
            normalised_removals.append(int(r["id"]))
        elif isinstance(r, (int, np.integer)):
            normalised_removals.append(int(r))
        else:
            # Fallback: try converting to int
            normalised_removals.append(int(r))

    with (
        NamedTemporaryFile("w+", suffix=".broken") as broken_fd,
        NamedTemporaryFile("w+", suffix=".out") as output_fd,
    ):
        broken_path = broken_fd.name
        out_path = output_fd.name

        for node_id in normalised_removals:
            broken_fd.write(f"{node_id}\n")
        broken_fd.flush()

        cd_cmd = f"cd {reinsertion_dir} && "

        # Build if necessary.  Prefer CMake (if build/ exists); fall back to make.
        build_dir = reinsertion_dir / "build"
        if build_dir.is_dir():
            build_cmd = "cd build && cmake .. && make"
        else:
            build_cmd = "mkdir -p build && cd build && cmake .. && make"

        cmds = [
            build_cmd,
            (
                f"./{executable} "
                f"--NetworkFile {network_path} "
                f'--IDFile "{broken_path}" '
                f'--OutFile "{out_path}" '
                f"--TargetSize {int(stop_condition)} "
                f"--SortStrategy {sort_strategy} "
            ),
        ]

        with (
            LogPipe(logger=logger, level=logging.INFO) as stdout_pipe,
            LogPipe(logger=logger, level=logging.ERROR) as stderr_pipe,
        ):
            for cmd in cmds:
                try:
                    logger.debug(f"Running: {cd_cmd + cmd}")
                    run(
                        cd_cmd + cmd,
                        shell=True,
                        stdout=stdout_pipe,
                        stderr=stderr_pipe,
                        text=True,
                        check=True,
                    )
                except CalledProcessError as e:
                    logger.error(f"Reinsertion binary failed: {e}", exc_info=True)
                    raise RuntimeError(f"Reinsertion failed: {e}") from e

        with open(out_path, "r") as f:
            lines = f.read().strip().splitlines()

        num_removals = len(lines)
        if num_removals == 0:
            raise RuntimeError("Reinsertion produced no output")

        nodes = []
        for i, line in enumerate(lines):
            node = int(line.strip())
            nodes.append(node)
            output[node] = num_removals - i

            if output[node] <= 0:
                raise RuntimeError(f"Node {node} has invalid priority {output[node]}")

    logger.debug(f"Reinsertion: {len(normalised_removals)} → {num_removals} removals")

    return output
