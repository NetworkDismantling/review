"""Graph-tool C++ extension wrapper for reverse-greedy reinsertion.

This module provides :func:`reverse_greedy_reinsertion_gt`, which calls the
compiled C++ extension ``libreinsertion_gt.so`` directly on a
:class:`graph_tool.Graph` object — no temp files, no subprocess.

The interface is compatible with :func:`reverse_greedy_reinsertion` (the
subprocess-based implementation in ``reverse_greedy.py``).

Build the extension with::

    cd network_dismantling/common/reinsertion && make libreinsertion_gt.so
"""

import logging
from typing import Dict, List, Union

import numpy as np
from graph_tool import Graph

logger = logging.getLogger(__name__)

# Sort-strategy constants (must match the C++ enum)
SORT_ORIGINAL = 0
SORT_ASCENDING_DEGREE = 1
SORT_DESCENDING_DEGREE = 2


def _load_extension():
    """Import the compiled C++ extension module.

    Returns ``None`` if the extension is not available.
    """
    try:
        # graph_tool must be imported first (provides symbols via dynamic lookup)
        import graph_tool as _gt  # noqa: F401
        from network_dismantling.common.reinsertion import libreinsertion_gt

        return libreinsertion_gt
    except ImportError:
        return None


def is_available() -> bool:
    """Return ``True`` if the C++ extension is compiled and loadable."""
    return _load_extension() is not None


def reverse_greedy_reinsertion_gt(
        network: Graph,
        removals: List[Union[int, Dict]],
        stop_condition: int,
        sort_strategy: int = SORT_DESCENDING_DEGREE,
        logger: logging.Logger = logging.getLogger("dummy"),
) -> np.ndarray:
    """Run reverse-greedy reinsertion via the graph-tool C++ extension.

    Same interface as :func:`reverse_greedy_reinsertion` but operates
    in-process without temp files or subprocess invocation.

    Args:
        network: The full network (before any removals).
        removals: Static IDs of nodes to consider for reinsertion.
            Can be plain ints or dicts with an ``"id"`` key.
        stop_condition: Target LCC size at which dismantling stops.
        sort_strategy: 0 = original, 1 = ascending degree, 2 = descending.
        logger: Logger instance.

    Returns:
        An array of length ``network.num_vertices()`` where ``output[v]``
        is the reinsertion priority of node ``v`` (higher = remove first),
        or 0 if the node was not selected for removal.  Indexing is by
        ``static_id``, matching the subprocess-based implementation.

    Raises:
        ImportError: If the C++ extension is not compiled.
        RuntimeError: If the extension produces no output.
    """
    ext = _load_extension()
    if ext is None:
        raise ImportError(
            "C++ reinsertion extension (libreinsertion_gt) not compiled.  "
            "Run 'make libreinsertion_gt.so' in common/reinsertion/."
        )

    static_id = network.vertex_properties["static_id"]

    # Build reverse map: static_id → vertex index (graph-tool descriptor)
    sid_to_vidx: Dict[int, int] = {}
    for v in network.vertices():
        sid_to_vidx[int(static_id[v])] = int(v)

    # Normalise removals to static_ids, then convert to vertex indices
    seed_vidxs: List[int] = []
    for r in removals:
        if isinstance(r, dict):
            sid = int(r["id"])
        elif isinstance(r, (int, np.integer)):
            sid = int(r)
        else:
            sid = int(r)

        vidx = sid_to_vidx.get(sid)
        if vidx is not None:
            seed_vidxs.append(vidx)
        else:
            logger.warning(
                "Reinsertion (gt): static_id %d not found in graph, skipping",
                sid,
            )

    # Call C++ extension (operates on vertex descriptors, not static_ids)
    result_vidxs = ext.reverse_greedy_reinsertion(
        network,
        seed_vidxs,
        int(stop_condition),
        sort_strategy,
    )

    # Build priority array indexed by static_id
    output = np.zeros(network.num_vertices(), dtype=int)

    num_removals = len(result_vidxs)
    if num_removals == 0:
        # No seeds remain after reinsertion (all reinserted).
        return output
    for i, vidx in enumerate(result_vidxs):
        sid = int(static_id[int(vidx)])
        output[sid] = num_removals - i

        if output[sid] <= 0:
            raise RuntimeError(
                f"Node {sid} has invalid priority {output[sid]}"
            )

    logger.debug(
        "Reinsertion (gt extension): %d → %d removals",
        len(seed_vidxs),
        num_removals,
    )

    return output
