"""Greedy (reverse-greedy) reinsertion algorithm.

Given a dismantled network and its set of removed nodes, this algorithm
re-inserts nodes one at a time — choosing at each step the node whose
reinsertion causes the *smallest* increase in the largest connected
component (LCC) — until the LCC reaches a target size.

Two interchangeable back-ends are provided:

1. **Graph-tool C++ extension** (``libreinsertion_gt.so``): operates directly
   on :class:`graph_tool.Graph` objects — no temp files, no subprocess.
   Preferred when available.  Build with ``cmake --build build``.

2. **Subprocess** (``reinsertion`` binary): writes graph to a temp file,
   invokes the compiled C++ binary, reads back results.  Always available
   after building.

The public API auto-selects the best available back-end::

    from network_dismantling.greedy_reinsertion import reverse_greedy_reinsertion

    predictions = reverse_greedy_reinsertion(
        network=graph,
        removals=[42, 7, 13],   # static_ids of removed nodes
        stop_condition=100,
    )

This module is auto-discovered via ``reinsertion_interface.py`` and
registered in :data:`network_dismantling.reinsertion_methods`.

.. note::

   The subprocess binary has a known off-by-one bug in its ``sort_nodes_by_degree``
   function (``degree(i + 1, g)`` and ``W[nodes[i] - 1]``).  Since the BGL graph
   uses 0-indexed vertices, vertex 0's degree is read from vertex 1, and index -1
   is accessed.  The bug only affects the *sort order* of final outputs (strategy
   ≠ 0), not the *set* of selected nodes.  The graph-tool C++ extension does NOT
   have this bug.
"""

import logging as _logging

from network_dismantling.greedy_reinsertion.reverse_greedy import (
    reverse_greedy_reinsertion as _reverse_greedy_subprocess,
    get_network_tempfile,
    cleanup_network_cache,
)

# Try importing the graph-tool C++ extension wrapper
try:
    from network_dismantling.greedy_reinsertion.reinsertion_gt import (
        reverse_greedy_reinsertion_gt,
        is_available as _gt_is_available,
    )
except ImportError:
    reverse_greedy_reinsertion_gt = None  # type: ignore[assignment]
    _gt_is_available = lambda: False  # noqa: E731

_logger = _logging.getLogger(__name__)


def reverse_greedy_reinsertion(network, removals, stop_condition, **kwargs):
    """Run reverse-greedy reinsertion, auto-selecting the best back-end.

    If the graph-tool C++ extension is compiled and loadable, it is used
    (in-process, no temp files).  Otherwise the subprocess-based
    implementation is used as fallback.

    See :func:`reverse_greedy_reinsertion_gt` and
    :func:`reverse_greedy.reverse_greedy_reinsertion` for full parameter
    documentation.
    """
    if _gt_is_available():
        _logger.debug("Using graph-tool C++ extension for reinsertion")
        return reverse_greedy_reinsertion_gt(
            network, removals, stop_condition, **kwargs,
        )
    _logger.debug("Falling back to subprocess-based reinsertion")
    return _reverse_greedy_subprocess(
        network, removals, stop_condition, **kwargs,
    )


__all__ = [
    "reverse_greedy_reinsertion",
    "reverse_greedy_reinsertion_gt",
    "get_network_tempfile",
    "cleanup_network_cache",
]
