"""Common reinsertion framework.

Provides a unified interface to reinsertion algorithms so that every dismantling
algorithm can share the same compiled binary and Python wrapper.

Two back-ends are available:

1. **Graph-tool C++ extension** (``libreinsertion_gt.so``): operates directly
   on :class:`graph_tool.Graph` objects — no temp files, no subprocess.
   Build with ``make libreinsertion_gt.so``.  Preferred when available.

2. **Subprocess** (``reinsertion`` binary): writes graph to a temp file,
   invokes the C++ binary, reads back results.  Always available after
   ``make reinsertion``.

Usage::

    from network_dismantling.common.reinsertion import reverse_greedy_reinsertion

    predictions = reverse_greedy_reinsertion(
        network=graph,
        removals=[42, 7, 13],   # static_ids of removed nodes
        stop_condition=100,
    )

:func:`reverse_greedy_reinsertion` automatically uses the C++ extension if
available, falling back to the subprocess implementation.
"""

import logging as _logging

from network_dismantling.common.reinsertion.reverse_greedy import (
    reverse_greedy_reinsertion as _reverse_greedy_subprocess,
    get_network_tempfile,
    cleanup_network_cache,
)

# Try importing the graph-tool C++ extension wrapper
try:
    from network_dismantling.common.reinsertion.reinsertion_gt import (
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
    (in-process, no temp files).  Otherwise, the subprocess-based
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
