"""Auto-discovery registration for the greedy reinsertion algorithm.

This module is imported automatically by the ``pkgutil.walk_packages``
auto-discovery in ``network_dismantling/__init__.py`` (suffix
``*.reinsertion_interface``).  The ``@reinsertion_method`` decorator
registers :func:`reverse_greedy_reinsertion` in the global
:data:`~network_dismantling.reinsertion_methods` dict.
"""

from network_dismantling._sorters import reinsertion_method
from network_dismantling.greedy_reinsertion import reverse_greedy_reinsertion


@reinsertion_method(
    name="Reverse Greedy Reinsertion",
    short_name="RGR",
    description=(
        "Reverse-greedy reinsertion: re-inserts removed nodes one at a time, "
        "choosing at each step the node whose reinsertion causes the smallest "
        "increase in the largest connected component."
    ),
    source="https://github.com/abraunst/decycler",
)
def greedy_reinsertion(network, removals, stop_condition, **kwargs):
    """Thin wrapper that delegates to :func:`reverse_greedy_reinsertion`."""
    return reverse_greedy_reinsertion(
        network=network,
        removals=removals,
        stop_condition=stop_condition,
        **kwargs,
    )
