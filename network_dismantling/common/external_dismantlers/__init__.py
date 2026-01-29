"""
External dismantlers module with C++ implementations.

This module automatically loads graph_tool before importing the C++ extension
to ensure all symbols are available (required on macOS and Linux).
"""

# Pre-load graph_tool to make its symbols available for the C++ extension
# This is necessary because the C++ dismantler uses graph_tool types
# and symbols are resolved at runtime with dynamic lookup
try:
    import graph_tool
except ImportError:
    import warnings
    warnings.warn(
        "graph_tool not available. The C++ dismantler module may fail to load.",
        ImportWarning
    )

# Now safe to import the C++ extension
try:
    # from network_dismantling.external_dismantlers import dismantler
    from . import dismantler
except ImportError as e:
    import warnings
    warnings.warn(
        f"Failed to import C++ dismantler module: {e}. "
        "Some features may not be available.",
        ImportWarning
    )
