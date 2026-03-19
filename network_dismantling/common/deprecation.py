"""Deprecation utilities — ``@deprecated`` decorator for marking obsolete APIs.

On Python 3.13+ ``warnings.deprecated`` is used natively.
Earlier versions get a compatible fallback with identical semantics.

Usage::

    from network_dismantling.common.deprecation import deprecated

    @deprecated("Use NewClass instead")
    def old_function():
        ...
"""

import functools

try:
    from warnings import deprecated  # noqa: F401  (Python 3.13+)
except ImportError:
    def deprecated(msg: str):  # type: ignore[misc]
        """Mark a callable as deprecated.

        Emits a :class:`DeprecationWarning` when the decorated function is called.
        Compatible shim for Python < 3.13 (replaces ``warnings.deprecated``).

        Args:
            msg: Human-readable deprecation message shown in the warning.
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                import warnings
                warnings.warn(
                    f"{func.__name__} is deprecated: {msg}",
                    DeprecationWarning,
                    stacklevel=2,
                )
                return func(*args, **kwargs)
            return wrapper
        return decorator
