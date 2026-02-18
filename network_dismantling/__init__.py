#   This file is part of the Network Dismantling review,
#   proposed in the paper "Robustness and resilience of complex networks"
#   by Oriol Artime, Marco Grassia, Manlio De Domenico, James P. Gleeson,
#   Hernán A. Makse, Giuseppe Mangioni, Matjaž Perc and Filippo Radicchi.
#
#   This is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   The project is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with the code.  If not, see <http://www.gnu.org/licenses/>.
import importlib
import logging
import pkgutil
from pathlib import Path
from typing import Callable, Union, List

import pandas as pd

from network_dismantling.common.data_structures import product_dict
from network_dismantling.common.dismantlers import dismantler_wrapper

dismantling_methods = {}
reinsertion_methods = {}

logger = logging.getLogger(__name__)


def setdefaultattr(obj, name, value):
    try:
        return getattr(obj, name)
    except AttributeError:
        setattr(obj, name, value)
    return value


class ReinsertionMethod:
    """A registered reinsertion algorithm.

    Reinsertion methods take a set of removed nodes and a graph, then
    return an optimised (smaller) removal set.  They are stored in
    :data:`reinsertion_methods` and auto-discovered from modules named
    ``*.reinsertion_interface``.

    Instances are callable — invoking one is equivalent to calling
    ``self.function(network, removals, stop_condition, **kwargs)``.
    """

    key: str | None = None

    name: str | None = None
    short_name: str | None = None

    description: str | None = None
    citation: str | None = None
    authors: Union[str, List[str]] | None = None
    source: str | None = None

    function: Callable | None = None

    license_file: Path | None = None
    citation_file: Path | None = None

    def __init__(self, **kwargs):
        super().__init__()

        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.function is None:
            raise RuntimeError("ReinsertionMethod must have a function defined")

        self.key = self.function.__name__

        if self.name is None:
            self.name = self.key
        if self.short_name is None:
            raise RuntimeError(f"Short name not defined for reinsertion method {self.key}")

    def __call__(self, network, removals, stop_condition, **kwargs):
        return self.function(
            network=network,
            removals=removals,
            stop_condition=stop_condition,
            **kwargs,
        )

    def __repr__(self) -> str:
        return f"ReinsertionMethod(key={self.key!r}, name={self.name!r})"


class DismantlingMethod:
    key: str | None = None

    name: str | None = None
    short_name: str | None = None

    _depends_on: str | None = None

    # Method categorization
    method_type: str | None = None  # e.g. "heuristic", "ml_based", "spectral", "optimization"

    doi: str | None = None
    citation: str | None = None
    description: str | None = None
    authors: Union[str, List[str]] | None = None

    function: Callable | None = None
    dynamic: bool | None = None

    display_name: str | None = None
    short_display_name: str | None = None

    plot_color: str | None = None
    plot_marker: str | None = None

    # reinsertion: ReinsertionSupport = None
    includes_reinsertion: bool = False
    optional_reinsertion: bool = False
    reinsertion_function = None
    reinsertion_display_name = None
    reinsertion_short_display_name = None

    license_file: Path | None = None

    # return_type: ReturnTypes = None

    source: str | None = None

    required_imports: List[str] | None = None

    def __init__(self,
                 # name=None,
                 # includes_reinsertion=False,
                 # description=None,
                 # citation=None,
                 # authors=None,
                 # source=None,
                 # return_type=None,
                 # depends_on: Union[str, Callable] = None,

                 **kwargs
                 ):

        super().__init__()

        for key, value in kwargs.items():
            # setdefaultattr(self, key, value)
            setattr(self, key, value)

        if self.function is None:
            raise RuntimeError("DismantlingMethod must have a function defined")

        self.key = self.function.__name__

        if self.name is None:
            self.name = self.key
        if self.display_name is None:
            self.display_name = "".join([w[0].capitalize() for w in self.name.split("_")])
        # if self.dynamic is None:
        #     raise RuntimeError(f"Dynamic/static not defined for {self.key}")

        if self.required_imports is None:
            self.required_imports = []

        if self.short_name is None:
            raise RuntimeError(f"Short name not defined for {self.key}")
        if not isinstance(self.short_name, str):
            raise RuntimeError(f"Short name not a string for {self.key}")

    def __call__(self, *args, **kwargs):
        output = self.function(*args, **kwargs)

        output["static"] = not self.dynamic
        output["heuristic"] = self.key

        # Auto-chain reinsertion when a reinsertion_function is configured and
        # the caller hasn't explicitly disabled it (reinsertion=False).
        if (
            self.reinsertion_function is not None
            and kwargs.get("reinsertion", True)
        ):
            reinsertion_logger = kwargs.get("logger", logging.getLogger("dummy"))

            # Extract the network (first positional arg or kwarg)
            network = args[0] if args else kwargs.get("network")
            if network is None:
                reinsertion_logger.warning(
                    f"Cannot run reinsertion for {self.key}: no network provided"
                )
            else:
                from network_dismantling.common.removal import Removal

                removals = output.get("removals", [])
                # Extract node IDs from removals
                removal_ids = []
                for r in removals:
                    if isinstance(r, Removal):
                        removal_ids.append(r.node_id)
                    elif isinstance(r, dict):
                        removal_ids.append(r["id"])
                    else:
                        removal_ids.append(int(r))

                stop_condition = kwargs.get("stop_condition", 1)

                reinsertion_logger.info(
                    f"Running reinsertion for {self.key} "
                    f"({len(removal_ids)} removals, target={stop_condition})"
                )

                reinsertion_output = self.reinsertion_function(
                    network=network,
                    removals=removal_ids,
                    stop_condition=stop_condition,
                    logger=reinsertion_logger,
                )
                output["reinsertion_predictions"] = reinsertion_output

        return output

    def _format_output(self, output):

        if isinstance(output, dict):
            import pandas as pd

            output = pd.DataFrame(output)

        return output

    def _format_input(self, input: pd.DataFrame):
        return input

    def _filter_input(self, input: pd.DataFrame):
        return input

    def handle_parameters(self, **kwargs):
        return product_dict(kwargs)

    @property
    def depends_on(self):
        if self._depends_on is None:
            return None

        return dismantling_methods[self._depends_on]

    @depends_on.setter
    def depends_on(self, value):
        if value is None:
            self._depends_on = None
        else:

            if isinstance(value, str):
                self._depends_on = value
            elif isinstance(value, DismantlingMethod):
                self._depends_on = value.key
            else:
                self._depends_on = value.__name__

    def with_reinsertion(
        self,
        reinsertion_function=None,
        name: str | None = None,
        short_name: str | None = None,
        **kwargs,
    ) -> "DismantlingMethod":
        """Create a ``+Reinsertion`` variant of this method.

        Returns a *new* :class:`DismantlingMethod` that:

        1. Runs the same underlying dismantling function.
        2. Auto-chains *reinsertion_function* after dismantling.
        3. Is registered in :data:`dismantling_methods` under a new key.

        This eliminates the need to write a separate Python function for
        every ``Algorithm + R`` variant.  Instead::

            from network_dismantling.greedy_reinsertion.reverse_greedy import (
                reverse_greedy_reinsertion,
            )

            GNDR = GND.with_reinsertion(reverse_greedy_reinsertion)

        Args:
            reinsertion_function: Callable that runs reinsertion (default:
                :func:`reverse_greedy_reinsertion` from ``common.reinsertion``).
            name: Display name (default: ``"<original> + Reinsertion"``).
            short_name: Short display name (default: ``"<original>+R"``).
            **kwargs: Extra attributes forwarded to the new
                :class:`DismantlingMethod`.

        Returns:
            The newly registered :class:`DismantlingMethod`.
        """
        if reinsertion_function is None:
            from network_dismantling.greedy_reinsertion import (
                reverse_greedy_reinsertion,
            )
            reinsertion_function = reverse_greedy_reinsertion

        if name is None:
            name = f"{self.name} + Reinsertion"
        if short_name is None:
            short_name = f"{self.short_name}+R"

        # Build the wrapper function that delegates to the original.
        # We need a real function object with a unique __name__ for the
        # registry key.
        func_name = f"{self.function.__name__}_R"

        def _reinsertion_wrapper(*args, **kw):
            return self.function(*args, **kw)

        _reinsertion_wrapper.__name__ = func_name
        _reinsertion_wrapper.__qualname__ = func_name
        _reinsertion_wrapper.__module__ = self.function.__module__

        # Inherit metadata from the parent, allow overrides.
        child_kwargs = dict(
            function=_reinsertion_wrapper,
            name=name,
            short_name=short_name,
            includes_reinsertion=True,
            reinsertion_function=reinsertion_function,
            depends_on=self,
            method_type=self.method_type,
            description=self.description,
            citation=self.citation,
            authors=self.authors,
            source=self.source,
            license_file=self.license_file,
            dynamic=self.dynamic,
        )
        child_kwargs.update(kwargs)

        child = DismantlingMethod(**child_kwargs)
        dismantling_methods[child.key] = child
        return child


__all__ = []

# Auto-discover dismantling and reinsertion modules.
# Any module named *.python_interface is imported, triggering @dismantling_method
# decorators that register functions in the dismantling_methods dict.
# Modules named *.reinsertion_interface are imported similarly for reinsertion methods.
_DISCOVERY_SUFFIXES = (".python_interface", ".reinsertion_interface")

for loader, module_name, is_pkg in pkgutil.walk_packages(__path__, prefix=__name__ + "."):
    # Skip private modules (check the leaf component, not the fully-qualified name)
    leaf = module_name.rsplit(".", 1)[-1]
    if leaf.startswith("_"):
        continue

    if any(module_name.endswith(suffix) for suffix in _DISCOVERY_SUFFIXES):
        human_module_name = module_name
        for suffix in _DISCOVERY_SUFFIXES:
            human_module_name = human_module_name.replace(suffix, "")

        _module = None
        try:
            _module = importlib.import_module(module_name)
        except ModuleNotFoundError as e:
            logger.warning(f"ModuleNotFoundError while importing {human_module_name}: {e}\n", exc_info=False)
            continue
        except Exception as e:
            logger.warning(f"Exception: while importing {human_module_name}: {e}\n", exc_info=True)
            continue

        if _module is not None:
            __all__.append(module_name)
            globals()[module_name] = _module
        else:
            logger.warning(f"Error importing {human_module_name}.")

__alldict__ = {k: globals()[k] for k in __all__}
