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

import logging
import pkgutil
from pathlib import Path

from network_dismantling.common.dismantlers import dismantler_wrapper

dismantling_methods = {}

logger = logging.getLogger(__name__)


def setdefaultattr(obj, name, value):
    try:
        return getattr(obj, name)
    except AttributeError:
        setattr(obj, name, value)
    return value


class DismantlingMethod:
    key: str = None

    name: str = None
    short_name: str = None

    doi = None
    citation = None
    description = None
    authors = None

    function = None
    dynamic = None

    display_name = None
    short_display_name = None

    plot_color: str = None
    plot_marker: str = None

    # reinsertion: ReinsertionSupport = None
    includes_reinsertion = False
    optional_reinsertion = False
    reinsertion_function = None
    reinsertion_display_name = None
    reinsertion_short_display_name = None

    license_file: Path = None

    # return_type: ReturnTypes = None

    source: str = None

    def __init__(self,
                 # name=None,
                 # includes_reinsertion=False,
                 # description=None,
                 # citation=None,
                 # authors=None,
                 # source=None,
                 # return_type=None,
                 **kwargs
                 ):

        super().__init__()

        for key, value in kwargs.items():
            # setdefaultattr(self, key, value)
            setattr(self, key, value)

        self.key = self.function.__name__

    def __call__(self, *args, **kwargs):
        output = self.function(*args, **kwargs)

        output["static"] = not self.dynamic
        output["heuristic"] = self.key

        return output

    def _format_output(self, output):

        if isinstance(output, dict):
            import pandas as pd

            output = pd.DataFrame(output)

        return output


__all__ = []

for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    if module_name.startswith("_"):
        continue

    if module_name.endswith(".python_interface"):
        # print("Importing", module_name)
        try:
            _module = loader.find_module(module_name).load_module(module_name)

            # __alldict__[module_name] = _module
            __all__.append(module_name)
            globals()[module_name] = _module
        except Exception as e:
            # print("Error importing:", module_name, e)
            logger.warning(f"Error importing: {module_name}")  # , exc_info=True)
            logger.debug("Exception:\n", exc_info=True)

            continue

__alldict__ = {k: globals()[k] for k in __all__}
