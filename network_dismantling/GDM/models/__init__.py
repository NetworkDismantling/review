#   This file is part of GDM (Graph Dismantling with Machine learning),
#   proposed in the paper "Machine learning dismantling and
#   early-warning signals of disintegration in complex systems"
#   by M. Grassia, M. De Domenico and G. Mangioni.
#
#   GDM is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   GDM is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with GDM.  If not, see <http://www.gnu.org/licenses/>.

from network_dismantling.GDM.models.GAT import GAT_Model

import importlib
import pkgutil

from network_dismantling.GDM.models.base import BaseModel

for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    if module_name.startswith("_"):
        continue

    _module = None
    try:
        _module = importlib.import_module(f"GDM.models.{module_name}")
    except Exception as e:
        # print(f"Error loading sub-module {module_name} ({e})")

        try:
            _module = loader.find_module(f"GDM.models.{module_name}").load_module(f"GDM.models.{module_name}")
        except Exception as e:
            # print(f"Error loading sub-module {module_name} ({e})")
            pass

    if _module is None:
        print(f"Error loading sub-module {module_name}")
        pass
    else:
        globals()[module_name] = _module

models_mapping = dict((cls.__name__, cls) for cls in BaseModel.__subclasses__())

__all__ = list(models_mapping.items())
__alldict__ = models_mapping
