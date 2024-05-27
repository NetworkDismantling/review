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

import inspect
from functools import wraps
from pathlib import Path
from typing import Union, List, Callable

from network_dismantling import DismantlingMethod, dismantling_methods


def dismantling_method(name: str = None,
                       short_name: str = None,
                       includes_reinsertion: bool = False,
                       description: str = None,
                       citation: str = None,
                       authors: Union[str, List[str]] = None,
                       source: str = None,
                       depends_on: Union[str, Callable] = None,
                       # plot_color: str = None,
                       # plot_marker: str = None,
                       **kwargs,
                       ):
    @wraps(dismantling_method)
    def wrapper(funct):
        key = funct.__name__
        key = key.replace("get_", "")

        if name is None:
            method_name = key
        elif short_name is None:
            method_name = name
        else:
            method_name = name

        frame = inspect.stack()[1]
        p = frame[0].f_code.co_filename
        p = Path(p).resolve()

        method_path = p.parent

        # if
        license_file = method_path / "LICENSE"
        if license_file.exists():
            # dismantling_methods_license_file[key] = license_file
            pass
        else:
            license_file = None

        citation_text = ""
        citation_file = None

        if citation is None:
            # TODO sort files according to some priority...
            for citation_file in method_path.glob("CITATION.*"):
                if citation_file.is_file():
                    citation_text = citation_file.read_text().strip()
                    # dismantling_methods_citation[key] = citation_file

                    break
                else:
                    citation_file = None

        else:
            citation_text = citation

        method = DismantlingMethod(name=method_name,
                                   short_name=short_name,
                                   description=description,
                                   citation=citation_text,
                                   authors=authors,
                                   function=funct,
                                   includes_reinsertion=includes_reinsertion,
                                   source=source,
                                   license_file=license_file,
                                   citation_file=citation_file,
                                   depends_on=depends_on,
                                   **kwargs,
                                   )

        dismantling_methods[key] = method

        # return funct
        return method

    return wrapper


__all__ = dismantling_methods.items()
__all_dict__ = dismantling_methods
