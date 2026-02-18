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
import logging
from functools import wraps
from pathlib import Path
from typing import List, Callable

from network_dismantling import DismantlingMethod, ReinsertionMethod, dismantling_methods, reinsertion_methods

_logger = logging.getLogger(__name__)


def _resolve_method_metadata(funct, name, short_name, citation):
    """Resolve common metadata (key, name, license, citation) for a method."""
    key = funct.__name__.replace("get_", "")

    method_name = name if name is not None else key

    frame = inspect.stack()[2]
    method_path = Path(frame[0].f_code.co_filename).resolve().parent

    license_file = method_path / "LICENSE"
    if not license_file.exists():
        license_file = None

    citation_text = ""
    citation_file = None

    if citation is None:
        for citation_file in method_path.glob("CITATION.*"):
            if citation_file.is_file():
                citation_text = citation_file.read_text().strip()
                break
            else:
                citation_file = None
    else:
        citation_text = citation

    return key, method_name, license_file, citation_text, citation_file


def dismantling_method(name: str | None = None,
                       short_name: str | None = None,
                       includes_reinsertion: bool = False,
                       method_type: str | None = None,
                       description: str | None = None,
                       citation: str | None = None,
                       authors: str | List[str] | None = None,
                       source: str | None = None,
                       depends_on: str | Callable | None = None,
                       reinsertion_function: Callable | None = None,
                       **kwargs,
                       ):
    """Register a function as a dismantling method.

    Args:
        name: Human-readable name.
        short_name: Abbreviated display name (required).
        includes_reinsertion: Whether this method already includes reinsertion.
        method_type: Category — "heuristic", "ml_based", "spectral",
                     "optimization", etc.  Used for filtering/grouping.
        description: Free-text description.
        citation: BibTeX or plain-text citation. Auto-discovered from
                  ``CITATION.*`` files if *None*.
        authors: Author name(s).
        source: URL to the original implementation.
        depends_on: A :class:`DismantlingMethod` or its key that must run first.
        reinsertion_function: Callable to auto-chain reinsertion after
            dismantling.  When set, :meth:`DismantlingMethod.__call__` runs
            the reinsertion after the main function and stores the output
            in ``output["reinsertion_predictions"]``.
        **kwargs: Extra attributes forwarded to :class:`DismantlingMethod`.
    """

    @wraps(dismantling_method)
    def wrapper(funct):
        key, method_name, license_file, citation_text, citation_file = (
            _resolve_method_metadata(funct, name, short_name, citation)
        )

        if key in dismantling_methods:
            _logger.warning(
                f"Duplicate dismantling method key '{key}' — overwriting "
                f"previous registration from {dismantling_methods[key].function.__module__}"
            )

        method = DismantlingMethod(
            name=method_name,
            short_name=short_name,
            description=description,
            citation=citation_text,
            authors=authors,
            function=funct,
            includes_reinsertion=includes_reinsertion,
            method_type=method_type,
            source=source,
            license_file=license_file,
            citation_file=citation_file,
            depends_on=depends_on,
            reinsertion_function=reinsertion_function,
            **kwargs,
        )

        dismantling_methods[key] = method

        return method

    return wrapper


def reinsertion_method(name: str | None = None,
                       short_name: str | None = None,
                       description: str | None = None,
                       citation: str | None = None,
                       authors: str | List[str] | None = None,
                       source: str | None = None,
                       **kwargs,
                       ):
    """Register a function as a reinsertion method.

    Reinsertion methods take a set of removed nodes and a graph,
    then return an optimised (smaller) removal set.  They are stored
    in ``reinsertion_methods`` as :class:`ReinsertionMethod` instances
    and auto-discovered from modules named ``*.reinsertion_interface``.

    Args:
        name: Human-readable name.
        short_name: Abbreviated display name (required).
        description: Free-text description.
        citation: BibTeX or plain-text citation.  Auto-discovered from
                  ``CITATION.*`` files if *None*.
        authors: Author name(s).
        source: URL to the original implementation.
        **kwargs: Extra attributes forwarded to :class:`ReinsertionMethod`.
    """

    @wraps(reinsertion_method)
    def wrapper(funct):
        key, method_name, license_file, citation_text, citation_file = (
            _resolve_method_metadata(funct, name, short_name, citation)
        )

        if key in reinsertion_methods:
            _logger.warning(
                f"Duplicate reinsertion method key '{key}' — overwriting "
                f"previous registration."
            )

        method = ReinsertionMethod(
            name=method_name,
            short_name=short_name,
            description=description,
            citation=citation_text,
            authors=authors,
            source=source,
            function=funct,
            license_file=license_file,
            citation_file=citation_file,
            **kwargs,
        )

        reinsertion_methods[key] = method

        return method

    return wrapper


__all__ = dismantling_methods.items()
__all_dict__ = dismantling_methods
