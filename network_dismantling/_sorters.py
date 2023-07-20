import inspect
from functools import wraps
from pathlib import Path

from network_dismantling import DismantlingMethod, dismantling_methods


def dismantling_method(name=None,
                       short_name=None,
                       includes_reinsertion=False,
                       description=None,
                       citation=None,
                       authors=None,
                       source=None,
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
                                   **kwargs,
                                   )

        dismantling_methods[key] = method

        return funct

    return wrapper


__all__ = dismantling_methods.items()
__all_dict__ = dismantling_methods
