from pathlib import Path

import pkgutil


dismantling_methods = {}
# dismantling_methods_citation = {}
# dismantling_methods_descriptions = {}
# dismantling_methods_authors = {}
# dismantling_methods_include_reinsertion = {}
# dismantling_methods_license_file = {}
# dismantling_methods_source = {}


class DismantlingMethod:
    name = None

    doi = None
    citation = None
    description = None
    authors = None

    function = None
    dynamic = False

    display_name = None
    short_display_name = None

    includes_reinsertion = False
    optional_reinsertion = False
    reinsertion_function = None
    reinsertion_display_name = None
    reinsertion_short_display_name = None

    license_file: Path = None

    source: str = None

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


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
            print("Error importing:", module_name, e)
            continue


__alldict__ = {k: globals()[k] for k in __all__}

