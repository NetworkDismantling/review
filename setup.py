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
from pathlib import Path

from network_dismantling._setup_hook import registry
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info
from setuptools.command.install import install

logger = logging.getLogger(__name__)

package_folder = Path(__file__).parent / "network_dismantling"

for folder in package_folder.iterdir():
    if folder.is_dir() and (folder / "setup_hook.py").exists():
        # logger.info(f"Loading {folder.name}")
        module = importlib.import_module(f"network_dismantling.{folder.name}.setup_hook")


class CustomInstallCommand(install):
    def run(self):
        install.run(self)

        for hook in registry:
            try:
                hook(
                    package_folder=package_folder,
                    logger=logger,
                )
            except Exception as e:
                logger.exception(f"Error during setup: {e}",
                                 exc_info=e,
                                 )

class CustomDevelopCommand(develop):
    def run(self):
        develop.run(self)

        for hook in registry:
            hook()


class CustomEggInfoCommand(egg_info):
    def run(self):
        egg_info.run(self)

        for hook in registry:
            hook()


setup(
    name="NetworkDismantling",
    version="0.1",
    packages=find_packages(),
    url="https://github.com/NetworkDismantling/review/",
    license="",
    author="Marco Grassia",
    author_email="",
    description="Network dismantling library. Cite: Artime, O., Grassia, M., De Domenico, M. et al. Robustness and resilience of complex networks. Nat Rev Phys (2024). https://doi.org/10.1038/s42254-023-00676-y",
    cmdclass={
        "install": CustomInstallCommand,
        "develop": CustomDevelopCommand,
        "egg_info": CustomEggInfoCommand,
    }
)
