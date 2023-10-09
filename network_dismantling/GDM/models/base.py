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

from argparse import ArgumentParser

from torch.nn import Module


class BaseModel(Module):

    _model_parameters = []
    _affected_by_seed = False

    @staticmethod
    def add_model_parameters(parser: ArgumentParser):
        pass

    @staticmethod
    def parameters_callback(args):
        pass

    @classmethod
    def get_name(cls):
        return cls.__name__

    @classmethod
    def get_parameters(cls):
        return cls._model_parameters

    @classmethod
    def is_affected_by_seed(cls):
        return cls._affected_by_seed

    def set_seed(self, seed):
        pass

    @staticmethod
    def parameters_combination_validator(params):
        return params

    def add_run_parameters(self, run: dict):
        pass

    @classmethod
    def add_run_parameters_from_args(cls, run: dict, args):
        pass

    def model_name(self):
        pass
