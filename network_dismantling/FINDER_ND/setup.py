import os
import sys
from distutils.core import setup
from distutils.extension import Extension

from Cython.Distutils import build_ext

sys.path.append(os.path.dirname(__file__) + os.sep + "../")
print("APPENDING", os.path.dirname(__file__) + os.sep + "../")

ext_modules = [
    Extension('PrepareBatchGraph',
              sources=['PrepareBatchGraph.pyx', 'src/lib/PrepareBatchGraph.cpp', 'src/lib/graph.cpp',
                       'src/lib/graph_struct.cpp', 'src/lib/disjoint_set.cpp'], language='c++',
              extra_compile_args=['-std=c++11']),
    Extension('graph', sources=['graph.pyx', 'src/lib/graph.cpp'], language='c++',
              extra_compile_args=['-std=c++11']),
    Extension('mvc_env',
              sources=['mvc_env.pyx', 'src/lib/mvc_env.cpp', 'src/lib/graph.cpp', 'src/lib/disjoint_set.cpp'],
              language='c++', extra_compile_args=['-std=c++11']),
    Extension('utils', sources=['utils.pyx', 'src/lib/utils.cpp', 'src/lib/graph.cpp', 'src/lib/graph_utils.cpp',
                                'src/lib/disjoint_set.cpp', 'src/lib/decrease_strategy.cpp'], language='c++',
              extra_compile_args=['-std=c++11']),
    Extension('nstep_replay_mem',
              sources=['nstep_replay_mem.pyx', 'src/lib/nstep_replay_mem.cpp', 'src/lib/graph.cpp',
                       'src/lib/mvc_env.cpp', 'src/lib/disjoint_set.cpp'], language='c++',
              extra_compile_args=['-std=c++11']),
    Extension('nstep_replay_mem_prioritized',
              sources=['nstep_replay_mem_prioritized.pyx', 'src/lib/nstep_replay_mem_prioritized.cpp',
                       'src/lib/graph.cpp', 'src/lib/mvc_env.cpp', 'src/lib/disjoint_set.cpp'], language='c++',
              extra_compile_args=['-std=c++11']),
    Extension('graph_struct', sources=['graph_struct.pyx', 'src/lib/graph_struct.cpp'], language='c++',
              extra_compile_args=['-std=c++11']),
    Extension('FINDER', sources=['FINDER.pyx'])
]

for e in ext_modules:
    e.cython_directives = {'language_level': "3"} #all are Python-3


setup(
    name='FINDER_ND',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
)