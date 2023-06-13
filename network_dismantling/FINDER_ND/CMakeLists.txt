cmake_minimum_required(VERSION 3.24)
project(FINDER_ND)

set(CMAKE_CXX_STANDARD 14)

include_directories(src/lib)

add_executable(FINDER_ND
        src/lib/config.cpp
        src/lib/config.h
        src/lib/decrease_strategy.cpp
        src/lib/disjoint_set.cpp
        src/lib/disjoint_set.h
        src/lib/graph.cpp
        src/lib/graph.h
        src/lib/graph_struct.cpp
        src/lib/graph_struct.h
        src/lib/graph_utils.cpp
        src/lib/graph_utils.h
        src/lib/i_env.h
        src/lib/msg_pass.cpp
        src/lib/msg_pass.h
        src/lib/mvc_env.cpp
        src/lib/mvc_env.h
        src/lib/nstep_replay_mem.cpp
        src/lib/nstep_replay_mem.h
        src/lib/nstep_replay_mem_prioritized.cpp
        src/lib/nstep_replay_mem_prioritized.h
        src/lib/PrepareBatchGraph.cpp
        src/lib/PrepareBatchGraph.h
        src/lib/utils.cpp
        src/lib/utils.h)
