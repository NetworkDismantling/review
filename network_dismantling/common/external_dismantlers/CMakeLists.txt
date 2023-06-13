cmake_minimum_required(VERSION 3.24)
project(external_dismantlers)

set(CMAKE_CXX_STANDARD 14)

include_directories(tsl)

add_executable(external_dismantlers
        tsl/robin_growth_policy.h
        tsl/robin_hash.h
        tsl/robin_map.h
        tsl/robin_set.h
        dismantler.cpp
        )
