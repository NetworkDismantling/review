cmake_minimum_required(VERSION 4.0)

project(external_dismantlers CXX)

# Use pixi environment
if(DEFINED ENV{CONDA_PREFIX})
    set(CONDA_PREFIX $ENV{CONDA_PREFIX})
    message(STATUS "Using CONDA_PREFIX: ${CONDA_PREFIX}/include")
else()
    message(FATAL_ERROR "CONDA_PREFIX environment variable not set. Please activate the pixi environment.")
endif()

# Search for libraries in conda environment
set(CMAKE_FIND_ROOT_PATH ${CONDA_PREFIX})
message(STATUS "CMAKE_FIND_ROOT_PATH: ${CMAKE_FIND_ROOT_PATH}")

# Force find_package to search in CMAKE_FIND_ROOT_PATH only
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# Python executable must be set for find_package(Python3)
#  and should point to the python in the conda environment
#if(NOT DEFINED Python3_EXECUTABLE)
#set(Python3_EXECUTABLE "${CONDA_PREFIX}/bin/python")
#message(STATUS "Setting Python3_EXECUTABLE to: ${Python3_EXECUTABLE}")
#endif()

if(POLICY CMP0167)
    # CMake 3.29 and below provide a FindBoost module, but it needs constant updates to keep up with upstream Boost releases. Upstream Boost 1.70 and above provide a BoostConfig.cmake package configuration file. find_package(Boost CONFIG) finds the upstream package directly, without the find module.
    # CMake 3.30 and above prefer to not provide the FindBoost module so that find_package(Boost) calls, without the CONFIG or NO_MODULE options, find the upstream BoostConfig.cmake directly. This policy provides compatibility for projects that have not been ported to use the upstream Boost package.
    # The OLD behavior of this policy is for find_package(Boost) to load CMake's FindBoost module. The NEW behavior is for find_package(Boost) to search for the upstream BoostConfig.cmake.
    # This policy was introduced in CMake version 3.30. It may be set by cmake_policy() or cmake_minimum_required(). If it is not set, CMake warns, and uses OLD behavior.
    cmake_policy(SET CMP0167 NEW)
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Position Independent Code for shared libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set cmake variables
set(CMAKE_PREFIX_PATH ${CONDA_PREFIX})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)

message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")

# Include TSL library
include_directories(tsl)

# Find Python3
find_package(Python3 COMPONENTS Development NumPy)

if(Python3_FOUND)
    include_directories(${Python3_INCLUDE_DIRS})

    message(STATUS "Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
    message(STATUS "Python3_LIBRARIES: ${Python3_LIBRARIES}")
    
    if(Python3_NumPy_FOUND)
        include_directories(${Python3_NumPy_INCLUDE_DIRS})
        message(STATUS "NumPy_INCLUDE_DIRS: ${Python3_NumPy_INCLUDE_DIRS}")
    else()
        # Fallback: get numpy include from python
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
            OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        include_directories(${NUMPY_INCLUDE_DIR})
        message(STATUS "NumPy include (fallback): ${NUMPY_INCLUDE_DIR}")
    endif()
endif()

# Find graph-tool include path
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import graph_tool, os; print(os.path.join(os.path.dirname(graph_tool.__file__), 'include'))"
    OUTPUT_VARIABLE GRAPH_TOOL_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import graph_tool, os; print(os.path.dirname(graph_tool.__file__))"
    OUTPUT_VARIABLE GRAPH_TOOL_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if (NOT EXISTS "${GRAPH_TOOL_INCLUDE_DIR}" OR NOT IS_DIRECTORY "${GRAPH_TOOL_INCLUDE_DIR}")
    message(FATAL_ERROR "Graph-tool include directory not found: ${GRAPH_TOOL_INCLUDE_DIR}")
endif()

include_directories(${GRAPH_TOOL_INCLUDE_DIR})
link_directories(${GRAPH_TOOL_LIB_DIR})

message(STATUS "Graph-tool include: ${GRAPH_TOOL_INCLUDE_DIR}")
message(STATUS "Graph-tool lib: ${GRAPH_TOOL_LIB_DIR}")

# Set GRAPH_TOOL flag for compilation
add_compile_definitions(GRAPH_TOOL)

# Get Python version for Boost
string(REPLACE "." "" PYTHON_VERSION_NO_DOT "${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}")
message(STATUS "Looking for Boost Python component: python${PYTHON_VERSION_NO_DOT}")

# Find specific Boost Python library
set(BOOST_PYTHON_LIBRARY_NAME "boost_python${PYTHON_VERSION_NO_DOT}")
find_library(BOOST_PYTHON_LIB 
    NAMES ${BOOST_PYTHON_LIBRARY_NAME}
    HINTS ${CMAKE_PREFIX_PATH}/lib ${Python3_LIBRARY_DIRS}
)

if(BOOST_PYTHON_LIB)
    message(STATUS "Found Boost Python library: ${BOOST_PYTHON_LIB}")
else()
    message(FATAL_ERROR "Could not find ${BOOST_PYTHON_LIBRARY_NAME}")
endif()

find_package(Boost ${BOOST_MIN_VERSION} REQUIRED)

# manage compilation and linkage
if(Boost_FOUND)
    # definitions
    add_compile_definitions(BOOST_LOG_DYN_LINK=1)
    add_compile_definitions(BOOST)

    include_directories(${Boost_INCLUDE_DIRS})
    
    message(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
    message(STATUS "Boost_LIBRARIES: ${Boost_LIBRARIES}")
    message(STATUS "Boost_LIBRARY_DIRS: ${Boost_LIBRARY_DIRS}")
    message(STATUS "Boost_VERSION: ${Boost_VERSION}")
else()
    message(FATAL_ERROR "Boost not found! Cannot build without Boost.")
endif()


# Build as a Python module (shared library)
add_library(dismantler MODULE
                dismantler.cpp
                tsl/robin_growth_policy.h
                tsl/robin_hash.h
                tsl/robin_map.h
                tsl/robin_set.h
)

# Set compilation flags
target_compile_options(dismantler PRIVATE
    -O3
    -march=native
    $<$<CONFIG:Debug>:-g>
    $<$<CONFIG:Debug>:-O0>
)

# Set output name without lib prefix and with proper extension
set_target_properties(dismantler PROPERTIES
    PREFIX ""
    OUTPUT_NAME "dismantler"
    SUFFIX ".so"
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# Platform-specific linking settings
if(APPLE)
    # macOS: Use dynamic lookup for Python and graph_tool symbols
    # graph_tool must be imported in Python before importing this module
    get_filename_component(PYTHON_LIB_DIR "${Python3_LIBRARIES}" DIRECTORY)
    
    set_target_properties(dismantler PROPERTIES
        MACOSX_RPATH ON
        BUILD_WITH_INSTALL_RPATH ON
        INSTALL_RPATH "@loader_path;${PYTHON_LIB_DIR};${Boost_LIBRARY_DIRS};${GRAPH_TOOL_LIB_DIR}"
        LINK_FLAGS "-undefined dynamic_lookup"
    )
    
    message(STATUS "Setting RPATH to: @loader_path;${PYTHON_LIB_DIR};${Boost_LIBRARY_DIRS};${GRAPH_TOOL_LIB_DIR}")
    message(STATUS "Using -undefined dynamic_lookup for macOS")
    
    # On macOS, we use -undefined dynamic_lookup, so we only need to link Boost.Python
    # Python and graph_tool symbols will be resolved at runtime
    target_link_libraries(dismantler PRIVATE
        ${BOOST_PYTHON_LIB}
    )
    
elseif(UNIX)
    # Linux: Use RPATH and allow undefined symbols
    # graph_tool must be imported in Python before importing this module
    get_filename_component(PYTHON_LIB_DIR "${Python3_LIBRARIES}" DIRECTORY)
    
    set_target_properties(dismantler PROPERTIES
        INSTALL_RPATH "$ORIGIN:${PYTHON_LIB_DIR}:${Boost_LIBRARY_DIRS}:${GRAPH_TOOL_LIB_DIR}"
        BUILD_WITH_INSTALL_RPATH ON
        LINK_FLAGS "-Wl,--allow-shlib-undefined"
    )
    
    message(STATUS "Setting RPATH to: $ORIGIN:${PYTHON_LIB_DIR}:${Boost_LIBRARY_DIRS}:${GRAPH_TOOL_LIB_DIR}")
    message(STATUS "Using --allow-shlib-undefined for Linux")
    
    # On Linux, we use --allow-shlib-undefined, so we only need to link Boost.Python
    # Python and graph_tool symbols will be resolved at runtime
    target_link_libraries(dismantler PRIVATE
        ${BOOST_PYTHON_LIB}
    )
    
elseif(WIN32)
    # Windows: Must link explicitly to all libraries
    # Find graph_tool library
    find_library(GRAPH_TOOL_CORE_LIB
        NAMES graph_tool_core
        HINTS ${GRAPH_TOOL_LIB_DIR}
        NO_DEFAULT_PATH
    )
    
    if(GRAPH_TOOL_CORE_LIB)
        message(STATUS "Found graph_tool_core library: ${GRAPH_TOOL_CORE_LIB}")
    else()
        message(WARNING "graph_tool_core library not found")
    endif()
    
    target_link_libraries(dismantler PRIVATE
        ${BOOST_PYTHON_LIB}
        ${Python3_LIBRARIES}
        ${GRAPH_TOOL_CORE_LIB}
    )
    
    message(STATUS "Windows: linking explicitly to Python and graph_tool")
endif()

# Add include directories to target (modern CMake approach)
target_include_directories(dismantler PRIVATE
    ${GRAPH_TOOL_INCLUDE_DIR}
    ${NUMPY_INCLUDE_DIR}
)