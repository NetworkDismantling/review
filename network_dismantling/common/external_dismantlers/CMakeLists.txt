cmake_minimum_required(VERSION 3.30)

project(external_dismantlers CXX)

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

if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()

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

include_directories(${GRAPH_TOOL_INCLUDE_DIR})
link_directories(${GRAPH_TOOL_LIB_DIR})

message(STATUS "Graph-tool include: ${GRAPH_TOOL_INCLUDE_DIR}")
message(STATUS "Graph-tool lib: ${GRAPH_TOOL_LIB_DIR}")

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

# macOS specific settings
if(APPLE)
    # Get the lib directory from Python3
    get_filename_component(PYTHON_LIB_DIR "${Python3_LIBRARIES}" DIRECTORY)
    
    set_target_properties(dismantler PROPERTIES
        MACOSX_RPATH ON
        BUILD_WITH_INSTALL_RPATH ON
        INSTALL_RPATH "@loader_path;${PYTHON_LIB_DIR};${Boost_LIBRARY_DIRS}"
        LINK_FLAGS "-undefined dynamic_lookup"
    )
    
    message(STATUS "Setting RPATH to: @loader_path;${PYTHON_LIB_DIR};${Boost_LIBRARY_DIRS}")
    message(STATUS "Using -undefined dynamic_lookup for macOS")
endif()

# Link libraries
# Note: On macOS, we use -undefined dynamic_lookup and link only to Boost.Python
# to avoid symbol conflicts between multiple Python libraries
target_link_libraries(dismantler PRIVATE
    ${BOOST_PYTHON_LIB}
)

# Add include directories to target (modern CMake approach)
target_include_directories(dismantler PRIVATE
    ${GRAPH_TOOL_INCLUDE_DIR}
    ${NUMPY_INCLUDE_DIR}
)