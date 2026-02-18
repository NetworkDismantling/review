# ============================================================================
# Makefile for common reinsertion module
#
# Targets:
#   reinsertion            — subprocess binary (C++11, Boost.ProgramOptions)
#   libreinsertion_gt.so   — graph-tool C++ extension (C++17, Boost.Python)
#   all                    — both targets
# ============================================================================

PYTHON_VERSION := $(shell python -c "import sys;t='{v[0]}.{v[1]}'.format(v=list(sys.version_info[:2]));sys.stdout.write(t)")
PYTHON_LOCATION := $(CONDA_PREFIX)
PYTHON_INC := $(PYTHON_LOCATION)/include/python$(PYTHON_VERSION)
BOOST_INC := $(PYTHON_LOCATION)/include/
BOOST_LIB_LOCATION := $(PYTHON_LOCATION)/lib
BOOST_LIB_VERSION := $(shell python -c "import sys;t='{v[0]}.{v[1]}'.format(v=list(sys.version_info[:2]));sys.stdout.write(t.replace('.', ''))")
BOOST_LIB_FILE := boost_python$(BOOST_LIB_VERSION)

# Allow overriding compiler (e.g., CXX=clang++ make)
CXX ?= g++

# --- Subprocess binary flags (C++11) ---
FLAGS=-fPIC -Wall -O3 -march=native -std=gnu++11 -g -Wno-deprecated -fpermissive
CInc := -I. -I$(BOOST_INC) -I$(PYTHON_INC)
CLinkFlags=-L$(BOOST_LIB_LOCATION)
LIBS= -lboost_program_options

BIN=reinsertion

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Linux)
    LDFLAGS=-Wl,-soname,$@
endif
ifeq ($(UNAME_S),Darwin)
    LDFLAGS=-Wl,-install_name,$@
endif

LDFLAGS += -Wl,-rpath,$(BOOST_LIB_LOCATION)

# --- Graph-tool extension flags (C++17) ---
GT_INCLUDE := $(shell python -c "import graph_tool, os; print(os.path.join(os.path.dirname(graph_tool.__file__), 'include'))")
NUMPY_INC := $(shell python -c "import numpy; print(numpy.get_include())")

GT_CInc := -I$(GT_INCLUDE) \
           -I$(GT_INCLUDE)/boost-workaround \
           -I$(GT_INCLUDE)/pcg-cpp \
           -I$(NUMPY_INC) \
           -I$(PYTHON_INC) \
           -I$(BOOST_INC) \
           -pthread

GT_FLAGS := -fPIC -Wall -O3 -march=native -std=gnu++17 -DBOOST_ALLOW_DEPRECATED_HEADERS

# Platform-specific dynamic-lookup flags for the extension module.
# Symbols from graph_tool and Python are resolved at runtime
# (graph_tool must be imported before loading this module).
ifeq ($(UNAME_S),Darwin)
    GT_LDFLAGS := -undefined dynamic_lookup
endif
ifeq ($(UNAME_S),Linux)
    GT_LDFLAGS := -Wl,--allow-shlib-undefined
endif

GT_SO = libreinsertion_gt.so

# ============================================================================

all: $(BIN) $(GT_SO)

reinsertion: reinsertion.cpp
	${CXX} ${FLAGS} ${CInc} $(LDFLAGS) reinsertion.cpp -o $@ ${CLinkFlags} ${LIBS}

$(GT_SO): reinsertion_module.cpp
	${CXX} ${GT_FLAGS} -shared ${GT_CInc} $< -o $@ \
		-L$(BOOST_LIB_LOCATION) -l$(BOOST_LIB_FILE) \
		$(GT_LDFLAGS) -Wl,-rpath,$(BOOST_LIB_LOCATION)

clean:
	rm -f $(BIN) $(GT_SO) *.o
