PYTHON_VERSION := $(shell python -c "import sys;t='{v[0]}.{v[1]}'.format(v=list(sys.version_info[:2]));sys.stdout.write(t)")
PYTHON_LOCATION := $(CONDA_PREFIX)
#PYTHON_INC := $(PYTHON_LOCATION)/include/python$(PYTHON_VERSION)m
PYTHON_INC := $(PYTHON_LOCATION)/include/python$(PYTHON_VERSION)
#BOOST_INC := /usr/include/boost
BOOST_INC := $(PYTHON_LOCATION)/include/
BOOST_LIB_LOCATION := $(PYTHON_LOCATION)/lib
BOOST_LIB_VERSION := $(shell python -c "import sys;t='{v[0]}.{v[1]}'.format(v=list(sys.version_info[:2]));sys.stdout.write(t.replace('.', ''))")
BOOST_LIB_FILE := boost_python$(BOOST_LIB_VERSION)

CXX=g++

FLAGS=-fPIC -Wall -O3 -march=native -std=gnu++11 -g -Wno-deprecated -fpermissive
DEBUG_FLAGS=-fPIC -Wall -O0 -march=native -std=gnu++11 -g -Wno-deprecated -fpermissive
CInc := -I. -I$(BOOST_INC) -I$(PYTHON_INC)
CLinkFlags=-L$(BOOST_LIB_LOCATION)
LIBS= -lboost_program_options

BIN=EnsembleGND
#EnsembleGNDR

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Linux)
    LDFLAGS=-Wl,-soname,$@
endif
ifeq ($(UNAME_S),Darwin)
    LDFLAGS=-Wl,-install_name,$@
endif

LDFLAGS += -Wl,-rpath,$(BOOST_LIB_LOCATION)

all: $(BIN)

EnsembleGND: EnsembleGND.cpp
	${CXX} EnsembleGND.cpp -O3 -std=gnu++11 -o $@

EnsembleGNDR: EnsembleGNDR.cpp
	${CXX} ${FLAGS} EnsembleGNDR.cpp ${LIBS} -o $@

clean:
	rm -f $(BIN) *.o


