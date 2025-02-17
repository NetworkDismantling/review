PYTHON_VERSION := $(shell python -c "import sys;t='{v[0]}.{v[1]}'.format(v=list(sys.version_info[:2]));sys.stdout.write(t)")
PYTHON_LOCATION := $(CONDA_PREFIX)
#PYTHON_INC := $(PYTHON_LOCATION)/include/python$(PYTHON_VERSION)m
PYTHON_INC := $(PYTHON_LOCATION)/include/python$(PYTHON_VERSION)
BOOST_INC := $(PYTHON_LOCATION)/include/
BOOST_LIB_LOCATION := $(PYTHON_LOCATION)/lib
BOOST_LIB_VERSION := $(shell python -c "import sys;t='{v[0]}.{v[1]}'.format(v=list(sys.version_info[:2]));sys.stdout.write(t.replace('.', ''))")
BOOST_LIB_FILE := boost_python$(BOOST_LIB_VERSION)

CXX := g++

# -fPIC
CFLAGS := -shared -Wall -O3 -march=native -g -Wno-deprecated -DBOOST -std=c++11
CInc := -I$(BOOST_INC) -I$(PYTHON_INC) -I.

# -lpython$(PYTHON_VERSION) -l$(BOOST_LIB_FILE)
CLinkFlags = -L$(BOOST_LIB_LOCATION) -Wl,-Bstatic -lboost_program_options -Wl,-Bdynamic

# uncomment the following line if you have an openmp-compliant compiler
OMP=1

ifeq ($(OMP),1)
	CFLAGS +=-fopenmp -DOMP
	CLinkFlags +=-lgomp
else
	CFLAGS +=-Wno-unknown-pragmas
endif

BIN=decycler reverse-greedy

all: $(BIN)

decycler: decycler.cpp proba.hpp real_type.hpp mes.hpp
	${CXX} ${FLAGS} ${CInc} decycler.cpp ${LIBS} -o $@ $(CLinkFlags)

reverse-greedy: reverse-greedy.cpp real_type.hpp
	${CXX} ${FLAGS} reverse-greedy.cpp ${LIBS} -o $@ $(CLinkFlags)
clean:
	rm -f $(BIN) *.o

