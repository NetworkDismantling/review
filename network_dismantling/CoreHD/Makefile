PYTHON_VERSION := $(shell python -c "import sys;t='{v[0]}.{v[1]}'.format(v=list(sys.version_info[:2]));sys.stdout.write(t)")
PYTHON_LOCATION := $(CONDA_PREFIX)
#PYTHON_INC := $(PYTHON_LOCATION)/include/python$(PYTHON_VERSION)m
PYTHON_INC := $(PYTHON_LOCATION)/include/python$(PYTHON_VERSION)
#BOOST_INC := /usr/include/boost
BOOST_INC := $(PYTHON_LOCATION)/include/boost
BOOST_LIB_LOCATION := $(PYTHON_LOCATION)/lib
BOOST_LIB_VERSION := $(shell python -c "import sys;t='{v[0]}.{v[1]}'.format(v=list(sys.version_info[:2]));sys.stdout.write(t.replace('.', ''))")
BOOST_LIB_FILE := boost_python$(BOOST_LIB_VERSION)
CXX=g++
FLAGS=-fPIC -Wall -O3 -march=native -std=gnu++11 -g -Wno-deprecated  -fpermissive
LIBS=-Wl,-soname,$@ -Wl,-rpath,$(BOOST_LIB_LOCATION) -L$(BOOST_LIB_LOCATION) -l$(BOOST_LIB_FILE) -I. -I$(BOOST_INC) -I$(PYTHON_INC) -lboost_program_options
BIN=coreHD


all: $(BIN)

coreHD: TAbyTwoCoreV01.cpp
	${CXX} ${FLAGS} TAbyTwoCoreV01.cpp ${LIBS} -o $@

clean:
	rm -f $(BIN) *.o *.so

