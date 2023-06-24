PYTHON_VERSION := $(shell python -c "import sys;t='{v[0]}.{v[1]}'.format(v=list(sys.version_info[:2]));sys.stdout.write(t)")
PYTHON_LOCATION := $(CONDA_PREFIX)
#PYTHON_INC := $(PYTHON_LOCATION)/include/python$(PYTHON_VERSION)m
PYTHON_INC := $(PYTHON_LOCATION)/include/python$(PYTHON_VERSION)
#BOOST_INC := /usr/include/boost
BOOST_INC := $(PYTHON_LOCATION)/include/boost
BOOST_LIB_LOCATION := $(PYTHON_LOCATION)/lib
BOOST_LIB_VERSION := $(shell python -c "import sys;t='{v[0]}.{v[1]}'.format(v=list(sys.version_info[:2]));sys.stdout.write(t.replace('.', ''))")
BOOST_LIB_FILE := boost_python$(BOOST_LIB_VERSION)

CXX := g++

CFLAGS := -c -fPIC -std=c++17 -DBOOST -march=native -O3
CInc := -I$(BOOST_INC) -I$(PYTHON_INC)

CLinkFlags = -shared -Wl,-soname,$@ -Wl,-rpath,$(BOOST_LIB_LOCATION) -L$(BOOST_LIB_LOCATION) -l$(BOOST_LIB_FILE)

PHONY: all
all: dismantler.so

dismantler.so: dismantler.o

%.so: %.o
	$(CXX) $^ $(CLinkFlags) -o $@

%.o: %.cpp
	$(CXX) $(CFLAGS) $(CInc) $^

clean:
	rm -f $(BIN) *.o *.so