PYTHON_VERSION := $(shell python -c "import sys;t='{v[0]}.{v[1]}'.format(v=list(sys.version_info[:2]));sys.stdout.write(t)")
PYTHON_LOCATION := $(CONDA_PREFIX)
#PYTHON_INC := $(PYTHON_LOCATION)/include/python$(PYTHON_VERSION)m
PYTHON_INC := $(PYTHON_LOCATION)/include/python$(PYTHON_VERSION)
#BOOST_INC := /usr/include/boost
BOOST_INC := $(PYTHON_LOCATION)/include/
BOOST_LIB_LOCATION := $(PYTHON_LOCATION)/lib
BOOST_LIB_VERSION := $(shell python -c "import sys;t='{v[0]}.{v[1]}'.format(v=list(sys.version_info[:2]));sys.stdout.write(t.replace('.', ''))")
BOOST_LIB_FILE := boost_python$(BOOST_LIB_VERSION)

CXX := g++

CFLAGS := -c -fPIC -std=c++17 -DBOOST -march=native -O3
CInc := -I$(BOOST_INC) -I$(PYTHON_INC)

CLinkFlags = -L$(BOOST_LIB_LOCATION) -lpython$(PYTHON_VERSION) -l$(BOOST_LIB_FILE)

PHONY: all
all: dismantler.so

dismantler.so: dismantler.o

#-Wl,--verbose
# -Wl,--as-needed
# -Wl,-soname

# -Wl, -soname,$@
%.so: %.o
	$(CXX) -shared -Wl,-install_name,$@  $^ -o $@ $(CLinkFlags)

#--verbose
%.o: %.cpp
	$(CXX) $(CFLAGS) $(CInc) $^ $(CLinkFlags)

clean:
	rm -f $(BIN) *.o *.so