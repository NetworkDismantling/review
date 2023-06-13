CXX=g++
FLAGS=-Wall -O3 -march=native -g -Wno-deprecated -I. 
LIBS=-lboost_program_options
BIN=decycler reverse-greedy

# uncomment the following line if you have an openmp-compliant compiler

OMP=1

ifeq ($(OMP),1)
	FLAGS+=-fopenmp -DOMP
	LIBS+=-lgomp
else
	FLAGS+=-Wno-unknown-pragmas

endif

all: $(BIN)

decycler: decycler.cpp proba.hpp real_type.hpp mes.hpp
	${CXX} ${FLAGS} decycler.cpp ${LIBS} -o $@

reverse-greedy: reverse-greedy.cpp real_type.hpp
	${CXX} ${FLAGS} reverse-greedy.cpp ${LIBS} -o $@
clean:
	rm -f $(BIN) *.o

