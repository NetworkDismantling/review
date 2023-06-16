CXX=g++
FLAGS=-Wall -O3 -march=native -std=gnu++11 -g -Wno-deprecated -I. -fpermissive
LIBS=
BIN=coreHD

# uncomment the following line if you have an openmp-compliant compiler

all: $(BIN)

coreHD: TAbyTwoCoreV01.cpp
	${CXX} ${FLAGS} TAbyTwoCoreV01.cpp ${LIBS} -o $@

clean:
	rm -f $(BIN) *.o

