CXX=gcc
FLAGS=-Wall -O3 -march=native -std=c11 -lm
BIN=CI

FLAGS+=-Wno-unknown-pragmas

all: $(BIN)

CI: CI_HEAP.c
	${CXX} CI_HEAP.c ${FLAGS} -o ${BIN}

clean:
	rm -f $(BIN) *.o


