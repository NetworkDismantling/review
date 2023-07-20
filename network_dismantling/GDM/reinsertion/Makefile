# ---- Edit the path to the boost library in the lines below --- #
BOOSTlib=/cygdrive/c/BoostTest/lib
BOOSTinclude=/cygdrive/c/BoostTest/include
# --------------------------------------------------------------#


CXX=g++
FLAGS=-Wall -O3 -march=native -std=gnu++11 -g -Wno-deprecated -I.
LIBS=-L $(BOOSTlib) -lboost_program_options
BIN=reinsertion

FLAGS+=-Wno-unknown-pragmas -I $(BOOSTinclude)

all: $(BIN)

reinsertion: reinsertion.cpp 
	${CXX} ${FLAGS} reinsertion.cpp ${LIBS} -o $@
clean:
	rm -f $(BIN) *.o


